package verboten

import (
	"context"
	"embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"text/template"
	"time"
	"unicode"

	"github.com/gorilla/websocket"
	"golang.org/x/sync/errgroup"
	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
	"google.golang.org/genai"
)

const DefaultLiveModel = "gemini-3.1-flash-live-preview"

// NewLogger returns a structured JSON slog.Logger whose top-level keys match
// Google Cloud Logging conventions ("severity", "message", "time").
// If w is nil, logs are written to os.Stdout.
func NewLogger(w io.Writer) *slog.Logger {
	if w == nil {
		w = os.Stdout
	}
	handler := slog.NewJSONHandler(w, &slog.HandlerOptions{
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) != 0 {
				return a
			}
			switch a.Key {
			case slog.LevelKey:
				a.Key = "severity"
				if lvl, ok := a.Value.Any().(slog.Level); ok {
					switch {
					case lvl < slog.LevelInfo:
						a.Value = slog.StringValue("DEBUG")
					case lvl < slog.LevelWarn:
						a.Value = slog.StringValue("INFO")
					case lvl < slog.LevelError:
						a.Value = slog.StringValue("WARNING")
					default:
						a.Value = slog.StringValue("ERROR")
					}
				}
			case slog.MessageKey:
				a.Key = "message"
			}
			return a
		},
	})
	return slog.New(handler)
}

type VerbotenGameServer struct {
	genaiClient               *genai.Client
	Logger                    *slog.Logger
	Verbose                   bool
	LiveModel                 string
	TTSModel                  string
	JudgeVoice                string
	RecordingsDir             string
	sameRootProbFunc          func(ctx context.Context, lang, spokenWord, forbiddenWord string) (float64, error)
	samePronunciationProbFunc func(ctx context.Context, lang, spokenWord, forbiddenWord string) (float64, error)
	judgeTTSFunc              func(ctx context.Context, lang, text string) ([]byte, string, error)

	ttsCacheMu sync.RWMutex
	ttsCache   map[string]cachedTTSAudio
}

func (vg *VerbotenGameServer) logger() *slog.Logger {
	if vg != nil && vg.Logger != nil {
		return vg.Logger
	}
	return slog.Default()
}

func NewServer(genaiClient *genai.Client) *VerbotenGameServer {
	liveModel := strings.TrimSpace(os.Getenv("VERBOTEN_LIVE_MODEL"))
	if liveModel == "" {
		liveModel = DefaultLiveModel
	}
	ttsModel := strings.TrimSpace(os.Getenv("VERBOTEN_TTS_MODEL"))
	if ttsModel == "" {
		ttsModel = DefaultTTSModel
	}
	judgeVoice := strings.TrimSpace(os.Getenv("VERBOTEN_JUDGE_VOICE"))
	if judgeVoice == "" {
		judgeVoice = DefaultJudgeVoice
	}
	return &VerbotenGameServer{
		genaiClient:   genaiClient,
		LiveModel:     liveModel,
		TTSModel:      ttsModel,
		JudgeVoice:    judgeVoice,
		RecordingsDir: DefaultRecordingsDirForEnv(),
	}
}

func recordGamesEnabled() bool {
	switch strings.TrimSpace(os.Getenv("RECORD_GAMES")) {
	case "1", "TRUE", "true":
		return true
	default:
		return false
	}
}

// DefaultRecordingsDirForEnv returns where game recordings should be written.
// Recording is disabled unless RECORD_GAMES is "1", "TRUE", or "true".
// When enabled, VERBOTEN_RECORDINGS_DIR can optionally override DefaultRecordingsDir ("recordings").
func DefaultRecordingsDirForEnv() string {
	if !recordGamesEnabled() {
		return ""
	}
	if dir := strings.TrimSpace(os.Getenv("VERBOTEN_RECORDINGS_DIR")); dir != "" {
		return dir
	}
	return DefaultRecordingsDir
}

// NewGenAIClient creates a genai.Client configured for Verboten's Live and Decisions APIs.
// Both gemini-3.1-flash-live-preview (BidiGenerateContent) and gemini-decision-1 (/decisions)
// require the Gemini Developer API (BackendGeminiAPI) on API version "v1beta"
// (authenticated via GEMINI_API_KEY or GOOGLE_API_KEY).
func NewGenAIClient(ctx context.Context) (*genai.Client, error) {
	apiVersion := "v1beta"
	if v := strings.TrimSpace(os.Getenv("GOOGLE_GENAI_API_VERSION")); v != "" {
		apiVersion = v
	}
	cfg := &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			APIVersion: apiVersion,
		},
	}
	return genai.NewClient(ctx, cfg)
}

const guesserPrompt = `
	You are playing the "guessing word" game in English, where the human player with their microphone
	is speaking in English to describe a secret English word.
	Expect English spoken input from the human player, and respond unmistakably in English.
	Your job is to listen to the description and say only one word as
	your guess, every few seconds. You have only 3 guesses.
	Don't say anything else than the word you're guessing.
`

const guesserPrompt_fr = `
	Vous jouez au jeu du "mot à deviner" en français, où le joueur humain avec son microphone
	s'exprime en français pour décrire un mot secret français.
	Attendez-vous à entendre du français en entrée audio, et répondez exclusivement en français.
	Votre travail consiste à écouter la description et à ne dire qu'un seul mot comme
	votre suggestion, toutes les quelques secondes. Vous n'avez que 3 essais.
	Ne dites rien d'autre que le mot que vous devinez.
`

const guesserPrompt_ar = `
	أنت تلعب لعبة "تخمين الكلمات" باللغة العربية حيث يتحدث اللاعب البشري بميكروفونه باللغة العربية لوصف كلمة سرية.
	توقع سماع اللغة العربية كمدخل صوتي، وأجب باللغة العربية فقط.
	مهمتك هي الاستماع إلى الوصف وقول كلمة واحدة فقط كتخمين، كل بضع ثوان.
	لديك 3 تخمينات فقط.
	لا تقل أي شيء آخر غير الكلمة التي تخمنها.
`

//go:embed assets
var embeddedAssets embed.FS

func (vg *VerbotenGameServer) Handler() http.Handler {
	assetsFS, err := fs.Sub(embeddedAssets, "assets")
	if err != nil {
		panic(fmt.Errorf("embedded assets sub-fs: %w", err))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", vg.serveGame)
	mux.HandleFunc("/live/", vg.liveGame)
	mux.HandleFunc("/tts", vg.serveTTS)
	mux.HandleFunc("/words.json", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFileFS(w, r, assetsFS, "words.json")
	})
	mux.Handle("/forbiddenwords/", http.StripPrefix("/forbiddenwords/", http.FileServerFS(assetsFS)))
	return mux
}

func (vg *VerbotenGameServer) Start(ctx context.Context) error {
	logger := vg.logger()

	// Determine port for HTTP service.
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
		logger.Info("defaulting to port", "port", port)
	}

	// Start HTTP server.
	logger.Info("listening", "port", port)
	return http.ListenAndServe(":"+port, vg.Handler())
}

func (vg *VerbotenGameServer) serveGame(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFS(embeddedAssets, "assets/verboten.html")
	if err != nil {
		http.Error(w, "Error loading template", http.StatusInternalServerError)
		return
	}
	err = tmpl.Execute(w, nil)
	if err != nil {
		http.Error(w, "Error executing template", http.StatusInternalServerError)
		return
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

const declareLossFuncName = "declare_loss"

type judgeMessage struct {
	Judge *judgeVerdict `json:"judge"`
}

type clientControlMessage struct {
	Outcome string `json:"outcome"`
}

type judgeVerdict struct {
	Lost          bool   `json:"lost"`
	ForbiddenWord string `json:"forbiddenWord,omitempty"`
	SpokenWord    string `json:"spokenWord,omitempty"`
	Reason        string `json:"reason,omitempty"`
	Message       string `json:"message,omitempty"`
	Audio         string `json:"audio,omitempty"`
	AudioMimeType string `json:"audioMimeType,omitempty"`
}

func judgeSystemInstruction(langName, languageCode, targetWord string, otherForbiddenWords []string) string {
	allForbidden := make([]string, 0, 1+len(otherForbiddenWords))
	if targetWord != "" {
		allForbidden = append(allForbidden, targetWord)
	}
	allForbidden = append(allForbidden, otherForbiddenWords...)

	return fmt.Sprintf(`You are the strict Judge in a live voice game of "Verboten" (Proscribed Words), played in %s (%s).
You are listening directly to the human player's microphone as they describe a secret target word to an AI guesser.
Expect %s spoken input most of the time, while staying strictly alert for any exact translation of the target word %q spoken in ANY other language (English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, etc.).

Target word to guess: %s
Other forbidden words: %s
Full list of forbidden words: %s

STRICT LOSS CONDITIONS — You MUST immediately interrupt the game and call %s if the human player pronounces ANY of the following:
1. EXACT FORBIDDEN WORD (reason="exact"): The target word (%s) OR any of the other forbidden words (%s).
2. TOO CLOSE / SAME-ROOT WORD (reason="resembling"): Two words are considered "too close" ONLY when they share the SAME ROOT/STEM as the target word (%s) OR any of the other forbidden words (%s) — i.e., an inflection, plural, feminine, verb conjugation, derivative, or compound from the same lexical root (e.g., "poussent" for "Pousser", "porte" for "Porter", "joue" for "Jouer", "cheesy" for "Cheese", "arbres" for "Arbre"). NEVER treat two words as "too close" merely because they are synonyms or semantically related!
3. EXACT TRANSLATION OF THE TARGET WORD IN ANOTHER LANGUAGE (reason="translation"): Any word that is the exact translation of the TARGET WORD (%s) in ANY other language — whether intentional or accidental, whether spoken alone or inside a %s sentence, and including plurals or inflections of that translation!
   Examples of FORBIDDEN exact translations of a target word (MUST immediately call %s with reason="translation"):
   - Target "Citron" -> player says "lemon", "lemons", "limón", "Zitrone", "limone" -> LOST!
   - Target "Chapeau" -> player says "hat", "hats", "sombrero", "Hut", "cappello" -> LOST!
   - Target "Arbre" -> player says "tree", "trees", "árbol", "Baum", "albero" -> LOST!
   - Target "Moon" -> player says "lune", "luna", "Mond" -> LOST!
   - Target "Cheese" -> player says "fromage", "queso", "Käse", "formaggio" -> LOST!
   IMPORTANT: Check foreign-language translations ONLY for the target word (%s). Do NOT check or penalize translations in other languages for the other forbidden words (%s).

ALLOWED (do NOT call %s):
- Any word in %s that does NOT share the same root as any forbidden word AND is NOT an exact translation of the target word (%s) in another language, even if it is a synonym or semantically related!
  Examples that are STRICTLY ALLOWED (must NOT trigger a loss):
  - "flamme" / "flammes" is NOT too close to "feu" (different roots).
  - "grand" is NOT too close to "flamme" (different roots).
  - "gros" is NOT too close to "grand" (synonym with a different root).
  - "mozzarella" is NOT too close to "Cheese"; "ficelle" is NOT too close to "Corde"; "tronc" is NOT too close to "Arbre".
- Translations in another language of the OTHER forbidden words (%s) — only translations of the target word (%s) are forbidden.
- Unclear background noise or hesitation sounds ("um", "euh") unless condition 1, 2, or 3 was clearly triggered.

RESPONSE BEHAVIOR:
- If the human player violates condition 1, 2, or 3 (exact forbidden word, very resembling word, OR exact translation of the target word %q in another language), IMMEDIATELY call %s with forbidden_word, spoken_word, and reason.
- Only if NONE of conditions 1, 2, or 3 were violated, do NOT call %s; simply reply "OK" and keep listening.`,
		langName,
		languageCode,
		langName,
		targetWord,
		targetWord,
		strings.Join(otherForbiddenWords, ", "),
		strings.Join(allForbidden, ", "),
		declareLossFuncName,
		targetWord,
		strings.Join(otherForbiddenWords, ", "),
		targetWord,
		strings.Join(otherForbiddenWords, ", "),
		targetWord,
		langName,
		declareLossFuncName,
		targetWord,
		strings.Join(otherForbiddenWords, ", "),
		declareLossFuncName,
		langName,
		targetWord,
		strings.Join(otherForbiddenWords, ", "),
		targetWord,
		targetWord,
		declareLossFuncName,
		declareLossFuncName,
	)
}

func (vg *VerbotenGameServer) liveGame(w http.ResponseWriter, r *http.Request) {
	lang := strings.TrimPrefix(r.URL.Path, "/live/")
	var prompt, langName, languageCode string
	switch lang {
	case "en":
		prompt = guesserPrompt
		langName = "English"
		languageCode = "en-US"
	case "fr":
		prompt = guesserPrompt_fr
		langName = "French"
		languageCode = "fr-FR"
	case "ar":
		prompt = guesserPrompt_ar
		langName = "Arabic"
		languageCode = "ar-EG"
	default:
		vg.logger().Warn("unsupported language", "lang", lang)
		http.NotFound(w, r)
		return
	}

	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		vg.logger().Error("websocket upgrade failed", "lang", lang, "err", err)
		return
	}
	defer c.Close()

	gameID := randomString(4)
	gameLog := vg.logger().With("game_id", gameID)
	targetWord := r.URL.Query().Get("target")
	forbiddenWords := r.URL.Query()["forbidden"]
	if targetWord == "" && len(forbiddenWords) > 0 {
		targetWord = forbiddenWords[0]
	}
	var otherForbiddenWords []string
	for _, w := range forbiddenWords {
		if targetWord != "" && normalizeWord(w) == normalizeWord(targetWord) {
			continue
		}
		otherForbiddenWords = append(otherForbiddenWords, w)
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	model := strings.TrimSpace(vg.LiveModel)
	if model == "" {
		model = DefaultLiveModel
	}
	gameLog.Info("starting game",
		"lang", lang,
		"target", targetWord,
		"forbidden", otherForbiddenWords,
		"model", model,
	)

	var recorder *gameRecorder
	if vg.RecordingsDir != "" {
		var err error
		recorder, err = newGameRecorder(vg.RecordingsDir, gameID, lang, languageCode, model, targetWord, otherForbiddenWords, gameLog)
		if err != nil {
			gameLog.Warn("game recording disabled", "dir", vg.RecordingsDir, "err", err)
		} else {
			gameLog.Info("recording game", "dir", recorder.Dir())
			defer func() {
				if err := recorder.Close(); err != nil {
					gameLog.Error("close recorder failed", "err", err)
				}
			}()
		}
	}

	var loopsWG sync.WaitGroup
	defer loopsWG.Wait()

	var (
		prefixPaddingMs  int32 = 100
		guesserSilenceMs int32 = 500
	)
	guesserRealtimeInputConfig := &genai.RealtimeInputConfig{
		AutomaticActivityDetection: &genai.AutomaticActivityDetection{
			StartOfSpeechSensitivity: genai.StartSensitivityHigh,
			EndOfSpeechSensitivity:   genai.EndSensitivityLow,
			PrefixPaddingMs:          &prefixPaddingMs,
			SilenceDurationMs:        &guesserSilenceMs,
		},
	}
	// The Judge uses manual activity detection (driven by judgeVAD) with NO_INTERRUPTION
	// because gemini-3.1-flash-live-preview ignores short SilenceDurationMs values in auto-VAD
	// and can otherwise delay evaluating continuous speech by 15+ seconds.
	judgeRealtimeInputConfig := &genai.RealtimeInputConfig{
		AutomaticActivityDetection: &genai.AutomaticActivityDetection{
			Disabled: true,
		},
		ActivityHandling: genai.ActivityHandlingNoInterruption,
	}

	audioTranscriptionConfig := &genai.AudioTranscriptionConfig{}

	// Gemini Live session 1 (Guesser): listens to the human and guesses the secret word,
	// without knowing any of the forbidden words.
	config := &genai.LiveConnectConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{Text: prompt},
			},
		},
		SpeechConfig: &genai.SpeechConfig{
			LanguageCode: languageCode,
			VoiceConfig: &genai.VoiceConfig{
				PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
					VoiceName: "Puck",
				},
			},
		},
		ResponseModalities:       []genai.Modality{genai.ModalityAudio},
		InputAudioTranscription:  audioTranscriptionConfig,
		OutputAudioTranscription: audioTranscriptionConfig,
		RealtimeInputConfig:      guesserRealtimeInputConfig,
	}

	// Gemini Live session 2 (Judge): listens to the same human voice stream, knows the
	// target word and forbidden words, and calls declare_loss on rule violations.
	configJudge := &genai.LiveConnectConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{Text: judgeSystemInstruction(langName, languageCode, targetWord, otherForbiddenWords)},
			},
		},
		SpeechConfig: &genai.SpeechConfig{
			LanguageCode: languageCode,
		},
		ResponseModalities:  []genai.Modality{genai.ModalityAudio},
		RealtimeInputConfig: judgeRealtimeInputConfig,
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        declareLossFuncName,
						Description: "Interrupt the game and declare the game is lost when the human player utters a forbidden word, a very resembling word (same root/inflection) of any forbidden word, or the exact translation of the target word in another language.",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"forbidden_word": {
									Type:        genai.TypeString,
									Description: "The exact word from the forbidden list (or the target word) that was violated.",
								},
								"spoken_word": {
									Type:        genai.TypeString,
									Description: "Only the triggering word itself spoken by the human player (in any language), stripped of any surrounding articles, determiners, or elided prefixes (e.g. 'instrument', NOT \"l'instrument\" or 'un instrument').",
								},
								"reason": {
									Type:        genai.TypeString,
									Enum:        []string{"exact", "resembling", "translation"},
									Description: "Why the game is lost: 'exact' for a forbidden word itself, 'resembling' for an inflection or same-root word of any forbidden word, or 'translation' for the exact translation of the target word in another language.",
								},
							},
							Required: []string{"forbidden_word", "spoken_word", "reason"},
						},
					},
				},
			},
		},
	}

	var session, sessionJudge *genai.Session
	connectGroup, connectCtx := errgroup.WithContext(ctx)
	connectGroup.Go(func() error {
		var err error
		session, err = vg.genaiClient.Live.Connect(connectCtx, model, config)
		if err != nil {
			return fmt.Errorf("connect guesser model: %w", err)
		}
		return nil
	})
	connectGroup.Go(func() error {
		var err error
		sessionJudge, err = vg.genaiClient.Live.Connect(connectCtx, model, configJudge)
		if err != nil {
			return fmt.Errorf("connect judge model: %w", err)
		}
		return nil
	})
	if err := connectGroup.Wait(); err != nil {
		gameLog.Error("connect live models failed", "err", err)
		if session != nil {
			session.Close()
		}
		if sessionJudge != nil {
			sessionJudge.Close()
		}
		return
	}
	defer session.Close()
	defer sessionJudge.Close()

	var (
		writeMu      sync.Mutex
		judgeWriteMu sync.Mutex
		gameOver     atomic.Bool
	)
	writeToBrowser := func(payload []byte) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return c.WriteMessage(websocket.TextMessage, payload)
	}
	sendToJudge := func(input genai.LiveRealtimeInput) error {
		judgeWriteMu.Lock()
		defer judgeWriteMu.Unlock()
		return sessionJudge.SendRealtimeInput(input)
	}
	jVAD := newJudgeVAD(sendToJudge)

	loopsWG.Add(1)
	go func() {
		defer loopsWG.Done()
		// Guessing Loop:
		// Receive audio and transcription data from the Guesser Gemini Live session.
		// Forward it to the player browser, via WebSocket.
		for {
			message, err := session.Receive()
			if err != nil {
				if !gameOver.Load() {
					gameLog.Warn("guesser model disconnected", "err", err)
				}
				return
			}
			if gameOver.Load() {
				return
			}
			if message.ServerContent != nil {
				if it := message.ServerContent.InputTranscription; it != nil && strings.TrimSpace(it.Text) != "" {
					recorder.RecordTranscript("human", it.Text)
					if vg.Verbose {
						gameLog.Info("human transcript", "text", it.Text)
					}
				}
				if ot := message.ServerContent.OutputTranscription; ot != nil && strings.TrimSpace(ot.Text) != "" {
					recorder.RecordTranscript("guesser", ot.Text)
					if vg.Verbose {
						gameLog.Info("guesser transcript", "text", ot.Text)
					}
				}
				if message.ServerContent.ModelTurn != nil {
					for _, part := range message.ServerContent.ModelTurn.Parts {
						if part != nil && part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/pcm") && len(part.InlineData.Data) > 0 {
							recorder.RecordGuesserAudio(part.InlineData.Data, part.InlineData.MIMEType)
						}
					}
				}
			}
			messageBytes, err := json.Marshal(message)
			if err != nil {
				gameLog.Error("marshal guesser response failed", "err", err)
				return
			}
			if err := writeToBrowser(messageBytes); err != nil {
				gameLog.Warn("write guesser message failed", "err", err)
				return
			}
		}
	}()

	loopsWG.Add(1)
	go func() {
		defer loopsWG.Done()
		// Judge Loop:
		// Receive tool calls from the Judge Gemini Live session when a forbidden word is uttered.
		// Signal to the browser to interrupt and end the game.
		for {
			message, err := sessionJudge.Receive()
			if err != nil {
				if !gameOver.Load() {
					gameLog.Warn("judge model disconnected", "err", err)
				}
				return
			}
			if message.ServerContent != nil && message.ServerContent.TurnComplete {
				if !gameOver.Load() {
					if err := jVAD.OnTurnComplete(); err != nil && !gameOver.Load() {
						gameLog.Warn("flush buffered audio to judge failed", "err", err)
					}
				}
			}
			if message.ToolCall != nil {
				for _, fc := range message.ToolCall.FunctionCalls {
					if fc.Name == declareLossFuncName {
						forbiddenWord, _ := fc.Args["forbidden_word"].(string)
						spokenWord, _ := fc.Args["spoken_word"].(string)
						reason, _ := fc.Args["reason"].(string)

						ignoreAndKeepListening := func(resultMsg string, decisionProb *float64) {
							recorder.RecordJudgeEvent(fc.Name, true, spokenWord, forbiddenWord, reason, decisionProb)
							judgeWriteMu.Lock()
							err := sessionJudge.SendToolResponse(genai.LiveToolResponseInput{
								FunctionResponses: []*genai.FunctionResponse{
									{
										ID:   fc.ID,
										Name: fc.Name,
										Response: map[string]any{
											"result": resultMsg,
										},
									},
								},
							})
							judgeWriteMu.Unlock()
							if err != nil && !gameOver.Load() {
								gameLog.Warn("send tool response to judge failed", "err", err)
							}
						}

						if reason == "translation" && targetWord != "" && normalizeWord(forbiddenWord) != normalizeWord(targetWord) {
							if vg.Verbose {
								gameLog.Info("ignoring judge translation call for non-target word",
									"forbidden", forbiddenWord,
									"target", targetWord,
								)
							}
							ignoreAndKeepListening("Ignored: foreign-language translations are only forbidden for the target word. Keep listening.", nil)
							continue
						}

						// When the Judge detects a "too close" (resembling) word — or any non-translation
						// call where the spoken word is not an exact token match for the forbidden word —
						// run all checks and candidate TTS generations concurrently:
						//   - Call A1: Gemini Decision 1 same-root check (threshold >= 75%)
						//   - Call A2: Gemini Decision 1 same-pronunciation check (threshold >= 75%, e.g. "clous"/"Clou", "poissons"/"Poisson")
						//   - Call B1: Gemini TTS for the "resembling" explanation
						//   - Call B2: Gemini TTS for the "exact" explanation (used if A2 confirms identical pronunciation)
						// Once A1 & A2 complete, unused TTS calls are cancelled immediately so zero extra latency is added.
						var (
							decisionProb  *float64
							ttsAudio      []byte
							ttsMimeType   string
							lossMessage   string
							isExactMatch  = isExactForbiddenMatch(spokenWord, forbiddenWord)
							needsDecision = !isExactMatch && reason != "translation" && spokenWord != "" && forbiddenWord != ""
						)
						if isExactMatch {
							reason = "exact"
						}

						if needsDecision {
							resemblingMessage := formatJudgeLossText(lang, spokenWord, forbiddenWord, "resembling")
							exactMessage := formatJudgeLossText(lang, spokenWord, forbiddenWord, "exact")

							type decisionResult struct {
								prob float64
								err  error
							}
							type ttsResult struct {
								audio    []byte
								mimeType string
								err      error
							}
							rootCh := make(chan decisionResult, 1)
							pronCh := make(chan decisionResult, 1)
							resemblingTTSCh := make(chan ttsResult, 1)
							exactTTSCh := make(chan ttsResult, 1)

							rootCtx, rootCancel := context.WithTimeout(ctx, 5*time.Second)
							pronCtx, pronCancel := context.WithTimeout(ctx, 5*time.Second)
							resemblingTTSCtx, resemblingTTSCancel := context.WithTimeout(ctx, judgeTTSTimeout)
							exactTTSCtx, exactTTSCancel := context.WithTimeout(ctx, judgeTTSTimeout)

							// Call A1: Decision 1 API (same root?)
							go func() {
								defer rootCancel()
								prob, err := vg.sameRootProbability(rootCtx, lang, spokenWord, forbiddenWord)
								rootCh <- decisionResult{prob: prob, err: err}
							}()

							// Call A2 (concurrent): Decision 1 API (same spoken pronunciation?)
							go func() {
								defer pronCancel()
								prob, err := vg.samePronunciationProbability(pronCtx, lang, spokenWord, forbiddenWord)
								pronCh <- decisionResult{prob: prob, err: err}
							}()

							// Call B1 (concurrent): Gemini TTS for resembling verdict
							go func() {
								defer resemblingTTSCancel()
								audio, mimeType, err := vg.synthesizeJudgeSpeech(resemblingTTSCtx, lang, resemblingMessage)
								resemblingTTSCh <- ttsResult{audio: audio, mimeType: mimeType, err: err}
							}()

							// Call B2 (concurrent): Gemini TTS for exact verdict (if homophone / identical pronunciation)
							go func() {
								defer exactTTSCancel()
								audio, mimeType, err := vg.synthesizeJudgeSpeech(exactTTSCtx, lang, exactMessage)
								exactTTSCh <- ttsResult{audio: audio, mimeType: mimeType, err: err}
							}()

							decRes := <-rootCh
							if decRes.err != nil {
								gameLog.Error("decision 1 check failed",
									"spoken", spokenWord,
									"forbidden", forbiddenWord,
									"err", decRes.err,
								)
							} else {
								prob := decRes.prob
								decisionProb = &prob
								proximityPct := int(math.Round(prob * 100))
								gameLog.Info(
									fmt.Sprintf("Proximité entre '%s' et '%s': %d%%.", spokenWord, forbiddenWord, proximityPct),
									"spoken", spokenWord,
									"forbidden", forbiddenWord,
									"proximity_pct", proximityPct,
									"probability", prob,
								)
								if prob < sameRootProbabilityThreshold {
									pronCancel()
									resemblingTTSCancel()
									exactTTSCancel()
									ignoreAndKeepListening(
										fmt.Sprintf("Ignored: %q is not too close to %q (same-root probability %.0f%% < 75%%). Keep listening.", spokenWord, forbiddenWord, prob*100),
										decisionProb,
									)
									continue
								}
								reason = "resembling"
							}

							if !gameOver.CompareAndSwap(false, true) {
								pronCancel()
								resemblingTTSCancel()
								exactTTSCancel()
								return
							}

							samePronunciation := false
							pronRes := <-pronCh
							if pronRes.err != nil {
								gameLog.Warn("decision 1 pronunciation check failed",
									"spoken", spokenWord,
									"forbidden", forbiddenWord,
									"err", pronRes.err,
								)
							} else {
								pronPct := int(math.Round(pronRes.prob * 100))
								gameLog.Info(
									fmt.Sprintf("Prononciation identique entre '%s' et '%s': %d%%.", spokenWord, forbiddenWord, pronPct),
									"spoken", spokenWord,
									"forbidden", forbiddenWord,
									"same_pronunciation_pct", pronPct,
									"probability", pronRes.prob,
								)
								samePronunciation = pronRes.prob >= samePronunciationProbabilityThreshold
							}

							var ttsRes ttsResult
							if samePronunciation {
								reason = "exact"
								lossMessage = exactMessage
								resemblingTTSCancel()
								ttsRes = <-exactTTSCh
							} else {
								lossMessage = resemblingMessage
								exactTTSCancel()
								ttsRes = <-resemblingTTSCh
							}
							if ttsRes.err != nil {
								gameLog.Warn("judge TTS failed", "reason", reason, "err", ttsRes.err)
							} else {
								ttsAudio = ttsRes.audio
								ttsMimeType = ttsRes.mimeType
							}
						} else {
							if !gameOver.CompareAndSwap(false, true) {
								return
							}
							lossMessage = formatJudgeLossText(lang, spokenWord, forbiddenWord, reason)
							ttsCtx, ttsCancel := context.WithTimeout(ctx, judgeTTSTimeout)
							audio, mimeType, err := vg.synthesizeJudgeSpeech(ttsCtx, lang, lossMessage)
							ttsCancel()
							if err != nil {
								gameLog.Warn("judge TTS failed", "reason", reason, "err", err)
							} else {
								ttsAudio = audio
								ttsMimeType = mimeType
							}
						}

						recorder.RecordJudgeEvent(fc.Name, false, spokenWord, forbiddenWord, reason, decisionProb)
						recorder.SetOutcome(OutcomeLost)
						gameLog.Info("judge declared loss",
							"reason", reason,
							"spoken", spokenWord,
							"forbidden", forbiddenWord,
							"tts_bytes", len(ttsAudio),
						)
						verdict := &judgeVerdict{
							Lost:          true,
							ForbiddenWord: forbiddenWord,
							SpokenWord:    spokenWord,
							Reason:        reason,
							Message:       lossMessage,
						}
						if len(ttsAudio) > 0 {
							verdict.Audio = base64.StdEncoding.EncodeToString(ttsAudio)
							verdict.AudioMimeType = ttsMimeType
						}
						verdictBytes, err := json.Marshal(judgeMessage{Judge: verdict})
						if err != nil {
							gameLog.Error("marshal judge verdict failed", "err", err)
							return
						}
						if err := writeToBrowser(verdictBytes); err != nil {
							gameLog.Warn("write judge verdict failed", "err", err)
						}
						return
					}
				}
			}
		}
	}()

	for {
		// Human speech Loop:
		// Receive voice packets from the player browser, via WebSocket.
		// Forward them to both the Guesser's and the Judge's Gemini Live sessions.
		_, message, err := c.ReadMessage()
		if err != nil {
			if !gameOver.Load() {
				gameLog.Warn("read from client failed", "err", err)
			}
			gameOver.Store(true)
			break
		}
		if gameOver.Load() {
			break
		}

		var ctrl clientControlMessage
		if err := json.Unmarshal(message, &ctrl); err == nil && ctrl.Outcome != "" {
			switch ctrl.Outcome {
			case OutcomeWon, OutcomeLost, OutcomeAbandoned:
				recorder.SetOutcome(ctrl.Outcome)
				gameLog.Info("game ended", "outcome", ctrl.Outcome)
				gameOver.Store(true)
				return
			}
			continue
		}

		var realtimeInput genai.LiveRealtimeInput
		if err := json.Unmarshal(message, &realtimeInput); err != nil {
			gameLog.Warn("unmarshal client message failed", "payload", string(message), "err", err)
			continue
		}
		if realtimeInput.Audio != nil && len(realtimeInput.Audio.Data) > 0 {
			recorder.RecordHumanAudio(realtimeInput.Audio.Data, realtimeInput.Audio.MIMEType)
		}
		if err := session.SendRealtimeInput(realtimeInput); err != nil && !gameOver.Load() {
			gameLog.Warn("send audio to guesser failed", "err", err)
		}
		if realtimeInput.Audio != nil && len(realtimeInput.Audio.Data) > 0 {
			if err := jVAD.FeedAudio(realtimeInput.Audio.Data, realtimeInput.Audio.MIMEType); err != nil && !gameOver.Load() {
				gameLog.Warn("send audio to judge failed", "err", err)
			}
		} else if err := sendToJudge(realtimeInput); err != nil && !gameOver.Load() {
			gameLog.Warn("send audio to judge failed", "err", err)
		}
	}
}

func normalizeWord(s string) string {
	tr := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	normalized, _, err := transform.String(tr, strings.ToLower(strings.TrimSpace(s)))
	if err != nil {
		normalized = strings.ToLower(strings.TrimSpace(s))
	}
	normalized = strings.ReplaceAll(normalized, "œ", "oe")
	normalized = strings.ReplaceAll(normalized, "æ", "ae")
	normalized = strings.ReplaceAll(normalized, "’", "'")
	normalized = strings.ReplaceAll(normalized, "‘", "'")
	normalized = strings.ReplaceAll(normalized, "ʼ", "'")
	return normalized
}

// tokenizeForExactMatch splits a normalized phrase on whitespace and apostrophes
// (e.g. French elisions like "l'instrument" -> ["l", "instrument"]) while keeping
// internal hyphens intact (e.g. "tic-tac", "couvre-chef").
func tokenizeForExactMatch(s string) []string {
	normStr := normalizeWord(s)
	if normStr == "" {
		return nil
	}
	parts := strings.FieldsFunc(normStr, func(r rune) bool {
		return unicode.IsSpace(r) || r == '\''
	})
	tokens := make([]string, 0, len(parts))
	for _, p := range parts {
		trimmed := strings.TrimFunc(p, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})
		if trimmed != "" {
			tokens = append(tokens, trimmed)
		}
	}
	return tokens
}

// isExactForbiddenMatch reports whether spokenWord contains forbiddenWord as an exact
// token (or exact contiguous token sequence for multi-word forbidden words), ignoring
// surrounding articles, determiners, punctuation, and elided prefixes such as "l'" or "d'".
func isExactForbiddenMatch(spokenWord, forbiddenWord string) bool {
	spokenTokens := tokenizeForExactMatch(spokenWord)
	forbiddenTokens := tokenizeForExactMatch(forbiddenWord)
	if len(spokenTokens) == 0 || len(forbiddenTokens) == 0 || len(forbiddenTokens) > len(spokenTokens) {
		return false
	}
	for i := 0; i <= len(spokenTokens)-len(forbiddenTokens); i++ {
		matched := true
		for j := range forbiddenTokens {
			if spokenTokens[i+j] != forbiddenTokens[j] {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

const alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

func randomString(n int) string {
	a := make([]byte, n)
	for i := range a {
		a[i] = alphanum[rand.Intn(len(alphanum))]
	}
	return string(a)
}
