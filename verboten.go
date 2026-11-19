package verboten

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"text/template"

	_ "embed"

	"github.com/gorilla/websocket"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"
)

type VerbotenGameServer struct {
	genaiClient *genai.Client
	Verbose     bool
}

func NewServer(genaiClient *genai.Client) *VerbotenGameServer {
	return &VerbotenGameServer{
		genaiClient: genaiClient,
	}
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

func (vg *VerbotenGameServer) Start(ctx context.Context) error {
	log.SetFlags(0)
	http.HandleFunc("/", vg.serveGame)
	http.HandleFunc("/live/", vg.liveGame)
	http.HandleFunc("/words.json", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "assets/words.json")
	})
	http.Handle("/forbiddenwords/", http.StripPrefix("/forbiddenwords/", http.FileServer(http.Dir("assets"))))

	// Determine port for HTTP service.
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
		log.Printf("defaulting to port %s", port)
	}

	// Start HTTP server.
	log.Printf("listening on port %s", port)
	return http.ListenAndServe(":"+port, nil)
}

//go:embed assets/verboten.html
var gameWebapp string

func (vg *VerbotenGameServer) serveGame(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.New("game").Parse(gameWebapp)
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

type judgeVerdict struct {
	Lost          bool   `json:"lost"`
	ForbiddenWord string `json:"forbiddenWord,omitempty"`
	SpokenWord    string `json:"spokenWord,omitempty"`
	Reason        string `json:"reason,omitempty"`
}

func judgeSystemInstruction(langName, languageCode string, forbiddenWords []string) string {
	return fmt.Sprintf(`You are the strict Judge in a live voice game of "Verboten" (Proscribed Words), played in %s (%s).
You are listening directly to the human player's microphone as they describe a secret word to an AI guesser.
Expect %s spoken input most of the time, but stay strictly alert for words spoken in ANY other language (English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, etc.), because translating a forbidden word into another language is an automatic loss.

Here are the forbidden words: %s.

STRICT LOSS CONDITIONS — You MUST immediately interrupt the game and call %s if the human player pronounces ANY of the following:
1. EXACT FORBIDDEN WORD (reason="exact"): Any word from the forbidden list above.
2. VERY RESEMBLING WORD (reason="resembling"): Any inflection, plural, feminine, verb conjugation, derivative, compound, or word sharing the same root/stem as a forbidden word (e.g., "poussent" for "Pousser", "cheesy" for "Cheese", "arbres" for "Arbre").
3. EXACT TRANSLATION IN ANOTHER LANGUAGE (reason="translation"): Any word that is the exact translation of ANY forbidden word in ANY other language — whether intentional or accidental, whether spoken alone or inside a %s sentence, and including plurals or inflections of that translation!
   Examples of FORBIDDEN exact translations (MUST immediately call %s with reason="translation"):
   - Forbidden "Fromage" -> player says "cheese", "queso", "Käse", "formaggio" -> LOST!
   - Forbidden "Eau" -> player says "water", "agua", "Wasser", "acqua" -> LOST!
   - Forbidden "Jaune" -> player says "yellow", "amarillo", "gelb", "giallo" -> LOST!
   - Forbidden "Arbre" -> player says "tree", "trees", "árbol", "Baum", "albero" -> LOST!
   - Forbidden "Chapeau" -> player says "hat", "sombrero", "Hut", "cappello" -> LOST!
   - Forbidden "Cheese" -> player says "fromage", "queso", "Käse", "formaggio" -> LOST!
   - Forbidden "Moon" -> player says "lune", "luna", "Mond" -> LOST!

ALLOWED (do NOT call %s):
- Same-language synonyms, descriptions, or semantically related words in %s that do NOT share the same root as any forbidden word AND are NOT an exact translation of any forbidden word in another language (for example, saying "mozzarella" when "Cheese" is forbidden, "ficelle" when "Corde" is forbidden, or "tronc" when "Arbre" is forbidden, is valid and must NOT end the game).
- Unclear background noise or hesitation sounds ("um", "euh") unless a forbidden word, resembling word, or exact translation in another language was clearly spoken.

RESPONSE BEHAVIOR:
- If the human player violates condition 1, 2, or 3 (exact forbidden word, very resembling word, OR exact translation in another language), IMMEDIATELY call %s with forbidden_word, spoken_word, and reason.
- Only if NONE of conditions 1, 2, or 3 were violated, do NOT call %s; simply reply "OK" and keep listening.`,
		langName,
		languageCode,
		langName,
		strings.Join(forbiddenWords, ", "),
		declareLossFuncName,
		langName,
		declareLossFuncName,
		declareLossFuncName,
		langName,
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
		log.Printf("unsupported language: %q", lang)
		http.NotFound(w, r)
		return
	}

	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("upgrade error: ", err)
		return
	}
	defer c.Close()

	gameID := randomString(4)
	forbiddenWords := r.URL.Query()["forbidden"]
	log.Printf("Starting game %s in %s with proscribed words %q", gameID, lang, forbiddenWords)

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var model string
	if vg.genaiClient.ClientConfig().Backend == genai.BackendVertexAI {
		model = "gemini-live-2.5-flash-preview-native-audio-09-2025"
	} else {
		model = "gemini-3.1-flash-live-preview"
	}
	log.Printf("Using model %q", model)

	var shortDuration int32 = 100
	realtimeInputConfig := &genai.RealtimeInputConfig{
		AutomaticActivityDetection: &genai.AutomaticActivityDetection{
			StartOfSpeechSensitivity: "START_SENSITIVITY_HIGH",
			EndOfSpeechSensitivity:   "END_SENSITIVITY_HIGH",
			PrefixPaddingMs:          &shortDuration,
			SilenceDurationMs:        &shortDuration,
		},
	}

	var audioTranscriptionConfig *genai.AudioTranscriptionConfig
	if vg.genaiClient.ClientConfig().Backend == genai.BackendVertexAI {
		audioTranscriptionConfig = &genai.AudioTranscriptionConfig{
			LanguageCodes: []string{languageCode},
		}
	} else {
		audioTranscriptionConfig = &genai.AudioTranscriptionConfig{}
	}

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
		RealtimeInputConfig:      realtimeInputConfig,
	}

	// Gemini Live session 2 (Judge): listens to the same human voice stream, knows the
	// forbidden words, and calls declare_loss if the human utters one (or a resembling word).
	configJudge := &genai.LiveConnectConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{Text: judgeSystemInstruction(langName, languageCode, forbiddenWords)},
			},
		},
		SpeechConfig: &genai.SpeechConfig{
			LanguageCode: languageCode,
		},
		ResponseModalities:  []genai.Modality{genai.ModalityAudio},
		RealtimeInputConfig: realtimeInputConfig,
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        declareLossFuncName,
						Description: "Interrupt the game and declare the game is lost when the human player utters a forbidden word, a very resembling word (same root/inflection), or the exact translation of a forbidden word in another language.",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"forbidden_word": {
									Type:        genai.TypeString,
									Description: "The exact word from the forbidden list that was violated.",
								},
								"spoken_word": {
									Type:        genai.TypeString,
									Description: "The word or phrase spoken by the human player (in any language) that triggered the loss.",
								},
								"reason": {
									Type:        genai.TypeString,
									Enum:        []string{"exact", "resembling", "translation"},
									Description: "Why the game is lost: 'exact' for a forbidden word itself, 'resembling' for an inflection or same-root word, or 'translation' for the exact translation of a forbidden word in another language.",
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
		log.Printf("Game %s connect error: %v", gameID, err)
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
		writeMu  sync.Mutex
		gameOver atomic.Bool
	)
	writeToBrowser := func(payload []byte) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return c.WriteMessage(websocket.TextMessage, payload)
	}

	go func() {
		// Guessing Loop:
		// Receive audio and transcription data from the Guesser Gemini Live session.
		// Forward it to the player browser, via WebSocket.
		for {
			message, err := session.Receive()
			if err != nil {
				if !gameOver.Load() {
					log.Println("guesser model deconnected: ", err)
				}
				return
			}
			if gameOver.Load() {
				return
			}
			if vg.Verbose && message.ServerContent != nil {
				if it := message.ServerContent.InputTranscription; it != nil && strings.TrimSpace(it.Text) != "" {
					log.Printf("Game %s Human says %q", gameID, it.Text)
				}
				if ot := message.ServerContent.OutputTranscription; ot != nil && strings.TrimSpace(ot.Text) != "" {
					log.Printf("Game %s Guesser says %q", gameID, ot.Text)
				}
			}
			messageBytes, err := json.Marshal(message)
			if err != nil {
				log.Println("marshal guesser model response error: ", message, err)
				return
			}
			if err := writeToBrowser(messageBytes); err != nil {
				log.Println("write message error: ", err)
				return
			}
		}
	}()

	go func() {
		// Judge Loop:
		// Receive tool calls from the Judge Gemini Live session when a forbidden word is uttered.
		// Signal to the browser to interrupt and end the game.
		for {
			message, err := sessionJudge.Receive()
			if err != nil {
				if !gameOver.Load() {
					log.Println("judge deconnected: ", err)
				}
				return
			}
			if message.ToolCall != nil {
				for _, fc := range message.ToolCall.FunctionCalls {
					if fc.Name == declareLossFuncName {
						forbiddenWord, _ := fc.Args["forbidden_word"].(string)
						spokenWord, _ := fc.Args["spoken_word"].(string)
						reason, _ := fc.Args["reason"].(string)
						if !gameOver.CompareAndSwap(false, true) {
							return
						}
						if reason != "" {
							log.Printf("Game %s Judge declared loss (%s): spoken=%q forbidden=%q", gameID, reason, spokenWord, forbiddenWord)
						} else {
							log.Printf("Game %s Judge declared loss: spoken=%q forbidden=%q", gameID, spokenWord, forbiddenWord)
						}
						verdictBytes, err := json.Marshal(judgeMessage{
							Judge: &judgeVerdict{
								Lost:          true,
								ForbiddenWord: forbiddenWord,
								SpokenWord:    spokenWord,
								Reason:        reason,
							},
						})
						if err != nil {
							log.Println("marshal judge verdict error: ", err)
							return
						}
						if err := writeToBrowser(verdictBytes); err != nil {
							log.Println("write judge verdict error: ", err)
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
				log.Println("read from client error: ", err)
			}
			gameOver.Store(true)
			break
		}
		if gameOver.Load() {
			break
		}

		var realtimeInput genai.LiveRealtimeInput
		if err := json.Unmarshal(message, &realtimeInput); err != nil {
			log.Println("unmarshal message error: ", string(message), err)
			continue
		}
		if err := session.SendRealtimeInput(realtimeInput); err != nil && !gameOver.Load() {
			log.Println("send to guesser error: ", err)
		}
		if err := sessionJudge.SendRealtimeInput(realtimeInput); err != nil && !gameOver.Load() {
			log.Println("send to judge error: ", err)
		}
	}
}

const alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

func randomString(n int) string {
	a := make([]byte, n)
	for i := range a {
		a[i] = alphanum[rand.Intn(len(alphanum))]
	}
	return string(a)
}
