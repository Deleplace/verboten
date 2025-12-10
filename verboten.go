package verboten

import (
	"context"
	"encoding/json"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"text/template"

	_ "embed"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

type VerbotenGameServer struct {
	genaiClient *genai.Client
}

func NewServer(genaiClient *genai.Client) *VerbotenGameServer {
	return &VerbotenGameServer{
		genaiClient: genaiClient,
	}
}

const guesserPrompt = `
	You are playing the "guessing word" game where the human player with their microphone
	is describing a word. Your job is to listen to the description and say only one word as
	your guess, every few seconds. You have only 3 guesses.
	Don't say anything else than the word you're guessing.
`

const guesserPrompt_fr = `
	Vous jouez au jeu du "mot à deviner" où le joueur humain avec son microphone
	décrit un mot. Votre travail consiste à écouter la description et à ne dire qu'un seul mot comme
	votre suggestion, toutes les quelques secondes. Vous n'avez que 3 essais.
	Ne dites rien d'autre que le mot que vous devinez.
`

const guesserPrompt_ar = `
	أنت تلعب لعبة "تخمين الكلمات" حيث يقوم اللاعب البشري بميكروفونه بوصف كلمة.
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

func (vg *VerbotenGameServer) liveGame(w http.ResponseWriter, r *http.Request) {
	lang := strings.TrimPrefix(r.URL.Path, "/live/")
	var prompt string
	switch lang {
	case "en":
		prompt = guesserPrompt
	case "fr":
		prompt = guesserPrompt_fr
	case "ar":
		prompt = guesserPrompt_ar
	default:
		log.Printf("unsupported language: %q", lang)
		http.NotFound(w, r)
		return
	}

	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Fatal("upgrade error: ", err)
		return
	}
	defer c.Close()

	gameID := randomString(4)
	forbiddenWords := r.URL.Query()["forbidden"]
	log.Printf("Starting game %s in %s with proscribed words %q", gameID, lang, forbiddenWords)

	ctx := context.Background()

	var model string
	if vg.genaiClient.ClientConfig().Backend == genai.BackendVertexAI {
		model = "gemini-live-2.5-flash-preview-native-audio-09-2025"
	} else {
		model = "gemini-2.5-flash-native-audio-preview-09-2025"
	}

	// Gemini Live session 1 : model listens to the human and guesses the secret word
	config := &genai.LiveConnectConfig{}
	config.SystemInstruction = &genai.Content{
		Parts: []*genai.Part{
			{Text: prompt},
		},
	}
	voiceName := "Puck"
	config.SpeechConfig = &genai.SpeechConfig{
		VoiceConfig: &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
				VoiceName: voiceName,
			},
		},
	}
	config.ResponseModalities = []genai.Modality{genai.ModalityAudio}
	config.InputAudioTranscription = &genai.AudioTranscriptionConfig{}
	config.OutputAudioTranscription = &genai.AudioTranscriptionConfig{}
	var shortDuration int32 = 100
	config.RealtimeInputConfig = &genai.RealtimeInputConfig{
		AutomaticActivityDetection: &genai.AutomaticActivityDetection{

			StartOfSpeechSensitivity: "START_SENSITIVITY_HIGH",
			EndOfSpeechSensitivity:   "END_SENSITIVITY_HIGH",
			PrefixPaddingMs:          &shortDuration,
			SilenceDurationMs:        &shortDuration,
		},
	}
	session, err := vg.genaiClient.Live.Connect(ctx, model, config)
	if err != nil {
		log.Fatal("connect to model error: ", err)
	}
	defer session.Close()

	// Gemini Live session 2 : model listens to the human and guesses the secret word
	configJudge := &genai.LiveConnectConfig{}
	configJudge.SystemInstruction = &genai.Content{
		Parts: []*genai.Part{
			{Text: `
				You're a judge listening to a human player of Proscribed Words, who is not allowed to
				say any of the words from the proscribed list. If the human player says any of them,
				or a very close word with the same radical, or one of the words translated in aother
				language, then pronounce only the phrase from the human that violated the rule.

				The proscribed words are: ` + strings.Join(forbiddenWords, ", ")},
		},
	}
	configJudge.ResponseModalities = []genai.Modality{genai.ModalityAudio}
	configJudge.OutputAudioTranscription = &genai.AudioTranscriptionConfig{}
	sessionJudge, err := vg.genaiClient.Live.Connect(ctx, model, configJudge)
	if err != nil {
		log.Fatal("connect to model error: ", err)
	}
	defer sessionJudge.Close()

	go func() {
		// Guessing Loop:
		// Receive audio data from the Gemini Live session.
		// Forward it to the player browser, via WebSocket.
		for {
			message, err := session.Receive()
			if err != nil {
				log.Println("guesser model deconnected: ", err)
				return
			}
			messageBytes, err := json.Marshal(message)
			if err != nil {
				log.Fatal("marshal guesser model response error: ", message, err)
			}
			err = c.WriteMessage(websocket.TextMessage, messageBytes)
			if err != nil {
				log.Println("write message error: ", err)
				break
			}
		}
	}()

	for {
		// Human speech Loop:
		// Receive audio  and transcript data from player browser, via WebSocket.
		// Forward it to the model guesser player's Gemini Live session.
		// Also forward it to the model judge's Gemini Live session.
		_, message, err := c.ReadMessage()
		if err != nil {
			log.Println("read from client error: ", err)
			break
		}

		var realtimeInput genai.LiveRealtimeInput
		if err := json.Unmarshal(message, &realtimeInput); err != nil {
			log.Fatal("unmarshal message error ", string(message), err)
		}
		session.SendRealtimeInput(realtimeInput)
		sessionJudge.SendRealtimeInput(realtimeInput)
	}

	go func() {
		// Judge Loop:
		// Receive audio and transcript data from the Gemini Live session.
		// Signal to the browser to end the game.
		for {
			message, err := sessionJudge.Receive()
			if err != nil {
				log.Println("judge deconnected: ", err)
				return
			}
			sc := message.ServerContent
			if sc != nil {
				ot := sc.OutputTranscription
				if ot != nil {
					log.Printf("Game %s Judge says %q", gameID, ot.Text)
					// TODO err = c.WriteMessage(websocket.TextMessage, messageBytes)
				}
			}
		}
	}()
}

const alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

func randomString(n int) string {
	a := make([]byte, n)
	for i := range a {
		a[i] = alphanum[rand.Intn(len(alphanum))]
	}
	return string(a)
}
