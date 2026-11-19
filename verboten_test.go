package verboten

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

func TestDoubleLivestreamJudgeAndGuesser(t *testing.T) {
	type capturedSession struct {
		isJudge      bool
		setupRaw     string
		audioPackets chan string
		conn         *websocket.Conn
	}

	var (
		mu       sync.Mutex
		sessions []*capturedSession
		bothUp   = make(chan struct{})
	)

	fakeGemini := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("fake gemini upgrade: %v", err)
			return
		}

		// Read initial LiveClientSetup message.
		_, setupBytes, err := c.ReadMessage()
		if err != nil {
			t.Errorf("fake gemini read setup: %v", err)
			c.Close()
			return
		}
		setupStr := string(setupBytes)
		sess := &capturedSession{
			isJudge:      strings.Contains(setupStr, declareLossFuncName),
			setupRaw:     setupStr,
			audioPackets: make(chan string, 10),
			conn:         c,
		}

		// Acknowledge setupComplete.
		_ = c.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`))

		mu.Lock()
		sessions = append(sessions, sess)
		if len(sessions) == 2 {
			close(bothUp)
		}
		mu.Unlock()

		for {
			_, msg, err := c.ReadMessage()
			if err != nil {
				return
			}
			sess.audioPackets <- string(msg)
		}
	}))
	defer fakeGemini.Close()

	wsBaseURL := "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://")
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-api-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    wsBaseURL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	gameServer := NewServer(client)
	gameServer.Verbose = true

	appServer := httptest.NewServer(http.HandlerFunc(gameServer.liveGame))
	defer appServer.Close()

	browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
		"/live/en?forbidden=apple&forbidden=banana&forbidden=lemon"
	browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
	if err != nil {
		t.Fatalf("dial liveGame: %v", err)
	}
	defer browserConn.Close()

	select {
	case <-bothUp:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for both Guesser and Judge Gemini Live sessions to connect")
	}

	mu.Lock()
	var guesserSess, judgeSess *capturedSession
	for _, s := range sessions {
		if s.isJudge {
			judgeSess = s
		} else {
			guesserSess = s
		}
	}
	mu.Unlock()

	if guesserSess == nil || judgeSess == nil {
		t.Fatalf("expected 1 guesser and 1 judge session, got guesser=%v judge=%v", guesserSess != nil, judgeSess != nil)
	}
	defer guesserSess.conn.Close()
	defer judgeSess.conn.Close()

	// Verify Guesser does NOT know any of the forbidden words.
	for _, w := range []string{"apple", "banana", "lemon"} {
		if strings.Contains(strings.ToLower(guesserSess.setupRaw), w) {
			t.Errorf("guesser setup leaked forbidden word %q: %s", w, guesserSess.setupRaw)
		}
	}
	// Verify Judge DOES know all forbidden words.
	for _, w := range []string{"apple", "banana", "lemon"} {
		if !strings.Contains(judgeSess.setupRaw, w) {
			t.Errorf("judge setup missing forbidden word %q: %s", w, judgeSess.setupRaw)
		}
	}

	// Drain initial setupComplete forwarded from Guesser to browser.
	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err = browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("read initial guesser message: %v", err)
	}

	// Browser sends a single voice packet -> BOTH Guesser and Judge must receive it.
	audioMsg := `{"audio":{"data":"AAAA","mimeType":"audio/pcm"}}`
	if err := browserConn.WriteMessage(websocket.TextMessage, []byte(audioMsg)); err != nil {
		t.Fatalf("browser write audio: %v", err)
	}

	for name, ch := range map[string]chan string{
		"guesser": guesserSess.audioPackets,
		"judge":   judgeSess.audioPackets,
	} {
		select {
		case pkt := <-ch:
			if !strings.Contains(pkt, "AAAA") {
				t.Errorf("%s received unexpected packet: %s", name, pkt)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for %s to receive voice packet", name)
		}
	}

	// Guesser streams a transcription/guess back to browser.
	guesserReply := `{"serverContent":{"inputTranscription":{"text":"yellow citrus"},"outputTranscription":{"text":"orange"}}}`
	if err := guesserSess.conn.WriteMessage(websocket.TextMessage, []byte(guesserReply)); err != nil {
		t.Fatalf("guesser write reply: %v", err)
	}

	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, gotGuesserBytes, err := browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("browser read guesser reply: %v", err)
	}
	if !strings.Contains(string(gotGuesserBytes), "orange") {
		t.Fatalf("expected guesser reply to reach browser, got: %s", gotGuesserBytes)
	}

	// Judge detects utterance of an exact translation of a forbidden word ("citron" -> "lemon") via tool call!
	judgeToolCall := `{"toolCall":{"functionCalls":[{"id":"call-1","name":"declare_loss","args":{"forbidden_word":"lemon","spoken_word":"citron","reason":"translation"}}]}}`
	if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(judgeToolCall)); err != nil {
		t.Fatalf("judge write toolCall: %v", err)
	}

	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, gotJudgeBytes, err := browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("browser read judge verdict: %v", err)
	}

	var msg judgeMessage
	if err := json.Unmarshal(gotJudgeBytes, &msg); err != nil {
		t.Fatalf("unmarshal judge verdict: %v (%s)", err, gotJudgeBytes)
	}
	if msg.Judge == nil || !msg.Judge.Lost || msg.Judge.ForbiddenWord != "lemon" || msg.Judge.SpokenWord != "citron" || msg.Judge.Reason != "translation" {
		t.Fatalf("unexpected judge verdict: %+v", msg.Judge)
	}
}

func TestFrenchLanguageConfiguration(t *testing.T) {
	for _, backend := range []genai.Backend{genai.BackendGeminiAPI, genai.BackendVertexAI} {
		t.Run(backend.String(), func(t *testing.T) {
			var (
				mu     sync.Mutex
				setups []string
				bothUp = make(chan struct{})
			)

			fakeGemini := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				c, err := upgrader.Upgrade(w, r, nil)
				if err != nil {
					t.Errorf("fake gemini upgrade: %v", err)
					return
				}
				defer c.Close()

				_, setupBytes, err := c.ReadMessage()
				if err != nil {
					t.Errorf("fake gemini read setup: %v", err)
					return
				}
				_ = c.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`))

				mu.Lock()
				setups = append(setups, string(setupBytes))
				if len(setups) == 2 {
					close(bothUp)
				}
				mu.Unlock()

				for {
					if _, _, err := c.ReadMessage(); err != nil {
						return
					}
				}
			}))
			defer fakeGemini.Close()

			wsBaseURL := "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://")
			cfg := &genai.ClientConfig{
				Backend: backend,
				HTTPOptions: genai.HTTPOptions{
					BaseURL: wsBaseURL,
				},
			}
			if backend == genai.BackendVertexAI {
				cfg.HTTPOptions.APIVersion = "v1beta1"
			} else {
				cfg.APIKey = "test-api-key"
				cfg.HTTPOptions.APIVersion = "v1beta"
			}

			client, err := genai.NewClient(context.Background(), cfg)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}

			appServer := httptest.NewServer(http.HandlerFunc(NewServer(client).liveGame))
			defer appServer.Close()

			browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
				"/live/fr?forbidden=Fromage&forbidden=P%C3%A2te"
			browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
			if err != nil {
				t.Fatalf("dial liveGame: %v", err)
			}
			defer browserConn.Close()

			select {
			case <-bothUp:
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for both French Live sessions")
			}

			mu.Lock()
			defer mu.Unlock()
			for i, setup := range setups {
				if !strings.Contains(setup, `"languageCode":"fr-FR"`) {
					t.Errorf("session %d missing speechConfig languageCode fr-FR: %s", i, setup)
				}
				isJudge := strings.Contains(setup, declareLossFuncName)
				if isJudge {
					if !strings.Contains(setup, "Expect French spoken input") {
						t.Errorf("judge setup missing French spoken input instruction: %s", setup)
					}
					if !strings.Contains(setup, "EXACT TRANSLATION IN ANOTHER LANGUAGE") {
						t.Errorf("judge setup missing exact translation loss rule: %s", setup)
					}
					if !strings.Contains(setup, `"translation"`) {
						t.Errorf("judge setup missing translation reason enum in tool declaration: %s", setup)
					}
				} else {
					if !strings.Contains(setup, "entendre du français en entrée audio") {
						t.Errorf("guesser setup missing French spoken input instruction: %s", setup)
					}
					if backend == genai.BackendVertexAI && !strings.Contains(setup, `"languageCodes":["fr-FR"]`) {
						t.Errorf("vertex guesser setup missing transcription languageCodes fr-FR: %s", setup)
					}
				}
			}
		})
	}
}
