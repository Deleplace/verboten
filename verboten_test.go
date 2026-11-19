package verboten

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

	recordingsDir := t.TempDir()
	gameServer := NewServer(client)
	gameServer.Verbose = true
	gameServer.RecordingsDir = recordingsDir

	handlerDone := make(chan struct{})
	appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		gameServer.liveGame(w, r)
	}))
	defer appServer.Close()

	browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
		"/live/en?target=lemon&forbidden=lemon&forbidden=apple&forbidden=banana"
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

	// Verify Guesser does NOT know any of the forbidden words, and uses less eager end-of-speech VAD.
	for _, w := range []string{"apple", "banana", "lemon"} {
		if strings.Contains(strings.ToLower(guesserSess.setupRaw), w) {
			t.Errorf("guesser setup leaked forbidden word %q: %s", w, guesserSess.setupRaw)
		}
	}
	if !strings.Contains(guesserSess.setupRaw, `"END_SENSITIVITY_LOW"`) || !strings.Contains(guesserSess.setupRaw, `"silenceDurationMs":500`) {
		t.Errorf("guesser setup missing 500ms low-sensitivity VAD: %s", guesserSess.setupRaw)
	}
	// Verify Judge DOES know all forbidden words and keeps fast VAD.
	for _, w := range []string{"apple", "banana", "lemon"} {
		if !strings.Contains(judgeSess.setupRaw, w) {
			t.Errorf("judge setup missing forbidden word %q: %s", w, judgeSess.setupRaw)
		}
	}
	if !strings.Contains(judgeSess.setupRaw, `"END_SENSITIVITY_HIGH"`) || !strings.Contains(judgeSess.setupRaw, `"silenceDurationMs":100`) {
		t.Errorf("judge setup missing 100ms high-sensitivity VAD: %s", judgeSess.setupRaw)
	}

	// Drain initial setupComplete forwarded from Guesser to browser.
	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err = browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("read initial guesser message: %v", err)
	}

	// Browser sends a single 16kHz voice packet -> BOTH Guesser and Judge must receive it.
	humanPCM := []byte{0x01, 0x02, 0x03, 0x04}
	humanB64 := base64.StdEncoding.EncodeToString(humanPCM)
	audioMsg := `{"audio":{"data":"` + humanB64 + `","mimeType":"audio/pcm;rate=16000"}}`
	if err := browserConn.WriteMessage(websocket.TextMessage, []byte(audioMsg)); err != nil {
		t.Fatalf("browser write audio: %v", err)
	}

	for name, ch := range map[string]chan string{
		"guesser": guesserSess.audioPackets,
		"judge":   judgeSess.audioPackets,
	} {
		select {
		case pkt := <-ch:
			if !strings.Contains(pkt, humanB64) {
				t.Errorf("%s received unexpected packet: %s", name, pkt)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for %s to receive voice packet", name)
		}
	}

	// Guesser streams audio + transcription/guess back to browser.
	guesserPCM := []byte{0x10, 0x20, 0x30, 0x40, 0x50, 0x60}
	guesserB64 := base64.StdEncoding.EncodeToString(guesserPCM)
	guesserReply := `{"serverContent":{"inputTranscription":{"text":"yellow citrus"},"outputTranscription":{"text":"orange"},"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` + guesserB64 + `"}}]}}}`
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

	// If Judge mistakenly reports a translation of a NON-target forbidden word ("pomme" -> "apple"),
	// the server must ignore it and send a toolResponse so Judge keeps listening.
	nonTargetTranslationCall := `{"toolCall":{"functionCalls":[{"id":"call-0","name":"declare_loss","args":{"forbidden_word":"apple","spoken_word":"pomme","reason":"translation"}}]}}`
	if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(nonTargetTranslationCall)); err != nil {
		t.Fatalf("judge write non-target toolCall: %v", err)
	}
	select {
	case resp := <-judgeSess.audioPackets:
		if !strings.Contains(resp, "toolResponse") || !strings.Contains(resp, "call-0") {
			t.Fatalf("expected toolResponse for ignored non-target translation, got: %s", resp)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for toolResponse to judge on ignored non-target translation")
	}

	// Judge detects utterance of an exact translation of the TARGET word ("citron" -> "lemon") via tool call!
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

	// Close browser connection so liveGame finishes and finalizes the game recording folder.
	_ = browserConn.Close()
	select {
	case <-handlerDone:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for liveGame handler to finish and flush recording")
	}

	entries, err := os.ReadDir(recordingsDir)
	if err != nil {
		t.Fatalf("read recordingsDir: %v", err)
	}
	if len(entries) != 1 || !entries[0].IsDir() {
		t.Fatalf("expected 1 game recording folder, got %d", len(entries))
	}
	gameDir := filepath.Join(recordingsDir, entries[0].Name())

	metaBytes, err := os.ReadFile(filepath.Join(gameDir, "game.json"))
	if err != nil {
		t.Fatalf("read game.json: %v", err)
	}
	var rec GameRecording
	if err := json.Unmarshal(metaBytes, &rec); err != nil {
		t.Fatalf("unmarshal game.json: %v", err)
	}
	if rec.TargetWord != "lemon" {
		t.Errorf("recorded target_word = %q, want %q", rec.TargetWord, "lemon")
	}
	if len(rec.ForbiddenWords) != 2 || rec.ForbiddenWords[0] != "apple" || rec.ForbiddenWords[1] != "banana" {
		t.Errorf("recorded forbidden_words = %q, want [apple banana]", rec.ForbiddenWords)
	}
	if rec.HumanAudio == nil || rec.HumanAudio.File != "human.wav" || rec.HumanAudio.PacketCount != 1 || rec.HumanAudio.TotalBytes != int64(len(humanPCM)) || rec.HumanAudio.FirstPacketMs == nil {
		t.Errorf("unexpected human_audio metadata: %+v", rec.HumanAudio)
	}
	if rec.GuesserAudio == nil || rec.GuesserAudio.File != "guesser.wav" || rec.GuesserAudio.PacketCount != 1 || rec.GuesserAudio.TotalBytes != int64(len(guesserPCM)) || rec.GuesserAudio.FirstPacketMs == nil {
		t.Errorf("unexpected guesser_audio metadata: %+v", rec.GuesserAudio)
	}
	if len(rec.JudgeEvents) != 2 {
		t.Fatalf("expected 2 judge events (1 ignored + 1 loss), got %+v", rec.JudgeEvents)
	}
	if !rec.JudgeEvents[0].Ignored || rec.JudgeEvents[0].ForbiddenWord != "apple" || rec.JudgeEvents[0].SpokenWord != "pomme" {
		t.Errorf("unexpected first judge event: %+v", rec.JudgeEvents[0])
	}
	if rec.JudgeEvents[1].Ignored || rec.JudgeEvents[1].ForbiddenWord != "lemon" || rec.JudgeEvents[1].SpokenWord != "citron" || rec.JudgeEvents[1].Reason != "translation" {
		t.Errorf("unexpected second judge event: %+v", rec.JudgeEvents[1])
	}

	assertWAVFile(t, filepath.Join(gameDir, "human.wav"), 16000, humanPCM)
	assertWAVFile(t, filepath.Join(gameDir, "guesser.wav"), 24000, guesserPCM)

	eventsBytes, err := os.ReadFile(filepath.Join(gameDir, "events.jsonl"))
	if err != nil {
		t.Fatalf("read events.jsonl: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(eventsBytes)), "\n")
	if len(lines) < 6 {
		t.Fatalf("expected at least 6 events in events.jsonl, got %d:\n%s", len(lines), eventsBytes)
	}
	for i, line := range lines {
		var ev RecordingEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("events.jsonl line %d invalid JSON: %v", i, err)
		}
		if ev.Type == "" {
			t.Errorf("events.jsonl line %d missing type", i)
		}
	}
}

func assertWAVFile(t *testing.T, path string, wantSampleRate uint32, wantPCM []byte) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read WAV %s: %v", path, err)
	}
	if len(b) != wavHeaderSize+len(wantPCM) {
		t.Fatalf("WAV %s len = %d, want %d", path, len(b), wavHeaderSize+len(wantPCM))
	}
	if string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" || string(b[12:16]) != "fmt " || string(b[36:40]) != "data" {
		t.Fatalf("WAV %s invalid RIFF/WAVE header", path)
	}
	if gotRate := binary.LittleEndian.Uint32(b[24:28]); gotRate != wantSampleRate {
		t.Errorf("WAV %s sampleRate = %d, want %d", path, gotRate, wantSampleRate)
	}
	if gotDataSize := binary.LittleEndian.Uint32(b[40:44]); gotDataSize != uint32(len(wantPCM)) {
		t.Errorf("WAV %s dataSize = %d, want %d", path, gotDataSize, len(wantPCM))
	}
	if !bytes.Equal(b[wavHeaderSize:], wantPCM) {
		t.Errorf("WAV %s PCM payload mismatch", path)
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

			gameServer := NewServer(client)
			gameServer.RecordingsDir = t.TempDir()
			appServer := httptest.NewServer(http.HandlerFunc(gameServer.liveGame))
			defer appServer.Close()

			browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
				"/live/fr?target=Pizza&forbidden=Pizza&forbidden=Fromage&forbidden=P%C3%A2te"
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
					if !strings.Contains(setup, "EXACT TRANSLATION OF THE TARGET WORD IN ANOTHER LANGUAGE") {
						t.Errorf("judge setup missing target-word exact translation loss rule: %s", setup)
					}
					if !strings.Contains(setup, "Target word to guess: Pizza") {
						t.Errorf("judge setup missing target word: %s", setup)
					}
					if !strings.Contains(setup, "Do NOT check or penalize translations in other languages for the other forbidden words (Fromage, Pâte)") {
						t.Errorf("judge setup missing exclusion of other forbidden words from translation checks: %s", setup)
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
