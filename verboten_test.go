package verboten

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
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
	if !strings.Contains(judgeSess.setupRaw, `"disabled":true`) || !strings.Contains(judgeSess.setupRaw, `"NO_INTERRUPTION"`) {
		t.Errorf("judge setup missing manual VAD + NO_INTERRUPTION: %s", judgeSess.setupRaw)
	}

	// Drain initial setupComplete forwarded from Guesser to browser.
	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err = browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("read initial guesser message: %v", err)
	}

	// Browser sends a single 16kHz voice packet -> BOTH Guesser and Judge must receive it
	// (with Judge receiving an ActivityStart signal before the speech packet).
	humanPCM := []byte{0x01, 0x02, 0x03, 0x04}
	humanB64 := base64.StdEncoding.EncodeToString(humanPCM)
	audioMsg := `{"audio":{"data":"` + humanB64 + `","mimeType":"audio/pcm;rate=16000"}}`
	if err := browserConn.WriteMessage(websocket.TextMessage, []byte(audioMsg)); err != nil {
		t.Fatalf("browser write audio: %v", err)
	}

	select {
	case pkt := <-guesserSess.audioPackets:
		if !strings.Contains(pkt, humanB64) {
			t.Errorf("guesser received unexpected packet: %s", pkt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for guesser to receive voice packet")
	}

	select {
	case pkt := <-judgeSess.audioPackets:
		if !strings.Contains(pkt, `"activityStart"`) {
			t.Errorf("judge expected activityStart before speech audio, got: %s", pkt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for judge activityStart")
	}
	select {
	case pkt := <-judgeSess.audioPackets:
		if !strings.Contains(pkt, humanB64) {
			t.Errorf("judge received unexpected audio packet: %s", pkt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for judge to receive voice packet")
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
	if rec.Outcome != OutcomeLost {
		t.Errorf("recorded outcome = %q, want %q", rec.Outcome, OutcomeLost)
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
	gameServer.RecordingsDir = t.TempDir()
	handlerDone := make(chan struct{})
	appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		gameServer.liveGame(w, r)
	}))
	defer appServer.Close()

	browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
		"/live/fr?target=Pizza&forbidden=Pizza&forbidden=Fromage&forbidden=P%C3%A2te"
	browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
	if err != nil {
		t.Fatalf("dial liveGame: %v", err)
	}
	defer func() {
		_ = browserConn.Close()
		select {
		case <-handlerDone:
		case <-time.After(3 * time.Second):
		}
	}()

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
		if !strings.Contains(setup, DefaultLiveModel) {
			t.Errorf("session %d missing default live model %q: %s", i, DefaultLiveModel, setup)
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
		}
	}
}

func TestDecisionSameRootWireFormat(t *testing.T) {
	var (
		gotPath string
		gotBody string
	)
	fakeDecisionServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answers":{"same_root":{"type":"noul","noul":0.12,"confidence":0.88}}}`))
	}))
	defer fakeDecisionServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-api-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    fakeDecisionServer.URL + "/",
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	server := NewServer(client)
	prob, err := server.sameRootProbability(context.Background(), "fr", "flamme", "feu")
	if err != nil {
		t.Fatalf("sameRootProbability: %v", err)
	}
	if prob != 0.12 {
		t.Fatalf("prob = %v, want 0.12", prob)
	}
	if gotPath != "/v1beta/decisions" {
		t.Errorf("request path = %q, want %q", gotPath, "/v1beta/decisions")
	}
	wantQuestion := "Le mot 'flamme' est-il trop proche du mot 'feu' pour le jeu des mots prohibés, car ayant la même racine?"
	if !strings.Contains(gotBody, `"model":"gemini-decision-1"`) {
		t.Errorf("request body missing model gemini-decision-1: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"type":"noul"`) {
		t.Errorf("request body missing noul question type: %s", gotBody)
	}
	if !strings.Contains(gotBody, wantQuestion) {
		t.Errorf("request body missing expected French question %q: %s", wantQuestion, gotBody)
	}
}

func TestDecisionSameRootFilteringInLiveGame(t *testing.T) {
	type capturedSession struct {
		isJudge  bool
		incoming chan string
		conn     *websocket.Conn
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
		_, setupBytes, err := c.ReadMessage()
		if err != nil {
			c.Close()
			return
		}
		sess := &capturedSession{
			isJudge:  strings.Contains(string(setupBytes), declareLossFuncName),
			incoming: make(chan string, 10),
			conn:     c,
		}
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
			sess.incoming <- string(msg)
		}
	}))
	defer fakeGemini.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-api-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	recordingsDir := t.TempDir()
	gameServer := NewServer(client)
	gameServer.RecordingsDir = recordingsDir
	gameServer.sameRootProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
		switch normalizeWord(spoken) + ":" + normalizeWord(forbidden) {
		case "flamme:feu":
			return 0.12, nil
		case "grand:flamme":
			return 0.02, nil
		case "porte:porter":
			return 0.94, nil
		default:
			return 0.0, nil
		}
	}

	handlerDone := make(chan struct{})
	appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		gameServer.liveGame(w, r)
	}))
	defer appServer.Close()

	browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
		"/live/fr?target=Dragon&forbidden=Dragon&forbidden=Feu&forbidden=Porter"
	browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
	if err != nil {
		t.Fatalf("dial liveGame: %v", err)
	}
	defer browserConn.Close()

	select {
	case <-bothUp:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for both sessions")
	}

	mu.Lock()
	var judgeSess, guesserSess *capturedSession
	for _, s := range sessions {
		if s.isJudge {
			judgeSess = s
		} else {
			guesserSess = s
		}
	}
	mu.Unlock()
	defer judgeSess.conn.Close()
	defer guesserSess.conn.Close()

	// Drain initial setupComplete.
	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, _, err := browserConn.ReadMessage(); err != nil {
		t.Fatalf("read setupComplete: %v", err)
	}

	// 1. Judge falsely flags "flamme" vs "Feu" (Decision 1 returns 0.12 < 0.75) -> must be ignored.
	call1 := `{"toolCall":{"functionCalls":[{"id":"c1","name":"declare_loss","args":{"forbidden_word":"Feu","spoken_word":"flamme","reason":"resembling"}}]}}`
	if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(call1)); err != nil {
		t.Fatalf("write call1: %v", err)
	}
	select {
	case resp := <-judgeSess.incoming:
		if !strings.Contains(resp, "toolResponse") || !strings.Contains(resp, "c1") {
			t.Fatalf("expected toolResponse for rejected flamme/Feu, got: %s", resp)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for toolResponse for flamme/Feu")
	}

	// 2. Judge falsely flags "Grand" vs "flamme" (Decision 1 returns 0.02 < 0.75) -> must be ignored.
	call2 := `{"toolCall":{"functionCalls":[{"id":"c2","name":"declare_loss","args":{"forbidden_word":"flamme","spoken_word":"Grand","reason":"resembling"}}]}}`
	if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(call2)); err != nil {
		t.Fatalf("write call2: %v", err)
	}
	select {
	case resp := <-judgeSess.incoming:
		if !strings.Contains(resp, "toolResponse") || !strings.Contains(resp, "c2") {
			t.Fatalf("expected toolResponse for rejected Grand/flamme, got: %s", resp)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for toolResponse for Grand/flamme")
	}

	// 3. Judge flags "porte" vs "Porter" (Decision 1 returns 0.94 >= 0.75) -> must end the game!
	call3 := `{"toolCall":{"functionCalls":[{"id":"c3","name":"declare_loss","args":{"forbidden_word":"Porter","spoken_word":"porte","reason":"resembling"}}]}}`
	if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(call3)); err != nil {
		t.Fatalf("write call3: %v", err)
	}
	_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, gotBytes, err := browserConn.ReadMessage()
	if err != nil {
		t.Fatalf("read browser verdict: %v", err)
	}
	var verdict judgeMessage
	if err := json.Unmarshal(gotBytes, &verdict); err != nil {
		t.Fatalf("unmarshal verdict: %v", err)
	}
	if verdict.Judge == nil || !verdict.Judge.Lost || verdict.Judge.SpokenWord != "porte" || verdict.Judge.ForbiddenWord != "Porter" {
		t.Fatalf("unexpected verdict: %+v", verdict.Judge)
	}

	_ = browserConn.Close()
	<-handlerDone

	// Verify all 3 judge events (2 ignored + 1 confirmed) were recorded with their Decision 1 probabilities.
	entries, err := os.ReadDir(recordingsDir)
	if err != nil || len(entries) != 1 {
		t.Fatalf("expected 1 recording dir, got %d (err=%v)", len(entries), err)
	}
	rec, err := os.ReadFile(filepath.Join(recordingsDir, entries[0].Name(), "game.json"))
	if err != nil {
		t.Fatalf("read game.json: %v", err)
	}
	var gr GameRecording
	if err := json.Unmarshal(rec, &gr); err != nil {
		t.Fatalf("unmarshal game.json: %v", err)
	}
	if gr.Outcome != OutcomeLost {
		t.Errorf("recorded outcome = %q, want %q", gr.Outcome, OutcomeLost)
	}
	if len(gr.JudgeEvents) != 3 {
		t.Fatalf("expected 3 judge events, got %+v", gr.JudgeEvents)
	}
	if !gr.JudgeEvents[0].Ignored || gr.JudgeEvents[0].DecisionProbability == nil || *gr.JudgeEvents[0].DecisionProbability != 0.12 {
		t.Errorf("unexpected event 0: %+v", gr.JudgeEvents[0])
	}
	if !gr.JudgeEvents[1].Ignored || gr.JudgeEvents[1].DecisionProbability == nil || *gr.JudgeEvents[1].DecisionProbability != 0.02 {
		t.Errorf("unexpected event 1: %+v", gr.JudgeEvents[1])
	}
	if gr.JudgeEvents[2].Ignored || gr.JudgeEvents[2].DecisionProbability == nil || *gr.JudgeEvents[2].DecisionProbability != 0.94 {
		t.Errorf("unexpected event 2: %+v", gr.JudgeEvents[2])
	}
}

func TestIsExactForbiddenMatch(t *testing.T) {
	cases := []struct {
		spoken    string
		forbidden string
		want      bool
	}{
		{"instrument", "Instrument", true},
		{"l'instrument", "Instrument", true},
		{"l’instrument", "Instrument", true},
		{"d'arbre", "Arbre", true},
		{"un instrument", "Instrument", true},
		{"C'est l'instrument !", "Instrument", true},
		{"l'eau", "Eau", true},
		{"Tic-tac", "tic-tac", true},
		{"an ice cream!", "Ice cream", true},
		{"l'œuf", "Oeuf", true},
		{"flamme", "Feu", false},
		{"joue", "Jouer", false},
		{"instruments", "Instrument", false},
		{"couvre-chef", "Chef", false},
		{"ice", "Ice cream", false},
		{"", "Instrument", false},
		{"instrument", "", false},
	}
	for _, tc := range cases {
		if got := isExactForbiddenMatch(tc.spoken, tc.forbidden); got != tc.want {
			t.Errorf("isExactForbiddenMatch(%q, %q) = %v, want %v", tc.spoken, tc.forbidden, got, tc.want)
		}
	}
}

func TestExactMatchWithArticleSkipsDecision1InLiveGame(t *testing.T) {
	cases := []struct {
		name              string
		spoken            string
		forbidden         string
		judgeReason       string
		wantDecisionCalls int
		wantReason        string
	}{
		{
			name:              "elided_article_exact",
			spoken:            "l'instrument",
			forbidden:         "Instrument",
			judgeReason:       "exact",
			wantDecisionCalls: 0,
			wantReason:        "exact",
		},
		{
			name:              "elided_article_mislabeled_resembling",
			spoken:            "d'arbre",
			forbidden:         "Arbre",
			judgeReason:       "resembling",
			wantDecisionCalls: 0,
			wantReason:        "exact",
		},
		{
			name:              "resembling_mislabeled_exact",
			spoken:            "porte",
			forbidden:         "Porter",
			judgeReason:       "exact",
			wantDecisionCalls: 1,
			wantReason:        "resembling",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			type capturedSession struct {
				isJudge bool
				conn    *websocket.Conn
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
				_, setupBytes, err := c.ReadMessage()
				if err != nil {
					c.Close()
					return
				}
				sess := &capturedSession{
					isJudge: strings.Contains(string(setupBytes), declareLossFuncName),
					conn:    c,
				}
				_ = c.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`))

				mu.Lock()
				sessions = append(sessions, sess)
				if len(sessions) == 2 {
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

			client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
				APIKey:  "test-api-key",
				Backend: genai.BackendGeminiAPI,
				HTTPOptions: genai.HTTPOptions{
					BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
					APIVersion: "v1beta",
				},
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}

			var (
				callMu        sync.Mutex
				decisionCalls int
			)
			gameServer := NewServer(client)
			gameServer.RecordingsDir = t.TempDir()
			gameServer.sameRootProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
				callMu.Lock()
				decisionCalls++
				callMu.Unlock()
				return 0.94, nil
			}

			handlerDone := make(chan struct{})
			appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				gameServer.liveGame(w, r)
			}))
			defer appServer.Close()

			browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
				"/live/fr?target=Guitare&forbidden=Guitare&forbidden=Instrument&forbidden=Arbre&forbidden=Porter"
			browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
			if err != nil {
				t.Fatalf("dial liveGame: %v", err)
			}
			defer func() {
				_ = browserConn.Close()
				select {
				case <-handlerDone:
				case <-time.After(3 * time.Second):
				}
			}()

			select {
			case <-bothUp:
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for both sessions")
			}

			mu.Lock()
			var judgeSess, guesserSess *capturedSession
			for _, s := range sessions {
				if s.isJudge {
					judgeSess = s
				} else {
					guesserSess = s
				}
			}
			mu.Unlock()
			defer judgeSess.conn.Close()
			defer guesserSess.conn.Close()

			_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
			if _, _, err := browserConn.ReadMessage(); err != nil {
				t.Fatalf("read setupComplete: %v", err)
			}

			toolCallBytes, err := json.Marshal(map[string]any{
				"toolCall": map[string]any{
					"functionCalls": []map[string]any{
						{
							"id":   "c1",
							"name": declareLossFuncName,
							"args": map[string]any{
								"forbidden_word": tc.forbidden,
								"spoken_word":    tc.spoken,
								"reason":         tc.judgeReason,
							},
						},
					},
				},
			})
			if err != nil {
				t.Fatalf("marshal toolCall: %v", err)
			}
			if err := judgeSess.conn.WriteMessage(websocket.TextMessage, toolCallBytes); err != nil {
				t.Fatalf("write toolCall: %v", err)
			}

			_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
			_, gotBytes, err := browserConn.ReadMessage()
			if err != nil {
				t.Fatalf("read browser verdict: %v", err)
			}
			var verdict judgeMessage
			if err := json.Unmarshal(gotBytes, &verdict); err != nil {
				t.Fatalf("unmarshal verdict: %v", err)
			}
			if verdict.Judge == nil || !verdict.Judge.Lost {
				t.Fatalf("expected loss verdict, got: %+v", verdict.Judge)
			}
			if verdict.Judge.Reason != tc.wantReason {
				t.Errorf("verdict.Reason = %q, want %q", verdict.Judge.Reason, tc.wantReason)
			}
			if verdict.Judge.ForbiddenWord != tc.forbidden {
				t.Errorf("verdict.ForbiddenWord = %q, want %q", verdict.Judge.ForbiddenWord, tc.forbidden)
			}

			callMu.Lock()
			gotCalls := decisionCalls
			callMu.Unlock()
			if gotCalls != tc.wantDecisionCalls {
				t.Errorf("Decision 1 calls = %d, want %d", gotCalls, tc.wantDecisionCalls)
			}
		})
	}
}

func TestRecordedGameOutcomes(t *testing.T) {
	cases := []struct {
		name           string
		guesserReplies []string
		sendOutcomeMsg string
		wantOutcome    string
	}{
		{
			name: "explicit_won",
			guesserReplies: []string{
				`{"serverContent":{"outputTranscription":{"text":"Lemon"}}}`,
			},
			sendOutcomeMsg: `{"outcome":"won"}`,
			wantOutcome:    OutcomeWon,
		},
		{
			name: "inferred_won_chunked_transcript",
			guesserReplies: []string{
				`{"serverContent":{"outputTranscription":{"text":"Lem"}}}`,
				`{"serverContent":{"outputTranscription":{"text":"on?"}}}`,
			},
			wantOutcome: OutcomeWon,
		},
		{
			name: "abandoned",
			guesserReplies: []string{
				`{"serverContent":{"outputTranscription":{"text":"Orange"}}}`,
			},
			wantOutcome: OutcomeAbandoned,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			type capturedSession struct {
				isJudge  bool
				incoming chan string
				conn     *websocket.Conn
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
				_, setupBytes, err := c.ReadMessage()
				if err != nil {
					c.Close()
					return
				}
				sess := &capturedSession{
					isJudge:  strings.Contains(string(setupBytes), declareLossFuncName),
					incoming: make(chan string, 10),
					conn:     c,
				}
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
					sess.incoming <- string(msg)
				}
			}))
			defer fakeGemini.Close()

			client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
				APIKey:  "test-api-key",
				Backend: genai.BackendGeminiAPI,
				HTTPOptions: genai.HTTPOptions{
					BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
					APIVersion: "v1beta",
				},
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}

			recordingsDir := t.TempDir()
			gameServer := NewServer(client)
			gameServer.RecordingsDir = recordingsDir

			handlerDone := make(chan struct{})
			appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				gameServer.liveGame(w, r)
			}))
			defer appServer.Close()

			browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
				"/live/en?target=Lemon&forbidden=Lemon&forbidden=Yellow"
			browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
			if err != nil {
				t.Fatalf("dial liveGame: %v", err)
			}
			defer browserConn.Close()

			select {
			case <-bothUp:
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for both sessions")
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
			defer guesserSess.conn.Close()
			defer judgeSess.conn.Close()

			// Drain initial setupComplete.
			_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
			if _, _, err := browserConn.ReadMessage(); err != nil {
				t.Fatalf("read setupComplete: %v", err)
			}

			for _, reply := range tc.guesserReplies {
				if err := guesserSess.conn.WriteMessage(websocket.TextMessage, []byte(reply)); err != nil {
					t.Fatalf("guesser write reply: %v", err)
				}
				_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
				if _, _, err := browserConn.ReadMessage(); err != nil {
					t.Fatalf("browser read guesser reply: %v", err)
				}
			}

			if tc.sendOutcomeMsg != "" {
				if err := browserConn.WriteMessage(websocket.TextMessage, []byte(tc.sendOutcomeMsg)); err != nil {
					t.Fatalf("browser write outcome: %v", err)
				}
			} else {
				_ = browserConn.Close()
			}

			select {
			case <-handlerDone:
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for liveGame handler to finish")
			}

			// Client control messages must never be forwarded to Gemini sessions.
			select {
			case leaked := <-guesserSess.incoming:
				t.Errorf("guesser unexpectedly received control message: %s", leaked)
			default:
			}
			select {
			case leaked := <-judgeSess.incoming:
				t.Errorf("judge unexpectedly received control message: %s", leaked)
			default:
			}

			entries, err := os.ReadDir(recordingsDir)
			if err != nil || len(entries) != 1 {
				t.Fatalf("expected 1 recording dir, got %d (err=%v)", len(entries), err)
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
			if rec.Outcome != tc.wantOutcome {
				t.Errorf("recorded outcome = %q, want %q", rec.Outcome, tc.wantOutcome)
			}

			eventsBytes, err := os.ReadFile(filepath.Join(gameDir, "events.jsonl"))
			if err != nil {
				t.Fatalf("read events.jsonl: %v", err)
			}
			if !strings.Contains(string(eventsBytes), `"type":"outcome"`) || !strings.Contains(string(eventsBytes), `"outcome":"`+tc.wantOutcome+`"`) {
				t.Errorf("events.jsonl missing outcome event %q:\n%s", tc.wantOutcome, eventsBytes)
			}
		})
	}
}

func TestNewGenAIClientForcesGeminiAPIV1Beta(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "test-key")
	// Even if Vertex env vars are set, Verboten must use BackendGeminiAPI + v1beta
	// because gemini-3.1-flash-live-preview and gemini-decision-1 require Gemini Developer API.
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
	t.Setenv("GOOGLE_GENAI_API_VERSION", "")

	client, err := NewGenAIClient(context.Background())
	if err != nil {
		t.Fatalf("NewGenAIClient: %v", err)
	}
	if got := client.ClientConfig().Backend; got != genai.BackendGeminiAPI {
		t.Fatalf("Backend = %v, want %v", got, genai.BackendGeminiAPI)
	}
	if got := client.ClientConfig().HTTPOptions.APIVersion; got != "v1beta" {
		t.Fatalf("APIVersion = %q, want %q", got, "v1beta")
	}
}

func TestDefaultRecordingsDirForEnv(t *testing.T) {
	unsetEnv := func(t *testing.T, key string) {
		t.Helper()
		t.Setenv(key, "")
		_ = os.Unsetenv(key)
	}

	cases := []struct {
		name        string
		recordGames *string // nil means unset
		customDir   string
		want        string
	}{
		{name: "absent", recordGames: nil, want: ""},
		{name: "empty", recordGames: genai.Ptr(""), want: ""},
		{name: "zero", recordGames: genai.Ptr("0"), want: ""},
		{name: "false", recordGames: genai.Ptr("false"), want: ""},
		{name: "other_value", recordGames: genai.Ptr("yes"), want: ""},
		{name: "mixed_case_True", recordGames: genai.Ptr("True"), want: ""},
		{name: "absent_with_custom_dir", recordGames: nil, customDir: "/mnt/recordings", want: ""},
		{name: "one", recordGames: genai.Ptr("1"), want: DefaultRecordingsDir},
		{name: "TRUE", recordGames: genai.Ptr("TRUE"), want: DefaultRecordingsDir},
		{name: "true", recordGames: genai.Ptr("true"), want: DefaultRecordingsDir},
		{name: "enabled_with_custom_dir", recordGames: genai.Ptr("1"), customDir: "/mnt/recordings", want: "/mnt/recordings"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.recordGames == nil {
				unsetEnv(t, "RECORD_GAMES")
			} else {
				t.Setenv("RECORD_GAMES", *tc.recordGames)
			}
			if tc.customDir == "" {
				unsetEnv(t, "VERBOTEN_RECORDINGS_DIR")
			} else {
				t.Setenv("VERBOTEN_RECORDINGS_DIR", tc.customDir)
			}
			if got := DefaultRecordingsDirForEnv(); got != tc.want {
				t.Errorf("DefaultRecordingsDirForEnv() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestEmbeddedAssetsServing(t *testing.T) {
	// Change into an empty temporary directory so any dynamic filesystem read of "assets/" would fail.
	t.Chdir(t.TempDir())

	srv := NewServer(nil)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	cases := []struct {
		path            string
		wantStatus      int
		wantContentType string
		wantContains    string
	}{
		{path: "/", wantStatus: http.StatusOK, wantContains: "<title>"},
		{path: "/words.json", wantStatus: http.StatusOK, wantContentType: "application/json", wantContains: `"pizza"`},
		{path: "/forbiddenwords/referee.png", wantStatus: http.StatusOK, wantContentType: "image/png"},
		{path: "/forbiddenwords/words_img/bridge.png", wantStatus: http.StatusOK, wantContentType: "image/png"},
		{path: "/forbiddenwords/.DS_Store", wantStatus: http.StatusNotFound},
	}

	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			resp, err := http.Get(ts.URL + tc.path)
			if err != nil {
				t.Fatalf("GET %s: %v", tc.path, err)
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("read body %s: %v", tc.path, err)
			}
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("GET %s status = %d, want %d", tc.path, resp.StatusCode, tc.wantStatus)
			}
			if tc.wantContentType != "" && !strings.Contains(resp.Header.Get("Content-Type"), tc.wantContentType) {
				t.Errorf("GET %s Content-Type = %q, want %q", tc.path, resp.Header.Get("Content-Type"), tc.wantContentType)
			}
			if tc.wantContains != "" && !strings.Contains(string(body), tc.wantContains) {
				t.Errorf("GET %s body missing %q", tc.path, tc.wantContains)
			}
			if tc.wantStatus == http.StatusOK && len(body) == 0 {
				t.Errorf("GET %s returned empty body", tc.path)
			}
		})
	}
}

func TestStructuredSlogOutput(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf)
	logger.Info("starting game", "game_id", "ab12", "lang", "fr", "target", "Guitare")
	logger.Warn("guesser model disconnected", "game_id", "ab12")
	logger.Error("decision 1 check failed", "game_id", "ab12", "spoken", "porte", "forbidden", "Porter")

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d JSON log lines, want 3:\n%s", len(lines), buf.String())
	}

	wantSeverities := []string{"INFO", "WARNING", "ERROR"}
	wantMessages := []string{"starting game", "guesser model disconnected", "decision 1 check failed"}
	for i, line := range lines {
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("line %d is not valid JSON: %v (%s)", i, err, line)
		}
		if got := entry["severity"]; got != wantSeverities[i] {
			t.Errorf("line %d severity = %v, want %q", i, got, wantSeverities[i])
		}
		if got := entry["message"]; got != wantMessages[i] {
			t.Errorf("line %d message = %v, want %q", i, got, wantMessages[i])
		}
		if got := entry["game_id"]; got != "ab12" {
			t.Errorf("line %d game_id = %v, want %q", i, got, "ab12")
		}
		if _, ok := entry["time"]; !ok {
			t.Errorf("line %d missing time field: %s", i, line)
		}
	}
}

func makeConstPCM16(samples int, amp int16) []byte {
	b := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		binary.LittleEndian.PutUint16(b[i*2:i*2+2], uint16(amp))
	}
	return b
}

func TestJudgeVAD(t *testing.T) {
	const mime = "audio/pcm;rate=16000"
	silentChunk := makeConstPCM16(1024, 5) // 64 ms silence (RMS=5)
	speechChunk := makeConstPCM16(1024, 800)

	t.Run("silence_pause_triggers_activity_end", func(t *testing.T) {
		var sent []genai.LiveRealtimeInput
		v := newJudgeVAD(func(in genai.LiveRealtimeInput) error {
			sent = append(sent, in)
			return nil
		})

		// Leading silence should not open an activity.
		for i := 0; i < 5; i++ {
			_ = v.FeedAudio(silentChunk, mime)
		}
		if len(sent) != 0 {
			t.Fatalf("expected 0 messages during leading silence, got %d", len(sent))
		}

		// 640 ms of speech -> ActivityStart + 3 pre-roll chunks (including onset) + 9 speech chunks.
		for i := 0; i < 10; i++ {
			_ = v.FeedAudio(speechChunk, mime)
		}
		if len(sent) == 0 || sent[0].ActivityStart == nil {
			t.Fatalf("expected first signal to be ActivityStart, got %+v", sent)
		}

		// Brief 256 ms mid-clause pause (4 chunks) must NOT close the turn prematurely.
		for i := 0; i < 4; i++ {
			_ = v.FeedAudio(silentChunk, mime)
		}
		if sent[len(sent)-1].ActivityEnd != nil {
			t.Fatalf("unexpected ActivityEnd after brief 256ms pause")
		}

		// Additional 320 ms of silence (total 576 ms >= 550 ms) -> ActivityEnd.
		for i := 0; i < 5; i++ {
			_ = v.FeedAudio(silentChunk, mime)
		}
		last := sent[len(sent)-1]
		if last.ActivityEnd == nil {
			t.Fatalf("expected ActivityEnd after 576ms silence, last=%+v", last)
		}
	})

	t.Run("single_chunk_dip_does_not_soft_split_mid_word", func(t *testing.T) {
		var sent []genai.LiveRealtimeInput
		v := newJudgeVAD(func(in genai.LiveRealtimeInput) error {
			sent = append(sent, in)
			return nil
		})

		// 40 speech chunks = 2560 ms (>= judgeSoftMaxSpeechMs).
		for i := 0; i < 40; i++ {
			_ = v.FeedAudio(speechChunk, mime)
		}
		// Single 64 ms quiet chunk (stop-consonant closure) must NOT trigger soft split.
		_ = v.FeedAudio(silentChunk, mime)
		if sent[len(sent)-1].ActivityEnd != nil {
			t.Fatalf("unexpected soft split on a single 64ms quiet chunk")
		}
		// Second consecutive quiet chunk (128 ms >= judgeSoftDipSilenceMs) MUST trigger soft split.
		_ = v.FeedAudio(silentChunk, mime)
		if sent[len(sent)-1].ActivityEnd == nil {
			t.Fatalf("expected soft split after 128ms quiet dip past soft max")
		}
	})

	t.Run("continuous_speech_splits_at_hard_max_and_replays_overlap", func(t *testing.T) {
		var sent []genai.LiveRealtimeInput
		v := newJudgeVAD(func(in genai.LiveRealtimeInput) error {
			sent = append(sent, in)
			return nil
		})

		// 66 nonstop loud chunks (~4.22 s) without any pause.
		for i := 0; i < 66; i++ {
			_ = v.FeedAudio(speechChunk, mime)
		}
		endCount := 0
		for _, m := range sent {
			if m.ActivityEnd != nil {
				endCount++
			}
		}
		if endCount != 1 {
			t.Fatalf("expected 1 ActivityEnd during 4.22s continuous speech, got %d", endCount)
		}

		// While waiting for TurnComplete, chunks are buffered; OnTurnComplete flushes them
		// with overlap into a fresh ActivityStart turn.
		before := len(sent)
		if err := v.OnTurnComplete(); err != nil {
			t.Fatalf("OnTurnComplete: %v", err)
		}
		if len(sent) <= before || sent[before].ActivityStart == nil {
			t.Fatalf("expected ActivityStart on TurnComplete flush, got %+v", sent[before:])
		}
	})

	t.Run("stale_overlap_discarded_after_prolonged_silence", func(t *testing.T) {
		var sent []genai.LiveRealtimeInput
		v := newJudgeVAD(func(in genai.LiveRealtimeInput) error {
			sent = append(sent, in)
			return nil
		})

		// Trigger a soft split (retains overlap).
		for i := 0; i < 40; i++ {
			_ = v.FeedAudio(speechChunk, mime)
		}
		_ = v.FeedAudio(silentChunk, mime)
		_ = v.FeedAudio(silentChunk, mime)
		if err := v.OnTurnComplete(); err != nil {
			t.Fatalf("OnTurnComplete: %v", err)
		}

		// Prolonged silence (>= 550 ms) after the split should discard stale overlap.
		for i := 0; i < 10; i++ {
			_ = v.FeedAudio(silentChunk, mime)
		}
		before := len(sent)
		_ = v.FeedAudio(speechChunk, mime)
		// Should emit ActivityStart + preRoll (3 chunks), NOT overlap (6) + onset (1).
		gotTurnPrefix := len(sent) - before
		if gotTurnPrefix != 1+judgePreRollChunks {
			t.Fatalf("expected ActivityStart + %d preRoll chunks after prolonged silence, got %d messages", judgePreRollChunks, gotTurnPrefix)
		}
	})

	t.Run("turn_timeout_unblocks_waiting_state", func(t *testing.T) {
		now := time.Unix(1000, 0)
		var sent []genai.LiveRealtimeInput
		v := newJudgeVAD(func(in genai.LiveRealtimeInput) error {
			sent = append(sent, in)
			return nil
		})
		v.now = func() time.Time { return now }

		for i := 0; i < 10; i++ {
			_ = v.FeedAudio(speechChunk, mime)
		}
		for i := 0; i < 9; i++ {
			_ = v.FeedAudio(silentChunk, mime)
		}
		// Buffer speech while waiting for TurnComplete.
		_ = v.FeedAudio(speechChunk, mime)
		countWhileWaiting := len(sent)

		// Advance past timeout without receiving TurnComplete -> next FeedAudio must unblock and start a new turn.
		now = now.Add(judgeTurnResponseTimeout + time.Millisecond)
		_ = v.FeedAudio(speechChunk, mime)
		if len(sent) <= countWhileWaiting || sent[countWhileWaiting].ActivityStart == nil {
			t.Fatalf("expected timeout to unblock and emit ActivityStart, got %+v", sent[countWhileWaiting:])
		}
	})
}

func TestFormatJudgeLossText(t *testing.T) {
	cases := []struct {
		lang, spoken, forbidden, reason string
		want                            string
	}{
		{
			lang: "fr", spoken: "porte", forbidden: "porter", reason: "resembling",
			want: "Vous avez dit le mot « porte », qui est trop proche du mot prohibé « porter ».",
		},
		{
			lang: "fr", spoken: "l'instrument", forbidden: "Instrument", reason: "exact",
			want: "Vous avez dit le mot prohibé « Instrument ».",
		},
		{
			lang: "fr", spoken: "lemon", forbidden: "Citron", reason: "translation",
			want: "Vous avez dit le mot « lemon », qui est une traduction du mot à deviner « Citron ».",
		},
		{
			lang: "en", spoken: "playing", forbidden: "Play", reason: "resembling",
			want: `You said the word "playing", which is too close to the forbidden word "Play".`,
		},
		{
			lang: "en", spoken: "home", forbidden: "Home", reason: "exact",
			want: `You said the forbidden word "Home".`,
		},
		{
			lang: "en", spoken: "ordinateur", forbidden: "Computer", reason: "translation",
			want: `You said the word "ordinateur", which is a translation of the target word "Computer".`,
		},
	}
	for _, tc := range cases {
		if got := formatJudgeLossText(tc.lang, tc.spoken, tc.forbidden, tc.reason); got != tc.want {
			t.Errorf("formatJudgeLossText(%q,%q,%q,%q) = %q, want %q",
				tc.lang, tc.spoken, tc.forbidden, tc.reason, got, tc.want)
		}
	}
}

func TestDecisionSamePronunciationWireFormat(t *testing.T) {
	var (
		gotPath string
		gotBody map[string]any
	)
	fakeDecisionServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode decision request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "dec-pron-1",
			"model": "gemini-decision-1",
			"status": "completed",
			"answers": {
				"same_pronunciation": {"type": "noul", "noul": 0.96, "confidence": 0.95}
			}
		}`))
	}))
	defer fakeDecisionServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-api-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    fakeDecisionServer.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	srv := NewServer(client)
	prob, err := srv.samePronunciationProbability(context.Background(), "fr", "poissons", "Poisson")
	if err != nil {
		t.Fatalf("samePronunciationProbability: %v", err)
	}
	if prob < 0.95 || prob > 0.97 {
		t.Fatalf("prob = %v, want ~0.96", prob)
	}
	if !strings.HasSuffix(gotPath, "/decisions") {
		t.Errorf("request path = %q, want suffix /decisions", gotPath)
	}
	if gotBody["model"] != decisionModelName {
		t.Errorf("request model = %v, want %q", gotBody["model"], decisionModelName)
	}
	questions, _ := gotBody["questions"].(map[string]any)
	if _, ok := questions["same_pronunciation"]; !ok {
		t.Errorf("expected same_pronunciation question in %+v", gotBody)
	}
}

func TestJudgeLossConcurrentDecisionAndTTS(t *testing.T) {
	fakeResemblingWAV := []byte("RIFF\x24\x00\x00\x00WAVEfmt mock-resembling")
	fakeExactWAV := []byte("RIFF\x24\x00\x00\x00WAVEfmt mock-exact-wav")

	t.Run("resembling_confirmed_runs_A_and_B_concurrently", func(t *testing.T) {
		type capturedSession struct {
			isJudge bool
			conn    *websocket.Conn
		}
		var (
			mu       sync.Mutex
			sessions []*capturedSession
			bothUp   = make(chan struct{})
		)

		fakeGemini := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			c, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			_, setupBytes, err := c.ReadMessage()
			if err != nil {
				c.Close()
				return
			}
			sess := &capturedSession{
				isJudge: strings.Contains(string(setupBytes), declareLossFuncName),
				conn:    c,
			}
			_ = c.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`))

			mu.Lock()
			sessions = append(sessions, sess)
			if len(sessions) == 2 {
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

		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  "test-api-key",
			Backend: genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{
				BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
				APIVersion: "v1beta",
			},
		})
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}

		// Synchronization barrier proving A1 (same_root), A2 (same_pronunciation),
		// B1 (resembling TTS), and B2 (exact TTS) ALL run concurrently!
		callA1Started := make(chan struct{})
		callA2Started := make(chan struct{})
		callB1Started := make(chan struct{})
		callB2Started := make(chan struct{})
		exactTTSCanceled := make(chan struct{})

		wantResemblingMsg := "Vous avez dit le mot « porte », qui est trop proche du mot prohibé « porter »."
		wantExactMsg := "Vous avez dit le mot prohibé « porter »."

		gameServer := NewServer(client)
		gameServer.RecordingsDir = t.TempDir()
		gameServer.sameRootProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			close(callA1Started)
			for _, ch := range []chan struct{}{callA2Started, callB1Started, callB2Started} {
				select {
				case <-ch:
				case <-time.After(2 * time.Second):
					t.Errorf("concurrent calls did not all start with Call A1")
				}
			}
			return 0.94, nil
		}
		gameServer.samePronunciationProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			close(callA2Started)
			select {
			case <-callA1Started:
			case <-time.After(2 * time.Second):
				t.Errorf("Call A1 did not start concurrently with Call A2")
			}
			return 0.03, nil // "porte" and "porter" are pronounced differently
		}
		gameServer.judgeTTSFunc = func(ctx context.Context, lang, text string) ([]byte, string, error) {
			switch text {
			case wantResemblingMsg:
				close(callB1Started)
				select {
				case <-callA1Started:
				case <-time.After(2 * time.Second):
					t.Errorf("Call A1 did not start concurrently with Call B1")
				}
				time.Sleep(30 * time.Millisecond)
				return fakeResemblingWAV, "audio/wav", nil
			case wantExactMsg:
				close(callB2Started)
				<-ctx.Done()
				close(exactTTSCanceled)
				return nil, "", ctx.Err()
			default:
				t.Errorf("unexpected TTS text: %q", text)
				return nil, "", context.Canceled
			}
		}

		handlerDone := make(chan struct{})
		appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer close(handlerDone)
			gameServer.liveGame(w, r)
		}))
		defer appServer.Close()

		browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
			"/live/fr?target=Dragon&forbidden=Dragon&forbidden=Porter"
		browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
		if err != nil {
			t.Fatalf("dial liveGame: %v", err)
		}
		defer func() {
			_ = browserConn.Close()
			select {
			case <-handlerDone:
			case <-time.After(3 * time.Second):
			}
		}()

		select {
		case <-bothUp:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for both sessions")
		}

		mu.Lock()
		var judgeSess, guesserSess *capturedSession
		for _, s := range sessions {
			if s.isJudge {
				judgeSess = s
			} else {
				guesserSess = s
			}
		}
		mu.Unlock()
		defer judgeSess.conn.Close()
		defer guesserSess.conn.Close()

		_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
		if _, _, err := browserConn.ReadMessage(); err != nil {
			t.Fatalf("read setupComplete: %v", err)
		}

		toolCall := `{"toolCall":{"functionCalls":[{"id":"c1","name":"declare_loss","args":{"forbidden_word":"porter","spoken_word":"porte","reason":"resembling"}}]}}`
		if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			t.Fatalf("write toolCall: %v", err)
		}

		_ = browserConn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, gotBytes, err := browserConn.ReadMessage()
		if err != nil {
			t.Fatalf("read judge verdict: %v", err)
		}

		select {
		case <-exactTTSCanceled:
		case <-time.After(2 * time.Second):
			t.Fatal("expected unused exact TTS call B2 to be cancelled")
		}

		var jm judgeMessage
		if err := json.Unmarshal(gotBytes, &jm); err != nil {
			t.Fatalf("unmarshal verdict: %v", err)
		}
		if jm.Judge == nil || !jm.Judge.Lost {
			t.Fatalf("expected loss verdict, got %+v", jm.Judge)
		}
		if jm.Judge.Reason != "resembling" {
			t.Errorf("Judge.Reason = %q, want resembling", jm.Judge.Reason)
		}
		if jm.Judge.Message != wantResemblingMsg {
			t.Errorf("Judge.Message = %q, want %q", jm.Judge.Message, wantResemblingMsg)
		}
		if jm.Judge.AudioMimeType != "audio/wav" {
			t.Errorf("Judge.AudioMimeType = %q, want audio/wav", jm.Judge.AudioMimeType)
		}
		decoded, err := base64.StdEncoding.DecodeString(jm.Judge.Audio)
		if err != nil {
			t.Fatalf("decode base64 Judge.Audio: %v", err)
		}
		if !bytes.Equal(decoded, fakeResemblingWAV) {
			t.Errorf("decoded Judge.Audio = %q, want %q", decoded, fakeResemblingWAV)
		}
	})

	t.Run("same_pronunciation_switches_to_exact_explanation_concurrently", func(t *testing.T) {
		type capturedSession struct {
			isJudge bool
			conn    *websocket.Conn
		}
		var (
			mu       sync.Mutex
			sessions []*capturedSession
			bothUp   = make(chan struct{})
		)

		fakeGemini := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			c, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			_, setupBytes, err := c.ReadMessage()
			if err != nil {
				c.Close()
				return
			}
			sess := &capturedSession{
				isJudge: strings.Contains(string(setupBytes), declareLossFuncName),
				conn:    c,
			}
			_ = c.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`))

			mu.Lock()
			sessions = append(sessions, sess)
			if len(sessions) == 2 {
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

		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  "test-api-key",
			Backend: genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{
				BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
				APIVersion: "v1beta",
			},
		})
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}

		wantResemblingMsg := "Vous avez dit le mot « poissons », qui est trop proche du mot prohibé « Poisson »."
		wantExactMsg := "Vous avez dit le mot prohibé « Poisson »."

		callA1Started := make(chan struct{})
		callA2Started := make(chan struct{})
		callB1Started := make(chan struct{})
		callB2Started := make(chan struct{})
		resemblingTTSCanceled := make(chan struct{})

		gameServer := NewServer(client)
		gameServer.RecordingsDir = t.TempDir()
		gameServer.sameRootProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			close(callA1Started)
			for _, ch := range []chan struct{}{callA2Started, callB1Started, callB2Started} {
				select {
				case <-ch:
				case <-time.After(2 * time.Second):
					t.Errorf("concurrent calls did not all start with Call A1")
				}
			}
			return 0.98, nil
		}
		gameServer.samePronunciationProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			close(callA2Started)
			return 0.96, nil // "poissons" and "Poisson" are pronounced identically in French
		}
		gameServer.judgeTTSFunc = func(ctx context.Context, lang, text string) ([]byte, string, error) {
			switch text {
			case wantResemblingMsg:
				close(callB1Started)
				<-ctx.Done()
				close(resemblingTTSCanceled)
				return nil, "", ctx.Err()
			case wantExactMsg:
				close(callB2Started)
				time.Sleep(25 * time.Millisecond)
				return fakeExactWAV, "audio/wav", nil
			default:
				t.Errorf("unexpected TTS text: %q", text)
				return nil, "", context.Canceled
			}
		}

		handlerDone := make(chan struct{})
		appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer close(handlerDone)
			gameServer.liveGame(w, r)
		}))
		defer appServer.Close()

		browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
			"/live/fr?target=Oc%C3%A9an&forbidden=Oc%C3%A9an&forbidden=Poisson"
		browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
		if err != nil {
			t.Fatalf("dial liveGame: %v", err)
		}
		defer func() {
			_ = browserConn.Close()
			select {
			case <-handlerDone:
			case <-time.After(3 * time.Second):
			}
		}()

		select {
		case <-bothUp:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for both sessions")
		}

		mu.Lock()
		var judgeSess, guesserSess *capturedSession
		for _, s := range sessions {
			if s.isJudge {
				judgeSess = s
			} else {
				guesserSess = s
			}
		}
		mu.Unlock()
		defer judgeSess.conn.Close()
		defer guesserSess.conn.Close()

		_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
		if _, _, err := browserConn.ReadMessage(); err != nil {
			t.Fatalf("read setupComplete: %v", err)
		}

		toolCall := `{"toolCall":{"functionCalls":[{"id":"c-homophone","name":"declare_loss","args":{"forbidden_word":"Poisson","spoken_word":"poissons","reason":"resembling"}}]}}`
		if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			t.Fatalf("write toolCall: %v", err)
		}

		_ = browserConn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, gotBytes, err := browserConn.ReadMessage()
		if err != nil {
			t.Fatalf("read judge verdict: %v", err)
		}

		select {
		case <-resemblingTTSCanceled:
		case <-time.After(2 * time.Second):
			t.Fatal("expected unused resembling TTS call B1 to be cancelled")
		}

		var jm judgeMessage
		if err := json.Unmarshal(gotBytes, &jm); err != nil {
			t.Fatalf("unmarshal verdict: %v", err)
		}
		if jm.Judge == nil || !jm.Judge.Lost {
			t.Fatalf("expected loss verdict, got %+v", jm.Judge)
		}
		if jm.Judge.Reason != "exact" {
			t.Errorf("Judge.Reason = %q, want exact", jm.Judge.Reason)
		}
		if jm.Judge.Message != wantExactMsg {
			t.Errorf("Judge.Message = %q, want %q", jm.Judge.Message, wantExactMsg)
		}
		decoded, err := base64.StdEncoding.DecodeString(jm.Judge.Audio)
		if err != nil {
			t.Fatalf("decode base64 Judge.Audio: %v", err)
		}
		if !bytes.Equal(decoded, fakeExactWAV) {
			t.Errorf("decoded Judge.Audio = %q, want %q", decoded, fakeExactWAV)
		}
	})

	t.Run("resembling_rejected_cancels_TTS_and_keeps_listening", func(t *testing.T) {
		type capturedSession struct {
			isJudge       bool
			conn          *websocket.Conn
			toolResponses chan string
		}
		var (
			mu       sync.Mutex
			sessions []*capturedSession
			bothUp   = make(chan struct{})
		)

		fakeGemini := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			c, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			_, setupBytes, err := c.ReadMessage()
			if err != nil {
				c.Close()
				return
			}
			sess := &capturedSession{
				isJudge:       strings.Contains(string(setupBytes), declareLossFuncName),
				conn:          c,
				toolResponses: make(chan string, 4),
			}
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
				if strings.Contains(string(msg), "toolResponse") {
					sess.toolResponses <- string(msg)
				}
			}
		}))
		defer fakeGemini.Close()

		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  "test-api-key",
			Backend: genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{
				BaseURL:    "ws://" + strings.TrimPrefix(fakeGemini.URL, "http://"),
				APIVersion: "v1beta",
			},
		})
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}

		ttsStarted := make(chan struct{}, 2)
		ttsCanceled := make(chan struct{}, 2)
		pronCanceled := make(chan struct{})

		gameServer := NewServer(client)
		gameServer.RecordingsDir = t.TempDir()
		gameServer.sameRootProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			<-ttsStarted
			<-ttsStarted
			return 0.07, nil // Rejected (< 75%)
		}
		gameServer.samePronunciationProbFunc = func(ctx context.Context, lang, spoken, forbidden string) (float64, error) {
			<-ctx.Done()
			close(pronCanceled)
			return 0, ctx.Err()
		}
		gameServer.judgeTTSFunc = func(ctx context.Context, lang, text string) ([]byte, string, error) {
			ttsStarted <- struct{}{}
			<-ctx.Done()
			ttsCanceled <- struct{}{}
			return nil, "", ctx.Err()
		}

		handlerDone := make(chan struct{})
		appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer close(handlerDone)
			gameServer.liveGame(w, r)
		}))
		defer appServer.Close()

		browserURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
			"/live/fr?target=Dragon&forbidden=Dragon&forbidden=Feu"
		browserConn, _, err := websocket.DefaultDialer.Dial(browserURL, nil)
		if err != nil {
			t.Fatalf("dial liveGame: %v", err)
		}
		defer func() {
			_ = browserConn.Close()
			select {
			case <-handlerDone:
			case <-time.After(3 * time.Second):
			}
		}()

		select {
		case <-bothUp:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for both sessions")
		}

		mu.Lock()
		var judgeSess, guesserSess *capturedSession
		for _, s := range sessions {
			if s.isJudge {
				judgeSess = s
			} else {
				guesserSess = s
			}
		}
		mu.Unlock()
		defer judgeSess.conn.Close()
		defer guesserSess.conn.Close()

		_ = browserConn.SetReadDeadline(time.Now().Add(2 * time.Second))
		if _, _, err := browserConn.ReadMessage(); err != nil {
			t.Fatalf("read setupComplete: %v", err)
		}

		toolCall := `{"toolCall":{"functionCalls":[{"id":"c-ignore","name":"declare_loss","args":{"forbidden_word":"Feu","spoken_word":"flamme","reason":"resembling"}}]}}`
		if err := judgeSess.conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			t.Fatalf("write toolCall: %v", err)
		}

		for i := 0; i < 2; i++ {
			select {
			case <-ttsCanceled:
			case <-time.After(2 * time.Second):
				t.Fatalf("expected both TTS calls to be cancelled after Decision 1 rejected (%d/2 cancelled)", i)
			}
		}
		select {
		case <-pronCanceled:
		case <-time.After(2 * time.Second):
			t.Fatal("expected same_pronunciation Decision 1 call A2 to be cancelled after A1 rejected")
		}

		select {
		case resp := <-judgeSess.toolResponses:
			if !strings.Contains(resp, "Ignored") {
				t.Errorf("expected Ignored toolResponse, got %s", resp)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("expected toolResponse to keep Judge listening")
		}

		// Confirm browser did NOT receive any loss verdict.
		_ = browserConn.SetReadDeadline(time.Now().Add(150 * time.Millisecond))
		if _, msg, err := browserConn.ReadMessage(); err == nil {
			t.Fatalf("expected no browser message after ignored call, got: %s", string(msg))
		}
	})
}

func TestServeTTSEndpointAndCache(t *testing.T) {
	var (
		mu       sync.Mutex
		apiCalls int
		gotModel string
		gotVoice string
		gotLang  string
		gotText  string
	)
	fakeAudio := []byte("RIFF\x24\x00\x00\x00WAVEfmt cached-tts-audio")

	fakeAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		apiCalls++
		gotModel = r.URL.Path
		mu.Unlock()

		var reqBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
			if contents, ok := reqBody["contents"].([]any); ok && len(contents) > 0 {
				if first, ok := contents[0].(map[string]any); ok {
					if parts, ok := first["parts"].([]any); ok && len(parts) > 0 {
						if part0, ok := parts[0].(map[string]any); ok {
							mu.Lock()
							gotText, _ = part0["text"].(string)
							mu.Unlock()
						}
					}
				}
			}
			if gc, ok := reqBody["generationConfig"].(map[string]any); ok {
				if sc, ok := gc["speechConfig"].(map[string]any); ok {
					mu.Lock()
					gotLang, _ = sc["languageCode"].(string)
					if vc, ok := sc["voiceConfig"].(map[string]any); ok {
						if pvc, ok := vc["prebuiltVoiceConfig"].(map[string]any); ok {
							gotVoice, _ = pvc["voiceName"].(string)
						}
					}
					mu.Unlock()
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []any{
				map[string]any{
					"content": map[string]any{
						"parts": []any{
							map[string]any{
								"inlineData": map[string]any{
									"mimeType": "audio/wav",
									"data":     base64.StdEncoding.EncodeToString(fakeAudio),
								},
							},
						},
					},
				},
			},
		})
	}))
	defer fakeAPI.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-api-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    fakeAPI.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	srv := NewServer(client)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. Valid POST /tts calls Gemini TTS with masculine voice Charon and fr-FR.
	resp, err := http.Post(ts.URL+"/tts", "application/json", strings.NewReader(`{"lang":"fr","text":"Bravo !"}`))
	if err != nil {
		t.Fatalf("POST /tts: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", resp.StatusCode, string(body))
	}
	if ct := resp.Header.Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("Content-Type = %q, want audio/wav", ct)
	}
	if !bytes.Equal(body, fakeAudio) {
		t.Errorf("body = %q, want %q", body, fakeAudio)
	}

	mu.Lock()
	if !strings.Contains(gotModel, DefaultTTSModel) {
		t.Errorf("model path = %q, want containing %q", gotModel, DefaultTTSModel)
	}
	if gotVoice != DefaultJudgeVoice {
		t.Errorf("voice = %q, want %q", gotVoice, DefaultJudgeVoice)
	}
	if gotLang != "fr-FR" {
		t.Errorf("languageCode = %q, want fr-FR", gotLang)
	}
	if gotText != "Bravo !" {
		t.Errorf("TTS prompt text = %q, want %q", gotText, "Bravo !")
	}
	mu.Unlock()

	// 2. Second identical request hits in-memory cache (no second Gemini API call).
	resp2, err := http.Post(ts.URL+"/tts", "application/json", strings.NewReader(`{"lang":"fr","text":"Bravo !"}`))
	if err != nil {
		t.Fatalf("second POST /tts: %v", err)
	}
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK || !bytes.Equal(body2, fakeAudio) {
		t.Fatalf("cached response mismatch: status=%d body=%q", resp2.StatusCode, string(body2))
	}
	mu.Lock()
	calls := apiCalls
	mu.Unlock()
	if calls != 1 {
		t.Errorf("Gemini TTS API calls = %d, want 1 (second request should hit cache)", calls)
	}

	// 3. Method / input validation.
	getResp, err := http.Get(ts.URL + "/tts")
	if err != nil {
		t.Fatalf("GET /tts: %v", err)
	}
	getResp.Body.Close()
	if getResp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("GET /tts status = %d, want 405", getResp.StatusCode)
	}

	emptyResp, err := http.Post(ts.URL+"/tts", "application/json", strings.NewReader(`{"lang":"fr","text":"   "}`))
	if err != nil {
		t.Fatalf("empty POST /tts: %v", err)
	}
	emptyResp.Body.Close()
	if emptyResp.StatusCode != http.StatusBadRequest {
		t.Errorf("empty POST /tts status = %d, want 400", emptyResp.StatusCode)
	}
}
