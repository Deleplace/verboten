//go:build replay

package verboten

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

func TestReplayGames(t *testing.T) {
	ctx := context.Background()
	client, err := NewGenAIClient(ctx)
	if err != nil {
		t.Fatalf("create genai client: %v", err)
	}
	liveClient, vertexInfo, err := NewVertexLiveClient(ctx)
	if err != nil {
		t.Fatalf("create vertex ai live client: %v", err)
	}
	if liveClient != nil {
		t.Logf("using Vertex AI Live/TTS client: project=%s location=%s endpoint=%s",
			vertexInfo.Project, vertexInfo.Location, vertexInfo.LiveWebSocketURL)
	}

	for _, outcome := range []string{"won", "lost"} {
		expectWin := outcome == "won"
		baseDir := filepath.Join("testdata", "games", outcome)
		entries, err := os.ReadDir(baseDir)
		if err != nil {
			t.Fatalf("read %s: %v", baseDir, err)
		}

		var dirs []string
		for _, entry := range entries {
			if entry.IsDir() {
				dirs = append(dirs, entry.Name())
			}
		}
		sort.Strings(dirs)

		t.Run(outcome, func(t *testing.T) {
			for _, name := range dirs {
				gameDir := filepath.Join(baseDir, name)
				t.Run(name, func(t *testing.T) {
					replayGame(t, client, liveClient, gameDir, expectWin)
				})
			}
		})
	}
}

func replayGame(t *testing.T, client, liveClient *genai.Client, gameDir string, expectWin bool) {
	rec, err := loadGameRecording(filepath.Join(gameDir, "game.json"))
	if err != nil {
		t.Fatalf("load game.json: %v", err)
	}
	if rec.HumanAudio == nil || rec.HumanAudio.File == "" {
		t.Fatalf("recording %s has no human audio file", gameDir)
	}

	pcm, wavSampleRate, err := readWAVPCM(filepath.Join(gameDir, rec.HumanAudio.File))
	if err != nil {
		t.Fatalf("read human WAV: %v", err)
	}
	sampleRate := rec.HumanAudio.SampleRate
	if sampleRate == 0 {
		sampleRate = wavSampleRate
	}
	mimeType := fmt.Sprintf("audio/pcm;rate=%d", sampleRate)

	packets, err := loadHumanAudioPackets(filepath.Join(gameDir, "events.jsonl"), len(pcm), sampleRate)
	if err != nil {
		t.Fatalf("load human audio packets: %v", err)
	}

	var expectedLoss *JudgeEvent
	if !expectWin {
		for i := range rec.JudgeEvents {
			ev := &rec.JudgeEvents[i]
			if !ev.Ignored && ev.Action == declareLossFuncName {
				expectedLoss = ev
				break
			}
		}
		if expectedLoss == nil {
			t.Fatalf("lost recording %s has no non-ignored %q judge event in game.json", gameDir, declareLossFuncName)
		}
	}

	gameServer := NewServer(client)
	gameServer.LiveClient = liveClient
	gameServer.Verbose = testing.Verbose()
	gameServer.RecordingsDir = t.TempDir()

	handlerDone := make(chan struct{})
	appServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		gameServer.liveGame(w, r)
	}))
	defer appServer.Close()

	params := url.Values{}
	params.Set("target", rec.TargetWord)
	params.Add("forbidden", rec.TargetWord)
	for _, w := range rec.ForbiddenWords {
		params.Add("forbidden", w)
	}
	wsURL := "ws://" + strings.TrimPrefix(appServer.URL, "http://") +
		"/live/" + rec.Language + "?" + params.Encode()

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial %s: %v", wsURL, err)
	}
	defer func() {
		_ = conn.Close()
		select {
		case <-handlerDone:
		case <-time.After(5 * time.Second):
		}
	}()

	type replayResult struct {
		won           bool
		judgeVerdict  *judgeVerdict
		guesserSpeech string
		humanSpeech   string
		readErr       error
	}

	resultCh := make(chan replayResult, 1)
	stopStreaming := make(chan struct{})
	setupReady := make(chan struct{})
	var setupOnce sync.Once

	// Read messages from the backend (Guesser responses + Judge verdicts).
	go func() {
		var (
			guesserSpeech strings.Builder
			humanSpeech   strings.Builder
			winTimer      *time.Timer
			winCh         <-chan time.Time
		)
		defer func() {
			if winTimer != nil {
				winTimer.Stop()
			}
		}()

		type incomingMsg struct {
			data []byte
			err  error
		}
		msgCh := make(chan incomingMsg, 32)
		go func() {
			for {
				_, data, err := conn.ReadMessage()
				msgCh <- incomingMsg{data: data, err: err}
				if err != nil {
					return
				}
			}
		}()

		normalizedTarget := normalizeWord(rec.TargetWord)
		for {
			select {
			case <-winCh:
				resultCh <- replayResult{
					won:           true,
					guesserSpeech: strings.TrimSpace(guesserSpeech.String()),
					humanSpeech:   strings.TrimSpace(humanSpeech.String()),
				}
				return
			case msg := <-msgCh:
				if msg.err != nil {
					resultCh <- replayResult{
						guesserSpeech: strings.TrimSpace(guesserSpeech.String()),
						humanSpeech:   strings.TrimSpace(humanSpeech.String()),
						readErr:       msg.err,
					}
					return
				}

				var jm judgeMessage
				if err := json.Unmarshal(msg.data, &jm); err == nil && jm.Judge != nil {
					if jm.Judge.Pending {
						if winTimer != nil {
							winTimer.Stop()
							winTimer = nil
							winCh = nil
						}
						continue
					}
					if jm.Judge.Lost {
						resultCh <- replayResult{
							judgeVerdict:  jm.Judge,
							guesserSpeech: strings.TrimSpace(guesserSpeech.String()),
							humanSpeech:   strings.TrimSpace(humanSpeech.String()),
						}
						return
					}
				}

				var sm genai.LiveServerMessage
				if err := json.Unmarshal(msg.data, &sm); err != nil {
					continue
				}
				if sm.SetupComplete != nil {
					setupOnce.Do(func() { close(setupReady) })
				}
				if sm.ServerContent == nil {
					continue
				}
				if it := sm.ServerContent.InputTranscription; it != nil && it.Text != "" {
					humanSpeech.WriteString(it.Text)
				}
				if ot := sm.ServerContent.OutputTranscription; ot != nil && ot.Text != "" {
					guesserSpeech.WriteString(ot.Text)
					if winTimer == nil && normalizedTarget != "" &&
						strings.Contains(normalizeWord(guesserSpeech.String()), normalizedTarget) {
						// Mirror browser's 1200ms window so Judge can still declare loss if the human cheated.
						winTimer = time.NewTimer(1200 * time.Millisecond)
						winCh = winTimer.C
					}
				}
			}
		}
	}()

	// Stream the recorded human voice packets in real-time pacing, followed by trailing silence
	// so Gemini Live VAD can finalize its last turn.
	streamDone := make(chan error, 1)
	go func() {
		select {
		case <-setupReady:
		case <-stopStreaming:
			streamDone <- nil
			return
		case <-time.After(15 * time.Second):
			streamDone <- fmt.Errorf("timed out waiting for Gemini Live setupComplete")
			return
		}

		sendPCM := func(chunk []byte) error {
			payload, err := json.Marshal(map[string]any{
				"audio": map[string]any{
					"data":     base64.StdEncoding.EncodeToString(chunk),
					"mimeType": mimeType,
				},
			})
			if err != nil {
				return err
			}
			return conn.WriteMessage(websocket.TextMessage, payload)
		}

		for i, pkt := range packets {
			if i > 0 {
				deltaMs := pkt.OffsetMs - packets[i-1].OffsetMs
				if deltaMs > 0 {
					select {
					case <-stopStreaming:
						streamDone <- nil
						return
					case <-time.After(time.Duration(deltaMs) * time.Millisecond):
					}
				}
			}
			select {
			case <-stopStreaming:
				streamDone <- nil
				return
			default:
			}

			end := pkt.ByteOffset + int64(pkt.ByteLength)
			if pkt.ByteOffset < 0 || end > int64(len(pcm)) {
				continue
			}
			if err := sendPCM(pcm[pkt.ByteOffset:end]); err != nil {
				streamDone <- err
				return
			}
		}

		// Trailing silence so VAD detects end-of-speech on the final utterance.
		const silenceChunkBytes = 2048
		silence := make([]byte, silenceChunkBytes)
		chunkDuration := time.Duration(silenceChunkBytes*1000/(sampleRate*2)) * time.Millisecond
		if chunkDuration <= 0 {
			chunkDuration = 50 * time.Millisecond
		}
		ticker := time.NewTicker(chunkDuration)
		defer ticker.Stop()

		deadline := time.After(6 * time.Second)
		for {
			select {
			case <-stopStreaming:
				streamDone <- nil
				return
			case <-deadline:
				streamDone <- nil
				return
			case <-ticker.C:
				if err := sendPCM(silence); err != nil {
					streamDone <- err
					return
				}
			}
		}
	}()

	var res replayResult
	select {
	case res = <-resultCh:
		close(stopStreaming)
		<-streamDone
	case err := <-streamDone:
		close(stopStreaming)
		if err != nil {
			t.Fatalf("error streaming human audio: %v", err)
		}
		// Give the reader a brief moment in case a final verdict arrived right at deadline.
		select {
		case res = <-resultCh:
		case <-time.After(2 * time.Second):
		}
	}

	if res.judgeVerdict != nil {
		t.Logf("target=%q forbidden=%q | won=%v judge={lost:%v reason:%q spoken:%q forbidden:%q message:%q audioMime:%q audioB64Len:%d} | guesser=%q | human=%q",
			rec.TargetWord, rec.ForbiddenWords, res.won,
			res.judgeVerdict.Lost, res.judgeVerdict.Reason, res.judgeVerdict.SpokenWord, res.judgeVerdict.ForbiddenWord,
			res.judgeVerdict.Message, res.judgeVerdict.AudioMimeType, len(res.judgeVerdict.Audio),
			res.guesserSpeech, res.humanSpeech)
	} else {
		t.Logf("target=%q forbidden=%q | won=%v judge=<nil> | guesser=%q | human=%q",
			rec.TargetWord, rec.ForbiddenWords, res.won, res.guesserSpeech, res.humanSpeech)
	}

	if expectWin {
		if res.judgeVerdict != nil && res.judgeVerdict.Lost {
			t.Fatalf("expected WIN for %q, but Judge declared loss: reason=%q spoken=%q forbidden=%q",
				rec.TargetWord, res.judgeVerdict.Reason, res.judgeVerdict.SpokenWord, res.judgeVerdict.ForbiddenWord)
		}
		if !res.won {
			t.Fatalf("expected WIN for %q, but Guesser did not guess it (guesser=%q, human=%q, err=%v)",
				rec.TargetWord, res.guesserSpeech, res.humanSpeech, res.readErr)
		}
		return
	}

	// Expected LOSS.
	if res.judgeVerdict == nil || !res.judgeVerdict.Lost {
		t.Fatalf("expected LOSS for %q (reason=%q, original spoken=%q forbidden=%q), but Judge did not declare loss (won=%v, guesser=%q, human=%q, err=%v)",
			rec.TargetWord, expectedLoss.Reason, expectedLoss.SpokenWord, expectedLoss.ForbiddenWord,
			res.won, res.guesserSpeech, res.humanSpeech, res.readErr)
	}
	if res.judgeVerdict.Reason != expectedLoss.Reason {
		t.Fatalf("expected LOSS reason=%q (original spoken=%q forbidden=%q), got reason=%q (spoken=%q forbidden=%q)",
			expectedLoss.Reason, expectedLoss.SpokenWord, expectedLoss.ForbiddenWord,
			res.judgeVerdict.Reason, res.judgeVerdict.SpokenWord, res.judgeVerdict.ForbiddenWord)
	}
}

func loadGameRecording(path string) (*GameRecording, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rec GameRecording
	if err := json.Unmarshal(b, &rec); err != nil {
		return nil, err
	}
	return &rec, nil
}

func readWAVPCM(path string) ([]byte, int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(b) < wavHeaderSize || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("invalid WAV file %s", path)
	}
	sampleRate := int(binary.LittleEndian.Uint32(b[24:28]))
	return b[wavHeaderSize:], sampleRate, nil
}

func loadHumanAudioPackets(eventsPath string, totalPCMBytes, sampleRate int) ([]RecordingEvent, error) {
	f, err := os.Open(eventsPath)
	if err == nil {
		defer f.Close()
		var packets []RecordingEvent
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var ev RecordingEvent
			if err := json.Unmarshal([]byte(line), &ev); err != nil {
				return nil, fmt.Errorf("parse %s: %w", eventsPath, err)
			}
			if ev.Type == "human_audio" && ev.ByteLength > 0 {
				packets = append(packets, ev)
			}
		}
		if err := scanner.Err(); err != nil {
			return nil, err
		}
		if len(packets) > 0 {
			return packets, nil
		}
	}

	// Fallback: slice raw PCM into 2048-byte packets at real-time sample rate.
	const chunkSize = 2048
	var packets []RecordingEvent
	byteRate := sampleRate * 2 // 16-bit mono
	if byteRate <= 0 {
		byteRate = defaultAudioSampleRate * 2
	}
	for offset := 0; offset < totalPCMBytes; offset += chunkSize {
		n := min(chunkSize, totalPCMBytes-offset)
		packets = append(packets, RecordingEvent{
			OffsetMs:   int64(offset*1000) / int64(byteRate),
			Type:       "human_audio",
			ByteOffset: int64(offset),
			ByteLength: n,
		})
	}
	return packets, nil
}
