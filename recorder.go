package verboten

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	DefaultRecordingsDir = "recordings"

	defaultAudioSampleRate    = 24000
	defaultAudioBitsPerSample = 16
	defaultAudioChannels      = 1
	wavHeaderSize             = 44
)

// GameRecording is persisted as game.json in each recorded game folder.
type GameRecording struct {
	GameID         string              `json:"game_id"`
	Language       string              `json:"language"`
	LanguageCode   string              `json:"language_code"`
	Model          string              `json:"model"`
	StartedAt      time.Time           `json:"started_at"`
	EndedAt        time.Time           `json:"ended_at"`
	DurationMs     int64               `json:"duration_ms"`
	TargetWord     string              `json:"target_word"`
	ForbiddenWords []string            `json:"forbidden_words"`
	HumanAudio     *AudioTrackMetadata `json:"human_audio"`
	GuesserAudio   *AudioTrackMetadata `json:"guesser_audio"`
	Transcripts    []TranscriptEvent   `json:"transcripts"`
	JudgeEvents    []JudgeEvent        `json:"judge_events"`
}

// AudioTrackMetadata describes a recorded WAV track and its basic timing info.
type AudioTrackMetadata struct {
	File            string `json:"file,omitempty"`
	MIMEType        string `json:"mime_type,omitempty"`
	SampleRate      int    `json:"sample_rate,omitempty"`
	BitsPerSample   int    `json:"bits_per_sample,omitempty"`
	Channels        int    `json:"channels,omitempty"`
	FirstPacketMs   *int64 `json:"first_packet_ms,omitempty"`
	LastPacketMs    *int64 `json:"last_packet_ms,omitempty"`
	PacketCount     int    `json:"packet_count"`
	TotalBytes      int64  `json:"total_bytes"`
	AudioDurationMs int64  `json:"audio_duration_ms"`
}

// TranscriptEvent records a transcribed text fragment and when it arrived.
type TranscriptEvent struct {
	OffsetMs int64  `json:"offset_ms"`
	Speaker  string `json:"speaker"`
	Text     string `json:"text"`
}

// JudgeEvent records an action taken by the Judge model and when it occurred.
type JudgeEvent struct {
	OffsetMs            int64     `json:"offset_ms"`
	Timestamp           time.Time `json:"timestamp"`
	Action              string    `json:"action"`
	Ignored             bool      `json:"ignored,omitempty"`
	SpokenWord          string    `json:"spoken_word,omitempty"`
	ForbiddenWord       string    `json:"forbidden_word,omitempty"`
	Reason              string    `json:"reason,omitempty"`
	DecisionProbability *float64  `json:"decision_probability,omitempty"`
	DecisionAnswer      string    `json:"decision_answer,omitempty"`
	DecisionDurationMs  *int64    `json:"decision_duration_ms,omitempty"`
}

// RecordingEvent is a single chronological line written to events.jsonl.
type RecordingEvent struct {
	OffsetMs            int64    `json:"offset_ms"`
	Type                string   `json:"type"`
	ByteOffset          int64    `json:"byte_offset,omitempty"`
	ByteLength          int      `json:"byte_length,omitempty"`
	MIMEType            string   `json:"mime_type,omitempty"`
	Speaker             string   `json:"speaker,omitempty"`
	Text                string   `json:"text,omitempty"`
	Action              string   `json:"action,omitempty"`
	Ignored             bool     `json:"ignored,omitempty"`
	SpokenWord          string   `json:"spoken_word,omitempty"`
	ForbiddenWord       string   `json:"forbidden_word,omitempty"`
	Reason              string   `json:"reason,omitempty"`
	DecisionProbability *float64 `json:"decision_probability,omitempty"`
	DecisionAnswer      string   `json:"decision_answer,omitempty"`
	DecisionDurationMs  *int64   `json:"decision_duration_ms,omitempty"`
}

type wavTrackRecorder struct {
	dir      string
	filename string
	file     *os.File
	meta     AudioTrackMetadata
}

func (t *wavTrackRecorder) writeChunk(offsetMs int64, pcm []byte, mimeType string) (byteOffset int64, err error) {
	if len(pcm) == 0 {
		return t.meta.TotalBytes, nil
	}
	if t.file == nil {
		path := filepath.Join(t.dir, t.filename)
		f, err := os.Create(path)
		if err != nil {
			return 0, err
		}
		var placeholder [wavHeaderSize]byte
		if _, err := f.Write(placeholder[:]); err != nil {
			f.Close()
			return 0, err
		}
		t.file = f
		t.meta.File = t.filename
		t.meta.MIMEType = mimeType
		t.meta.SampleRate = parseSampleRate(mimeType, defaultAudioSampleRate)
		t.meta.BitsPerSample = defaultAudioBitsPerSample
		t.meta.Channels = defaultAudioChannels
		first := offsetMs
		t.meta.FirstPacketMs = &first
	}

	byteOffset = t.meta.TotalBytes
	if _, err := t.file.Write(pcm); err != nil {
		return byteOffset, err
	}
	last := offsetMs
	t.meta.LastPacketMs = &last
	t.meta.PacketCount++
	t.meta.TotalBytes += int64(len(pcm))
	return byteOffset, nil
}

func (t *wavTrackRecorder) close() error {
	if t.file == nil {
		return nil
	}
	byteRate := int64(t.meta.SampleRate * t.meta.Channels * (t.meta.BitsPerSample / 8))
	if byteRate > 0 {
		t.meta.AudioDurationMs = (t.meta.TotalBytes * 1000) / byteRate
	}
	var hdr [wavHeaderSize]byte
	writeWAVHeader(hdr[:], uint32(t.meta.TotalBytes), t.meta.SampleRate, t.meta.BitsPerSample, t.meta.Channels)
	_, writeErr := t.file.WriteAt(hdr[:], 0)
	closeErr := t.file.Close()
	t.file = nil
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

type gameRecorder struct {
	mu         sync.Mutex
	closed     bool
	dir        string
	startedAt  time.Time
	recording  GameRecording
	human      wavTrackRecorder
	guesser    wavTrackRecorder
	eventsFile *os.File
}

func newGameRecorder(baseDir, gameID, lang, languageCode, model, targetWord string, forbiddenWords []string) (*gameRecorder, error) {
	if baseDir == "" {
		baseDir = DefaultRecordingsDir
	}
	startedAt := time.Now().UTC()
	folderName := fmt.Sprintf("%s_%s_%s_%s",
		startedAt.Format("20060102-150405"),
		sanitizeDirSegment(gameID),
		sanitizeDirSegment(lang),
		sanitizeDirSegment(targetWord),
	)
	dir := filepath.Join(baseDir, folderName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create recording directory %s: %w", dir, err)
	}

	eventsFile, err := os.Create(filepath.Join(dir, "events.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("create events.jsonl: %w", err)
	}

	forbiddenCopy := make([]string, len(forbiddenWords))
	copy(forbiddenCopy, forbiddenWords)

	return &gameRecorder{
		dir:       dir,
		startedAt: startedAt,
		recording: GameRecording{
			GameID:         gameID,
			Language:       lang,
			LanguageCode:   languageCode,
			Model:          model,
			StartedAt:      startedAt,
			TargetWord:     targetWord,
			ForbiddenWords: forbiddenCopy,
			Transcripts:    []TranscriptEvent{},
			JudgeEvents:    []JudgeEvent{},
		},
		human:      wavTrackRecorder{dir: dir, filename: "human.wav"},
		guesser:    wavTrackRecorder{dir: dir, filename: "guesser.wav"},
		eventsFile: eventsFile,
	}, nil
}

func (r *gameRecorder) Dir() string {
	if r == nil {
		return ""
	}
	return r.dir
}

func (r *gameRecorder) offsetMsLocked(now time.Time) int64 {
	return now.Sub(r.startedAt).Milliseconds()
}

func (r *gameRecorder) appendEventLocked(ev RecordingEvent) {
	if r.eventsFile == nil {
		return
	}
	line, err := json.Marshal(ev)
	if err != nil {
		log.Printf("Game %s marshal recording event error: %v", r.recording.GameID, err)
		return
	}
	line = append(line, '\n')
	if _, err := r.eventsFile.Write(line); err != nil {
		log.Printf("Game %s write events.jsonl error: %v", r.recording.GameID, err)
	}
}

func (r *gameRecorder) RecordHumanAudio(pcm []byte, mimeType string) {
	if r == nil || len(pcm) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	offsetMs := r.offsetMsLocked(time.Now().UTC())
	byteOffset, err := r.human.writeChunk(offsetMs, pcm, mimeType)
	if err != nil {
		log.Printf("Game %s record human audio error: %v", r.recording.GameID, err)
		return
	}
	r.appendEventLocked(RecordingEvent{
		OffsetMs:   offsetMs,
		Type:       "human_audio",
		ByteOffset: byteOffset,
		ByteLength: len(pcm),
		MIMEType:   mimeType,
	})
}

func (r *gameRecorder) RecordGuesserAudio(pcm []byte, mimeType string) {
	if r == nil || len(pcm) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	offsetMs := r.offsetMsLocked(time.Now().UTC())
	byteOffset, err := r.guesser.writeChunk(offsetMs, pcm, mimeType)
	if err != nil {
		log.Printf("Game %s record guesser audio error: %v", r.recording.GameID, err)
		return
	}
	r.appendEventLocked(RecordingEvent{
		OffsetMs:   offsetMs,
		Type:       "guesser_audio",
		ByteOffset: byteOffset,
		ByteLength: len(pcm),
		MIMEType:   mimeType,
	})
}

func (r *gameRecorder) RecordTranscript(speaker, text string) {
	if r == nil || strings.TrimSpace(text) == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	offsetMs := r.offsetMsLocked(time.Now().UTC())
	r.recording.Transcripts = append(r.recording.Transcripts, TranscriptEvent{
		OffsetMs: offsetMs,
		Speaker:  speaker,
		Text:     text,
	})
	r.appendEventLocked(RecordingEvent{
		OffsetMs: offsetMs,
		Type:     "transcript",
		Speaker:  speaker,
		Text:     text,
	})
}

func (r *gameRecorder) RecordJudgeEvent(action string, ignored bool, spokenWord, forbiddenWord, reason string, decisionAnswer string, decisionDurationMs *int64) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	now := time.Now().UTC()
	offsetMs := r.offsetMsLocked(now)
	ev := JudgeEvent{
		OffsetMs:           offsetMs,
		Timestamp:          now,
		Action:             action,
		Ignored:            ignored,
		SpokenWord:         spokenWord,
		ForbiddenWord:      forbiddenWord,
		Reason:             reason,
		DecisionAnswer:     decisionAnswer,
		DecisionDurationMs: decisionDurationMs,
	}
	r.recording.JudgeEvents = append(r.recording.JudgeEvents, ev)
	r.appendEventLocked(RecordingEvent{
		OffsetMs:           offsetMs,
		Type:               "judge_event",
		Action:             action,
		Ignored:            ignored,
		SpokenWord:         spokenWord,
		ForbiddenWord:      forbiddenWord,
		Reason:             reason,
		DecisionAnswer:     decisionAnswer,
		DecisionDurationMs: decisionDurationMs,
	})
}

func (r *gameRecorder) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true

	endedAt := time.Now().UTC()
	r.recording.EndedAt = endedAt
	r.recording.DurationMs = endedAt.Sub(r.startedAt).Milliseconds()

	var firstErr error
	recordErr := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	recordErr(r.human.close())
	recordErr(r.guesser.close())
	if r.eventsFile != nil {
		recordErr(r.eventsFile.Close())
		r.eventsFile = nil
	}

	humanMeta := r.human.meta
	guesserMeta := r.guesser.meta
	r.recording.HumanAudio = &humanMeta
	r.recording.GuesserAudio = &guesserMeta

	metaBytes, err := json.MarshalIndent(r.recording, "", "  ")
	if err != nil {
		recordErr(fmt.Errorf("marshal game.json: %w", err))
		return firstErr
	}
	metaBytes = append(metaBytes, '\n')
	if err := os.WriteFile(filepath.Join(r.dir, "game.json"), metaBytes, 0o644); err != nil {
		recordErr(fmt.Errorf("write game.json: %w", err))
	}
	return firstErr
}

func writeWAVHeader(buf []byte, dataSize uint32, sampleRate, bitsPerSample, channels int) {
	blockAlign := uint16(channels * (bitsPerSample / 8))
	byteRate := uint32(sampleRate) * uint32(blockAlign)

	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], 36+dataSize)
	copy(buf[8:12], "WAVE")
	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(buf[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:32], byteRate)
	binary.LittleEndian.PutUint16(buf[32:34], blockAlign)
	binary.LittleEndian.PutUint16(buf[34:36], uint16(bitsPerSample))
	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], dataSize)
}

func parseSampleRate(mimeType string, defaultRate int) int {
	for _, part := range strings.Split(mimeType, ";") {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) == 2 && strings.EqualFold(strings.TrimSpace(kv[0]), "rate") {
			if rate, err := strconv.Atoi(strings.TrimSpace(kv[1])); err == nil && rate > 0 {
				return rate
			}
		}
	}
	return defaultRate
}

func sanitizeDirSegment(s string) string {
	s = normalizeWord(s)
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		} else if r == ' ' {
			b.WriteRune('_')
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}
