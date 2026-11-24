package verboten

import (
	"encoding/binary"
	"math"
	"sync"
	"time"

	"google.golang.org/genai"
)

const (
	// Adaptive speech energy thresholds for 16-bit PCM mono microphone input.
	judgeMinSpeechRMS        = 90.0
	judgeInitialNoiseFloor   = 25.0
	judgeMinNoiseFloor       = 10.0
	judgeMaxNoiseFloor       = 400.0
	judgeNoiseFloorFactor    = 3.0
	judgePreRollChunks       = 3 // ~192 ms at 64 ms/chunk
	judgeOverlapChunks       = 6 // ~384 ms overlap when splitting mid-speech
	judgeMinTurnSpeechMs     = 300
	judgeEndSilenceMs        = 550 // Real clause/sentence pause; keeps brief mid-clause pauses intact
	judgeSoftDipSilenceMs    = 128 // >= 2 quiet chunks so soft splits avoid intra-word stop consonants
	judgeSoftMaxSpeechMs     = 2500
	judgeHardMaxSpeechMs     = 4000
	judgeTurnResponseTimeout = 4 * time.Second
	judgeMaxPendingChunks    = 100 // ~6.4 s cap while waiting for turnComplete
)

type pcmAudioChunk struct {
	data     []byte
	mimeType string
}

// judgeVAD drives manual activity detection (ActivityStart / ActivityEnd) for the Judge
// Gemini Live session. gemini-3.1-flash-live-preview ignores short SilenceDurationMs values
// in AutomaticActivityDetection and can keep a turn open for 15+ seconds during continuous
// speech with brief pauses. By controlling turn boundaries server-side with NO_INTERRUPTION,
// the Judge evaluates speech after any ~550ms pause OR every ~2.5-4s of nonstop speech.
type judgeVAD struct {
	mu   sync.Mutex
	send func(genai.LiveRealtimeInput) error
	now  func() time.Time

	noiseFloor   float64
	inActivity   bool
	waitingTurn  bool
	turnEndedAt  time.Time
	speechMs     int64
	silenceMs    int64
	preRoll      []pcmAudioChunk
	overlap      []pcmAudioChunk
	pendingAudio []pcmAudioChunk
}

func newJudgeVAD(send func(genai.LiveRealtimeInput) error) *judgeVAD {
	return &judgeVAD{
		send:       send,
		now:        time.Now,
		noiseFloor: judgeInitialNoiseFloor,
	}
}

// FeedAudio processes a single PCM16 audio chunk from the human microphone and sends
// ActivityStart, audio blobs, and ActivityEnd signals to the Judge session as needed.
func (v *judgeVAD) FeedAudio(pcm []byte, mimeType string) error {
	if v == nil || len(pcm) == 0 {
		return nil
	}
	// Copy slice so buffered pre-roll / overlap / pending chunks never alias caller buffers.
	copied := make([]byte, len(pcm))
	copy(copied, pcm)
	chunk := pcmAudioChunk{data: copied, mimeType: mimeType}

	v.mu.Lock()
	defer v.mu.Unlock()

	// Safety valve: if Gemini somehow never delivered TurnComplete, unblock after timeout.
	if v.waitingTurn && !v.turnEndedAt.IsZero() && v.now().Sub(v.turnEndedAt) >= judgeTurnResponseTimeout {
		v.waitingTurn = false
		if err := v.flushPendingLocked(); err != nil {
			return err
		}
	}

	return v.feedChunkLocked(chunk)
}

// OnTurnComplete must be called when the Judge session emits ServerContent.TurnComplete.
// Any speech buffered while the Judge was evaluating the previous turn is flushed into a new turn.
func (v *judgeVAD) OnTurnComplete() error {
	if v == nil {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()

	if !v.waitingTurn {
		return nil
	}
	v.waitingTurn = false
	return v.flushPendingLocked()
}

func (v *judgeVAD) flushPendingLocked() error {
	for len(v.pendingAudio) > 0 && !v.waitingTurn {
		ch := v.pendingAudio[0]
		v.pendingAudio = v.pendingAudio[1:]
		if err := v.feedChunkLocked(ch); err != nil {
			return err
		}
	}
	return nil
}

func (v *judgeVAD) speechThresholdLocked() float64 {
	return math.Max(judgeMinSpeechRMS, v.noiseFloor*judgeNoiseFloorFactor)
}

func (v *judgeVAD) feedChunkLocked(chunk pcmAudioChunk) error {
	if v.waitingTurn {
		v.pendingAudio = append(v.pendingAudio, chunk)
		if len(v.pendingAudio) > judgeMaxPendingChunks {
			v.pendingAudio = v.pendingAudio[len(v.pendingAudio)-judgeMaxPendingChunks:]
		}
		return nil
	}

	rms := pcm16RMS(chunk.data)
	threshold := v.speechThresholdLocked()
	isSpeech := rms >= threshold
	if !isSpeech {
		v.noiseFloor = 0.9*v.noiseFloor + 0.1*rms
		if v.noiseFloor < judgeMinNoiseFloor {
			v.noiseFloor = judgeMinNoiseFloor
		} else if v.noiseFloor > judgeMaxNoiseFloor {
			v.noiseFloor = judgeMaxNoiseFloor
		}
	}

	chunkMs := pcm16DurationMs(len(chunk.data), chunk.mimeType)

	if !v.inActivity {
		// Keep a short pre-roll before speech onset so leading consonants are not clipped.
		v.preRoll = append(v.preRoll, chunk)
		if len(v.preRoll) > judgePreRollChunks {
			v.preRoll = v.preRoll[len(v.preRoll)-judgePreRollChunks:]
		}
		if !isSpeech {
			v.silenceMs += chunkMs
			if v.silenceMs >= judgeEndSilenceMs {
				// Discard any overlap saved from a previous soft/hard cut once the speaker truly paused.
				v.overlap = nil
			}
			return nil
		}

		// Start a new Judge turn with either overlap from a mid-speech split or pre-roll.
		var prefix []pcmAudioChunk
		if len(v.overlap) > 0 {
			prefix = append(prefix, v.overlap...)
			// Append the current speech chunk too (overlap comes from the previous turn).
			prefix = append(prefix, chunk)
			v.overlap = nil
		} else {
			prefix = append(prefix, v.preRoll...)
		}
		v.preRoll = nil

		v.inActivity = true
		v.speechMs = chunkMs
		v.silenceMs = 0

		if err := v.send(genai.LiveRealtimeInput{ActivityStart: &genai.ActivityStart{}}); err != nil {
			return err
		}
		for _, p := range prefix {
			if err := v.send(genai.LiveRealtimeInput{
				Audio: &genai.Blob{Data: p.data, MIMEType: p.mimeType},
			}); err != nil {
				return err
			}
		}
		v.overlap = append(v.overlap[:0], chunk)
		return nil
	}

	// Currently inside an active Judge turn: stream chunk immediately.
	if err := v.send(genai.LiveRealtimeInput{
		Audio: &genai.Blob{Data: chunk.data, MIMEType: chunk.mimeType},
	}); err != nil {
		return err
	}

	v.speechMs += chunkMs
	v.overlap = append(v.overlap, chunk)
	if len(v.overlap) > judgeOverlapChunks {
		v.overlap = v.overlap[len(v.overlap)-judgeOverlapChunks:]
	}

	if isSpeech {
		v.silenceMs = 0
	} else {
		v.silenceMs += chunkMs
	}

	if v.silenceMs >= judgeEndSilenceMs && v.speechMs >= judgeMinTurnSpeechMs {
		// Clean silence boundary: no overlap replay needed on the next turn.
		v.overlap = nil
		return v.endActivityLocked()
	}
	if v.speechMs >= judgeSoftMaxSpeechMs && v.silenceMs >= judgeSoftDipSilenceMs {
		// Brief inter-word dip (>= 128ms) after >= 2.5s of speech: cut here and preserve overlap.
		return v.endActivityLocked()
	}
	if v.speechMs >= judgeHardMaxSpeechMs {
		// Continuous speech with no pauses for 4s: force evaluation with overlap.
		return v.endActivityLocked()
	}
	return nil
}

func (v *judgeVAD) endActivityLocked() error {
	v.inActivity = false
	v.waitingTurn = true
	v.turnEndedAt = v.now()
	v.speechMs = 0
	v.silenceMs = 0
	return v.send(genai.LiveRealtimeInput{ActivityEnd: &genai.ActivityEnd{}})
}

func pcm16RMS(pcm []byte) float64 {
	n := len(pcm) / 2
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		s := float64(int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2])))
		sum += s * s
	}
	return math.Sqrt(sum / float64(n))
}

func pcm16DurationMs(numBytes int, mimeType string) int64 {
	sr := parseSampleRate(mimeType, 16000)
	if sr <= 0 {
		sr = 16000
	}
	ms := int64(numBytes) * 1000 / int64(sr*2)
	if ms <= 0 && numBytes > 0 {
		return 1
	}
	return ms
}
