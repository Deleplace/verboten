package verboten

import (
	"encoding/binary"
	"math"
	"sync"

	"google.golang.org/genai"
)

const (
	// Adaptive speech energy thresholds for 16-bit PCM mono microphone input.
	judgeMinSpeechRMS      = 90.0
	judgeInitialNoiseFloor = 25.0
	judgeMinNoiseFloor     = 10.0
	judgeMaxNoiseFloor     = 400.0
	judgeNoiseFloorFactor  = 3.0
	judgePreRollChunks     = 3 // ~192 ms at 64 ms/chunk
	judgeOverlapChunks     = 3 // ~192 ms overlap when hard-splitting nonstop speech
	judgeMinTurnSpeechMs   = 300
	judgeEndSilenceMs      = 550 // Real clause/sentence pause; keeps brief mid-clause pauses intact
	judgeSoftDipSilenceMs  = 128 // >= 2 quiet chunks so soft splits land between words, not inside stop consonants
	judgeSoftMaxSpeechMs   = 2500
	judgeHardMaxSpeechMs   = 4000
)

type pcmAudioChunk struct {
	data     []byte
	mimeType string
}

// judgeVAD drives manual activity detection (ActivityStart / ActivityEnd) for the Judge
// Gemini Live session using NO_INTERRUPTION. Audio is always streamed at 1.0x real-time
// (never buffered and burst-flushed, which Vertex AI LlmBidiService drops), while turn
// boundaries are emitted on any ~550ms pause OR every ~2.5-4s of continuous speech so the
// Judge never waits 15+ seconds to flag a forbidden word.
type judgeVAD struct {
	mu   sync.Mutex
	send func(genai.LiveRealtimeInput) error

	noiseFloor float64
	inActivity bool
	speechMs   int64
	silenceMs  int64
	preRoll    []pcmAudioChunk
	overlap    []pcmAudioChunk
}

func newJudgeVAD(send func(genai.LiveRealtimeInput) error) *judgeVAD {
	return &judgeVAD{
		send:       send,
		noiseFloor: judgeInitialNoiseFloor,
	}
}

// FeedAudio processes a single PCM16 audio chunk from the human microphone and sends
// ActivityStart, audio blobs, and ActivityEnd signals to the Judge session in real time.
func (v *judgeVAD) FeedAudio(pcm []byte, mimeType string) error {
	if v == nil || len(pcm) == 0 {
		return nil
	}
	copied := make([]byte, len(pcm))
	copy(copied, pcm)
	chunk := pcmAudioChunk{data: copied, mimeType: mimeType}

	v.mu.Lock()
	defer v.mu.Unlock()

	return v.feedChunkLocked(chunk)
}

func (v *judgeVAD) speechThresholdLocked() float64 {
	return math.Max(judgeMinSpeechRMS, v.noiseFloor*judgeNoiseFloorFactor)
}

func (v *judgeVAD) feedChunkLocked(chunk pcmAudioChunk) error {
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
		v.preRoll = append(v.preRoll, chunk)
		if len(v.preRoll) > judgePreRollChunks {
			v.preRoll = v.preRoll[len(v.preRoll)-judgePreRollChunks:]
		}
		if !isSpeech {
			v.silenceMs += chunkMs
			if v.silenceMs >= judgeEndSilenceMs {
				v.overlap = nil
			}
			return nil
		}

		// Start a new Judge turn with either overlap from a hard mid-speech split or pre-roll.
		var prefix []pcmAudioChunk
		if len(v.overlap) > 0 {
			prefix = append(prefix, v.overlap...)
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

	// Currently inside an active Judge turn: stream chunk immediately at 1.0x real-time.
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
		// Clean silence boundary: no overlap needed on the next turn.
		v.overlap = nil
		return v.endActivityLocked()
	}
	if v.speechMs >= judgeSoftMaxSpeechMs && v.silenceMs >= judgeSoftDipSilenceMs {
		// Inter-word dip (>= 128ms) after >= 2.5s of speech: clean boundary without word duplication.
		v.overlap = nil
		return v.endActivityLocked()
	}
	if v.speechMs >= judgeHardMaxSpeechMs {
		// Continuous speech with no pauses for 4s: force evaluation with short overlap.
		return v.endActivityLocked()
	}
	return nil
}

func (v *judgeVAD) endActivityLocked() error {
	v.inActivity = false
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
