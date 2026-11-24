package verboten

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"
)

const (
	DefaultTTSModel   = "gemini-3.8-flash-lite-tts"
	DefaultJudgeVoice = "Charon" // Deep masculine voice, distinct from the Guesser's "Puck" voice.
	judgeTTSStyleTag  = "[slightly fast] "

	judgeTTSTimeout       = 8 * time.Second
	judgeTTSCacheMaxItems = 256
	judgeTTSMaxTextRunes  = 300
)

type cachedTTSAudio struct {
	audio    []byte
	mimeType string
}

func languageCodeForLang(lang string) string {
	switch lang {
	case "fr":
		return "fr-FR"
	case "ar":
		return "ar-EG"
	default:
		return "en-US"
	}
}

// formatJudgeLossText builds the spoken and displayed verdict sentence for the Judge
// in the game's language.
func formatJudgeLossText(lang, spokenWord, forbiddenWord, reason string) string {
	spoken := strings.TrimSpace(spokenWord)
	forbidden := strings.TrimSpace(forbiddenWord)

	switch lang {
	case "fr":
		if spoken == "" || forbidden == "" {
			word := forbidden
			if word == "" {
				word = spoken
			}
			return fmt.Sprintf("Vous avez dit le mot interdit « %s ».", word)
		}
		switch reason {
		case "translation":
			return fmt.Sprintf("Vous avez dit le mot « %s », qui est une traduction du mot à deviner « %s ».", spoken, forbidden)
		case "exact":
			return fmt.Sprintf("Vous avez dit le mot interdit « %s ».", forbidden)
		default:
			if normalizeWord(spoken) == normalizeWord(forbidden) {
				return fmt.Sprintf("Vous avez dit le mot interdit « %s ».", forbidden)
			}
			return fmt.Sprintf("Vous avez dit le mot « %s », qui est trop proche du mot interdit « %s ».", spoken, forbidden)
		}
	case "ar":
		if spoken == "" || forbidden == "" {
			word := forbidden
			if word == "" {
				word = spoken
			}
			return fmt.Sprintf("لقد قلت الكلمة الممنوعة «%s».", word)
		}
		switch reason {
		case "translation":
			return fmt.Sprintf("لقد قلت كلمة «%s»، وهي ترجمة للكلمة المطلوب تخمينها «%s».", spoken, forbidden)
		case "exact":
			return fmt.Sprintf("لقد قلت الكلمة الممنوعة «%s».", forbidden)
		default:
			if normalizeWord(spoken) == normalizeWord(forbidden) {
				return fmt.Sprintf("لقد قلت الكلمة الممنوعة «%s».", forbidden)
			}
			return fmt.Sprintf("لقد قلت كلمة «%s»، وهي قريبة جداً من الكلمة الممنوعة «%s».", spoken, forbidden)
		}
	default:
		if spoken == "" || forbidden == "" {
			word := forbidden
			if word == "" {
				word = spoken
			}
			return fmt.Sprintf("You said the forbidden word %q.", word)
		}
		switch reason {
		case "translation":
			return fmt.Sprintf("You said the word %q, which is a translation of the target word %q.", spoken, forbidden)
		case "exact":
			return fmt.Sprintf("You said the forbidden word %q.", forbidden)
		default:
			if normalizeWord(spoken) == normalizeWord(forbidden) {
				return fmt.Sprintf("You said the forbidden word %q.", forbidden)
			}
			return fmt.Sprintf("You said the word %q, which is too close to the forbidden word %q.", spoken, forbidden)
		}
	}
}

func (vg *VerbotenGameServer) ttsModel() string {
	if vg != nil && strings.TrimSpace(vg.TTSModel) != "" {
		return strings.TrimSpace(vg.TTSModel)
	}
	return DefaultTTSModel
}

func (vg *VerbotenGameServer) judgeVoice() string {
	if vg != nil && strings.TrimSpace(vg.JudgeVoice) != "" {
		return strings.TrimSpace(vg.JudgeVoice)
	}
	return DefaultJudgeVoice
}

// synthesizeJudgeSpeech generates spoken audio for the Judge using Gemini TTS with a
// masculine voice distinct from the Guesser AI, caching results in memory by (lang, text).
func (vg *VerbotenGameServer) synthesizeJudgeSpeech(ctx context.Context, lang, text string) ([]byte, string, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, "", fmt.Errorf("empty text for TTS")
	}

	if vg.judgeTTSFunc != nil {
		return vg.judgeTTSFunc(ctx, lang, trimmed)
	}

	cacheKey := lang + "\x00" + trimmed
	vg.ttsCacheMu.RLock()
	if cached, ok := vg.ttsCache[cacheKey]; ok {
		vg.ttsCacheMu.RUnlock()
		return cached.audio, cached.mimeType, nil
	}
	vg.ttsCacheMu.RUnlock()

	if vg.genaiClient == nil || vg.genaiClient.Models == nil {
		return nil, "", fmt.Errorf("genai models client is not initialized")
	}

	resp, err := vg.genaiClient.Models.GenerateContent(ctx, vg.ttsModel(), genai.Text(judgeTTSStyleTag+trimmed), &genai.GenerateContentConfig{
		ResponseModalities: []string{"AUDIO"},
		SpeechConfig: &genai.SpeechConfig{
			LanguageCode: languageCodeForLang(lang),
			VoiceConfig: &genai.VoiceConfig{
				PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
					VoiceName: vg.judgeVoice(),
				},
			},
		},
	})
	if err != nil {
		return nil, "", err
	}
	if resp == nil {
		return nil, "", fmt.Errorf("empty TTS response")
	}

	var (
		audio    []byte
		mimeType string
	)
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part != nil && part.InlineData != nil && len(part.InlineData.Data) > 0 {
				if mimeType == "" {
					mimeType = part.InlineData.MIMEType
				}
				audio = append(audio, part.InlineData.Data...)
			}
		}
	}
	if len(audio) == 0 {
		return nil, "", fmt.Errorf("TTS response contained no audio")
	}
	if mimeType == "" {
		mimeType = "audio/wav"
	}

	vg.ttsCacheMu.Lock()
	if vg.ttsCache == nil {
		vg.ttsCache = make(map[string]cachedTTSAudio)
	}
	if len(vg.ttsCache) >= judgeTTSCacheMaxItems {
		clear(vg.ttsCache)
	}
	vg.ttsCache[cacheKey] = cachedTTSAudio{audio: audio, mimeType: mimeType}
	vg.ttsCacheMu.Unlock()

	return audio, mimeType, nil
}

type ttsRequest struct {
	Lang string `json:"lang"`
	Text string `json:"text"`
}

func (vg *VerbotenGameServer) serveTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req ttsRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req); err != nil {
		http.Error(w, "invalid JSON request", http.StatusBadRequest)
		return
	}
	req.Text = strings.TrimSpace(req.Text)
	if req.Text == "" || len([]rune(req.Text)) > judgeTTSMaxTextRunes {
		http.Error(w, "text must be between 1 and 300 characters", http.StatusBadRequest)
		return
	}
	if req.Lang == "" {
		req.Lang = "en"
	}

	ttsCtx, cancel := context.WithTimeout(r.Context(), judgeTTSTimeout)
	defer cancel()

	audio, mimeType, err := vg.synthesizeJudgeSpeech(ttsCtx, req.Lang, req.Text)
	if err != nil {
		vg.logger().Warn("judge TTS HTTP synthesis failed", "lang", req.Lang, "err", err)
		http.Error(w, "TTS synthesis failed", http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Cache-Control", "public, max-age=3600")
	_, _ = w.Write(audio)
}
