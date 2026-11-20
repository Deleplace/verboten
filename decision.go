package verboten

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"google.golang.org/genai"
)

const decisionModelName = "gemini-3.5-flash-lite"

func sameRootPrompt(lang, spokenWord, forbiddenWord string) string {
	switch lang {
	case "fr":
		return fmt.Sprintf(`Mot prononcé : '%s'
Mot prohibé : '%s'

Le mot '%s' est-il trop proche du mot '%s' pour le jeu des mots prohibés, car ayant la même racine?
- Yes : les deux mots ont la même racine lexicale (flexion, conjugaison, pluriel, féminin, mot composé ou dérivé de la même famille de mots).
- No : les deux mots n'ont pas la même racine lexicale (même s'ils sont synonymes ou liés par le sens, comme 'flamme' et 'feu', ou 'grand' et 'gros').

Réponds uniquement par "Yes" ou "No".`, spokenWord, forbiddenWord, spokenWord, forbiddenWord)
	default:
		return fmt.Sprintf(`Spoken word: '%s'
Proscribed word: '%s'

Is the word '%s' too close to the word '%s' for the proscribed words game, because they share the same root?
- Yes: both words share the same lexical root/stem (inflection, verb conjugation, plural, compound, or derivative from the same word family).
- No: the two words do not share the same lexical root/stem (even if they are synonyms or semantically related, like 'flame' and 'fire', or 'big' and 'large').

Answer only "Yes" or "No".`, spokenWord, forbiddenWord, spokenWord, forbiddenWord)
	}
}

func parseYesNo(raw string) (sameRoot bool, normalized string, err error) {
	cleaned := strings.TrimSpace(raw)
	cleaned = strings.Trim(cleaned, ".,!?:;\"'`")
	lower := strings.ToLower(cleaned)
	fields := strings.Fields(lower)
	if len(fields) == 0 {
		return false, "", fmt.Errorf("empty yes/no response")
	}
	first := strings.Trim(fields[0], ".,!?:;\"'`")
	switch first {
	case "yes", "oui":
		return true, "Yes", nil
	case "no", "non":
		return false, "No", nil
	default:
		return false, "", fmt.Errorf("unexpected yes/no response: %q", raw)
	}
}

func (vg *VerbotenGameServer) isSameRoot(ctx context.Context, lang, spokenWord, forbiddenWord string) (sameRoot bool, answer string, elapsed time.Duration, err error) {
	start := time.Now()
	defer func() {
		elapsed = time.Since(start)
		if vg.onSameRootDuration != nil {
			vg.onSameRootDuration(elapsed)
		}
	}()

	if vg.sameRootCheckFunc != nil {
		sameRoot, err = vg.sameRootCheckFunc(ctx, lang, spokenWord, forbiddenWord)
		if err != nil {
			return false, "", 0, err
		}
		if sameRoot {
			return true, "Yes", 0, nil
		}
		return false, "No", 0, nil
	}
	if vg.genaiClient == nil || vg.genaiClient.Models == nil {
		return false, "", 0, fmt.Errorf("models client is not initialized")
	}

	prompt := sameRootPrompt(lang, spokenWord, forbiddenWord)
	res, err := vg.genaiClient.Models.GenerateContent(ctx, decisionModelName, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](0),
		ThinkingConfig: &genai.ThinkingConfig{
			ThinkingLevel: genai.ThinkingLevelMinimal,
		},
	})
	if err != nil {
		return false, "", 0, err
	}
	sameRoot, answer, err = parseYesNo(res.Text())
	if err != nil {
		return false, "", 0, err
	}
	return sameRoot, answer, 0, nil
}

type durationStats struct {
	Count  int
	Avg    time.Duration
	Median time.Duration
	Min    time.Duration
	Max    time.Duration
}

func summarizeDurations(durations []time.Duration) durationStats {
	n := len(durations)
	if n == 0 {
		return durationStats{}
	}
	sorted := make([]time.Duration, n)
	copy(sorted, durations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var total time.Duration
	for _, d := range sorted {
		total += d
	}
	var median time.Duration
	if n%2 == 1 {
		median = sorted[n/2]
	} else {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return durationStats{
		Count:  n,
		Avg:    total / time.Duration(n),
		Median: median,
		Min:    sorted[0],
		Max:    sorted[n-1],
	}
}
