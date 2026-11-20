package verboten

import (
	"context"
	"fmt"

	"google.golang.org/genai/interactions/models/decisions"
	"google.golang.org/genai/interactions/models/operations"
)

const (
	decisionModelName            = "gemini-decision-1"
	sameRootProbabilityThreshold = 0.75
)

func sameRootDecisionQuestion(lang, spokenWord, forbiddenWord string) (inputText, questionText, trueDesc, falseDesc string) {
	switch lang {
	case "fr":
		inputText = fmt.Sprintf("Mot prononcé : '%s'\nMot prohibé : '%s'", spokenWord, forbiddenWord)
		questionText = fmt.Sprintf("Le mot '%s' est-il trop proche du mot '%s' pour le jeu des mots prohibés, car ayant la même racine?", spokenWord, forbiddenWord)
		trueDesc = "Oui, les deux mots ont la même racine lexicale (flexion, conjugaison, pluriel, féminin, mot composé ou dérivé de la même famille de mots)."
		falseDesc = "Non, les deux mots n'ont pas la même racine lexicale (même s'ils sont synonymes ou liés par le sens, comme 'flamme' et 'feu', ou 'grand' et 'gros')."
	default:
		inputText = fmt.Sprintf("Spoken word: '%s'\nProscribed word: '%s'", spokenWord, forbiddenWord)
		questionText = fmt.Sprintf("Is the word '%s' too close to the word '%s' for the proscribed words game, because they share the same root?", spokenWord, forbiddenWord)
		trueDesc = "Yes, both words share the same lexical root/stem (inflection, verb conjugation, plural, compound, or derivative from the same word family)."
		falseDesc = "No, the two words do not share the same lexical root/stem (even if they are synonyms or semantically related, like 'flame' and 'fire', or 'big' and 'large')."
	}
	return inputText, questionText, trueDesc, falseDesc
}

func (vg *VerbotenGameServer) sameRootProbability(ctx context.Context, lang, spokenWord, forbiddenWord string) (float64, error) {
	if vg.sameRootProbFunc != nil {
		return vg.sameRootProbFunc(ctx, lang, spokenWord, forbiddenWord)
	}
	if vg.genaiClient == nil || vg.genaiClient.Decisions == nil {
		return 0, fmt.Errorf("decisions client is not initialized")
	}

	inputText, questionText, trueDesc, falseDesc := sameRootDecisionQuestion(lang, spokenWord, forbiddenWord)
	res, err := vg.genaiClient.Decisions.Create(ctx, operations.CreateDecisionRequest{
		Body: decisions.CreateDecisionRequest{
			Model: decisionModelName,
			Input: decisions.NewInput(inputText),
			Questions: map[string]decisions.Question{
				"same_root": decisions.NewQuestion(decisions.NoulQuestion{
					Instructions: decisions.NewNoulQuestionInstructions(questionText),
					Criteria: &decisions.NoulCriteria{
						TrueDescription:  &trueDesc,
						FalseDescription: &falseDesc,
					},
				}),
			},
		},
	})
	if err != nil {
		return 0, err
	}
	if res == nil || res.CreateDecisionResponse == nil {
		return 0, fmt.Errorf("empty decision response")
	}
	ans, ok := res.CreateDecisionResponse.Answers["same_root"]
	if !ok || ans.NoulAnswer == nil {
		return 0, fmt.Errorf("missing noul answer in decision response")
	}
	return ans.NoulAnswer.Noul, nil
}
