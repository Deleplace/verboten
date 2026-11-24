package verboten

import (
	"context"
	"fmt"

	"google.golang.org/genai/interactions/models/decisions"
	"google.golang.org/genai/interactions/models/operations"
)

const (
	decisionModelName                     = "gemini-decision-1"
	sameRootProbabilityThreshold          = 0.75
	samePronunciationProbabilityThreshold = 0.75
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

func samePronunciationDecisionQuestion(lang, spokenWord, forbiddenWord string) (inputText, questionText, trueDesc, falseDesc string) {
	switch lang {
	case "fr":
		inputText = fmt.Sprintf("Mot 1 : '%s'\nMot 2 : '%s'", spokenWord, forbiddenWord)
		questionText = fmt.Sprintf("En français, le mot '%s' se prononce-t-il exactement de la même façon que le mot '%s' (homophones / même prononciation orale) ?", spokenWord, forbiddenWord)
		trueDesc = "Oui, les deux mots ont exactement la même prononciation à l'oral en français (par exemple un pluriel silencieux comme 'clous' et 'Clou', 'poissons' et 'Poisson', 'arbres' et 'Arbre', ou 'mangé' et 'Manger')."
		falseDesc = "Non, les deux mots se prononcent différemment à l'oral en français (par exemple 'porte' et 'Porter', 'joue' et 'Jouer', 'tranchante' et 'Tranchant', 'cheval' et 'Chevaux')."
	default:
		inputText = fmt.Sprintf("Word 1: '%s'\nWord 2: '%s'", spokenWord, forbiddenWord)
		questionText = fmt.Sprintf("In English, is the word '%s' pronounced the exact same way as the word '%s' (identical spoken pronunciation / homophones)?", spokenWord, forbiddenWord)
		trueDesc = "Yes, both words have the exact same spoken pronunciation in English (homophones)."
		falseDesc = "No, the two words are pronounced differently in English (for example 'trees' and 'Tree', 'playing' and 'Play', 'birds' and 'Bird', 'cheesy' and 'Cheese')."
	}
	return inputText, questionText, trueDesc, falseDesc
}

func (vg *VerbotenGameServer) samePronunciationProbability(ctx context.Context, lang, spokenWord, forbiddenWord string) (float64, error) {
	if vg.samePronunciationProbFunc != nil {
		return vg.samePronunciationProbFunc(ctx, lang, spokenWord, forbiddenWord)
	}
	if vg.genaiClient == nil || vg.genaiClient.Decisions == nil {
		return 0, fmt.Errorf("decisions client is not initialized")
	}

	inputText, questionText, trueDesc, falseDesc := samePronunciationDecisionQuestion(lang, spokenWord, forbiddenWord)
	res, err := vg.genaiClient.Decisions.Create(ctx, operations.CreateDecisionRequest{
		Body: decisions.CreateDecisionRequest{
			Model: decisionModelName,
			Input: decisions.NewInput(inputText),
			Questions: map[string]decisions.Question{
				"same_pronunciation": decisions.NewQuestion(decisions.NoulQuestion{
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
	ans, ok := res.CreateDecisionResponse.Answers["same_pronunciation"]
	if !ok || ans.NoulAnswer == nil {
		return 0, fmt.Errorf("missing noul answer in decision response")
	}
	return ans.NoulAnswer.Noul, nil
}
