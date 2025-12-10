package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"
	"unicode"

	"golang.org/x/sync/errgroup"
	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
	"google.golang.org/genai"
)

type forbiddenWord struct {
	Word         string   `json:"word"`
	Forbidden    []string `json:"forbidden"`
	languageName string
}
type wordsByLang struct {
	En []forbiddenWord `json:"en"`
	Fr []forbiddenWord `json:"fr"`
	Ar []forbiddenWord `json:"ar"`
}

type uiPhrases struct {
	chooseLanguage          string
	wordToDescribe          string
	forbiddenWordsAre       string
	describeTheWord         string
	usedForbiddenWord       string
	usedForbiddenInflection string
	aiGuess                 string
	aiGuessedTheWord        string
	wordWas                 string
}

var phrases = map[string]uiPhrases{
	"en": {
		chooseLanguage:          "Choose your language (en/fr/ar): ",
		wordToDescribe:          "The word to describe is: %s\n",
		forbiddenWordsAre:       "The proscribed words are: %s\n",
		describeTheWord:         "\nDescribe the word.\n> ",
		usedForbiddenWord:       "Oh! You used the proscribed word '%s'. You lose!\n",
		usedForbiddenInflection: "Oh! You sais '%s' which is too close to the proscribed word '%s'. You lose!\n",
		aiGuess:                 "AI: %s\n",
		aiGuessedTheWord:        "\nThe AI guessed the word! You win!\n",
		wordWas:                 "\nThe word was %s. You lose!\n",
	},
	"fr": {
		chooseLanguage:          "Choisissez votre langue (en/fr/ar): ",
		wordToDescribe:          "Le mot à décrire est : %s\n",
		forbiddenWordsAre:       "Les mots prohibés sont : %s\n",
		describeTheWord:         "\nDécrivez le mot.\n> ",
		usedForbiddenWord:       "Oh! Vous avez utilisé le mot prohibé '%s'. Vous avez perdu !\n",
		usedForbiddenInflection: "Oh! Vous avez dit '%s' qui est trop proche du mot prohibé '%s'. Vous avez perdu !\n",
		aiGuess:                 "IA : %s\n",
		aiGuessedTheWord:        "\nL'IA a deviné le mot ! Vous avez gagné !\n",
		wordWas:                 "\nLe mot était %s. Vous avez perdu !\n",
	},
	"ar": {
		chooseLanguage:          "اختر لغتك (en/fr/ar): ",
		wordToDescribe:          "الكلمة التي يجب وصفها هي: %s\n",
		forbiddenWordsAre:       "الكلمات الممنوعة هي: %s\n",
		describeTheWord:         "\nصف الكلمة.\n> ",
		usedForbiddenWord:       "أوه! لقد استخدمت الكلمة الممنوعة '%s'. لقد خسرت!\n",
		usedForbiddenInflection: "أوه! لقد قلت '%s' وهي قريبة جدًا من الكلمة الممنوعة '%s'. لقد خسرت!\n",
		aiGuess:                 "الذكاء الاصطناعي: %s\n",
		aiGuessedTheWord:        "\nلقد خمن الذكاء الاصطناعي الكلمة! لقد فزت!\n",
		wordWas:                 "\nكانت الكلمة %s. لقد خسرت!\n",
	},
}

const modelName = "gemini-2.5-flash-lite"

var client *genai.Client

func main() {
	ctx := context.Background()

	//
	// Create the Gemini client
	//
	var err error
	for _, k := range []string{
		"GOOGLE_API_KEY",
		"GOOGLE_GENAI_USE_VERTEXAI",
		"GOOGLE_CLOUD_PROJECT",
		"GOOGLE_CLOUD_LOCATION",
	} {
		// fmt.Printf("%s=%s\n", k, os.Getenv(k))
		_ = k
	}
	client, err = genai.NewClient(ctx, &genai.ClientConfig{
		// empty ClientConfig automatically uses the env vars listed above
	})
	if err != nil {
		log.Fatal(err)
	}
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		// fmt.Println("(using VertexAI backend)")
	} else {
		// fmt.Println("(using GeminiAPI backend)")
	}

	// Load words from JSON file
	file, err := os.ReadFile("assets/words.json")
	if err != nil {
		log.Fatalf("failed to read words file: %v", err)
	}

	var allWords wordsByLang
	if err := json.Unmarshal(file, &allWords); err != nil {
		log.Fatalf("failed to parse words file: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	var words []forbiddenWord
	var instructions string
	var currentPhrases uiPhrases
	var langName string

	var langChosen = false
	for !langChosen {
		fmt.Print(phrases["en"].chooseLanguage)
		lang, _ := reader.ReadString('\n')
		lang = strings.TrimSpace(lang)

		switch lang {
		case "fr":
			langChosen = true
			words = allWords.Fr
			currentPhrases = phrases["fr"]
			instructions = `
				Tu es le devineur dans une partie de "Mots Prohibés".
				Je vais te décrire un mot. Tu dois deviner ce que c'est.
				Tu n'as que 3 essais.
				Je connais le mot à faire deviner, mais je ne peux pas te le dire.
				Je ne peux pas non plus te dire plusieurs mots prohibés.
				Réponds uniquement en Français.
				Réponds uniquement le mot que tu supposes être celui que j'essaie de faire deviner.
				Commençons.
				`
			langName = "French"
		case "en":
			langChosen = true
			words = allWords.En
			currentPhrases = phrases["en"]
			instructions = `
				You are the guesser in a game of "Proscribed Words".
				I will describe a word to you. You have to guess what it is.
				You only have 3 guesses.
				I know the word to guess, but I cannot say it to you.
				I also cannot say several other proscribed words.
				Answer only in English.
				Answer only with the word you think is the one I'm trying to let you guess.
				Let's start.
				`
			langName = "English"
		case "ar":
			langChosen = true
			words = allWords.Ar
			currentPhrases = phrases["ar"]
			instructions = `
				أنت المخمن في لعبة "الكلمات الممنوعة".
				سأصف لك كلمة. عليك أن تخمن ما هي.
				لديك 3 محاولات فقط.
				أعرف الكلمة التي يجب تخمينها، لكن لا يمكنني قولها لك.
				كما لا يمكنني قول العديد من الكلمات الممنوعة الأخرى.
				أجب باللغة العربية فقط.
				أجب فقط بالكلمة التي تعتقد أنها الكلمة التي أحاول أن أجعلك تخمنها.
				لنبدأ.
				`
			langName = "Arabic"
		}
	}
	//fmt.Println(instructions)

	// Pick a random word
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	gameWord := words[r.Intn(len(words))]
	gameWord.languageName = langName

	fmt.Println()
	fmt.Printf(currentPhrases.wordToDescribe, gameWord.Word)
	fmt.Printf(currentPhrases.forbiddenWordsAre, strings.Join(gameWord.Forbidden, ", "))

	var config *genai.GenerateContentConfig = &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{Text: instructions},
			},
		},
	}

	chat, err := client.Chats.Create(ctx, modelName, config, nil)
	if err != nil {
		log.Fatal(err)
	}

	guesses := 3
	for guesses > 0 {
		fmt.Println(currentPhrases.describeTheWord)
		description, _ := reader.ReadString('\n')
		description = strings.TrimSpace(description)

		g := new(errgroup.Group)

		// Check for proscribed words
		var lost bool
		var forbiddenSaid, forbiddenMatched string
		g.Go(func() error {
			lost, forbiddenSaid, forbiddenMatched, err = gameWord.saidForbidden(ctx, description)
			return err
		})

		// Let Gemini guess, concurrently
		var result *genai.GenerateContentResponse
		g.Go(func() error {
			result, err = chat.SendMessage(ctx, genai.Part{Text: description})
			return err
		})

		err := g.Wait()
		if err != nil {
			log.Fatal(err)
		}

		if lost {
			if normalize(forbiddenSaid) == normalize(forbiddenMatched) {
				// Exact match
				fmt.Printf(currentPhrases.usedForbiddenWord, forbiddenMatched)
			} else {
				// Fuzzy match
				fmt.Printf(currentPhrases.usedForbiddenInflection, forbiddenSaid, forbiddenMatched)
			}
			return
		}

		// AI's guess
		aiResponse := textOf(result)
		fmt.Printf(currentPhrases.aiGuess, aiResponse)

		winning, err := gameWord.isWinning(ctx, aiResponse)
		if err != nil {
			log.Fatal(err)
		}

		if winning {
			fmt.Println(currentPhrases.aiGuessedTheWord)
			return
		}
		guesses--
	}

	fmt.Printf(currentPhrases.wordWas, gameWord.Word)
}

func (fw *forbiddenWord) isWinning(ctx context.Context, guess string) (bool, error) {
	lowGuess := normalize(guess)
	lowGoal := normalize(fw.Word)
	return strings.Contains(lowGuess, lowGoal), nil
}

// normalize returns its argument lowercased and without diacritics
func normalize(s string) string {
	// Local transformers, not shared with other goroutines
	tr := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	normalized, _, err := transform.String(tr, strings.ToLower(s))
	if err != nil {
		// We do not expect string transformation to fail in general
		panic(err)
	}
	return normalized
}

func (fw *forbiddenWord) saidForbidden(ctx context.Context, said string) (lost bool, forbiddenSaid string, forbiddenMatched string, err error) {
	systemInstruction := `
		You are the judge in the Proscribed Words game.
		The human player will say a description.

		If the prompt contains any of the proscribed words, or an inflection of a forbidden
		word, or a proscribed word translated in another language, then the game is lost.

		The proscribed words are:
		` + fw.Word + ", " + strings.Join(fw.Forbidden, ", ") + `

		In the field "forbiddenWord", provide exactly one of the original proscribed words.

		In the field "fragment", provide the part of the prompt that violated the rule.

		The description must be rejected as using a proscribed word only if it actually contains
		an inflection, or misspelling, or translation of a proscribed word.

		Synonyms of proscribed words must not trigger a lost game.

		E.g. "ficelle" does not match the proscribed word "Corde", because the two words have
		a similar meaning but the word "ficelle" is not an inflection of the word "corde" and
		the game is not lost.

		E.g. "orange" does not match the proscribed word "Agrume", because the two words have
		a similar meaning but the word "orange" is not an inflection of the word "Agrume" and
		the game is not lost.

		E.g. "tronc" does not match the proscribed word "Arbre", because the two words have
		related meaning but the word "tronc" is not an inflection of the word "Arbre" and
		the game is not lost.

		E.g. "poussent" matches the proscribed word "Pousser", because "poussent" is a
		conjugation of the verb "Pousser", thus it is an inflection of "Pousser" and the game
		is lost.
`

	// Force JSON structured output
	config := &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromParts([]*genai.Part{
			{Text: systemInstruction},
		}, genai.RoleModel),
		ResponseMIMEType: "application/json",
		ResponseJsonSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"lost": {
					Type:        genai.TypeBoolean,
					Description: "Indicates if the user has lost the game.",
				},
				"forbiddenWord": {
					Type:        genai.TypeString,
					Description: "The word that triggered the loss condition.",
				},
				"fragment": {
					Type:        genai.TypeString,
					Description: "The text fragment analyzed.",
				},
			},
			Required: []string{"lost"},
		},
	}

	prompt := []*genai.Content{
		genai.NewContentFromParts([]*genai.Part{
			{Text: said},
		}, genai.RoleUser),
	}

	resp, err := client.Models.GenerateContent(ctx, modelName, prompt, config)

	if err != nil {
		return false, "", "", err
	}

	structureAnswer := resp.Candidates[0].Content.Parts[0].Text

	// Parse structureAnswer to return the fields
	var result struct {
		Lost          bool   `json:"lost"`
		ForbiddenWord string `json:"forbiddenWord"`
		Fragment      string `json:"fragment"`
	}
	if err := json.Unmarshal([]byte(structureAnswer), &result); err != nil {
		return false, "", "", fmt.Errorf("failed to parse AI response: %w", err)
	}

	if !result.Lost {
		return false, "", "", nil
	}

	// Sometimes words are incorrectly detected as proscribed, just because they are
	// semantically close to one of the proscribed words.
	// E.g. " 'nuages' est trop proche du mot prohibé 'Ciel' "
	//
	// Let's double-check if the suspicious fragment is actually either an inflection,
	// or a translation, of the proscribed word.
	var isSameRoot, isTranslated bool

	g := new(errgroup.Group)
	g.Go(func() error {
		isSameRoot, err = haveSameRoot(ctx, result.Fragment, result.ForbiddenWord)
		return err
	})
	g.Go(func() error {
		isTranslated, err = isTranslation(ctx, result.Fragment, result.ForbiddenWord, fw.languageName)
		return err
	})
	err = g.Wait()
	if err != nil {
		return false, "", "", err
	}

	if !isSameRoot && !isTranslated {
		// False alarm
		fmt.Printf("\nJudge says: the words '%s' and '%s' looked suspiciously similar, but not for sure\n", result.Fragment, result.ForbiddenWord)
		return false, "", "", nil
	}

	if isSameRoot {
		fmt.Printf("\nJudge says: the words '%s' and '%s' have the same root\n", result.Fragment, result.ForbiddenWord)
	}

	if isTranslated {
		fmt.Printf("\nJudge says: '%s' is a translation of the proscribed word '%s'\n", result.Fragment, result.ForbiddenWord)
	}

	return result.Lost, result.Fragment, result.ForbiddenWord, nil
}

func haveSameRoot(ctx context.Context, word1, word2 string) (bool, error) {
	// TODO stemmer e.g. PorterStemmer
	prompt := []*genai.Content{
		genai.NewContentFromParts([]*genai.Part{
			{Text: fmt.Sprintf(
				`Can we say that the words '%s' and '%s' share the same root?
				 Answer just Yes or No, and nothing else.`, word1, word2)},
		}, genai.RoleUser),
	}

	resp, err := client.Models.GenerateContent(ctx, modelName, prompt, nil)

	if err != nil {
		return false, err
	}

	answer := strings.ToLower(resp.Candidates[0].Content.Parts[0].Text)

	return answer == "yes", nil
}

func isTranslation(ctx context.Context, word1, word2 string, word2Lang string) (bool, error) {
	prompt := []*genai.Content{
		genai.NewContentFromParts([]*genai.Part{
			{Text: fmt.Sprintf(
				`Can we say that the word '%s' is a translation of the %s word '%s' in another language?
				 Answer just Yes or No, and nothing else.`, word1, word2Lang, word2)},
		}, genai.RoleUser),
	}

	resp, err := client.Models.GenerateContent(ctx, modelName, prompt, nil)

	if err != nil {
		return false, err
	}

	answer := resp.Candidates[0].Content.Parts[0].Text

	return strings.ToLower(answer) == "yes", nil
}

func checkNotEmpty(res *genai.GenerateContentResponse) {
	if len(res.Candidates) == 0 ||
		len(res.Candidates[0].Content.Parts) == 0 {
		log.Fatalf("empty response from model")
	}
}

func textOf(res *genai.GenerateContentResponse) string {
	checkNotEmpty(res)
	return res.Candidates[0].Content.Parts[0].Text
}
