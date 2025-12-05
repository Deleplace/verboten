package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"google.golang.org/genai"

	"github.com/Deleplace/verboten"
)

func main() {
	flag.Parse()
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
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
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
	fmt.Println()

	server := verboten.NewServer(client)
	err = server.Start(ctx)
	if err != nil {
		log.Fatal(err)
	}
}
