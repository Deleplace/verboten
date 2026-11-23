package main

import (
	"context"
	"flag"
	"log"

	"github.com/Deleplace/verboten"
)

func main() {
	var verbose bool
	const verboseUsage = "log transcribed words spoken by the human player and the AI guesser"
	flag.BoolVar(&verbose, "verbose", false, verboseUsage)
	flag.BoolVar(&verbose, "v", false, verboseUsage+" (shorthand)")
	recordingsDir := flag.String("recordings-dir", verboten.DefaultRecordingsDirForEnv(), "directory for game recordings (empty disables recording; enabled when RECORD_GAMES is 1, TRUE, or true)")
	flag.Parse()

	ctx := context.Background()

	client, err := verboten.NewGenAIClient(ctx)
	if err != nil {
		log.Fatal(err)
	}

	server := verboten.NewServer(client)
	server.Verbose = verbose
	server.RecordingsDir = *recordingsDir
	if server.RecordingsDir == "" {
		log.Printf("Game recording disabled (set RECORD_GAMES=1 to enable)")
	} else {
		log.Printf("Game recording enabled (dir=%q)", server.RecordingsDir)
	}
	if err := server.Start(ctx); err != nil {
		log.Fatal(err)
	}
}
