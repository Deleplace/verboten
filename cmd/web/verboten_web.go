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
	recordingsDir := flag.String("recordings-dir", verboten.DefaultRecordingsDirForEnv(), "directory for game recordings (empty disables recording; disabled by default on Cloud Run)")
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
		log.Printf("Game recording disabled (set VERBOTEN_RECORDINGS_DIR or -recordings-dir to enable)")
	}
	if err := server.Start(ctx); err != nil {
		log.Fatal(err)
	}
}
