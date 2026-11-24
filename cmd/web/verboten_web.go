package main

import (
	"context"
	"flag"
	"log/slog"
	"os"

	"github.com/Deleplace/verboten"
)

func main() {
	var verbose bool
	const verboseUsage = "log transcribed words spoken by the human player and the AI guesser"
	flag.BoolVar(&verbose, "verbose", false, verboseUsage)
	flag.BoolVar(&verbose, "v", false, verboseUsage+" (shorthand)")
	recordingsDir := flag.String("recordings-dir", verboten.DefaultRecordingsDirForEnv(), "directory for game recordings (empty disables recording; enabled when RECORD_GAMES is 1, TRUE, or true)")
	flag.Parse()

	logger := verboten.NewLogger(os.Stdout)
	slog.SetDefault(logger)

	ctx := context.Background()

	client, err := verboten.NewGenAIClient(ctx)
	if err != nil {
		logger.Error("failed to create genai client", "err", err)
		os.Exit(1)
	}

	server := verboten.NewServer(client)
	server.Logger = logger
	server.Verbose = verbose
	server.RecordingsDir = *recordingsDir
	if server.RecordingsDir == "" {
		logger.Info("game recording disabled", "hint", "set RECORD_GAMES=1 to enable")
	} else {
		logger.Info("game recording enabled", "dir", server.RecordingsDir)
	}
	if err := server.Start(ctx); err != nil {
		logger.Error("server exited with error", "err", err)
		os.Exit(1)
	}
}
