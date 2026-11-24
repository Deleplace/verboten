# Use the official Golang image to create a build artifact.
# This is a multi-stage build, so we use a temporary image for the build process.
FROM golang:1.25 AS builder

# Set the working directory inside the container.
WORKDIR /app

# Copy the Go module files and download dependencies.
# This is done as a separate step to leverage Docker layer caching.
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code.
COPY . .

# Build the Go application.
# The -o flag specifies the output file name.
# CGO_ENABLED=0 is used to build a statically linked binary.
RUN CGO_ENABLED=0 go build -o /server ./cmd/web

# Use a minimal, secure, non-root base image with up-to-date CA certificates.
FROM gcr.io/distroless/static-debian12:nonroot

# Set the working directory inside the container.
WORKDIR /app

# Copy the self-contained binary (with embedded assets) from the builder stage.
COPY --from=builder /server /app/server

# Set the command to run the application.
CMD ["/app/server"]
