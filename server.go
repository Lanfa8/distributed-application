package main

import (
	"bytes"
	"distributed-app/schemas"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

const (
	BUFFER_SIZE     = 4096
	DATA_DIRECTORY  = "data"
	FILENAME_LENGTH = 256
)

const (
	CMD_SEND_FILE byte = 1
	CMD_CLOSE     byte = 2
)

func receiveFile(conn net.Conn, sendFileAction schemas.SendFileAction) error {
	sizeBuffer := sendFileAction.Filesize

	var fileSize int64
	for i := 0; i < 8; i++ {
		fileSize |= int64(sizeBuffer[i]) << (i * 8)
	}

	if err := os.MkdirAll(DATA_DIRECTORY, 0755); err != nil {
		return fmt.Errorf("error creating data directory: %v", err)
	}

	outputPath := filepath.Join(DATA_DIRECTORY, sendFileAction.FileName)
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating output file: %v", err)
	}
	defer outputFile.Close()

	buffer := make([]byte, BUFFER_SIZE)
	totalBytesReceived := int64(0)
	startTime := time.Now()

	for {
		n, err := conn.Read(buffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error receiving data: %v", err)
		}

		_, err = outputFile.Write(buffer[:n])
		if err != nil {
			return fmt.Errorf("error writing to file: %v", err)
		}

		totalBytesReceived += int64(n)

		progress := float64(totalBytesReceived) / float64(fileSize) * 100
		speed := float64(totalBytesReceived) / time.Since(startTime).Seconds() / 1024 // KB/s

		fmt.Printf("\rProgress: %.2f%% (%.2f KB/s)", progress, speed)

		if totalBytesReceived >= fileSize {
			break
		}
	}

	fmt.Printf("\nFile received successfully: %s\n", outputPath)
	return nil
}

func runBashCommand(command string) (string, error) {
	cmd := exec.Command("bash", "-c", command)

	var output bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("error executing command: %v\nCommand: %s\nStdout: %s\nStderr: %s",
			err, command, output.String(), stderr.String())
	}

	return output.String(), nil
}

func handleOpenMPMethod(schema schemas.ProcessAction) (string, error) {
	command := fmt.Sprintf("./openmp/openmp ./data/%s %d", schema.FileName, schema.Processes)

	fmt.Printf("command: %s\n", command)

	return runBashCommand(command)
}

func handlePythonMethod(schema schemas.ProcessAction) (string, error) {
	command := fmt.Sprintf("python3 ./multiprocessing_python/server.py --processes %d --csv_path ./data/%s", schema.Processes, schema.FileName)

	fmt.Printf("command: %s\n", command)

	return runBashCommand(command)
}

func handleMPIMethod(schema schemas.ProcessAction) (string, error) {
	command := fmt.Sprintf("mpirun -np %d go run ./mpi/server.go -csv ./data/%s", schema.Processes, schema.FileName)

	fmt.Printf("command: %s\n", command)

	return runBashCommand(command)
}

func handleGoroutinesMethod(schema schemas.ProcessAction) (string, error) {
	command := fmt.Sprintf("go run ./goroutines/server.go -processes %d -csv ./data/%s", schema.Processes, schema.FileName)

	fmt.Printf("command: %s\n", command)

	return runBashCommand(command)
}

func handleConnection(conn net.Conn) {
	defer conn.Close()
	fmt.Println("New connection from:", conn.RemoteAddr())

	for {
		sizeBuf := make([]byte, 4)
		_, err := io.ReadFull(conn, sizeBuf)
		if err != nil {
			if err == io.EOF {
				fmt.Println("Client disconnected")
				return
			}
			fmt.Printf("Error reading message size: %v\n", err)
			return
		}

		messageSize := binary.LittleEndian.Uint32(sizeBuf)
		fmt.Printf("Received message size: %d bytes\n", messageSize)

		messageBuffer := make([]byte, messageSize)
		if _, err := io.ReadFull(conn, messageBuffer); err != nil {
			fmt.Printf("Error reading message: %v\n", err)
			return
		}

		// First, unmarshal to get the action type
		var baseAction schemas.Action
		if err := json.Unmarshal(messageBuffer, &baseAction); err != nil {
			fmt.Printf("Error parsing action: %v\n", err)
			return
		}

		switch baseAction.Action {
		case schemas.SEND_FILE_ACTION:
			var sendFileAction schemas.SendFileAction
			if err := json.Unmarshal(messageBuffer, &sendFileAction); err != nil {
				fmt.Printf("Error parsing file data: %v\n", err)
				return
			}
			err := receiveFile(conn, sendFileAction)
			if err != nil {
				fmt.Printf("Error receiving file: %v\n", err)
			}
		case schemas.PROCESS_ACTION:
			var processAction schemas.ProcessAction
			if err := json.Unmarshal(messageBuffer, &processAction); err != nil {
				fmt.Printf("Error parsing action data: %v\n", err)
				return
			}

			switch processAction.Method {
			case schemas.OPENMP:
				out, err := handleOpenMPMethod(processAction)
				if err != nil {
					fmt.Printf("Error processing action: %v\n", err)
				}
				conn.Write([]byte(out))
			case schemas.GO_ROUTINES:
				out, err := handleGoroutinesMethod(processAction)
				if err != nil {
					fmt.Printf("Error processing action: %v\n", err)
				}
				conn.Write([]byte(out))
			case schemas.MPI:
				out, err := handleMPIMethod(processAction)
				if err != nil {
					fmt.Printf("Error processing action: %v\n", err)
				}
				conn.Write([]byte(out))
			case schemas.PYTHON_MULTIPROCESSING:
				out, err := handlePythonMethod(processAction)
				if err != nil {
					fmt.Printf("Error processing action: %v\n", err)
				}
				conn.Write([]byte(out))
			default:
				fmt.Printf("Not allowed method \n")
				conn.Write([]byte("Not allowed method \n"))
			}

		default:
			fmt.Printf("Unknown action type: %s\n", baseAction.Action)
		}
	}
}

func main() {
	// Start listening on port 8080
	listen, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println("Error starting server:", err)
		os.Exit(1)
	}
	defer listen.Close()

	fmt.Println("Server listening on port 8080...")

	for {
		conn, err := listen.Accept()
		if err != nil {
			fmt.Println("Error accepting connection:", err)
			continue
		}

		go handleConnection(conn)
	}
}
