package main

import (
	"bufio"
	"distributed-app/client/process_data_menu"
	"distributed-app/schemas"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"time"
)

const (
	BUFFER_SIZE    = 4096 // 4KB chunks
	SERVER_ADDRESS = "localhost:8080"
)

type Client struct {
	conn net.Conn
}

func NewClient() (*Client, error) {
	conn, err := net.Dial("tcp", SERVER_ADDRESS)
	if err != nil {
		return nil, fmt.Errorf("error connecting to server: %v", err)
	}

	return &Client{conn: conn}, nil
}

func (c *Client) sendAction(action interface{}) error {
	jsonData, err := json.Marshal(action)
	if err != nil {
		return fmt.Errorf("error marshaling: %v", err)
	}

	// Send the size of the JSON message first (4 bytes)
	messageSizeBuffer := make([]byte, 4)
	binary.LittleEndian.PutUint32(messageSizeBuffer, uint32(len(jsonData)))
	_, err = c.conn.Write(messageSizeBuffer)
	if err != nil {
		return fmt.Errorf("error sending message size: %v", err)
	}

	// Send the JSON message
	_, err = c.conn.Write(jsonData)
	if err != nil {
		return fmt.Errorf("error sending action json: %v", err)
	}

	return nil
}

func (c *Client) sendFileAction() error {
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Enter the file path: ")
	cfilepath, err := reader.ReadString('\n')
	if err != nil {
		return fmt.Errorf("error reading input: %v", err)
	}

	cfilepath = strings.TrimSpace(cfilepath)

	file, err := os.Open(cfilepath)
	if err != nil {
		return fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("error getting file info: %v", err)
	}
	fileSize := fileInfo.Size()

	sizeBuffer := make([]byte, 8)
	for i := 0; i < 8; i++ {
		sizeBuffer[i] = byte(fileSize >> (i * 8))
	}

	action := schemas.SendFileAction{
		Action:   schemas.SEND_FILE_ACTION,
		FileName: fileInfo.Name(),
		Filesize: sizeBuffer,
	}

	err = c.sendAction(action)
	if err != nil {
		return err
	}

	fmt.Printf("Sending file %s (%.2f MB)...\n", action.FileName, float64(fileSize)/1024/1024)

	err = c.sendFileInChunks(file, fileSize)
	if err != nil {
		return fmt.Errorf("error during file transfer: %v", err)
	}

	return nil
}

func (c *Client) sendFileInChunks(file *os.File, fileSize int64) error {
	buffer := make([]byte, BUFFER_SIZE)
	totalBytesSent := int64(0)
	startTime := time.Now()

	for {
		n, err := file.Read(buffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading file: %v", err)
		}

		_, err = c.conn.Write(buffer[:n])
		if err != nil {
			return fmt.Errorf("error sending chunk: %v", err)
		}

		totalBytesSent += int64(n)

		progress := float64(totalBytesSent) / float64(fileSize) * 100
		speed := float64(totalBytesSent) / time.Since(startTime).Seconds() / 1024 // KB/s

		fmt.Printf("\rProgress: %.2f%% (%.2f KB/s)", progress, speed)
	}

	fmt.Println("\nFile transfer completed!")
	return nil
}

func (c *Client) processDataMenu() error {
	reader := bufio.NewReader(os.Stdin)

	device, err := process_data_menu.GetDevice(reader)
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return err
	}

	method, err := process_data_menu.GetMethod(reader, *device)
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return err
	}

	fileName, err := process_data_menu.GetFileName(reader)
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return err
	}

	processes, err := process_data_menu.GetProcessesNumber(reader, *method)
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return err
	}

	// attempts, err := process_data_menu.GetAttemptsNumber(reader)
	// if err != nil {
	// 	fmt.Printf("Error reading input: %v\n", err)
	// 	return err
	// }

	processAction := schemas.ProcessAction{
		Action:    schemas.PROCESS_ACTION,
		FileName:  *fileName,
		Method:    *method,
		Attempts:  1,
		Processes: *processes,
	}

	fmt.Printf("Processing...\n")
	c.sendAction(processAction)

	responseBuffer := make([]byte, 64768)
	n, err := c.conn.Read(responseBuffer)
	if err != nil {
		log.Printf("Error reading response: %v\n", err)
		return err
	}

	fmt.Printf("End processing\n")
	fmt.Print(string(responseBuffer[:n]))

	return nil
}

func (c *Client) Close() {
	c.conn.Close()
}

func main() {
	client, err := NewClient()
	if err != nil {
		fmt.Println("Error creating client:", err)
		os.Exit(1)
	}
	defer client.Close()

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\nChoose an action:")
		fmt.Println("1. Send file")
		fmt.Println("2. Process data")
		fmt.Println("5. Exit")

		action, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			return
		}

		action = strings.TrimSpace(action)

		switch action {
		case "1":
			err = client.sendFileAction()
			if err != nil {
				fmt.Printf("Error sending file: %v\n", err)
			}

		case "2":
			err = client.processDataMenu()
			if err != nil {
				fmt.Printf("Error sending other action: %v\n", err)
			}
		case "5":
			return
		default:
			fmt.Println("Invalid option. Please try again.")
		}
	}
}
