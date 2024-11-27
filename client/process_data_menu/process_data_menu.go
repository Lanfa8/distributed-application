package process_data_menu

import (
	"bufio"
	"distributed-app/schemas"
	"fmt"
	"strconv"
	"strings"
)

func GetDevice(reader *bufio.Reader) (*schemas.Device, error) {
	for {
		fmt.Println("\nChoose a device:")
		fmt.Println("1. CPU")
		fmt.Println("2. GPU")

		device, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			return nil, err
		}

		device = strings.TrimSpace(device)

		switch device {
		case "1":
			res := schemas.CPU
			return &res, nil
		case "2":
			res := schemas.GPU
			return &res, nil
		default:
			fmt.Println("Invalid option. Please try again.")
		}
	}
}

func GetMethod(reader *bufio.Reader, device schemas.Device) (*schemas.Method, error) {
	for {
		fmt.Println("\nChoose a parallelization method:")

		if device == schemas.CPU {
			fmt.Println("1. OpenMP (C)")
			fmt.Println("2. MPI (Go)")
			fmt.Println("3. Goroutines (Go)")
		} else {
			fmt.Println("4. Cuda (CUDA)")
		}

		method, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			return nil, err
		}
		method = strings.TrimSpace(method)

		switch method {
		case "1":
			if device != schemas.CPU {
				fmt.Println("Invalid option. Please try again.")
				continue
			}

			res := schemas.OPENMP
			return &res, nil
		case "2":
			if device != schemas.CPU {
				fmt.Println("Invalid option. Please try again.")
				continue
			}

			res := schemas.MPI
			return &res, nil
		case "3":
			if device != schemas.CPU {
				fmt.Println("Invalid option. Please try again.")
				continue
			}

			res := schemas.GO_ROUTINES
			return &res, nil
		case "4":
			if device != schemas.GPU {
				fmt.Println("Invalid option. Please try again.")
				continue
			}
			res := schemas.CUDA
			return &res, nil
		default:
			fmt.Println("Invalid option. Please try again.")
		}
	}
}

func GetFileName(reader *bufio.Reader) (*string, error) {
	fmt.Println("\nEnter file name:")

	fileName, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return nil, err
	}

	fileName = strings.TrimSpace(fileName)

	return &fileName, nil
}

func GetNumberValue(reader *bufio.Reader) (*int, error) {
	for {
		numberStr, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			return nil, err
		}

		numberStr = strings.TrimSpace(numberStr)

		number, err := strconv.Atoi(numberStr)
		if err != nil {
			fmt.Println("Invalid value. Enter valid number, please")
			continue
		}

		return &number, nil
	}
}

func GetProcessesNumber(reader *bufio.Reader, method schemas.Method) (*int, error) {
	// TODO: validate respecting method type
	fmt.Println("\nEnter the desired number of parallel processes:")

	return GetNumberValue(reader)
}

func GetAttemptsNumber(reader *bufio.Reader) (*int, error) {
	fmt.Println("\nEnter the desired number of attempts:")

	return GetNumberValue(reader)
}
