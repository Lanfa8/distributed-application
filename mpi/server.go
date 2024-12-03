package main

import (
	"bytes"
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"sort"
	"strconv"
	"time"

	mpi "github.com/sbromberger/gompi"
)

const (
	c_id           = 0
	c_device       = 1
	c_contagem     = 2
	c_data         = 3
	c_temperatura  = 4
	c_umidade      = 5
	c_luminosidade = 6
	c_ruido        = 7
	c_eco2         = 8
	c_etvoc        = 9
	c_latitude     = 10
	c_longitude    = 11
)

type Interval struct {
	DeviceID string
	Value    float64
	Start    time.Time
	End      time.Time
	Duration time.Duration
}


type ByDuration []Interval

func (a ByDuration) Len() int           { return len(a) }
func (a ByDuration) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByDuration) Less(i, j int) bool { return a[i].Duration > a[j].Duration } // Descending


type Record struct {
	Timestamp time.Time
	Etvoc     *float64
	Eco2      *float64
	Ruido     *float64
}

func main() {
	runWithThreads()
	runWithSingleThread()
}

func runWithThreads() {
	mpi.Start(false)
	mpiCommunicator := mpi.NewCommunicator(nil)
	defer mpi.Stop()

	numProcs := mpi.WorldSize()
	rank := mpi.WorldRank()

	startTime := mpi.WorldTime()

	if rank == 0 {
		fmt.Printf("Número de threads utilizados: %d\n", numProcs)
	}

	var csvPath string
	flag.StringVar(&csvPath, "csv", "", "Path to the CSV file")
	flag.Parse()
	if csvPath == "" {
		if rank == 0 {
			fmt.Println("Usage: mpirun -np <num_procs> go run main.go -csv <path_to_csv>")
		}
		return
	}

	file, err := os.Open(csvPath)
	if err != nil {
		if rank != 0 {
			fmt.Println("Error opening CSV file:", err)
		}
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '|'
	reader.FieldsPerRecord = -1

	_, err = reader.Read()
	if err != nil {
		if rank == 0 {
			fmt.Println("Error reading CSV header:", err)
		}
		return
	}

	deviceData := make(map[string][]Record)

	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			continue 
		}

		deviceID := line[c_device]
		dataStr := line[c_data]

		if deviceID == "" || dataStr == "" || (line[c_etvoc] == "" && line[c_eco2] == "" && line[c_ruido] == "") {
			continue
		}

		h := hash(deviceID)
		if h%uint32(numProcs) != uint32(rank) {
			continue
		}

		layout := "2006-01-02 15:04:05.999999"
		timestamp, err := time.Parse(layout, dataStr)
		if err != nil {
			continue
		}

		var etvoc *float64
		var eco2 *float64
		var ruido *float64

		etvocStr := line[c_etvoc]
		if etvocStr != "" {
			v, err := strconv.ParseFloat(etvocStr, 64)
			if err == nil {
				etvoc = &v
			}
		}

		eco2Str := line[c_eco2]
		if eco2Str != "" {
			v, err := strconv.ParseFloat(eco2Str, 64)
			if err == nil {
				eco2 = &v
			}
		}

		ruidoStr := line[c_ruido]
		if ruidoStr != "" {
			v, err := strconv.ParseFloat(ruidoStr, 64)
			if err == nil {
				ruido = &v
			}
		}

		record := Record{
			Timestamp: timestamp,
			Etvoc:     etvoc,
			Eco2:      eco2,
			Ruido:     ruido,
		}
		deviceData[deviceID] = append(deviceData[deviceID], record)
	}

	csvReadEndTime := mpi.WorldTime()
	if rank == 0 {
		fmt.Printf("Tempo de leitura do CSV: %.4f segundos\n", csvReadEndTime-startTime)
	}

	var intervalsEtvoc []Interval
	var intervalsEco2 []Interval
	var intervalsRuido []Interval

	for deviceID, records := range deviceData {
		intervals := findIntervals(records, deviceID, "etvoc")
		intervalsEtvoc = append(intervalsEtvoc, intervals...)

		intervals = findIntervals(records, deviceID, "eco2")
		intervalsEco2 = append(intervalsEco2, intervals...)

		intervals = findIntervals(records, deviceID, "ruido")
		intervalsRuido = append(intervalsRuido, intervals...)
	}

	intervalsEtvoc = topNIntervals(intervalsEtvoc, 50)
	intervalsEco2 = topNIntervals(intervalsEco2, 50)
	intervalsRuido = topNIntervals(intervalsRuido, 50)


	if rank != 0 {
		sendIntervals(intervalsEtvoc, 0, 100+rank*3+1)
		sendIntervals(intervalsEco2, 0, 100+rank*3+2)
		sendIntervals(intervalsRuido, 0, 100+rank*3+3)
	} else {
		allIntervalsEtvoc := intervalsEtvoc
		allIntervalsEco2 := intervalsEco2
		allIntervalsRuido := intervalsRuido

		for proc := 1; proc < numProcs; proc++ {
			received := receiveIntervals(proc, 100+proc*3+1)
			allIntervalsEtvoc = append(allIntervalsEtvoc, received...)

			received = receiveIntervals(proc, 100+proc*3+2)
			allIntervalsEco2 = append(allIntervalsEco2, received...)

			received = receiveIntervals(proc, 100+proc*3+3)
			allIntervalsRuido = append(allIntervalsRuido, received...)
		}

		// Keep top 50 intervals
		allIntervalsEtvoc = topNIntervals(allIntervalsEtvoc, 50)
		allIntervalsEco2 = topNIntervals(allIntervalsEco2, 50)
		allIntervalsRuido = topNIntervals(allIntervalsRuido, 50)

		// Output results
		fmt.Printf("Resultados para etvoc:\n")
		printIntervals(allIntervalsEtvoc)
		fmt.Printf("\nResultados para eco2:\n")
		printIntervals(allIntervalsEco2)
		fmt.Printf("\nResultados para ruido:\n")
		printIntervals(allIntervalsRuido)
	}

	mpiCommunicator.Barrier()
	endTime := mpi.WorldTime()

	
	if rank == 0 {
		fmt.Printf("Tempo de processamento com %d threads: %.4f segundos\n", numProcs, endTime-csvReadEndTime)
		fmt.Printf("\nTempo total com %d threads: %.4f segundos\n", numProcs, endTime-startTime)
	}
}

func runWithSingleThread() {
	mpi.Start(false)
	mpiCommunicator := mpi.NewCommunicator(nil)
	defer mpi.Stop()

	numProcs := 1
	rank := mpi.WorldRank()

	startTime := mpi.WorldTime()

	if rank == 0 {
		fmt.Printf("Número de threads utilizados: %d\n", numProcs)
	}

	var csvPath string
	flag.StringVar(&csvPath, "csv", "", "Path to the CSV file")
	flag.Parse()
	if csvPath == "" {
		if rank == 0 {
			fmt.Println("Usage: mpirun -np <num_procs> go run main.go -csv <path_to_csv>")
		}
		return
	}

	file, err := os.Open(csvPath)
	if err != nil {
		if rank != 0 {
			fmt.Println("Error opening CSV file:", err)
		}
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '|'
	reader.FieldsPerRecord = -1

	_, err = reader.Read()
	if err != nil {
		if rank == 0 {
			fmt.Println("Error reading CSV header:", err)
		}
		return
	}

	deviceData := make(map[string][]Record)

	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			continue 
		}

		deviceID := line[c_device]
		dataStr := line[c_data]

		if deviceID == "" || dataStr == "" || (line[c_etvoc] == "" && line[c_eco2] == "" && line[c_ruido] == "") {
			continue
		}

		h := hash(deviceID)
		if h%uint32(numProcs) != uint32(rank) {
			continue
		}

		layout := "2006-01-02 15:04:05.999999"
		timestamp, err := time.Parse(layout, dataStr)
		if err != nil {
			continue
		}

		var etvoc *float64
		var eco2 *float64
		var ruido *float64

		etvocStr := line[c_etvoc]
		if etvocStr != "" {
			v, err := strconv.ParseFloat(etvocStr, 64)
			if err == nil {
				etvoc = &v
			}
		}

		eco2Str := line[c_eco2]
		if eco2Str != "" {
			v, err := strconv.ParseFloat(eco2Str, 64)
			if err == nil {
				eco2 = &v
			}
		}

		ruidoStr := line[c_ruido]
		if ruidoStr != "" {
			v, err := strconv.ParseFloat(ruidoStr, 64)
			if err == nil {
				ruido = &v
			}
		}

		record := Record{
			Timestamp: timestamp,
			Etvoc:     etvoc,
			Eco2:      eco2,
			Ruido:     ruido,
		}
		deviceData[deviceID] = append(deviceData[deviceID], record)
	}

	csvReadEndTime := mpi.WorldTime()
	if rank == 0 {
		fmt.Printf("Tempo de leitura do CSV: %.4f segundos\n", csvReadEndTime-startTime)
	}

	var intervalsEtvoc []Interval
	var intervalsEco2 []Interval
	var intervalsRuido []Interval

	for deviceID, records := range deviceData {
		intervals := findIntervals(records, deviceID, "etvoc")
		intervalsEtvoc = append(intervalsEtvoc, intervals...)

		intervals = findIntervals(records, deviceID, "eco2")
		intervalsEco2 = append(intervalsEco2, intervals...)

		intervals = findIntervals(records, deviceID, "ruido")
		intervalsRuido = append(intervalsRuido, intervals...)
	}

	intervalsEtvoc = topNIntervals(intervalsEtvoc, 50)
	intervalsEco2 = topNIntervals(intervalsEco2, 50)
	intervalsRuido = topNIntervals(intervalsRuido, 50)


	if rank != 0 {
		sendIntervals(intervalsEtvoc, 0, 100+rank*3+1)
		sendIntervals(intervalsEco2, 0, 100+rank*3+2)
		sendIntervals(intervalsRuido, 0, 100+rank*3+3)
	} else {
		allIntervalsEtvoc := intervalsEtvoc
		allIntervalsEco2 := intervalsEco2
		allIntervalsRuido := intervalsRuido

		for proc := 1; proc < numProcs; proc++ {
			received := receiveIntervals(proc, 100+proc*3+1)
			allIntervalsEtvoc = append(allIntervalsEtvoc, received...)

			received = receiveIntervals(proc, 100+proc*3+2)
			allIntervalsEco2 = append(allIntervalsEco2, received...)

			received = receiveIntervals(proc, 100+proc*3+3)
			allIntervalsRuido = append(allIntervalsRuido, received...)
		}

		// Keep top 50 intervals
		allIntervalsEtvoc = topNIntervals(allIntervalsEtvoc, 50)
		allIntervalsEco2 = topNIntervals(allIntervalsEco2, 50)
		allIntervalsRuido = topNIntervals(allIntervalsRuido, 50)

		// Output results
		fmt.Printf("Resultados para etvoc:\n")
		printIntervals(allIntervalsEtvoc)
		fmt.Printf("\nResultados para eco2:\n")
		printIntervals(allIntervalsEco2)
		fmt.Printf("\nResultados para ruido:\n")
		printIntervals(allIntervalsRuido)
	}

	mpiCommunicator.Barrier()
	endTime := mpi.WorldTime()

	if rank == 0 {
		fmt.Printf("Tempo de processamento com 1 thread: %.4f segundos\n", endTime-csvReadEndTime)
		fmt.Printf("\nTempo total com 1 thread: %.4f segundos\n", endTime-startTime)
	}

	if rank == 0 {
		// Calculate and print speedup
		speedup := (endTime - startTime) / (endTime - csvReadEndTime)
		fmt.Printf("Speedup: %.4f\n", speedup)
	}
}

func hash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}


func findIntervals(records []Record, deviceID, variable string) []Interval {
	var intervals []Interval

	var lastValue *float64
	var startTime time.Time
	var endTime time.Time

	for _, record := range records {

		var currentValue *float64
		switch variable {
		case "etvoc":
			currentValue = record.Etvoc
		case "eco2":
			currentValue = record.Eco2
		case "ruido":
			currentValue = record.Ruido
		}

		if currentValue == nil {
			if lastValue != nil {
				duration := endTime.Sub(startTime)
				intervals = append(intervals, Interval{
					DeviceID: deviceID,
					Value:    *lastValue,
					Start:    startTime,
					End:      endTime,
					Duration: duration,
				})
				lastValue = nil
			}
			continue
		}

		if lastValue == nil || *currentValue != *lastValue {
			if lastValue != nil {
				duration := endTime.Sub(startTime)
				intervals = append(intervals, Interval{
					DeviceID: deviceID,
					Value:    *lastValue,
					Start:    startTime,
					End:      endTime,
					Duration: duration,
				})
			}
			lastValue = currentValue
			startTime = record.Timestamp
			endTime = record.Timestamp
		} else {
			endTime = record.Timestamp
		}
	}

	if lastValue != nil {
		duration := endTime.Sub(startTime)
		intervals = append(intervals, Interval{
			DeviceID: deviceID,
			Value:    *lastValue,
			Start:    startTime,
			End:      endTime,
			Duration: duration,
		})
	}

	return intervals
}


func topNIntervals(intervals []Interval, N int) []Interval {
	sort.Sort(ByDuration(intervals))
	if len(intervals) > N {
		intervals = intervals[:N]
	}
	return intervals
}

func sendIntervals(intervals []Interval, toID, tag int) {
	data, err := encodeIntervals(intervals)
	if err != nil {
		fmt.Println("Error encoding intervals:", err)
		return
	}
	comm := mpi.NewCommunicator(nil)
	comm.SendBytes(data, toID, tag)
}

func receiveIntervals(fromID, tag int) []Interval {
	comm := mpi.NewCommunicator(nil)
	data, _ := comm.RecvBytes(fromID, tag)
	intervals, err := decodeIntervals(data)
	if err != nil {
		fmt.Println("Error decoding intervals:", err)
	}
	return intervals
}

func encodeIntervals(intervals []Interval) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(intervals)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeIntervals(data []byte) ([]Interval, error) {
	var intervals []Interval
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	err := dec.Decode(&intervals)
	if err != nil {
		return nil, err
	}
	return intervals, nil
}


func printIntervals(intervals []Interval) {
	for _, interval := range intervals {
		fmt.Printf("Dispositivo: %s, Valor: %.2f, Início: %s, Fim: %s, Duração: %s\n",
			interval.DeviceID, interval.Value, interval.Start.Format("2006-01-02 15:04:05"),
			interval.End.Format("2006-01-02 15:04:05"), interval.Duration)
	}
}