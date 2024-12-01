package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	resultLimit   = 50
	minValidCols  = 10
	dateLayout    = "2006-01-02 15:04:05.000000"
)

type Record struct {
	Device      string
	Timestamp   time.Time
	Ruido       *float64
	Eco2        *float64
	Etvoc       *float64
}

type Interval struct {
	Field    string
	Device    string
	Value     float64
	StartTime time.Time
	EndTime   time.Time
	Duration  time.Duration
}

func main() {
	csvPath := flag.String("csv", "", "Path to the CSV file")
	processes := flag.Int("processes", 1, "Number of worker processes")
	flag.Parse()

	if *csvPath == "" {
		log.Fatalf("Usage: %s -csv <csv_path> -processes <num_workers>", os.Args[0])
	}

	numWorkers := *processes
	log.Printf("Number of workers: %d", numWorkers)
	startTime := time.Now()
	records, err := readAndParseCSV(*csvPath)
	if err != nil {
		log.Fatalf("Error reading CSV: %v", err)
	}
	log.Printf("CSV parsed in %v, total records: %d", time.Since(startTime), len(records))

	groupStart := time.Now()
	deviceData := groupByDevice(records)
	log.Printf("Data grouped by device in %v", time.Since(groupStart))

	fields := []string{"etvoc", "eco2", "ruido"}

	processStart := time.Now()
	results := make(map[string][]Interval)

	for _, field := range fields {
		intervals := processField(deviceData, field, numWorkers)
		results[field] = intervals
	}
	log.Printf("Processing completed in %v", time.Since(processStart))

	for _, field := range fields {
		displayResults(field, results[field])
	}
	log.Printf("Total execution time: %v", time.Since(startTime))
}

func readAndParseCSV(path string) ([]Record, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var records []Record
	scanner := bufio.NewScanner(file)
	headerParsed := false
	for scanner.Scan() {
		line := scanner.Text()
		if !headerParsed {
			headerParsed = true
			continue
		}
		record, err := parseLine(line)
		if err == nil {
			records = append(records, record)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return records, nil
}

func parseLine(line string) (Record, error) {
	fields := strings.Split(line, "|")
	if len(fields) < minValidCols {
		return Record{}, fmt.Errorf("insufficient columns")
	}

	device := fields[1]
	if device == "" {
		return Record{}, fmt.Errorf("empty device")
	}

	timestamp, err := time.Parse(dateLayout, fields[3])
	if err != nil {
		return Record{}, fmt.Errorf("invalid timestamp")
	}

	ruido, err := parseFloatField(fields[7])
	eco2, err2 := parseFloatField(fields[8])
	etvoc, err3 := parseFloatField(fields[9])

	if err != nil && err2 != nil && err3 != nil {
		return Record{}, fmt.Errorf("all target fields invalid")
	}

	return Record{
		Device:    device,
		Timestamp: timestamp,
		Ruido:     ruido,
		Eco2:      eco2,
		Etvoc:     etvoc,
	}, nil
}

func parseFloatField(s string) (*float64, error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "nan") {
		return nil, fmt.Errorf("invalid float")
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func groupByDevice(records []Record) map[string][]Record {
	deviceData := make(map[string][]Record)
	for _, record := range records {
		deviceData[record.Device] = append(deviceData[record.Device], record)
	}
	return deviceData
}

func processField(deviceData map[string][]Record, field string, numWorkers int) []Interval {
	var intervals []Interval
	var mutex sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan []Record, len(deviceData))

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for records := range jobs {
				deviceIntervals := findIntervals(records, field)
				mutex.Lock()
				intervals = append(intervals, deviceIntervals...)
				mutex.Unlock()
			}
		}()
	}

	for _, records := range deviceData {
		jobs <- records
	}
	close(jobs)
	wg.Wait()

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i].Duration > intervals[j].Duration
	})
	if len(intervals) > resultLimit {
		intervals = intervals[:resultLimit]
	}

	for i := range intervals {
		intervals[i].Field = field
	}
	return intervals
}

func findIntervals(records []Record, field string) []Interval {
	var intervals []Interval
	if len(records) == 0 {
		return intervals
	}

	var lastValue *float64
	var startTime time.Time
	var endTime time.Time

	for _, record := range records {

		var currentValue *float64
		switch field {
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
					Device: records[0].Device,
					Value:    *lastValue,
					StartTime:    startTime,
					EndTime:      endTime,
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
					Device: records[0].Device,
					Value:    *lastValue,
					StartTime:    startTime,
					EndTime:      endTime,
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
			Device: records[0].Device,
			Value:    *lastValue,
			StartTime:    startTime,
			EndTime:      endTime,
			Duration: duration,
		})
	}

	return intervals
}

func getFieldPointer(record Record, field string) *float64 {
	switch field {
	case "etvoc":
		return record.Etvoc
	case "eco2":
		return record.Eco2
	case "ruido":
		return record.Ruido
	default:
		return nil
	}
}

func sortAndLimitIntervals(intervals []Interval, limit int) []Interval {
	for i := 0; i < len(intervals); i++ {
		maxIdx := i
		for j := i + 1; j < len(intervals); j++ {
			if intervals[j].Duration > intervals[maxIdx].Duration {
				maxIdx = j
			}
		}
		intervals[i], intervals[maxIdx] = intervals[maxIdx], intervals[i]
		if i == limit-1 {
			break
		}
	}
	if len(intervals) > limit {
		intervals = intervals[:limit]
	}
	return intervals
}

func displayResults(field string, intervals []Interval) {
	fmt.Printf("\nTop %d intervals for %s:\n", resultLimit, strings.ToUpper(field))
	for _, interval := range intervals {
		fmt.Printf("Device: %s, Value: %.2f, Start: %s, End: %s, Duration: %v\n",
			interval.Device, interval.Value, interval.StartTime.Format(dateLayout),
			interval.EndTime.Format(dateLayout), interval.Duration)
	}
}