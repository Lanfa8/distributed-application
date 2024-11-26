#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
#define MAX_DEVICE_NAME 64
#define MAX_TIMESTAMP 32
#define TOP_N 50

// Estrutura alinhada para melhor performance em CUDA
typedef struct __align__(8) {
    char device[MAX_DEVICE_NAME];
    float value;
    char start_time[MAX_TIMESTAMP];
    char end_time[MAX_TIMESTAMP];
    long duration_seconds;
} StuckInterval;

// Estrutura alinhada para melhor performance em CUDA
typedef struct __align__(8) {
    char device[MAX_DEVICE_NAME];
    char timestamp[MAX_TIMESTAMP];
    float etvoc;
    float eco2;
    float noise;
} SensorReading;

__device__ bool deviceStringCompare(const char* str1, const char* str2, int maxLen) {
    for (int i = 0; i < maxLen; i++) {
        if (str1[i] != str2[i]) return false;
        if (str1[i] == '\0') return true;
    }
    return true;
}

__device__ void deviceStringCopy(char* dest, const char* src, int maxLen) {
    for (int i = 0; i < maxLen - 1; i++) {
        dest[i] = src[i];
        if (src[i] == '\0') break;
    }
    dest[maxLen - 1] = '\0';
}

__global__ void findStuckIntervals(const SensorReading* d_readings, 
                                 int n_readings,
                                 StuckInterval* d_stuck_intervals,
                                 int* d_stuck_count,
                                 int sensor_type,
                                 int max_intervals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_readings - 1) return;

    float current_value, next_value;
    
    switch(sensor_type) {
        case 0: 
            current_value = d_readings[tid].etvoc;
            next_value = d_readings[tid + 1].etvoc;
            break;
        case 1:
            current_value = d_readings[tid].eco2;
            next_value = d_readings[tid + 1].eco2;
            break;
        case 2:
            current_value = d_readings[tid].noise;
            next_value = d_readings[tid + 1].noise;
            break;
        default:
            return;
    }

    if (current_value == next_value && 
        deviceStringCompare(d_readings[tid].device, 
                          d_readings[tid + 1].device, 
                          MAX_DEVICE_NAME)) {
        
        int interval_start = tid;
        int interval_end = tid + 1;
        
        while (interval_end < n_readings) {
            float end_value;
            switch(sensor_type) {
                case 0: end_value = d_readings[interval_end].etvoc; break;
                case 1: end_value = d_readings[interval_end].eco2; break;
                case 2: end_value = d_readings[interval_end].noise; break;
                default: return;
            }
            
            if (end_value != current_value || 
                !deviceStringCompare(d_readings[interval_start].device,
                                   d_readings[interval_end].device,
                                   MAX_DEVICE_NAME)) {
                break;
            }
            interval_end++;
        }
        
        int duration = interval_end - interval_start;
        if (duration > 1) {
            int idx = atomicAdd(d_stuck_count, 1);
            if (idx < max_intervals) {
                deviceStringCopy(d_stuck_intervals[idx].device,
                               d_readings[interval_start].device,
                               MAX_DEVICE_NAME);
                d_stuck_intervals[idx].value = current_value;
                deviceStringCopy(d_stuck_intervals[idx].start_time,
                               d_readings[interval_start].timestamp,
                               MAX_TIMESTAMP);
                deviceStringCopy(d_stuck_intervals[idx].end_time,
                               d_readings[interval_end-1].timestamp,
                               MAX_TIMESTAMP);
                d_stuck_intervals[idx].duration_seconds = duration;
            }
        }
    }
}

void handleCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

void checkLastCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

time_t parseTimestamp(const char* timestamp) {
    struct tm tm = {0};
    sscanf(timestamp, "%d-%d-%d %d:%d:%d",
           &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
           &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return mktime(&tm);
}

int compareIntervals(const void* a, const void* b) {
    const StuckInterval* interval1 = (const StuckInterval*)a;
    const StuckInterval* interval2 = (const StuckInterval*)b;
  
    if (interval1->duration_seconds < interval2->duration_seconds) return 1;
    if (interval1->duration_seconds > interval2->duration_seconds) return -1;
    return 0;
}

void remove_newlines(char* str) {
    char* src = str;
    char* dst = str;
    while (*src) {
        if (*src != '\n') {
            *dst = *src;
            dst++;
        }
        src++;
    }
    *dst = '\0';
}

void analyzeStuckReadings(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Skip header
    char line[MAX_LINE_LENGTH];
    if (!fgets(line, MAX_LINE_LENGTH, file)) {
        perror("Error reading header");
        fclose(file);
        return;
    }

    // Count number of lines first
    int n_readings = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        n_readings++;
    }
    rewind(file);
    fgets(line, MAX_LINE_LENGTH, file); // Skip header again

    // Allocate host memory
    SensorReading* h_readings = (SensorReading*)calloc(n_readings, sizeof(SensorReading));
    if (!h_readings) {
        perror("Failed to allocate host memory");
        fclose(file);
        return;
    }

    // Read data
    int reading_idx = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) && reading_idx < n_readings) {
        char* token = strtok(line, "|");
        int field = 0;
        
        while (token && field < 10) {
            switch(field) {
                case 1: // device
                    strncpy(h_readings[reading_idx].device, token, MAX_DEVICE_NAME-1);
                    break;
                case 3: // timestamp
                    strncpy(h_readings[reading_idx].timestamp, token, MAX_TIMESTAMP-1);
                    break;
                case 8: // eco2
                    h_readings[reading_idx].eco2 = atof(token);
                    break;
                case 9: // etvoc
                    h_readings[reading_idx].etvoc = atof(token);
                    break;
                case 7: // noise
                    h_readings[reading_idx].noise = atof(token);
                    break;
            }
            token = strtok(NULL, "|");
            field++;
        }
        reading_idx++;
    }
    fclose(file);

    // Allocate device memory
    SensorReading* d_readings;
    StuckInterval* d_stuck_intervals;
    int* d_stuck_count;

    handleCudaError(cudaMalloc(&d_readings, n_readings * sizeof(SensorReading)),
                   "Failed to allocate device memory for readings");
    handleCudaError(cudaMalloc(&d_stuck_intervals, TOP_N * sizeof(StuckInterval)),
                   "Failed to allocate device memory for intervals");
    handleCudaError(cudaMalloc(&d_stuck_count, sizeof(int)),
                   "Failed to allocate device memory for count");

    const char* sensor_names[] = {"ETVOC", "ECO2", "Noise"};
    
    for (int sensor_type = 0; sensor_type < 3; sensor_type++) {
        // Copy data to device
        handleCudaError(cudaMemcpy(d_readings, h_readings, 
                                 n_readings * sizeof(SensorReading),
                                 cudaMemcpyHostToDevice),
                       "Failed to copy readings to device");

        int h_stuck_count = 0;
        handleCudaError(cudaMemcpy(d_stuck_count, &h_stuck_count,
                                 sizeof(int), cudaMemcpyHostToDevice),
                       "Failed to copy count to device");

        // Launch kernel
        int threadsPerBlock = 256;
        int numBlocks = (n_readings + threadsPerBlock - 1) / threadsPerBlock;
        
        findStuckIntervals<<<numBlocks, threadsPerBlock>>>(
            d_readings, n_readings, d_stuck_intervals,
            d_stuck_count, sensor_type, TOP_N);
        
        checkLastCudaError("Kernel launch failed");
        
        handleCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");

        // Copy results back
        StuckInterval* h_stuck_intervals = (StuckInterval*)malloc(TOP_N * sizeof(StuckInterval));
        if (!h_stuck_intervals) {
            perror("Failed to allocate host memory for results");
            continue;
        }

        handleCudaError(cudaMemcpy(h_stuck_intervals, d_stuck_intervals,
                                 TOP_N * sizeof(StuckInterval),
                                 cudaMemcpyDeviceToHost),
                       "Failed to copy results from device");
        
        handleCudaError(cudaMemcpy(&h_stuck_count, d_stuck_count,
                                 sizeof(int), cudaMemcpyDeviceToHost),
                       "Failed to copy count from device");



        // Calcular durações reais
        for (int i = 0; i < h_stuck_count && i < TOP_N; i++) {
            time_t start_time = parseTimestamp(h_stuck_intervals[i].start_time);
            time_t end_time = parseTimestamp(h_stuck_intervals[i].end_time);
            h_stuck_intervals[i].duration_seconds = difftime(end_time, start_time);
        }

        // Ordenar usando qsort
        qsort(h_stuck_intervals, 
            (h_stuck_count < TOP_N ? h_stuck_count : TOP_N),
            sizeof(StuckInterval), 
            compareIntervals);

        // Print results
        printf("\nTop intervals for %s (found %d intervals):\n",
               sensor_names[sensor_type], h_stuck_count);
        printf("%-30s | %-10s | %-30s | %-30s | %-15s\n", "Device", "Value", "Start Time", "End Time", "Duration(s)");
        printf("---------------------------------------------------------------\n");

        for (int i = 0; i < h_stuck_count && i < TOP_N; i++) {
            // Create temporary copies to avoid modifying original data
            char device[256], start_time[256], end_time[256];
            strncpy(device, h_stuck_intervals[i].device, sizeof(device)-1);
            strncpy(start_time, h_stuck_intervals[i].start_time, sizeof(start_time)-1);
            strncpy(end_time, h_stuck_intervals[i].end_time, sizeof(end_time)-1);
            
            remove_newlines(device);
            remove_newlines(start_time);
            remove_newlines(end_time);
            
            printf("%-30s | %-10.2f | %-30s | %-30s | %-15ld\n",
                device,
                h_stuck_intervals[i].value,
                start_time,
                end_time,
                h_stuck_intervals[i].duration_seconds);
        }
      
        // for (int i = 0; i < h_stuck_count && i < TOP_N; i++) {
        //     printf("%-30s | %-10.2f | %-30s | %-30s | %-15ld\n",
        //            h_stuck_intervals[i].device,
        //            h_stuck_intervals[i].value,
        //            h_stuck_intervals[i].start_time,
        //            h_stuck_intervals[i].end_time,
        //            h_stuck_intervals[i].duration_seconds);
        //     printf("\ntest ----------- test\n");
        // }

        free(h_stuck_intervals);
    }

    // Cleanup
    cudaFree(d_readings);
    cudaFree(d_stuck_intervals);
    cudaFree(d_stuck_count);
    free(h_readings);
}

int main(int argc, char** argv) {
    // if (argc != 2) {
    //     printf("Usage: %s <input_csv_file>\n", argv[0]);
    //     return 1;
    // }

    // analyzeStuckReadings(argv[1]);
    analyzeStuckReadings("../data/devices.csv");
    return 0;
}