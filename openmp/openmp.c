#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <signal.h>

#define MAX_DEVICES 100
#define MAX_RECORDS 1000000
#define TOP_INTERVALS 50
#define MAX_LINE_LENGTH 1024
#define MAX_FIELD_LENGTH 256
#define MAX_DEVICE_NAME 100
#define MAX_RECORDS_PER_DEVICE 10000

typedef struct {
    int id;
    char device[100];
    int contagem;
    char data[30];
    float temperatura;
    float umidade;
    float luminosidade;
    float ruido;
    float eco2;
    float etvoc;
    float latitude;
    float longitude;
} SensorData;

typedef struct {
    char device[100];
    float static_value;
    double interval_duration;
} StaticInterval;

typedef struct {
    char device[100];
    SensorData* records;
    int record_count;
} DeviceGroup;

// Função de comparação para qsort
int compare_intervals(const void* a, const void* b) {
    const StaticInterval* interval1 = (const StaticInterval*)a;
    const StaticInterval* interval2 = (const StaticInterval*)b;
    
    if (interval1->interval_duration > interval2->interval_duration) return -1;
    if (interval1->interval_duration < interval2->interval_duration) return 1;
    return 0;
}

// Conversão de timestamp segura
double parse_timestamp(const char* timestamp) {
    struct tm time_info = {0};
    char temp_timestamp[64];
    strncpy(temp_timestamp, timestamp, sizeof(temp_timestamp) - 1);
    temp_timestamp[sizeof(temp_timestamp) - 1] = '\0';
    
    char* dot = strchr(temp_timestamp, '.');
    if (dot) *dot = '\0';
    
    if (sscanf(temp_timestamp, "%d-%d-%d %d:%d:%d", 
               &time_info.tm_year, &time_info.tm_mon, &time_info.tm_mday,
               &time_info.tm_hour, &time_info.tm_min, &time_info.tm_sec) != 6) {
        fprintf(stderr, "Erro ao parsear timestamp: %s\n", temp_timestamp);
        return 0.0;
    }
    
    time_info.tm_year -= 1900;
    time_info.tm_mon -= 1;
    
    return (double)mktime(&time_info);
}

// Function to group data by device
int group_data_by_device(SensorData* input_data, int total_records, DeviceGroup* device_groups) {
    // Initialize device groups
    for (int i = 0; i < MAX_DEVICES; i++) {
        device_groups[i].records = calloc(MAX_RECORDS_PER_DEVICE, sizeof(SensorData));
        device_groups[i].record_count = 0;
        device_groups[i].device[0] = '\0';
    }

    // Sort and group data
    for (int i = 0; i < total_records; i++) {
        // Find or create group for this device
        int group_index = -1;
        
        // Try to find existing group
        for (int j = 0; j < MAX_DEVICES; j++) {
            if (device_groups[j].device[0] == '\0' || 
                strcmp(device_groups[j].device, input_data[i].device) == 0) {
                group_index = j;
                break;
            }
        }

        // Validate group index
        if (group_index == -1) {
            fprintf(stderr, "Maximum device limit reached\n");
            continue;
        }

        // Set device name if not set
        if (device_groups[group_index].device[0] == '\0') {
            strncpy(device_groups[group_index].device, 
                    input_data[i].device, 
                    sizeof(device_groups[group_index].device) - 1);
        }

        // Add record to device group
        if (device_groups[group_index].record_count < MAX_RECORDS_PER_DEVICE) {
            device_groups[group_index].records[device_groups[group_index].record_count] = input_data[i];
            device_groups[group_index].record_count++;
        }
    }

    // Count and print actual number of device groups
    int device_count = 0;
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (device_groups[i].device[0] != '\0') {
            device_count++;
            printf("Device %s: %d records\n", 
                   device_groups[i].device, 
                   device_groups[i].record_count);
        }
    }

    return device_count;
}

// Modify find_static_intervals to work with a single device group
void find_static_intervals_for_device(
    DeviceGroup* device_group, 
    char* metric,
    StaticInterval* top_intervals
) {
    StaticInterval* local_intervals = calloc(MAX_RECORDS_PER_DEVICE, sizeof(StaticInterval));
    int interval_count = 0;

    for (int i = 0; i < device_group->record_count - 1; i++) {
        float current_value = 0, next_value = 0;
        
        // Select metric
        if (strcmp(metric, "etvoc") == 0) {
            current_value = device_group->records[i].etvoc;
            next_value = device_group->records[i+1].etvoc;
        } else if (strcmp(metric, "eco2") == 0) {
            current_value = device_group->records[i].eco2;
            next_value = device_group->records[i+1].eco2;
        } else if (strcmp(metric, "ruido") == 0) {
            current_value = device_group->records[i].ruido;
            next_value = device_group->records[i+1].ruido;
        } else {
            fprintf(stderr, "Invalid metric: %s\n", metric);
            continue;
        }
        
        // Calculate time interval
        double start_time = parse_timestamp(device_group->records[i].data);
        double end_time = parse_timestamp(device_group->records[i+1].data);
        double interval = fabs(end_time - start_time);
        
        // Check for static value
        if (fabs(current_value - next_value) < 0.001 && interval > 0) {
            strncpy(local_intervals[interval_count].device, device_group->device, 100);
            local_intervals[interval_count].static_value = current_value;
            local_intervals[interval_count].interval_duration = interval;
            interval_count++;
        }
    }
    
    // Sort and copy top intervals
    qsort(local_intervals, interval_count, sizeof(StaticInterval), compare_intervals);
    
    int top_count = (interval_count < TOP_INTERVALS) ? interval_count : TOP_INTERVALS;
    memcpy(top_intervals, local_intervals, top_count * sizeof(StaticInterval));
    
    free(local_intervals);
}

int split_line(char* line, char** fields, int max_fields) {
    if (!line || !*line) return 0;
    
    int field_count = 0;
    char* token = line;
    
    while (*line && field_count < max_fields) {
        if (*line == '|') {
            *line = '\0';
            fields[field_count++] = token;
            token = line + 1;
        }
        line++;
    }
    
    // Handle last field
    if (token < line && field_count < max_fields) {
        fields[field_count++] = token;
    }
    
    // Zero out remaining fields to prevent garbage
    while (field_count < max_fields) {
        fields[field_count++] = "";
    }
    
    return field_count;
}

int read_csv(const char* filename, SensorData* records, int max_records) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir o arquivo");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    char* fields[12] = {0};  // Initialize to NULL
    int record_count = 0;

    // Skip header
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }

    memset(records, 0, max_records * sizeof(SensorData));

    while (fgets(line, sizeof(line), file) && record_count < max_records) {
        line[strcspn(line, "\n")] = 0;  // Remove newline

        // Skip completely empty lines
        if (strlen(line) == 0 || strspn(line, " \t|") == strlen(line)) {
            continue;
        }

        int field_count = split_line(line, fields, 12);

        // Require at least minimal data
        if (field_count < 12 || strlen(fields[3]) < 10 || strlen(fields[11]) < 1 || strlen(fields[10]) < 1 || strlen(fields[9]) < 1) {
            fprintf(stderr, "Skipping line with insufficient data: %s\n", line);
            continue;
        }

        SensorData* current = &records[record_count];

        // Safe parsing with defaults
        current->id = (fields[0] && *fields[0]) ? atoi(fields[0]) : 0;
        
        strncpy(current->device, 
                (fields[1] && *fields[1]) ? fields[1] : "UNKNOWN", 
                sizeof(current->device) - 1);
        current->device[sizeof(current->device) - 1] = '\0';

        current->contagem = (fields[2] && *fields[2]) ? atoi(fields[2]) : 0;
        
        strncpy(current->data, 
                (fields[3] && *fields[3]) ? fields[3] : "", 
                sizeof(current->data) - 1);
        current->data[sizeof(current->data) - 1] = '\0';

        // Safe conversions with error checking
        current->temperatura = (fields[4] && *fields[4]) ? atof(fields[4]) : 0.0;
        current->umidade = (fields[5] && *fields[5]) ? atof(fields[5]) : 0.0;
        current->luminosidade = (fields[6] && *fields[6]) ? atof(fields[6]) : 0.0;
        current->ruido = (fields[7] && *fields[7]) ? atof(fields[7]) : 0.0;
        current->eco2 = (fields[8] && *fields[8]) ? atof(fields[8]) : 0.0;
        current->etvoc = (fields[9] && *fields[9]) ? atof(fields[9]) : 400.0;  // Default to 400
        current->latitude = (fields[10] && *fields[10]) ? atof(fields[10]) : 0.0;
        current->longitude = (fields[11] && *fields[11]) ? atof(fields[11]) : 0.0;

        record_count++;
    }

    fclose(file);
    printf("Total de registros válidos lidos: %d\n", record_count);
    return record_count;
}

int main() {
    // Allocation with verification
    SensorData* data = calloc(MAX_RECORDS, sizeof(SensorData));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Read CSV
    int total_records = read_csv("../data/devices.csv", data, MAX_RECORDS);
    
    if (total_records <= 0) {
        fprintf(stderr, "Failed to read CSV or no records found\n");
        free(data);
        return 1;
    }

    // Device grouping
    DeviceGroup* device_groups = calloc(MAX_DEVICES, sizeof(DeviceGroup));
    int device_count = group_data_by_device(data, total_records, device_groups);

    // Parallel processing of device groups
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < device_count; i++) {
            StaticInterval* top_intervals = calloc(TOP_INTERVALS, sizeof(StaticInterval));
            
            // Process each metric for the device
            find_static_intervals_for_device(&device_groups[i], "etvoc", top_intervals);
            
            // Print results for this device
            #pragma omp critical
            {
                printf("\nDevice %s - Top ETVOC Static Intervals:\n", device_groups[i].device);
                for (int j = 0; j < TOP_INTERVALS; j++) {
                    if (top_intervals[j].interval_duration > 0) {
                        printf("  Value: %.1f, Interval: %.2f seconds\n", 
                               top_intervals[j].static_value, 
                               top_intervals[j].interval_duration);
                    }
                }
            }
            
            free(top_intervals);
        }
    }

    // Free memory
    for (int i = 0; i < MAX_DEVICES; i++) {
        free(device_groups[i].records);
    }
    free(data);

    return 0;
}