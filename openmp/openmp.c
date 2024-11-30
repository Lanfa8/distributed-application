#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <signal.h>

#define MAX_DEVICES 100
#define MAX_RECORDS 4500000
#define TOP_INTERVALS_COUNT 50
#define MAX_LINE_LENGTH 1024
#define MAX_RECORDS_PER_DEVICE 1000000

typedef struct {
    int id;
    char device[100];
    int contagem;
    char data[30];
    double timestamp;
    // float temperatura;
    // float umidade;
    // float luminosidade;
    float ruido;
    float eco2;
    float etvoc;
    // float latitude;
    // float longitude;
} SensorData;

typedef struct {
    char device[100];
    float static_value;
    double interval_duration;
    int start_id;
    int end_id;
    char start_date[30];
    char end_date[30];
} StaticInterval;

typedef struct {
    char device[100];
    SensorData* records;
    int record_count;
} DeviceGroup;

int compare_intervals(const void* a, const void* b) {
    const StaticInterval* interval1 = (const StaticInterval*)a;
    const StaticInterval* interval2 = (const StaticInterval*)b;
    
    if (interval1->interval_duration > interval2->interval_duration) return -1;
    if (interval1->interval_duration < interval2->interval_duration) return 1;
    return 0;
}

double parse_timestamp_fast(const char* timestamp) {
    int year = 0, month = 0, day = 0;
    int hour = 0, minute = 0, second = 0;
    int microsecond = 0;
    
    year = (timestamp[0] - '0') * 1000 + 
           (timestamp[1] - '0') * 100 + 
           (timestamp[2] - '0') * 10 + 
           (timestamp[3] - '0');
    
    month = (timestamp[5] - '0') * 10 + 
            (timestamp[6] - '0');
    
    day = (timestamp[8] - '0') * 10 + 
          (timestamp[9] - '0');
    
    hour = (timestamp[11] - '0') * 10 + 
           (timestamp[12] - '0');
    
    minute = (timestamp[14] - '0') * 10 + 
             (timestamp[15] - '0');
    
    second = (timestamp[17] - '0') * 10 + 
             (timestamp[18] - '0');
    
    // Parse microseconds
    microsecond = (timestamp[20] - '0') * 100000 +
                  (timestamp[21] - '0') * 10000 +
                  (timestamp[22] - '0') * 1000 +
                  (timestamp[23] - '0') * 100 +
                  (timestamp[24] - '0') * 10 +
                  (timestamp[25] - '0');
    
    struct tm time_info = {0};
    time_info.tm_year = year - 1900;
    time_info.tm_mon = month - 1;
    time_info.tm_mday = day;
    time_info.tm_hour = hour;
    time_info.tm_min = minute;
    time_info.tm_sec = second;
    
    return (double)mktime(&time_info) + (microsecond / 1000000.0);
}

int group_data_by_device(SensorData* input_data, int total_records, DeviceGroup* device_groups) {
    for (int i = 0; i < MAX_DEVICES; i++) {
        device_groups[i].records = calloc(MAX_RECORDS_PER_DEVICE, sizeof(SensorData));
        device_groups[i].record_count = 0;
        device_groups[i].device[0] = '\0';
    }

    for (int i = 0; i < total_records; i++) {
        int group_index = -1;
        
        for (int j = 0; j < MAX_DEVICES; j++) {
            if (device_groups[j].device[0] == '\0' || 
                strcmp(device_groups[j].device, input_data[i].device) == 0) {
                group_index = j;
                break;
            }
        }

        if (group_index == -1) {
            // fprintf(stderr, "Maximum device limit reached\n");
            continue;
        }

        if (device_groups[group_index].device[0] == '\0') {
            strncpy(device_groups[group_index].device, 
                    input_data[i].device, 
                    sizeof(device_groups[group_index].device) - 1);
        }

        if (device_groups[group_index].record_count < MAX_RECORDS_PER_DEVICE) {
            device_groups[group_index].records[device_groups[group_index].record_count] = input_data[i];
            device_groups[group_index].record_count++;
        }
    }

    int device_count = 0;
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (device_groups[i].device[0] != '\0') {
            device_count++;
            // printf("Device %s: %d records\n", 
            //        device_groups[i].device, 
            //        device_groups[i].record_count);
        }
    }

    return device_count;
}

void find_intervals(
    DeviceGroup* device_group, 
    char* metric, 
    StaticInterval* top_intervals 
) {   
    float last_value = -1;
    double start_time = 0, end_time = 0;
    int interval_index = 0;

    for (int i = 0; i < device_group->record_count; i++) {
        SensorData* record = &device_group->records[i];
        float current_value = -1;

        if (strcmp(metric, "etvoc") == 0 && record->etvoc != -1) {
            current_value = record->etvoc;
        } else if (strcmp(metric, "eco2") == 0 && record->eco2 != -1) {
            current_value = record->eco2;
        } else if (strcmp(metric, "ruido") == 0 && record->ruido != -1) {
            current_value = record->ruido;
        }

        if (current_value == -1) {
            if (last_value != -1) {
                StaticInterval* interval = &top_intervals[interval_index++];
                strcpy(interval->device, device_group->device);
                interval->static_value = last_value;
                interval->interval_duration = end_time - start_time;
                // interval->start_id = record->id;
                last_value = -1;
            }
            continue;
        }

        if (last_value == -1 || current_value != last_value) {
            if (last_value != -1) {
                StaticInterval* interval = &top_intervals[interval_index++];
                strcpy(interval->device, device_group->device);
                interval->static_value = last_value;
                interval->interval_duration = end_time - start_time;
                // interval->end_id = record->id;
            }

            last_value = current_value;
            start_time = record->timestamp;
            end_time = record->timestamp;
        } else {
            end_time = record->timestamp;
        }
    }

    if (last_value != -1) {
        StaticInterval* interval = &top_intervals[interval_index++];
        strcpy(interval->device, device_group->device);
        interval->static_value = last_value;
        interval->interval_duration = end_time - start_time;
    }

    qsort(top_intervals, interval_index, sizeof(StaticInterval), compare_intervals);
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
    
    if (token < line && field_count < max_fields) {
        fields[field_count++] = token;
    }
    
    while (field_count < max_fields) {
        fields[field_count++] = "";
    }
    
    return field_count;
}

int read_csv(const char* filename, SensorData* records, int max_records) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir o arquivo");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    char* fields[12] = {0};
    int record_count = 0;

    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }

    memset(records, 0, max_records * sizeof(SensorData));

    while (fgets(line, sizeof(line), file) && record_count < max_records) {
        line[strcspn(line, "\n")] = 0;

        if (strlen(line) == 0 || strspn(line, " \t|") == strlen(line)) {
            continue;
        }

        int field_count = split_line(line, fields, 12);

        if (field_count < 10 || strlen(fields[3]) < 1 || (strlen(fields[11]) < 1 && strlen(fields[10]) < 1 && strlen(fields[9]) < 1)) {
            // fprintf(stderr, "Skipping line with insufficient data: %s\n", line);
            continue;
        }

        SensorData* current = &records[record_count];

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

        current->timestamp = parse_timestamp_fast(current->data);

        // current->temperatura = (fields[4] && *fields[4]) ? atof(fields[4]) : 0.0;
        // current->umidade = (fields[5] && *fields[5]) ? atof(fields[5]) : 0.0;
        // current->luminosidade = (fields[6] && *fields[6]) ? atof(fields[6]) : 0.0;
        current->ruido = (fields[7] && *fields[7]) ? atof(fields[7]) : -1;
        current->eco2 = (fields[8] && *fields[8]) ? atof(fields[8]) : -1;
        current->etvoc = (fields[9] && *fields[9]) ? atof(fields[9]) : -1;
        // current->latitude = (fields[10] && *fields[10]) ? atof(fields[10]) : 0.0;
        // current->longitude = (fields[11] && *fields[11]) ? atof(fields[11]) : 0.0;
       
        record_count++;
    }

    fclose(file);
    // printf("Total de registros válidos lidos: %d\n", record_count);
    return record_count;
}

StaticInterval* get_top_intervals(StaticInterval** devices_top_intervals, int num_devices, int intervals_per_device) {
    int total_intervals = num_devices * intervals_per_device;
    
    StaticInterval* all_intervals = malloc(total_intervals * sizeof(StaticInterval));
    if (all_intervals == NULL) {
        return NULL;
    }
    
    int index = 0;
    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < intervals_per_device; j++) {
            all_intervals[index++] = devices_top_intervals[i][j];
        }
    }
    
    qsort(all_intervals, total_intervals, sizeof(StaticInterval), compare_intervals);
    
    StaticInterval* top_intervals = malloc(TOP_INTERVALS_COUNT * sizeof(StaticInterval));
    if (top_intervals == NULL) {
        free(all_intervals);
        return NULL;
    }
    
    int copy_count = (total_intervals < TOP_INTERVALS_COUNT) ? total_intervals : TOP_INTERVALS_COUNT;
    memcpy(top_intervals, all_intervals, copy_count * sizeof(StaticInterval));
    
    free(all_intervals);
    
    return top_intervals;
}

int main(int argc, char *argv[]) {
    int threads = atoi(argv[2]);
    char* file_to_process = argv[1];

    omp_set_num_threads(threads);

    SensorData* data = calloc(MAX_RECORDS, sizeof(SensorData));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    int total_records = read_csv(file_to_process, data, MAX_RECORDS);
    
    if (total_records <= 0) {
        fprintf(stderr, "Failed to read CSV or no records found\n");
        free(data);
        return 1;
    }

    struct timespec inicio, fim;
    double tempo;
    srand(time(NULL));
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    char* metrics[] = {"etvoc", "eco2", "ruido"};

    for (int m = 0; m < 3; m++) {
        DeviceGroup* device_groups = calloc(MAX_DEVICES, sizeof(DeviceGroup));
        int device_count = group_data_by_device(data, total_records, device_groups);

        StaticInterval** devices_top_intervals = calloc(device_count, sizeof(StaticInterval*));
        
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < device_count; i++) {
                StaticInterval* top_intervals_local = calloc(MAX_RECORDS_PER_DEVICE, sizeof(StaticInterval));
                
                find_intervals(&device_groups[i], metrics[m], top_intervals_local);
                
                devices_top_intervals[i] = top_intervals_local;
            }
        }

        StaticInterval* over_all_top_intervals = get_top_intervals(devices_top_intervals, device_count, TOP_INTERVALS_COUNT);
        
        if (over_all_top_intervals != NULL) {
            printf("%s\n", metrics[m]);
            printf("device|interval_duration|value\n");
            for (int i = 0; i < TOP_INTERVALS_COUNT; i++) {
                printf("%s|%f|%f\n",over_all_top_intervals[i].device,over_all_top_intervals[i].interval_duration,over_all_top_intervals[i].static_value);
            }
            
            free(over_all_top_intervals);
        }

        for (int i = 0; i < MAX_DEVICES; i++) {
            free(device_groups[i].records);
        }
        free(device_groups);

        for (int i = 0; i < device_count; i++) {
            free(devices_top_intervals[i]);
        }
        free(devices_top_intervals);
    }

    clock_gettime(CLOCK_MONOTONIC, &fim);
    tempo = (fim.tv_sec - inicio.tv_sec) + (fim.tv_nsec - inicio.tv_nsec) / 1e9;
    printf("%.9f\n", tempo);

    free(data);

    return 0;
}