#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_ROWS 4450000
#define TOP_N 50

// Estrutura para armazenar as leituras
typedef struct {
    char id[TOP_N], device[TOP_N], date[TOP_N];
    int count;
    float temperature, humidity, luminosity, noise, eco2, etvoc, latitude, longitude;
} DataRow;

int readCSV(const char *filename, DataRow *data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir o arquivo");
        return -1;
    }

    char line[2048];
    int row = 0;

    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Erro ao ler o cabeçalho do arquivo.\n");
        fclose(file);
        return -1;
    }

    while (fgets(line, sizeof(line), file)) {
        if (row >= MAX_ROWS) {
            fprintf(stderr, "Número de linhas excede o limite (%d).\n", MAX_ROWS);
            break;
        }

        char *token = strtok(line, "|");
        if (!token) continue;
        strncpy(data[row].id, token, sizeof(data[row].id) - 1);
        data[row].id[sizeof(data[row].id) - 1] = '\0';

        token = strtok(NULL, "|");
        if (!token) continue;
        strncpy(data[row].device, token, sizeof(data[row].device) - 1);
        data[row].device[sizeof(data[row].device) - 1] = '\0';

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].count = atoi(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        strncpy(data[row].date, token, sizeof(data[row].date) - 1);
        data[row].date[sizeof(data[row].date) - 1] = '\0';

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].temperature = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].humidity = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].luminosity = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].noise = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].eco2 = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].etvoc = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].latitude = atof(token);

        token = strtok(NULL, "|");
        if (!token) continue;
        data[row].longitude = atof(token);

        // printf("%s|%s|%d|%s|%f|%f|%f|%f|%f|%f|%f|%f \n\n", data[row].id, data[row].device, data[row].count, data[row].date,
        //             data[row].temperature, data[row].humidity, data[row].luminosity,
        //             data[row].noise, data[row].eco2, data[row].etvoc,
        //             data[row].latitude, data[row].longitude);
        row++;
    }

    fclose(file);
    return row;
}

// Kernel CUDA para identificar intervalos constantes
__global__ void findStalledIntervals(const float *values, int n, int *start, int *length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n - 1) {
        if (values[idx] == values[idx + 1]) {
            atomicAdd(&length[idx], 1);
        } else {
            start[idx + 1] = idx + 1; // Início de um novo intervalo
        }
    }
}

void classifyIntervals(int *start, int *length, int n, DataRow *data, const char *column_name) {
    typedef struct {
        int start_idx, length;
    } Interval;

    Interval *intervals = (Interval *)malloc(n * sizeof(Interval));
    int count = 0;

    for (int i = 0; i < n; ++i) {
        if (length[i] > 0) {
            intervals[count].start_idx = start[i];
            intervals[count].length = length[i];
            count++;
        }
    }

    qsort(intervals, count, sizeof(Interval), [](const void *a, const void *b) {
        return ((Interval *)b)->length - ((Interval *)a)->length;
    });

    printf("Top %d intervals for %s:\n", TOP_N, column_name);
    for (int i = 0; i < TOP_N && i < count; ++i) {
        int idx = intervals[i].start_idx;
        printf("Device: %s, Value: %.2f, Interval: %d - %d, Length: %d\n",
               data[idx].device, data[idx].etvoc, idx, idx + intervals[i].length, intervals[i].length);
    }

    free(intervals);
}

int main() {
    const char *filename = "../data/devices.csv";
    DataRow *data = (DataRow *)malloc(MAX_ROWS * sizeof(DataRow));
    int n = readCSV(filename, data);

    if (n < 0) {
        free(data);
        return 1;
    }

    float *etvoc = (float *)malloc(n * sizeof(float));
    float *eco2 = (float *)malloc(n * sizeof(float));
    float *noise = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        etvoc[i] = data[i].etvoc;
        eco2[i] = data[i].eco2;
        noise[i] = data[i].noise;
    }

    // Alocar memória na GPU
    float *d_values;
    int *d_start, *d_length;

    cudaMalloc(&d_values, n * sizeof(float));
    cudaMalloc(&d_start, n * sizeof(int));
    cudaMalloc(&d_length, n * sizeof(int));

    // Processar `etvoc`
    cudaMemcpy(d_values, etvoc, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_start, 0, n * sizeof(int));
    cudaMemset(d_length, 0, n * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    findStalledIntervals<<<blocksPerGrid, threadsPerBlock>>>(d_values, n, d_start, d_length);

    // Copiar resultados para a CPU
    int *start = (int *)malloc(n * sizeof(int));
    int *length = (int *)malloc(n * sizeof(int));

    cudaMemcpy(start, d_start, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(length, d_length, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Classificar e exibir os maiores intervalos para `etvoc`
    classifyIntervals(start, length, n, data, "eTVOC");

    // Repita o processo para `eco2` e `noise`
    cudaMemcpy(d_values, eco2, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_start, 0, n * sizeof(int));
    cudaMemset(d_length, 0, n * sizeof(int));

    findStalledIntervals<<<blocksPerGrid, threadsPerBlock>>>(d_values, n, d_start, d_length);
    cudaMemcpy(start, d_start, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(length, d_length, n * sizeof(int), cudaMemcpyDeviceToHost);
    classifyIntervals(start, length, n, data, "eCO2");

    cudaMemcpy(d_values, noise, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_start, 0, n * sizeof(int));
    cudaMemset(d_length, 0, n * sizeof(int));

    findStalledIntervals<<<blocksPerGrid, threadsPerBlock>>>(d_values, n, d_start, d_length);
    cudaMemcpy(start, d_start, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(length, d_length, n * sizeof(int), cudaMemcpyDeviceToHost);
    classifyIntervals(start, length, n, data, "Noise");

    // Liberar memória
    cudaFree(d_values);
    cudaFree(d_start);
    cudaFree(d_length);

    free(start);
    free(length);
    free(etvoc);
    free(eco2);
    free(noise);
    free(data);

    return 0;
}
