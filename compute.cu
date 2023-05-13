#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void computeKernel(vector3* accels, vector3* hPos, vector3* hVel, double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j, k;

    if (i < NUMENTITIES) {
        for (j = 0; j < NUMENTITIES; j++) {
            if (i == j) {
                FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
            }
            else {
                vector3 distance;
                for (k = 0; k < 3; k++)
                    distance[k] = hPos[i][k] - hPos[j][k];
                double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
                double magnitude = sqrt(magnitude_sq);
                double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
                FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude,
                            accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
            }
        }

        vector3 accel_sum = {0, 0, 0};
        for (j = 0; j < NUMENTITIES; j++) {
            for (k = 0; k < 3; k++)
                accel_sum[k] += accels[i * NUMENTITIES + j][k];
        }

        for (k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] = hVel[i][k] * INTERVAL;
        }
    }
}

void compute() {
    vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
    for (int i = 0; i < NUMENTITIES; i++)
        accels[i] = &values[i * NUMENTITIES];

    // Allocate memory on the GPU
    vector3* d_accels;
    cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3* d_hPos;
    cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
    vector3* d_hVel;
    cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
    double* d_mass;
    cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);

    // Transfer data from CPU to GPU
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the CUDA kernel
    int blockSize = 256;
    int gridSize = (NUMENTITIES + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    computeKernel<<<gridSize, blockSize>>>(d_accels, d_hPos, d_hVel, d_mass);

    // Transfer the results back from GPU to CPU
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_accels);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);

    free(accels);
    free(values);
}


