#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "vector.h"
#include "config.h"

#define NUM_BLOCKS 256
#define THREADS_PER_BLOCK 256

__global__ void computeAccels(vector3* accels, vector3* hPos, double* mass) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i == j) {
			FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
		}
		else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[i][k] - hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	}
}

__global__ void updateVelPos(vector3* accels, vector3* hVel, vector3* hPos) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NUMENTITIES) {
		vector3 accel_sum = { 0, 0, 0 };
		for (int j = 0; j < NUMENTITIES; j++) {
			for (int k = 0; k < 3; k++) {
				accel_sum[k] += accels[i * NUMENTITIES + j][k];
			}
		}
		for (int k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}
	}
}
void compute() {
    vector3* hPos = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    vector3* hVel = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    double* hMass = (double*)malloc(sizeof(double) * NUMENTITIES);

    vector3* dPos;
    vector3* dVel;
    double* dMass;
    vector3* dAccels;
    cudaMalloc((void**)&dPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&dVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&dMass, sizeof(double) * NUMENTITIES);
    cudaMalloc((void**)&dAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    cudaMemcpy(dPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, hMass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 gridDim((NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    computeAccels<<<gridDim, blockDim>>>(dAccels, dPos, dMass);
    cudaDeviceSynchronize();

    updateVelPos<<<(NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dAccels, dVel, dPos);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
    cudaFree(dAccels);

    free(hPos);
    free(hVel);
    free(hMass);
}
