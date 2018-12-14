#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG

#define INDEX(fst, snd, n) ((fst) * (n) + (snd))
#define SIZE (5000)
#define TILL (100)
#define N_TILL (SIZE / TILL)
__global__ void multiple(float* matrix, float* vector, float* out) {
    /*
     * a thread get 100 element in a line (one line 50 thread)
     * thread is (50, 20)
     * 20 lines fill a block.
     * a matrix has 250 blocks
     */
    int x = threadIdx.x;
    int y = threadIdx.y;
    int blk = blockIdx.x;
    float sum = 0;
    for (int i = x * 100; i < (x + 1) * 100; ++i) {
        sum += matrix[INDEX(y + blk * 20, i, SIZE)] * vector[i];
    }
    atomicAdd(&out[y + blk * 20], (float)sum);
}

void validator(float* matrix, float* vector, float* out) {
    for (int i = 0; i < SIZE; ++i) {
        float sum = 0;
        for (int j = 0; j < SIZE; ++j) {
            sum += matrix[INDEX(i, j, SIZE)] * vector[j];
        }
        out[i] = sum;
    }
}


int main() {
    float* hA = (float*) malloc(sizeof(float) * SIZE * SIZE);
    float* dA;
    cudaMalloc((void**) &dA, sizeof(float) * SIZE * SIZE);
    float* hx = (float*) malloc(sizeof(float) * SIZE);
    float* dx;
    cudaMalloc((void**) &dx, sizeof(float) * SIZE * SIZE);
    float* out;
    cudaMalloc((void**) &out, sizeof(float) * SIZE);
    float* valout = (float*) malloc(sizeof(float)  * SIZE);

    // init hA and hx
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            hA[INDEX(i, j, SIZE)] = i - 0.1 * j + 1;
        }
        hx[i] = 0.2 * i - 0.1 * sqrt(i);
    }

    // init out
    cudaMemset(out, 0, sizeof(float)* SIZE);
    memset(valout, 0, sizeof(float) * SIZE);

    // transfer to gpu
    cudaMemcpy(dA, hA, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    dim3 threads(50, 20);
    multiple<<<250, threads>>>(dA, dx, out);
    validator(hA, hx, valout);

    free(hA);
    free(hx);
    cudaFree(dA);
    cudaFree(dx);
    float* hout = (float*) malloc(sizeof(float) * SIZE);
    cudaMemcpy(hout, out, sizeof(float)* SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("%f, (%f) \n", hout[i], valout[i]);
    }
    free(valout);
    free(hout);
    cudaFree(out);
}

