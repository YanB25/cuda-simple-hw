#include <stdio.h>
__global__ void helloCUDA() {
    printf("Hello from thread (%d, %d) block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}
int main() {
    dim3 grid(2, 4);
    dim3 block(8, 16);
    helloCUDA<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}
