#include <stdio.h>
__global__ void helloCUDA(float f) {
    printf(("Hello thread %d, f=%f\n", threadIdx.x, f);
}
int main() {
    dim3 grid(2, 4);
    dim3 block(8, 16);
    helloCUDA<<<grid, block>>>(1.23456f);
    return 0;
}