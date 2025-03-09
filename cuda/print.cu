#include "print.cuh"
#include <stdio.h>

__global__ void print_kernel(const char* str) {
    printf("Hello from the GPU!\n");
    printf("str: %s\n", str);
}

void print_gpu(const char* str) {
    print_kernel<<<1, 1>>>(str);
    cudaDeviceSynchronize();
}