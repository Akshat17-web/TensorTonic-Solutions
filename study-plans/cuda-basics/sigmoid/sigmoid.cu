#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        float denom = 1 + exp(input[i] * (-1));
        output[i] = 1/denom;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}