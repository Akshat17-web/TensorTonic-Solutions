#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Write code here
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < M && j < N){
        float sum = 0;
        for(int itr = 0; itr<K; itr++){
            sum += A[i * K + itr] * B[itr * N + j];
        }
        C[N * i + j] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
