/**
 * Example CUDA solution for the "vector-addition" problem.
 *
 * Compile & verify:
 *     python3 run_local.py vector-addition examples/vector_addition.cu
 *
 * Compile, verify & benchmark:
 *     python3 run_local.py vector-addition examples/vector_addition.cu --benchmark
 */

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void vector_addition(const float* d_input1, const float* d_input2,
                                 float* d_output, size_t n) {
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_input1, d_input2, d_output, (int)n);
    cudaDeviceSynchronize();
}
