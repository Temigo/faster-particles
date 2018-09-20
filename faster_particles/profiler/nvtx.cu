#include <cstdio>
#include "nvToolsExt.h"
// nvcc -Xcompiler -fPIC -shared -o cuda_nvtx.so nvtx.cu -lnvToolsExt
__global__ void test_kernel(float *a, float *b, float *c, size_t size){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  c[idx] = a[idx] + b[idx];
}
extern "C" {
  void test_func(float *a, float *b, float *c, size_t size){
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));
    cudaMalloc((void **)&d_c, size * sizeof(float));
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    test_kernel <<< ceil(size / 256.0), 256 >>> (d_a, d_b, d_c, size);
    cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }

  void nvtx_push(char *name){
    printf("debug_msg:nvtx_push\n");
    nvtxRangePushA(name);
  }

  void nvtx_pop(){
    nvtxRangePop();
    printf("debug_msg:nvtx_pop\n");
  }
}
