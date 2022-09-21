#ifndef MATH_FUNCTION_HPP
#define MATH_FUNCTION_HPP
#include <stdint.h>
#include <cmath> 

#include "cblas.h"

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

namespace DNN_FrameWork{
    // NUMBER OF THREADS
    const int CUDA_NUM_THREADS = 512;

    // NUMBER OF BLOCKS
    inline int GET_NUM_BLOCKS(const int N) {
        return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    }   
    //gemm -> C = alpha * op(A)*op(B) +beta *C
    void gpu_gemm(const CBLAS_TRANSPOSE transA,
        const CBLAS_TRANSPOSE transB,
        const int M,const int N,const int K,
        const float alpha,const float*A,const float*B,
        const float beta,
        float* C);

    void gpu_gemv(const CBLAS_TRANSPOSE transA,
    const int M,const int N,
    const float alpha,const float* A,const float* x,const float beta,
    float *C);

    void gpu_set(const int N,const float alpha,float * x);
    void gpu_add(const int N,const float* A,const float*B,float*C);
    void gpu_sub(const int N,const float* A,const float*B,float*C);
    void gpu_mul(const int N,const float* A,const float*B,float*C);
    void gpu_div(const int N,const float* A,const float*B,float*C);

    void gpu_abs(const int N,const float* A,float*C);
    void gpu_Saxpy(const int N,float alpha,const float* A,float*C);
    void gpu_exp(const int N,const float* A,float*C);
    void gpu_log(const int N,const float* A,float*C);

    void gpu_rng_gaussian(const int N,const float mu,const float sigma,float* C);

}
#endif
