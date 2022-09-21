#include"../include/MathFunctions.cuh"

namespace DNN_FrameWork
{
    void gpu_gemm(const CBLAS_TRANSPOSE transA,
        const CBLAS_TRANSPOSE transB,
        const int M,const int N,const int K,
        const float alpha,const float*A,const float*B,
        const float beta,
        float* C)
    {
        //LEADING DIMENSION OF MATRIX A
        int leadingDimA = (transA == CblasNoTrans)? K : M;
        //LEADING DIMENSION OF MATRIX B
        int leadingDimB = (transB == CblasNoTrans)? N : K;
        //MATRIX OPEARATION FOR MATRIX A
        cublasOperation_t opA = (transA==CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        //MATRIX OPEARATION FOR MATRIX B
        cublasOperation_t opB = (transB==CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        //CUBLAS HANDLE
        cublasHandle_t handle;
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        // CUBLAS USES COLUM MAJOR FORMAT SO A AND B ARE REVRSED IN cublasSgemm
        cublasSgemm(handle,
                    opB,opA,
                    N,M,K,
                    &alpha,
                    B,leadingDimB,
                    A,leadingDimA,
                    &beta,
                    C,N);
    }

    void gpu_gemv(const CBLAS_TRANSPOSE transA,
        const int M,const int N,
        const float alpha,const float* A,const float* x,const float beta,
        float *C)
    {
        //MATRIX OPEARATION FOR MATRIX A
        cublasOperation_t opA = (transA==CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        //CUBLAS HANDLE
        cublasHandle_t handle;
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        cublasSgemv(handle,opA,N,M,&alpha,A,N,x,1,&beta,C,1);
    }

    __global__ void set_kernel(const int N, const float val,float*Y){
        for(int i = blockIdx.x*blockDim.x+threadIdx.x ;i<N;i+=blockDim.x*gridDim.x){
            Y[i] = val;
        }
    }

    void gpu_set(const int N,const float val,float * Y){
        if(val == 0 ){
            cudaMemset(Y,0,sizeof(float)*N);
        }else{
            set_kernel<<<GET_NUM_BLOCKS(N),CUDA_NUM_THREADS>>>(N,val,Y);
        }
    }

    __global__ void add_kernel(const int N,const float * A, const float *B,float*C){
        for(int i = blockIdx.x*blockDim.x+threadIdx.x ;i<N;i+=blockDim.x*gridDim.x){
            C[i] = A[i]+B[i];
        }
    }
    void gpu_add(const int N,const float* A,const float*B,float*C){
        add_kernel<<<GET_NUM_BLOCKS(N),CUDA_NUM_THREADS>>>(N,A,B,C);
    }
    __global__ void sub_kernel(const int N,const float * A, const float *B,float*C){
        for(int i = blockIdx.x*blockDim.x+threadIdx.x ;i<N;i+=blockDim.x*gridDim.x){
            C[i] = A[i]-B[i];
        }
    }
    void gpu_sub(const int N,const float* A,const float*B,float*C){
        sub_kernel<<<GET_NUM_BLOCKS(N),CUDA_NUM_THREADS>>>(N,A,B,C);
    }

    void gpu_mul(const int N,const float* A,const float*B,float*C){}
    void gpu_div(const int N,const float* A,const float*B,float*C){}

    void gpu_abs(const int N,const float* A,float*C){}
    __global__ void exp_kernel(const int N,const float*A,float*B){
        for(int i = blockIdx.x*blockDim.x+threadIdx.x ;i<N;i+=blockDim.x*gridDim.x){
            B[i]=exp(A[i]);
        }
    }
    void gpu_exp(const int N,const float* A,float*C){
        exp_kernel<<<GET_NUM_BLOCKS(N),CUDA_NUM_THREADS>>>(N,A,C);
    }
    void gpu_log(const int N,const float* A,float*C){}

    void gpu_rng_gaussian(const int N,const float mu,const float sigma,float* C){
        curandGenerator_t gen; 
        curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateNormal(gen,C,N,mu,sigma);
    }
    void gpu_Saxpy(const int N,float alpha,const float* A,float*C){
        cublasHandle_t handle;
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        cublasSaxpy(handle,N,&alpha,A,1,C,1);
    }
    
} // namespace DNN_FrameWork

    

    
