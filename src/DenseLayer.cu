#include"../include/DenseLayer.hpp"
namespace DNN_FrameWork{
    void DenseLayer :: Forward_gpu(const vector<DataTensor*>&input,
                                   const vector<DataTensor*>&output){

        const float* inputData = input[0]->gpu_data();
        float * outputData = output[0]->mutable_gpu_data();
        const float* LayerWeight = this->data_[0]->gpu_data();

        if(M==1){
            gpu_gemv(CblasNoTrans,N,K,(float)1,LayerWeight,inputData,(float)0,outputData);
            if(bias_term){
                gpu_Saxpy(N,baisMultipler.cpu_data()[0],this->data_[1]->gpu_data(),outputData);
            }
        }else{

            gpu_gemm(CblasNoTrans,(transpose?CblasTrans:CblasNoTrans),
            M,N,K,float(1),
            inputData,LayerWeight,float(0),
            outputData);

            if(bias_term){
                gpu_gemm(CblasNoTrans,CblasNoTrans,
                M,N,1,(float)1,
                baisMultipler.gpu_data(),
                this->data_[1]->gpu_data(),
                (float)1,outputData);
            }
        }
    }
    void DenseLayer :: Backward_gpu(const vector<DataTensor*>&input,
                                    vector<bool>propagateDown,
                                    const vector<DataTensor*>&output){
        const float* prevDiff = output[0]->gpu_diff();
        if(this->propagate_grad[0]){
            const float* InputData = input[0]->gpu_data();
            if(this->transpose){
                gpu_gemm(CblasTrans,CblasNoTrans,
                        K,N,M,
                        (float)1,InputData,prevDiff,
                        (float)1,
                        this->data_[0]->mutable_gpu_diff());
            }
            else{
                gpu_gemm(CblasTrans,CblasNoTrans,
                        N,K,M,
                        (float)1,prevDiff,InputData,
                        (float)1,
                        this->data_[0]->mutable_gpu_diff());
            }
        }
        if(this->propagate_grad[1]&&this->bias_term){
            gpu_gemv(CblasTrans,
                    M,N,(float)1,
                    prevDiff,baisMultipler.gpu_data(),
                    (float)1,
                    this->data_[1]->mutable_gpu_diff());
        }
        if(propagateDown[0]){
           if(this->transpose){
                gpu_gemm(CblasNoTrans,CblasTrans,
                        M,K,N,(float)1,
                        prevDiff,this->data_[0]->gpu_data(),
                        (float)0,
                        output[0]->mutable_gpu_diff());
           } 
           else{
            gpu_gemm(CblasNoTrans,CblasNoTrans,
                    M,K,N,(float)1,
                    prevDiff,this->data_[0]->gpu_data(),
                    (float)0,
                    output[0]->mutable_gpu_diff());
           }
            
        }


    }
}