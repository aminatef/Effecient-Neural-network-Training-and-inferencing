#include"../include/DenseLayer.hpp"
namespace DNN_FrameWork{
    void DenseLayer :: Forward_gpu(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
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
    void DenseLayer :: Backward_gpu(const vector<DataTensor*>&input,vector<bool>propagateDown,const vector<DataTensor*>&output){

    }
}