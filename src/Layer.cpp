#include"../include/Layer.hpp"
namespace DNN_FrameWork
{
    void Layer :: Forward(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
        Reshape(input,output);
        Forward_gpu(input,output);
        for(int i = 0 ;i<output.size();i++){
            const int count = output[i]->DataCount();
            const float* data = output[i]->gpu_data();
            const float* loss = output[i]->gpu_diff();
        }

    }
    void Layer :: Backward(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
        Backward_gpu(input,output);
    }
}