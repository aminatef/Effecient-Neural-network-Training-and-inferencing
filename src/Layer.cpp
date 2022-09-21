#include"../include/Layer.hpp"
namespace DNN_FrameWork
{
    void Layer :: Forward(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
        Reshape(input,output);
        Forward_gpu(input,output);
    }
    void Layer :: Backward(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
        Backward_gpu(input,output);
    }
}