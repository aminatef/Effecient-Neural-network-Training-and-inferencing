#include"../include/initializer.hpp"
namespace DNN_FrameWork{
    initializer::initializer(int mean , int std)
    {
        this->std = std;
        this->mean=mean;
    }
    void initializer::fill(DataTensor * data)
    {
        float* dataPtr = data->mutable_gpu_data();
        gpu_rng_gaussian(data->DataCount(),this->mean,this->std,dataPtr);
    }

}
