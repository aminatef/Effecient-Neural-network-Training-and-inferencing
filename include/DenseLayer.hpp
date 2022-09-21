#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP
#include"Layer.hpp"
#include"MathFunctions.cuh"
#include"initializer.hpp"
namespace DNN_FrameWork{
    class DenseLayer : public Layer
    {
    protected:
        virtual void Forward_gpu(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        virtual void Backward_gpu(const vector<DataTensor*>&input,vector<bool> propgateDown,const vector<DataTensor*>&output);
        int M;
        int K;
        int N;
        bool bias_term;
        bool transpose;
        int mean,std;
        DataTensor baisMultipler;
    public:
        virtual void LayerSetUp(const vector<DataTensor*>&input,const vector<DataTensor*>&output,int mean=5,int std = 7);
        virtual void Reshape(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
    };

}
#endif
