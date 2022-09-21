#include"../include/DenseLayer.hpp"
namespace DNN_FrameWork{
    void DenseLayer :: LayerSetUp(const vector<DataTensor*>&input,const vector<DataTensor*>&output,int mean = 5,int std=7){
        int numOutput = this->config.numOutput();
        this->bias_term = this->config.bais_term();
        this->transpose = this->config.transpose();
        N = numOutput;
        int axis = this->config.axis();
        K = input[0]->DataCount(axis);
        if(bias_term){
            this->data_.resize(2);
        }else{
            this->data_.resize(1);
        }
        vector<int>weightShape(2);
        if(transpose){
            weightShape[0]=K;
            weightShape[1]=N;
        }else{
            weightShape[0]=N;
            weightShape[1]=K;
        }
        this->data_[0].reset(new DataTensor(weightShape));
        initializer init = initializer(this->mean,this->std);
        init.fill(this->data_[0].get());
        if(this->bias_term){
            vector<int> baisShape(1,N);
            this->data_[1].reset(new DataTensor(baisShape));
            init.fill(this->data_[1].get());
        }
        this->propagate_grad.resize(this->data_.size(),true);
    }
    void DenseLayer :: Reshape(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
        int axis = config.axis();
        int newK = input[0]->DataCount(axis);//axis is the start axis
        M = input[0]->DataCount(0,axis);
        vector<int> topShape = input[0]->DataShape();
        topShape[axis] = N;
        output[0]->reshape(topShape);
        if(bias_term){
            vector<int>baisShape(1,M);
            baisMultipler.reshape(baisShape);
            gpu_set(M,float(1),baisMultipler.mutable_gpu_data());
        }
    }
}