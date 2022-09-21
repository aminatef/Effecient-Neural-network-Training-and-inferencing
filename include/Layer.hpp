#ifndef LAYER_HPP
#define LAYER_HPP
#include"LayerConfig.hpp"
#include "DataTensor.hpp"
namespace DNN_FrameWork{
    class Layer
    {   
    public:
        explicit Layer(const LayerConfig &config){
            this->config = config;
            if(this->config.numDataTensor()>0){
                data_.resize(this->config.numDataTensor());
                for(int i =0;i<this->config.numDataTensor();i++){
                    data_[i].reset(new DataTensor());
                }
            }
        }
        ~Layer();
        virtual void LayerSetUp(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        virtual void Reshape(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        void SetUp(const vector<DataTensor*>&input,const vector<DataTensor*>&output){
            LayerSetUp(input,output);
            Reshape(input,output);
        }
        void Forward(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        void Backward(const vector<DataTensor*>&input,const vector<DataTensor*>&output);

        virtual void Forward_gpu(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        virtual void Backward_gpu(const vector<DataTensor*>&input,const vector<DataTensor*>&output);
        protected:
        vector<shared_ptr<DataTensor>> data_;
        vector<bool> propagate_grad;
        LayerConfig config;
    };

}
#endif
