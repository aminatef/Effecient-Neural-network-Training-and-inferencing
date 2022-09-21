#include"../include/LayerConfig.hpp"
namespace DNN_FrameWork
{
    LayerConfig::LayerConfig(string LayerType)
    {
        this->LayerType = LayerType;
    }
    void LayerConfig::DenseLayerConfig(int numOutput,int numDataTensor,int axis,bool bais_term,bool transpose,bool phase)
    {
        this->numOutput_ = numOutput;
        this->phase_=phase;
        this->numDataTensor_=numDataTensor;
        this->bais_term_=bais_term;
        this->transpose_=transpose;
        this->axis_=axis;
    }
        
} // namespace DNN_FrameWork
