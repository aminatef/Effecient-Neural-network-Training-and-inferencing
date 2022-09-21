#ifndef LAYER_CONFIG_HPP
#define LAYER_CONFIG_HPP
#include<string>
using std::string;
namespace DNN_FrameWork{
class LayerConfig
{
private:
    string LayerType;
    bool phase_;//True in in training phase
    int numDataTensor_;
    bool bais_term_;
    int numOutput_;
    bool transpose_;
    int axis_;

public:
    explicit LayerConfig(string LayerType);
    LayerConfig(){}
    bool phase(){return phase_;}
    int numDataTensor(){return numDataTensor_;}
    int numOutput(){return numOutput_;}
    bool bais_term(){return bais_term_;}
    bool transpose(){return transpose_;}
    bool axis(){return axis_;}
    void DenseLayerConfig(int numOutputs,int numDataTensor,int axis,bool bais_term,bool transpose,bool phase);
};
}
#endif
