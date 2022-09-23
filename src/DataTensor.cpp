#include <vector>
#include <memory>
#include "../include/DataTensor.hpp"
#include"../include/MathFunctions.cuh"
using std::vector;
using std::shared_ptr;
namespace DNN_FrameWork
{
    
    void DataTensor::reshape( int num,  int channels,
                                     int height,  int width) 
    {
        shape[0]=num;
        shape[1]=channels;
        shape[2]=height;
        shape[3]=width;
        reshape(shape);
    }

    void DataTensor::reshape(vector<int> newShape)
    {
        int newCount = 1;
        shape.resize(newShape.size());
        for(int i = 0 ;i<shape.size();i++){
            newCount*=newShape[i];
            shape[i] = newShape[i];
        }
        if(newCount>count){
            count = newCount;
            data_.reset(new MemManager(newCount*(sizeof(float))));
            diff_.reset(new MemManager(newCount*(sizeof(float))));
        }
    }

    
    void DataTensor::reshapeLike(DataTensor data)
    {
        reshape(data.DataShape());
    }

    
    int DataTensor::offset(vector<int> index)
    {
        int offset=1;
        for(int i = 0;i<shape.size();i++){
            offset*=shape[i];
            offset+=index[i];
        }
        return offset;
    }

    
    int DataTensor::DataCount()
    {
        count = 1;
        for(int i =0;i<shape.size();i++){
            count*=shape[i];
        }
        return count;
    }
    int DataTensor::DataCount(int Start_axis,int end_axis)
    {
        count = 1;
        for(int i =Start_axis;i<end_axis;i++){
            count*=shape[i];
        }
        return count;
    }

    int DataTensor::DataCount(int Start_axis)
    {
        return DataCount(Start_axis,shape.size());
    }


    
    void DataTensor::copy_data(DataTensor &data,bool copy_diff,bool reshape)
    {
        vector<int> dShape = data.DataShape();
        int dCount = data.DataCount();
        if(dCount!=count||data.DataShape()!=shape){
            if(reshape)
                reshapeLike(data);
        }
        if(copy_diff){
            MemManager::DNNcudaMemCpyDefault((float*)diff_->mutable_gpu_data(),
                        count*(sizeof(float)),(void *)data.gpu_diff());
        }else{
            MemManager::DNNcudaMemCpyDefault((float*)diff_->mutable_gpu_data(),
                        count*(sizeof(float)),(void*)data.gpu_data());
        }
    }

    const float *DataTensor::gpu_data()
    {
        return (const float*)this->data_->gpu_data();
    }

    
    const float *DataTensor::cpu_data()
    {
        return (const float*)this->data_->cpu_data();
    }

    
    float *DataTensor::mutable_gpu_data()
    {
        return (float*)this->data_->mutable_gpu_data();
    }

    
    float *DataTensor::mutable_cpu_data()
    {
        return (float*)this->data_->mutable_cpu_data();
    }



    
    const float *DataTensor::gpu_diff()
    {
        return (const float*)this->diff_->gpu_data();
    }

    
    const float *DataTensor::cpu_diff()
    {
        return (const float*)this->diff_->cpu_data();
    }

    
    float *DataTensor::mutable_gpu_diff()
    {
        return (float*)this->diff_->mutable_gpu_data();
    }

    
    float *DataTensor::mutable_cpu_diff()
    {
        return (float*)this->diff_->mutable_cpu_data();
    }

    void DataTensor::set_cpu_data(float *data)
    {
        int size = count*sizeof(float);
        if(data_->get_size()!=size){
            data_.reset(new MemManager(size));
            diff_.reset(new MemManager(size));
        }
        data_->set_cpu_data(data);
    }

    
    void DataTensor::set_gpu_data(float *data)
    {
        int size = count*sizeof(float);
        if(data_->get_size()!=size){
            data_.reset(new MemManager(size));
            diff_.reset(new MemManager(size));
        }   
        data_->set_gpu_data(data);
    }

    void DataTensor::update(){
        if (this->data_->get_state()==MemManager::IN_GPU)
        {
            gpu_Saxpy(this->count,(float)-1,
            (const float*)this->diff_->gpu_data(),
            (float*)this->data_->mutable_gpu_data());
        }
    }
    float DataTensor::sumData(){
        float output;
        gpu_sum(this->DataCount(),gpu_data(),&output);
        return output;
    }
    float DataTensor::sumDiff(){
        float output;
        gpu_sum(this->DataCount(),gpu_diff(),&output);
        return output;
    }
    float DataTensor::sumSquareData(){
        float output;
        const float* data = gpu_data();
        gpu_dot(this->DataCount(),data,data,&output);
        return output;
    }
    float DataTensor::sumSquareDiff(){
        float output;
        const float* data = gpu_diff();
        gpu_dot(this->DataCount(),data,data,&output);
        return output;
    }
    float DataTensor::scaleData(float alpha){
        const float* data = gpu_data();
        gpu_scale(this->DataCount(),data,alpha,mutable_gpu_data());
    }
    float DataTensor::scaleDiff(float alpha){
        const float* data = gpu_diff();
        gpu_scale(this->DataCount(),data,alpha,mutable_gpu_diff());
    }

}