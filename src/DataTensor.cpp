#include "../include/DataTensor.hpp"
#include"../include/MemManager.cuh"
#include <vector>
#include <memory>
using std::vector;
using std::shared_ptr;
namespace DNN_FrameWork
{
    template <typename Dtype>
    DataTensor<Dtype>::DataTensor(const int num, const int channels,
                                  const int height, const int width)
    {
        count = num*channels*height*width;
        reshape(num,channels,height,width);
    }
    
    template <typename Dtype>
    void DataTensor<Dtype>::reshape(const int num, const int channels,
                                    const int height, const int width) 
    {
        shape[0]=num;
        shape[1]=channels;
        shape[2]=height;
        shape[3]=width;
        reshape(shape);
    }

    template <typename Dtype>
    void DataTensor<Dtype>::reshape(vector<int> newShape)
    {
        int newCount = 1;
        shape.resize(newShape.size());
        for(int i = 0 ;i<shape.size();i++){
            newCount*=newShape[i];
            shape[i] = newShape[i];
        }
        if(newCount>count){
            count = newCount;
            data.reset(new MemManager(newCount*(sizeof(Dtype))));
            diff.reset(new MemManager(newCount*(sizeof(Dtype))));
        }
    }

    template <typename Dtype>
    void DataTensor<Dtype>::reshapeLike(DataTensor<Dtype> data)
    {
        reshape(data.DataShape());
    }

    template <typename Dtype>
    int DataTensor<Dtype>::offset(vector<int> index)
    {
        int offset=1;
        for(int i = 0;i<shape.size();i++){
            offset*=shape[i];
            offset+=index[i];
        }
        return offset;
    }

    template <typename Dtype>
    int DataTensor<Dtype>::DataCount()
    {
        count = 1;
        for(int i =0;i<shape.size();i++){
            count*=shape[i];
        }
        return count;
    }


    template <typename Dtype>
    void DataTensor<Dtype>::copy_data(const DataTensor<Dtype> &data,bool copy_diff,bool reshape)
    {
        vector<int> dShape = data.DataShape();
        int dCount = data.DataCount();
        if(dCount!=count||data.DataShape()!=shape){
            if(reshape)
                reshapeLike(data);
        }
        if(copy_diff){
            MemManager::DNNcudaMemCpyDefault((Dtype*)diff->mutable_gpu_data(),
                        count*(sizeof(Dtype)),data.gpu_diff());
            
        }else{
            MemManager::DNNcudaMemCpyDefault((Dtype*)diff->mutable_gpu_data(),
                        count*(sizeof(Dtype)),data.gpu_data());
        }
    }

    template <typename Dtype>
    const Dtype *DataTensor<Dtype>::gpu_data()
    {
        return (const Dtype*)this->data->gpu_data();
    }

    template <typename Dtype>
    const Dtype *DataTensor<Dtype>::cpu_data()
    {
        return (const Dtype*)this->data->cpu_data();
    }

    template <typename Dtype>
    Dtype *DataTensor<Dtype>::mutable_gpu_data()
    {
        return (Dtype*)this->data->mutable_gpu_data();
    }

    template <typename Dtype>
    Dtype *DataTensor<Dtype>::mutable_cpu_data()
    {
        return (Dtype*)this->data->mutable_cpu_data();
    }



    template <typename Dtype>
    const Dtype *DataTensor<Dtype>::gpu_diff()
    {
        return (const Dtype*)this->diff->gpu_data();
    }

    template <typename Dtype>
    const Dtype *DataTensor<Dtype>::cpu_diff()
    {
        return (const Dtype*)this->diff->cpu_data();
    }

    template <typename Dtype>
    Dtype *DataTensor<Dtype>::mutable_gpu_diff()
    {
        return (Dtype*)this->diff->mutable_gpu_data();
    }

    template <typename Dtype>
    Dtype *DataTensor<Dtype>::mutable_cpu_diff()
    {
        return (Dtype*)this->diff->mutable_cpu_data();
    }











    template <typename Dtype>
    void DataTensor<Dtype>::set_cpu_data(Dtype *data)
    {
        int dCount = data->DataCount();
        if(dCount!=count){
            data.reset(new MemManager(dCount*(sizeof(Dtype))));
            diff.reset(new MemManager(dCount*(sizeof(Dtype))));
        }   
        data->set_cpu_data(data);
    }

    template <typename Dtype>
    void DataTensor<Dtype>::set_gpu_data(Dtype *data)
    {
        int dCount = data->DataCount();
        if(dCount!=count){
            data.reset(new MemManager(dCount*(sizeof(Dtype))));
            diff.reset(new MemManager(dCount*(sizeof(Dtype))));
        }   
        data->set_gpu_data(data);
    }

}