#ifndef DATA_TENSOR_HPP
#define DATA_TENSOR_HPP
#include"MemManager.cuh"
#include<vector>
#include <memory>
using std::vector;
using std::shared_ptr;
namespace DNN_FrameWork{
    class DataTensor
    {
    protected:
    vector<int> shape;
    int count;
    shared_ptr<MemManager> data_;
    shared_ptr<MemManager> diff_;
    public:
        DataTensor(int num, int channels,int height, int width){
            count = 0;
            shape = vector<int>(4);
            reshape(num,channels,height,width);
        }
        DataTensor(){
            shape = vector<int>(4);
            shape[0]=0;
            shape[1]=0;
            shape[2]=0;
            shape[3]=0;
            count = 0;
        }
        DataTensor(vector<int>shape){
            count = 0;
            reshape(shape[0],shape[1],shape[2],shape[3]);
        }

        void reshape(int num,int channels,int height,int width);
        vector<int> DataShape(){return shape;}

        void reshape(vector<int>size);
        void reshapeLike(DataTensor data);
        inline int offset(int n=0,int c=0, int h=0 ,int w=0){return ((n*channels()+c)*height()+h)*width()+w;}
        int offset(vector<int>index);
        int DataCount();
        int DataCount(int Start_axis,int end_axis);
        int DataCount(int start_axis);
        int numElements(){return count;}
        int num(){return shape[0];}
        int channels(){return shape[1];}
        int height(){return shape[2];}
        int width(){return shape[3];}
        
        void copy_data(DataTensor &data,bool copy_diff,bool reshape);
        const float * gpu_data();
        const float * cpu_data();
        float * mutable_gpu_data();
        float * mutable_cpu_data();
        void set_cpu_data(float* data);
        void set_gpu_data(float* data);

        const float * gpu_diff();
        const float * cpu_diff();
        float * mutable_gpu_diff();
        float * mutable_cpu_diff();
        void update();
        float sumData();
        float sumDiff();
        float sumSquareData();
        float sumSquareDiff();
        float scaleData(float alpha);
        float scaleDiff(float alpha);


    };

}
#endif
