#ifndef DATA_TENSOR_HPP
#define DATA_TENSOR_HPP
#include"MemManager.cuh"
#include<vector>
#include <memory>
using std::vector;
using std::shared_ptr;
namespace DNN_FrameWork{
    template<typename Dtype>
    class DataTensor
    {
    private:
    vector<int> shape;
    int count;
    shared_ptr<MemManager> data;
    shared_ptr<MemManager> diff;
    public:
        DataTensor(const int num, const int channels,
                   const int height,const int width);

        void reshape(const int num, const int channels,
                     const int height,const int width);
        vector<int> DataShape(){return shape;}

        void reshape(vector<int>size);
        void reshapeLike(DataTensor<Dtype> data);
        inline int offset(int n=0,int c=0, int h=0 ,int w=0){return ((n*channels+c)*height+h)*width+w;}
        int offset(vector<int>index);
        int DataCount();
        int numElements(){return count;}
        int num(){return shape[0];}
        int channels(){return shape[1];}
        int height(){return shape[2];}
        int width(){return shape[3];}
        
        void copy_data(const DataTensor<Dtype> &data,bool copy_diff,bool reshape);
        const Dtype * gpu_data();
        const Dtype * cpu_data();
        Dtype * mutable_gpu_data();
        Dtype * mutable_cpu_data();
        void set_cpu_data(Dtype* data);
        void set_gpu_data(Dtype* data);

        const Dtype * gpu_diff();
        const Dtype * cpu_diff();
        Dtype * mutable_gpu_diff();
        Dtype * mutable_cpu_diff();



        ~DataTensor();
    };

}
#endif
