#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP
#include"DataTensor.hpp"
#include"MathFunctions.cuh"
namespace DNN_FrameWork{
    class initializer
    {
    private:
        int mean;
        int std;
    public:
        initializer(int mean,int std);
        void fill(DataTensor * data);

    };
    

    
}
#endif