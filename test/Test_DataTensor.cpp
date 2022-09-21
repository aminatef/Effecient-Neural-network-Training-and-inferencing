#include"../include/DataTensor.hpp"
#include<iostream>
#undef NDEBUG
#include"assert.h"
#include<cstring>

using namespace std;

namespace DNN_FrameWork{
    class Test_DataTensor
    {
    private:
    DataTensor tensor;
    DataTensor preshaped_tensor;
        
    public:
        Test_DataTensor(){
            cout<<"============Testing DataTensor=================="<<endl;
            tensor = DataTensor();
            preshaped_tensor=DataTensor(2,3,4,5);
        }

        void Test_Init(){
            cout<<"TEST 1 => TEST initialization";
            assert(&tensor);
            assert(&preshaped_tensor);
            assert(preshaped_tensor.num()== 2);
            assert(preshaped_tensor.channels()== 3);
            assert(preshaped_tensor.height()== 4);
            assert(preshaped_tensor.width() == 5);
            assert(preshaped_tensor.DataCount() == 120);
            assert(tensor.DataCount() == 0);
            cout<<" PASSED"<<endl;
        }
        void Test_ptr(){
            cout<<"TEST 2 => TEST GPU/CPU ptr";
            assert(preshaped_tensor.gpu_data());
            
            assert(preshaped_tensor.cpu_data()!=NULL);
            
            assert(preshaped_tensor.mutable_gpu_data()!=NULL);
            
            assert(preshaped_tensor.mutable_cpu_data()!=NULL);
            cout<<" PASSED"<<endl;
        }
        void test_reshape(){
            cout<<"TEST 3 => TEST Reshape";
            tensor.reshape(2, 3, 4, 5);
            assert(tensor.num() == 2);
            assert(tensor.channels() == 3);
            assert(tensor.height() == 4);
            assert(tensor.width() == 5);
            assert(tensor.DataCount() == 120);
            tensor.reshapeLike(tensor);
            cout<<" PASSED"<<endl;
        }
        void test_reshape2(){
            cout<<"TEST 4 => TEST Reshape Zero";
            vector<int> shape(2);
            shape[0] = 0;
            shape[1] = 5;
            tensor.reshape(shape);
            assert(tensor.DataCount() == 0);
            cout<<" PASSED"<<endl;
        }
        void RUN_TESTS(){
            Test_Init();
            Test_ptr();
            test_reshape();
            test_reshape2();
        }

        
    };
    

}