#include"../include/MemManager.cuh"
#include<iostream>
#undef NDEBUG
#include"assert.h"
#include <cstring>

using namespace std;
namespace DNN_FrameWork{
    class Test_MemManager
    {
    public:
        Test_MemManager(){
            cout<<"============Testing MemManager=================="<<endl;
        }
        void Test_initialzation(){
            cout<<"TEST 1 => TEST initialization";
            size_t size_ = 100*sizeof(int);
            MemManager *mem = new MemManager(size_);
            assert(size_ == mem->get_size());
            assert(mem->get_state()==MemManager::UNINITIALIZED);

            free(mem);
            
            cout<<" PASSED"<<endl;
        }
        void test_CPU_allocation(){
            cout<<"TEST 2 => TEST CPU allocation";
            size_t size_ = 100*sizeof(int);
            MemManager *mem = new MemManager(size_);
            const void * const_data = mem->cpu_data();
            assert(const_data!=NULL);
            void * data = mem->mutable_cpu_data();
            assert(data!=NULL);
            mem->set_cpu_data(data);
            assert(mem->get_state()==MemManager::IN_CPU);
            free(mem);
            
            cout<<" PASSED"<<endl;
        }

        void test_GPU_allocation(){
            cout<<"TEST 3 => TEST GPU allocation";
            size_t size_ = 100*sizeof(int);
            MemManager *mem = new MemManager(size_);
            const void * const_data = mem->gpu_data();
            assert(const_data!=NULL);
            void * data = mem->mutable_gpu_data();
            assert(data!=NULL);
            mem->set_gpu_data(data);
            assert(mem->get_state()==MemManager::IN_GPU);
            free(mem);
            
            cout<<" PASSED"<<endl;
        }
        void test_cpu_Write(){
            cout<<"TEST 4 => TEST CPU read";

            size_t size_ = 100*sizeof(int);
            MemManager *mem = new MemManager(size_);
            void * data_alloc = malloc(size_);
            memset(data_alloc,2,size_);


            
            mem->set_cpu_data(data_alloc);
            const void * const_data = mem->cpu_data();
            assert(const_data!=NULL);
            assert(mem->get_state()==MemManager::IN_CPU);
            
            for(int i = 0 ;i<100;i++){
                assert(((char*)const_data)[i]==2);
            }

            void * data = mem->mutable_cpu_data();
            assert(data!=NULL);
            free(mem);
            
            cout<<" PASSED"<<endl;
        }

        void test_gpu_Write(){
            cout<<"TEST 5 => TEST GPU read";

            size_t size_ = 100*sizeof(int);
            MemManager *mem = new MemManager(size_);
            void * data_alloc = malloc(size_);
            memset(data_alloc,2,size_);
            void * test_data=malloc(size_);


            
            mem->set_cpu_data(data_alloc);
            void * dev_data = mem->mutable_gpu_data();
            assert(dev_data!=NULL);
            //cout<<mem->get_state()<<"="<<MemManager::IN_CPU<<endl;
            assert(mem->get_state()==MemManager::IN_GPU);
            mem->DNNcudaMemCpyToHost(test_data,size_,dev_data);

            for(int i = 0 ;i<100;i++){
                assert(((char*)test_data)[i]==2);
            }

            void * data = mem->mutable_cpu_data();
            assert(data!=NULL);
            free(mem);
            
            cout<<" PASSED"<<endl;
        }


        void RUN_TESTS(){
            Test_initialzation();
            test_CPU_allocation();
            test_GPU_allocation();
            test_cpu_Write();
            test_gpu_Write();
        }

        

    };
}
