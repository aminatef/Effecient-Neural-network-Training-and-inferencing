#include <cstdlib>
#include"../include/MemManager.cuh"
using namespace std;
namespace DNN_FrameWork
{
    MemManager::MemManager(size_t size):cpu_ptr(NULL), gpu_ptr(NULL) ,size(size) , state(UNINITIALIZED) {}
    MemManager::MemManager():cpu_ptr(NULL), gpu_ptr(NULL) ,size(0) , state(UNINITIALIZED) {}

    MemManager::~MemManager(){
        if(this->state == IN_CPU)
            free(this->cpu_ptr);
        else if (this->state == IN_GPU)
            cudaFree(this->gpu_ptr);

        
    }
    const void * MemManager::gpu_data(){
        to_gpu();
        state = IN_GPU;
        return (const void*)this->gpu_ptr;
    }

    const void * MemManager::cpu_data(){
        to_cpu();
        state = IN_CPU;
        return (const void*)this->cpu_ptr;
    }
    void MemManager::freeGPU(){
        cudaFree(this->gpu_ptr);
    }
    void MemManager::freeCPU(){
        free(this->cpu_ptr);
    }
    void MemManager::to_gpu(){
        switch (state)
        {
        case UNINITIALIZED:
            cudaMalloc((void**)&this->gpu_ptr,size);
            break;
        case IN_CPU:
            if(gpu_ptr==NULL){
                cudaMalloc((void**)&this->gpu_ptr,this->size);
            }
            cudaMemcpy(gpu_ptr,cpu_ptr,size,cudaMemcpyHostToDevice);
            break;
        case IN_GPU:
            break;
        default:
            break;
        }

    }
    void MemManager::to_cpu(){
        switch (state)
        {
        case UNINITIALIZED:
            this->cpu_ptr = malloc(this->size);
            break;
        case IN_CPU:
            break;
        case IN_GPU:
            if(cpu_ptr==NULL){
                this->cpu_ptr=malloc(size);
            }
            cudaMemcpy(cpu_ptr,gpu_ptr,size,cudaMemcpyDeviceToHost);
            break;
        default:
            break;
        }
        
    }
    void * MemManager::mutable_gpu_data(){
        to_gpu();
        state = IN_GPU;
        return (void*)this->gpu_ptr;
    }
    void * MemManager::mutable_cpu_data(){
        to_cpu();
        state = IN_CPU;
        return (void*)this->cpu_ptr;
    }
    void MemManager::set_cpu_data(void* data){
        if(cpu_ptr!=NULL){
            free(cpu_ptr);
        }
        cpu_ptr = data;
        state = IN_CPU;
    }
    void MemManager::set_gpu_data(void* data){
        if(gpu_ptr!=NULL){
            cudaFree(gpu_ptr);
        }
        gpu_ptr = data;
        state = IN_GPU;
    }
    void MemManager::DNNcudaMemCpyToHost(void*HostPtr,size_t bytes,void * devPtr){
        cudaMemcpy(HostPtr,devPtr,bytes,cudaMemcpyDeviceToHost);
    }
    void MemManager::DNNcudaMemCpyToDevice(void * devPtr,size_t bytes,void*HostPtr){
        cudaMemcpy(devPtr,HostPtr,bytes,cudaMemcpyHostToDevice);
    }

    void MemManager::DNNcudaMemCpyDefault(void * devPtr,size_t bytes,void*HostPtr){
        cudaMemcpy(devPtr,HostPtr,bytes,cudaMemcpyDefault);
    }
    
} // namespace DNN_FrameWork

