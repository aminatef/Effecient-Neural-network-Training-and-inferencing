#ifndef MEM_MANGER_HPP
#define MEM_MANGER_HPP
#include <cstdlib>
namespace DNN_FrameWork{
    class MemManager
    {
    public:
        enum MemoryState {UNINITIALIZED,IN_CPU,IN_GPU};
        MemManager();
        MemManager(size_t size);
        ~MemManager();
        void set_size(size_t size){this->size=size;}
        size_t get_size(){return size;}
        const void * gpu_data();
        const void * cpu_data();
        void * mutable_gpu_data();
        void * mutable_cpu_data();
        void set_cpu_data(void* data);
        void set_gpu_data(void* data);
        static void DNNcudaMemCpyToHost(void*HostPtr,size_t bytes,void * devPtr);
        static void DNNcudaMemCpyToDevice(void * devPtr,size_t bytes,void*HostPtr);
        static void DNNcudaMemCpyDefault(void * devPtr,size_t bytes,void*HostPtr);
        void freeGPU();
        void freeCPU();
        MemoryState get_state(){return state;}
    private:
        void * gpu_ptr;
        void * cpu_ptr;
        size_t size;
        MemoryState state;
        void to_gpu();
        void to_cpu();
    };
}
#endif



