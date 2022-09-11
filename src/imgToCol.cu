#include<iostream>
#include<assert.h>
#include "../include/imgToCol.cuh"


# define NUM_THREADS 512
template<typename Dtype>
__global__ void im2Col_kernel(const int n,const Dtype* data_im,
    const int height,const int width,const int  kernel_size, const int pad,const int stride,
    const int height_col,const int width_col,Dtype* data_col
){
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    for (int i = index ; i<n;i+=blockDim.x*gridDim.x){
        int h_idx = index / width_col;
        int h_col = h_idx % height_col;
        int w_col = index % width_col;
        int channel_img = h_idx/height_col;
        int channel_out = channel_img * kernel_size *kernel_size;
        int h_offset = h_col*stride - pad;
        int w_offset = w_col*stride - pad;
        Dtype * data_col_ptr = data_col;
        data_col_ptr += channel_out * height_col * width_col + h_col*width_col + w_col;
        const Dtype * data_im_ptr = data_im;
        data_im_ptr+= (channel_img * height + h_offset) *width + w_offset;
        for(int i = 0;i<kernel_size;i++){
            for(int j =0;j<kernel_size;j++){
                int h = i+h_offset;
                int w = j+w_offset;
                *data_col_ptr = (h>=0 && w>=0 && h<height && w<width)?
                                 data_im_ptr[i*width+j] : 0;

                data_col_ptr += height_col*width_col;
            }
        }
    }
}

template<typename Dtype>
void im2col_gpu(Dtype*im,
    int channels,int height,int width,
    int kernel_size,int stride,int pad,
    Dtype* data_col
){
    int height_col = (height+2*pad-kernel_size)/stride+1;
    int width_col = (width+2*pad-kernel_size)/stride+1;
    int num_kernels = channels*height_col*width_col;
    im2Col_kernel<int><<<(num_kernels+NUM_THREADS-1)/NUM_THREADS,NUM_THREADS>>>(num_kernels,
        im,height,width,
        kernel_size,pad,stride,
        height_col,width_col,data_col);
}




// int main(){
//     int * h_img,*h_img_t ;
//     int height = 10;
//     int width = 10;
//     int channels = 3;
//     int kernel_size= 3;
//     int stride = 1 ;
//     int pad = 1 ;
//     int height_col = (height + 2 * pad - kernel_size) / stride + 1;
//     int width_col = (width + 2 * pad - kernel_size) / stride + 1;
//     int data_col_size = kernel_size*kernel_size*channels*height_col*width_col;
//     size_t img_size = height*width*channels;
//     h_img = new int[img_size];
//     h_img_t = new int[img_size];
//     for (int i = 0 ;i<img_size;i++){
//         h_img[i] = rand()%255;
//     }


//     int * dev_img,*col_img;
//     cudaMalloc((void**)&dev_img,img_size*sizeof(int));
//     cudaMalloc((void**)&col_img,data_col_size*sizeof(int));

//     cudaMemcpy(dev_img,h_img,img_size*sizeof(int),cudaMemcpyHostToDevice);
//     im2col_gpu<int>(dev_img,channels,height,width,kernel_size,stride,pad,col_img);
//     int * t1;
//     t1 = new int[img_size];
//     cudaMemcpy(t1,col_img,img_size*sizeof(int),cudaMemcpyDeviceToHost);

//     for(int i=0;i<img_size;i++){
//         std::cout<<t1[i]<<std::endl;
//     }
// }




