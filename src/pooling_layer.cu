template<typename Dtype>
__global__ void maxPooling_forward_kernel(int numElements,
        int in_h,int in_w,int in_c,
        int stride,int pad,int pooling_size,
        Dtype * input,Dtype* output,
        int *indexes
){
    int out_h = (in_h+pad-pooling_size)/stride + 1;
    int out_w = (in_w+pad-pooling_size)/stride + 1;
    int out_c = in_c;

    int threadIndex = (blockDim.x*gridDim.x)*blockIdx.y+blockIdx.x*blockDim.x+threadIdx.x;

    if(threadIndex>=numElements)
        return;
    int j = threadIndex % out_w;
    threadIndex/=out_w;
    int i = threadIndex % out_h;
    threadIndex/=out_h;
    int k = threadIndex % out_c; 
    threadIndex/=out_c;
    int b = threadIndex;

    int out_idx = j + w*(i+k*h+h*out_c*b);
    Dtype max = -INFINITY;
    int max_idx =-1;
    for(int x=0;x<pooling_size;x++){
        for(int y=0;y<pooling_size;y++){
            int height = i*stride - pad/2 + x;
            int width  = j*stride - pad/2 + y;
            int index = (height+in_c*in_h+in_h*in_c*b)*in_w + width;
            bool cond = height>=0 && height<in_h && width>=0 && width<in_w;
            Dtype val = (cond)? input[index] : -INFINITY;
            max_idx = (val>max) index:max_idx;
            max = (val>max) val:max;
        }   
    }
    output[out_idx] = max;
    indexes[out_idx] = max_idx;
}
__global__ void maxPooling_GPU(){

}