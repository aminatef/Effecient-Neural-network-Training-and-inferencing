
#include"test/Test_MemManager.cpp"
#include"test/Test_DataTensor.cpp"
#include <assert.h> 
#include<iostream>
using namespace std;
using namespace DNN_FrameWork;
int main(){
    DNN_FrameWork::Test_MemManager *test = new Test_MemManager();
    test->RUN_TESTS();
    DNN_FrameWork::Test_DataTensor *t2 = new Test_DataTensor();
    t2->RUN_TESTS();
}