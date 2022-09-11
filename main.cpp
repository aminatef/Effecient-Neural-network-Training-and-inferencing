
#include"test/Test_MemManager.cpp"
#include <assert.h> 
#include<iostream>
using namespace std;
using namespace DNN_FrameWork;
int main(){
    DNN_FrameWork::Test_MemManager *test = new Test_MemManager();
    test->RUN_TESTS();
}