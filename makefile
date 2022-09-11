###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=-g
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test
CXX_SRCS := MaxPooling.cpp SGD.cpp DataTensor.cpp ActivationLayer.cpp LossLayer.cpp ConvLayer.cpp Network.cpp DenseLayer.cpp MathFunctions.cpp  Layer.cpp
CU_SRCS := pooling_layer.cu imgToCol.cu MemManager.cu

CXX_OBJS := $(addprefix $(OBJ_DIR)/, ${CXX_SRCS:.cpp=.o})

Cu_OBJS := $(addprefix $(OBJ_DIR)/, ${CU_SRCS:.cu=.o})


# str = /src
# to=
# CXX_OBJS := $(subst $(str) ,$(to) ,$(CU_SRCS))
# CU_OBJS := $(subst $(str), $(to),$(CXX_SRCS))

OBJS := $(CXX_OBJS) $(Cu_OBJS) main.o


# # Object files:
# OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o

##########################################################

## Compile ##

# # Link c++ and CUDA compiled object files to target executable:
# h:
# 	echo $(OBJS)
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.hpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
