#Get location of Tensorflow headers and library files
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB)
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart


SRC       = crop_op.cc
GPUSRC    = crop_op_gpu.cu.cc
PROD      = crop_op.so
GPUPROD = crop_op_cu.so



#Use this if building for Tensorflow 1.2 or 1.3
#TF_CFLAGS=-I$(TF_INC) -I$(TF_INC)/external/nsync/public
#Tensorflor 1.4 added libtensorflow_framework.so
# TF_CFLAGS=-I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -ltensorflow_framework

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)
