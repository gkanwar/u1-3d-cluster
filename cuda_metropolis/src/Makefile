CUDA_HOME ?= /opt/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS := -arch=sm_60
ifeq ($(dbg),1)
	NVCCFLAGS += -g -G
else
	NVCCFLAGS += -DNDEBUG -O3
endif

ALL_CCFLAGS := $(NVCCFLAGS)
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_CCFLAGS += --threads 0 --std=c++11 -O3

INCLUDES :=
LIBRARIES := -lcurand


all: build

build: u1_3d_cuda u1_3d_wloop_cuda

main.o: main.cpp config.h util.h args.hxx
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
main_wloop.o: main_wloop.cpp config.h util.h args.hxx
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
metropolis.o: metropolis.cu config.h util.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
u1_3d_cuda: main.o metropolis.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)
u1_3d_wloop_cuda: main_wloop.o metropolis.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f u1_3d_cuda u1_3d_wloop_cuda \
	main.o main_wloop.o metropolis.o
