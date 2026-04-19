NVCC       = /usr/local/cuda-13.2/bin/nvcc
CXX        = g++
CUDA_PATH  = /usr/local/cuda-13.2

NVCC_FLAGS = -O3 -arch=sm_120 --expt-relaxed-constexpr
CXX_FLAGS  = -O2 -std=c++17
SDL_FLAGS  = $(shell pkg-config --cflags sdl2 SDL2_ttf)
SDL_LIBS   = $(shell pkg-config --libs sdl2 SDL2_ttf)
CUDA_INC   = -I$(CUDA_PATH)/include
CUDA_LIB   = -L$(CUDA_PATH)/lib64 -lcudart -lcurand

TARGET = evosim_cuda

all: $(TARGET)

sim.o: sim.cu types.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) -c sim.cu -o sim.o

main.o: main.cpp types.h
	$(CXX) $(CXX_FLAGS) $(SDL_FLAGS) $(CUDA_INC) -c main.cpp -o main.o

$(TARGET): sim.o main.o
	$(CXX) -O2 -o $(TARGET) sim.o main.o $(SDL_LIBS) $(CUDA_LIB)

clean:
	rm -f *.o $(TARGET)
