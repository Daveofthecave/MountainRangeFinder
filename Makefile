# Makefile for MountainRangeFinder
# Compiles CPU C code with gcc, C++ code with g++, and GPU CUDA code with nvcc.
# 
#   make            -> builds the searcher (mountain_rangefinder)
#   make selftest   -> builds the validation harness (selftest)
#   make test       -> builds and runs the harness
#   make clean      -> removes all build artifacts
#
# NOTE: no automatic header dependency tracking. After editing noise_common.h,
# common.h, random.h, gpu.h, or cpu.h, run `make clean` before rebuilding.

# Compiler executables
NVCC = nvcc
CXX  = g++
CC   = gcc

# Target executable name
TARGET = mountain_rangefinder

SELFTEST = selftest

# Directory for the compiled executable
BUILD_DIR = build

# C++/CUDA Compiler flags
CXXFLAGS  = -O3 -std=c++17 -pthread -Wall
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -pthread

# Optional: FM=1 enables -use_fast_math for the CUDA pipeline.
# This sets denormals-flush-to-zero, -prec-div=false, -prec-sqrt=false,
# and implies --fmad=true. It does not affect the CPU verifier (math.h),
# so the CPU's double-precision ground truth remains unchanged.
# After enabling, run `make test` to confirm the selftest tolerance (~1e-4).
ifeq ($(FM),1)
NVCCFLAGS += --use_fast_math
endif

# C compiler flags
# -D_POSIX_C_SOURCE=199309L exposes POSIX functions like clock_gettime
CFLAGS    = -O3 -std=c11 -Wall -Wno-unused-result -D_POSIX_C_SOURCE=199309L

# CUDA Architecture: SASS for sm_80 (loads natively on all sm_8x cards:
# Ampere 30xx, Ada 40xx) plus embedded PTX (compute_89) so newer architectures
# (e.g. Blackwell 50xx) can JIT-compile it at first launch via the driver.
# Costs a one-time JIT delay on newer cards; no runtime cost afterwards.
NVCCFLAGS += -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89

# Cubiomes Library setup
CUBIOMES_DIR = cubiomes

# Exclude tests.c to avoid multiple definition of `main` errors
CUBIOMES_SRCS = $(filter-out $(CUBIOMES_DIR)/tests.c, $(wildcard $(CUBIOMES_DIR)/*.c))
CUBIOMES_OBJS = $(patsubst $(CUBIOMES_DIR)/%.c,$(BUILD_DIR)/%.o,$(CUBIOMES_SRCS))

# Include directories
INCLUDES = -I. -I$(CUBIOMES_DIR)

# Source files for our project
APP_CPU_SRCS = main.cpp cpu.cpp probe.cpp
APP_GPU_SRCS = gpu.cu

# Object files
APP_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(APP_CPU_SRCS)) \
           $(patsubst %.cu,$(BUILD_DIR)/%.o,$(APP_GPU_SRCS))

# Default target
all: $(TARGET)

# Link everything together
$(TARGET): $(APP_OBJS) $(CUBIOMES_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) -lpthread

$(SELFTEST): $(BUILD_DIR)/selftest.o $(CUBIOMES_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lpthread

test: $(SELFTEST)
	./$(SELFTEST)

# End-to-end regression: the known-good magic seed must survive the full
# GPU -> CPU funnel. Requires a CUDA device.
# Comment out the recall: block below to disable this test.
recall: $(TARGET)
	@rm -f recall_test.txt
	./$(TARGET) --seeds test_seeds.txt --output recall_test.txt
	@n=$$(grep -vc '^#' recall_test.txt); \
	echo "recall: $$n verified row(s)"; \
	test "$$n" -ge 1 && echo "RECALL OK" || { echo "RECALL FAIL"; exit 1; }

# Rule for compiling CPU C++ files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling GPU CUDA files
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling Cubiomes C files
$(BUILD_DIR)/%.o: $(CUBIOMES_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(SELFTEST)

# Python visualization shim (viz/make_*.py need viz/libmrfshim.so)
vizshim:
	$(CC) -O2 -fPIC -shared -I $(CUBIOMES_DIR) viz/vizshim.c $(CUBIOMES_SRCS) -o viz/libmrfshim.so -lm

# Release build: static CUDA runtime + static libstdc++/libgcc, so the binary
# only needs the NVIDIA *driver* (libcuda.so), not the toolkit. glibc stays
# dynamic -- build on the oldest distro you intend to support (a container is
# fine) and the binary will run anywhere newer.
release: NVCCFLAGS += --cudart static
release: LDFLAGS += -static-libstdc++ -static-libgcc
release: $(TARGET)
	strip $(TARGET)

.PHONY: all clean test recall vizshim release
