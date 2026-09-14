// gpu.h
// Interface for the GPU worker thread.
// Each GpuThread manages a single CUDA device, pulling blocks of seeds from
// the SeedIterator, running the filtering kernels, and passing candidate seeds
// to the CPU verifier threads via the GpuOutputs queue.

#pragma once

#include "common.h"

struct GpuThread : Thread<GpuThread> {
    int device;                 // The CUDA device ID (e.g., 0 for first GPU)
    SeedIterator &input;        // Source for the next batch of seeds to process
    GpuOutputs &outputs;        // Thread-safe queue to push GPU candidates to the CPU

    // Set when run() returns (seed-list exhaustion in --seeds mode, or a stop
    // request). main.cpp uses this to detect pipeline completion.
    std::atomic_bool finished{false};
    std::atomic_uint64_t completed{0};   // seeds fully processed by this GPU

    // Constructor initializes the device ID and references, then starts the thread.
    GpuThread(int device, SeedIterator &input, GpuOutputs &outputs)
        : Thread(), device(device), input(input), outputs(outputs) {
        start();
    }

    // The main loop executed by the GPU thread.
    // Defined in gpu.cu.
    void run();
};
