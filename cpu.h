// cpu.h
// Interface for the CPU verifier threads.
// Each CpuThread pulls GPU candidate seeds from the GpuOutputs queue and
// performs the slow, exact 3D terrain height and biome checks using the
// Cubiomes C library. Fully verified seeds are pushed to CpuOutputs.

#pragma once

#include "common.h"
#include <vector>

// Preload the deduplication map with prior results (e.g. an existing 
// output.txt), so a resumed search doesn't re-emit points it already 
// reported. Call before spawning any CpuThread.
size_t dedup_preload(const std::vector<GpuOutput> &prior);

// Reject tallies (defined in cpu.cpp), displayed in the live status line.
extern std::atomic_uint64_t g_rej_dark, g_rej_area, g_rej_core, g_rej_height, g_rej_dedup;

struct CpuThread : Thread<CpuThread> {
    int id;                     // Thread identifier (for logging/debugging)
    const VerifyConfig &cfg;    // Verification tunables (owned by main)
    GpuOutputs &inputs;         // Queue of GPU candidates to verify
    CpuOutputs &outputs;        // Queue of fully verified seeds to save to disk

    // Candidates fully processed (pass or fail). main.cpp compares this
    // against GpuOutputs::total_pushed to detect a drained pipeline.
    std::atomic_uint64_t processed{0};

    // Constructor initializes the thread ID and references, then starts the thread.
    CpuThread(int id, const VerifyConfig &cfg, GpuOutputs &inputs, CpuOutputs &outputs)
        : Thread(), id(id), cfg(cfg), inputs(inputs), outputs(outputs) {
        start();
    }

    // The main loop executed by the CPU thread.
    // Defined in cpu.cpp.
    void run();
};
