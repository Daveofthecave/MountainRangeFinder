// selftest.cu
// Validation harness. Builds as a separate binary (via `make selftest`) so the
// searcher itself is never touched.
//
// WHAT IT DOES
//   For a handful of (seed, point) pairs, it computes erosion,
//   continentalness, and temperature three ways:
//     1. cubiomes ground truth : setClimateParaSeed + sampleClimatePara,
//        full octaves (nmax = -1), double precision;
//     2. host code path        : init_mountain_noise + sample_climate from
//        noise_common.h, float precision;
//     3. device code path      : the same functions compiled for CUDA.
//   Because all project code shares noise_common.h, an exact host match means
//   the exact same math runs on the GPU; the device check additionally covers
//   nvcc-specific issues (constexpr tables, __float2int_rd, shared/global
//   memory, launch wiring).
//
// WHY
//   Both bugs you hit (missing /4 coord scale; 8/7 temperature normalization)
//   would have failed this test in under a second. Run it after ANY edit to
//   noise_common.h or gpu.cu.
//
// HOW TO READ THE OUTPUT
//   Errors print in raw noise units and in "CV units" (x10000), so you can
//   compare directly against Cubiomes-Viewer's climate display at block
//   (4x, 4z). Expected: ~1e-4 raw (~1 CV unit) -- pure float32-vs-double
//   noise. Anything above ~0.005 raw (50 CV) means a real bug.
//   Use --var to print every sample individually.

#include "noise_common.h"

#include "biomenoise.h" // cubiomes ground truth (via -Icubiomes)

#include <cuda_runtime.h>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(expr) check_cuda(expr, __FILE__, __LINE__)
static void check_cuda(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Defined in gpu.cu; selftest links against the same object.
void gpu_set_blob_cfg(float ero_max, float cont_min, float temp_min, float temp_max,
                      uint32_t target_cells);
void gpu_set_mc(int mc_version);

// Device-side copies of the shared configs. Built from the same noise_common.h
// expressions as the host copies, so they are identical by construction.
__device__ constexpr GradDotTable    D_GRAD = make_grad_dot_table();
__device__ constexpr ClimateConfig<4> D_ERO  = make_climate_config<5, 4>(-9,  EROSION_AMPS,         HASH_EROSION);
__device__ constexpr ClimateConfig<9> D_CONT = make_climate_config<9, 9>(-9,  CONTINENTALNESS_AMPS, HASH_CONTINENTALNESS);
__device__ constexpr ClimateConfig<2> D_TEMP = make_climate_config<6, 2>(-10, TEMPERATURE_AMPS,     HASH_TEMPERATURE);

__global__ void sample_kernel(uint64_t seed, int32_t xq, int32_t zq, float *out3) {
    if (threadIdx.x != 0) return;
    MountainNoiseResult r; // ~8 KiB of local memory; fine for a test kernel
    init_mountain_noise(r, seed, D_ERO, D_CONT, D_TEMP);
    out3[0] = sample_climate(D_GRAD, r.erosion_a, r.erosion_b, D_ERO,  xq, zq);
    out3[1] = sample_climate(D_GRAD, r.cont_a,    r.cont_b,    D_CONT, xq, zq);
    out3[2] = sample_climate(D_GRAD, r.temp_a,    r.temp_b,    D_TEMP, xq, zq);
}

int main(int argc, char **argv) {
    const bool verbose = (argc > 1 && std::strcmp(argv[1], "--var") == 0);

    const uint64_t SEEDS[] = {
        1ULL,
        8608349212744664511ULL,
        (uint64_t)(-1234567890123456789LL),
        0x5DEECE66DULL,
        9223372036854775783ULL
    };
    // Block coordinates, all divisible by 4; tested at quart = block/4.
    const int32_t PTS[][2] = {
        { 0, 0 }, { 4000, -4000 }, { -1000, 2000 }, { 124, -456 }, { 776, 332 }
    };
    constexpr double TOL = 0.005; // raw; expected actual ~1e-4

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) device_count = 0;
    const bool have_gpu = device_count > 0;

    float *d_out = nullptr;
    if (have_gpu) CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(float)));

    const char *names[3] = { "erosion", "continentalness", "temperature" };
    double max_err_host[3] = {};
    double max_err_dev[3]  = {};
    double max_err_trunc[4] = {}; // truncated samplers (the 0B/1B workhorse paths)

    std::printf("MountainRangeFinder selftest\n");
    std::printf("Ground truth: cubiomes setClimateParaSeed/sampleClimatePara (full octaves)\n");
    std::printf("GPU available: %s\n\n", have_gpu ? "yes" : "no (host-only check)");

    for (uint64_t seed : SEEDS) {
        MountainNoiseResult r;
        init_mountain_noise(r, seed, EROSION_CFG, CONTINENTALNESS_CFG, TEMPERATURE_CFG);
        init_weird_noise(r, seed, WEIRDNESS_CFG);

        for (const auto &pt : PTS) {
            const int32_t xq = pt[0] / 4;
            const int32_t zq = pt[1] / 4;

            BiomeNoise bn;
            setClimateParaSeed(&bn, seed, 0, NP_EROSION, -1);
            const double ref_e = sampleClimatePara(&bn, nullptr, xq, zq);
            setClimateParaSeed(&bn, seed, 0, NP_CONTINENTALNESS, -1);
            const double ref_c = sampleClimatePara(&bn, nullptr, xq, zq);
            setClimateParaSeed(&bn, seed, 0, NP_TEMPERATURE, -1);
            const double ref_t = sampleClimatePara(&bn, nullptr, xq, zq);
            const double ref[3] = { ref_e, ref_c, ref_t };

            const float host[3] = {
                sample_climate(GRAD_DOT_TABLE, r.erosion_a, r.erosion_b, EROSION_CFG,         xq, zq),
                sample_climate(GRAD_DOT_TABLE, r.cont_a,    r.cont_b,    CONTINENTALNESS_CFG, xq, zq),
                sample_climate(GRAD_DOT_TABLE, r.temp_a,    r.temp_b,    TEMPERATURE_CFG,     xq, zq),
            };

            float dev[3] = { 0, 0, 0 };
            if (have_gpu) {
                sample_kernel<<<1, 32>>>(seed, xq, zq, d_out);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(dev, d_out, sizeof(dev), cudaMemcpyDeviceToHost));
            }

            // Truncated samplers: the paths the pipeline actually exercises.
            // Ground truth: cubiomes with nmax (2 -> 0A+0B; 4 -> 0A+0B+1A+1B).
            {
                setClimateParaSeed(&bn, seed, 0, NP_EROSION, 2);
                const double r0 = sampleClimatePara(&bn, nullptr, xq, zq);
                const float h0 = sample_climate_trunc(GRAD_DOT_TABLE, r.erosion_a, r.erosion_b,
                                                      EROSION_CFG, 1, 1, xq, zq);
                const double e0 = std::fabs((double)h0 - r0);

                setClimateParaSeed(&bn, seed, 0, NP_CONTINENTALNESS, 4);
                const double r1 = sampleClimatePara(&bn, nullptr, xq, zq);
                const float h1 = sample_climate_trunc(GRAD_DOT_TABLE, r.cont_a, r.cont_b,
                                                      CONTINENTALNESS_CFG, 2, 2, xq, zq);
                const double e1 = std::fabs((double)h1 - r1);

                setClimateParaSeed(&bn, seed, 0, NP_TEMPERATURE, 2);
                const double r2 = sampleClimatePara(&bn, nullptr, xq, zq);
                const float h2 = sample_climate_trunc(GRAD_DOT_TABLE, r.temp_a, r.temp_b,
                                                      TEMPERATURE_CFG, 1, 1, xq, zq);
                const double e2 = std::fabs((double)h2 - r2);

                setClimateParaSeed(&bn, seed, 0, NP_WEIRDNESS, 4);
                const double r3 = sampleClimatePara(&bn, nullptr, xq, zq);
                const float h3 = sample_climate_trunc(GRAD_DOT_TABLE, r.weird_a, r.weird_b,
                                                      WEIRDNESS_CFG, 2, 2, xq, zq);
                const double e3 = std::fabs((double)h3 - r3);
                if (e3 > max_err_trunc[3]) max_err_trunc[3] = e3;

                if (e0 > max_err_trunc[0]) max_err_trunc[0] = e0;
                if (e1 > max_err_trunc[1]) max_err_trunc[1] = e1;
                if (e2 > max_err_trunc[2]) max_err_trunc[2] = e2;
                if (verbose)
                    std::printf("  trunc 0B-ero %.6f | 1B-cont %.6f | 0B-temp %.6f\n", e0, e1, e2);
            }

            for (int c = 0; c < 3; c++) {
                const double eh = std::fabs((double)host[c] - ref[c]);
                if (eh > max_err_host[c]) max_err_host[c] = eh;
                double ed = 0.0;
                if (have_gpu) {
                    ed = std::fabs((double)dev[c] - ref[c]);
                    if (ed > max_err_dev[c]) max_err_dev[c] = ed;
                }
                if (verbose) {
                    std::printf("seed %20" PRIu64 "  block (%6d, %6d)  %-15s ref %+10.6f  host %+10.6f (%8.4f)",
                                seed, pt[0], pt[1], names[c], ref[c], (double)host[c], eh);
                    if (have_gpu) std::printf("  dev %+10.6f (%8.4f)", (double)dev[c], ed);
                    std::printf("\n");
                }
            }
        }
    }

    bool ok = true;
    std::printf("\n%-16s %24s %24s\n", "climate", "host max |err|", "device max |err|");
    for (int c = 0; c < 3; c++) {
        std::printf("%-16s %10.6f (%7.2f CV)", names[c], max_err_host[c], max_err_host[c] * 10000.0);
        if (have_gpu) std::printf("   %10.6f (%7.2f CV)", max_err_dev[c], max_err_dev[c] * 10000.0);
        else          std::printf("   %24s", "(skipped)");
        std::printf("\n");
        if (max_err_host[c] > TOL || (have_gpu && max_err_dev[c] > TOL)) ok = false;
    }
    const char *tnames[4] = { "erosion 0B", "continentalness 1B", "temperature 0B", "weirdness (2,2)" };
    for (int c = 0; c < 4; c++) {
        std::printf("%-16s %10.6f (%7.2f CV)  (truncated path)\n",
                    tnames[c], max_err_trunc[c], max_err_trunc[c] * 10000.0);
        if (max_err_trunc[c] > TOL) ok = false;
    }

    if (d_out) cudaFree(d_out);

    std::printf("\nTolerance: %.3f raw (%.0f CV units). Result: %s\n",
                TOL, TOL * 10000.0, ok ? "PASS" : "FAIL");
    if (!ok)
        std::printf("  -> investigate seeding chain, hashes, normalization, and quart scaling.\n");
    return ok ? 0 : 1;
}
