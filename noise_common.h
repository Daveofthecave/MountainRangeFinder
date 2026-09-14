// noise_common.h
// Shared climate-noise implementation used by both the CUDA kernels (gpu.cu)
// and the validation harness (selftest.cu). Single source of truth: host and
// device call the same functions, so the two paths cannot diverge.
//
// COORDINATE CONVENTION:
//   All samplers take quart coordinates: quart = block / 4.
//   Minecraft 1.18+ evaluates the biome climate maps at 1:4 ("quart")
//   resolution. See genBiomeNoise3D()/getVoronoiSrcRange() in cubiomes'
//   biomenoise.c, and COMMISSION's KernelFilter1 ("noise (1:4) coords",
//   multiplying by 4 at output handoff).
//
// FIDELITY NOTES:
//   * Seeding chain, octave MD5 hashes, lacunarity/persistence and the
//     (5/3)*L/(L+1) normalization replicate cubiomes' xDoublePerlinInit() +
//     xOctaveInit() exactly. L is the ZERO-TRIMMED amplitude count
//     (temperature {1.5,0,1,0,0,0} -> L=3 -> amp 5/4, not 10/7).
//   * One deliberate approximation, an idea inspired from COMMISSION (which
//     demonstrated it works): init_noise() draws offsets with nextFloat() 
//     instead of nextDouble(). Both consume exactly one RNG output 
//     (identical state chain); the offset differs by < 1.5e-5 noise cells, 
//     irrelevant at biome scales. The selftest quantifies the total effect 
//     empirically.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

#include <math.h>

#include "random.h"

#if defined(__CUDACC__)
#define PRAGMA_UNROLL _Pragma("unroll")
#else
#define PRAGMA_UNROLL
#endif

// ---------------------------------------------------------------------------
// Gradient direction table for ImprovedNoise.
// Equivalent to cubiomes' indexedLerp(idx, a, b, c) with a->x, b->y, c->z.
// ---------------------------------------------------------------------------
struct GradDotTable {
    float x[16];
    float y[16];
    float z[16];
};

constexpr GradDotTable make_grad_dot_table() {
    GradDotTable t{};
    t.x[0]  =  1; t.y[0]  =  1; t.z[0]  =  0;  // case 0:   a + b
    t.x[1]  = -1; t.y[1]  =  1; t.z[1]  =  0;  // case 1:  -a + b
    t.x[2]  =  1; t.y[2]  = -1; t.z[2]  =  0;  // case 2:   a - b
    t.x[3]  = -1; t.y[3]  = -1; t.z[3]  =  0;  // case 3:  -a - b
    t.x[4]  =  1; t.y[4]  =  0; t.z[4]  =  1;  // case 4:   a + c
    t.x[5]  = -1; t.y[5]  =  0; t.z[5]  =  1;  // case 5:  -a + c
    t.x[6]  =  1; t.y[6]  =  0; t.z[6]  = -1;  // case 6:   a - c
    t.x[7]  = -1; t.y[7]  =  0; t.z[7]  = -1;  // case 7:  -a - c
    t.x[8]  =  0; t.y[8]  =  1; t.z[8]  =  1;  // case 8:   b + c
    t.x[9]  =  0; t.y[9]  = -1; t.z[9]  =  1;  // case 9:  -b + c
    t.x[10] =  0; t.y[10] =  1; t.z[10] = -1;  // case 10:  b - c
    t.x[11] =  0; t.y[11] = -1; t.z[11] = -1;  // case 11: -b - c
    t.x[12] =  1; t.y[12] =  1; t.z[12] =  0;  // case 12:  a + b
    t.x[13] =  0; t.y[13] = -1; t.z[13] =  1;  // case 13: -b + c
    t.x[14] = -1; t.y[14] =  1; t.z[14] =  0;  // case 14: -a + b
    t.x[15] =  0; t.y[15] = -1; t.z[15] = -1;  // case 15: -b - c
    return t;
}

constexpr GradDotTable GRAD_DOT_TABLE = make_grad_dot_table();

// ---------------------------------------------------------------------------
// MD5-derived fork hashes, identical to cubiomes' biomenoise.c.
// ---------------------------------------------------------------------------
constexpr XrsrForkHash HASH_CONTINENTALNESS { 0x83886c9d0ae3a662ULL, 0xafa638a61b42e8adULL }; // "minecraft:continentalness"
constexpr XrsrForkHash HASH_EROSION         { 0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL }; // "minecraft:erosion"
constexpr XrsrForkHash HASH_TEMPERATURE     { 0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL }; // "minecraft:temperature"
constexpr XrsrForkHash HASH_WEIRDNESS       { 0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL }; // "minecraft:ridge"

// md5 "octave_-12" .. "octave_0", identical to xOctaveInit()'s md5_octave_n.
constexpr XrsrForkHash HASH_OCTAVE[13] = {
    { 0xb198de63a8012672ULL, 0x7b84cad43ef7b5a8ULL }, // octave_-12
    { 0x0fd787bfbc403ec3ULL, 0x74a4a31ca21b48b8ULL }, // octave_-11
    { 0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL }, // octave_-10
    { 0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL }, // octave_-9
    { 0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL }, // octave_-8
    { 0xf11268128982754fULL, 0x257a1d670430b0aaULL }, // octave_-7
    { 0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL }, // octave_-6
    { 0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL }, // octave_-5
    { 0xbd90d5377ba1b762ULL, 0xc07317d419a7548dULL }, // octave_-4
    { 0x53d39c6752dac858ULL, 0xbcd1c5a80ab65b3eULL }, // octave_-3
    { 0xb4a24d7a84e7677bULL, 0x023ff9668e89b5c4ULL }, // octave_-2
    { 0xdffa22b534c5f608ULL, 0xb9b67517d3665ca9ULL }, // octave_-1
    { 0xd50708086cef4d7cULL, 0x6e1651ecc7f43309ULL }, // octave_0
};

// ---------------------------------------------------------------------------
// Noise table storage. 272 bytes, 16-byte aligned.
// ---------------------------------------------------------------------------
struct alignas(16) ImprovedNoise {
    uint8_t p[256];
    float xo, yo, zo, pad;
};
static_assert(sizeof(ImprovedNoise) == 272, "ImprovedNoise layout");

struct OctaveConfig {
    XrsrForkHash fork_hash; // octave MD5 hash, XORed into the octave's RNG seed
    double input_factor;    // lacunarity = 2^(first_octave + i)
    double value_factor;    // total_amp * amplitude[i] * persistence_i
};

// M = number of nonzero amplitudes (zero-amplitude octaves are never
// stored, initialized, or sampled; they consume no work).
template <size_t M>
struct ClimateConfig {
    XrsrForkHash fork_hash; // climate-level MD5 hash
    OctaveConfig a[M];      // Perlin-A octaves
    OctaveConfig b[M];      // Perlin-B octaves (sampled at 337/331 x frequency)
};

// Replicates cubiomes' xDoublePerlinInit() + xOctaveInit() scaling exactly:
//   persist_i  = 2^(N-1-i) / (2^N - 1)   (N = FULL amplitudes length; zero
//                                          entries still advance lacunarity
//                                          and persistence)
//   total_amp  = (5/3) * L / (L+1), L = trimmed length (leading/trailing
//                zero amplitudes removed)
//   B octave i is sampled at (337/331) * the A frequency (cubiomes' "f").
template <size_t N, size_t M>
constexpr ClimateConfig<M> make_climate_config(int32_t first_octave,
        const double (&amplitudes)[N], const XrsrForkHash &fork_hash) {
    ClimateConfig<M> cfg{};
    cfg.fork_hash = fork_hash;

    int first = 0;
    while (amplitudes[first] == 0.0) first++;
    int last = (int)N - 1;
    while (amplitudes[last] == 0.0) last--;
    const int trimmed_len = last - first + 1;

    const double total_amp = (5.0 / 3.0) * (double)trimmed_len / (double)(trimmed_len + 1);
    double input_factor = 1.0 / (double)(1 << -first_octave);
    double persist      = (double)(1u << (N - 1)) / (double)((1u << N) - 1u);

    size_t n = 0;
    for (size_t i = 0; i < N; i++) {
        if (amplitudes[i] != 0.0) {
            const OctaveConfig oc {
                HASH_OCTAVE[first_octave + 12 + (int32_t)i],
                input_factor,
                total_amp * amplitudes[i] * persist
            };
            cfg.a[n] = oc;
            cfg.b[n] = { oc.fork_hash, input_factor * (337.0 / 331.0), oc.value_factor };
            n++;
        }
        input_factor *= 2.0;
        persist      *= 0.5;
    }
    return cfg;
}

// 1.18+ amplitude tables (from cubiomes' init_climate_seed).
constexpr double EROSION_AMPS[5]         = { 1.0, 1.0, 0.0, 1.0, 1.0 };
constexpr double CONTINENTALNESS_AMPS[9] = { 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0 };
constexpr double TEMPERATURE_AMPS[6]     = { 1.5, 0.0, 1.0, 0.0, 0.0, 0.0 };
constexpr double WEIRDNESS_AMPS[6]       = { 1.0, 2.0, 1.0, 0.0, 0.0, 0.0 };

// Canonical host-side constexpr configs. gpu.cu/selftest.cu create matching
// __device__ constexpr copies from these same expressions.
constexpr ClimateConfig<4> EROSION_CFG         = make_climate_config<5, 4>(-9,  EROSION_AMPS,         HASH_EROSION);
constexpr ClimateConfig<9> CONTINENTALNESS_CFG = make_climate_config<9, 9>(-9,  CONTINENTALNESS_AMPS, HASH_CONTINENTALNESS);
constexpr ClimateConfig<2> TEMPERATURE_CFG     = make_climate_config<6, 2>(-10, TEMPERATURE_AMPS,     HASH_TEMPERATURE);
constexpr ClimateConfig<3> WEIRDNESS_CFG       = make_climate_config<6, 3>(-7,  WEIRDNESS_AMPS,       HASH_WEIRDNESS);

// Compile-time tripwires against config regressions (exact powers of two;
// value_factor expressions mirror the maker's exact operation order).
static_assert(EROSION_CFG.a[0].input_factor == 1.0 / 512.0, "erosion octave 0 lacunarity");
static_assert(EROSION_CFG.a[2].input_factor == 1.0 / 64.0,  "erosion amp idx 3 lacunarity (compaction map)");
static_assert(EROSION_CFG.a[3].input_factor == 1.0 / 32.0,  "erosion amp idx 4 lacunarity");
static_assert(CONTINENTALNESS_CFG.a[8].input_factor == 0.5, "continentalness octave 8 lacunarity");
static_assert(TEMPERATURE_CFG.a[1].input_factor == 1.0 / 256.0, "temperature amp idx 2 lacunarity");
static_assert(EROSION_CFG.a[0].value_factor ==
    (5.0 / 3.0) * 5.0 / 6.0 * 1.0 * (16.0 / 31.0), "erosion normalization");      // ~0.7168
static_assert(CONTINENTALNESS_CFG.a[0].value_factor ==
    (5.0 / 3.0) * 9.0 / 10.0 * 1.0 * (256.0 / 511.0), "continentalness normalization"); // ~0.7515
static_assert(TEMPERATURE_CFG.a[0].value_factor ==
    (5.0 / 3.0) * 3.0 / 4.0 * 1.5 * (32.0 / 63.0), "temperature normalization (trimmed len 3)"); // ~0.9524
static_assert(WEIRDNESS_CFG.a[0].input_factor == 1.0 / 128.0, "weirdness octave 0 lacunarity");
static_assert(WEIRDNESS_CFG.a[0].value_factor ==
    (5.0 / 3.0) * 3.0 / 4.0 * 1.0 * (32.0 / 63.0), "weirdness normalization"); // ~0.6349

// ---------------------------------------------------------------------------
// Xoroshiro seeding helpers (moved from gpu.cu; now usable on host too).
// ---------------------------------------------------------------------------

// The first two nextLong()s of xSetSeed(seed) -- i.e. the (xlo, xhi) base state
// that cubiomes' setBiomeSeed() XORs into every climate hash.
HOST_DEVICE inline XrsrRandomFork xrsr_seed_fork(uint64_t seed) {
    seed ^= XrsrRandom::XRSR_SILVER_RATIO;
    const uint64_t l  = XrsrRandom::mix64(seed);
    const uint64_t h  = XrsrRandom::mix64(seed + XrsrRandom::XRSR_GOLDEN_RATIO);
    const uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
    const uint64_t h1 = h ^ l;
    const uint64_t l2 = XrsrRandom::rol64(l, 49) ^ h1 ^ (h1 << 21);
    const uint64_t h2 = XrsrRandom::rol64(h1, 28);
    const uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;
    return { r1, r2 };
}

// Two consecutive nextLong() pairs: fork_a = octA (xlo,xhi), fork_b = octB's.
// Matches xDoublePerlinInit()'s draw order (octA first, then octB).
HOST_DEVICE inline void xrsr_double_fork(XrsrRandom &rng, XrsrRandomFork &fork_a, XrsrRandomFork &fork_b) {
    const uint64_t l = rng.lo, h = rng.hi;
    const uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
    const uint64_t h1 = h ^ l;
    const uint64_t l2 = XrsrRandom::rol64(l, 49) ^ h1 ^ (h1 << 21);
    const uint64_t h2 = XrsrRandom::rol64(h1, 28);
    const uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;
    const uint64_t h2b = h2 ^ l2;
    const uint64_t l3 = XrsrRandom::rol64(l2, 49) ^ h2b ^ (h2b << 21);
    const uint64_t h3 = XrsrRandom::rol64(h2b, 28);
    fork_a = { r1, r2 };
    const uint64_t r3 = XrsrRandom::rol64(l3 + h3, 17) + l3;
    const uint64_t h3b = h3 ^ l3;
    const uint64_t l4 = XrsrRandom::rol64(l3, 49) ^ h3b ^ (h3b << 21);
    const uint64_t h4 = XrsrRandom::rol64(h3b, 28);
    const uint64_t r4 = XrsrRandom::rol64(l4 + h4, 17) + l4;
    fork_b = { r3, r4 };
}

// Equivalent to cubiomes' xPerlinInit(). See header note re: nextFloat offsets.
HOST_DEVICE inline void init_noise(ImprovedNoise &noise, XrsrRandom random) {
    noise.xo = random.nextFloat() * 256.0f;
    noise.yo = random.nextFloat() * 256.0f;
    noise.zo = random.nextFloat() * 256.0f;
    for (uint32_t i = 0; i < 256; i++) noise.p[i] = (uint8_t)i;
    for (uint32_t i = 0; i < 256; i++) {
        const uint32_t j = random.nextInt(256 - i);
        const uint8_t b = noise.p[i];
        noise.p[i] = noise.p[i + j];
        noise.p[i + j] = b;
    }
}

// ---------------------------------------------------------------------------
// Sampling (float32; identical expression structure on host and device).
// ---------------------------------------------------------------------------
HOST_DEVICE inline float grad_dot(const GradDotTable &t, uint8_t p, float x, float y, float z) {
    const uint32_t h = p & 15u;
    return std::fmaf(x, t.x[h], std::fmaf(y, t.y[h], z * t.z[h]));
}

HOST_DEVICE inline float smoothstep5(float v) {
    return v * v * v * (v * (v * 6.0f - 15.0f) + 10.0f);
}

HOST_DEVICE inline float lerp1(float f, float a, float b) {
    return std::fmaf(f, b - a, a);
}

HOST_DEVICE inline int32_t floor_i32(float x) {
#if defined(__CUDA_ARCH__)
    return __float2int_rd(x);
#else
    return (int32_t)std::floor(x);
#endif
}

HOST_DEVICE inline float sample_noise(const GradDotTable &t, const ImprovedNoise &n,
        float x, float y, float z) {
    x += n.xo; y += n.yo; z += n.zo;
    const int32_t ix = floor_i32(x), iy = floor_i32(y), iz = floor_i32(z);
    const float fx0 = x - (float)ix, fy0 = y - (float)iy, fz0 = z - (float)iz;
    const uint8_t p0  = n.p[(ix)     & 0xFF];
    const uint8_t p1  = n.p[(ix + 1) & 0xFF];
    const uint8_t p00 = n.p[(p0 + iy)     & 0xFF];
    const uint8_t p01 = n.p[(p0 + iy + 1) & 0xFF];
    const uint8_t p10 = n.p[(p1 + iy)     & 0xFF];
    const uint8_t p11 = n.p[(p1 + iy + 1) & 0xFF];
    const float n000 = grad_dot(t, n.p[(p00 + iz)     & 0xFF], fx0,        fy0,        fz0);
    const float n100 = grad_dot(t, n.p[(p10 + iz)     & 0xFF], fx0 - 1.0f, fy0,        fz0);
    const float n010 = grad_dot(t, n.p[(p01 + iz)     & 0xFF], fx0,        fy0 - 1.0f, fz0);
    const float n110 = grad_dot(t, n.p[(p11 + iz)     & 0xFF], fx0 - 1.0f, fy0 - 1.0f, fz0);
    const float n001 = grad_dot(t, n.p[(p00 + iz + 1) & 0xFF], fx0,        fy0,        fz0 - 1.0f);
    const float n101 = grad_dot(t, n.p[(p10 + iz + 1) & 0xFF], fx0 - 1.0f, fy0,        fz0 - 1.0f);
    const float n011 = grad_dot(t, n.p[(p01 + iz + 1) & 0xFF], fx0,        fy0 - 1.0f, fz0 - 1.0f);
    const float n111 = grad_dot(t, n.p[(p11 + iz + 1) & 0xFF], fx0 - 1.0f, fy0 - 1.0f, fz0 - 1.0f);
    const float fx = smoothstep5(fx0), fy = smoothstep5(fy0), fz = smoothstep5(fz0);
    const float x00 = lerp1(fx, n000, n100);
    const float x10 = lerp1(fx, n010, n110);
    const float x01 = lerp1(fx, n001, n101);
    const float x11 = lerp1(fx, n011, n111);
    return lerp1(fz, lerp1(fy, x00, x10), lerp1(fy, x01, x11));
}

HOST_DEVICE inline float sample_octave(const GradDotTable &t, const ImprovedNoise &n,
        int32_t xq, int32_t yq, int32_t zq, const OctaveConfig cfg) {
    const float f = (float)cfg.input_factor;
    return sample_noise(t, n, xq * f, yq * f, zq * f) * (float)cfg.value_factor;
}

// Full double-perlin climate sample (all non-zero octaves, A + B), at quart
// coordinates (xq, zq).
template <size_t M>
HOST_DEVICE inline float sample_climate(const GradDotTable &t, const ImprovedNoise *a,
        const ImprovedNoise *b, const ClimateConfig<M> &cfg, int32_t xq, int32_t zq) {
    float v = 0.0f;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < M; i++) {
        v += sample_octave(t, a[i], xq, 0, zq, cfg.a[i]);
        v += sample_octave(t, b[i], xq, 0, zq, cfg.b[i]);
    }
    return v;
}

// Truncated variant: only the first kA A-octaves and first kB B-octaves.
// Cubiomes-Viewer octave-truncation equivalents for the compacted arrays:
//   "up to 0B" -> (1, 1)   "up to 1B" -> (2, 2)   "up to 1A" -> (2, 1)
// (The compacted arrays keep the non-zero amplitudes in order, so a[1] is
// always octave 1A -- e.g. for erosion, whose amplitude table has a zero at
// index 2, a[1] still carries octave 1A's hash, lacunarity and persistence.)
template <size_t M>
HOST_DEVICE inline float sample_climate_trunc(const GradDotTable &t, const ImprovedNoise *a,
        const ImprovedNoise *b, const ClimateConfig<M> &cfg, int kA, int kB, int32_t xq, int32_t zq) {
    float v = 0.0f;
    for (int i = 0; i < kA; i++) v += sample_octave(t, a[i], xq, 0, zq, cfg.a[i]);
    for (int i = 0; i < kB; i++) v += sample_octave(t, b[i], xq, 0, zq, cfg.b[i]);
    return v;
}

// ---------------------------------------------------------------------------
// Low-discrepancy disc sampling for the coverage kernels (phyllotaxis /
// golden-angle spiral). The table holds DISC_MAX_COVERS interleaved 32-point
// covers of the unit disc, int16-quantized. Each cover alone is a reasonable
// full-disc sampling, so kernels can treat covers as fail-fast batches of 32
// and still get an unbiased coverage estimate from any prefix of batches.
//
// How many covers each check actually uses (the rest of the table is spare
// capacity). Fewer points = faster scans with a slightly noisier estimate
// (binomial standard error ~3% around the core threshold). The GPU kernels
// and the CPU-side stats both read these, so they can never drift apart.
// ---------------------------------------------------------------------------
constexpr int DISC_COVER_POINTS = 32;
constexpr int DISC_MAX_COVERS   = 7;

constexpr int CORE_COVERS = 3; // core disc:   3 covers = 96 samples

constexpr int CORE_COV_SAMPLES = CORE_COVERS * DISC_COVER_POINTS; // 96
constexpr int CORE_COV_NEED    = 77;  // >= ~80% of 96
static_assert(CORE_COV_NEED * 5 >= CORE_COV_SAMPLES * 4, "core coverage is at least 80%");

struct DiscOffsets {
    int16_t x[DISC_MAX_COVERS * DISC_COVER_POINTS];
    int16_t z[DISC_MAX_COVERS * DISC_COVER_POINTS];
};

// Host-side generator; upload the result to device memory before use.
inline DiscOffsets make_disc_offsets() {
    DiscOffsets d;
    const float golden = 2.39996322972865332f; // pi * (3 - sqrt(5))
    for (int g = 0; g < DISC_MAX_COVERS; g++) {
        for (int j = 0; j < DISC_COVER_POINTS; j++) {
            const int k = g * DISC_COVER_POINTS + j;
            // radius within the cover (each cover spans the full disc) and
            // angle from the global index (gives each cover its own phase).
            const float r = std::sqrt((j + 0.5f) / (float)DISC_COVER_POINTS);
            const float a = (float)k * golden;
            d.x[k] = (int16_t)std::lrint(r * std::cos(a) * 32767.0f);
            d.z[k] = (int16_t)std::lrint(r * std::sin(a) * 32767.0f);
        }
    }
    return d;
}

// ---------------------------------------------------------------------------
// Per-seed noise state for the three climates the pipeline filters on.
// Zero-amplitude octaves are compacted out:
//   erosion: 8 tables | continentalness: 18 tables | temperature: 4 tables
// ---------------------------------------------------------------------------
struct MountainNoiseResult {
    ImprovedNoise erosion_a[4], erosion_b[4];
    ImprovedNoise cont_a[9],    cont_b[9];
    ImprovedNoise temp_a[2],    temp_b[2];
    ImprovedNoise weird_a[2],   weird_b[2]; // (2,2) truncation: octaves 0-1 of amplitude {1,2,1}
};
static_assert(sizeof(MountainNoiseResult) == 34 * sizeof(ImprovedNoise), "noise table size");

template <size_t M>
HOST_DEVICE inline void init_climate(ImprovedNoise *out_a, ImprovedNoise *out_b,
        const XrsrRandomFork &base, const ClimateConfig<M> &cfg) {
    XrsrRandom rng = base.from(cfg.fork_hash);
    XrsrRandomFork fa, fb;
    xrsr_double_fork(rng, fa, fb);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < M; i++) {
        init_noise(out_a[i], fa.from(cfg.a[i].fork_hash));
        init_noise(out_b[i], fb.from(cfg.b[i].fork_hash));
    }
}

// Builds exactly one octave table of a climate (A-side index i, or B-side when
// b_side is set), following the same fork chain as init_climate(). Lets the
// GPU spread table construction across many threads (KernelInit/KernelLateInit
// in gpu.cu) instead of building a whole climate serially per seed.
template <size_t M>
HOST_DEVICE inline void init_climate_table(ImprovedNoise *out, uint64_t seed,
        const ClimateConfig<M> &cfg, uint32_t i, bool b_side) {
    const XrsrRandomFork base = xrsr_seed_fork(seed);
    XrsrRandom rng = base.from(cfg.fork_hash);
    XrsrRandomFork fa, fb;
    xrsr_double_fork(rng, fa, fb);
    init_noise(*out, (b_side ? fb : fa).from((b_side ? cfg.b[i] : cfg.a[i]).fork_hash));
}

// One erosion table by flat index 0..7 (a[0..4) then b[0..4)).
HOST_DEVICE inline void init_erosion_table(MountainNoiseResult &r, uint64_t seed,
        const ClimateConfig<4> &ero, uint32_t table) {
    if (table < 4)
        init_climate_table(&r.erosion_a[table],     seed, ero, table,     false);
    else
        init_climate_table(&r.erosion_b[table - 4], seed, ero, table - 4, true);
}

// Staged initialization: the GPU pipeline only needs the erosion tables for
// its first filter pass, so continentalness/temperature are deferred to
// KernelLateInit (gpu.cu) and only initialized for seeds that survive stage 2.
HOST_DEVICE inline void init_erosion_noise(MountainNoiseResult &r, uint64_t seed,
        const ClimateConfig<4> &ero) {
    init_climate(r.erosion_a, r.erosion_b, xrsr_seed_fork(seed), ero);
}

HOST_DEVICE inline void init_cont_temp_noise(MountainNoiseResult &r, uint64_t seed,
        const ClimateConfig<9> &cont, const ClimateConfig<2> &temp) {
    const XrsrRandomFork base = xrsr_seed_fork(seed);
    init_climate(r.cont_a, r.cont_b, base, cont);
    init_climate(r.temp_a, r.temp_b, base, temp);
}

// Weirdness (2,2)-truncated pair for the GPU texture gates (KernelExtrema).
// Built per-table like KernelLateInit's lane mapping, so the same helper
// serves both the staged pipeline and the selftest harness.
HOST_DEVICE inline void init_weird_noise(MountainNoiseResult &r, uint64_t seed,
        const ClimateConfig<3> &weird) {
    init_climate_table(&r.weird_a[0], seed, weird, 0, false);
    init_climate_table(&r.weird_a[1], seed, weird, 1, false);
    init_climate_table(&r.weird_b[0], seed, weird, 0, true);
    init_climate_table(&r.weird_b[1], seed, weird, 1, true);
}

HOST_DEVICE inline void init_mountain_noise(MountainNoiseResult &r, uint64_t seed,
        const ClimateConfig<4> &ero, const ClimateConfig<9> &cont, const ClimateConfig<2> &temp) {
    init_erosion_noise(r, seed, ero);
    init_cont_temp_noise(r, seed, cont, temp);
}
