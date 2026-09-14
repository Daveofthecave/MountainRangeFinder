// ===========================================================================
// random.h
// ===========================================================================
// Xoroshiro128++, the RNG Minecraft 1.18+ world generation runs on, in a
// shape that compiles unchanged for the CPU and for CUDA kernels.
//
// What MRF does with it:
//   * Every climate noise table (erosion, continentalness, temperature,
//     weirdness) is seeded through the fork chain in noise_common.h
//     (xrsr_seed_fork -> xrsr_double_fork -> XrsrRandomFork::from), which
//     replicates cubiomes' seeding chain call for call.
//   * XrsrRandom::mix64, the splitmix64 finalizer, is a bijection on 2^64.
//     That is the property --shuffled-search leans on: it maps a plain
//     counter through the mixer and thereby visits every seed exactly
//     once, in scrambled order, with no repeats and a resumable counter.
//
// Provenance: a restatement of cubiomes' MIT-licensed implementation
// (cubiomes/rng.h: Xoroshiro, xSetSeed, xNextLong, xNextInt, xNextFloat;
// the fork pattern is init_climate_seed in cubiomes/biomenoise.c). The
// class form and the HOST_DEVICE marker exist only so this one source file
// builds for both host and device. cubiomes' license text ships at
// cubiomes/LICENSE.
// ===========================================================================

#ifndef RANDOM_H_
#define RANDOM_H_

#include <cstdint>

// Marks a function as callable from both host code and CUDA kernels.
// noise_common.h relies on this file to define the macro. The outer check
// lets a translation unit predefine its own equivalent.
#ifndef HOST_DEVICE
#ifdef __CUDA_ARCH__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

// A "name hash": the MD5 of a climate or octave name (the concrete values
// live in noise_common.h's HASH_* tables). Forking an RNG means XORing one
// of these into a drawn (lo, hi) state pair, which is how a world seed
// gives every climate parameter its own independent noise stream.
struct XrsrForkHash {
    uint64_t lo;
    uint64_t hi;
};

struct XrsrRandom;

// A drawn (lo, hi) pair, ready to become a child generator via from().
struct XrsrRandomFork {
    uint64_t lo;
    uint64_t hi;

    HOST_DEVICE XrsrRandom from(const XrsrForkHash &hash) const;
};

// ===========================================================================
// XrsrRandom -- the generator and its helpers
// ===========================================================================
// Two 64-bit state words; each draw emits rotl(lo + hi, 17) + lo (the "++"
// output function) and then advances the state. The seeding, drawing, and
// forking below track cubiomes' functions line for line, so the GPU's float
// pipeline and cubiomes' double-precision reference can never disagree
// about what a seed produces.
//
// Naming note: the class deliberately is not called Xoroshiro, because
// cubiomes already defines a plain `struct Xoroshiro` in rng.h, and
// translation units like gpu.cu include both headers.
// ===========================================================================
struct XrsrRandom {
    // The splitmix64 finalizer constants (Minecraft: RandomSupport.mix64).
    constexpr static uint64_t XRSR_MIX1 = 0xbf58476d1ce4e5b9;
    constexpr static uint64_t XRSR_MIX2 = 0x94d049bb133111eb;
    // Seeding offsets, named the way the game's own source names them
    // (SILVER_RATIO_64 / GOLDEN_RATIO_64).
    constexpr static uint64_t XRSR_SILVER_RATIO = 0x6a09e667f3bcc909;
    constexpr static uint64_t XRSR_GOLDEN_RATIO = 0x9e3779b97f4a7c15;

    uint64_t lo;
    uint64_t hi;

    HOST_DEVICE explicit XrsrRandom() {}

    HOST_DEVICE constexpr XrsrRandom(uint64_t lo_, uint64_t hi_)
        : lo(lo_), hi(hi_) {
        substituteZeroState();
    }

    HOST_DEVICE explicit XrsrRandom(uint64_t seed) {
        setSeed(seed);
    }

    HOST_DEVICE XrsrRandom(const XrsrRandom &other)
        : lo(other.lo), hi(other.hi) {}

    // The splitmix64 finalizer: invertible, full avalanche. Also the engine
    // behind --shuffled-search (see the file header).
    HOST_DEVICE static uint64_t mix64(uint64_t a) {
        a = (a ^ a >> 30) * XRSR_MIX1;
        a = (a ^ a >> 27) * XRSR_MIX2;
        return a ^ a >> 31;
    }

    HOST_DEVICE constexpr static uint64_t rol64(uint64_t a, int bits) {
        return (a << bits) | (a >> (64 - bits));
    }

    // cubiomes' xSetSeed: XOR the silver offset in, run one mix64 chain per
    // state word, with the hi chain's input offset by the golden constant.
    HOST_DEVICE void setSeed(uint64_t seed) {
        seed ^= XRSR_SILVER_RATIO;
        lo = mix64(seed);
        hi = mix64(seed + XRSR_GOLDEN_RATIO);
        substituteZeroState();
    }

    // One Xoroshiro128++ round (cubiomes' xNextLong): emit the ++ output,
    // then scramble the state.
    HOST_DEVICE constexpr uint64_t nextLong() {
        uint64_t l = lo;
        uint64_t h = hi;
        uint64_t r = rol64(l + h, 17) + l;
        h ^= l;
        lo = rol64(l, 49) ^ h ^ h << 21;
        hi = rol64(h, 28);
        return r;
    }

    // The top `bits` bits of a draw, matching the game's nextBits.
    HOST_DEVICE uint64_t nextBits(int32_t bits) {
        return nextLong() >> (64 - bits);
    }

    HOST_DEVICE uint32_t nextInt() {
        return static_cast<uint32_t>(nextLong());
    }

    // Bounded draw (cubiomes' xNextInt): a 64-bit multiply whose high half
    // is kept, plus a rejection loop guarding the thin slice of products
    // that would bias small bounds. The loop body essentially never runs.
    HOST_DEVICE uint32_t nextInt(uint32_t bound) {
        uint64_t l = nextInt();
        uint64_t m = l * bound;
        uint64_t n = m & 0xFFFFFFFF;
        if (n < bound) {
            uint32_t j = (~bound + 1) % bound;
            while (n < j) {
                l = nextInt();
                m = l * bound;
                n = m & 0xFFFFFFFF;
            }
        }
        return static_cast<uint32_t>(m >> 32);
    }

    // The top 24 bits of a draw, scaled: the game's float draw. (init_noise
    // deliberately draws its table offsets this way rather than with
    // nextDouble; both consume exactly one RNG output, so the draw order is
    // unchanged, and the value differs only in low-order rounding, well
    // under a hundredth of a noise cell.)
    HOST_DEVICE float nextFloat() {
        return static_cast<float>(nextBits(24)) * 5.9604645E-8f;
    }

private:
    // The all-zero state is a fixed point of the generator (it would emit
    // zeros forever), so the game substitutes a fixed marked state; the
    // same two constants Minecraft uses.
    HOST_DEVICE constexpr void substituteZeroState() {
        if ((lo | hi) == 0) {
            lo = static_cast<uint64_t>(-7046029254386353131); // golden ratio
            hi = static_cast<uint64_t>(7640891576956012809);  // silver ratio
        }
    }
};

// Child generator = fork XOR name hash: the exact fork cubiomes'
// init_climate_seed performs (pxr.lo = xlo ^ hash.lo, pxr.hi = xhi ^ hash_hi).
HOST_DEVICE inline XrsrRandom XrsrRandomFork::from(const XrsrForkHash &hash) const {
    return { lo ^ hash.lo, hi ^ hash.hi };
}

#endif // RANDOM_H_
