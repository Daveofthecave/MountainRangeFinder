// common.h
// Shared definitions, data structures, and threading utilities for
// MountainRangeFinder. Bridges the main orchestrator, the GPU kernels,
// and the CPU verifier threads.

#pragma once

#include <cstdint>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <utility>
#include <limits>

#include "biomes.h"   // biome ID constants + isOceanic(), for the census bins
#include "random.h"   // XrsrRandom::mix64 (splitmix64 finalizer, a bijection
                      // on 2^64) for --shuffled-search

// Program version, printed by --version and the startup banner. Bump this
// together with the git tag when cutting a release.
#define MRF_VERSION "1.0.0"

// ============================================================================
// Search configuration constants (blocks), mapped from the CV recipe
// ============================================================================

// Main search area: centered square of side 32,000 -> +-16,000 from (0, 0).
constexpr int32_t SEARCH_RADIUS = 16000;

// Step size of the main grid iterator. 1600 keeps the 21x21 anchor lattice
// dense enough that any megaregion-scale trough is reliably caught. (A 2000
// step was measured to miss known-good seeds whose blobs sit off-center
// between anchors.)
constexpr int32_t GRID_STEP_SIZE = 1600;

// Radius of the core low-erosion blob (recipe step 2); also the radius of the
// CPU-side surface-height spiral (recipe steps 7-8).
constexpr int32_t CORE_BLOB_RADIUS = 1600;

// Radius of the legacy adjacent-check circles; now only used by the CPU
// verifier's Dark Forest exclusion (recipe steps 58-59).
constexpr int32_t ADJACENT_BLOB_RADIUS = 1365;

// Surface-height acceptance window (recipe step 8).
constexpr int32_t MIN_TARGET_Y = 230;
constexpr int32_t MAX_TARGET_Y = 320;

// Default result deduplication radius (blocks). Two yielded points for the 
// same seed closer than this are treated as one megaregion reported twice 
// (adjacent core anchors of a single blob fill measure the same connected 
// field and emit near-identical results). CLI: --dedup-radius, 0 disables.
constexpr int32_t DEDUP_RADIUS_BLOCKS = 8000;

// ============================================================================
// Megaregion size gate (shared by the GPU flood fill and the CPU verifier)
//
// The GPU flood fill (KernelBlob, gpu.cu) counts connected low-erosion cells
// on a 64-block lattice; each cell stands for 64*64 = 4096 blocks^2. The GPU
// early-exits at TARGET_BLOB_CELLS, which is deliberately a looser prefilter
// than the CPU's MIN_BLOB_AREA: the CPU re-measures in double precision and
// applies the real gate. A GPU-side float32 gate that is tighter than the CPU
// gate would silently discard real blobs near the threshold without ever
// measuring them precisely, so make sure that:
// TARGET_BLOB_CELLS * 4096 is 10-12% less than MIN_BLOB_AREA.
// TARGET_BLOB_CELLS gets adjusted automatically if
// the user sets --min-blob-area to a different number.
// ============================================================================

// GPU prefilter fallback; main.cpp recomputes it as 0.88 * MIN_BLOB_AREA / 4096,
// so the value here only matters if that recomputation is ever removed.
constexpr uint32_t TARGET_BLOB_CELLS = 3840;            // 3840 * 4096 = ~15.7M blocks^2
constexpr int64_t  MIN_BLOB_AREA     = 18'000'000;      // blocks^2 (CPU gate)

// Minimum inscribed core radius, in lattice cells (64 blocks each). This is
// measured for real by a two-pass chamfer distance transform over the
// connected blob mask; e.g. the default of 20 cells * 64 blocks each = 1280 blocks. 
// --min-core-width 0 disables the gate entirely.
constexpr int32_t MIN_CORE_WIDTH_CELLS = 20;

// ============================================================================
// Multi-climate blob field thresholds, shared by the GPU KernelBlob and the
// CPU flood fill. Both sides must sample the same field or their measurements
// cannot be compared. Raw noise units (CV display / 10000).
// The CLI knobs can only make the CPU gate stricter than this definition;
// to change the blob definition itself, edit these and rebuild.
// ============================================================================
constexpr float BLOB_ERO_MAX  = -0.35f;  // 0B erosion ceiling
constexpr float BLOB_CONT_MIN = -0.12f;  // 1B continentalness floor (inland)
constexpr float BLOB_TEMP_MIN = -0.45f;  // 0B temperature window (reject frozen)
constexpr float BLOB_TEMP_MAX =  0.20f;  // 0B temperature window (reject hot)

// ============================================================================
// Quart-space mirrors (quart = block / 4)
//
// Minecraft 1.18+ samples the biome climate maps at 1:4 ("quart") resolution.
// The GPU kernels operate entirely in quart coordinates and convert back to
// blocks (x4) exactly once, when emitting candidates.
// ============================================================================
constexpr int32_t SEARCH_RADIUS_Q = SEARCH_RADIUS / 4;   // 4000
constexpr int32_t GRID_STEP_Q     = GRID_STEP_SIZE / 4;  // 500

static_assert(SEARCH_RADIUS % 4 == 0, "search radius must be a multiple of 4");
static_assert(GRID_STEP_SIZE % 4 == 0, "grid step must be a multiple of 4");

// ============================================================================
// Inter-thread communication
// ============================================================================

// GPU -> CPU candidate. x/z are ordinary BLOCK coordinates (converted from
// quart inside the blob-fill kernel at emission time).
struct GpuOutput {
    uint64_t seed;
    int32_t x;
    int32_t z;
    int32_t blob_h;  // blob-scale max approx height from the GPU fill (0 = n/a)
    int32_t blob_dh; // tier-2 mean |dh| at 128-block pitch, x8 fixed point (0 = n/a)
};

// ============================================================================
// Surface-biome census bins, shared by the CPU verifier (cpu.cpp) and the
// prober (probe.cpp). Keeping the binning here means the verifier's bio_*
// columns and the prober's census columns can never drift apart.
//
// Bins:
//   0 mtn  -- jagged/frozen/stony peaks, grove, snowy slopes
//   1 mtg  -- "mega taiga": old-growth pine + old-growth spruce
//   2 tg   -- regular taiga + snowy taiga
//   3 mdw  -- meadow
//   4 chy  -- cherry grove
//   5 drk  -- dark forest
//   6 pln  -- plains + sunflower plains
//   7 riv  -- river + frozen river
//   8 ocn  -- any oceanic biome
//
// Everything else (birch/regular forest, jungle, savanna, swamp, stony shore,
// beach, badlands, snowy plains, ice spikes, mushroom fields, ...) falls into
// the uncounted "other" remainder.
// ============================================================================
enum CensusBin {
    CEN_MOUNTAIN = 0,
    CEN_OG_TAIGA,
    CEN_TAIGA,
    CEN_MEADOW,
    CEN_CHERRY,
    CEN_DARK,
    CEN_PLAINS,
    CEN_RIVER,
    CEN_OCEAN,
    CEN_N
};

inline const char *census_bin_name(int b) {
    static const char *names[CEN_N] = {
        "mtn", "mtg", "tg", "mdw", "chy", "drk", "pln", "riv", "ocn"
    };
    return (b >= 0 && b < CEN_N) ? names[b] : "?";
}

// Returns the bin index, or -1 for "other". Host-side only (cpu.cpp/probe.cpp);
// declared `inline` so the header can be included from several translation
// units (including gpu.cu, which never calls it) without ODR issues.
inline int census_bin(int id) {
    switch (id) {
    case jagged_peaks:
    case frozen_peaks:
    case stony_peaks:
    case grove:
    case snowy_slopes:                return CEN_MOUNTAIN;
    case old_growth_pine_taiga:
    case old_growth_spruce_taiga:     return CEN_OG_TAIGA;
    case taiga:
    case snowy_taiga:                 return CEN_TAIGA;
    case meadow:                      return CEN_MEADOW;
    case cherry_grove:                return CEN_CHERRY;
    case dark_forest:                 return CEN_DARK;
    case plains:
    case sunflower_plains:            return CEN_PLAINS;
    case river:
    case frozen_river:                return CEN_RIVER;
    default: break;
    }
    if (isOceanic(id))                return CEN_OCEAN;
    return -1;
}

// ============================================================================
// Enrichment statistics: the expensive measurement bundle produced once per
// fully verified seed by blob_enrich() (probe.cpp), on the same flood fill
// that the gates used. Every field below feeds either the composite score or
// the output columns; both the verifier (cpu.cpp) and the prober (--probe
// mode) fill this same struct, so their numbers are directly comparable.
//
// All stats are measured over the stride-2 sublattice (128-block pitch) of
// the connected blob, except where noted. NaN marks "not measured".
// ============================================================================
struct EnrichStats {
    // full-octave climate stats over the sublattice
    double eroF_mean, eroF_p10, eroF_min;
    double contF_mean, contF_max;

    // weirdness texture: mean |w|, ridge/valley fractions (PV thresholds),
    // and ridge/valley CROSSING fractions between 128-block neighbors
    double w_abs, ridge, valley, rcross, vcross;

    // approximate surface height over the sublattice
    double h_min, h_max, h_mean, h_p90;
    double h_std;   // height spread: "contrasty, fine-grained" as a number
    double relief;  // fraction of neighbor pairs with |dh| >= 96 blocks
    double dh_mean; // mean |dh| over measured neighbor pairs (threshold-free ruggedness)
    double high;    // fraction of cells at y >= 200
    double low;     // fraction of cells at y <= 116

    // "Packed individual peaks" texture (the massif discriminators):
    double peak_density; // strict local height maxima per million blocks^2
    double peak_prom;    // mean prominence of the maxima (peak - mean of 4 neighbors)
    double high_comps;   // connected components of the y>=200 mask, per M blocks^2
    double high_largest; // largest high component's share of all high cells
                         // (~1.0 for a plateau massif; lower for packed peaks)

    // Dark forest around the main 1,664-block window center (r = 1152 blocks, 
    // 64-block pitch): 0 = none; ~0.05+ = dark forest right where the output 
    // row sends you.
    double dark_core_frac;
    // Worst local blotch: max dark-forest fraction over 896-block windows
    // whose centers lie within 1792 blocks of the headline.
    double max_dark_896_near;
    // Ocean proximity of the headline: blocks to the nearest ocean-biome cell
    // inside the blob (chamfer DT over the sublattice ocean mask; misses open
    // ocean beyond the blob edge), and blocks to the nearest 1B-cont <= -0.19
    // point within 3200 blocks (spiral scan; sees open ocean). NaN = none.
    double ocean_dt_hl, ocean_spiral;
    // "How much of the blob is actually magic": superlevel-set stats of the
    // 1664-block window pattern-score field. qual_area = largest 4-connected
    // component of windows with sc >= max(0.75, 0.35*best), in blocks^2;
    // qual_n = qualifying window count; sc_q90 = 90th percentile score.
    double qual_area, qual_n, sc_q90;
    // Mountain-range connectivity: largest 4-connected component of the
    // "mtnish" mask (peaks + meadow + old-growth taiga + cherry grove) in
    // blocks^2, and the component count.
    double mtn_comp_m, mtn_comps;
    // Gorge floors: fraction of low (h <= 105) sublattice cells with a tall
    // (h >= 190) cell within 256 blocks -- the "deep floor, tall wall, short
    // distance" adjacency of the magic region's dissecting valleys.
    double gorge_frac;
    // Valley-network connectivity: 4-connected components of the INLAND low
    // mask (h <= 100; oceanic-biome cells excluded so an inland sea cannot
    // impersonate a valley network). low_comp_m = largest component's area in
    // blocks^2; low_comps = component count. One huge component = the deep
    // valleys are all interlinked (the magic-region signature); many small
    // ones = isolated bowls/cirques.
    double low_comp_m, low_comps;
    // Vicinity oddball flags from the headline ring scan (<= 4800 blocks):
    // 1.0 if a mushroom-island climate (1B cont <= -1.05) was found, 1.0 if a
    // warm-ocean climate (shore-ish cont and 0B temp >= 0.55) was found.
    // Measured, then provisionally weighted -- see compute_score's charm term.
    double vic_mush, vic_warmoc;

    // Setting (scenery) feature measurements -- the unified successor to the
    // icing + charm pair, combined into `setting` by compute_score.
    //   lake_n:       isolated river-lake count (small interior river
    //                 components on valley floors with tall walls within
    //                 256 blocks, in an otherwise river-scarce blob)
    //   cliff_frac:   share of in-blob ocean sublattice cells with tall
    //                 (>= 190) terrain within 256 blocks (NaN if no ocean)
    //   vic_sea_min:  nearest ocean-ish point along the headline's 16 rays
    //   vic_open_oc:  rays (of 16) still in non-frozen ocean at 4800 blocks
    //   vic_open_warm: of those, warm-ocean rays (0B temp >= 0.55)
    //   vic_fjord:    best ocean-excursion quality in [0, 1] (ray dips into
    //                 ocean and returns to land: fjord / inlet / inland sea)
    //   vic_isthmus:  1.0 if ocean appears in sky sectors >= 135 degrees apart
    //   vic_badl / vic_badl_oth / vic_mush_b / vic_flower / vic_cherry /
    //   vic_mtg:      256-pitch biome scan outside the blob (<= 4800 blocks):
    //                 eroded badlands, other badlands, biome-confirmed
    //                 mushroom fields, flower forest, cherry grove, mega taiga
    //   mtg_comp_m / chy_comp_m: largest same-biome old-growth-taiga /
    //                 cherry-grove component inside the blob, blocks^2
    //   setting:      the final gated, capped bonus actually added to score
    double lake_n, cliff_frac;
    double vic_sea_min, vic_open_oc, vic_open_warm, vic_fjord, vic_isthmus;
    double vic_badl, vic_badl_oth, vic_mush_b, vic_flower, vic_cherry, vic_mtg;
    double mtg_comp_m, chy_comp_m;
    double setting;


    // 9-bin surface census fractions over the sublattice
    double bio[CEN_N];

    // Windowed best-subregion stats: the best 896x896- and 1664x1664-block
    // square windows inside the blob (2D prefix sums over the sublattice).
    double w896_ero_min,  w896_h_max,  w896_mtn_max;
    int32_t w896_ero_x,   w896_ero_z;   // center of the best-erosion 896 window
    double w1664_ero_min, w1664_h_max, w1664_mtn_max;
    int32_t w1664_ero_x,  w1664_ero_z;  // center of the best-erosion 1664 window
    // Per-window local contrast/depth (over cells with measured heights):
    //   hstd_max  = max over windows of the height standard deviation
    //               (peaks and valleys packed into one neighborhood)
    //   hmean_min = min over windows of the mean height (deepest valley bowl)
    double w896_hstd_max,  w896_hmean_min;
    double w1664_hstd_max, w1664_hmean_min;
    // Coherent best-pattern windows (score v3.1): per tier, the single window
    // maximizing the pattern criterion (probe.cpp window_pattern_score);
    // all stats co-located. bw_ tiers: 896, 1664, 3200 blocks.
    // hifrac = frac of cells at y>=200; lofrac = frac of inland cells at
    // y<=63 (the sea-level-valley signature); ocn = frac of oceanic-biome
    // cells (coastal-cliff detector: ocean windows are excluded from argmax).
    double bw896_sc,  bw896_ero,  bw896_hmean,  bw896_hstd,  bw896_dh,  bw896_mtn;
    double bw896_hifrac, bw896_lofrac, bw896_ocn;
    int32_t bw896_x, bw896_z;
    double bw1664_sc, bw1664_ero, bw1664_hmean, bw1664_hstd, bw1664_dh, bw1664_mtn;
    double bw1664_hifrac, bw1664_lofrac, bw1664_ocn;
    double bw1664_dark;                 // dark-forest fraction of the pattern window (v3.2)
    int32_t bw1664_x, bw1664_z;
    double bw3200_sc, bw3200_ero, bw3200_hmean, bw3200_hstd, bw3200_dh, bw3200_mtn;
    double bw3200_hifrac, bw3200_lofrac, bw3200_ocn;
    int32_t bw3200_x, bw3200_z;

    double score;   // composite rank (see compute_score in probe.cpp)
    bool valid;     // false when the blob was empty / nothing was measured

    EnrichStats() { clear(); }

    void clear() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        eroF_mean = eroF_p10 = eroF_min = nan;
        contF_mean = contF_max = nan;
        w_abs = ridge = valley = rcross = vcross = nan;
        h_min = h_max = h_mean = h_p90 = h_std = nan;
        relief = dh_mean = high = low = nan;
        peak_density = peak_prom = nan;
        high_comps = high_largest = nan;
        dark_core_frac = nan;
        max_dark_896_near = nan;
        ocean_dt_hl = ocean_spiral = nan;
        qual_area = qual_n = sc_q90 = nan;
        mtn_comp_m = mtn_comps = nan;
        gorge_frac = nan;
        low_comp_m = low_comps = nan;
        vic_mush = vic_warmoc = nan;
        lake_n = cliff_frac = nan;
        vic_sea_min = vic_open_oc = vic_open_warm = vic_fjord = vic_isthmus = nan;
        vic_badl = vic_badl_oth = vic_mush_b = vic_flower = vic_cherry = vic_mtg = nan;
        mtg_comp_m = chy_comp_m = nan;
        setting = nan;
        for (int b = 0; b < CEN_N; b++) bio[b] = nan;
        w896_ero_min = w896_h_max = w896_mtn_max = nan;
        w896_ero_x = w896_ero_z = 0;
        w1664_ero_min = w1664_h_max = w1664_mtn_max = nan;
        w1664_ero_x = w1664_ero_z = 0;
        w896_hstd_max = w896_hmean_min = nan;
        w1664_hstd_max = w1664_hmean_min = nan;
        bw896_sc = bw896_ero = bw896_hmean = bw896_hstd = bw896_dh = bw896_mtn = nan;
        bw896_hifrac = bw896_lofrac = bw896_ocn = nan;
        bw896_x = bw896_z = 0;
        bw1664_sc = bw1664_ero = bw1664_hmean = bw1664_hstd = bw1664_dh = bw1664_mtn = nan;
        bw1664_hifrac = bw1664_lofrac = bw1664_ocn = nan;
        bw1664_dark = nan;
        bw1664_x = bw1664_z = 0;
        bw3200_sc = bw3200_ero = bw3200_hmean = bw3200_hstd = bw3200_dh = bw3200_mtn = nan;
        bw3200_hifrac = bw3200_lofrac = bw3200_ocn = nan;
        bw3200_x = bw3200_z = 0;
        score = nan;
        valid = false;
    }
};

// ============================================================================
// Final CPU-verified result, written to the output file.
//
// x/z are the main window's coordinates: the center of the blob's best-pattern
// 1664x1664-block window (the heart of the densest-packed peaks), falling
// back to the 896 window and then to the raw anchor. anchor_x/anchor_z always
// hold the original candidate anchor (the blob-attach point), so the exact
// pipeline hit can be reproduced.
//
// score and stats are NaN/invalid when the blob measurement is disabled
// (--no-blob).
// ============================================================================
struct CpuOutput {
    uint64_t seed;
    int32_t x, z;           // headline coords (best 1664-block window center)
    int32_t anchor_x, anchor_z; // original candidate anchor
    double score;           // composite rank; NaN when enrichment didn't run
    int32_t y_height;       // true max approx surface height within the probe radius
    int32_t gpu_h;          // GPU blob-fill max approx height (whole blocks)
    int32_t gpu_dh8;        // GPU tier-2 mean |dh| at 128-block pitch, x8 (0 = n/a)
    float ero_cov;          // measured 0B-erosion disc coverage fraction at the core
    float ero_min;          // full-octave erosion window minimum (raw noise units)
    float cont_max;         // full-octave continentalness window maximum (raw units)
    int32_t adj_dirs;       // legacy: adjacent directions (unused since blob fill; -1)
    int64_t blob_area;      // contiguous low-erosion area in blocks^2 (0 if off)
    int32_t core_radius;    // measured inscribed core radius in lattice cells (64 blocks each)
    int32_t blob_edge;      // 1 = blob reached the +-48000-block window edge (area is a lower bound)
    EnrichStats stats;      // enrichment bundle (valid iff stats.valid)
};

struct GpuOutputs {
    std::queue<GpuOutput> queue;
    std::mutex mutex;
    std::atomic_uint64_t total_pushed{0}; // candidates ever enqueued (completion tracking)
};

struct CpuOutputs {
    std::queue<CpuOutput> queue;
    std::mutex mutex;
    std::atomic_uint64_t total_pushed{0};
};

// ============================================================================
// CPU verifier configuration (populated from the CLI in main.cpp)
// ============================================================================
struct VerifyConfig {
    int mc_version = 0;             // cubiomes MC_* enum, from --mc
    bool check_height = true;       // recipe steps 7-8 (approx surface height)
    bool close1 = true;             // 1-cell morphological closing on the blob mask
    int32_t min_height = MIN_TARGET_Y;
    int32_t max_height = MAX_TARGET_Y;
    bool check_dark_forest = true;  // recipe steps 58-59
    double dark_forest_frac = 0.07; // max dark-forest share of the anchor disc
    bool measure_blob = true;       // contiguous-area measurement AND gate
    bool climate_stats = true;      // CPU re-measurement of the GPU's claims
    int32_t enrich_phases = 4;      // sublattice phases measured per blob
                                    // (4 = keep the best; 1 = phase 0 only)
    bool aggregate = false;         // search/list mode: main.cpp clusters each
                                    // seed's verified rows and keeps the best-
                                    // scoring row per cluster (dedup is skipped)
    // Configurable CPU-side gates (defaults = the shared blob field above).
    int64_t min_blob_area = MIN_BLOB_AREA;
    int32_t min_core_width = MIN_CORE_WIDTH_CELLS;
    int32_t dedup_radius = DEDUP_RADIUS_BLOCKS;
    float ero_max  = BLOB_ERO_MAX;
    float cont_min = BLOB_CONT_MIN;
    float temp_min = BLOB_TEMP_MIN;
    float temp_max = BLOB_TEMP_MAX;
};

// ============================================================================
// Seed source: hands out blocks of seeds to the GPU workers.
//
// Incremental mode dispatches consecutive 64-bit seeds starting from a base.
// List mode (--seeds) dispatches exactly the seeds from a file, then reports
// exhaustion so the pipeline can shut down on its own.
// ============================================================================
struct SeedIterator {
    std::atomic_uint64_t pos{0};
    std::atomic_uint64_t done{0};        // seeds fully processed by all GPU workers

    // Seeds fully processed so far (for stats; lags pos by up to one batch/GPU).
    uint64_t completed() const {
        return done.load(std::memory_order_relaxed);
    }
    void mark_done(uint32_t n) {
        done.fetch_add(n, std::memory_order_relaxed);
    }

    // List mode
    const bool list_mode;
    std::vector<uint64_t> seeds;
    std::atomic_uint64_t list_idx{0};

    const uint64_t start; // incremental-mode base (counter when shuffled)
    const bool shuffled;  // --shuffled-search: counter -> mix64 permutation

    explicit SeedIterator(uint64_t start)
        : pos(start), list_mode(false), start(start), shuffled(false) {}
    explicit SeedIterator(uint64_t start, bool shuffled)
        : pos(start), list_mode(false), start(start), shuffled(shuffled) {}
    explicit SeedIterator(std::vector<uint64_t> list)
        : list_mode(true), seeds(std::move(list)), start(0), shuffled(false) {}

    // Fills out[0..count) with the next seeds; returns the number of seeds
    // written (fewer than count at the tail of a list; 0 once a list is
    // exhausted). Incremental mode always returns count.
    uint32_t next_batch(uint64_t *out, uint32_t count) {
        if (!list_mode) {
            const uint64_t s = pos.fetch_add(count, std::memory_order_relaxed);
            if (shuffled) {
                // mix64 is a bijection on 2^64: the counter visits every seed
                // exactly once, in pseudo-random order. Zero collisions for
                // the whole run; resume = remembering the counter.
                for (uint32_t i = 0; i < count; i++) out[i] = XrsrRandom::mix64(s + i);
            } else {
                for (uint32_t i = 0; i < count; i++) out[i] = s + i;
            }
            return count;
        }
        const uint64_t i0 = list_idx.fetch_add(count, std::memory_order_relaxed);
        if (i0 >= seeds.size()) return 0;
        const uint32_t n = (uint32_t)std::min<uint64_t>(count, (uint64_t)(seeds.size() - i0));
        for (uint32_t i = 0; i < n; i++) out[i] = seeds[i0 + i];
        return n;
    }

    // True once every listed seed has been handed out (list mode only).
    bool exhausted() const {
        return list_mode && list_idx.load(std::memory_order_relaxed) >= seeds.size();
    }

    // Raw seed counter position (incremental mode) or absolute list index.
    uint64_t progress() const {
        return list_mode ? std::min<uint64_t>(list_idx.load(std::memory_order_relaxed), seeds.size())
                         : pos.load(std::memory_order_relaxed);
    }

    // Total seeds to be searched (list mode; 0 for incremental).
    uint64_t total() const { return list_mode ? (uint64_t)seeds.size() : 0; }
};

// ============================================================================
// Generic thread wrapper (CRTP): start()/stop()/join() around run()
// ============================================================================
template<typename T>
struct Thread {
private:
    std::atomic_bool stop_flag;
    std::thread thread;

protected:
    Thread() : stop_flag(false), thread() {}

    void start() {
        thread = std::thread(&T::run, (T*)this);
    }

    bool should_stop() {
        return stop_flag.load(std::memory_order_relaxed);
    }

public:
    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
    }

    void join() {
        if (thread.joinable()) {
            thread.join();
        }
    }
};

// Global run flag, defined in main.cpp. Worker loops should poll this.
extern std::atomic_bool g_running;

// ===========================================================================
// Console UI: verbosity levels + live status line (implemented in main.cpp)
//
//   UI_NORMAL  -- one compact line per verified seed + a self-refreshing
//                 status line (TTY), or a 5-second progress line (log file)
//   UI_DEBUG   -- the above, plus the GPU per-stage funnel tables (~5 s)
//   UI_VERBOSE -- the full firehose: every individual reject line, as before
//
// The status line is a curl-style carriage-return rewrite: no ANSI escapes,
// no curses, so it works on any terminal (including legacy cmd.exe). All
// console output during the search goes through ui_eventf()/ui_block_*() so
// the status line is cleared and redrawn around it without tearing.
// ===========================================================================
enum UiLevel { UI_NORMAL = 0, UI_DEBUG = 1, UI_VERBOSE = 2 };
extern std::atomic_int g_ui_level;

void ui_init();                       // call once in main() before workers start
void ui_status(const char *s);        // set + redraw the status line (TTY only)
void ui_eventf(const char *fmt, ...); // print one line above the status line
void ui_block_begin();                // clear the status line, lock the console
void ui_block_end();                  // redraw the status line, unlock
