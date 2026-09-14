// probe.h
// Two things live here:
//
//   1. The shared measurement engine: the per-thread noise context
//      (MeasureCtx), the connected-blob flood fill (blob_fill), and the
//      heavy sublattice measurement bundle + composite score (blob_enrich,
//      compute_score). The CPU verifier (cpu.cpp) uses these for its gates
//      and per-seed scoring; --probe mode uses them for gates-free
//      measurement. Single source of truth: both paths measure the SAME
//      field with the SAME code, so their numbers are directly comparable.
//
//   2. --probe mode itself (run_probe): reads "seed x z [poly-x/z pairs...]"
//      lines and prints a wide stat vector per candidate. Nothing is ever
//      rejected in probe mode.

#pragma once

#include "common.h"

#include "generator.h"
#include "biomenoise.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// Per-thread measurement context: the generator plus the six climate samplers
// the engine needs, seeded once per candidate by measure_ctx_seed().
//   eroB/contB/tempB -- the truncated blob-field samplers (0B / 1B / 0B)
//   eroF/contF/weird -- the full-octave samplers for enrichment stats
// ---------------------------------------------------------------------------
struct MeasureCtx {
    Generator g;
    BiomeNoise eroB;   // erosion 0A+0B (blob field)
    BiomeNoise eroF;   // erosion, all octaves
    BiomeNoise contB;  // continentalness through 1B (blob field + coast scan)
    BiomeNoise contF;  // continentalness, all octaves
    BiomeNoise tempB;  // temperature 0A+0B (blob field)
    BiomeNoise weird;  // weirdness, all octaves
};

void measure_ctx_setup(MeasureCtx &c, int mc_version);
void measure_ctx_seed(MeasureCtx &c, uint64_t seed);

// ---------------------------------------------------------------------------
// Connected multi-climate blob on the 64-block lattice (+-48,000-block
// reach), filled from the anchor (nudging out two lattice rings if the
// anchor cell itself fails). cells holds the lattice offsets of every
// connected cell; area/core_cells/edge are derived from the fill.
// ---------------------------------------------------------------------------
struct BlobFill {
    // Absolute block coords of lattice cell (0, 0): the anchor snapped down to
    // a multiple of 64 (world-aligned lattice). Two anchors that land in the
    // same blob then produce the same fill and the same stats -- the
    // measurement becomes a property of the seed, not the anchor's phase.
    // (Set by blob_fill before any cell is added.)
    int32_t origin_x = 0, origin_z = 0;
    std::vector<std::pair<int16_t, int16_t>> cells; // lattice offsets (i, j)
    int64_t area = 0;          // cells * 4096 blocks^2

    int32_t core_cells = 0;    // inscribed core radius, lattice cells (chamfer DT)
    int32_t edge = 0;          // 1 = region hit the measurement window edge
    // bounding box in lattice offsets
    int32_t minI = 0, maxI = -1, minJ = 0, maxJ = -1;
};

// The flood fill. Uses cfg.ero_max/cont_min/temp_min/temp_max (the shared
// BLOB_* field by default). Cost scales with the blob, not the reach window.
BlobFill blob_fill(MeasureCtx &c, const VerifyConfig &cfg, int32_t bx, int32_t bz);

// The heavy measurement bundle over the blob's stride-2 sublattice
// (128-block pitch): full-octave climates, weirdness texture, approximate
// heights, biome census, neighbor-pair crossings, windowed best-subregion
// stats, and the composite score (o.score). This is the expensive part of
// verification and runs only for seeds that passed every gate.
// `phase` selects which of the four interleaved sublattices is measured:
// bit 0 / bit 1 shift the 128-block sampling grid by 64 blocks in x / z.
void blob_enrich(MeasureCtx &c, const BlobFill &blob, EnrichStats &o, int phase);

// Phase-max enrichment: measures the blob on all four sublattice phases (or
// just phase 0 when phases == 1) and keeps the bundle with the best score.
// A blob's score is then a property of the blob itself -- never of which
// anchor happened to trigger the fill -- and no seed gets stuck with an
// unlucky sampling phase. Costs ~4x a single enrichment; the verifier pool
// has the headroom (use --phases 1 for fast bulk re-verification).
void blob_enrich_best(MeasureCtx &c, const BlobFill &blob, EnrichStats &o, int phases);

// Composite rank (rank, not gate). Fixed z-score reference constants measured
// on the 71-control batch; see the definition for the recalibration recipe.
double compute_score(int64_t area, int32_t core_cells, const EnrichStats &e);

// The unified scenery bonus (successor to icing + charm): recomputed from the
// EnrichStats setting fields. compute_score already includes this value; this
// accessor exists so the output column can report it.
double setting_score(const EnrichStats &e);

// The gated aspects behind compute_score: pattern quality (amp/tex/frag/depth,
// measured in the coherent best-pattern windows) and extent (pattern at scale;
// needs area/core, so it lives outside score_aspects). Exposed so the console
// can show why a seed scored the way it did (tuning: see compute_score).
void score_aspects(const EnrichStats &e, double &amp, double &tex, double &frag, double &depth);
// Negative aspect: dark forest aimed at the headline (ring + worst blotch).
// Joins the min() gate inside compute_score. NaN inputs -> neutral 0.0.
double score_dark_aspect(const EnrichStats &e);
double score_extent(int64_t area, int32_t core_cells, const EnrichStats &e);

// Thumbnail renderer (--thumbs <dir>): when enabled, blob_enrich writes a
// small BMP of each measured blob (heightfield shading + ocean/dark-forest
// tints + biome-class context ring) into <dir>. Filenames sort by score.
void probe_enable_thumbs(const char *dir, double min_score);

// ---------------------------------------------------------------------------
// --probe mode driver (gates-free measurement of a candidate list).
// ---------------------------------------------------------------------------
int run_probe(const std::string &path, const VerifyConfig &vcfg, int threads, std::FILE *out);
