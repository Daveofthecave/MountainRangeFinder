// probe.cpp
// The measurement engine, shared by the CPU verifier (cpu.cpp) and the
// --probe mode driver at the bottom of this file. Both paths measure the
// same field with the same code, so their numbers are directly comparable:
//
//   1. blob_fill flood-fills the connected region around an anchor on a
//      64-block lattice, out to +-48,000 blocks. A cell joins the region
//      if it passes the shared climate field (the BLOB_* thresholds in
//      common.h: low truncated erosion, inland continentalness, temperate
//      temperature). If the anchor cell itself fails, the search nudges
//      out two lattice rings for a valid start.
//   2. blob_enrich measures the region on every other lattice cell (a
//      128-block pitch): full-stack climate stats, weirdness texture,
//      approximate heights, a 9-bin biome census, neighbor-pair crossings,
//      and the sliding-window stats described further down.
//   3. compute_score folds all of that into the composite score.
//
// Vocabulary used throughout this file:
//
//   * the sublattice: the measurement grid at a 128-block pitch, every
//     other lattice cell.
//   * a phase: which of the four interleaved sublattices is measured (the
//     grid can start offset by 64 blocks on either axis). blob_enrich_best
//     measures all four and keeps the best score.
//   * a window: a square of sublattice cells (896, 1,664, or 3,200 blocks
//     across) slid over the region to find its best neighborhood.
//   * the headline: the coordinates an output row reports for a region,
//     the center of its best 1,664-block window.
//   * the reference region: the known-good seed in magic_seed.txt, used
//     as a canary for pipeline and scoring changes. Several thresholds
//     below were centered on its measured values; where that matters, the
//     comments say which magic_seed.txt columns to re-read.
//
// --probe mode (run_probe) skips every gate and only measures: the same
// bundle as above, plus blob-field means over all cells, coastal stats,
// and stats inside an optional user-drawn polygon.

#include "probe.h"
#include "common.h"

#include "finders.h"
#include "generator.h"
#include "biomenoise.h"
#include "biomes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <filesystem>

namespace {

// ---------------------------------------------------------------------------
// Geometry / tuning constants
// ---------------------------------------------------------------------------
constexpr int32_t BLOB_STEP  = 64;               // blocks per lattice cell
constexpr int32_t BLOB_MAX_R = 48000;            // how far the fill can reach from the anchor
constexpr int32_t BLOB_HALF  = BLOB_MAX_R / BLOB_STEP;  // 750 cells
constexpr int32_t BLOB_DIM   = 2 * BLOB_HALF + 1;       // 1501

// The expensive stats are measured on every SUB-th lattice cell, i.e. at a
// 128-block pitch.
constexpr int32_t SUB = 2;

// The coarse coastal scan: a 256-block pitch out to +-24,000 blocks.
constexpr int32_t COAST_STEP  = 256;
constexpr int32_t COAST_MAX_R = 24000;
constexpr int32_t COAST_HALF  = COAST_MAX_R / COAST_STEP; // 93 (the grid is 187 wide)

// Coastal thresholds for truncated continentalness, in raw noise units.
constexpr double CONT_SHORE = -0.19; // the land/ocean boundary
constexpr double CONT_DEEP  = -0.45; // deep-ocean territory

// Weirdness is read through the peaks-and-valleys projection,
// PV = 1 - 3*||w| - 2/3|: it approaches +1 on ridge lines (peaks) and -1 at
// w = 0 (the valley and river axis).
constexpr double PV_RIDGE  = 0.85;
constexpr double PV_VALLEY = -0.50;

// Height scale for the relief stats, in blocks.
constexpr float H_RELIEF = 96.0f;   // a neighbor-pair drop this big counts as relief
constexpr float H_HIGH   = 200.0f;  // high ground
constexpr float H_LOW    = 116.0f;  // valley-floor territory (measured valley floors sit near y100)

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
struct ProbeJob {
    uint64_t seed;
    int32_t x, z;
    std::vector<std::pair<double, double>> poly; // block coords; may be empty
};

// Whitespace tokenization used to locate the seed/x/z inside a row. Output
// rows lead with a decimal score column, so the seed is the first
// integer-like token, followed by two more integers (x, z).
struct Tok { const char *p; int len; };

int tokenize(const char *line, Tok *toks, int maxtoks) {
    int n = 0;
    const char *p = line;
    while (n < maxtoks) {
        while (*p == ' ' || *p == '\t' || *p == '\r') p++;
        if (*p == 0 || *p == '\n' || *p == '#') break;
        const char *s = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        toks[n++] = { s, (int)(p - s) };
    }
    return n;
}

bool tok_is_int(const Tok &t) {
    int i = (t.len > 0 && (t.p[0] == '+' || t.p[0] == '-')) ? 1 : 0;
    if (i >= t.len) return false;
    for (; i < t.len; i++)
        if (t.p[i] < '0' || t.p[i] > '9') return false;
    return true;
}

bool tok_to_i64(const Tok &t, int64_t &v) {
    char buf[40];
    const int len = std::min(t.len, 39);
    std::memcpy(buf, t.p, (size_t)len);
    buf[len] = 0;
    char *end = nullptr;
    const long long x = std::strtoll(buf, &end, 10);
    if (end == buf || *end != 0) return false;
    v = (int64_t)x;
    return true;
}

bool tok_to_f64(const Tok &t, double &v) {
    char buf[64];
    const int len = std::min(t.len, 63);
    std::memcpy(buf, t.p, (size_t)len);
    buf[len] = 0;
    char *end = nullptr;
    const double x = std::strtod(buf, &end);
    if (end == buf || *end != 0) return false;
    v = x;
    return true;
}

bool load_probe_list(const char *path, std::vector<ProbeJob> &out) {
    std::FILE *fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[2048];
    uint64_t skipped = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        Tok tok[128]; // probe rows run to ~74 tokens
        const int nt = tokenize(line, tok, 128);
        if (nt == 0) continue;

        // Locate seed/x/z: the first run of three integer tokens. This skips
        // a leading decimal score column on new-format output rows.
        int base = -1;
        for (int i = 0; i + 2 < nt; i++) {
            if (tok_is_int(tok[i]) && tok_is_int(tok[i + 1]) && tok_is_int(tok[i + 2])) {
                base = i;
                break;
            }
        }
        if (base < 0) { skipped++; continue; }

        int64_t seed, x, z;
        if (!tok_to_i64(tok[base], seed) ||
            !tok_to_i64(tok[base + 1], x) ||
            !tok_to_i64(tok[base + 2], z)) { skipped++; continue; }

        ProbeJob job{(uint64_t)seed, (int32_t)x, (int32_t)z, {}};

        // Remaining tokens after z: either a user polygon (3-5 x/z pairs) or
        // the trailing stats of a previous output row (>= 12 numbers), which
        // are ignored so output files can be fed straight back in.
        std::vector<double> nums;
        for (int i = base + 3; i < nt; i++) {
            double v;
            if (tok_to_f64(tok[i], v)) nums.push_back(v);
        }
        if (!nums.empty()) {
            const size_t tn = nums.size();
            if (tn >= 12) {
                // prior verifier/prober output row: polygon stays empty
            } else if (tn % 2 != 0 || tn < 6) {
                std::fprintf(stderr, "Note: skipped malformed polygon on seed %lld "
                                     "(need 3-5 x/z pairs)\n", (long long)seed);
                skipped++;
                continue;
            } else {
                for (size_t i = 0; i < tn; i += 2)
                    job.poly.emplace_back(nums[i], nums[i + 1]);
            }
        }

        out.push_back(std::move(job));
    }
    std::fclose(fp);
    if (skipped)
        std::fprintf(stderr, "Note: skipped %" PRIu64 " malformed lines in %s\n", skipped, path);
    return true;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// The shared blob field. Identical definition to the search (VerifyConfig).
inline bool cell_ok(MeasureCtx &c, const VerifyConfig &cfg, int32_t bx, int32_t bz) {
    const int32_t xq = bx >> 2, zq = bz >> 2;
    if (sampleClimatePara(&c.eroB, nullptr, xq, zq) > cfg.ero_max) return false;
    if (sampleClimatePara(&c.contB, nullptr, xq, zq) < cfg.cont_min) return false;
    const double t = sampleClimatePara(&c.tempB, nullptr, xq, zq);
    return t >= cfg.temp_min && t <= cfg.temp_max;
}

inline double pv_of(double w) {
    const double a = std::fabs(w);
    return 1.0 - 3.0 * std::fabs(a - 2.0 / 3.0);
}

inline float percentile(std::vector<float> &v, float q) {
    if (v.empty()) return -1.0f;
    const size_t k = (size_t)((double)(v.size() - 1) * (double)q);
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)k, v.end());
    return v[k];
}

// ---------------------------------------------------------------------------
// Windowed best-subregion stats, computed with 2D prefix sums over the
// sublattice (ev/hv/mv and friends are the per-cell value arrays built by
// blob_enrich, indexed over the region's bounding box).
//
// The reason for windows at all: a big region's best part is usually a
// sub-region, and whole-region averages dilute it. The reference region's
// highlight is a roughly 2000x3000-block neighborhood inside a 30M-block
// region, and measurements confined to that neighborhood are dramatically
// stronger than its region-wide averages. So the question these stats
// answer is not "is the region good on average?" but "does the region
// contain a spectacular neighborhood?" Every (2R+1)-cell square window
// over the region is scored, and the best of each stat is kept.
//
// Prefix sums make that affordable. With S[y][x] = the sum of all cells in
// rows 0..y, columns 0..x, any rectangle's sum costs four reads:
//   sum(x0..x1, y0..y1) = S[y1][x1] - S[y0-1][x1] - S[y1][x0-1] + S[y0-1][x0-1]
// so after one O(N) build, every window costs O(1). A window counts only
// when at least half of its cells lie inside the region; thin fringe
// slivers would otherwise win on artifacts.
// ---------------------------------------------------------------------------

// Window half-widths, in sublattice cells. R896 = 3 means a 7x7-cell
// window, 896x896 blocks; R1664 = 6 means 13x13 cells, 1664x1664 blocks.
struct WinHalf {
    static constexpr int32_t R896  = 3;
    static constexpr int32_t R1664 = 6;
    static constexpr double  MIN_COVER = 0.5; // a window must be at least half inside the region
};

template <typename T>
void prefix_sum_2d(std::vector<T> &a, int32_t W, int32_t H) {
    for (int32_t y = 0; y < H; y++) {
        T run = 0;
        for (int32_t x = 0; x < W; x++) {
            run += a[(size_t)y * W + x];
            a[(size_t)y * W + x] = (y > 0) ? run + a[(size_t)(y - 1) * W + x] : run;
        }
    }
}

template <typename T>
inline T rect_sum(const std::vector<T> &ps, int32_t W,
                  int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    T s = ps[(size_t)y1 * W + x1];
    if (x0 > 0) s -= ps[(size_t)y1 * W + x0 - 1];
    if (y0 > 0) s -= ps[(size_t)(y0 - 1) * W + x1];
    if (x0 > 0 && y0 > 0) s += ps[(size_t)(y0 - 1) * W + x0 - 1];
    return s;
}

struct WinResult {
    double ero_min = NaN, h_max = NaN, mtn_max = NaN;
    double hstd_max = NaN, hmean_min = NaN;
    int32_t ero_x = 0, ero_z = 0;
};

WinResult best_windows(int32_t R, int32_t SW, int32_t SH,
        const std::vector<double>  &ero_ps, const std::vector<double>  &h_ps,
        const std::vector<double>  &h2_ps,
        const std::vector<int32_t> &mtn_ps, const std::vector<int32_t> &cnt_ps,
        const std::vector<int32_t> &hcnt_ps, int32_t iE, int32_t jE,
        int32_t bx, int32_t bz) {
    WinResult wb;
    const int32_t cells = 2 * R + 1;
    const int64_t min_cover = (int64_t)std::ceil(WinHalf::MIN_COVER * cells * cells);
    for (int32_t cy = R; cy + R < SH; cy++) {
        for (int32_t cx = R; cx + R < SW; cx++) {
            const int64_t cnt = rect_sum(cnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
            if (cnt < min_cover) continue;
            const double ero_m = rect_sum(ero_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            if (std::isnan(wb.ero_min) || ero_m < wb.ero_min) {
                wb.ero_min = ero_m;
                wb.ero_x = bx + (iE + 2 * cx) * BLOB_STEP;
                wb.ero_z = bz + (jE + 2 * cy) * BLOB_STEP;
            }
            const int64_t hcnt = rect_sum(hcnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
            if (hcnt >= min_cover) {
                const double h_m  = rect_sum(h_ps,  SW, cx - R, cy - R, cx + R, cy + R) / (double)hcnt;
                const double h2_m = rect_sum(h2_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)hcnt;
                const double var  = h2_m - h_m * h_m;
                const double h_sd = var > 0.0 ? std::sqrt(var) : 0.0;
                if (std::isnan(wb.h_max)     || h_m  > wb.h_max)     wb.h_max = h_m;
                if (std::isnan(wb.hmean_min) || h_m  < wb.hmean_min) wb.hmean_min = h_m;
                if (std::isnan(wb.hstd_max)  || h_sd > wb.hstd_max)  wb.hstd_max = h_sd;
            }
            const double mtn_f = (double)rect_sum(mtn_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            if (std::isnan(wb.mtn_max) || mtn_f > wb.mtn_max) wb.mtn_max = mtn_f;
        }
    }
    return wb;
}

// ---------------------------------------------------------------------------
// Coherent best-pattern windows.
//
// Each window gets one combined pattern score from its own ingredients
// (window_pattern_score below), and the argmax window's full stat vector
// is kept, along with its center, which becomes the headline coordinate.
//
// Two guards keep the argmax in check. The first is about the sea: a range
// bordering the ocean produces huge height differences (cliff face to
// water), so without a guard, shoreline windows win and the headline lands
// in the ocean. Oceanic cells are therefore counted per window, windows
// above OCN_SKIP are excluded outright, and the rest are penalized. The
// second guard is about the region's edge: windows hanging off it (cover
// below 0.85) lose credit in proportion to how far they dangle. Two more
// terms reward the signature juxtaposition directly: hi, the share of
// cells at y >= 200 (tall ground), and lo, the share of inland cells at
// y <= 63 (sea-level valley floors; ocean cells never count toward it).
//
// Tuning: the reference constants below were centered on the reference
// region's measured best-window values (bw1664Dh 42.7-46.2, bw1664hStd
// 44.8-50.8 across its rows in magic_seed.txt). To re-center, re-run
// --verify on magic_seed.txt and read the bw1664Dh / bw1664hStd / bw1664Hi
// / bw1664Lo columns. If you change WREF_HI or WREF_LO, change the matching
// references in score_aspects (z_bwhi, z_bwlo) to follow.
// ---------------------------------------------------------------------------
constexpr double WREF_DH   = 36.0;  // reference for mean neighbor height difference (blocks)
constexpr double WREF_HSTD = 40.0;  // reference for height spread (blocks)
constexpr double WREF_ERO  = -0.90; // reference for mean full-stack erosion
constexpr double WREF_HI   = 0.10;  // reference for high-ground share
constexpr double WREF_LO   = 0.02;  // reference for sea-level valley-floor share
constexpr double OCN_SKIP  = 0.15;  // windows with more ocean than this are excluded outright

// One window's pattern score: each ingredient is compared against its
// reference constant above, weighted, and summed. Higher means a closer
// match to the target pattern: steep (dh), height-varied (hstd), with high
// ground (hi) and valley floors (lo) packed in together, deep erosion
// beneath it (ero), mountain-biome cover (mtn), and no sea in the frame
// (ocn).
static inline double window_pattern_score(double ero_m, double h_sd,
        double dh_m, double mtn_f, double hi_f, double lo_f, double ocn_f,
        double cover) {
    const double mtn_cap = mtn_f > 0.62 ? 0.62 : mtn_f; // saturates: past 62% cover, more doesn't help
    double hi_z = (hi_f - WREF_HI) / 0.10;
    hi_z = hi_z > 2.0 ? 2.0 : (hi_z < -1.5 ? -1.5 : hi_z);
    double lo_z = (lo_f - WREF_LO) / 0.03;
    lo_z = lo_z > 2.0 ? 2.0 : (lo_z < -1.5 ? -1.5 : lo_z);
    // Windows hanging off the blob's edge (cover < 0.85) -- including coastal
    // cliff windows, whose dh comes from the drop to the sea -- lose credit.
    const double edge_pen = cover < 0.85 ? 1.2 * (0.85 - cover) / 0.15 : 0.0;
    return 1.00 * ((dh_m - WREF_DH)   / 6.0)
         + 0.70 * ((h_sd - WREF_HSTD) / 12.0)
         + 0.40 * hi_z
         + 0.35 * lo_z
         + 0.35 * ((WREF_ERO - ero_m) / 0.15)
         + 0.30 * ((mtn_cap - 0.35)   / 0.15)
         - 3.00 * ocn_f
         - edge_pen;
}

// The full ingredient list of the single best-scoring window at one size
// tier, plus its center. NaN fields mean no eligible window was measured
// (e.g. the region is smaller than the window).
struct BwResult {
    double sc = NaN, ero = NaN, hmean = NaN, hstd = NaN, dh = NaN, mtn = NaN;
    double hifrac = NaN, lofrac = NaN, ocn = NaN, dark = NaN;
    int32_t x = 0, z = 0;
};

BwResult best_pattern_window(int32_t R, int32_t SW, int32_t SH,
        const std::vector<double>  &ero_ps, const std::vector<double>  &h_ps,
        const std::vector<double>  &h2_ps, const std::vector<double>  &dhsum_ps,
        const std::vector<int32_t> &dhcnt_ps,
        const std::vector<int32_t> &mtn_ps, const std::vector<int32_t> &cnt_ps,
        const std::vector<int32_t> &hcnt_ps,
        const std::vector<int32_t> &hi_ps, const std::vector<int32_t> &lo_ps,
        const std::vector<int32_t> &ocn_ps, const std::vector<int32_t> &drk_ps,
        int32_t iE, int32_t jE,
        int32_t bx, int32_t bz, std::vector<float> *field_out = nullptr) {
    BwResult r;
    // Optionally record every eligible window's score (NaN where the window
    // was never evaluated or was skipped), so the caller can analyze the
    // whole field instead of just the argmax.
    if (field_out)
        field_out->assign((size_t)SW * (size_t)SH, std::numeric_limits<float>::quiet_NaN());
    const int32_t cells = 2 * R + 1;
    const int64_t min_cover = (int64_t)std::ceil(WinHalf::MIN_COVER * cells * cells);
    for (int32_t cy = R; cy + R < SH; cy++) {
        for (int32_t cx = R; cx + R < SW; cx++) {
            const int64_t cnt = rect_sum(cnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
            if (cnt < min_cover) continue;
            const int64_t hcnt = rect_sum(hcnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
            if (hcnt < min_cover) continue;
            const int64_t dhcnt = rect_sum(dhcnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
            if (dhcnt < min_cover) continue;
            const double ocn_f = (double)rect_sum(ocn_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            if (ocn_f > OCN_SKIP) continue; // coastal window: cliff-to-sea dh is not the pattern
            const double ero_m = rect_sum(ero_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            const double h_m   = rect_sum(h_ps,  SW, cx - R, cy - R, cx + R, cy + R) / (double)hcnt;
            const double h2_m  = rect_sum(h2_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)hcnt;
            const double var   = h2_m - h_m * h_m;
            const double h_sd  = var > 0.0 ? std::sqrt(var) : 0.0;
            const double dh_m  = rect_sum(dhsum_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)dhcnt;
            const double mtn_f = (double)rect_sum(mtn_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            const double hi_f  = (double)rect_sum(hi_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            const double lo_f  = (double)rect_sum(lo_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            const double drk_f = (double)rect_sum(drk_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
            const double cover = (double)cnt / (double)(cells * cells);
            const double sc    = window_pattern_score(ero_m, h_sd, dh_m, mtn_f,
                                                      hi_f, lo_f, ocn_f, cover);
            if (field_out) (*field_out)[(size_t)cy * SW + cx] = (float)sc;
            if (std::isnan(r.sc) || sc > r.sc) {
                r.sc = sc; r.ero = ero_m; r.hmean = h_m; r.hstd = h_sd;
                r.dh = dh_m; r.mtn = mtn_f;
                r.hifrac = hi_f; r.lofrac = lo_f; r.ocn = ocn_f; r.dark = drk_f;
                r.x = bx + (iE + 2 * cx) * BLOB_STEP;
                r.z = bz + (jE + 2 * cy) * BLOB_STEP;
            }
        }
    }
    return r;
}

void windowed_blob_stats(const BlobFill &blob, int32_t bx, int32_t bz,
        const std::vector<double> &ev, const std::vector<double> &hv,
        const std::vector<double> &mv, const std::vector<double> &ov,
        const std::vector<double> &dv, EnrichStats &o,
        int32_t px, int32_t pz) {
    // Subgrid: cell (a, b) <-> blob-lattice offset (iE + 2a, jE + 2b), where
    // iE/jE are the smallest offsets >= the bbox min corner whose ABSOLUTE
    // lattice parity matches the phase being measured.
    const int32_t iE = blob.minI + ((((blob.minI + bx / BLOB_STEP) & 1) != px) ? 1 : 0);
    const int32_t jE = blob.minJ + ((((blob.minJ + bz / BLOB_STEP) & 1) != pz) ? 1 : 0);
    const int32_t SW = (blob.maxI - iE) / 2 + 1;
    const int32_t SH = (blob.maxJ - jE) / 2 + 1;
    if (SW < 2 * WinHalf::R896 + 1 || SH < 2 * WinHalf::R896 + 1) return;

    const int32_t BW = blob.maxI - blob.minI + 1; // gidx() row stride
    const size_t N = (size_t)SW * (size_t)SH;

    thread_local std::vector<double>  ero_ps, h_ps, h2_ps, dhsum_ps;
    thread_local std::vector<int32_t> mtn_ps, cnt_ps, hcnt_ps, dhcnt_ps;
    thread_local std::vector<int32_t> hi_ps, lo_ps, ocn_ps, drk_ps;
    thread_local std::vector<float>   b1664_field; // 1664-tier window score field
    ero_ps.assign(N, 0.0); h_ps.assign(N, 0.0); h2_ps.assign(N, 0.0);
    dhsum_ps.assign(N, 0.0);
    mtn_ps.assign(N, 0); cnt_ps.assign(N, 0); hcnt_ps.assign(N, 0); dhcnt_ps.assign(N, 0);
    hi_ps.assign(N, 0); lo_ps.assign(N, 0); ocn_ps.assign(N, 0); drk_ps.assign(N, 0);

    for (int32_t b = 0; b < SH; b++) {
        const int32_t j = jE + 2 * b;
        const size_t row = (size_t)(j - blob.minJ) * BW;
        for (int32_t a = 0; a < SW; a++) {
            const int32_t i = iE + 2 * a;
            const size_t gi = row + (size_t)(i - blob.minI);
            const double e = ev[gi];
            if (std::isnan(e)) continue; // not a sampled blob cell
            const size_t g = (size_t)b * SW + a;
            ero_ps[g]  = e;
            mtn_ps[g]  = (int32_t)mv[gi];
            ocn_ps[g]  = (int32_t)(ov[gi] > 0.5); // unconditional: drives the coastal skip
            drk_ps[g]  = (int32_t)(dv[gi] > 0.5);
            cnt_ps[g]  = 1;
            const double h = hv[gi];
            const int ocn = (int)(ov[gi] > 0.5); // for the inland-valley-floor test below
            if (!std::isnan(h)) {
                h_ps[g] = h; h2_ps[g] = h * h; hcnt_ps[g] = 1;
                if (h >= H_HIGH) hi_ps[g] = 1;        // y>=200: tall mass
                if (h <= 63.0 && !ocn) lo_ps[g] = 1;  // inland sea-level valley floor
                // |dh| to the E and S sublattice neighbors (128-block pitch);
                // hv is NaN outside the blob, which skips those pairs.
                if (i + 2 <= blob.maxI) {
                    const double hE = hv[gi + 2];
                    if (!std::isnan(hE)) { dhsum_ps[g] += std::fabs(h - hE); dhcnt_ps[g]++; }
                }
                if (j + 2 <= blob.maxJ) {
                    const double hS = hv[gi + (size_t)2 * BW];
                    if (!std::isnan(hS)) { dhsum_ps[g] += std::fabs(h - hS); dhcnt_ps[g]++; }
                }
            }
        }
    }

    prefix_sum_2d(ero_ps, SW, SH);
    prefix_sum_2d(h_ps, SW, SH);
    prefix_sum_2d(h2_ps, SW, SH);
    prefix_sum_2d(dhsum_ps, SW, SH);
    prefix_sum_2d(dhcnt_ps, SW, SH);
    prefix_sum_2d(hi_ps, SW, SH);
    prefix_sum_2d(lo_ps, SW, SH);
    prefix_sum_2d(ocn_ps, SW, SH);
    prefix_sum_2d(drk_ps, SW, SH);
    prefix_sum_2d(mtn_ps, SW, SH);
    prefix_sum_2d(cnt_ps, SW, SH);
    prefix_sum_2d(hcnt_ps, SW, SH);

    const WinResult w896 = best_windows(WinHalf::R896, SW, SH, ero_ps, h_ps, h2_ps,
                                        mtn_ps, cnt_ps, hcnt_ps, iE, jE, bx, bz);
    const WinResult w1664 = best_windows(WinHalf::R1664, SW, SH, ero_ps, h_ps, h2_ps,
                                         mtn_ps, cnt_ps, hcnt_ps, iE, jE, bx, bz);
    o.w896_ero_min  = w896.ero_min;   o.w896_h_max  = w896.h_max;   o.w896_mtn_max  = w896.mtn_max;
    o.w896_ero_x    = w896.ero_x;     o.w896_ero_z  = w896.ero_z;
    o.w896_hstd_max = w896.hstd_max;  o.w896_hmean_min = w896.hmean_min;
    o.w1664_ero_min = w1664.ero_min;  o.w1664_h_max = w1664.h_max;  o.w1664_mtn_max = w1664.mtn_max;
    o.w1664_ero_x   = w1664.ero_x;    o.w1664_ero_z = w1664.ero_z;
    o.w1664_hstd_max = w1664.hstd_max; o.w1664_hmean_min = w1664.hmean_min;

    // Coherent best-pattern windows: 896 (R=3), 1664 (R=6), 3200 (R=12).
    // A tier is skipped automatically when the blob's bounding box is smaller
    // than the window (the scan loop never runs, sc stays NaN).
    const BwResult b896  = best_pattern_window(3,  SW, SH, ero_ps, h_ps, h2_ps,
        dhsum_ps, dhcnt_ps, mtn_ps, cnt_ps, hcnt_ps, hi_ps, lo_ps, ocn_ps, drk_ps, iE, jE, bx, bz);
    const BwResult b1664 = best_pattern_window(6,  SW, SH, ero_ps, h_ps, h2_ps,
        dhsum_ps, dhcnt_ps, mtn_ps, cnt_ps, hcnt_ps, hi_ps, lo_ps, ocn_ps, drk_ps, iE, jE, bx, bz,
        &b1664_field);
    const BwResult b3200 = best_pattern_window(12, SW, SH, ero_ps, h_ps, h2_ps,
        dhsum_ps, dhcnt_ps, mtn_ps, cnt_ps, hcnt_ps, hi_ps, lo_ps, ocn_ps, drk_ps, iE, jE, bx, bz);
    o.bw896_sc  = b896.sc;   o.bw896_ero  = b896.ero;   o.bw896_hmean  = b896.hmean;
    o.bw896_hstd = b896.hstd; o.bw896_dh  = b896.dh;    o.bw896_mtn    = b896.mtn;
    o.bw896_hifrac = b896.hifrac; o.bw896_lofrac = b896.lofrac; o.bw896_ocn = b896.ocn;
    o.bw896_x   = b896.x;    o.bw896_z    = b896.z;
    o.bw1664_sc  = b1664.sc;   o.bw1664_ero  = b1664.ero;   o.bw1664_hmean  = b1664.hmean;
    o.bw1664_hstd = b1664.hstd; o.bw1664_dh  = b1664.dh;    o.bw1664_mtn    = b1664.mtn;
    o.bw1664_hifrac = b1664.hifrac; o.bw1664_lofrac = b1664.lofrac; o.bw1664_ocn = b1664.ocn;
    o.bw1664_dark = b1664.dark;
    o.bw1664_x   = b1664.x;    o.bw1664_z    = b1664.z;
    o.bw3200_sc  = b3200.sc;   o.bw3200_ero  = b3200.ero;   o.bw3200_hmean  = b3200.hmean;
    o.bw3200_hstd = b3200.hstd; o.bw3200_dh  = b3200.dh;    o.bw3200_mtn    = b3200.mtn;
    o.bw3200_hifrac = b3200.hifrac; o.bw3200_lofrac = b3200.lofrac; o.bw3200_ocn = b3200.ocn;
    o.bw3200_x   = b3200.x;    o.bw3200_z    = b3200.z;

    // Headline coordinate for the remaining stats (same fallback chain as
    // blob_enrich / cpu.cpp).
    int32_t hx = 0, hz = 0;
    bool have_hl = !std::isnan(b1664.sc);
    if (have_hl) { hx = b1664.x; hz = b1664.z; }
    else if (!std::isnan(b896.sc)) { hx = b896.x; hz = b896.z; have_hl = true; }

    // The worst local dark-forest patch near the headline: the maximum
    // dark-forest share over 896-block windows whose centers lie within
    // 1792 blocks of the headline. A genuine dark-forest blob is 50%+
    // locally, so this separates a couple of thick patches from a light
    // dusting spread across the region.
    if (have_hl) {
        const int32_t R = 3;
        const int64_t NEAR2 = 1792LL * 1792;
        double best = std::numeric_limits<double>::quiet_NaN();
        for (int32_t cy = R; cy + R < SH; cy++) {
            for (int32_t cx = R; cx + R < SW; cx++) {
                const int32_t wx = bx + (iE + 2 * cx) * BLOB_STEP;
                const int32_t wz = bz + (jE + 2 * cy) * BLOB_STEP;
                const int64_t dx = (int64_t)wx - hx, dz = (int64_t)wz - hz;
                if (dx * dx + dz * dz > NEAR2) continue;
                const int64_t cnt = rect_sum(cnt_ps, SW, cx - R, cy - R, cx + R, cy + R);
                if (cnt <= 0) continue;
                const double df = (double)rect_sum(drk_ps, SW, cx - R, cy - R, cx + R, cy + R) / (double)cnt;
                if (std::isnan(best) || df > best) best = df;
            }
        }
        o.max_dark_896_near = best;
    }

    // Superlevel-set stats of the 1,664-block window score field: how much
    // of the region is window-score material at all. qual_n counts the
    // windows above the threshold; qual_area measures the largest connected
    // patch of them. The threshold rises together with the best window's
    // score, so one lucky window cannot inflate either number.
    if (!std::isnan(b1664.sc)) {
        const float T = (float)std::max(0.75, 0.35 * b1664.sc);
        const int64_t N = (int64_t)SW * SH;
        thread_local std::vector<float> vals;
        vals.clear();
        vals.reserve(4096);
        int64_t qual = 0;
        for (int64_t k = 0; k < N; k++) {
            const float v = b1664_field[k];
            if (std::isnan(v)) continue;
            vals.push_back(v);
            if (v >= T) qual++;
        }
        if (!vals.empty())
            o.sc_q90 = percentile(vals, 0.90f);
        o.qual_n = (double)qual;
        o.qual_area = 0.0;
        if (qual > 0) {
            // Largest 4-connected qualifying component, converted to blocks^2
            // (window centers sit on the sublattice: 128-block spacing).
            thread_local std::vector<uint8_t> qseen;
            qseen.assign((size_t)N, 0);
            thread_local std::vector<int32_t> qqueue;
            qqueue.clear();
            int64_t largest = 0;
            for (int32_t cy = 0; cy < SH; cy++) {
                for (int32_t cx = 0; cx < SW; cx++) {
                    const size_t g0 = (size_t)cy * SW + cx;
                    if (qseen[g0] || !(b1664_field[g0] >= T)) continue; // NaN fails >=
                    int64_t sz = 0;
                    qseen[g0] = 1;
                    qqueue.push_back((int32_t)g0);
                    while (!qqueue.empty()) {
                        const int32_t qq = qqueue.back();
                        qqueue.pop_back();
                        sz++;
                        const int32_t qx = qq % SW, qy = qq / SW;
                        const int32_t nb[4][2] = {{qx-1,qy},{qx+1,qy},{qx,qy-1},{qx,qy+1}};
                        for (const auto &nbx : nb) {
                            if (nbx[0] < 0 || nbx[0] >= SW || nbx[1] < 0 || nbx[1] >= SH) continue;
                            const size_t g2 = (size_t)nbx[1] * SW + nbx[0];
                            if (qseen[g2] || !(b1664_field[g2] >= T)) continue;
                            qseen[g2] = 1;
                            qqueue.push_back((int32_t)g2);
                        }
                    }
                    if (sz > largest) largest = sz;
                }
            }
            o.qual_area = (double)largest * (double)(BLOB_STEP * SUB) * (double)(BLOB_STEP * SUB);
        }
    }

    // Ocean proximity inside the region: a chamfer distance transform over
    // the oceanic-biome mask on the sublattice, read at the headline and
    // reported in blocks. Open ocean past the region's edge is invisible
    // to this scan; ocean_spiral_scan covers that side.
    if (have_hl) {
        const size_t N = (size_t)SW * SH;
        thread_local std::vector<float> dt;
        dt.assign(N, 1e9f);
        bool any = false;
        for (int32_t b = 0; b < SH; b++) {
            const int32_t j = jE + 2 * b;
            const size_t row = (size_t)(j - blob.minJ) * BW;
            for (int32_t a = 0; a < SW; a++) {
                const int32_t i = iE + 2 * a;
                if (ov[row + (size_t)(i - blob.minI)] > 0.5) {
                    dt[(size_t)b * SW + a] = 0.0f;
                    any = true;
                }
            }
        }
        if (any) {
            const float SQRT2 = 1.41421356f;
            for (int32_t b = 0; b < SH; b++) {
                for (int32_t a = 0; a < SW; a++) {
                    const size_t cc = (size_t)b * SW + a;
                    if (dt[cc] == 0.0f) continue;
                    float m = dt[cc];
                    if (a > 0)             m = std::min(m, dt[cc - 1]      + 1.0f);
                    if (b > 0)             m = std::min(m, dt[cc - SW]     + 1.0f);
                    if (a > 0 && b > 0)    m = std::min(m, dt[cc - SW - 1] + SQRT2);
                    if (a < SW-1 && b > 0) m = std::min(m, dt[cc - SW + 1] + SQRT2);
                    dt[cc] = m;
                }
            }
            for (int32_t b = SH - 1; b >= 0; b--) {
                for (int32_t a = SW - 1; a >= 0; a--) {
                    const size_t cc = (size_t)b * SW + a;
                    if (dt[cc] == 0.0f) continue;
                    float m = dt[cc];
                    if (a < SW-1)              m = std::min(m, dt[cc + 1]      + 1.0f);
                    if (b < SH-1)              m = std::min(m, dt[cc + SW]     + 1.0f);
                    if (a < SW-1 && b < SH-1)  m = std::min(m, dt[cc + SW + 1] + SQRT2);
                    if (a > 0 && b < SH-1)     m = std::min(m, dt[cc + SW - 1] + SQRT2);
                    dt[cc] = m;
                }
            }
            // Snap the headline to the nearest sublattice cell.
            int32_t ia = (int32_t)std::lrint(((double)(hx - bx) / BLOB_STEP - iE) / 2.0);
            int32_t ib = (int32_t)std::lrint(((double)(hz - bz) / BLOB_STEP - jE) / 2.0);
            ia = std::max(0, std::min(SW - 1, ia));
            ib = std::max(0, std::min(SH - 1, ib));
            o.ocean_dt_hl = (double)dt[(size_t)ib * SW + ia] * (double)(BLOB_STEP * SUB);
        }
    }
}

// ---------------------------------------------------------------------------
// Setting (scenery) feature detection: rivers, lakes, fjords, cliffs, and
// other neighbors that make a region's surroundings interesting.
// Everything here is measured relative to the headline coordinate (the
// point the output row teleports to), chosen by the windowed stats above.
// ---------------------------------------------------------------------------

// Sixteen rays cast from the headline at a 128-block pitch, out to 4,800
// blocks, reading truncated continentalness along the way. A ray that dips
// into ocean-like continentalness and comes back to land has crossed a
// fjord, inlet, or inland sea; a ray that never comes back ends in open
// ocean. Ocean hits are also bucketed into eight compass sectors, which is
// how isthmus geometry (ocean in opposite directions) is detected.
struct VicinityRays {
    double sea_min = NaN;   // nearest ocean-ish hit, blocks
    int    open_nonfrz = 0; // rays ending in non-frozen ocean at 4800 blocks
    int    open_warm = 0;   // ... of which warm (0B temp >= 0.55)
    double fjord = 0.0;     // best excursion quality, [0, 1]
    bool   isthmus = false; // ocean in sectors >= 135 degrees apart
};

static VicinityRays vicinity_ray_scan(MeasureCtx &c, int32_t hx, int32_t hz) {
    constexpr int RAYS = 16;
    constexpr int32_t STEP = 128, RMAX = 4800;
    VicinityRays v;
    bool sector_hit[8] = {};
    for (int k = 0; k < RAYS; k++) {
        const double ang = k * (3.14159265358979323846 / (RAYS / 2));
        const double ca = std::cos(ang), sa = std::sin(ang);
        bool in_oc = false, end_oc = false;
        int32_t oc_start = 0;
        double oc_temp = 0.0;
        for (int32_t d = STEP; d <= RMAX; d += STEP) {
            const int32_t xq = (int32_t)std::lrint(hx + d * ca) >> 2;
            const int32_t zq = (int32_t)std::lrint(hz + d * sa) >> 2;
            const bool oc = sampleClimatePara(&c.contB, nullptr, xq, zq) <= CONT_SHORE;
            if (oc) {
                if (!in_oc) { in_oc = true; oc_start = d; }
                oc_temp = sampleClimatePara(&c.tempB, nullptr, xq, zq);
                if (std::isnan(v.sea_min) || d < v.sea_min) v.sea_min = d;
                sector_hit[k >> 1] = true;
            } else if (in_oc) {
                // The excursion ended: fjord/inlet/inland-sea crossing.
                const int32_t L = d - oc_start;
                const double prox = oc_start <= 640 ? 1.0
                                  : oc_start >= 3200 ? 0.35
                                  : 1.0 - 0.65 * (double)(oc_start - 640) / 2560.0;
                const double lenq = L < 192 ? 0.0
                                  : L < 384 ? (double)(L - 192) / 192.0
                                  : L <= 2560 ? 1.0
                                  : std::max(0.25, 1.0 - (double)(L - 2560) / 1280.0);
                if (prox * lenq > v.fjord) v.fjord = prox * lenq;
                in_oc = false;
            }
            end_oc = oc;
        }
        if (end_oc) {
            if (oc_temp >= -0.45) v.open_nonfrz++;
            if (oc_temp >= 0.55)  v.open_warm++;
        }
    }
    for (int i = 0; i < 8 && !v.isthmus; i++)
        for (int j = i + 3; j <= i + 5 && j < 8; j++)
            if (sector_hit[i] && sector_hit[j]) { v.isthmus = true; break; }
    return v;
}

// Blob-anchored scenery features: isolated river lakes, coastal cliffs, large
// gem forests, the waterscape ray scan, and the oddball-biome vicinity scan.
static void setting_features(MeasureCtx &c, const BlobFill &blob,
        const std::vector<uint8_t> &inblob, const std::vector<double> &hv,
        const std::vector<double> &ov, const std::vector<double> &rv,
        const std::vector<uint8_t> &gv, EnrichStats &o,
        int32_t hx, int32_t hz, int32_t bx, int32_t bz) {
    const int32_t W2 = blob.maxI - blob.minI + 1;
    const int32_t H2 = blob.maxJ - blob.minJ + 1;
    if (W2 <= 0 || H2 <= 0) return;
    auto gidx = [&](int32_t i, int32_t j) {
        return (size_t)(j - blob.minJ) * W2 + (i - blob.minI);
    };

    // -- Isolated river lakes: small interior river components on valley
    // floors with tall walls nearby, in an otherwise river-scarce region.
    {
        const bool river_scarce = !std::isnan(o.bio[CEN_RIVER]) && o.bio[CEN_RIVER] <= 0.05;
        thread_local std::vector<uint8_t> rseen;
        rseen.assign((size_t)W2 * H2, 0);
        thread_local std::vector<int32_t> rqueue;
        thread_local std::vector<int32_t> rcomp;
        int lakes = 0;
        for (const auto &cl : blob.cells) {
            const int32_t i = cl.first, j = cl.second;
            if ((i % SUB) != 0 || (j % SUB) != 0) continue;
            const size_t g0 = gidx(i, j);
            if (rseen[g0] || rv[g0] < 0.5) continue;
            rcomp.clear();
            int64_t sz = 0;
            double hsum = 0.0;
            bool exterior = false; // touches the blob edge or the bbox rim
            rseen[g0] = 1;
            rqueue.push_back((int32_t)g0);
            while (!rqueue.empty()) {
                const int32_t g2 = rqueue.back();
                rqueue.pop_back();
                rcomp.push_back(g2);
                sz++;
                const int32_t ci = (int32_t)(g2 % W2) + blob.minI;
                const int32_t cj = (int32_t)(g2 / W2) + blob.minJ;
                if (!std::isnan(hv[g2])) hsum += hv[g2];
                const int32_t nbr[4][2] = {{ci-SUB,cj},{ci+SUB,cj},{ci,cj-SUB},{ci,cj+SUB}};
                for (const auto &nnp : nbr) {
                    if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                        nnp[1] < blob.minJ || nnp[1] > blob.maxJ ||
                        !inblob[gidx(nnp[0], nnp[1])]) { exterior = true; continue; }
                    const size_t g3 = gidx(nnp[0], nnp[1]);
                    if (rseen[g3] || rv[g3] < 0.5) continue;
                    rseen[g3] = 1;
                    rqueue.push_back((int32_t)g3);
                }
            }
            if (!river_scarce || sz > 14 || exterior) continue;
            if (hsum / (double)sz > 110.0) continue; // valley floor, not a plateau pond
            double ring_max = -1e9;
            double cx_sum = 0.0, cz_sum = 0.0;
            for (const int32_t g2 : rcomp) {
                const int32_t ci = (int32_t)(g2 % W2) + blob.minI;
                const int32_t cj = (int32_t)(g2 / W2) + blob.minJ;
                cx_sum += bx + ci * BLOB_STEP;
                cz_sum += bz + cj * BLOB_STEP;
                for (int32_t dj = -4; dj <= 4; dj += 2) {
                    for (int32_t di = -4; di <= 4; di += 2) {
                        if (di == 0 && dj == 0) continue;
                        if (ci + di < blob.minI || ci + di > blob.maxI ||
                            cj + dj < blob.minJ || cj + dj > blob.maxJ) continue;
                        const double hn = hv[gidx(ci + di, cj + dj)];
                        if (!std::isnan(hn) && hn > ring_max) ring_max = hn;
                    }
                }
            }
            if (ring_max < 190.0) continue; // needs tall walls nearby
            const double ddx = cx_sum / sz - hx, ddz = cz_sum / sz - hz;
            if (ddx * ddx + ddz * ddz > 4000.0 * 4000.0) continue; // near the core
            lakes++;
        }
        o.lake_n = (double)lakes;
    }

    // -- Coastal cliffs: in-blob ocean cells with tall terrain within 256.
    {
        int64_t oc_cells = 0, cliffy = 0;
        for (const auto &cl : blob.cells) {
            const int32_t i = cl.first, j = cl.second;
            if ((i % SUB) != 0 || (j % SUB) != 0) continue;
            if (ov[gidx(i, j)] < 0.5) continue;
            oc_cells++;
            bool tall = false;
            for (int32_t dj = -4; dj <= 4 && !tall; dj += 2)
                for (int32_t di = -4; di <= 4 && !tall; di += 2) {
                    if (di == 0 && dj == 0) continue;
                    if (i + di < blob.minI || i + di > blob.maxI ||
                        j + dj < blob.minJ || j + dj > blob.maxJ) continue;
                    const double hn = hv[gidx(i + di, j + dj)];
                    if (!std::isnan(hn) && hn >= 190.0) tall = true;
                }
            if (tall) cliffy++;
        }
        o.cliff_frac = oc_cells > 0 ? (double)cliffy / (double)oc_cells
                                    : std::numeric_limits<double>::quiet_NaN();
    }

    // -- Large old-growth taiga and cherry grove patches: the biggest
    // connected same-biome component of each inside the region.
    {
        thread_local std::vector<uint8_t> gseen;
        gseen.assign((size_t)W2 * H2, 0);
        thread_local std::vector<int32_t> gqueue;
        int64_t largest_mtg = 0, largest_chy = 0;
        for (const auto &cl : blob.cells) {
            const int32_t i = cl.first, j = cl.second;
            if ((i % SUB) != 0 || (j % SUB) != 0) continue;
            const size_t g0 = gidx(i, j);
            if (gseen[g0] || gv[g0] == 0) continue;
            const uint8_t kind = gv[g0];
            int64_t sz = 0;
            gseen[g0] = 1;
            gqueue.push_back((int32_t)g0);
            while (!gqueue.empty()) {
                const int32_t g2 = gqueue.back();
                gqueue.pop_back();
                sz++;
                const int32_t ci = (int32_t)(g2 % W2) + blob.minI;
                const int32_t cj = (int32_t)(g2 / W2) + blob.minJ;
                const int32_t nbr[4][2] = {{ci-SUB,cj},{ci+SUB,cj},{ci,cj-SUB},{ci,cj+SUB}};
                for (const auto &nnp : nbr) {
                    if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                        nnp[1] < blob.minJ || nnp[1] > blob.maxJ) continue;
                    const size_t g3 = gidx(nnp[0], nnp[1]);
                    if (gseen[g3] || !inblob[g3] || gv[g3] != kind) continue;
                    gseen[g3] = 1;
                    gqueue.push_back((int32_t)g3);
                }
            }
            if (kind == 1 && sz > largest_mtg) largest_mtg = sz;
            if (kind == 2 && sz > largest_chy) largest_chy = sz;
        }
        const double cell_area = (double)(BLOB_STEP * SUB) * (double)(BLOB_STEP * SUB);
        o.mtg_comp_m = (double)largest_mtg * cell_area;
        o.chy_comp_m = (double)largest_chy * cell_area;
    }

    // -- The waterscape ray scan from the headline.
    {
        const VicinityRays vr = vicinity_ray_scan(c, hx, hz);
        o.vic_sea_min   = vr.sea_min;
        o.vic_open_oc   = (double)vr.open_nonfrz;
        o.vic_open_warm = (double)vr.open_warm;
        o.vic_fjord     = vr.fjord;
        o.vic_isthmus   = vr.isthmus ? 1.0 : 0.0;
    }

    // -- Vicinity biome scan: square rings at a 256-block pitch out to
    // 4,800 blocks, counting the interesting neighboring biomes (eroded
    // badlands, mushroom fields, flower forest, cherry grove, old-growth
    // taiga). Points inside the region are skipped because the census covers
    // those.
    {
        constexpr int32_t VSTEP = 256, VRMAX = 4800;
        double badl = 0, badl_oth = 0, mushb = 0, flow = 0, vch = 0, vmtg = 0;
        for (int32_t r = VSTEP; r <= VRMAX; r += VSTEP) {
            const int32_t half = r / VSTEP;
            const int32_t per = 8 * half;
            for (int32_t k = 0; k < per; k++) {
                const int32_t side = k / (2 * half);
                const int32_t t = (k % (2 * half)) - half;
                int32_t dx, dz;
                switch (side) {
                case 0:  dx = t;     dz = -half; break;
                case 1:  dx = half;  dz = t;     break;
                case 2:  dx = -t;    dz = half;  break;
                default: dx = -half; dz = -t;    break;
                }
                const int32_t x = hx + dx * VSTEP;
                const int32_t z = hz + dz * VSTEP;
                const int32_t li = (int32_t)std::lrint((double)(x - bx) / BLOB_STEP);
                const int32_t lj = (int32_t)std::lrint((double)(z - bz) / BLOB_STEP);
                if (li >= blob.minI && li <= blob.maxI &&
                    lj >= blob.minJ && lj <= blob.maxJ && inblob[gidx(li, lj)])
                    continue;
                switch (getBiomeAt(&c.g, 1, x, 256, z)) {
                case eroded_badlands:         badl++;     break;
                case badlands:
                case wooded_badlands:         badl_oth++; break;
                case mushroom_fields:
                case mushroom_field_shore:    mushb++;    break;
                case flower_forest:           flow++;     break;
                case cherry_grove:            vch++;      break;
                case old_growth_pine_taiga:
                case old_growth_spruce_taiga: vmtg++;     break;
                default: break;
                }
            }
        }
        o.vic_badl = badl; o.vic_badl_oth = badl_oth; o.vic_mush_b = mushb;
        o.vic_flower = flow; o.vic_cherry = vch; o.vic_mtg = vmtg;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Measurement context setup (exported; shared by cpu.cpp and the prober)
// ---------------------------------------------------------------------------
void measure_ctx_setup(MeasureCtx &c, int mc_version) {
    setupGenerator(&c.g, mc_version, 0); // 1.18+ biome-noise path
}

void measure_ctx_seed(MeasureCtx &c, uint64_t seed) {
    applySeed(&c.g, DIM_OVERWORLD, seed);
    setClimateParaSeed(&c.eroB,  seed, 0, NP_EROSION,         2);  // 0A+0B
    setClimateParaSeed(&c.eroF,  seed, 0, NP_EROSION,        -1);  // all octaves
    setClimateParaSeed(&c.contB, seed, 0, NP_CONTINENTALNESS, 4);  // 0A..1B
    setClimateParaSeed(&c.contF, seed, 0, NP_CONTINENTALNESS,-1);
    setClimateParaSeed(&c.tempB, seed, 0, NP_TEMPERATURE,     2);  // 0A+0B
    setClimateParaSeed(&c.weird, seed, 0, NP_WEIRDNESS,      -1);
}

// ---------------------------------------------------------------------------
// Blob fill + core radius (exported)
// ---------------------------------------------------------------------------
BlobFill blob_fill(MeasureCtx &c, const VerifyConfig &cfg, int32_t bx, int32_t bz) {
    const int32_t DIM = BLOB_DIM, HALF = BLOB_HALF;
    const size_t NCELL = (size_t)DIM * (size_t)DIM;

    // World-align the measurement lattice: snap the anchor down to a multiple
    // of 64 blocks (two's-complement floor), so every fill samples the same
    // absolute grid regardless of which anchor triggered it.
    const int32_t bx0 = bx & ~63;
    const int32_t bz0 = bz & ~63;

    thread_local std::vector<uint8_t> seen;
    thread_local std::vector<int32_t> queue;
    seen.assign(NCELL, 0);
    queue.clear();
    queue.reserve(1u << 16);

    auto idx = [&](int32_t i, int32_t j) { return (j + HALF) * DIM + (i + HALF); };

    BlobFill out;
    out.origin_x = bx0;
    out.origin_z = bz0;

    int32_t si = 0, sj = 0;
    if (!cell_ok(c, cfg, bx0, bz0)) {
        bool found = false;
        for (int32_t r = 1; r <= 2 && !found; r++)
            for (int32_t j = -r; j <= r && !found; j++)
                for (int32_t i = -r; i <= r && !found; i++)
                    if (cell_ok(c, cfg, bx0 + i * BLOB_STEP, bz0 + j * BLOB_STEP)) {
                        si = i; sj = j; found = true;
                    }
        if (!found) return out; // empty blob
    }

    seen[idx(si, sj)] = 1;
    queue.push_back(idx(si, sj));
    out.cells.emplace_back((int16_t)si, (int16_t)sj);
    out.minI = out.maxI = si;
    out.minJ = out.maxJ = sj;
    out.edge = (si == -HALF || si == HALF || sj == -HALF || sj == HALF);

    while (!queue.empty()) {
        const int32_t cc = queue.back();
        queue.pop_back();
        const int32_t i = (cc % DIM) - HALF;
        const int32_t j = (cc / DIM) - HALF;
        const int32_t nb[4][2] = {{i-1,j},{i+1,j},{i,j-1},{i,j+1}};
        for (const auto &n : nb) {
            if (n[0] < -HALF || n[0] > HALF || n[1] < -HALF || n[1] > HALF) continue;
            const int32_t nc = idx(n[0], n[1]);
            if (seen[nc]) continue;
            seen[nc] = 1;
            if (!cell_ok(c, cfg, bx0 + n[0] * BLOB_STEP, bz0 + n[1] * BLOB_STEP)) continue;
            queue.push_back(nc);
            out.cells.emplace_back((int16_t)n[0], (int16_t)n[1]);
            if (n[0] < out.minI) out.minI = n[0];
            if (n[0] > out.maxI) out.maxI = n[0];
            if (n[1] < out.minJ) out.minJ = n[1];
            if (n[1] > out.maxJ) out.maxJ = n[1];
            if (n[0] == -HALF || n[0] == HALF || n[1] == -HALF || n[1] == HALF) out.edge = 1;
        }
    }

    if (cfg.close1) {
        // One-cell morphological closing: dilate (an unset cell with any set
        // 4-neighbor joins) over a mask with a 2-cell margin, then erode
        // (drop cells adjacent to any unset 4-neighbor). The dilation ring
        // is stripped back off by the erosion, so original region cells
        // always survive, and the measured area can only grow by bridged
        // one-cell gaps, never shrink.
        const int32_t W = out.maxI - out.minI + 5;
        const int32_t H = out.maxJ - out.minJ + 5;
        thread_local std::vector<uint8_t> mask, dil;
        mask.assign((size_t)W * H, 0);
        dil.assign((size_t)W * H, 0);
        auto midx = [&](int32_t i, int32_t j) {
            return (j - out.minJ + 2) * W + (i - out.minI + 2);
        };
        for (const auto &cl : out.cells)
            mask[midx(cl.first, cl.second)] = 1;
        // Dilate: any cell with at least one set 4-neighbor becomes set.
        for (int32_t j = 1; j < H - 1; j++) {
            for (int32_t i = 1; i < W - 1; i++) {
                const size_t c = (size_t)j * W + i;
                dil[c] = (uint8_t)(mask[c] | mask[c-1] | mask[c+1] | mask[c-W] | mask[c+W]);
            }
        }
        // Erode: keep only dilated cells whose 4 neighbors are all dilated.
        out.cells.clear();
        for (int32_t j = 1; j < H - 1; j++) {
            for (int32_t i = 1; i < W - 1; i++) {
                const size_t c = (size_t)j * W + i;
                if (!dil[c]) continue;
                if (dil[c-1] & dil[c+1] & dil[c-W] & dil[c+W]) {
                    const int32_t li = i - 2 + out.minI;
                    const int32_t lj = j - 2 + out.minJ;
                    out.cells.emplace_back((int16_t)li, (int16_t)lj);
                }
            }
        }
        // Recompute bbox after closing (can only grow via bridged cells).
        if (!out.cells.empty()) {
            out.minI = out.maxI = out.cells[0].first;
            out.minJ = out.maxJ = out.cells[0].second;
            for (const auto &cl : out.cells) {
                out.minI = std::min(out.minI, (int32_t)cl.first);
                out.maxI = std::max(out.maxI, (int32_t)cl.first);
                out.minJ = std::min(out.minJ, (int32_t)cl.second);
                out.maxJ = std::max(out.maxJ, (int32_t)cl.second);
            }
        }
    }

    out.area = (int64_t)out.cells.size() * BLOB_STEP * BLOB_STEP;

    // The inscribed core radius: a two-pass chamfer distance transform over
    // the region's bounding box (plus a 1-cell margin). The largest value
    // on the map is the radius, in lattice cells, of the biggest circle
    // that fits entirely inside the region.
    {
        const int32_t W = out.maxI - out.minI + 3;
        const int32_t H = out.maxJ - out.minJ + 3;
        thread_local std::vector<uint8_t> mask;
        thread_local std::vector<float> dt;
        mask.assign((size_t)W * H, 0);
        dt.assign((size_t)W * H, 0.0f);
        auto midx = [&](int32_t i, int32_t j) {
            return (j - out.minJ + 1) * W + (i - out.minI + 1);
        };
        for (const auto &cl : out.cells)
            mask[midx(cl.first, cl.second)] = 1;

        const float BIG = 1e9f;
        const float SQRT2 = 1.41421356f;
        for (int32_t j = 0; j < H; j++) {
            for (int32_t i = 0; i < W; i++) {
                const size_t cc = (size_t)j * W + i;
                if (!mask[cc]) continue;
                dt[cc] = BIG;
            }
        }
        for (int32_t j = 0; j < H; j++) {
            for (int32_t i = 0; i < W; i++) {
                const size_t cc = (size_t)j * W + i;
                if (dt[cc] == 0.0f) continue;
                float m = dt[cc];
                if (i > 0)          m = std::min(m, dt[cc - 1]     + 1.0f);
                if (j > 0)          m = std::min(m, dt[cc - W]     + 1.0f);
                if (i > 0 && j > 0) m = std::min(m, dt[cc - W - 1] + SQRT2);
                if (i < W-1 && j>0) m = std::min(m, dt[cc - W + 1] + SQRT2);
                dt[cc] = m;
            }
        }
        float maxdt = 0.0f;
        for (int32_t j = H - 1; j >= 0; j--) {
            for (int32_t i = W - 1; i >= 0; i--) {
                const size_t cc = (size_t)j * W + i;
                if (dt[cc] == 0.0f) continue;
                float m = dt[cc];
                if (i < W-1)            m = std::min(m, dt[cc + 1]     + 1.0f);
                if (j < H-1)            m = std::min(m, dt[cc + W]     + 1.0f);
                if (i < W-1 && j < H-1) m = std::min(m, dt[cc + W + 1] + SQRT2);
                if (i > 0 && j < H-1)   m = std::min(m, dt[cc + W - 1] + SQRT2);
                dt[cc] = m;
                if (m > maxdt) maxdt = m;
            }
        }
        out.core_cells = (int32_t)lrintf(maxdt);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Composite Score Constants
// 
// These are responsible for determining a seed's score. In my current
// calibration, for example, densely-clustered prominent peaks with moderate
// footprints surrounded by a deep network of valleys yield a high score.
//
// Each stat is z-scored against fixed reference constants (pooled references
// from the labeled backlog; weights = Cohen's d per feature/Fisher discriminant).
// The constants are deliberately fixed, rather than recomputed per input file,
// so scores are comparable across runs. 
// 
// To recalibrate: 
// 1. Take output.txt from a long run (or a concatenation of several runs)
// 2. Run this command:
//      ./mountain_rangefinder --verify output.txt --output output_verified.txt
// 3. Generate a semi-randomized seedlist CSV from verified.txt with
//      python3 seedlab.py recal output_verified.txt --todo 150 --out todo.csv
// 4. Open todo.csv in Calc or Excel, and copy the seedlist column into Cubiomes
//      Viewer, under the Search > Seeds tab; also load the session called
//      "Cubiomes_Viewer_MountainRangeFinder_seed_verifier_v#.session"
//      from the project root directory
// 5. In Cubiomes Viewer, switch to the Locations tab (next to the Seeds tab)
//      and click Analyze, then Expand all, and then click on the Spiral Iterator
//      (main) entry to move the map over the proper mountain region.
// 6. Then, one by one, rank each seed either 0 or 1 -- 0 if you don't like it,
//      and 1 if you do. Try to get at least 15 or 20 1s, or more if you like.
// 7. Once you marked every seed (i.e. the label column is filled with 0s and 1s),
//      delete all columns but the seed and the label, and delete the header row.
//      Then close todo.csv and rename it to labels.csv.
// 8. Run this command to generate an analysis based on your rankings:
//      python3 seedlab.py recal output_verified.txt --labels labels.csv
// 9. Take the C++ code generated in the output (specifically the block that
//      starts with "const Term terms[] = {" without quotes) and replace it with
//      the existing const Term block below.
// 10.Run this command to apply your changes:
//      make clean && make
// ---------------------------------------------------------------------------

// A NaN-safe weighted average of one aspect's terms. If an aspect has no
// measured terms at all, it scores low (-3) rather than being silently
// skipped, so missing data can never inflate a score.
struct AspectAcc {
    double sum = 0.0, wsum = 0.0;
    void add(double v, double w) { if (!std::isnan(v)) { sum += w * v; wsum += w; } }
    double get() const { return wsum > 0.0 ? sum / wsum : -3.0; }
};

// The gated aspects. amp, tex, frag, and depth are computed here; the dark
// forest aspect joins them in compute_score via score_dark_aspect. The
// five describe the region's core pattern along separate axes, and a great
// region has to be good at all of them at once. A single linear score
// would let one axis compensate for another (that's how "small but
// dissected" could outrank "tall, deep, AND dissected"), so the final
// score is built on the weakest aspect:
//
//   amp   = tall, prominent peaks
//           (hMax, peakProm, bw1664hStd, bw1664Hi)
//   tex   = fine-grained ruggedness and valley-network density
//           (bw1664Dh, bw896Dh, vcross)
//   frag  = many packed individual peaks rather than one plateau massif
//           (highComps up, highLargest down)
//   depth = deep valley floors and gorge adjacency
//           (w896hMeanMin, hMin, bw1664Lo, gorgeFrac)
//
// The reference constants come from the calibration set (the same fit that
// produced the consensus table in compute_score). The window-contrast
// reference is in physical units (blocks): an in-window height spread of
// 15 is ordinary, while 30+ means peaks and valleys packed into one
// neighborhood. If the reference region's amp aspect ever reads low,
// re-center that constant near 0.6x its measured w896hStdMax column in
// magic_seed.txt.
void score_aspects(const EnrichStats &e, double &amp, double &tex, double &frag, double &depth) {
    const double z_hmax   = (e.h_max        - 242.078)  / 10.348;
    const double z_prom   = (e.peak_prom    - 36.9412)  / 2.00566;
    const double z_vcross = (e.vcross       - 0.300176) / 0.0167818;
    const double z_hcomps = (e.high_comps   - 0.615029) / 0.243762;
    const double z_hlarg  = (e.high_largest - 0.237651) / 0.104692;
    // Coherent best-pattern-window stats (bw1664 = pattern window; bw896 =
    // fine-grain check). References recentered on the magic seed's measured
    // values; if you recenter WREF_HI/WREF_LO above, change z_bwhi/z_bwlo too.
    const double z_bwstd  = (e.bw1664_hstd   - 40.0)    / 12.0;
    const double z_bwhi   = (e.bw1664_hifrac - 0.10)    / 0.10;
    const double z_bwdh   = (e.bw1664_dh     - 36.0)    / 6.0;
    const double z_bwdh9  = (e.bw896_dh      - 36.0)    / 6.0;
    // Depth: the region's deepest 896-block bowl (its valley floor), how
    // close the region gets to sea level at all, the share of sea-level
    // valley floors inside the best window, and gorge adjacency. Measuring
    // depth from the window's mean height does not work: the best window
    // sits on the peaks, which makes even a deeply cut region read as
    // shallow.
    const double z_bowl   = (90.0 - e.w896_hmean_min)   / 12.0;
    const double z_hmin   = (90.0 - e.h_min)            / 20.0;
    const double z_bwlo   = (e.bw1664_lofrac - 0.02)    / 0.03;
    // Gorge adjacency: the share of low cells with a tall wall within 256
    // blocks, i.e. narrow deep floors beside tall walls. Tuning: re-center
    // the reference on the reference region's gorgeFrac column after
    // re-verifying the corpus.
    const double z_gorge  = (e.gorge_frac - 0.25)       / 0.10;

    AspectAcc a, t, f, d;
    a.add(z_hmax, 0.30);   a.add(z_prom, 0.25);   a.add(z_bwstd, 0.25); a.add(z_bwhi, 0.20);
    t.add(z_bwdh, 0.55);   t.add(z_bwdh9, 0.30);  t.add(z_vcross, 0.15);
    f.add(z_hcomps, 0.55); f.add(-z_hlarg, 0.45);
    d.add(z_bowl, 0.45);   d.add(z_hmin, 0.35);   d.add(z_bwlo, 0.20);  d.add(z_gorge, 0.20);

    amp = a.get(); tex = t.get(); frag = f.get(); depth = d.get();
}

// Extent: how large the pattern is. Combines the region's area
// (with saturation, so outliers stop gaining), its inscribed
// core width, and the pattern score of the best 3,200-block window, which
// can only score well if the pattern holds up at scale.
double score_extent(int64_t area, int32_t core_cells, const EnrichStats &e) {
    double z_area = ((double)area / 1e6 - 20.0) / 4.0;
    z_area = z_area > 3.5 ? 3.5 : (z_area < -2.0 ? -2.0 : z_area); // cap 2.5 -> 3.5
    AspectAcc x;
    x.add(z_area, 0.25);
    x.add(((double)core_cells - 21.0) / 4.0, 0.15);
    x.add(e.bw3200_sc, 0.35);
    // Three more extent signals: qual_area, how much of the region scores
    // like its best part (the superlevel set of the 1,664-block window
    // score field); scQ90, the 90th-percentile window score, which measures
    // how consistently the pattern shows up (in the calibration corpus it
    // was one of the strongest separators, Cohen's d = +0.98, precisely
    // because it punishes one-lucky-window regions); and mtnCompM, the
    // size of the largest connected mountain-biome stretch.
    if (!std::isnan(e.qual_area)) {
        double zq = ((double)e.qual_area / 1e6 - 2.0) / 1.0;
        x.add(zq > 2.5 ? 2.5 : (zq < -2.0 ? -2.0 : zq), 0.15);
    }
    if (!std::isnan(e.sc_q90)) {
        // Corpus-pooled constants from the 628k-row fit.
        double zs = (e.sc_q90 - (-1.28639)) / 0.778071;
        x.add(zs > 2.5 ? 2.5 : (zs < -2.0 ? -2.0 : zs), 0.15);
    }
    if (!std::isnan(e.mtn_comp_m)) {
        double zm = ((double)e.mtn_comp_m / 1e6 - 2.5) / 1.0;
        x.add(zm > 2.5 ? 2.5 : (zm < -2.0 ? -2.0 : zm), 0.10);
    }
    return x.get();
}

// The dark-forest aspect, measured on the best pattern window
// (bw1664_dark), not on the ring around the headline. A ring over-counts
// peripheral dark forest and would penalize good regions whose dark forest
// sits at the region's edges rather than its core. The ring stays in the
// output as a diagnostic, and as a safety net at extreme levels below.
double score_dark_aspect(const EnrichStats &e) {
    if (std::isnan(e.bw1664_dark)) return 0.0;
    // Window share: 5% is neutral, then one notch per extra 0.5% (5.5% ->
    // -1, 6% -> -2, 6.5% -> -3), clamped at -4 from 7% on. The cliff is
    // deliberately steep: the reference region measures 3.6% dark forest
    // in its best window and reads as clean, while a rejected calibration
    // seed measured 7.1% inside its best window and read as infested.
    const double z_win = (0.05 - e.bw1664_dark) / 0.005;
    // The safety net: a headline ring above 25% dark forest means the area
    // around the reported point is dark-forest-dominated, and the seed is
    // vetoed even if the best window happens to sit in a clear patch.
    const double z_ring = std::isnan(e.dark_core_frac) ? 1.5
                          : (0.25 - e.dark_core_frac) / 0.05;
    const double z = std::min(z_win, z_ring);
    return z > 1.5 ? 1.5 : (z < -4.0 ? -4.0 : z);
}

// The nearest measured ocean to the headline, whichever probe saw it
// closest. Not used by the current score; kept for reference.
[[maybe_unused]] static double ocean_distance(const EnrichStats &e) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double d = nan;
    if (!std::isnan(e.ocean_dt_hl))  d = e.ocean_dt_hl;
    if (!std::isnan(e.ocean_spiral)) d = std::isnan(d) ? e.ocean_spiral
                                                       : std::min(d, e.ocean_spiral);
    return d;
}

// The earlier ocean-proximity bonus: a small score lift for ocean near
// (but not inside) the region's best window. Not used by the current
// score; setting_parts() below took over this job. Kept for reference.
[[maybe_unused]] static double icing_score(double odist, double ocn_frac) {
    if (std::isnan(odist) || std::isnan(ocn_frac)) return 0.0;
    if (ocn_frac > 0.08) return 0.0; // window is already a sea cliff; not "icing"
    double t;
    if      (odist <  200.0)  t = 0.0;
    else if (odist <  640.0)  t = (odist - 200.0) / 440.0;
    else if (odist <= 3000.0) t = 1.0;
    else if (odist <  4800.0) t = (4800.0 - odist) / 1800.0;
    else                      t = 0.0;
    return 0.5 * t;
}

// ---------------------------------------------------------------------------
// The setting (scenery) bonus. Rather than generic proximity flags, it
// detects actual features: waterscapes (a proximity ramp plus fjord and
// inlet excursions plus broad non-frozen open ocean), isthmus geometry,
// coastal cliffs, isolated river lakes in the valleys, and unusual
// neighboring biomes (eroded badlands, mushroom fields, flower forest, big
// cherry groves or old-growth taiga). Each group is capped, and so is the
// total, and the whole bonus is gated on the core pattern's quality: a
// weak core gets zero scenery credit, so the setting can break ties among
// good cores but never rescue a bad one.
// ---------------------------------------------------------------------------
struct SettingParts {
    double water = 0.0, isthmus = 0.0, cliffs = 0.0, lakes = 0.0, oddball = 0.0;
    double gate = 0.0, total = 0.0;
};

static double ramp_sea(double od) {
    if (std::isnan(od)) return 0.0;
    if (od < 200.0)     return 0.0;
    if (od < 640.0)     return (od - 200.0) / 440.0;
    if (od <= 3000.0)   return 1.0;
    if (od < 4800.0)    return (4800.0 - od) / 1800.0;
    return 0.0;
}

static SettingParts setting_parts(const EnrichStats &e, double key) {
    SettingParts sp;
    double od = std::numeric_limits<double>::quiet_NaN();
    if (!std::isnan(e.ocean_dt_hl))  od = e.ocean_dt_hl;
    if (!std::isnan(e.ocean_spiral)) od = std::isnan(od) ? e.ocean_spiral : std::min(od, e.ocean_spiral);
    if (!std::isnan(e.vic_sea_min))  od = std::isnan(od) ? e.vic_sea_min  : std::min(od, e.vic_sea_min);
    double prox = ramp_sea(od);
    if (!std::isnan(e.bw1664_ocn) && e.bw1664_ocn > 0.08)
        prox *= 0.25; // the pattern window is already waterfront: the sea is
                      // the stage there, not the scenery
    const double fjord = std::isnan(e.vic_fjord) ? 0.0 : e.vic_fjord;
    const double ooc   = std::isnan(e.vic_open_oc) ? 0.0 : e.vic_open_oc;
    const double owm   = std::isnan(e.vic_open_warm) ? 0.0 : e.vic_open_warm;
    double open = ooc >= 2.0 ? 0.20 : (ooc >= 1.0 ? 0.10 : 0.0);
    if (owm >= 1.0) open += 0.05;
    sp.water   = std::min(0.45 * prox + 0.25 * fjord + open, 0.55);
    sp.isthmus = (e.vic_isthmus >= 0.5 && sp.water > 0.0) ? 0.15 : 0.0;
    const double cliff = std::isnan(e.cliff_frac) ? 0.0 : e.cliff_frac;
    sp.cliffs  = 0.15 * std::min(1.0, cliff / 0.25);
    const double lakes = std::isnan(e.lake_n) ? 0.0 : e.lake_n;
    sp.lakes   = std::min(0.12 * lakes, 0.30);
    const double badl  = std::isnan(e.vic_badl) ? 0.0 : e.vic_badl;
    const double badlo = std::isnan(e.vic_badl_oth) ? 0.0 : e.vic_badl_oth;
    const double mush  = std::isnan(e.vic_mush_b) ? 0.0 : e.vic_mush_b;
    const double flow  = std::isnan(e.vic_flower) ? 0.0 : e.vic_flower;
    const double vchy  = std::isnan(e.vic_cherry) ? 0.0 : e.vic_cherry;
    const double vmtg  = std::isnan(e.vic_mtg) ? 0.0 : e.vic_mtg;
    const double ccomp = std::isnan(e.chy_comp_m) ? 0.0 : e.chy_comp_m;
    const double mcomp = std::isnan(e.mtg_comp_m) ? 0.0 : e.mtg_comp_m;
    double odd = 0.0;
    if (badl > 0)   odd += std::min(0.15, 0.05 + 0.01 * badl); // eroded badlands
    if (badlo >= 8) odd += 0.06;                                 // badlands band
    if (mush > 0)   odd += std::min(0.15, 0.08 + 0.01 * mush);   // mushroom island (biome)
    if (flow >= 4)  odd += std::min(0.10, 0.04 + 0.01 * flow);   // big flower forest
    if (vchy >= 4 || ccomp >= 400000.0) odd += 0.10;             // big cherry grove
    if (vmtg >= 6 || mcomp >= 800000.0) odd += 0.10;             // vast mega taiga
    sp.oddball = std::min(odd, 0.30);
    const double raw = sp.water + sp.isthmus + sp.cliffs + sp.lakes + sp.oddball;
    sp.gate  = key >= 1.0 ? 1.0 : key <= 0.5 ? 0.0 : (key - 0.5) * 2.0;
    sp.total = std::min(raw, 0.80) * sp.gate;
    return sp;
}

double compute_score(int64_t area, int32_t core_cells, const EnrichStats &e) {

    //   score = 2.00 * min(amp, tex, frag, depth, dark)  the weakest-aspect gate:
    //                                                    the core pattern's quality
    //         + 0.30 * (amp + tex + frag + depth + dark) tiebreak across aspects
    //         + 1.10 * extent                            the pattern at scale
    //                                                    (see score_extent)
    //         + 1.25 * consensus                         the fitted Term table
    //                                                    below, as a weighted MEAN
    //         - 40 * max(0, bw1664_dark - 0.055)         dark forest inside the
    //                                                    best window, past ~5.5%
    //         - massif penalty                           one plateau hogging the
    //                                                    high ground
    //         + setting                                  the scenery bonus, up to
    //                                                    +0.80, core-gated
    //                                                    (see setting_parts)
    //
    // To make raw size matter more or less, tune the 1.10 extent weight and
    // the reference/scale constants in score_extent.
    double a_amp, a_tex, a_frag, a_depth;
    score_aspects(e, a_amp, a_tex, a_frag, a_depth);
    const double a_dark = score_dark_aspect(e);
    const double key = std::min({a_amp, a_tex, a_frag, a_depth, a_dark});

    // Legacy linear model. This Term table is what recalibrate_score.py
    // regenerates (step 9 of the recipe above): paste over it only.
    struct Term { double v, mean, sd, weight; };
    // =============================================================================
    // \/ \/ \/ \/ THIS IS THE TERMS TABLE TO REPLACE WHEN RECALIBRATING \/ \/ \/ \/ 
    // =============================================================================
    // Last recalibrated on 2026-08-27 from 78 positive & 169 negative labels.
    const Term terms[] = {
        { e.bw3200_dh,  29.3085,  2.15306,              +1.17 }, // bw3200Dh
        { e.bw3200_hmean,  113.044,  6.05741,           +1.04 }, // bw3200hMean
        { e.bw3200_sc,  -2.26601,  0.713437,            +1.04 }, // bw3200Sc
        { e.dh_mean,  26.4897,  1.36548,                +0.99 }, // dhMean
        { e.bw1664_dh,  36.0524,  3.02595,              +0.97 }, // bw1664Dh
        { e.bw3200_hifrac,  0.0311663,  0.0160673,      +0.96 }, // bw3200Hi
        { e.peak_prom,  36.8023,  2.02987,              +0.91 }, // peakProm
        { e.high_comps,  0.586944,  0.240445,           +0.90 }, // highComps
        { e.h_p90,  154.232,  6.91857,                  +0.89 }, // hP90
        { e.low,  0.691115,  0.0404762,                 -0.88 }, // low
        { e.high,  0.0165787,  0.00786205,              +0.87 }, // high
        { e.h_mean,  104.98,  4.15171,                  +0.86 }, // hMean
        { e.bw3200_mtn,  0.259228,  0.0430392,          +0.86 }, // bw3200Mtn
        { e.h_std,  35.6727,  2.1216,                   +0.85 }, // hStd
        { e.bw3200_hstd,  38.2035,  2.98361,            +0.84 }, // bw3200hStd
        { e.relief,  0.013502,  0.00466682,             +0.83 }, // relief
        { e.w1664_h_max,  134.301,  7.84006,            +0.81 }, // w1664hMax
        { e.bw3200_ero,  -0.660867,  0.0571423,         -0.81 }, // bw3200Ero
        { e.bw1664_sc,  -0.0102298,  0.88565,           +0.79 }, // bw1664Sc
        { e.h_max,  241.241,  10.3986,                  +0.77 }, // hMax
        { e.eroF_mean,  -0.582397,  0.036329,           -0.76 }, // eroFmean
        { e.bio[CEN_MOUNTAIN],  0.20646,  0.0285928,    +0.74 }, // bioMtn
        { e.bio[CEN_RIVER],  0.0340993,  0.00772917,    -0.71 }, // bioRiv
        { e.bw1664_hmean,  127.056,  9.96771,           +0.70 }, // bw1664hMean
        { e.bw3200_lofrac,  0.0496073,  0.0172566,      -0.68 }, // bw3200Lo
        { e.bw896_dh,  43.9345,  3.78023,               +0.64 }, // bw896Dh
        { e.bw896_sc,  2.20821,  0.985784,              +0.63 }, // bw896Sc
        { e.bw1664_hifrac,  0.0756042,  0.0385493,      +0.59 }, // bw1664Hi
        { e.contF_max,  1.11621,  0.147552,             +0.59 }, // contFmax
        { e.contF_mean,  0.260204,  0.056164,           +0.57 }, // contFmean
        { e.w896_h_max,  153.584,  9.93666,             +0.57 }, // w896hMax
        { e.eroF_p10,  -0.868355,  0.0577004,           -0.55 }, // eroFp10
        { e.high_largest,  0.250865,  0.112722,         -0.52 }, // highLargest
        { e.bw1664_dark,  0.0276971,  0.0375307,        -0.50 }, // bw1664Dark
        { e.bw896_hifrac,  0.157411,  0.0759688,        +0.47 }, // bw896Hi
        { e.bw1664_lofrac,  0.0343086,  0.0230571,      -0.46 }, // bw1664Lo
        { e.w1664_hmean_min,  80.4092,  6.17331,        +0.45 }, // w1664hMeanMin
        { e.bw896_hmean,  143.39,  13.8513,             +0.43 }, // bw896hMean
        { e.w1664_hstd_max,  46.7241,  3.63239,         +0.41 }, // w1664hStdMax
    };
    // A weighted mean: the ~39 terms are heavily intercorrelated,
    // so a sum would pay out several times over for what
    // is really one underlying quality (ruggedness) and would quietly
    // override the weakest-aspect gate. As a mean, this term stays bounded
    // near [-2.5, +2.5]: a consensus index, not a pileup.
    double legacy = 0.0, legacy_wsum = 0.0;
    for (const Term &t : terms) {
        if (!std::isnan(t.v)) {
            double z = (t.v - t.mean) / t.sd;
            if (z > 2.5) z = 2.5; else if (z < -2.5) z = -2.5; // per-term clamp
            legacy += t.weight * z;
            legacy_wsum += std::fabs(t.weight);
        }
    }
    if (legacy_wsum > 0.0) legacy /= legacy_wsum;

    // Dark forest is charged in two places. The weakest-aspect gate
    // handles the headline ring and the local blotches (via the dark
    // aspect); the charge here covers dark forest inside the best pattern
    // window itself, tolerated up to ~5.5%, then billed at 40 per unit of
    // excess share. The gate's dark aspect still carries the hard veto for
    // truly infested windows.
    const double dark_win = std::isnan(e.bw1664_dark) ? 0.0 : e.bw1664_dark;
    const double dark_pen = 40.0 * std::max(0.0, dark_win - 0.055);

    // The massif penalty: a soft charge for one-plateau regions, where the
    // largest connected piece of high ground holds more than 55% of all
    // high ground. The reference region's 0.098 share never triggers it;
    // true plateau regions do.
    const double massif_pen = 3.0 *
        std::max(0.0, (std::isnan(e.high_largest) ? 0.0 : e.high_largest) - 0.55);

    // The setting bonus: the scenery around the region, feature-detected
    // (waterscapes, isthmuses, cliffs, river lakes, unusual neighboring
    // biomes), capped at +0.80, and gated on the core pattern's quality,
    // so it breaks ties among good cores but never rescues a weak one.
    const double setting = setting_parts(e, key).total;

    return 2.0 * key + 0.30 * (a_amp + a_tex + a_frag + a_depth + a_dark)
         + 1.10 * score_extent(area, core_cells, e)
         + 1.25 * legacy - dark_pen - massif_pen + setting;
}

double setting_score(const EnrichStats &e) {
    double a_amp, a_tex, a_frag, a_depth;
    score_aspects(e, a_amp, a_tex, a_frag, a_depth);
    const double key = std::min({a_amp, a_tex, a_frag, a_depth, score_dark_aspect(e)});
    return setting_parts(e, key).total;
}

// Dark-forest fraction of the r=1152 disc around a point (64-block pitch).
// Same sampling pattern as the candidate gate in cpu.cpp, but centered on the
// headline instead of the anchor.
static double dark_ring_frac(MeasureCtx &c, int32_t hx, int32_t hz) {
    constexpr int32_t R = 1152;
    constexpr int32_t STEP = 64;
    const int64_t R2 = (int64_t)R * R;
    int64_t dark = 0, tot = 0;
    for (int32_t x = hx - R; x <= hx + R; x += STEP) {
        for (int32_t z = hz - R; z <= hz + R; z += STEP) {
            const int64_t dx = (int64_t)x - hx, dz = (int64_t)z - hz;
            if (dx * dx + dz * dz > R2) continue;
            tot++;
            if (getBiomeAt(&c.g, 1, x, 256, z) == dark_forest) dark++;
        }
    }
    return tot > 0 ? (double)dark / (double)tot
                   : std::numeric_limits<double>::quiet_NaN();
}

// An outward square-ring scan (128-block pitch, out to 4,800 blocks)
// around a point. Returns the distance to the nearest point whose
// truncated continentalness reads as shore or below; unlike the in-region
// distance transform, this sees open ocean beyond the region's edge.
// Along the way it also flags two noteworthy climates: mushroom-island
// conditions (continentalness <= -1.05) and warm-ocean conditions
// (shore-like continentalness with truncated temperature >= 0.55). The
// distance is NaN when nothing ocean-like is in range. The 4,800-block
// reach covers the farthest distance the scenery bonus still rewards.
static double ocean_spiral_scan(MeasureCtx &c, int32_t hx, int32_t hz,
                                double &vic_mush, double &vic_warmoc) {
    constexpr int32_t STEP = 128, RMAX = 4800;
    double nearest = std::numeric_limits<double>::quiet_NaN();
    vic_mush = 0.0;
    vic_warmoc = 0.0;
    for (int32_t r = STEP; r <= RMAX; r += STEP) {
        const int32_t half = r / STEP;        // ring radius in steps
        const int32_t per  = 8 * half;        // perimeter samples
        for (int32_t k = 0; k < per; k++) {
            const int32_t side = k / (2 * half);
            const int32_t t = (int32_t)(k % (2 * half)) - half;
            int32_t dx, dz;
            switch (side) {
            case 0:  dx = t;     dz = -half; break; // north edge
            case 1:  dx = half;  dz = t;     break; // east edge
            case 2:  dx = -t;    dz = half;  break; // south edge
            default: dx = -half; dz = -t;    break; // west edge
            }
            const int32_t xq = (hx + dx * STEP) >> 2;
            const int32_t zq = (hz + dz * STEP) >> 2;
            const double cont = sampleClimatePara(&c.contB, nullptr, xq, zq);
            if (cont <= -1.05) vic_mush = 1.0;
            if (cont <= CONT_SHORE) {
                if (std::isnan(nearest)) nearest = (double)r;
                if (vic_warmoc < 0.5 &&
                    sampleClimatePara(&c.tempB, nullptr, xq, zq) >= 0.55)
                    vic_warmoc = 1.0;
            }
        }
    }
    return nearest;
}

// ---------------------------------------------------------------------------
// Thumbnails (--thumbs): a fixed-scale, fixed-size BMP per measured
// region, so a folder sorted by filename IS the ranking, and every image
// is at the same map scale.
//   * 64 blocks per pixel, 256x256 pixels = 16,384 blocks across, centered
//     on the headline: no per-region rescaling, so relative region sizes
//     are directly comparable between images.
//   * Inside the region: the heightfield re-sampled at 64-block pitch
//     (valley floors warm, peaks near-white), ocean in blue, dark forest
//     in dark green.
//   * Outside the region: biome-class context colors at 55% brightness,
//     so the surroundings (inland seas, coasts, mushroom fields in purple,
//     badlands in orange) are visible at a glance.
//   * A magenta crosshair marks the headline coordinate.
//   * The filename sort key is K = round(score * 20 + 1000), zero-padded:
//     a higher score sorts later in the folder. K 1000 means score 0; a
//     score of 8.30 maps to K 1166.
// Cost: about 65k getBiomeAt calls plus a few thousand mapApproxHeight per
// thumbnail (roughly 0.3-0.6 s of one CPU thread): trivial for offline
// triage, a few percent of the verifier pool when enabled on a live
// search.
// ---------------------------------------------------------------------------
static bool g_thumbs_on = false;
static std::string g_thumbs_dir = "thumbs";
static double g_thumb_min_score = -1e300; // --thumb-min-score

void probe_enable_thumbs(const char *dir, double min_score) {
    g_thumbs_on = true;
    g_thumb_min_score = min_score;
    if (dir && dir[0]) g_thumbs_dir = dir;
    std::error_code ec;
    std::filesystem::create_directories(g_thumbs_dir, ec);
}

// Context-ring colors (already dimmed): biome class per pixel outside the blob.
static void ctx_color(int id, uint8_t &r, uint8_t &g, uint8_t &b) {
    auto set = [&](int R, int G, int B) {
        r = (uint8_t)(R * 55 / 100); g = (uint8_t)(G * 55 / 100); b = (uint8_t)(B * 55 / 100);
    };
    switch (id) {
    case ocean: case deep_ocean: case frozen_ocean: case deep_frozen_ocean:
    case cold_ocean: case deep_cold_ocean: case lukewarm_ocean:
    case deep_lukewarm_ocean: case warm_ocean: case deep_warm_ocean:
        set(45, 90, 150); break;
    case river: case frozen_river: set(70, 120, 190); break;
    case mushroom_fields: case mushroom_field_shore: set(190, 90, 190); break;
    case dark_forest: set(30, 70, 35); break;
    case forest: case flower_forest: set(55, 110, 50); break;
    case birch_forest: case old_growth_birch_forest: set(85, 130, 65); break;
    case taiga: case snowy_taiga:
    case old_growth_pine_taiga: case old_growth_spruce_taiga:
        set(50, 100, 85); break;
    case plains: case sunflower_plains: set(140, 165, 85); break;
    case meadow: set(150, 170, 95); break;
    case snowy_plains: case ice_spikes: set(215, 220, 225); break;
    case grove: case snowy_slopes: set(205, 210, 215); break;
    case jagged_peaks: case frozen_peaks: case stony_peaks: set(185, 185, 195); break;
    case savanna: case savanna_plateau: set(180, 160, 85); break;
    case windswept_savanna: case shattered_savanna_plateau: set(160, 130, 95); break;
    case desert: set(225, 210, 140); break;
    case badlands: case eroded_badlands: case wooded_badlands: set(200, 115, 75); break;
    case jungle: case bamboo_jungle: case sparse_jungle: set(70, 130, 60); break;
    case swamp: case mangrove_swamp: set(80, 100, 65); break;
    case cherry_grove: set(230, 160, 190); break;
    case beach: case snowy_beach: set(215, 205, 165); break;
    case stony_shore: set(150, 150, 150); break;
    case windswept_hills: case windswept_forest: case windswept_gravelly_hills:
        set(120, 130, 110); break;
    default: set(95, 95, 95); break;
    }
}

static void write_thumb(MeasureCtx &c, const BlobFill &blob,
                        const std::vector<uint8_t> &inblob,
                        double score, uint64_t seed,
                        int32_t ax, int32_t az, int32_t hx, int32_t hz) {
    constexpr int32_t THUMB_PX = 64;  // blocks per pixel
    constexpr int32_t THUMB_N  = 256; // pixels per side

    const int32_t W = blob.maxI - blob.minI + 1;
    if (W <= 0 || blob.maxJ < blob.minJ) return;
    const int32_t ox = hx - (THUMB_N / 2) * THUMB_PX;
    const int32_t oz = hz - (THUMB_N / 2) * THUMB_PX;

    const int32_t stride = (THUMB_N * 3 + 3) & ~3; // BMP rows pad to 4 bytes
    std::vector<uint8_t> px((size_t)stride * THUMB_N, 0);

    for (int32_t b = 0; b < THUMB_N; b++) {
        const int32_t zb = oz + b * THUMB_PX;
        for (int32_t a = 0; a < THUMB_N; a++) {
            const int32_t xb = ox + a * THUMB_PX;
            // Blob membership of the 64-block lattice cell containing this pixel.
            const int32_t li = (int32_t)std::floor((double)(xb - ax) / (double)BLOB_STEP);
            const int32_t lj = (int32_t)std::floor((double)(zb - az) / (double)BLOB_STEP);
            const bool in = li >= blob.minI && li <= blob.maxI &&
                            lj >= blob.minJ && lj <= blob.maxJ &&
                            inblob[(size_t)(lj - blob.minJ) * W + (li - blob.minI)] != 0;
            const int id = getBiomeAt(&c.g, 1, xb, 256, zb);
            uint8_t r, g, bl;
            if (!in) {
                ctx_color(id, r, g, bl); // context ring
            } else {
                const int bin = census_bin(id);
                if (bin == CEN_OCEAN)     { r =  30; g =  70; bl = 160; }
                else if (bin == CEN_DARK) { r =  25; g =  60; bl =  30; }
                else {
                    float hy;
                    if (mapApproxHeight(&hy, nullptr, &c.g, nullptr, xb >> 2, zb >> 2, 1, 1) != 0) {
                        r = 80; g = 80; bl = 80;
                    } else if (hy <= 100.0f) { r = 205; g = 195; bl = 155; } // valley floors pop
                    else if (hy <= 140.0f)   { r = 115; g = 150; bl =  95; }
                    else if (hy <= 180.0f)   { r = 135; g = 135; bl = 110; }
                    else if (hy <= 220.0f)   { r = 170; g = 165; bl = 155; }
                    else                     { r = 245; g = 245; bl = 250; } // high peaks
                }
            }
            uint8_t *p = &px[(size_t)b * stride + (size_t)a * 3];
            p[0] = bl; p[1] = g; p[2] = r; // BMP stores BGR
        }
    }

    // Headline crosshair (magenta).
    {
        const int32_t cx = (hx - ox) / THUMB_PX;
        const int32_t cz = (hz - oz) / THUMB_PX;
        for (int32_t d = -3; d <= 3; d++) {
            if (cx + d >= 0 && cx + d < THUMB_N) {
                uint8_t *p = &px[(size_t)cz * stride + (size_t)(cx + d) * 3];
                p[0] = 255; p[1] = 0; p[2] = 255;
            }
            if (cz + d >= 0 && cz + d < THUMB_N) {
                uint8_t *p = &px[(size_t)(cz + d) * stride + (size_t)cx * 3];
                p[0] = 255; p[1] = 0; p[2] = 255;
            }
        }
    }

    // The filename sort key: K = score * 20 + 1000, so higher scores sort
    // later in the folder. K 1000 means score 0.
    int key = std::isnan(score) ? 0 : (int)std::lrint(score * 20.0 + 1000.0);
    key = key < 0 ? 0 : (key > 9999 ? 9999 : key);
    char path[512];
    std::snprintf(path, sizeof(path), "%s/t_%04d_%" PRIi64 "_%" PRIi32 "_%" PRIi32 ".bmp",
                  g_thumbs_dir.c_str(), key, (int64_t)seed, hx, hz);
    std::FILE *fp = std::fopen(path, "wb");
    if (!fp) return;
    const uint32_t img = (uint32_t)stride * (uint32_t)THUMB_N;
    auto w16 = [&](uint32_t v) { std::fputc((int)(v & 255), fp); std::fputc((int)((v >> 8) & 255), fp); };
    auto w32 = [&](uint32_t v) { w16(v & 0xFFFF); w16(v >> 16); };
    std::fputc('B', fp); std::fputc('M', fp);
    w32(54 + img); w32(0); w32(54);
    w32(40); w32((uint32_t)THUMB_N); w32((uint32_t)-THUMB_N); // top-down
    w16(1); w16(24); w32(0); w32(img);
    w32(2835); w32(2835); w32(0); w32(0);
    std::fwrite(px.data(), 1, img, fp);
    std::fclose(fp);
}

// ---------------------------------------------------------------------------
// The full measurement bundle (exported): the sublattice stats, the 9-bin
// biome census, neighbor-pair crossings, the windowed best-subregion
// stats, and the composite score. This is the expensive part of
// verification, so the verifier runs it once per verified seed after every
// gate has passed; --probe mode always runs it.
// ---------------------------------------------------------------------------
void blob_enrich(MeasureCtx &c, const BlobFill &blob, EnrichStats &o, int phase) {

    o.clear();
    const int32_t px = phase & 1;        // 1 = shift the 128-block grid by 64 in x
    const int32_t pz = (phase >> 1) & 1; // 1 = shift the 128-block grid by 64 in z

    // Measurement base: the world-aligned origin chosen by blob_fill. All
    // stats below key off it, so they're anchor-independent.
    const int32_t bx = blob.origin_x;
    const int32_t bz = blob.origin_z;
    // Absolute lattice indices of the origin cell. Sublattice parities are
    // pinned to the absolute lattice ((oi0 + i) & 1 == px), so a phase
    // measures the same interleaved sublattice no matter which anchor
    // triggered the fill; blob_enrich_best() then keeps the best phase.
    const int32_t oi0 = bx / BLOB_STEP;
    const int32_t oj0 = bz / BLOB_STEP;

    const int64_t ncells = (int64_t)blob.cells.size();
    if (ncells == 0) return;
    o.valid = true;

    // A cell joins the measurement when its absolute lattice parity
    // matches the current phase on both axes (the phase pins which of the
    // four interleaved grids is measured, independent of which anchor
    // triggered the fill). Neighbor pairs use offset +2, keeping a uniform
    // 128-block pitch.
    // Bounding-box-indexed value arrays for pair lookups.
    const int32_t W = blob.maxI - blob.minI + 1;
    const int32_t H = blob.maxJ - blob.minJ + 1;
    thread_local std::vector<uint8_t> inblob;
    thread_local std::vector<double>  hv, wv, ev, mv, ov, dv, tv, rv;
    thread_local std::vector<uint8_t> gv; // 1 = old-growth taiga, 2 = cherry grove
    inblob.assign((size_t)W * H, 0);
    hv.assign((size_t)W * H, NaN);
    wv.assign((size_t)W * H, NaN);
    ev.assign((size_t)W * H, NaN); // full-octave erosion per cell
    mv.assign((size_t)W * H, 0.0); // mountain-census indicator per cell
    ov.assign((size_t)W * H, 0.0); // ocean-census indicator per cell
    dv.assign((size_t)W * H, 0.0); // dark-forest indicator per cell
    tv.assign((size_t)W * H, 0.0); // "mtnish" indicator per cell
    rv.assign((size_t)W * H, 0.0); // non-frozen river indicator per cell
    gv.assign((size_t)W * H, 0);   // gem-forest indicator per cell
    auto gidx = [&](int32_t i, int32_t j) {
        return (size_t)(j - blob.minJ) * W + (i - blob.minI);
    };
    for (const auto &cl : blob.cells)
        inblob[gidx(cl.first, cl.second)] = 1;

    double ef_sum = 0.0, ef_min = 1e9;
    double cf_sum = 0.0, cf_max = -1e9;
    double wa_sum = 0.0;
    double h_sum = 0.0, h_sum2 = 0.0, h_min = 1e9, h_max = -1e9;
    int64_t ridge_n = 0, valley_n = 0, high_n = 0, low_n = 0;
    int64_t bins[CEN_N] = {};
    std::vector<float> ef_vals, h_vals;
    ef_vals.reserve(4096);
    h_vals.reserve(4096);
    int64_t n = 0;

    for (const auto &cl : blob.cells) {
        const int32_t i = cl.first, j = cl.second;
        if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;

        const int32_t x = bx + i * BLOB_STEP;
        const int32_t z = bz + j * BLOB_STEP;
        const int32_t xq = x >> 2, zq = z >> 2;

        const double ef = sampleClimatePara(&c.eroF, nullptr, xq, zq);
        ef_sum += ef;
        if (ef < ef_min) ef_min = ef;
        ef_vals.push_back((float)ef);
        ev[gidx(i, j)] = ef;       // per-cell copy for the windowed pass

        const double cf = sampleClimatePara(&c.contF, nullptr, xq, zq);
        cf_sum += cf;
        if (cf > cf_max) cf_max = cf;

        const double w = sampleClimatePara(&c.weird, nullptr, xq, zq);
        const double pv = pv_of(w);
        wa_sum += std::fabs(w);
        if (pv > PV_RIDGE)  ridge_n++;
        if (pv < PV_VALLEY) valley_n++;
        wv[gidx(i, j)] = w;

        float hy;
        if (mapApproxHeight(&hy, nullptr, &c.g, nullptr, xq, zq, 1, 1) == 0) {
            hv[gidx(i, j)] = (double)hy;
            h_sum += hy;
            h_sum2 += (double)hy * hy;
            if (hy < h_min) h_min = hy;
            if (hy > h_max) h_max = hy;
            if (hy >= H_HIGH) high_n++;
            if (hy <= H_LOW)  low_n++;
            h_vals.push_back(hy);
        }

        const int bid = getBiomeAt(&c.g, 1, x, 256, z);
        const int b = census_bin(bid);
        if (b >= 0) bins[b]++;
        rv[gidx(i, j)] = (bid == river) ? 1.0 : 0.0;
        gv[gidx(i, j)] = (bid == old_growth_pine_taiga || bid == old_growth_spruce_taiga) ? 1
                       : (bid == cherry_grove) ? 2 : 0;
        mv[gidx(i, j)] = (b == CEN_MOUNTAIN) ? 1.0 : 0.0;
        // "mtnish": the range backbone (peaks biomes) plus the biomes that
        // typically ring them, so one continuous range reads as one component.
        tv[gidx(i, j)] = (b == CEN_MOUNTAIN || b == CEN_OG_TAIGA ||
                          b == CEN_MEADOW    || b == CEN_CHERRY) ? 1.0 : 0.0;
        ov[gidx(i, j)] = (b == CEN_OCEAN) ? 1.0 : 0.0;
        dv[gidx(i, j)] = (b == CEN_DARK) ? 1.0 : 0.0;
        // (rv/gv were set above from the raw biome id: the census bins merge
        // river with frozen river and both old-growth taigas, which the
        // setting detectors deliberately distinguish.)

        n++;
    }

    if (n > 0) {
        o.eroF_mean  = ef_sum / (double)n;
        o.eroF_min   = ef_min;
        o.eroF_p10   = percentile(ef_vals, 0.10f);
        o.contF_mean = cf_sum / (double)n;
        o.contF_max  = cf_max;
        o.w_abs      = wa_sum / (double)n;
        o.ridge      = (double)ridge_n  / (double)n;
        o.valley     = (double)valley_n / (double)n;
        if (!h_vals.empty()) {
            const double m = (double)h_vals.size();
            o.h_min  = h_min;
            o.h_max  = h_max;
            o.h_mean = h_sum / m;
            o.h_p90  = percentile(h_vals, 0.90f);
            const double var = h_sum2 / m - o.h_mean * o.h_mean;
            o.h_std  = var > 0.0 ? std::sqrt(var) : 0.0;
            o.high   = (double)high_n / (double)n;
            o.low    = (double)low_n  / (double)n;
        }
        for (int b = 0; b < CEN_N; b++)
            o.bio[b] = (double)bins[b] / (double)n;
    }

    // --- Neighbor pairs: ridge/valley crossings + height relief --------
    int64_t pairs = 0, rcross_n = 0, vcross_n = 0, relief_n = 0;
    double dh_sum = 0.0;    // mean |dh| over measured pairs (ruggedness)
    int64_t dh_pairs = 0;   // (pairs with heights measured on both ends)
    for (const auto &cl : blob.cells) {
        const int32_t i = cl.first, j = cl.second;
        if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
        const double wa = wv[gidx(i, j)];
        const double ha = hv[gidx(i, j)];
        if (std::isnan(wa)) continue;

        const int32_t nbr[2][2] = {{i + SUB, j}, {i, j + SUB}};
        for (const auto &nn : nbr) {
            if (nn[0] < blob.minI || nn[0] > blob.maxI ||
                nn[1] < blob.minJ || nn[1] > blob.maxJ) continue;
            if (!inblob[gidx(nn[0], nn[1])]) continue;
            const double wb = wv[gidx(nn[0], nn[1])];
            if (std::isnan(wb)) continue;
            pairs++;

            // Ridge crossing: |w| straddles 2/3.
            const double ra = std::fabs(wa) - 2.0 / 3.0;
            const double rb = std::fabs(wb) - 2.0 / 3.0;
            if ((ra < 0) != (rb < 0)) rcross_n++;
            // Valley crossing: w straddles 0 (river axis).
            if ((wa < 0) != (wb < 0)) vcross_n++;

            const double hb = hv[gidx(nn[0], nn[1])];
            if (!std::isnan(ha) && !std::isnan(hb)) {
                const double dh = std::fabs(ha - hb);
                dh_sum += dh;
                dh_pairs++;
                if (dh >= H_RELIEF) relief_n++;
            }
        }
    }
    if (pairs > 0) {
        o.rcross = (double)rcross_n / (double)pairs;
        o.vcross = (double)vcross_n / (double)pairs;
        o.relief = (double)relief_n / (double)pairs;
        if (dh_pairs > 0) o.dh_mean = dh_sum / (double)dh_pairs;
    }

    // --- Peak texture: local height maxima and high-ground connectivity ---
    // These separate "one plateau massif" from "densely packed individual
    // peaks with dissecting valleys":
    //   peak_density / peak_prom: strict local maxima on the sublattice (a
    //     peak must sit above all four neighbors), with a prominence floor
    //     (PEAK_MIN_PROM) so ripple noise never registers as a peak.
    //   high_comps / high_largest: the component count and the largest
    //     component's share of the y >= 200 mask. A massif is one
    //     component (share near 1.0); packed peaks are many small ones.
    if (n > 0 && !h_vals.empty()) {
        constexpr double PEAK_MIN_PROM = 8.0; // blocks; tunable
        int64_t peaks = 0;
        double prom_sum = 0.0;
        for (const auto &cl : blob.cells) {
            const int32_t i = cl.first, j = cl.second;
            if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
            const double h0 = hv[gidx(i, j)];
            if (std::isnan(h0)) continue;
            double nsum = 0.0;
            int nn = 0;
            bool is_max = true;
            const int32_t nbr[4][2] = {{i-SUB,j},{i+SUB,j},{i,j-SUB},{i,j+SUB}};
            for (const auto &nnp : nbr) {
                if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                    nnp[1] < blob.minJ || nnp[1] > blob.maxJ) continue;
                if (!inblob[gidx(nnp[0], nnp[1])]) continue;
                const double hn = hv[gidx(nnp[0], nnp[1])];
                if (std::isnan(hn)) continue;
                if (hn >= h0) is_max = false; // ties/plateaus don't count
                nsum += hn;
                nn++;
            }
            if (!is_max || nn < 3) continue; // edge cells need 3+ neighbors
            const double prom = h0 - nsum / nn;
            if (prom < PEAK_MIN_PROM) continue;
            peaks++;
            prom_sum += prom;
        }
        const double area_m = (double)blob.area / 1e6;
        if (area_m > 0) {
            o.peak_density = (double)peaks / area_m;
            o.peak_prom    = peaks > 0 ? prom_sum / (double)peaks : 0.0; // 0 peaks is informative
        }

        // High-ground components: BFS over the sublattice cells at >= 200.
        {
            const int32_t W2 = blob.maxI - blob.minI + 1;
            const int32_t H2 = blob.maxJ - blob.minJ + 1;
            thread_local std::vector<uint8_t> hseen;
            hseen.assign((size_t)W2 * H2, 0);
            thread_local std::vector<int32_t> hqueue;
            hqueue.clear();
            int64_t comps = 0, largest = 0;
            for (const auto &cl : blob.cells) {
                const int32_t i = cl.first, j = cl.second;
                if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
                const size_t g0 = gidx(i, j);
                if (hseen[g0] || std::isnan(hv[g0]) || hv[g0] < H_HIGH) continue;
                int64_t sz = 0;
                hseen[g0] = 1;
                hqueue.push_back((int32_t)g0);
                while (!hqueue.empty()) {
                    const int32_t g = hqueue.back();
                    hqueue.pop_back();
                    sz++;
                    const int32_t ci = (int32_t)(g % W2) + blob.minI;
                    const int32_t cj = (int32_t)(g / W2) + blob.minJ;
                    const int32_t nbr[4][2] = {{ci-SUB,cj},{ci+SUB,cj},{ci,cj-SUB},{ci,cj+SUB}};
                    for (const auto &nnp : nbr) {
                        if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                            nnp[1] < blob.minJ || nnp[1] > blob.maxJ) continue;
                        const size_t g2 = gidx(nnp[0], nnp[1]);
                        if (hseen[g2] || !inblob[g2]) continue;
                        const double hn = hv[g2];
                        if (std::isnan(hn) || hn < H_HIGH) continue;
                        hseen[g2] = 1;
                        hqueue.push_back((int32_t)g2);
                    }
                }
                comps++;
                if (sz > largest) largest = sz;
            }
            if (area_m > 0) o.high_comps = (double)comps / area_m;
            if (high_n > 0) o.high_largest = (double)largest / (double)high_n;
        }

        // Mountain-range connectivity: 4-connected components of the mtnish
        // mask (tv). "One big continuous range" = few components with a large
        // largest one; scattered patches = many small ones. mtn_comp_m is the
        // largest component's area in blocks^2 (each sublattice cell stands
        // for 128*128 = 16384 blocks^2).
        {
            const int32_t W2 = blob.maxI - blob.minI + 1;
            const int32_t H2 = blob.maxJ - blob.minJ + 1;
            thread_local std::vector<uint8_t> mseen;
            mseen.assign((size_t)W2 * H2, 0);
            thread_local std::vector<int32_t> mqueue;
            mqueue.clear();
            int64_t comps = 0, largest = 0;
            for (const auto &cl : blob.cells) {
                const int32_t i = cl.first, j = cl.second;
                if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
                const size_t g0 = gidx(i, j);
                if (mseen[g0] || tv[g0] < 0.5) continue;
                int64_t sz = 0;
                mseen[g0] = 1;
                mqueue.push_back((int32_t)g0);
                while (!mqueue.empty()) {
                    const int32_t g = mqueue.back();
                    mqueue.pop_back();
                    sz++;
                    const int32_t ci = (int32_t)(g % W2) + blob.minI;
                    const int32_t cj = (int32_t)(g / W2) + blob.minJ;
                    const int32_t nbr[4][2] = {{ci-SUB,cj},{ci+SUB,cj},{ci,cj-SUB},{ci,cj+SUB}};
                    for (const auto &nnp : nbr) {
                        if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                            nnp[1] < blob.minJ || nnp[1] > blob.maxJ) continue;
                        const size_t g2 = gidx(nnp[0], nnp[1]);
                        if (mseen[g2] || !inblob[g2]) continue;
                        if (tv[g2] < 0.5) continue;
                        mseen[g2] = 1;
                        mqueue.push_back((int32_t)g2);
                    }
                }
                comps++;
                if (sz > largest) largest = sz;
            }
            o.mtn_comps  = (double)comps;
            o.mtn_comp_m = (double)largest * (double)(BLOB_STEP * SUB) * (double)(BLOB_STEP * SUB);
        }

        // Valley-network connectivity: components of the inland low mask
        // (h <= 100, non-oceanic cells only). This is the "interconnected
        // dissecting valleys" axis -- one large component means the valleys
        // form a single linked network threading the peaks.
        {
            const int32_t W2 = blob.maxI - blob.minI + 1;
            thread_local std::vector<uint8_t> lseen;
            lseen.assign((size_t)W2 * (blob.maxJ - blob.minJ + 1), 0);
            thread_local std::vector<int32_t> lqueue;
            lqueue.clear();
            int64_t comps = 0, largest = 0;
            for (const auto &cl : blob.cells) {
                const int32_t i = cl.first, j = cl.second;
                if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
                const size_t g0 = gidx(i, j);
                if (lseen[g0]) continue;
                const double h0 = hv[g0];
                if (std::isnan(h0) || h0 > 100.0 || ov[g0] > 0.5) continue;
                int64_t sz = 0;
                lseen[g0] = 1;
                lqueue.push_back((int32_t)g0);
                while (!lqueue.empty()) {
                    const int32_t g = lqueue.back();
                    lqueue.pop_back();
                    sz++;
                    const int32_t ci = (int32_t)(g % W2) + blob.minI;
                    const int32_t cj = (int32_t)(g / W2) + blob.minJ;
                    const int32_t nbr[4][2] = {{ci-SUB,cj},{ci+SUB,cj},{ci,cj-SUB},{ci,cj+SUB}};
                    for (const auto &nnp : nbr) {
                        if (nnp[0] < blob.minI || nnp[0] > blob.maxI ||
                            nnp[1] < blob.minJ || nnp[1] > blob.maxJ) continue;
                        const size_t g2 = gidx(nnp[0], nnp[1]);
                        if (lseen[g2] || !inblob[g2]) continue;
                        const double hn = hv[g2];
                        if (std::isnan(hn) || hn > 100.0 || ov[g2] > 0.5) continue;
                        lseen[g2] = 1;
                        lqueue.push_back((int32_t)g2);
                    }
                }
                comps++;
                if (sz > largest) largest = sz;
            }
            o.low_comps  = (double)comps;
            o.low_comp_m = (double)largest * (double)(BLOB_STEP * SUB) * (double)(BLOB_STEP * SUB);
        }

        // Gorge floors (the "deep valleys dissecting packed peaks" signature):
        // fraction of low cells (h <= 105) that have a tall cell (h >= 190)
        // within +-2 sublattice steps (256 blocks). Area-based valley stats
        // structurally can't see narrow gorges; this adjacency stat can.
        {
            int64_t low_cells = 0, walled = 0;
            for (const auto &cl : blob.cells) {
                const int32_t i = cl.first, j = cl.second;
                if (((oi0 + i) & 1) != px || ((oj0 + j) & 1) != pz) continue;
                const double h0 = hv[gidx(i, j)];
                if (std::isnan(h0) || h0 > 105.0) continue;
                low_cells++;
                bool near_tall = false;
                for (int32_t dj = -4; dj <= 4 && !near_tall; dj += 2) {
                    for (int32_t di = -4; di <= 4 && !near_tall; di += 2) {
                        if (di == 0 && dj == 0) continue;
                        if (i + di < blob.minI || i + di > blob.maxI ||
                            j + dj < blob.minJ || j + dj > blob.maxJ) continue;
                        const double hn = hv[gidx(i + di, j + dj)];
                        if (!std::isnan(hn) && hn >= 190.0) near_tall = true;
                    }
                }
                if (near_tall) walled++;
            }
            o.gorge_frac = low_cells > 0 ? (double)walled / (double)low_cells : 0.0;
        }
    }

    // --- Windowed best-subregion stats (prefix sums over the sublattice) --
    if (n > 0) windowed_blob_stats(blob, bx, bz, ev, hv, mv, ov, dv, o, px, pz);

    // --- Headline-targeted stats ------------------------------------------
    // Anchored at the headline coordinate (the coherent best-pattern window
    // center, falling back to the anchor), because that is the point the
    // output row actually teleports you to.
    {
        int32_t hx = bx, hz = bz;
        if (!std::isnan(o.bw1664_sc))     { hx = o.bw1664_x; hz = o.bw1664_z; }
        else if (!std::isnan(o.bw896_sc)) { hx = o.bw896_x;  hz = o.bw896_z; }

        // Dark forest around the headline (darkCoreFrac). Unlike the
        // blob-mask stats, this ring also sees dark forest sitting just
        // outside the region, which the masked census cannot.
        o.dark_core_frac = dark_ring_frac(c, hx, hz);

        // Open ocean beyond the blob edge (ocean inside the blob is covered
        // by ocean_dt_hl in windowed_blob_stats); the ring scan also flags
        // nearby mushroom-island and warm-ocean climates.
        o.ocean_spiral = ocean_spiral_scan(c, hx, hz, o.vic_mush, o.vic_warmoc);

        // Setting (scenery) features: isolated river lakes, coastal cliffs,
        // large gem forests, the waterscape ray scan, and the oddball-biome
        // vicinity scan. Combined by compute_score into the setting bonus.
        setting_features(c, blob, inblob, hv, ov, rv, gv, o, hx, hz, bx, bz);
    }

    o.setting = setting_score(o);
    o.score   = compute_score(blob.area, blob.core_cells, o);
}

// Thumbnails are emitted by blob_enrich_best() for the winning phase only, so
// a --thumbs run doesn't write four near-duplicate images per blob.
static void thumb_from_stats(MeasureCtx &c, const BlobFill &blob,
                             const EnrichStats &o, uint64_t seed) {
    int32_t hx = blob.origin_x, hz = blob.origin_z;
    if (!std::isnan(o.bw1664_sc))     { hx = o.bw1664_x; hz = o.bw1664_z; }
    else if (!std::isnan(o.bw896_sc)) { hx = o.bw896_x;  hz = o.bw896_z; }
    const int32_t W = blob.maxI - blob.minI + 1;
    const int32_t H = blob.maxJ - blob.minJ + 1;
    if (W <= 0 || H <= 0) return;
    thread_local std::vector<uint8_t> inblob;
    inblob.assign((size_t)W * H, 0);
    for (const auto &cl : blob.cells)
        inblob[(size_t)(cl.second - blob.minJ) * W + (cl.first - blob.minI)] = 1;
    write_thumb(c, blob, inblob, o.score, seed, blob.origin_x, blob.origin_z, hx, hz);
}

void blob_enrich_best(MeasureCtx &c, const BlobFill &blob, EnrichStats &o, int phases) {
    EnrichStats best;
    double best_score = -std::numeric_limits<double>::infinity();
    const int n = phases >= 4 ? 4 : 1;
    for (int ph = 0; ph < n; ph++) {
        EnrichStats cur;
        blob_enrich(c, blob, cur, ph);
        const double s = (cur.valid && !std::isnan(cur.score))
                       ? cur.score : -std::numeric_limits<double>::infinity();
        if (s > best_score) {
            best_score = s;
            best = cur;
        }
    }
    o = best;
    if (g_thumbs_on && o.valid && !std::isnan(o.score) && o.score >= g_thumb_min_score)
        thumb_from_stats(c, blob, o, (uint64_t)c.g.seed);
}

namespace {

// ---------------------------------------------------------------------------
// Probe-only: blob-field means over all cells + coastal-erosion stat.
// (Cheap extras the prober prints but the verifier's score doesn't need.)
// ---------------------------------------------------------------------------
void blob_field_stats(MeasureCtx &c, const VerifyConfig &, const BlobFill &blob,
                      int32_t bx, int32_t bz,
                      double &eroB_mean, double &eroB_min, double &tempB_mean,
                      double &coast_ero, int64_t &coast_n) {
    const int64_t ncells = (int64_t)blob.cells.size();
    if (ncells == 0) return;

    {
        double esum = 0.0, emin = 1e9, tsum = 0.0;
        for (const auto &cl : blob.cells) {
            const int32_t xq = (bx + cl.first * BLOB_STEP) >> 2;
            const int32_t zq = (bz + cl.second * BLOB_STEP) >> 2;
            const double e = sampleClimatePara(&c.eroB, nullptr, xq, zq);
            esum += e;
            if (e < emin) emin = e;
            tsum += sampleClimatePara(&c.tempB, nullptr, xq, zq);
        }
        eroB_mean  = esum / (double)ncells;
        eroB_min   = emin;
        tempB_mean = tsum / (double)ncells;
    }

    // Coast: mean 0B erosion of blob cells with an ocean-ish neighbor.
    {
        double sum = 0.0;
        int64_t n = 0;
        for (const auto &cl : blob.cells) {
            const int32_t i = cl.first, j = cl.second;
            bool coastal = false;
            const int32_t nb[4][2] = {{i-1,j},{i+1,j},{i,j-1},{i,j+1}};
            for (const auto &nn : nb) {
                if (nn[0] < -BLOB_HALF || nn[0] > BLOB_HALF ||
                    nn[1] < -BLOB_HALF || nn[1] > BLOB_HALF) continue;
                const int32_t xq = (bx + nn[0] * BLOB_STEP) >> 2;
                const int32_t zq = (bz + nn[1] * BLOB_STEP) >> 2;
                if (sampleClimatePara(&c.contB, nullptr, xq, zq) <= CONT_SHORE) {
                    coastal = true;
                    break;
                }
            }
            if (coastal) {
                const int32_t xq = (bx + i * BLOB_STEP) >> 2;
                const int32_t zq = (bz + j * BLOB_STEP) >> 2;
                sum += sampleClimatePara(&c.eroB, nullptr, xq, zq);
                n++;
            }
        }
        coast_n = n;
        if (n > 0) coast_ero = sum / (double)n;
    }
}

// ---------------------------------------------------------------------------
// Coast scan (coarse lattice, whole window -- independent of the blob)
// ---------------------------------------------------------------------------
struct CoastOut {
    int32_t shore = -1;   // blocks to nearest cont(1B) <= -0.19 cell
    int32_t deep  = -1;   // blocks to nearest cont(1B) <= -0.45 cell
};

CoastOut coast_scan(MeasureCtx &c, int32_t bx, int32_t bz) {
    CoastOut o;
    double d_shore = 1e18, d_deep = 1e18;
    for (int32_t j = -COAST_HALF; j <= COAST_HALF; j++) {
        const int32_t zq = (bz + j * COAST_STEP) >> 2;
        for (int32_t i = -COAST_HALF; i <= COAST_HALF; i++) {
            const int32_t xq = (bx + i * COAST_STEP) >> 2;
            const double cont = sampleClimatePara(&c.contB, nullptr, xq, zq);
            if (cont <= CONT_SHORE) {
                const double d = (double)COAST_STEP * std::sqrt((double)i * i + (double)j * j);
                if (d < d_shore) d_shore = d;
                if (cont <= CONT_DEEP && d < d_deep) d_deep = d;
            }
        }
    }
    if (d_shore < 1e17) o.shore = (int32_t)std::lrint(d_shore);
    if (d_deep  < 1e17) o.deep  = (int32_t)std::lrint(d_deep);
    return o;
}

// ---------------------------------------------------------------------------
// Point-in-polygon (ray casting; x east, z south -- orientation irrelevant)
// ---------------------------------------------------------------------------
bool point_in_poly(const std::vector<std::pair<double, double>> &p, double x, double z) {
    bool inside = false;
    const size_t n = p.size();
    for (size_t a = 0, b = n - 1; a < n; b = a++) {
        const double x1 = p[a].first, z1 = p[a].second;
        const double x2 = p[b].first, z2 = p[b].second;
        if ((z1 > z) != (z2 > z)) {
            const double xi = x1 + (z - z1) * (x2 - x1) / (z2 - z1);
            if (x < xi) inside = !inside;
        }
    }
    return inside;
}

// ---------------------------------------------------------------------------
// Polygon-side stats (bounded grid inside the polygon)
// ---------------------------------------------------------------------------
struct PolyOut {
    int64_t n = 0;
    double eroF_mean = NaN, eroF_min = NaN;
    double ridge = NaN, valley = NaN, rcross = NaN, vcross = NaN;
    double hmin = NaN, hmax = NaN, hmean = NaN;
    double relief = NaN, high = NaN, low = NaN;
    double bio[CEN_N];
    PolyOut() { for (int b = 0; b < CEN_N; b++) bio[b] = NaN; }
};

PolyOut poly_stats(MeasureCtx &c, const ProbeJob &job) {
    PolyOut o;
    if (job.poly.size() < 3) return o;

    double minx = 1e18, maxx = -1e18, minz = 1e18, maxz = -1e18;
    for (const auto &v : job.poly) {
        minx = std::min(minx, v.first);  maxx = std::max(maxx, v.first);
        minz = std::min(minz, v.second); maxz = std::max(maxz, v.second);
    }

    // Pick a grid step that keeps the sample count bounded (~<= 4096 inside
    // the bounding box).
    double step = 96.0;
    while (((maxx - minx) / step + 1.0) * ((maxz - minz) / step + 1.0) > 4096.0)
        step *= 2.0;

    const int32_t nx = (int32_t)((maxx - minx) / step) + 1;
    const int32_t nz = (int32_t)((maxz - minz) / step) + 1;

    thread_local std::vector<double> hv, wv;
    thread_local std::vector<uint8_t> inside;
    hv.assign((size_t)nx * nz, NaN);
    wv.assign((size_t)nx * nz, NaN);
    inside.assign((size_t)nx * nz, 0);

    double ef_sum = 0.0, ef_min = 1e9;
    double h_sum = 0.0, h_min = 1e9, h_max = -1e9;
    int64_t ridge_n = 0, valley_n = 0, high_n = 0, low_n = 0;
    int64_t bins[CEN_N] = {};
    int64_t n = 0;

    for (int32_t j = 0; j < nz; j++) {
        const double z = minz + j * step;
        for (int32_t i = 0; i < nx; i++) {
            const double x = minx + i * step;
            if (!point_in_poly(job.poly, x, z)) continue;

            const int32_t xb = (int32_t)std::llrint(x);
            const int32_t zb = (int32_t)std::llrint(z);
            const int32_t xq = xb >> 2, zq = zb >> 2;
            const size_t g = (size_t)j * nx + i;
            inside[g] = 1;

            const double ef = sampleClimatePara(&c.eroF, nullptr, xq, zq);
            ef_sum += ef;
            if (ef < ef_min) ef_min = ef;

            const double w = sampleClimatePara(&c.weird, nullptr, xq, zq);
            const double pv = pv_of(w);
            if (pv > PV_RIDGE)  ridge_n++;
            if (pv < PV_VALLEY) valley_n++;
            wv[g] = w;

            float hy;
            if (mapApproxHeight(&hy, nullptr, &c.g, nullptr, xq, zq, 1, 1) == 0) {
                hv[g] = (double)hy;
                h_sum += hy;
                if (hy < h_min) h_min = hy;
                if (hy > h_max) h_max = hy;
                if (hy >= H_HIGH) high_n++;
                if (hy <= H_LOW)  low_n++;
            }

            const int b = census_bin(getBiomeAt(&c.g, 1, xb, 256, zb));
            if (b >= 0) bins[b]++;

            n++;
        }
    }

    o.n = n;
    if (n == 0) return o;

    o.eroF_mean = ef_sum / (double)n;
    o.eroF_min  = ef_min;
    o.ridge     = (double)ridge_n  / (double)n;
    o.valley    = (double)valley_n / (double)n;
    o.hmin      = h_min;
    o.hmax      = h_max;
    o.hmean     = h_sum / (double)n;
    o.high      = (double)high_n / (double)n;
    o.low       = (double)low_n  / (double)n;
    for (int b = 0; b < CEN_N; b++)
        o.bio[b] = (double)bins[b] / (double)n;

    int64_t pairs = 0, rcross_n = 0, vcross_n = 0, relief_n = 0;
    for (int32_t j = 0; j < nz; j++) {
        for (int32_t i = 0; i < nx; i++) {
            const size_t g = (size_t)j * nx + i;
            if (!inside[g]) continue;
            const double wa = wv[g];
            if (std::isnan(wa)) continue;
            const int32_t nbr[2][2] = {{i + 1, j}, {i, j + 1}};
            for (const auto &nn : nbr) {
                if (nn[0] >= nx || nn[1] >= nz) continue;
                const size_t g2 = (size_t)nn[1] * nx + nn[0];
                if (!inside[g2]) continue;
                const double wb = wv[g2];
                if (std::isnan(wb)) continue;
                pairs++;
                const double ra = std::fabs(wa) - 2.0 / 3.0;
                const double rb = std::fabs(wb) - 2.0 / 3.0;
                if ((ra < 0) != (rb < 0)) rcross_n++;
                if ((wa < 0) != (wb < 0)) vcross_n++;
                const double ha2 = hv[g], hb2 = hv[g2];
                if (!std::isnan(ha2) && !std::isnan(hb2) &&
                    std::fabs(ha2 - hb2) >= H_RELIEF) relief_n++;
            }
        }
    }
    if (pairs > 0) {
        o.rcross = (double)rcross_n / (double)pairs;
        o.vcross = (double)vcross_n / (double)pairs;
        o.relief = (double)relief_n / (double)pairs;
    }
    return o;
}

// ---------------------------------------------------------------------------
// Output record (probe mode)
// ---------------------------------------------------------------------------
struct ProbeOut {
    uint64_t seed;
    int32_t x, z;

    // blob geometry
    int64_t area = 0;
    int32_t core_cells = 0;
    int32_t edge = 0;
    int64_t ncells = 0;

    // blob-field stats over ALL blob cells (probe-only extras)
    double eroB_mean = NaN, eroB_min = NaN, tempB_mean = NaN;

    // shared enrichment bundle (sublattice stats, census, windowed, score)
    EnrichStats e;

    // coast (probe only)
    int32_t coast_shore = -1;   // blocks to nearest cont(1B) <= -0.19 cell
    int32_t coast_deep  = -1;   // blocks to nearest cont(1B) <= -0.45 cell
    double  coast_ero = NaN;    // mean 0B erosion of coastal blob cells
    int64_t coast_n = 0;

    // polygon (probe only; poly_n == 0 means "no polygon given")
    PolyOut p;
};

// ---------------------------------------------------------------------------
// One candidate
// ---------------------------------------------------------------------------
ProbeOut probe_one(MeasureCtx &c, const VerifyConfig &cfg, const ProbeJob &job) {
    measure_ctx_seed(c, job.seed);

    ProbeOut o;
    o.seed = job.seed;
    o.x = job.x;
    o.z = job.z;

    BlobFill blob = blob_fill(c, cfg, job.x, job.z);
    o.area = blob.area;
    o.core_cells = blob.core_cells;
    o.edge = blob.edge;
    o.ncells = (int64_t)blob.cells.size();

    blob_field_stats(c, cfg, blob, blob.origin_x, blob.origin_z,
                     o.eroB_mean, o.eroB_min, o.tempB_mean, o.coast_ero, o.coast_n);
    blob_enrich_best(c, blob, o.e, cfg.enrich_phases);

    const CoastOut coast = coast_scan(c, job.x, job.z);
    o.coast_shore = coast.shore;
    o.coast_deep  = coast.deep;

    o.p = poly_stats(c, job);
    return o;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------
void print_probe_header(std::FILE *fp) {
    std::fprintf(fp,
        "# [-999 = not measured] Columns:\n"
        "# seed x z\n"
        "#   blob: area coreCells coreBlocks edge nCells\n"
        "#   blob field (all cells): ero0Bmean ero0Bmin temp0Bmean\n"
        "#   blob sublattice (128b): eroFmean eroFp10 eroFmin contFmean contFmax\n"
        "#     weird: meanAbs ridgeFrac valleyFrac ridgeCrossFrac valleyCrossFrac\n"
        "#     height: hMin hMax hMean hP90 reliefFrac highFrac lowFrac\n"
        "#     census: mtn mtg tg mdw chy drk pln riv ocn\n"
        "#   coast: shoreDistBlocks deepDistBlocks coastEroMean coastCells\n"
        "#   polygon: n eroFmean eroFmin ridgeFrac valleyFrac ridgeCross valleyCross\n"
        "#     hMin hMax hMean reliefFrac highFrac lowFrac mtn..ocn (9)\n"
        "#   windowed best-subregion: w896eroMin w896hMax w896mtnMax w896eroX w896eroZ\n"
        "#     w1664eroMin w1664hMax w1664mtnMax w1664eroX w1664eroZ (896-/1664-block\n"
        "#     square windows; positions = best-erosion window centers)\n"
        "#     w896hStdMax w896hMeanMin w1664hStdMax w1664hMeanMin (local contrast/depth)\n"
        "#     bw{896,1664,3200}{Sc Ero hMean hStd Dh Mtn Hi Lo Ocn X Z}: coherent\n"
        "#     best-pattern windows (all stats from ONE argmax window)\n"
        "#   peak texture: hStd dhMean peakDensity peakProm highComps highLargest\n"
        "#   headline-targeted: maxDark896Near oceanDtHl oceanSpiral (blocks)\n"
        "#     superlevel: qualN qualArea scQ90 | range: mtnCompM mtnComps\n"
        "#     gorgeFrac | valley net: lowCompM lowComps | vicMush vicWarmOc\n"
        "#     (areas in blocks^2; vic* = 0/1 flags within 4800 blocks)\n"
        "#   setting: lakeN cliffFrac vicSeaMin vicOpenOc vicOpenWarm vicFjord\n"
        "#     vicIsthmus vicBadl vicBadlOth vicMushB vicFlower vicCherry vicMtg\n"
        "#     mtgCompM chyCompM setting (the unified, core-gated scenery bonus)\n"
        "#   score: composite rank, v0 (constants in probe.cpp compute_score)\n"
        "# Input: rows with >= 12 trailing numbers are read as prior output\n"
        "# (extra columns ignored); other rows may append a 3-5 vertex polygon\n");
}

void fnum(std::FILE *fp, double v) {
    if (std::isnan(v)) std::fprintf(fp, " -999");
    else               std::fprintf(fp, " %.4f", v);
}

void print_probe(std::FILE *fp, const ProbeOut &o) {
    std::fprintf(fp, "%" PRIi64 " %" PRIi32 " %" PRIi32,
                 (int64_t)o.seed, o.x, o.z);
    std::fprintf(fp, " %" PRIi64 " %" PRIi32 " %" PRIi32 " %" PRIi32 " %" PRIi64,
                 (int64_t)o.area, o.core_cells, o.core_cells * 64, o.edge,
                 (int64_t)o.ncells);
    fnum(fp, o.eroB_mean); fnum(fp, o.eroB_min); fnum(fp, o.tempB_mean);
    fnum(fp, o.e.eroF_mean); fnum(fp, o.e.eroF_p10); fnum(fp, o.e.eroF_min);
    fnum(fp, o.e.contF_mean); fnum(fp, o.e.contF_max);
    fnum(fp, o.e.w_abs); fnum(fp, o.e.ridge); fnum(fp, o.e.valley);
    fnum(fp, o.e.rcross); fnum(fp, o.e.vcross);
    fnum(fp, o.e.h_min); fnum(fp, o.e.h_max); fnum(fp, o.e.h_mean); fnum(fp, o.e.h_p90);
    fnum(fp, o.e.relief); fnum(fp, o.e.high); fnum(fp, o.e.low);
    for (int b = 0; b < CEN_N; b++) fnum(fp, o.e.bio[b]);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.coast_shore, o.coast_deep);
    fnum(fp, o.coast_ero);
    std::fprintf(fp, " %" PRIi64, (int64_t)o.coast_n);

    // polygon
    std::fprintf(fp, " %" PRIi64, (int64_t)o.p.n);
    fnum(fp, o.p.eroF_mean); fnum(fp, o.p.eroF_min);
    fnum(fp, o.p.ridge); fnum(fp, o.p.valley);
    fnum(fp, o.p.rcross); fnum(fp, o.p.vcross);
    fnum(fp, o.p.hmin); fnum(fp, o.p.hmax); fnum(fp, o.p.hmean);
    fnum(fp, o.p.relief); fnum(fp, o.p.high); fnum(fp, o.p.low);
    for (int b = 0; b < CEN_N; b++) fnum(fp, o.p.bio[b]);

    // windowed best-subregion + composite score
    fnum(fp, o.e.w896_ero_min);  fnum(fp, o.e.w896_h_max);  fnum(fp, o.e.w896_mtn_max);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.e.w896_ero_x, o.e.w896_ero_z);
    fnum(fp, o.e.w1664_ero_min); fnum(fp, o.e.w1664_h_max); fnum(fp, o.e.w1664_mtn_max);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.e.w1664_ero_x, o.e.w1664_ero_z);
    fnum(fp, o.e.h_std); fnum(fp, o.e.dh_mean);
    fnum(fp, o.e.peak_density); fnum(fp, o.e.peak_prom);
    fnum(fp, o.e.high_comps); fnum(fp, o.e.high_largest);
    fnum(fp, o.e.dark_core_frac);
    fnum(fp, o.e.w896_hstd_max); fnum(fp, o.e.w896_hmean_min);
    fnum(fp, o.e.w1664_hstd_max); fnum(fp, o.e.w1664_hmean_min);
    fnum(fp, o.e.bw896_sc);  fnum(fp, o.e.bw896_ero);  fnum(fp, o.e.bw896_hmean);
    fnum(fp, o.e.bw896_hstd); fnum(fp, o.e.bw896_dh);  fnum(fp, o.e.bw896_mtn);
    fnum(fp, o.e.bw896_hifrac); fnum(fp, o.e.bw896_lofrac); fnum(fp, o.e.bw896_ocn);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.e.bw896_x, o.e.bw896_z);
    fnum(fp, o.e.bw1664_sc);  fnum(fp, o.e.bw1664_ero);  fnum(fp, o.e.bw1664_hmean);
    fnum(fp, o.e.bw1664_hstd); fnum(fp, o.e.bw1664_dh);  fnum(fp, o.e.bw1664_mtn);
    fnum(fp, o.e.bw1664_hifrac); fnum(fp, o.e.bw1664_lofrac); fnum(fp, o.e.bw1664_ocn);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.e.bw1664_x, o.e.bw1664_z);
    fnum(fp, o.e.bw3200_sc);  fnum(fp, o.e.bw3200_ero);  fnum(fp, o.e.bw3200_hmean);
    fnum(fp, o.e.bw3200_hstd); fnum(fp, o.e.bw3200_dh);  fnum(fp, o.e.bw3200_mtn);
    fnum(fp, o.e.bw3200_hifrac); fnum(fp, o.e.bw3200_lofrac); fnum(fp, o.e.bw3200_ocn);
    std::fprintf(fp, " %" PRIi32 " %" PRIi32, o.e.bw3200_x, o.e.bw3200_z);
    fnum(fp, o.e.bw1664_dark);
    fnum(fp, o.e.max_dark_896_near);
    fnum(fp, o.e.ocean_dt_hl);
    fnum(fp, o.e.ocean_spiral);
    fnum(fp, o.e.qual_n);
    fnum(fp, o.e.qual_area);
    fnum(fp, o.e.sc_q90);
    fnum(fp, o.e.mtn_comp_m);
    fnum(fp, o.e.mtn_comps);
    fnum(fp, o.e.gorge_frac);
    fnum(fp, o.e.low_comp_m);
    fnum(fp, o.e.low_comps);
    fnum(fp, o.e.vic_mush);
    fnum(fp, o.e.vic_warmoc);
    fnum(fp, o.e.lake_n);
    fnum(fp, o.e.cliff_frac);
    fnum(fp, o.e.vic_sea_min);
    fnum(fp, o.e.vic_open_oc);
    fnum(fp, o.e.vic_open_warm);
    fnum(fp, o.e.vic_fjord);
    fnum(fp, o.e.vic_isthmus);
    fnum(fp, o.e.vic_badl);
    fnum(fp, o.e.vic_badl_oth);
    fnum(fp, o.e.vic_mush_b);
    fnum(fp, o.e.vic_flower);
    fnum(fp, o.e.vic_cherry);
    fnum(fp, o.e.vic_mtg);
    fnum(fp, o.e.mtg_comp_m);
    fnum(fp, o.e.chy_comp_m);
    fnum(fp, o.e.setting);
    fnum(fp, o.e.score);
    std::fprintf(fp, "\n");
    std::fflush(fp);
}

void print_probe_stdout(const ProbeOut &o) {
    std::printf("PROBE %" PRIi64 " (%" PRIi32 ", %" PRIi32 "): "
                "blob %.1fM coreR %" PRIi32 "b%s | eroFm %.3f | ridge %.2f vall %.2f "
                "rc %.2f vc %.2f | h %.0f..%.0f rel %.2f | coast %d/%d | riv %.3f"
                " | bw1664 sc %.2f dh %.1f hi %.2f lo %.2f ocn %.2f | set %.2f | score %.1f\n",
                (int64_t)o.seed, o.x, o.z,
                (double)o.area / 1e6, o.core_cells * 64, o.edge ? " EDGE" : "",
                o.e.eroF_mean, o.e.ridge, o.e.valley, o.e.rcross, o.e.vcross,
                o.e.h_min, o.e.h_max, o.e.relief,
                o.coast_shore, o.coast_deep,
                std::isnan(o.e.bio[CEN_RIVER]) ? -1.0 : o.e.bio[CEN_RIVER],
                o.e.bw1664_sc, o.e.bw1664_dh, o.e.bw1664_hifrac,
                o.e.bw1664_lofrac, o.e.bw1664_ocn,
                std::isnan(o.e.setting) ? -999.0 : o.e.setting,
                std::isnan(o.e.score) ? -999.0 : o.e.score);
    std::fflush(stdout);
}

} // namespace

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
int run_probe(const std::string &path, const VerifyConfig &vcfg, int threads, std::FILE *out) {
    std::vector<ProbeJob> jobs;
    if (!load_probe_list(path.c_str(), jobs)) {
        std::fprintf(stderr, "Could not open %s\n", path.c_str());
        return 1;
    }
    if (jobs.empty()) {
        std::fprintf(stderr, "No candidates found in %s\n", path.c_str());
        return 1;
    }

    std::printf("Probing %zu candidates from %s (gates OFF, %d threads)\n",
                jobs.size(), path.c_str(), threads);
    print_probe_header(out);

    std::atomic_size_t next{0};
    std::atomic_uint64_t done{0};
    std::mutex out_mutex;

    auto worker = [&]() {
        MeasureCtx *c = new MeasureCtx();
        measure_ctx_setup(*c, vcfg.mc_version);
        while (g_running.load(std::memory_order_relaxed)) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) break;
            ProbeOut o = probe_one(*c, vcfg, jobs[i]);
            {
                std::lock_guard<std::mutex> lock(out_mutex);
                print_probe(out, o);
                print_probe_stdout(o);
            }
            done.fetch_add(1, std::memory_order_relaxed);
        }
        delete c;
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker);

    auto last = std::chrono::steady_clock::now();
    while (done.load(std::memory_order_relaxed) < jobs.size() &&
           g_running.load(std::memory_order_relaxed)) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() >= 5.0) {
            std::printf("probe progress: %" PRIu64 "/%zu\n",
                        done.load(std::memory_order_relaxed), jobs.size());
            last = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    for (auto &t : pool) t.join();

    std::printf("Probe complete: %" PRIu64 "/%zu measured.\n",
                done.load(std::memory_order_relaxed), jobs.size());
    return 0;
}
