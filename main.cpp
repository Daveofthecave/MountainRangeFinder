// main.cpp
// Entry point and orchestrator for the MountainRangeFinder project.
// This program utilizes a hybrid GPU/CPU architecture to rapidly scan the
// 2^64 Minecraft seed space for massive, contiguous mountainous regions.
//
// Run modes:
//   (default)   incremental GPU -> CPU search over the 64-bit seed space
//   --seeds F   restricted search: only the seeds listed in F go through the
//               GPU pipeline (recall testing against known-good seeds)
//   --verify F  skip the GPU entirely; CPU-verify "seed x z" candidate lines
//               from F (threshold calibration; no CUDA device required)
//   --probe F   gates-free measurement (see probe.cpp)
//
// OUTPUT FORMAT (default and --verify), one row per verified candidate:
//   score seed x z blobArea coreRadius maxY eroCovFrac eroMin contMax edge
//   bioMtn bioMtOg bioTg bioMdw bioChy bioDrk bioPln bioRiv bioOcn
//   eroFmean eroFp10 eroFmin contFmean contFmax wAbs ridge valley rcross vcross
//   hMin hMax hMean hP90 relief high low
//   w896eroMin w896hMax w896mtnMax w896eroX w896eroZ
//   w1664eroMin w1664hMax w1664mtnMax anchorX anchorZ
//   (plus the many enrichment/stat columns; the output file's own header
//   comment carries the full, always-current column list)
//
// The score is the composite rank from probe.cpp, computed by the CPU
// verifier itself (blob_enrich over the same flood fill the gates used), so
// searching and ranking happen in one pass. The (x, z) pair is the headline
// coordinate: the center of the region's best-pattern 1,664x1,664-block
// window, the heart of the densest-packed peaks. anchorX/anchorZ is the
// point where the pipeline first detected the region. Sort by column 1
// descending to rank.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <cstdarg>
#ifdef _WIN32
#include <io.h>       // _isatty / _fileno / _write
#else
#include <unistd.h>
#endif

// Async-signal-safe-ish stderr write used by the Ctrl+C handler.
static void write_stderr(const char *msg, size_t len) {
#ifdef _WIN32
    _write(2, msg, (unsigned int)len);
#else
    const ssize_t ignored = write(STDERR_FILENO, msg, len);
    (void)ignored;
#endif
}

#include <cstdlib>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>
#include <queue>
#include <deque>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <optional>
#include <charconv>
#include <algorithm>
#include <random>

// Project-specific headers
#include "common.h"
#include "gpu.h"
#include "cpu.h"
#include "util.h"   // cubiomes str2mc() / mc2str()
#include "biomes.h" // cubiomes MC_* version enum
#include "probe.h"

// Device config pushers, defined in gpu.cu (declared here because this file
// is compiled as pure C++ and gpu.h stays CUDA-free).
void gpu_set_blob_cfg(float ero_max, float cont_min, float temp_min, float temp_max,
                      uint32_t target_cells);
void gpu_set_mc(int mc_version);

// Global atomic flag to control graceful shutdown across all threads.
std::atomic_bool g_running{true};

// Output-time score floor (--min-score). Verified rows scoring below this are
// never written to the output file. NaN scores (--no-blob) are treated as
// -inf, so they only survive the default (no floor). Checked in the output
// path only: verification work is unchanged, and --verify/--probe stay
// unfiltered so calibration corpora keep their full score distribution.
static double g_min_score = -std::numeric_limits<double>::infinity();
static std::atomic_uint32_t g_signal_count{0};

// The Ctrl+C handler: one press starts a graceful shutdown (the worker
// threads finish in-flight work, then the final stats line prints), a
// second press force-exits immediately.
static void signal_handler(int) {
    const uint32_t count = g_signal_count.fetch_add(1, std::memory_order_relaxed) + 1;
    g_running.store(false, std::memory_order_relaxed);

    if (count == 1) {
        // Announce immediately: with 32 CPU verifiers and a GPU batch in
        // flight, a graceful shutdown can take several seconds, and the
        // final stats only print at the very end. write(2) is
        // async-signal-safe; printf would not be.
        static const char msg[] =
            "\nInterrupt caught: shutting down gracefully (this can take a few\n"
            "seconds). Press Ctrl+C again to force-exit immediately -- but note\n"
            "the final stats line only prints on a clean shutdown.\n";
        write_stderr(msg, sizeof(msg) - 1);
        return;
    }

    // If the user presses Ctrl+C twice, force an immediate exit.
    std::_Exit(130);
}

// ---------------------------------------------------------------------------
// Console UI: verbosity levels plus a live status line.
//
// The status line rewrites itself in place with a carriage return ("\r" +
// overwrite): no ANSI escapes, no curses, no dependencies, so it works on
// any terminal (including legacy cmd.exe), and it degrades to plain
// 5-second progress lines when stdout is redirected to a file or pipe.
// ---------------------------------------------------------------------------
std::atomic_int g_ui_level{UI_NORMAL};

static std::mutex  g_ui_mutex;
static std::string g_ui_text;          // current status text
static bool        g_ui_shown = false; // the status line is on screen right now
static bool        g_ui_tty   = false; // stdout is an interactive terminal

void ui_init() {
#ifdef _WIN32
    g_ui_tty = _isatty(_fileno(stdout)) != 0;
#else
    g_ui_tty = isatty(fileno(stdout)) != 0;
#endif
}

// Caller must hold g_ui_mutex.
static void ui_wipe_locked() {
    if (!g_ui_shown) return;
    std::fputc('\r', stdout);
    for (size_t i = 0; i < g_ui_text.size(); i++) std::fputc(' ', stdout);
    std::fputc('\r', stdout);
    g_ui_shown = false;
}

// Caller must hold g_ui_mutex.
static void ui_draw_locked() {
    if (!g_ui_tty || g_ui_text.empty() || g_ui_shown) return;
    std::fputs(g_ui_text.c_str(), stdout);
    std::fflush(stdout);
    g_ui_shown = true;
}

void ui_status(const char *s) {
    std::lock_guard lock(g_ui_mutex);
    g_ui_text = s ? s : "";
    ui_wipe_locked();
    ui_draw_locked();
}

void ui_eventf(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard lock(g_ui_mutex);
    ui_wipe_locked();
    std::fputs(buf, stdout);
    std::fputc('\n', stdout);
    ui_draw_locked();
    std::fflush(stdout);
}

void ui_block_begin() {
    g_ui_mutex.lock();
    ui_wipe_locked();
}

void ui_block_end() {
    ui_draw_locked();
    std::fflush(stdout);
    g_ui_mutex.unlock();
}

// ---------------------------------------------------------------------------
// Small parsing helpers
// ---------------------------------------------------------------------------
static bool parse_i64(const char *s, int64_t &out) {
    const char *end = s + std::strlen(s);
    const auto res = std::from_chars(s, end, out);
    return res.ec == std::errc() && res.ptr == end;
}

// Row tokenization used to locate the seed/x/z inside a row. Output rows
// lead with a decimal score column, so the seed is found as the first
// integer-like token (a score always prints with a decimal point, so it can
// never be mistaken for a seed). Both row layouts parse identically: the
// current score-first rows and the earlier seed-first ones. --seeds,
// --verify, and the dedup preload all build on these helpers.
struct RowTok { const char *p; int len; };

static int row_tokenize(const char *line, RowTok *toks, int maxtoks) {
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

static bool row_tok_is_int(const RowTok &t) {
    int i = (t.len > 0 && (t.p[0] == '+' || t.p[0] == '-')) ? 1 : 0;
    if (i >= t.len) return false;
    for (; i < t.len; i++)
        if (t.p[i] < '0' || t.p[i] > '9') return false;
    return true;
}

static bool row_tok_i64(const RowTok &t, int64_t &v) {
    char buf[40];
    const int len = std::min(t.len, 39);
    std::memcpy(buf, t.p, (size_t)len);
    buf[len] = 0;
    return parse_i64(buf, v);
}

// Seed list file: the first integer-like token of each line is the seed; the
// rest of the line is ignored, so a previous output.txt in either row
// format can be fed straight back in.
static bool load_seed_list(const char *path, std::vector<uint64_t> &out) {
    std::FILE *fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[2048];
    while (std::fgets(line, sizeof(line), fp)) {
        RowTok tok[8];
        const int nt = row_tokenize(line, tok, 8);
        for (int i = 0; i < nt; i++) {
            if (!row_tok_is_int(tok[i])) continue;
            int64_t v;
            if (!row_tok_i64(tok[i], v)) continue;
            out.push_back((uint64_t)v);
            break;
        }
    }
    std::fclose(fp);
    return true;
}

// Verify list file: "seed x z" per line, located as the first run of three
// integer-like tokens ('#' comments and blank lines skipped). Both output
// formats parse: the current score-first rows and the earlier seed-first
// ones.
//
// When the file carries a "# score seed ..." header with anchorX/anchorZ
// columns, the anchor coordinates win over the headline x/z. The live
// pipeline evaluates its gates (dark forest, height spiral, blob-fill seed)
// at the anchor, so verifying at the anchor reproduces the original verdict
// exactly. The headline x/z is a display coordinate (the best-pattern
// window's center) and can sit closer to peripheral features than the
// anchor did, such as a dark-forest patch the anchor-centered gate never
// saw. The enrichment recomputes the headline coordinates anyway, so
// nothing is lost.
static bool load_verify_list(const char *path, std::vector<GpuOutput> &out) {
    std::FILE *fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[4096];
    uint64_t skipped = 0;
    int seed_col = -1, anchor_x_col = -1, anchor_z_col = -1;
    bool have_header = false;
    while (std::fgets(line, sizeof(line), fp)) {
        // Column header: remember where the seed/anchor columns live. Updated
        // on every header line, so concatenated mixed-format files work.
        if (std::strncmp(line, "# score seed", 12) == 0) {
            RowTok tok[128];
            const int nt = row_tokenize(line + 1, tok, 128);
            seed_col = anchor_x_col = anchor_z_col = -1;
            have_header = true;
            for (int i = 0; i < nt; i++) {
                if (tok[i].len == 4 && std::strncmp(tok[i].p, "seed", 4) == 0)
                    seed_col = i;
                else if (tok[i].len == 7 && std::strncmp(tok[i].p, "anchorX", 7) == 0)
                    anchor_x_col = i;
                else if (tok[i].len == 7 && std::strncmp(tok[i].p, "anchorZ", 7) == 0)
                    anchor_z_col = i;
            }
            continue;
        }
        RowTok tok[128];
        const int nt = row_tokenize(line, tok, 128);
        if (nt == 0) continue; // blank line or '#' comment
        bool ok = false;
        if (have_header && seed_col >= 0 && anchor_x_col >= 0 && anchor_z_col >= 0) {
            const int need = std::max(seed_col, std::max(anchor_x_col, anchor_z_col));
            if (nt > need &&
                row_tok_is_int(tok[seed_col]) &&
                row_tok_is_int(tok[anchor_x_col]) &&
                row_tok_is_int(tok[anchor_z_col])) {
                int64_t seed, ax, az;
                if (row_tok_i64(tok[seed_col], seed) &&
                    row_tok_i64(tok[anchor_x_col], ax) &&
                    row_tok_i64(tok[anchor_z_col], az)) {
                    out.push_back({ (uint64_t)seed, (int32_t)ax, (int32_t)az });
                    ok = true;
                }
            }
        }
        if (!ok) {
            // Legacy: first run of three integer tokens = seed x z.
            for (int i = 0; i + 2 < nt && !ok; i++) {
                if (!row_tok_is_int(tok[i]) || !row_tok_is_int(tok[i + 1]) || !row_tok_is_int(tok[i + 2]))
                    continue;
                int64_t seed, x, z;
                if (!row_tok_i64(tok[i], seed) ||
                    !row_tok_i64(tok[i + 1], x) ||
                    !row_tok_i64(tok[i + 2], z))
                    continue;
                out.push_back({ (uint64_t)seed, (int32_t)x, (int32_t)z });
                ok = true;
            }
        }
        if (!ok) skipped++;
    }
    std::fclose(fp);
    if (skipped)
        std::fprintf(stderr, "Note: skipped %" PRIu64 " malformed lines in %s\n", skipped, path);
    return true;
}

// Dedup-preload parser: like load_verify_list, but keyed on the trailing
// anchorX/anchorZ of current-format output rows. The pipeline re-emits at
// anchor coordinates on resume, so preloading anchors (rather than headline
// coordinates) keeps the full dedup radius effective. (Only correct for the
// current score-first format; an earlier seed-first row would misread its
// trailing integer columns as coordinates.)
static int preload_seed_col = -1, preload_ax_col = -1, preload_az_col = -1;

static bool load_preload_list(const char *path, std::vector<GpuOutput> &out) {
    std::FILE *fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[2048];
    while (std::fgets(line, sizeof(line), fp)) {
        // Header-aware: prefer the anchorX/anchorZ columns (dedup anchors at
        // the same coordinates the pipeline re-emits). Falls back to the last
        // two integer tokens for header-less rows.
        if (std::strncmp(line, "# score seed", 12) == 0) {
            RowTok htok[128];
            const int hn = row_tokenize(line + 1, htok, 128);
            preload_seed_col = preload_ax_col = preload_az_col = -1;
            for (int i = 0; i < hn; i++) {
                if (htok[i].len == 4 && std::strncmp(htok[i].p, "seed", 4) == 0)
                    preload_seed_col = i;
                else if (htok[i].len == 7 && std::strncmp(htok[i].p, "anchorX", 7) == 0)
                    preload_ax_col = i;
                else if (htok[i].len == 7 && std::strncmp(htok[i].p, "anchorZ", 7) == 0)
                    preload_az_col = i;
            }
            continue;
        }
        RowTok tok[128];
        const int nt = row_tokenize(line, tok, 128);
        if (preload_seed_col >= 0 && preload_ax_col >= 0 && preload_az_col >= 0) {
            const int need = std::max(preload_seed_col, std::max(preload_ax_col, preload_az_col));
            if (nt > need &&
                row_tok_is_int(tok[preload_seed_col]) &&
                row_tok_is_int(tok[preload_ax_col]) &&
                row_tok_is_int(tok[preload_az_col])) {
                int64_t seed, ax, az;
                if (row_tok_i64(tok[preload_seed_col], seed) &&
                    row_tok_i64(tok[preload_ax_col], ax) &&
                    row_tok_i64(tok[preload_az_col], az)) {
                    out.push_back({ (uint64_t)seed, (int32_t)ax, (int32_t)az });
                    continue;
                }
            }
        }
        int64_t vals[128];
        int nv = 0;
        for (int i = 0; i < nt && nv < 128; i++)
            if (row_tok_is_int(tok[i]) && row_tok_i64(tok[i], vals[nv])) nv++;
        if (nv < 3) continue; // need a seed plus a coordinate pair
        out.push_back({ (uint64_t)vals[0], (int32_t)vals[nv - 2], (int32_t)vals[nv - 1] });
    }
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// --list-seeds row parsing and sorting
// ---------------------------------------------------------------------------

enum ListSortMode { LIST_SEED, LIST_SCORE, LIST_AREA, LIST_CORE };

struct ListRow {
    uint64_t seed;
    double  score = std::numeric_limits<double>::quiet_NaN(); // NaN = none on the row
    int64_t area  = -1;  // -1 = absent/unparsed
    int64_t core  = -1;
    uint64_t ord  = 0;   // original line order (stable tiebreak)
};

static bool row_tok_f64(const RowTok &t, double &v) {
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

// Parses one output or probe row. The seed/x/z run anchors the row; the
// layouts are told apart by the shape of the integer run after the
// headline z:
//   current search rows:  score seed x z AREA CORE maxY <decimal eroCov> ...   (3+ ints after z)
//   probe rows:           seed x z AREA CORE coreBlocks edge nCells ...        (3+ ints after z)
//   earlier search rows:  score seed x z maxY <decimal eroCov> ... AREA CORE   (1 int after z)
// The score is the decimal token right before the seed (search rows), or the
// trailing decimal token (probe rows print it last).
static bool parse_output_row(const char *line, ListRow &out) {
    RowTok tok[128]; // probe rows run to ~74 tokens
    const int nt = row_tokenize(line, tok, 128);
    if (nt < 3) return false;

    int base = -1;
    for (int i = 0; i + 2 < nt; i++) {
        if (row_tok_is_int(tok[i]) && row_tok_is_int(tok[i + 1]) && row_tok_is_int(tok[i + 2])) {
            base = i;
            break;
        }
    }
    if (base < 0) return false;

    int64_t seed, x, z;
    if (!row_tok_i64(tok[base], seed) ||
        !row_tok_i64(tok[base + 1], x) ||
        !row_tok_i64(tok[base + 2], z)) return false;
    out.seed = (uint64_t)seed;

    if (base > 0 && !row_tok_is_int(tok[base - 1])) {
        double v;
        if (row_tok_f64(tok[base - 1], v)) out.score = v;
    } else if (nt > base + 3 && !row_tok_is_int(tok[nt - 1])) {
        double v;
        if (row_tok_f64(tok[nt - 1], v)) out.score = v;
    }

    int ints_after = 0;
    for (int i = base + 3; i < nt && row_tok_is_int(tok[i]); i++) ints_after++;
    if (ints_after >= 3) {           // new search layout / probe layout
        row_tok_i64(tok[base + 3], out.area);
        row_tok_i64(tok[base + 4], out.core);
    } else if (ints_after == 1) {    // old search layout
        if (base + 7 < nt) row_tok_i64(tok[base + 7], out.area);
        if (base + 8 < nt) row_tok_i64(tok[base + 8], out.core);
    }
    return true;
}

static bool load_output_rows(const char *path, std::vector<ListRow> &out) {
    std::FILE *fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[2048];
    while (std::fgets(line, sizeof(line), fp)) {
        ListRow r;
        r.ord = (uint64_t)out.size();
        if (parse_output_row(line, r)) out.push_back(r);
    }
    std::fclose(fp);
    return true;
}

// Sort key per mode. Unmeasured values (NaN score, absent area/core) map to
// -inf so they sink to the bottom of a descending sort.
static double list_row_key(const ListRow &r, ListSortMode mode) {
    switch (mode) {
    case LIST_SCORE: return std::isnan(r.score) ? -std::numeric_limits<double>::infinity() : r.score;
    case LIST_AREA:  return (double)r.area;
    case LIST_CORE:  return (double)r.core;
    default:         return 0.0;
    }
}

// ---------------------------------------------------------------------------
// Command-line arguments
// ---------------------------------------------------------------------------

// Defined below; referenced by Args::parse for --help.
static void print_usage(const char *prog);

struct Args {
    std::vector<int> devices;
    std::optional<int> threads;
    bool close1 = true;                     // --closing / --no-closing
    std::optional<std::string> output_file;
    std::optional<int64_t> start_seed;      // presence-tracked: --start 0 is valid
    std::string mc_str = "1.21.3";
    std::optional<std::string> seeds_file;  // --seeds: list mode through the GPU
    std::optional<std::string> verify_file; // --verify: CPU-only mode
    std::optional<int32_t> min_height;
    std::optional<int32_t> max_height;
    std::optional<std::string> probe_file;  // --probe: gates-free measurement
    std::optional<std::string> thumbs_dir;  // --thumbs: BMP thumbnails dir
    double thumb_min_score = -1e300;        // --thumb-min-score
    std::optional<std::string> list_seeds_file; // --list-seeds: dump seed column
    ListSortMode sort = LIST_SEED;          // --sort (with --list-seeds)
    bool sort_seen = false;
    bool shuffled_search = false;           // --shuffled-search
    int  ui_level = UI_NORMAL;              // --debug / --verbose
    bool no_height = false;
    bool no_dark_forest = false;
    bool no_blob = false;
    bool no_stats = false;
    bool no_aggregate = false;

    // Configurable CPU-side gates (defaults = the shared blob field, common.h)
    int64_t min_blob_area = MIN_BLOB_AREA;
    int32_t min_core_width = MIN_CORE_WIDTH_CELLS;
    int32_t dedup_radius = DEDUP_RADIUS_BLOCKS;
    float ero_max  = BLOB_ERO_MAX;
    float cont_min = BLOB_CONT_MIN;
    float temp_min = BLOB_TEMP_MIN;
    float temp_max = BLOB_TEMP_MAX;
    double dark_forest_frac = 0.07; // --dark-forest-frac
    double min_score = -std::numeric_limits<double>::infinity(); // --min-score
    int32_t enrich_phases = 4; // --phases

    bool parse(int argc, const char **const argv) {
        for (int i = 1; i < argc;) {
            const char *arg = argv[i++];

            auto need_value = [&](const char *name) -> bool {
                if (i >= argc) {
                    std::fprintf(stderr, "Missing argument to %s\n", name);
                    return false;
                }
                return true;
            };

            if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
                print_usage(argv[0]);
                std::exit(0);
            } else if (std::strcmp(arg, "--version") == 0) {
                std::printf("MountainRangeFinder %s\n", MRF_VERSION);
                std::exit(0);
            } else if (std::strcmp("--device", arg) == 0) {
                if (!need_value(arg)) return false;
                const char *devices_str = argv[i++];
                const char *last = devices_str + std::strlen(devices_str);
                const char *first = devices_str;
                while (first != last) {
                    int device;
                    auto [ptr, ec] = std::from_chars(first, last, device, 10);
                    if (ec != std::errc() || device < 0 ||
                        std::find(devices.begin(), devices.end(), device) != devices.end() ||
                        (ptr != last && *ptr != ',')) {
                        std::fprintf(stderr, "Invalid argument to --device: %s\n", devices_str);
                        return false;
                    }
                    devices.push_back(device);
                    first = ptr;
                    if (first != last) first++;
                }
            } else if (std::strcmp("--threads", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v) || v < 1 || v > 1024) {
                    std::fprintf(stderr, "Invalid thread count. Must be 1-1024.\n");
                    return false;
                }
                threads = (int)v;
            } else if (std::strcmp("--output", arg) == 0) {
                if (!need_value(arg)) return false;
                output_file = argv[i++];
            } else if (std::strcmp("--start", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v)) {
                    std::fprintf(stderr, "Invalid argument to --start\n");
                    return false;
                }
                start_seed = v; // zero and negatives are legitimate seeds
            } else if (std::strcmp("--mc", arg) == 0) {
                if (!need_value(arg)) return false;
                mc_str = argv[i++];
            } else if (std::strcmp("--seeds", arg) == 0) {
                if (!need_value(arg)) return false;
                seeds_file = argv[i++];
            } else if (std::strcmp("--verify", arg) == 0) {
                if (!need_value(arg)) return false;
                verify_file = argv[i++];
            } else if (std::strcmp("--min-height", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v)) { std::fprintf(stderr, "Invalid --min-height\n"); return false; }
                min_height = (int32_t)v;
            } else if (std::strcmp("--max-height", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v)) { std::fprintf(stderr, "Invalid --max-height\n"); return false; }
                max_height = (int32_t)v;
            } else if (std::strcmp("--probe", arg) == 0) {
                if (!need_value(arg)) return false;
                probe_file = argv[i++];
            } else if (std::strcmp("--thumbs", arg) == 0) {
                if (!need_value(arg)) return false;
                thumbs_dir = argv[i++];
            } else if (std::strcmp("--thumb-min-score", arg) == 0) {
                if (!need_value(arg)) return false;
                try { thumb_min_score = std::stod(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --thumb-min-score\n"); return false; }
            } else if (std::strcmp("--list-seeds", arg) == 0) {
                if (!need_value(arg)) return false;
                list_seeds_file = argv[i++];
            } else if (std::strcmp("--sort", arg) == 0) {
                if (!need_value(arg)) return false;
                const char *m = argv[i++];
                if (std::strcmp(m, "seed") == 0)       sort = LIST_SEED;
                else if (std::strcmp(m, "score") == 0) sort = LIST_SCORE;
                else if (std::strcmp(m, "area") == 0)  sort = LIST_AREA;
                else if (std::strcmp(m, "core") == 0)  sort = LIST_CORE;
                else {
                    std::fprintf(stderr, "Invalid --sort mode '%s' (expected seed, score, area, or core)\n", m);
                    return false;
                }
                sort_seen = true;
            } else if (std::strcmp("--shuffled-search", arg) == 0) {
                shuffled_search = true;
            } else if (std::strcmp("--debug", arg) == 0) {
                ui_level = std::max(ui_level, (int)UI_DEBUG);
            } else if (std::strcmp("--verbose", arg) == 0) {
                ui_level = std::max(ui_level, (int)UI_VERBOSE);
            } else if (std::strcmp("--no-height", arg) == 0) {
                no_height = true;
            } else if (std::strcmp("--no-dark-forest", arg) == 0) {
                no_dark_forest = true;
            } else if (std::strcmp("--no-blob", arg) == 0) {
                no_blob = true;
            } else if (std::strcmp("--no-stats", arg) == 0) {
                no_stats = true;
            } else if (std::strcmp("--no-aggregate", arg) == 0) {
                no_aggregate = true;
            } else if (std::strcmp("--closing", arg) == 0) {
                close1 = true;
            } else if (std::strcmp("--no-closing", arg) == 0) {
                close1 = false;
            } else if (std::strcmp("--min-blob-area", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v)) { std::fprintf(stderr, "Invalid --min-blob-area\n"); return false; }
                min_blob_area = v;
            } else if (std::strcmp("--min-core-width", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v)) { std::fprintf(stderr, "Invalid --min-core-width\n"); return false; }
                min_core_width = (int32_t)v;
            } else if (std::strcmp("--dedup-radius", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v) || v < 0) { std::fprintf(stderr, "Invalid --dedup-radius\n"); return false; }
                dedup_radius = (int32_t)v;
            } else if (std::strcmp("--ero-max", arg) == 0) {
                if (!need_value(arg)) return false;
                try { ero_max = std::stof(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --ero-max\n"); return false; }
            } else if (std::strcmp("--cont-min", arg) == 0) {
                if (!need_value(arg)) return false;
                try { cont_min = std::stof(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --cont-min\n"); return false; }
            } else if (std::strcmp("--temp-range", arg) == 0) {
                if (!need_value(arg)) return false;
                try { temp_min = std::stof(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --temp-range min\n"); return false; }
                if (!need_value(arg)) return false;
                try { temp_max = std::stof(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --temp-range max\n"); return false; }
            } else if (std::strcmp("--dark-forest-frac", arg) == 0) {
                if (!need_value(arg)) return false;
                try { dark_forest_frac = std::stod(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --dark-forest-frac\n"); return false; }
            } else if (std::strcmp("--min-score", arg) == 0) {
                if (!need_value(arg)) return false;
                try { min_score = std::stod(argv[i++]); } catch(...) { std::fprintf(stderr, "Invalid --min-score\n"); return false; }
            } else if (std::strcmp("--phases", arg) == 0) {
                if (!need_value(arg)) return false;
                int64_t v;
                if (!parse_i64(argv[i++], v) || (v != 1 && v != 4)) { std::fprintf(stderr, "Invalid --phases (1 or 4)\n"); return false; }
                enrich_phases = (int32_t)v;
            } else {
                std::fprintf(stderr, "Unknown option: %s\n", arg);
                return false;
            }
        }

        if (seeds_file && verify_file) {
            std::fprintf(stderr, "--seeds and --verify are mutually exclusive\n");
            return false;
        }
        if (probe_file && (seeds_file || verify_file)) {
            std::fprintf(stderr, "--probe is mutually exclusive with --seeds and --verify\n");
            return false;
        }
        if (sort_seen && !list_seeds_file) {
            std::fprintf(stderr, "--sort only applies to --list-seeds\n");
            return false;
        }
        // Default to device 0 if none specified (verify mode uses no GPU).
        if (devices.empty() && !verify_file) {
            devices.push_back(0);
        }
        return true;
    }
};

static void print_usage(const char *prog) {
    std::fprintf(stderr,
        "Usage:\n%s [options]\n"
        "  --help                    display this message\n"
        "  --version                 print the program version and exit\n"
        "  --device <d,d,...>        CUDA devices (default: 0)\n"
        "  --threads <n>             CPU verifier threads (default: all hardware threads)\n"
        "  --output <file>           verified-seed output file (default: output.txt)\n"
        "  --start <seed>            first seed of the incremental search (default: random)\n"
        "  --mc <version>            MC version for CPU biome checks (default: 1.21.3; any 1.18+)\n"
        "  --seeds <file>            search only the seeds listed in <file> (one per line)\n"
        "  --verify <file>           CPU-verify \"seed x z\" lines from <file>; no GPU needed\n"
        "  --probe <file>            measure-only mode: reads \"seed x z [poly-x/z pairs...]\"\n"
        "                            lines and prints a wide stat vector per candidate\n"
        "  --thumbs <dir>            write a small BMP thumbnail of each measured blob\n"
        "                            into <dir> (filenames sort by score)\n"
        "  --list-seeds <file>       print the unique seeds from an output file, one per\n"
        "                            line, to stdout, then exit (pipe to clip/xclip for CV)\n"
        "  --shuffled-search         visit seeds in deterministic pseudo-random order:\n"
        "                            --start is a counter mapped 1:1 through a bijection,\n"
        "                            so no seed is ever searched twice (ignored with --seeds)\n"
        "  --debug                   also print the GPU per-stage funnel tables (~5 s)\n"
        "  --verbose                 full console firehose: every individual reject line\n"
        "                            (default: one compact line per verified seed + a live\n"
        "                            status line; plain 5-second progress lines in log files)\n"
        "  --sort <mode>             with --list-seeds: output order. seed (default,\n"
        "                            ascending), score, area, or core (all descending);\n"
        "                            duplicate rows of a seed keep the best-scoring one\n"
        "  --min-height <y>          approx-height window low (default: %d)\n"
        "  --max-height <y>          approx-height window high (default: %d)\n"
        "  --no-height               disable the surface-height check\n"
        "  --no-dark-forest          disable the dark-forest exclusion\n"
        "  --no-blob                 disable the contiguous-area measurement (and its gates;\n"
        "                            also disables scoring, which needs the blob)\n"
        "  --no-stats                disable CPU-side climate re-measurement\n"
        "  --closing                 enable 1-cell morphological closing (default)\n"
        "  --no-closing              disable closing (strict contiguity)\n"
        "  --min-blob-area <area>    minimum contiguous blob area, blocks^2 (default: %lld)\n"
        "  --min-core-width <cells>  minimum measured inscribed core radius, in 64-block\n"
        "                            lattice cells (default: %d; 0 disables the gate)\n"
        "  --dedup-radius <blocks>   drop verified points within this many blocks of an\n"
        "                            already-emitted point for the same seed\n"
        "                            (default: %d; 0 disables)\n"
        "  --min-score <score>       only write verified rows scoring at least this\n"
        "                            (default: -inf, keep everything; 0 drops the\n"
        "                            negative tail; output-only, no speed change)\n"
        "  --phases <1|4>            enrichment sampling phases per blob (default: 4 =\n"
        "                            measure all four 128-block sublattices and keep\n"
        "                            the best score)\n"
        "  --no-aggregate            keep the first-verified row per cluster instead\n"
        "                            of the best-scoring one (old dedup behavior)\n"
        "  --ero-max <float>         max 0B erosion for blob cells (default: %.2f)\n"
        "  --cont-min <float>        min 1B continentalness for blob cells (default: %.2f)\n"
        "  --temp-range <min> <max>  0B temperature window for blob cells (default: %.2f %.2f)\n"
        "\n"
        "Output rows are score-first: composite score, seed, then (x, z) = the center\n"
        "of the blob's best-pattern 1664x1664-block window (heart of the densest\n"
        "peaks); the original anchor coords trail at the end. --verify/--seeds/--probe\n"
        "accept both this format and the older seed-first one. Sort by column 1\n"
        "descending to rank.\n",
        prog, MIN_TARGET_Y, MAX_TARGET_Y, (long long)MIN_BLOB_AREA, MIN_CORE_WIDTH_CELLS,
        DEDUP_RADIUS_BLOCKS, BLOB_ERO_MAX, BLOB_CONT_MIN, BLOB_TEMP_MIN, BLOB_TEMP_MAX);
}

static uint64_t random_start_seed() {
    std::random_device rd;
    return ((uint64_t)rd() << 32) + (uint64_t)rd();
}

// Field printer: NaN doubles print as -999.99 (the not-measured sentinel),
// so partially measured rows stay machine-parseable.
static void fd(std::FILE *fp, double v) {
    if (std::isnan(v)) std::fprintf(fp, " -999.99");
    else               std::fprintf(fp, " %.4f", v);
}

// Console and file reporting for one verified candidate (shared by both
// modes). `index` is the 1-based running count of printed rows, shown as
// [#N] on the console.
static void print_cpu_output(std::FILE *out_fp, const CpuOutput &o, uint64_t index) {
    const EnrichStats &s = o.stats;

    char cov[24];
    if (o.ero_cov >= 0.0f) std::snprintf(cov, sizeof(cov), "%.1f%%", (double)(o.ero_cov * 100.0f));
    else                   std::snprintf(cov, sizeof(cov), "n/a");
    char scorebuf[24];
    if (std::isnan(o.score)) std::snprintf(scorebuf, sizeof(scorebuf), "n/a");
    else                     std::snprintf(scorebuf, sizeof(scorebuf), "%.2f", o.score);

    if (g_ui_level.load(std::memory_order_relaxed) < UI_VERBOSE) {
        // The compact one-liner; the full stat vector is on the file row
        // below. The coordinates printed here are the headline (the
        // best-pattern window's center, the point worth teleporting to);
        // the anchor stays in the file row.
        ui_eventf("[#%" PRIu64 "] seed %" PRIi64 " @ (%" PRIi32 ", %" PRIi32 ")"
                  " | score %s | blob %.2fM | core %" PRIi32 " | maxY %" PRIi32 "%s",
                  index, (int64_t)o.seed, o.x, o.z, scorebuf,
                  (double)o.blob_area / 1e6, o.core_radius, o.y_height,
                  o.blob_edge ? " | EDGE" : "");
    } else {
        ui_block_begin();
        std::printf("VERIFIED SEED: %" PRIi64 " at X:%" PRIi32 " Z:%" PRIi32
                    " | score %s | maxY %" PRIi32 " | cov %s | eroMin %.3f | contMax %.3f"
                    " | blob %.2fM blocks^2 | coreR %" PRIi32 " (%" PRIi32 " blocks)%s"
                    " | anchor (%" PRIi32 ", %" PRIi32 ") | gpuH %" PRIi32 " gpuDh %.1f",
                    (int64_t)o.seed, o.x, o.z, scorebuf, o.y_height, cov,
                    (double)o.ero_min, (double)o.cont_max,
                    (double)o.blob_area / 1e6, o.core_radius, o.core_radius * 64,
                    o.blob_edge ? " | EDGE-CLIPPED" : "",
                    o.anchor_x, o.anchor_z, o.gpu_h, o.gpu_dh8 / 8.0);
        if (s.valid) {
            std::printf(" | bio");
            for (int b = 0; b < CEN_N; b++)
                std::printf(" %s %.1f%%", census_bin_name(b), s.bio[b] * 100.0);
            double asp_a, asp_t, asp_f, asp_d;
            score_aspects(s, asp_a, asp_t, asp_f, asp_d);
            std::printf(" | aspects amp %.2f tex %.2f frag %.2f dep %.2f drk %.2f ext %.2f set %.2f",
                        asp_a, asp_t, asp_f, asp_d, score_dark_aspect(s),
                        score_extent(o.blob_area, o.core_radius, s), s.setting);
        }
        std::printf("\n");
        ui_block_end();
    }

    // File: score first, then seed, then the headline (window-center) coords.
    fd(out_fp, o.score);
    std::fprintf(out_fp, " %" PRIi64 " %" PRIi32 " %" PRIi32,
                 (int64_t)o.seed, o.x, o.z);
    std::fprintf(out_fp, " %" PRIi64 " %" PRIi32 " %" PRIi32,
                 (int64_t)o.blob_area, o.core_radius, o.y_height);
    std::fprintf(out_fp, " %.4f %.4f %.4f %" PRIi32,
                 (double)o.ero_cov, (double)o.ero_min, (double)o.cont_max,
                 o.blob_edge);
    for (int b = 0; b < CEN_N; b++) fd(out_fp, s.bio[b]);
    fd(out_fp, s.eroF_mean); fd(out_fp, s.eroF_p10); fd(out_fp, s.eroF_min);
    fd(out_fp, s.contF_mean); fd(out_fp, s.contF_max);
    fd(out_fp, s.w_abs); fd(out_fp, s.ridge); fd(out_fp, s.valley);
    fd(out_fp, s.rcross); fd(out_fp, s.vcross);
    fd(out_fp, s.h_min); fd(out_fp, s.h_max); fd(out_fp, s.h_mean); fd(out_fp, s.h_p90);
    fd(out_fp, s.relief); fd(out_fp, s.high); fd(out_fp, s.low);
    fd(out_fp, s.w896_ero_min); fd(out_fp, s.w896_h_max); fd(out_fp, s.w896_mtn_max);
    std::fprintf(out_fp, " %" PRIi32 " %" PRIi32, s.w896_ero_x, s.w896_ero_z);
    fd(out_fp, s.w1664_ero_min); fd(out_fp, s.w1664_h_max); fd(out_fp, s.w1664_mtn_max);
    std::fprintf(out_fp, " %" PRIi32 " %" PRIi32, o.anchor_x, o.anchor_z);
    fd(out_fp, s.h_std); fd(out_fp, s.dh_mean);
    fd(out_fp, s.peak_density); fd(out_fp, s.peak_prom);
    fd(out_fp, s.high_comps); fd(out_fp, s.high_largest);
    fd(out_fp, s.dark_core_frac);
    fd(out_fp, s.w896_hstd_max); fd(out_fp, s.w896_hmean_min);
    fd(out_fp, s.w1664_hstd_max); fd(out_fp, s.w1664_hmean_min);
    fd(out_fp, s.bw896_sc);  fd(out_fp, s.bw896_ero);  fd(out_fp, s.bw896_hmean);
    fd(out_fp, s.bw896_hstd); fd(out_fp, s.bw896_dh);  fd(out_fp, s.bw896_mtn);
    fd(out_fp, s.bw896_hifrac); fd(out_fp, s.bw896_lofrac); fd(out_fp, s.bw896_ocn);
    std::fprintf(out_fp, " %" PRIi32 " %" PRIi32, s.bw896_x, s.bw896_z);
    fd(out_fp, s.bw1664_sc);  fd(out_fp, s.bw1664_ero);  fd(out_fp, s.bw1664_hmean);
    fd(out_fp, s.bw1664_hstd); fd(out_fp, s.bw1664_dh);  fd(out_fp, s.bw1664_mtn);
    fd(out_fp, s.bw1664_hifrac); fd(out_fp, s.bw1664_lofrac); fd(out_fp, s.bw1664_ocn);
    std::fprintf(out_fp, " %" PRIi32 " %" PRIi32, s.bw1664_x, s.bw1664_z);
    fd(out_fp, s.bw3200_sc);  fd(out_fp, s.bw3200_ero);  fd(out_fp, s.bw3200_hmean);
    fd(out_fp, s.bw3200_hstd); fd(out_fp, s.bw3200_dh);  fd(out_fp, s.bw3200_mtn);
    fd(out_fp, s.bw3200_hifrac); fd(out_fp, s.bw3200_lofrac); fd(out_fp, s.bw3200_ocn);
    std::fprintf(out_fp, " %" PRIi32 " %" PRIi32, s.bw3200_x, s.bw3200_z);
    fd(out_fp, s.bw1664_dark);
    // GPU tier-2 texture stats (-999 = n/a, e.g. --verify rows)
    if (o.gpu_h > 0)    std::fprintf(out_fp, " %" PRIi32, o.gpu_h);
    else                std::fprintf(out_fp, " -999");
    if (o.gpu_dh8 > 0)  std::fprintf(out_fp, " %.2f", o.gpu_dh8 / 8.0);
    else                std::fprintf(out_fp, " -999.99");
    fd(out_fp, s.max_dark_896_near);
    fd(out_fp, s.ocean_dt_hl);
    fd(out_fp, s.ocean_spiral);
    fd(out_fp, s.qual_n);
    fd(out_fp, s.qual_area);
    fd(out_fp, s.sc_q90);
    fd(out_fp, s.mtn_comp_m);
    fd(out_fp, s.mtn_comps);
    fd(out_fp, s.gorge_frac);
    fd(out_fp, s.low_comp_m);
    fd(out_fp, s.low_comps);
    fd(out_fp, s.vic_mush);
    fd(out_fp, s.vic_warmoc);
    fd(out_fp, s.lake_n);
    fd(out_fp, s.cliff_frac);
    fd(out_fp, s.vic_sea_min);
    fd(out_fp, s.vic_open_oc);
    fd(out_fp, s.vic_open_warm);
    fd(out_fp, s.vic_fjord);
    fd(out_fp, s.vic_isthmus);
    fd(out_fp, s.vic_badl);
    fd(out_fp, s.vic_badl_oth);
    fd(out_fp, s.vic_mush_b);
    fd(out_fp, s.vic_flower);
    fd(out_fp, s.vic_cherry);
    fd(out_fp, s.vic_mtg);
    fd(out_fp, s.mtg_comp_m);
    fd(out_fp, s.chy_comp_m);
    fd(out_fp, s.setting);
    std::fprintf(out_fp, "\n");
    // Flush every verified row: rows are rare enough (around one per several
    // million seeds) that a buffered row could sit in memory for minutes,
    // and a crash would silently lose it. One fflush per row costs nothing
    // at these rates.
    std::fflush(out_fp);
}

static uint64_t cpu_processed_total(const std::vector<std::unique_ptr<CpuThread>> &threads) {
    uint64_t n = 0;
    for (const auto &t : threads) n += t->processed.load(std::memory_order_relaxed);
    return n;
}

// ---------------------------------------------------------------------------
// --verify mode: CPU-only re-verification of "seed x z" candidates.
// ---------------------------------------------------------------------------
static int run_verify(const std::string &path, const VerifyConfig &vcfg, int threads,
                      std::FILE *output_file) {
    std::vector<GpuOutput> list;
    if (!load_verify_list(path.c_str(), list)) {
        std::fprintf(stderr, "Could not open %s\n", path.c_str());
        return 1;
    }
    if (list.empty()) {
        std::fprintf(stderr, "No candidates found in %s\n", path.c_str());
        return 1;
    }
    std::printf("Verifying %zu candidates from %s (CPU only)\n", list.size(), path.c_str());

    GpuOutputs gpu_outputs;
    CpuOutputs cpu_outputs;
    {
        std::lock_guard<std::mutex> lock(gpu_outputs.mutex);
        for (const auto &v : list) gpu_outputs.queue.push(v);
        gpu_outputs.total_pushed = (uint64_t)list.size();
    }

    std::vector<std::unique_ptr<CpuThread>> cpu_threads;
    for (int i = 0; i < threads; i++)
        cpu_threads.emplace_back(std::make_unique<CpuThread>(i, vcfg,
                                 std::ref(gpu_outputs), std::ref(cpu_outputs)));

    uint64_t verified = 0;
    const uint64_t total = (uint64_t)list.size();
    auto last_print = std::chrono::steady_clock::now();

    while (g_running.load(std::memory_order_relaxed)) {
        {
            std::lock_guard<std::mutex> lock(cpu_outputs.mutex);
            while (!cpu_outputs.queue.empty()) {
                verified++;
                print_cpu_output(output_file, cpu_outputs.queue.front(), verified);
                cpu_outputs.queue.pop();
            }
        }
        if (cpu_processed_total(cpu_threads) == total) break;

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_print).count() >= 2.0) {
            std::printf("verify progress: %" PRIu64 "/%" PRIu64 " processed, %" PRIu64 " verified\n",
                        cpu_processed_total(cpu_threads), total, verified);
            last_print = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    for (auto &t : cpu_threads) t->stop();
    for (auto &t : cpu_threads) t->join();
    {
        std::lock_guard<std::mutex> lock(cpu_outputs.mutex);
        while (!cpu_outputs.queue.empty()) {
            verified++;
            print_cpu_output(output_file, cpu_outputs.queue.front(), verified);
            cpu_outputs.queue.pop();
        }
    }
    std::printf("Verify complete: %" PRIu64 "/%" PRIu64 " candidates passed.\n", verified, total);
    return 0;
}

// ---------------------------------------------------------------------------
// The main application loop.
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Args args{};
    if (!args.parse(argc, const_cast<const char **const>(argv))) {
        print_usage(argv[0]);
        return 1;
    }
    g_min_score = args.min_score;

    // --list-seeds: a standalone utility mode. Parse an output/probe file,
    // keep the best row per seed under the sort key, print the seeds in
    // sorted order, and exit, before any GPU or output plumbing is touched.
    if (args.list_seeds_file) {
        std::vector<ListRow> rows;
        if (!load_output_rows(args.list_seeds_file->c_str(), rows)) {
            std::fprintf(stderr, "Could not open %s\n", args.list_seeds_file->c_str());
            return 1;
        }
        const ListSortMode mode = args.sort;

        // Seed-major pre-pass: within each seed's duplicate rows, the one
        // with the best key lands first, and std::unique keeps exactly it.
        std::stable_sort(rows.begin(), rows.end(), [&](const ListRow &a, const ListRow &b) {
            if (a.seed != b.seed) return a.seed < b.seed;
            return list_row_key(a, mode) > list_row_key(b, mode);
        });
        rows.erase(std::unique(rows.begin(), rows.end(),
                               [](const ListRow &a, const ListRow &b) { return a.seed == b.seed; }),
                   rows.end());

        // Presentation order (stable: equal keys fall back to ascending seed).
        if (mode != LIST_SEED)
            std::stable_sort(rows.begin(), rows.end(), [&](const ListRow &a, const ListRow &b) {
                return list_row_key(a, mode) > list_row_key(b, mode);
            });

        for (const ListRow &r : rows)
            std::printf("%" PRIi64 "\n", (int64_t)r.seed);

        const char *mode_name = mode == LIST_SEED  ? "seed" :
                                mode == LIST_SCORE ? "score" :
                                mode == LIST_AREA  ? "blob area" : "core radius";
        std::fprintf(stderr, "%zu unique seeds (sorted by %s).\n", rows.size(), mode_name);
        return 0;
    }

    // Console verbosity + status-line plumbing (harmless in every mode).
    g_ui_level.store(args.ui_level, std::memory_order_relaxed);
    ui_init();

    // CPU-side biome generation version. The GPU's climate-noise math is
    // identical across 1.18-1.21.x, so this only affects the CPU checks
    // (dark forest, biome census), which use the version's biome tree.
    const int mc = str2mc(args.mc_str.c_str());
    if (mc == MC_UNDEF || mc < MC_1_18) {
        std::fprintf(stderr, "Unsupported --mc version '%s' (need 1.18+, e.g. 1.21.3)\n",
                     args.mc_str.c_str());
        return 1;
    }

    VerifyConfig vcfg;
    vcfg.mc_version        = mc;
    vcfg.check_height      = !args.no_height;
    vcfg.min_height        = args.min_height.value_or(MIN_TARGET_Y);
    vcfg.max_height        = args.max_height.value_or(MAX_TARGET_Y);
    vcfg.check_dark_forest = !args.no_dark_forest;
    vcfg.measure_blob      = !args.no_blob;
    vcfg.climate_stats     = !args.no_stats;
    vcfg.close1            = args.close1;

    // Configurable CPU-side gates
    vcfg.min_blob_area     = args.min_blob_area;
    vcfg.min_core_width    = args.min_core_width;
    vcfg.dedup_radius      = args.dedup_radius;
    vcfg.ero_max           = args.ero_max;
    vcfg.cont_min          = args.cont_min;
    vcfg.temp_min          = args.temp_min;
    vcfg.temp_max          = args.temp_max;
    vcfg.dark_forest_frac  = args.dark_forest_frac;
    vcfg.enrich_phases     = args.enrich_phases;

    // --verify re-measures an existing list; self-dedup against that list
    // would silently drop its same-seed near-duplicates, so it is disabled
    // outright there. (The CPU threads consult cfg.dedup_radius per point.)
    if (args.verify_file)
        vcfg.dedup_radius = 0;

    // Best-of-seed aggregation replaces first-claim-wins dedup in the search
    // and --seeds modes (see the aggregator around drain_outputs below).
    // --verify wants every row verbatim, so it stays off there.
    vcfg.aggregate = !args.no_aggregate && !args.verify_file;

    // Default CPU verifier pool: all hardware threads. The per-candidate cost
    // is dominated by the enrichment samplers now, so more threads genuinely
    // help; override with --threads if you want to leave the box headroom.
    const unsigned hw = std::thread::hardware_concurrency();
    const int default_threads = (int)std::max(1u, hw);
    const int threads = args.threads.value_or(default_threads);
    const std::string output_path = args.output_file.value_or(
        args.probe_file ? "probe_results.txt" : "output.txt");

    std::printf("Starting MountainRangeFinder v%s...\n", MRF_VERSION);
    std::printf("MC version: %s | CPU verifier threads: %d\n", mc2str(mc), threads);
    std::printf("Verify: height=%s [%d..%d], dark forest=%s (max %.1f%%), blob area=%s (gate=%lld), climate stats=%s\n",
                vcfg.check_height ? "on" : "off", vcfg.min_height, vcfg.max_height,
                vcfg.check_dark_forest ? "on" : "off", vcfg.dark_forest_frac * 100.0,
                vcfg.measure_blob ? "on" : "off",
                (long long)vcfg.min_blob_area,
                vcfg.climate_stats ? "on" : "off");
    std::printf("Blob Climate Gates: ero_max=%.2f, cont_min=%.2f, temp_range=[%.2f..%.2f], "
                "min_core_width=%d, dedup_radius=%d, phases=%d\n",
                vcfg.ero_max, vcfg.cont_min, vcfg.temp_min, vcfg.temp_max,
                vcfg.min_core_width, vcfg.dedup_radius, vcfg.enrich_phases);
    std::printf("Output file: %s\n", output_path.c_str());
    if (args.min_score > -std::numeric_limits<double>::infinity())
        std::printf("Output filter: score >= %.2f (seeds with lower scores are ignored)\n",
                    args.min_score);

    // Open the output file in append mode so searches can be resumed.
    std::FILE *output_file = std::fopen(output_path.c_str(), "a");
    if (output_file == nullptr) {
        std::fprintf(stderr, "Could not open output file %s\n", output_path.c_str());
        return 1;
    }
    // Column header, written as a '#' comment so re-loading the file (in any
    // input slot) skips it silently. Written only when the file is empty
    // (fresh output); append mode positions the stream at EOF, so ftell == 0
    // means "new file".
    if (std::ftell(output_file) == 0) {
    std::fprintf(output_file,
        "# score seed x z blobAreaBlocks2 coreRadiusCells maxY eroCovFrac eroMin contMax edgeClipped"
        " bioMtn bioMtOg bioTg bioMdw bioChy bioDrk bioPln bioRiv bioOcn"
        " eroFmean eroFp10 eroFmin contFmean contFmax wAbs ridge valley rcross vcross"
        " hMin hMax hMean hP90 relief high low"
        " w896eroMin w896hMax w896mtnMax w896eroX w896eroZ"
        " w1664eroMin w1664hMax w1664mtnMax anchorX anchorZ"
        " hStd dhMean peakDensity peakProm highComps highLargest darkCoreFrac"
        " w896hStdMax w896hMeanMin w1664hStdMax w1664hMeanMin"
        " bw896Sc bw896Ero bw896hMean bw896hStd bw896Dh bw896Mtn bw896Hi bw896Lo bw896Ocn bw896X bw896Z"
        " bw1664Sc bw1664Ero bw1664hMean bw1664hStd bw1664Dh bw1664Mtn bw1664Hi bw1664Lo bw1664Ocn bw1664X bw1664Z"
        " bw3200Sc bw3200Ero bw3200hMean bw3200hStd bw3200Dh bw3200Mtn bw3200Hi bw3200Lo bw3200Ocn bw3200X bw3200Z"
        " bw1664Dark gpuH gpuDh"
        " maxDark896Near oceanDtHl oceanSpiral qualN qualArea scQ90 mtnCompM mtnComps gorgeFrac"
        " lowCompM lowComps vicMush vicWarmOc"
        " lakeN cliffFrac vicSeaMin vicOpenOc vicOpenWarm vicFjord vicIsthmus"
        " vicBadl vicBadlOth vicMushB vicFlower vicCherry vicMtg mtgCompM chyCompM setting\n"
        "#   (x, z)     = center of the blob's best-pattern 1664x1664-block window\n"
        "#   anchorX/Z  = original pipeline hit point\n"
        "#   areas in blocks^2; ocean* in blocks; darkCoreFrac = DF ring at the headline\n"
        "#   -999.99 / -999 = not measured; sort by column 1 DESCENDING to rank (mc=%s)\n",
        mc2str(mc));
    std::fflush(output_file);
    }

    // Register signal handler for graceful shutdown.
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (args.thumbs_dir) probe_enable_thumbs(args.thumbs_dir->c_str(), args.thumb_min_score);

    // --verify mode: CPU only, no GPU.
    if (args.verify_file) {
        const int rc = run_verify(*args.verify_file, vcfg, threads, output_file);
        std::fclose(output_file);
        return rc;
    }

    // --probe mode: gates-free measurement, no GPU.
    if (args.probe_file) {
        const int rc = run_probe(*args.probe_file, vcfg, threads, output_file);
        std::fclose(output_file);
        return rc;
    }

    // Preload prior results from the (append-mode) output file, so a resumed
    // or repeated search doesn't re-emit points it already reported. The
    // loader keys on the trailing anchorX/anchorZ of each row, because the
    // pipeline re-emits at anchor coordinates on resume (load_preload_list).
    {
        std::vector<GpuOutput> prior;
        if (load_preload_list(output_path.c_str(), prior) && !prior.empty()) {
            const size_t n = dedup_preload(prior);
            std::printf("Dedup: preloaded %zu prior seeds from %s\n", n, output_path.c_str());
        }
    }

    // Initialize shared data structures for inter-thread communication.
    GpuOutputs gpu_outputs;
    CpuOutputs cpu_outputs;

    // Seed source: incremental (default) or a fixed list (--seeds).
    std::optional<SeedIterator> seed_range;
    if (args.seeds_file) {
        std::vector<uint64_t> seeds;
        if (!load_seed_list(args.seeds_file->c_str(), seeds)) {
            std::fprintf(stderr, "Could not open %s\n", args.seeds_file->c_str());
            std::fclose(output_file);
            return 1;
        }
        if (seeds.empty()) {
            std::fprintf(stderr, "No seeds found in %s\n", args.seeds_file->c_str());
            std::fclose(output_file);
            return 1;
        }
        std::printf("Searching %zu listed seeds from %s\n", seeds.size(), args.seeds_file->c_str());
        seed_range.emplace(std::move(seeds));
    } else {
        const uint64_t start_seed = args.start_seed.value_or(random_start_seed());
        seed_range.emplace(start_seed, args.shuffled_search);
        if (args.shuffled_search)
            std::printf("Starting shuffled search from counter: %" PRIu64
                        " (each counter value maps to a unique seed; resume with this counter)\n",
                        start_seed);
        else
            std::printf("Starting search from seed: %" PRIi64 "\n", (int64_t)start_seed);
    }

    // Push the blob field + size target to the device before workers start.
    // GPU target = 0.88x the CPU area gate, so the float32 prefilter can never
    // be stricter than the double-precision CPU measurement.
    gpu_set_mc(mc);
    gpu_set_blob_cfg(vcfg.ero_max, vcfg.cont_min, vcfg.temp_min, vcfg.temp_max,
                     (uint32_t)std::max<int64_t>(1024, (int64_t)(vcfg.min_blob_area * 0.88 / 4096.0)));

    // Spawn GPU worker threads.
    std::vector<std::unique_ptr<GpuThread>> gpu_threads;
    for (int device : args.devices) {
        gpu_threads.emplace_back(std::make_unique<GpuThread>(device, std::ref(*seed_range), std::ref(gpu_outputs)));
    }

    // Spawn CPU worker threads for exact biome and height verification.
    std::vector<std::unique_ptr<CpuThread>> cpu_threads;
    for (int i = 0; i < threads; i++) {
        cpu_threads.emplace_back(std::make_unique<CpuThread>(i, vcfg, std::ref(gpu_outputs), std::ref(cpu_outputs)));
    }

    // ------------------------------------------------------------------
    // Best-of-seed aggregation (search and --seeds modes).
    //
    // The GPU emits every candidate for a seed within one batch, so all of a
    // seed's verified rows arrive within a second or two of each other.
    // Which anchor wins still matters, though: the blob fill's measurement
    // lattice is phased by the candidate point, so the same seed can
    // genuinely score differently per anchor (one known-good region has
    // produced both a -5.9 row and a +7.3 row from different anchors).
    // Rather than letting whichever verifier thread finishes first claim
    // the seed, verified rows are held briefly, clustered by anchor
    // proximity (chain-linked at the dedup radius), and only the
    // best-scoring row of each cluster is printed. CPU cost is unchanged
    // either way: the runner-up candidates were already fully verified by
    // the time they are dropped.
    //
    // Rows are flushed once their seed has been quiet for
    // AGGREGATE_FLUSH_AGE_S seconds, and unconditionally at list completion
    // or shutdown. Output rows therefore lag verification by a few seconds,
    // which is cosmetic only.
    // ------------------------------------------------------------------
    const bool aggregate = vcfg.aggregate;
    constexpr double AGGREGATE_FLUSH_AGE_S = 3.0;
    struct PendingRow {
        CpuOutput row;
        std::chrono::steady_clock::time_point t;
    };
    // pending holds rows keyed by seed until they flush; pending_order tracks
    // (first-seen time, seed) pairs in arrival order, so flush_pending can
    // age out the oldest seeds first.
    std::unordered_map<uint64_t, std::vector<PendingRow>> pending;
    std::deque<std::pair<std::chrono::steady_clock::time_point, uint64_t>> pending_order;

    uint64_t verified = 0;
    double best_score = -std::numeric_limits<double>::infinity();

    auto flush_seed = [&](uint64_t seed) {
        auto it = pending.find(seed);
        if (it == pending.end()) return;
        const std::vector<PendingRow> &rows = it->second;
        const int64_t r2 = (int64_t)vcfg.dedup_radius * vcfg.dedup_radius;

        // Cluster rows by anchor proximity: a row joins a cluster if its
        // anchor lies within the dedup radius of any member's anchor.
        std::vector<int32_t> cl(rows.size(), -1);
        int32_t ncl = 0;
        for (size_t i = 0; i < rows.size(); i++) {
            if (cl[i] >= 0) continue;
            cl[i] = ncl;
            std::vector<size_t> stack(1, i);
            while (!stack.empty()) {
                const size_t a = stack.back();
                stack.pop_back();
                for (size_t j = 0; j < rows.size(); j++) {
                    if (cl[j] >= 0) continue;
                    const int64_t dx = (int64_t)rows[j].row.anchor_x - rows[a].row.anchor_x;
                    const int64_t dz = (int64_t)rows[j].row.anchor_z - rows[a].row.anchor_z;
                    if (dx * dx + dz * dz < r2) {
                        cl[j] = ncl;
                        stack.push_back(j);
                    }
                }
            }
            ncl++;
        }
        // Keep the best-scoring row per cluster. NaN scores (possible under
        // --no-blob) never win the comparison, so the first row stands.
        for (int32_t c2 = 0; c2 < ncl; c2++) {
            size_t best = (size_t)-1;
            for (size_t i = 0; i < rows.size(); i++)
                if (cl[i] == c2 &&
                    (best == (size_t)-1 || rows[i].row.score > rows[best].row.score))
                    best = i;
            if (best != (size_t)-1) {
                const CpuOutput &row = rows[best].row;
                const double sc = std::isnan(row.score) ? -std::numeric_limits<double>::infinity()
                                                        : row.score;
                if (sc >= g_min_score) {
                    verified++;
                    print_cpu_output(output_file, row, verified);
                    if (!std::isnan(row.score) && row.score > best_score)
                        best_score = row.score;
                }
            }
        }
        pending.erase(it);
    };

    auto flush_pending = [&](bool all) {
        while (!pending_order.empty()) {
            const auto &front = pending_order.front();
            const double age = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - front.first).count();
            if (!all && age < AGGREGATE_FLUSH_AGE_S) break;
            pending_order.pop_front();
            flush_seed(front.second);
        }
    };

    auto drain_outputs = [&]() {
        std::lock_guard<std::mutex> lock(cpu_outputs.mutex);
        while (!cpu_outputs.queue.empty()) {
            if (aggregate) {
                const CpuOutput &row = cpu_outputs.queue.front();
                if (!pending.count(row.seed))
                    pending_order.emplace_back(std::chrono::steady_clock::now(), row.seed);
                pending[row.seed].push_back({ row, std::chrono::steady_clock::now() });
                cpu_outputs.queue.pop();
            } else {
                const CpuOutput &row = cpu_outputs.queue.front();
                const double sc = std::isnan(row.score) ? -std::numeric_limits<double>::infinity()
                                                        : row.score;
                if (sc >= g_min_score) {
                    verified++;
                    print_cpu_output(output_file, row, verified);
                    if (!std::isnan(row.score) && row.score > best_score)
                        best_score = row.score;
                }
                cpu_outputs.queue.pop();
            }
        }
        if (aggregate) flush_pending(false);
    };

    // Compact human-readable counts: 1.23G / 45.6M / 12.3k.
    auto fmt_count = [](double v, char *buf, size_t n) {
        if (v >= 1e9)      std::snprintf(buf, n, "%.2fG", v / 1e9);
        else if (v >= 1e6) std::snprintf(buf, n, "%.2fM", v / 1e6);
        else if (v >= 1e3) std::snprintf(buf, n, "%.1fk", v / 1e3);
        else               std::snprintf(buf, n, "%.0f", v);
    };

    const auto t_start = std::chrono::steady_clock::now();
    auto t_last = t_start;
    uint64_t prog_last = 0;
    double rate_smooth = 0.0;

    // A live status line on a TTY (a carriage-return rewrite; --verbose opts
    // out because the full reject log owns the screen there), or a plain
    // progress line every 5 seconds when stdout is a log file.
    const bool live_status = g_ui_tty && g_ui_level.load(std::memory_order_relaxed) < UI_VERBOSE;

    // Status line legend: the reject tallies read as 
    // rej a/c/d/h = area gate, core gate, dark forest, height.
    auto build_status = [&]() -> std::string {
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        const int eh = (int)(el / 3600.0), em = ((int)(el / 60.0)) % 60, es = ((int)el) % 60;
        const uint64_t prog = seed_range->list_mode ? seed_range->progress()
                                                    : seed_range->completed();
        size_t gq, cq;
        {
            std::lock_guard<std::mutex> lock(gpu_outputs.mutex);
            gq = gpu_outputs.queue.size();
        }
        {
            std::lock_guard<std::mutex> lock(cpu_outputs.mutex);
            cq = cpu_outputs.queue.size();
        }
        char nbuf[32];
        fmt_count((double)prog, nbuf, sizeof(nbuf));
        char line[352];
        if (seed_range->list_mode) {
            char tbuf[32];
            fmt_count((double)seed_range->total(), tbuf, sizeof(tbuf));
            std::snprintf(line, sizeof(line),
                "[%02d:%02d:%02d] %s/%s seeds @ %.2fM/s | verified %" PRIu64
                " | rej a:%" PRIu64 " c:%" PRIu64 " d:%" PRIu64 " h:%" PRIu64
                " | gpuq %zu cpuq %zu",
                eh, em, es, nbuf, tbuf, rate_smooth / 1e6, verified,
                g_rej_area.load(std::memory_order_relaxed),
                g_rej_core.load(std::memory_order_relaxed),
                g_rej_dark.load(std::memory_order_relaxed),
                g_rej_height.load(std::memory_order_relaxed), gq, cq);
        } else {
            std::snprintf(line, sizeof(line),
                "[%02d:%02d:%02d] %s seeds @ %.2fM/s | verified %" PRIu64
                " | rej a:%" PRIu64 " c:%" PRIu64 " d:%" PRIu64 " h:%" PRIu64
                " | gpuq %zu cpuq %zu",
                eh, em, es, nbuf, rate_smooth / 1e6, verified,
                g_rej_area.load(std::memory_order_relaxed),
                g_rej_core.load(std::memory_order_relaxed),
                g_rej_dark.load(std::memory_order_relaxed),
                g_rej_height.load(std::memory_order_relaxed), gq, cq);
        }
        std::string s(line);
        if (std::isfinite(best_score)) {
            char b[40];
            std::snprintf(b, sizeof(b), " | best %.2f", best_score);
            s += b;
        }
        return s;
    };

    while (g_running.load(std::memory_order_relaxed)) {
        drain_outputs();

        // Throughput / progress reporting. The seeds/sec rate updates on a
        // 5-second window; the status line itself refreshes once per loop
        // (~1 s) so the clock, queue depths and reject tallies stay live.
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 5.0) {
            const uint64_t prog = seed_range->list_mode ? seed_range->progress()
                                                        : seed_range->completed();
            rate_smooth = (double)(prog - prog_last) / dt;
            t_last = now;
            prog_last = prog;
            if (!live_status)
                std::printf("%s\n", build_status().c_str());
        }
        if (live_status)
            ui_status(build_status().c_str());

        // List-mode completion: source drained, GPU batches done, and every
        // emitted candidate accounted for by the CPU verifiers.
        if (seed_range->list_mode && seed_range->exhausted()) {
            bool gpu_done = true;
            for (auto &t : gpu_threads)
                gpu_done = gpu_done && t->finished.load(std::memory_order_relaxed);
            if (gpu_done &&
                cpu_processed_total(cpu_threads) == gpu_outputs.total_pushed.load(std::memory_order_relaxed)) {
                drain_outputs();
                if (aggregate) flush_pending(true);
                ui_eventf("Seed list exhausted.");
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // --- GRACEFUL SHUTDOWN SEQUENCE ---
    ui_status(""); // erase the status line; plain event prints take it from here
    ui_eventf("Shutting down... signaling worker threads to stop.");

    // 1. Stop the GPU threads. They will finish their current batch of seeds.
    for (auto &thread : gpu_threads) {
        (*thread).stop();
    }

    // 2. Stop the CPU threads. They will finish verifying their current candidate queue.
    for (auto &thread : cpu_threads) {
        (*thread).stop();
    }

    // 3. Wait for all GPU threads to join.
    ui_eventf("Waiting for GPU threads to finish...");
    for (auto &thread : gpu_threads) {
        (*thread).join();
    }

    // 4. Wait for all CPU threads to join.
    ui_eventf("Waiting for CPU threads to finish...");
    for (auto &thread : cpu_threads) {
        (*thread).join();
    }

    // 5. Final drain of the CPU output queue to catch any seeds verified
    // during the shutdown sequence. Aggregation: flush everything still held.
    drain_outputs();
    if (aggregate) flush_pending(true);

    // 6. Report the search extent; the start seed is printed so the search
    // can be resumed later with --start.
    const uint64_t total_checked = seed_range->list_mode
        ? seed_range->progress()
        : seed_range->completed();
    if (seed_range->list_mode) {
        ui_eventf("List search complete. Seeds checked: %" PRIu64 ", verified: %" PRIu64,
                  total_checked, verified);
    } else if (args.shuffled_search) {
        ui_eventf("Search complete. Seeds checked: %" PRIu64 ", verified: %" PRIu64
                  ". Resume with: --shuffled-search --start %" PRIu64 " (counter offset)",
                  total_checked, verified, seed_range->start + total_checked);
    } else {
        ui_eventf("Search complete. Start seed: %" PRIi64 ", seeds checked: %" PRIu64 ", verified: %" PRIu64,
                  (int64_t)seed_range->start, total_checked, verified);
    }

    std::fclose(output_file);
    return 0;
}
