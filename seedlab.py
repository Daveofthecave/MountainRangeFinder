#!/usr/bin/env python3
r"""
seedlab.py -- the comprehensive MountainRangeFinder seed analysis toolbox.

  python3 seedlab.py <command> [args]

Corpus overview & ranking
  stats       FILE...                     rows, unique seeds, %-negative, percentiles,
                                          gem counts, area stats, x/z uniformity smell test
  showtop     FILE... [--top 25] [--cols a,b,c] [--searched N]
  watchlist   FILE... [--k 8]             union of the per-aspect top-k lists

Score engine mirror (CONSTS/LEGACY mirror probe.cpp compute_score -- keep in sync!
use `cxx2py` on the generated Term block, then `check` a fresh --verify file)
  check       FILE...                     max |rescore - file score|
  explain     FILE... --seed S
  rerank      FILE... [--labels labels.csv]
  kfold       FILE... --labels labels.csv [--k 5]
  jitter      FILE... --seed S [--trials 120] [--frac 0.30]
  softmin     FILE... [--k 3]

Calibration / gates
  recal       CORPUS [LABELS.csv] [--todo N OUT.csv] [--hardneg X] [--canary S]...
  logistic    CORPUS --labels labels.csv [--k 5] [--l2 2.0] [--iters 2500]
  lens        FILE... [--gem 8] [--try "col>=t" ...]
  calib       FILE [--gem 8]              GPU tier-2 gate calibration (v6+ rows)

Labeling workflow
  audit sample FILE... [--bands 1-20,21-50,51-100,101-200,201-400] [--n 10] [--out audit]
  audit tally  --key audit_key.csv --labels audit_labels.csv
  positives   LABELS.csv [--min 1]        seeds with label >= min, one per line
                                          (build GPU recall lists with this!)

Seed lists
  consolidate PATH... [-o seed_atlas.txt] [--first-seen]

Misc
  cxx2py                                  stdin: C++ Term lines -> python LEGACY tuples

===============================================
USEFUL COMMANDS (copy-paste; adjust file names)
===============================================
Nightly triage of a fresh hunt:
  python3 seedlab.py stats output.txt
  python3 seedlab.py showtop output.txt --top 30
  python3 seedlab.py watchlist output.txt --k 12
  python3 seedlab.py explain output.txt --seed <S>

Magic-seed regression (run after any pipeline or score change):
  ./mountain_rangefinder --verify magic_seed.txt --output magic_check.txt
  python3 seedlab.py check magic_check.txt
  python3 seedlab.py explain magic_check.txt --seed 6696478651374553046

Full-funnel recall over every seed you have ever vouched for:
  python3 seedlab.py positives labels.csv --min 1 > recall_pos.txt
  ./mountain_rangefinder --seeds recall_pos.txt --output recall_run.txt
  python3 seedlab.py stats recall_run.txt

Score-model experiments (never ship a recalibration that fails these):
  python3 seedlab.py kfold output_all.txt --labels labels.csv
  python3 seedlab.py logistic output_all.txt --labels labels.csv
  python3 seedlab.py jitter output_all.txt --seed 6696478651374553046
  python3 seedlab.py softmin output_all.txt --k 3

Recalibration, long form:
  ./mountain_rangefinder --verify output.txt --output verified.txt
  python3 seedlab.py recal verified.txt --todo 150 --out todo.csv
  # Then label todo.csv blind (1 = magic-like, 0 = no) + append it to labels.csv
  python3 seedlab.py recal verified.txt --labels labels.csv --canary 6696478651374553046
  # if (and only if) the canary passes: paste the Term table into
  # probe.cpp compute_score(), regenerate this file's LEGACY table:
  python3 seedlab.py cxx2py < terms.txt
  make clean && make -j && make test && make recall
  ./mountain_rangefinder --verify verified.txt --output verified_v2.txt
  python3 seedlab.py check verified_v2.txt        # must stay ~0.00

GPU gate tuning:
  python3 seedlab.py lens output_all.txt --gem 8
  python3 seedlab.py lens output_all.txt --gem 8 --try "hMax>=220" "ridge>=0.015"
  python3 seedlab.py calib output_v6.txt --gem 8   # tier-2 BLOB2_* gates

"Are gems hiding below the fold?" (blind rank-band audit, ~50 eyeballs):
  python3 seedlab.py audit sample output_all.txt --n 10
  # ... label audit_labels.csv blind, then:
  python3 seedlab.py audit tally --key audit_key.csv --labels audit_labels.csv

Seed-list plumbing:
  python3 seedlab.py consolidate seedlists/ -o seed_atlas.txt
  python3 seedlab.py positives labels.csv --min 1 > good.txt

Quick & dirty one-liners:
# % of negative-score rows in any output file (stats also reports this):
awk '!/^#/ && NF>5 {n++; if ($1+0<0) neg++} END {printf "%.1f%% negative (%d/%d)\n", 100*neg/n, neg, n}' output_all.txt
# Where are the finds clustering? (headline x/z histogram, 4k buckets):
awk '!/^#/ && NF>5 {print int(\$3/4000), int(\$4/4000)}' output_all.txt | sort | uniq -c | sort -rn | head
"""

import math, os, random, re, statistics, sys

# ==========================================================================
# Shared row parsing
# ==========================================================================
FALLBACK_COLS = ("score seed x z blobAreaBlocks2 coreRadiusCells maxY eroCovFrac eroMin contMax "
    "edgeClipped bioMtn bioMtOg bioTg bioMdw bioChy bioDrk bioPln bioRiv bioOcn "
    "eroFmean eroFp10 eroFmin contFmean contFmax wAbs ridge valley rcross vcross "
    "hMin hMax hMean hP90 relief high low "
    "w896eroMin w896hMax w896mtnMax w896eroX w896eroZ "
    "w1664eroMin w1664hMax w1664mtnMax anchorX anchorZ "
    "hStd dhMean peakDensity peakProm highComps highLargest darkCoreFrac "
    "w896hStdMax w896hMeanMin w1664hStdMax w1664hMeanMin "
    "bw896Sc bw896Ero bw896hMean bw896hStd bw896Dh bw896Mtn bw896Hi bw896Lo bw896Ocn bw896X bw896Z "
    "bw1664Sc bw1664Ero bw1664hMean bw1664hStd bw1664Dh bw1664Mtn bw1664Hi bw1664Lo bw1664Ocn bw1664X bw1664Z "
    "bw3200Sc bw3200Ero bw3200hMean bw3200hStd bw3200Dh bw3200Mtn bw3200Hi bw3200Lo bw3200Ocn bw3200X bw3200Z "
    "bw1664Dark").split()

def norm_seed(s):
    try:
        v = int(str(s).strip())
    except (ValueError, TypeError):
        return None
    if v > (1 << 63) - 1: v -= 1 << 64
    if v < -(1 << 63): v += 1 << 64
    return str(v)

def load_rows(paths):
    """Header-aware parser for score-first output rows. Each file's own
    '# score seed ...' header is honored; headerless files fall back to the
    classic column layout. Returns dicts of floats; _seed = normalized string."""
    if isinstance(paths, str): paths = [paths]
    rows, skipped = [], 0
    for path in paths:
        names = None
        try:
            fh = open(path, errors="replace")
        except OSError:
            print("warning: cannot open %s" % path, file=sys.stderr)
            continue
        with fh:
            for line in fh:
                if line.startswith("# score seed"):
                    names = [str(x) for x in line[1:].split()]
                    continue
                if line.startswith("#"):
                    continue
                t = line.split()
                if not t:
                    continue
                if names is None:
                    names = list(FALLBACK_COLS)
                if len(t) < len(names):
                    skipped += 1
                    continue
                r = {}
                ok = True
                for i in range(len(names)):
                    nm = names[i]
                    if nm == "seed":
                        sv = norm_seed(t[i])
                        if sv is None: ok = False
                        else: r["_seed"] = sv
                        continue
                    try:
                        r[nm] = float(t[i])
                    except (ValueError, TypeError):
                        ok = False
                        break
                if ok and "_seed" in r: rows.append(r)
                else: skipped += 1
    if skipped:
        print("load_rows: skipped %d short/unparseable rows" % skipped, file=sys.stderr)
    return rows

def load_labels(path):
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or "," not in line: continue
        s, l = line.split(",", 1)
        sv = norm_seed(s)
        if sv is None: continue
        try: out[sv] = int(l.strip())
        except ValueError: pass
    return out

def g(r, c):
    v = r.get(c)
    if v is None or v <= -998.0: return None
    return v

def miss(v):
    return v is None or v <= -998.0

def dedupe_file(rows):
    """Best row per seed, keyed on the file's score column."""
    best = {}
    for r in rows:
        s = r["_seed"]
        sc = r.get("score")
        if sc is None: sc = float("-inf")
        if s not in best or sc > best[s][0]:
            best[s] = (sc, r)
    return [v[1] for v in best.values()]

# ==========================================================================
# Score engine mirror -- SYNC: mirrors score_aspects()/compute_score() in
# probe.cpp. After any recalibration, update both and re-run `check`.
# ==========================================================================
CONSTS = dict(HMAX_MU=242.078, HMAX_SD=10.348, PROM_MU=36.9412, PROM_SD=2.00566,
    VC_MU=0.300176, VC_SD=0.0167818, HC_MU=0.615029, HC_SD=0.243762,
    HL_MU=0.237651, HL_SD=0.104692, BWSTD_MU=40.0, BWSTD_SD=12.0,
    BWHI_MU=0.10, BWHI_SD=0.10, BWDH_MU=36.0, BWDH_SD=6.0,
    BOWL_MU=90.0, BOWL_SD=12.0, HMIN_MU=90.0, HMIN_SD=20.0,
    BWLO_MU=0.02, BWLO_SD=0.03, GORGE_MU=0.25, GORGE_SD=0.10)

LEGACY = [  # (column, mean, sd, weight) -- mirror of compute_score()'s Term table
    ("bw3200Dh",29.3085,2.15306,+1.17),
    ("bw3200hMean",113.044,6.05741,+1.04),
    ("bw3200Sc",-2.26601,0.713437,+1.04),
    ("dhMean",26.4897,1.36548,+0.99),
    ("bw1664Dh",36.0524,3.02595,+0.97),
    ("bw3200Hi",0.0311663,0.0160673,+0.96),
    ("peakProm",36.8023,2.02987,+0.91),
    ("highComps",0.586944,0.240445,+0.90),
    ("hP90",154.232,6.91857,+0.89),
    ("low",0.691115,0.0404762,-0.88),
    ("high",0.0165787,0.00786205,+0.87),
    ("hMean",104.98,4.15171,+0.86),
    ("bw3200Mtn",0.259228,0.0430392,+0.86),
    ("hStd",35.6727,2.1216,+0.85),
    ("bw3200hStd",38.2035,2.98361,+0.84),
    ("relief",0.013502,0.00466682,+0.83),
    ("w1664hMax",134.301,7.84006,+0.81),
    ("bw3200Ero",-0.660867,0.0571423,-0.81),
    ("bw1664Sc",-0.0102298,0.88565,+0.79),
    ("hMax",241.241,10.3986,+0.77),
    ("eroFmean",-0.582397,0.036329,-0.76),
    ("bioMtn",0.20646,0.0285928,+0.74),
    ("bioRiv",0.0340993,0.00772917,-0.71),
    ("bw1664hMean",127.056,9.96771,+0.70),
    ("bw3200Lo",0.0496073,0.0172566,-0.68),
    ("bw896Dh",43.9345,3.78023,+0.64),
    ("bw896Sc",2.20821,0.985784,+0.63),
    ("bw1664Hi",0.0756042,0.0385493,+0.59),
    ("contFmax",1.11621,0.147552,+0.59),
    ("contFmean",0.260204,0.056164,+0.57),
    ("w896hMax",153.584,9.93666,+0.57),
    ("eroFp10",-0.868355,0.0577004,-0.55),
    ("highLargest",0.250865,0.112722,-0.52),
    ("bw1664Dark",0.0276971,0.0375307,-0.50),
    ("bw896Hi",0.157411,0.0759688,+0.47),
    ("bw1664Lo",0.0343086,0.0230571,-0.46),
    ("w1664hMeanMin",80.4092,6.17331,+0.45),
    ("bw896hMean",143.39,13.8513,+0.43),
    ("w1664hStdMax",46.7241,3.63239,+0.41),
]

def _wm(pairs):
    s = sum(w * v for v, w in pairs if v is not None)
    w = sum(w for v, w in pairs if v is not None)
    return s / w if w > 0 else -3.0

def _wm_flat(pairs):
    s = 0.0; w = 0.0
    for v, wt in pairs:
        if v is not None and v == v:  # not None, not NaN
            s += wt * v; w += wt
    return s / w if w > 0 else -3.0

def _dark_aspect(dw, dr):
    if dw is None and dr is None: return 0.0
    z_win = 1.5 if dw is None else (0.05 - dw) / 0.005
    z_ring = 1.5 if dr is None else (0.25 - dr) / 0.05
    return max(-4.0, min(1.5, min(z_win, z_ring)))

def _ocean_dist(r):
    vs = [g(r, c2) for c2 in ("oceanDtHl", "oceanSpiral")]
    vs = [v for v in vs if v is not None]
    return min(vs) if vs else None

def _icing(od, ocn):
    if od is None or ocn is None: return 0.0
    if ocn > 0.08: return 0.0
    if od < 200: t = 0.0
    elif od < 640: t = (od - 200) / 440.0
    elif od <= 3000: t = 1.0
    elif od < 4800: t = (4800 - od) / 1800.0
    else: t = 0.0
    return 0.5 * t

# NOTE: icing/charm were retired in favor of the unified, core-gated setting
# bonus; _ocean_dist/_icing are kept (currently unused) for reference.
def _ramp_sea(od):
    if od is None: return 0.0
    if od < 200: return 0.0
    if od < 640: return (od - 200) / 440.0
    if od <= 3000: return 1.0
    if od < 4800: return (4800 - od) / 1800.0
    return 0.0

def _setting(r, key):
    """Mirror of setting_parts() in probe.cpp: the unified scenery bonus."""
    ods = [v for v in (g(r,"oceanDtHl"), g(r,"oceanSpiral"), g(r,"vicSeaMin")) if v is not None]
    od = min(ods) if ods else None
    prox = _ramp_sea(od)
    ocn = g(r,"bw1664Ocn")
    if ocn is not None and ocn > 0.08: prox *= 0.25
    fjord = g(r,"vicFjord") or 0.0
    ooc = g(r,"vicOpenOc") or 0.0
    owm = g(r,"vicOpenWarm") or 0.0
    op = 0.20 if ooc >= 2 else (0.10 if ooc >= 1 else 0.0)
    if owm >= 1: op += 0.05
    water = min(0.45*prox + 0.25*fjord + op, 0.55)
    isth = 0.15 if (g(r,"vicIsthmus") or 0.0) >= 0.5 and water > 0 else 0.0
    cliff = g(r,"cliffFrac")
    cliffs = 0.15*min(1.0, (cliff if cliff is not None else 0.0)/0.25)
    lakes = min(0.12*(g(r,"lakeN") or 0.0), 0.30)
    badl = g(r,"vicBadl") or 0.0; badlo = g(r,"vicBadlOth") or 0.0
    mush = g(r,"vicMushB") or 0.0; flow = g(r,"vicFlower") or 0.0
    vchy = g(r,"vicCherry") or 0.0; vmtg = g(r,"vicMtg") or 0.0
    ccomp = g(r,"chyCompM") or 0.0; mcomp = g(r,"mtgCompM") or 0.0
    odd = 0.0
    if badl > 0: odd += min(0.15, 0.05 + 0.01*badl)
    if badlo >= 8: odd += 0.06
    if mush > 0: odd += min(0.15, 0.08 + 0.01*mush)
    if flow >= 4: odd += min(0.10, 0.04 + 0.01*flow)
    if vchy >= 4 or ccomp >= 400000: odd += 0.10
    if vmtg >= 6 or mcomp >= 800000: odd += 0.10
    odd = min(odd, 0.30)
    gate = 1.0 if key >= 1.0 else (0.0 if key <= 0.5 else (key - 0.5)*2.0)
    total = min(water + isth + cliffs + lakes + odd, 0.80) * gate
    return dict(setting=total, set_water=water, set_isth=isth, set_cliff=cliffs,
                set_lake=lakes, set_odd=odd, set_gate=gate)

def score_full(r, C=CONSTS, terms=LEGACY, soft_k=None):
    def z(c, mu, sd):
        v = g(r, c)
        return None if v is None else (v - mu) / sd
    def zr(c, mu, sd):
        v = g(r, c)
        return None if v is None else (mu - v) / sd
    amp   = _wm([(z("hMax",C["HMAX_MU"],C["HMAX_SD"]),0.30),
                 (z("peakProm",C["PROM_MU"],C["PROM_SD"]),0.25),
                 (z("bw1664hStd",C["BWSTD_MU"],C["BWSTD_SD"]),0.25),
                 (z("bw1664Hi",C["BWHI_MU"],C["BWHI_SD"]),0.20)])
    tex   = _wm([(z("bw1664Dh",C["BWDH_MU"],C["BWDH_SD"]),0.55),
                 (z("bw896Dh",C["BWDH_MU"],C["BWDH_SD"]),0.30),
                 (z("vcross",C["VC_MU"],C["VC_SD"]),0.15)])
    hl    = z("highLargest",C["HL_MU"],C["HL_SD"])
    frag  = _wm([(z("highComps",C["HC_MU"],C["HC_SD"]),0.55),
                 (-hl if hl is not None else None,0.45)])
    depth = _wm([(zr("w896hMeanMin",C["BOWL_MU"],C["BOWL_SD"]),0.45),
                 (zr("hMin",C["HMIN_MU"],C["HMIN_SD"]),0.35),
                 (z("bw1664Lo",C["BWLO_MU"],C["BWLO_SD"]),0.20),
                 (z("gorgeFrac",C["GORGE_MU"],C["GORGE_SD"]),0.20)])
    darkasp = _dark_aspect(g(r,"bw1664Dark"), g(r,"darkCoreFrac"))
    if soft_k:
        key = -math.log(sum(math.exp(-soft_k*a) for a in (amp,tex,frag,depth,darkasp))/5.0)/soft_k
    else:
        key = min(amp,tex,frag,depth,darkasp)
    a_ = g(r,"blobAreaBlocks2"); c_ = g(r,"coreRadiusCells")
    qa = g(r,"qualArea");        mc = g(r,"mtnCompM")
    za = None if a_ is None else max(-2.0, min(3.5, (a_/1e6-20)/4))
    zc = None if c_ is None else (c_-21)/4
    zq = None if qa is None else max(-2.0, min(2.5, qa/1e6-2.0))
    zm = None if mc is None else max(-2.0, min(2.5, mc/1e6-2.5))
    sq9 = g(r,"scQ90")
    zs9 = None if sq9 is None else max(-2.0, min(2.5, (sq9 - -1.28639)/0.778071))
    ext = _wm([(za,0.25),(zc,0.15),(g(r,"bw3200Sc"),0.35),(zq,0.15),(zs9,0.15),(zm,0.10)])
    leg = 0.0; wsum = 0.0
    for col,mu,sd,w in terms:
        v = g(r,col)
        if v is not None and sd > 0:
            zl = (v-mu)/sd
            if zl > 2.5: zl = 2.5
            elif zl < -2.5: zl = -2.5
            leg += w*zl; wsum += abs(w)
    if wsum > 0: leg /= wsum
    dw = g(r,"bw1664Dark") or 0.0
    dark = 40.0*max(0.0, dw-0.055)
    setp = _setting(r, key)   # unified scenery bonus (mirror of probe.cpp)
    hl = g(r,"highLargest"); hl = hl if hl is not None else 0.0
    massif = 3.0*max(0.0, hl - 0.55)
    total = (2.0*key + 0.30*(amp+tex+frag+depth+darkasp)
             + 1.10*ext + 1.25*leg - dark - massif + setp["setting"])
    return dict(score=total, amp=amp, tex=tex, frag=frag, depth=depth,
                key=key, ext=ext, leg=leg, dark=dark, darkasp=darkasp,
                massif=massif, **setp)

def score_rows(rows, terms=LEGACY, C=CONSTS, soft_k=None):
    for r in rows:
        r["_p"] = score_full(r, C, terms, soft_k)

def dedupe_p(rows):
    best = {}
    for r in rows:
        s = r["_seed"]
        if s not in best or r["_p"]["score"] > best[s]["_p"]["score"]: best[s] = r
    return list(best.values())

def ranked(rows):
    return sorted(dedupe_p(rows), key=lambda r: -r["_p"]["score"])

# ==========================================================================
# Score-mirror commands
# ==========================================================================
def cmd_check(files, opts):
    rows = load_rows(files)
    worst, n = 0.0, 0
    for r in rows:
        fs = g(r, "score")
        if fs is None: continue
        d = abs(score_full(r)["score"] - fs)
        if d > worst: worst = d
        n += 1
    print("check: %d rows, max |rescore - file score| = %.4f" % (n, worst))
    print("  (<= ~0.02: faithful mirror -- print rounding dominates; larger: a constant drifted)")
    return 0

def cmd_explain(files, opts):
    seed = norm_seed(opts.get("--seed"))
    if seed is None:
        print("explain needs --seed S"); return 1
    rows = load_rows(files)
    found = 0
    for r in rows:
        if r["_seed"] != seed: continue
        found += 1
        p = score_full(r)
        print("seed %s  (%.0f, %.0f)  [row %d]" % (r["_seed"], r.get("x",0), r.get("z",0), found))
        print("  amp %.2f tex %.2f frag %.2f depth %.2f | min %.2f | ext %.2f" %
              (p["amp"], p["tex"], p["frag"], p["depth"], p["key"], p["ext"]))
        print("  2*min %+6.2f | 0.30*sum %+5.2f | 1.10*ext %+5.2f | 1.25*legacy %+5.2f (legacy mean %.2f) | dark %+.2f | massif %+.2f | setting %+.2f | darkAsp %.2f" %
              (2*p["key"], 0.30*(p["amp"]+p["tex"]+p["frag"]+p["depth"]+p["darkasp"]),
               1.10*p["ext"], 1.25*p["leg"], p["leg"], -p["dark"], -p.get("massif", 0.0),
               p.get("setting", 0.0), p["darkasp"]))
        print("  setting: gate %.2f | water %.2f | isthmus %.2f | cliffs %.2f | lakes %.2f | oddball %.2f" %
              (p.get("set_gate",0.0), p.get("set_water",0.0), p.get("set_isth",0.0),
               p.get("set_cliff",0.0), p.get("set_lake",0.0), p.get("set_odd",0.0)))
        fs = r.get("score")
        print("  total %.2f  (file score column: %s)" % (p["score"], "%.4f" % fs if fs is not None else "?"))
    if not found: print("seed not found")
    return 0

def cmd_rerank(files, opts):
    labels_path = opts.get("--labels")
    rows_all = load_rows(files)
    score_rows(rows_all)
    rows = ranked(rows_all)
    sc = sorted(r["_p"]["score"] for r in rows)
    def pct(q): return sc[min(len(sc)-1, max(0, int(q*(len(sc)-1))))]
    print("score distribution over %d unique seeds: p50 %.2f p90 %.2f p99 %.2f max %.2f" %
          (len(rows), pct(0.5), pct(0.9), pct(0.99), sc[-1]))
    print("\ntop 25:")
    for i, r in enumerate(rows[:25]):
        p = r["_p"]
        print("  %2d. %7.2f %s  (%.0f, %.0f)  amp %.1f tex %.1f frag %.1f dep %.1f ext %.1f" %
              (i+1, p["score"], r["_seed"], r.get("x",0), r.get("z",0),
               p["amp"], p["tex"], p["frag"], p["depth"], p["ext"]))
    if labels_path:
        lab = load_labels(labels_path)
        pos = [s for s,l in lab.items() if l >= 1]
        rank_of = {r["_seed"]: i+1 for i, r in enumerate(rows)}
        found = sorted(((rank_of[s], s) for s in pos if s in rank_of))
        missing = [s for s in pos if s not in rank_of]
        print("\nlabeled positives: %d found in file(s), %d absent (absent ones need --verify+merge)"
              % (len(found), len(missing)))
        if found:
            print("recall among found: top-20 %d/%d | top-50 %d/%d" %
                  (sum(1 for rk,_ in found if rk<=20), len(found),
                   sum(1 for rk,_ in found if rk<=50), len(found)))
            nc = len(rows)
            for rk, s in found:
                print("  rank %6d  (pct %8.4f)  %s%s" %
                      (rk, 100.0 * (nc - rk) / nc, s,
                       "  <-- BURIED (in-sample!)" if rk > 50 else ""))
    return 0

def fit_terms(train_seeds, lab, rows):
    terms = []
    for col, _, _, _ in LEGACY:
        vals = [g(r,col) for r in rows]; vals = [v for v in vals if v is not None]
        if len(vals) < 5: continue
        mu = statistics.fmean(vals); sd = statistics.pstdev(vals)
        if sd <= 0: continue
        v1 = [g(r,col) for r in rows if r["_seed"] in train_seeds and lab.get(r["_seed"],0) >= 1]
        v0 = [g(r,col) for r in rows if r["_seed"] in train_seeds and lab.get(r["_seed"],9) <= 0]
        v1 = [v for v in v1 if v is not None]; v0 = [v for v in v0 if v is not None]
        if len(v1) < 2 or len(v0) < 2: continue
        m1, m0 = statistics.fmean(v1), statistics.fmean(v0)
        s1 = statistics.stdev(v1); s0 = statistics.stdev(v0)
        sp = math.sqrt(((len(v1)-1)*s1*s1 + (len(v0)-1)*s0*s0) / max(len(v1)+len(v0)-2, 1))
        if sp <= 0: continue
        d = (m1 - m0) / sp
        if abs(d) >= 0.4: terms.append((col, mu, sd, d))
    return terms

def cmd_kfold(files, opts):
    labels_path = opts.get("--labels")
    if not labels_path: print("kfold needs --labels"); return 1
    k = int(opts.get("--k", 5))
    lab = load_labels(labels_path)
    rows_all = load_rows(files)
    score_rows(rows_all)
    rows0 = ranked(rows_all)
    labeled = [r for r in rows0 if r["_seed"] in lab]
    npos = sum(1 for r in labeled if lab[r["_seed"]] >= 1)
    labeled.sort(key=lambda r: int(r["_seed"]))
    folds = [set(r["_seed"] for i, r in enumerate(labeled) if i % k == f) for f in range(k)]
    print("kfold: %d labeled rows (%d positives), %d folds" % (len(labeled), npos, k))
    if len(labeled) < 10:
        print("not enough labeled rows in-corpus -- recover them with --verify (see instructions)")
    tot20 = tot50 = tott = 0
    for f in range(k):
        test = folds[f]
        train = set().union(*(folds[j] for j in range(k) if j != f)) if k > 1 else set()
        terms = fit_terms(train, lab, rows0)
        score_rows(rows_all, terms=terms)
        rank_of = {r["_seed"]: i+1 for i, r in enumerate(ranked(rows_all))}
        tp = [s for s in test if lab[s] >= 1 and s in rank_of]
        r20 = sum(1 for s in tp if rank_of[s] <= 20)
        r50 = sum(1 for s in tp if rank_of[s] <= 50)
        tot20 += r20; tot50 += r50; tott += len(tp)
        print("  fold %d: %d terms fit | %d test positives | recall@20 %d | recall@50 %d" %
              (f+1, len(terms), len(tp), r20, r50))
    print("  overall: recall@20 %.2f | recall@50 %.2f  (n=%d held-out positives)" %
          (tot20/max(1,tott), tot50/max(1,tott), tott))
    return 0

JIT_ASP = ["hMax","peakProm","bw1664hStd","bw1664Hi","bw1664Dh","bw896Dh","vcross",
           "highComps","highLargest","w896hMeanMin","hMin","bw1664Lo",
           "blobAreaBlocks2","coreRadiusCells","bw3200Sc","darkCoreFrac","bw1664Dark",
           "maxDark896Near","gorgeFrac","qualArea","mtnCompM","oceanDtHl","oceanSpiral","bw1664Ocn",
           "vicMush","vicWarmOc","scQ90",
           "lakeN","cliffFrac","vicSeaMin","vicOpenOc","vicOpenWarm","vicFjord","vicIsthmus",
           "vicBadl","vicBadlOth","vicMushB","vicFlower","vicCherry","vicMtg",
           "mtgCompM","chyCompM"]

def _score_flat(fv, av, C, terms):
    def zz(i, mu, sd):
        v = av[i]
        return None if v != v else (v - mu) / sd
    def zzr(i, mu, sd):
        v = av[i]
        return None if v != v else (mu - v) / sd
    amp  = _wm_flat([(zz(0,C["HMAX_MU"],C["HMAX_SD"]),0.30),
                     (zz(1,C["PROM_MU"],C["PROM_SD"]),0.25),
                     (zz(2,C["BWSTD_MU"],C["BWSTD_SD"]),0.25),
                     (zz(3,C["BWHI_MU"],C["BWHI_SD"]),0.20)])
    tex  = _wm_flat([(zz(4,C["BWDH_MU"],C["BWDH_SD"]),0.55),
                     (zz(5,C["BWDH_MU"],C["BWDH_SD"]),0.30),
                     (zz(6,C["VC_MU"],C["VC_SD"]),0.15)])
    hlz  = zz(8,C["HL_MU"],C["HL_SD"])
    frag = _wm_flat([(zz(7,C["HC_MU"],C["HC_SD"]),0.55),
                     (-hlz if hlz is not None else None,0.45)])
    depth= _wm_flat([(zzr(9,C["BOWL_MU"],C["BOWL_SD"]),0.45),
                     (zzr(10,C["HMIN_MU"],C["HMIN_SD"]),0.35),
                     (zz(11,C["BWLO_MU"],C["BWLO_SD"]),0.20),
                     (zz(18,C["GORGE_MU"],C["GORGE_SD"]),0.20)])
    dw, dr = av[16], av[15]  # bw1664Dark, darkCoreFrac (ring)
    z_win = None if dw != dw else (0.05 - dw) / 0.005
    z_ring = None if dr != dr else (0.25 - dr) / 0.05
    if z_win is None and z_ring is None:
        darkasp = 0.0
    else:
        darkasp = max(-4.0, min(1.5, min(x for x in (z_win, z_ring) if x is not None)))
    key = min(amp, tex, frag, depth, darkasp)
    a12, c13, b14 = av[12], av[13], av[14]
    qa, mc = av[19], av[20]
    za = None if a12 != a12 else max(-2.0, min(3.5, (a12/1e6-20)/4))
    zc = None if c13 != c13 else (c13-21)/4
    zq = None if qa != qa else max(-2.0, min(2.5, qa/1e6-2.0))
    zm = None if mc != mc else max(-2.0, min(2.5, mc/1e6-2.5))
    sq9 = av[26]
    zs9 = None if sq9 != sq9 else max(-2.0, min(2.5, (sq9 - -1.28639)/0.778071))
    ext = _wm_flat([(za,0.25),(zc,0.15),(b14 if b14 == b14 else None,0.35),(zq,0.15),(zs9,0.15),(zm,0.10)])
    leg = 0.0; wsum = 0.0
    for (col,mu,sd,w), v in zip(terms, fv):
        if v == v and sd > 0:
            zl = (v-mu)/sd
            if zl > 2.5: zl = 2.5
            elif zl < -2.5: zl = -2.5
            leg += w*zl; wsum += abs(w)
    if wsum > 0: leg /= wsum
    dw = av[16]; dw = 0.0 if dw != dw else dw
    darkpen = 40.0*max(0.0, dw-0.055)
    # setting mirror (flat): av[27..41] = the setting feature columns
    def fv(i):
        v = av[i]
        return v if v == v else 0.0   # NaN -> 0
    ods = [v for v in (av[21], av[22], av[29]) if v == v]
    od = min(ods) if ods else None
    if od is None: prox = 0.0
    elif od < 200: prox = 0.0
    elif od < 640: prox = (od-200)/440.0
    elif od <= 3000: prox = 1.0
    elif od < 4800: prox = (4800-od)/1800.0
    else: prox = 0.0
    if av[23] == av[23] and av[23] > 0.08: prox *= 0.25
    ooc, owm = fv(30), fv(31)
    op = 0.20 if ooc >= 2 else (0.10 if ooc >= 1 else 0.0)
    if owm >= 1: op += 0.05
    water = min(0.45*prox + 0.25*fv(32) + op, 0.55)
    isth = 0.15 if fv(33) >= 0.5 and water > 0 else 0.0
    cliffs = 0.15*min(1.0, fv(28)/0.25)
    lakes = min(0.12*fv(27), 0.30)
    odd = 0.0
    if fv(34) > 0: odd += min(0.15, 0.05 + 0.01*fv(34))
    if fv(35) >= 8: odd += 0.06
    if fv(36) > 0: odd += min(0.15, 0.08 + 0.01*fv(36))
    if fv(37) >= 4: odd += min(0.10, 0.04 + 0.01*fv(37))
    if fv(38) >= 4 or fv(41) >= 400000: odd += 0.10
    if fv(39) >= 6 or fv(40) >= 800000: odd += 0.10
    odd = min(odd, 0.30)
    gate = 1.0 if key >= 1.0 else (0.0 if key <= 0.5 else (key-0.5)*2.0)
    setting = min(water + isth + cliffs + lakes + odd, 0.80)*gate
    hlv = av[8]; hlv = 0.0 if hlv != hlv else hlv
    massif = 3.0*max(0.0, hlv-0.55)          # massif tax (mirror)
    return (2.0*key + 0.30*(amp+tex+frag+depth+darkasp)
            + 1.10*ext + 1.25*leg - darkpen - massif + setting)

def cmd_jitter(files, opts):
    seed = norm_seed(opts.get("--seed"))
    if seed is None: print("jitter needs --seed S"); return 1
    trials = int(opts.get("--trials", 120)); frac = float(opts.get("--frac", 0.30))
    rows_all = load_rows(files)
    score_rows(rows_all)
    base = ranked(rows_all)
    rank_of = {r["_seed"]: i+1 for i, r in enumerate(base)}
    if seed not in rank_of:
        print("seed not found"); return 1
    base_rank = rank_of[seed]
    tscore = max(r["_p"]["score"] for r in base if r["_seed"] == seed)
    MARGIN = 7.0
    pool = [r for r in base if r["_p"]["score"] >= tscore - MARGIN]
    base_top = set(r["_seed"] for r in pool[:20])
    print("jitter: target %s base score %.2f, rank %d/%d" % (seed, tscore, base_rank, len(base)))
    print("  pool: %d rows within %.1f points (the rest cannot catch it)" % (len(pool), MARGIN))
    cols = [c for c,_,_,_ in LEGACY]
    nan = float("nan")
    recs = []
    for r in pool:
        fv = [g(r,c) if g(r,c) is not None else nan for c in cols]
        av = [g(r,c) if g(r,c) is not None else nan for c in JIT_ASP]
        recs.append((r["_seed"], fv, av))
    rng = random.Random(7)
    ranks, overlaps = [], []
    for _ in range(trials):
        Cj = {k2: v*(1+rng.uniform(-frac,frac)) for k2,v in CONSTS.items()}
        tj = [(c, mu*(1+rng.uniform(-frac,frac)), max(1e-9, sd*(1+rng.uniform(-frac,frac))),
               w*(1+rng.uniform(-frac,frac))) for c,mu,sd,w in LEGACY]
        order = sorted(range(len(recs)), key=lambda i: -_score_flat(recs[i][1], recs[i][2], Cj, tj))
        pos = {recs[i][0]: n+1 for n, i in enumerate(order)}
        ranks.append(pos[seed])
        overlaps.append(len(set(recs[i][0] for i in order[:20]) & base_top) / 20.0)
    ranks.sort()
    print("  target rank over %d trials at +/-%.0f%%: min %d | median %d | p90 %d | max %d" %
          (trials, frac*100, ranks[0], ranks[len(ranks)//2], ranks[int(len(ranks)*0.9)], ranks[-1]))
    print("  stays top-20 in %.0f%% of trials | mean top-20 overlap %.2f" %
          (100*sum(1 for x in ranks if x<=20)/len(ranks), sum(overlaps)/len(overlaps)))
    return 0

def cmd_softmin(files, opts):
    k = float(opts.get("--k", 3.0))
    rows_all = load_rows(files)
    score_rows(rows_all)
    hard = ranked(rows_all)
    score_rows(rows_all, soft_k=k)
    soft = ranked(rows_all)
    old_rank = {r["_seed"]: i+1 for i, r in enumerate(hard)}
    print("softmin(k=%g): old rank -> new rank (top 25 of the soft ranking)" % k)
    for i, r in enumerate(soft[:25]):
        p = r["_p"]
        print("  %2d. %7.2f %s  (was #%d)  amp %.1f tex %.1f frag %.1f dep %.1f" %
              (i+1, p["score"], r["_seed"], old_rank.get(r["_seed"], -1),
               p["amp"], p["tex"], p["frag"], p["depth"]))
    return 0

def cmd_watchlist(files, opts):
    k = int(opts.get("--k", 8))
    rows_all = load_rows(files)
    score_rows(rows_all)
    rows = ranked(rows_all)
    srank = {r["_seed"]: i+1 for i, r in enumerate(rows)}
    keys = [("score","score"),("amp","amp"),("tex","tex"),("frag","frag"),
            ("depth","depth"),("extent","ext"),("area",None),("core",None)]
    lists = {}
    for name, kk in keys:
        if kk: lists[name] = sorted(rows, key=lambda r: -r["_p"][kk])[:k]
        elif name == "area": lists[name] = sorted(rows, key=lambda r: -(g(r,"blobAreaBlocks2") or -1))[:k]
        else: lists[name] = sorted(rows, key=lambda r: -(g(r,"coreRadiusCells") or -1))[:k]
    seen = {}
    for name, lst in lists.items():
        for r in lst: seen.setdefault(r["_seed"], {"row": r, "lists": []})["lists"].append(name)
    out = sorted(seen.values(), key=lambda e: -e["row"]["_p"]["score"])
    print("watchlist: %d unique seeds in the union of top-%d lists" % (len(out), k))
    print("rank  score  seed  x  z  lists")
    for e in out:
        r = e["row"]; sr = srank[r["_seed"]]
        flag = "  <== aspect-top but score-rank %d -- EYEBALL THIS ONE" % sr if sr > 40 else ""
        print("%4d %7.2f %s %.0f %.0f  %s%s" %
              (sr, r["_p"]["score"], r["_seed"], r.get("x",0), r.get("z",0),
               ",".join(e["lists"]), flag))
    return 0

# ==========================================================================
# Gate mining (score_lens) and GPU tier-2 calibration (calib_v6)
# ==========================================================================
GATEABLE = {
    "ridge"        : ("W_TEX_RIDGE_MIN",  "min", 0.70),
    "valley"       : ("W_TEX_VALLEY_MIN", "min", 0.70),
    "vcross"       : (None, "min", 1.0),
    "rcross"       : (None, "min", 1.0),
    "wAbs"         : (None, "min", 1.0),
    "hMax"         : ("H_GATE_MIN",       "min", 0.90),
    "hMean"        : (None, "min", 1.0),
    "hP90"         : (None, "min", 1.0),
    "hStd"         : (None, "min", 1.0),
    "dhMean"       : (None, "min", 1.0),
    "peakDensity"  : (None, "min", 1.0),
    "highComps"    : (None, "min", 1.0),
    "highLargest"  : (None, "max", 1.0),
    "darkCoreFrac" : (None, "max", 1.0),
    "bioDrk"       : (None, "max", 1.0),
    "bw1664Dh"     : (None, "min", 1.0),
    "bw1664hStd"   : (None, "min", 1.0),
    "bw1664Hi"     : (None, "min", 1.0),
    "bw3200Dh"     : (None, "min", 1.0),
    "eroCovFrac"   : (None, "min", 1.0),
    "eroMin"       : (None, "min", 1.0),
    "maxY"         : (None, "min", 1.0),
    "blobAreaBlocks2": (None, "min", 1.0),
}

def cmd_lens(files, opts):
    rows = load_rows(files)
    gem_at = float(opts.get("--gem", 8.0))
    tries = opts.get("--try", [])
    if isinstance(tries, str): tries = [tries]
    gems  = [r for r in rows if r.get("score") is not None and r["score"] >= gem_at]
    nogem = [r for r in rows if r.get("score") is not None and r["score"] <  gem_at]
    if not rows: print("no rows"); return 1
    s = sorted(r["score"] for r in rows)
    def pctv(v, q):
        return v[min(len(v)-1, max(0, int(q*(len(v)-1))))] if v else float("nan")
    print("rows %d | gems(score>=%.0f) %d (%.1f%%)" % (len(rows), gem_at, len(gems), 100*len(gems)/max(1,len(rows))))
    print("score pct: p50 %.2f p90 %.2f p99 %.2f max %.2f min %.2f" %
          (pctv(s,.5), pctv(s,.9), pctv(s,.99), s[-1], s[0]))
    def valid(rr, col):
        return [r[col] for r in rr if col in r and r[col] > -998]
    print("\n%-16s %4s %6s %8s %8s %6s %6s %6s  suggestion" %
          ("feature","dir","d","gemMean","restMean","recall","kill","t"))
    suggested = {}
    for col,(knob,direction,_) in GATEABLE.items():
        gv = sorted(valid(gems, col)); nv = valid(nogem, col)
        if len(gv) < 5 or not nv: continue
        mg = sum(gv)/len(gv); mr = sum(nv)/len(nv)
        sg = (sum((x-mg)**2 for x in gv)/max(1,len(gv)-1))**.5
        sr = (sum((x-mr)**2 for x in nv)/max(1,len(nv)-1))**.5
        sp = ((sg*sg+sr*sr)/2)**.5
        d = (mg-mr)/sp if sp > 0 else 0.0
        best = None
        for q in (0.0, 0.01, 0.02, 0.05):
            t = pctv(gv, q) if direction == "min" else pctv(gv, 1-q)
            keep = [x for x in gv if (x >= t if direction=="min" else x <= t)]
            kill = [x for x in nv if (x <  t if direction=="min" else x >  t)]
            rec = len(keep)/len(gv); kl = len(kill)/len(nv)
            if rec >= 0.98 and (best is None or kl > best[2]):
                best = (t, rec, kl)
            if best is None:
                t0 = pctv(gv, 0.0 if direction=="min" else 1.0)
                best = (t0, 1.0,
                        len([x for x in nv if (x < t0 if direction=="min" else x > t0)])/len(nv))
        t, rec, kl = best
        suggested[col] = (t, direction)
        print("%-16s %4s %+6.2f %8.4g %8.4g %6.2f %6.2f %6.4g  %s %s" %
              (col, ">=" if direction=="min" else "<=", d, mg, mr, rec, kl, t,
               knob or "-", "GATE" if knob else ""))
    if tries:
        keep = [r for r in rows if r.get("score") is not None]
        for expr in tries:
            op = ">=" if ">=" in expr else ("<=" if "<=" in expr else None)
            if op is None:
                print("bad gate expr: %s (need >= or <=)" % expr)
                continue
            col, val = expr.split(op, 1)
            col = col.strip()
            tv = float(val)
            keep = [r for r in keep if col in r and r[col] > -998 and
                    ((r[col] >= tv) if op == ">=" else (r[col] <= tv))]
        gk = [r for r in keep if r["score"] >= gem_at]
        print("\njoint %s: %d/%d candidates survive, gem recall %d/%d = %.3f" %
              (tries, len(keep), len(rows), len(gk), len(gems), len(gk)/max(1,len(gems))))
    print("\n// ---- paste-ready knobs (transfer factor applied) ----")
    for col,(knob,direction,xf) in GATEABLE.items():
        if knob and col in suggested:
            print("constexpr float %-18s = %.4ff; // corpus at %.4g" %
                  (knob, suggested[col][0]*xf, suggested[col][0]))
    return 0

def cmd_calib(files, opts):
    gem_at = float(opts.get("--gem", 8.0))
    rows = []
    for r in load_rows(files):
        try:
            rows.append((r["score"], r["dhMean"], r["gpuDh"], r["hMax"], r["gpuH"]))
        except KeyError:
            continue
    rows = [r for r in rows if 0 < r[2] < 900]
    if len(rows) < 20:
        print("need >= 20 rows with gpuDh measured (%d found)" % len(rows)); return 1
    gems = [r for r in rows if r[0] >= gem_at]
    rest = [r for r in rows if r[0] < gem_at]
    def q(v, p):
        v = sorted(v)
        return v[min(len(v)-1, max(0, int(p*(len(v)-1))))]
    def corr(a, b):
        ma = sum(a)/len(a); mb = sum(b)/len(b)
        sa = sum((x-ma)**2 for x in a)**.5
        sb = sum((y-mb)**2 for y in b)**.5
        return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(sa*sb) if sa*sb > 0 else 0.0
    print("rows with gpuDh: %d (gems %d, rest %d)" % (len(rows), len(gems), len(rest)))
    print("corr(gpuDh, dhMean) = %.3f   corr(gpuH, hMax) = %.3f" %
          (corr([r[2] for r in rows], [r[1] for r in rows]),
           corr([r[4] for r in rows], [r[3] for r in rows])))
    for name, i in (("gpuDh", 2), ("gpuH", 4)):
        gg = [r[i] for r in gems]; rr = [r[i] for r in rest]
        if not gg: continue
        print("%s: gems mean %.2f min %.2f p2 %.2f | rest mean %.2f med %.2f min %.2f" %
              (name, sum(gg)/len(gg), min(gg), q(gg,0.02), sum(rr)/len(rr), q(rr,0.5), min(rr)))
        for t in (min(gg)*0.95, q(gg,0.02)*0.95, q(gg,0.5)*0.8):
            kill = sum(1 for x in rr if x < t)/max(1,len(rr))
            rec = sum(1 for x in gg if x >= t)/max(1,len(gg))
            print("   gate %.2f : kills %.0f%% of rest, recalls %.1f%% of gems" % (t, 100*kill, 100*rec))
    print("\nPaste into gpu.cu:  BLOB2_DH_MIN / BLOB2_H_MIN_I  (choose ~95%% recall)")
    return 0

# ==========================================================================
# Recalibration (formerly recalibrate_score.py) + canary guard
# ==========================================================================
MIN_D = 0.4

FEATURES = [
    ("blobAreaBlocks2", "(double)area"),
    ("eroFmean",   "e.eroF_mean"), ("eroFp10",    "e.eroF_p10"),
    ("eroFmin",    "e.eroF_min"), ("contFmean",  "e.contF_mean"),
    ("contFmax",   "e.contF_max"), ("wAbs",       "e.w_abs"),
    ("ridge",      "e.ridge"), ("valley",     "e.valley"),
    ("rcross",     "e.rcross"), ("vcross",     "e.vcross"),
    ("hMin",       "e.h_min"), ("hMax",       "e.h_max"),
    ("hMean",      "e.h_mean"), ("hP90",       "e.h_p90"),
    ("hStd",       "e.h_std"), ("dhMean",     "e.dh_mean"),
    ("relief",     "e.relief"), ("high",       "e.high"),
    ("low",        "e.low"), ("peakDensity","e.peak_density"),
    ("peakProm",   "e.peak_prom"), ("highComps",  "e.high_comps"),
    ("highLargest","e.high_largest"), ("darkCoreFrac","e.dark_core_frac"),
    ("bioMtn",     "e.bio[CEN_MOUNTAIN]"), ("bioMtOg",    "e.bio[CEN_OG_TAIGA]"),
    ("bioTg",      "e.bio[CEN_TAIGA]"), ("bioMdw",     "e.bio[CEN_MEADOW]"),
    ("bioChy",     "e.bio[CEN_CHERRY]"), ("bioDrk",     "e.bio[CEN_DARK]"),
    ("bioPln",     "e.bio[CEN_PLAINS]"), ("bioRiv",     "e.bio[CEN_RIVER]"),
    ("bioOcn",     "e.bio[CEN_OCEAN]"),
    ("w896eroMin", "e.w896_ero_min"), ("w896hMax",   "e.w896_h_max"),
    ("w896mtnMax", "e.w896_mtn_max"), ("w1664eroMin","e.w1664_ero_min"),
    ("w1664hMax",  "e.w1664_h_max"), ("w1664mtnMax","e.w1664_mtn_max"),
    ("w896hStdMax",  "e.w896_hstd_max"), ("w896hMeanMin", "e.w896_hmean_min"),
    ("w1664hStdMax", "e.w1664_hstd_max"), ("w1664hMeanMin","e.w1664_hmean_min"),
    ("coreRadiusCells", "(double)core_cells"),
    ("bw896Sc",   "e.bw896_sc"), ("bw896Ero",  "e.bw896_ero"),
    ("bw896hMean","e.bw896_hmean"), ("bw896hStd", "e.bw896_hstd"),
    ("bw896Dh",   "e.bw896_dh"), ("bw896Mtn",  "e.bw896_mtn"),
    ("bw1664Sc",   "e.bw1664_sc"), ("bw1664Ero",  "e.bw1664_ero"),
    ("bw1664hMean","e.bw1664_hmean"), ("bw1664hStd", "e.bw1664_hstd"),
    ("bw1664Dh",   "e.bw1664_dh"), ("bw1664Mtn",  "e.bw1664_mtn"),
    ("bw3200Sc",   "e.bw3200_sc"), ("bw3200Ero",  "e.bw3200_ero"),
    ("bw3200hMean","e.bw3200_hmean"), ("bw3200hStd", "e.bw3200_hstd"),
    ("bw3200Dh",   "e.bw3200_dh"), ("bw3200Mtn",  "e.bw3200_mtn"),
    ("bw896Hi",  "e.bw896_hifrac"), ("bw896Lo",  "e.bw896_lofrac"),
    ("bw896Ocn", "e.bw896_ocn"),
    ("bw1664Hi",  "e.bw1664_hifrac"), ("bw1664Lo",  "e.bw1664_lofrac"),
    ("bw1664Ocn", "e.bw1664_ocn"),
    ("bw3200Hi",  "e.bw3200_hifrac"), ("bw3200Lo",  "e.bw3200_lofrac"),
    ("bw3200Ocn", "e.bw3200_ocn"), ("bw1664Dark","e.bw1664_dark"),
    ("maxDark896Near","e.max_dark_896_near"),
    ("oceanDtHl",  "e.ocean_dt_hl"), ("oceanSpiral","e.ocean_spiral"),
    ("qualN",      "e.qual_n"), ("qualArea",   "e.qual_area"),
    ("scQ90",      "e.sc_q90"), ("mtnCompM",   "e.mtn_comp_m"),
    ("mtnComps",   "e.mtn_comps"), ("gorgeFrac",  "e.gorge_frac"),
    ("lowCompM",   "e.low_comp_m"), ("lowComps",   "e.low_comps"),
    ("vicMush",    "e.vic_mush"),
    ("vicWarmOc",  "e.vic_warmoc"),
    ("lakeN",      "e.lake_n"),
    ("cliffFrac",  "e.cliff_frac"),
    ("vicSeaMin",  "e.vic_sea_min"),
    ("vicOpenOc",  "e.vic_open_oc"),
    ("vicOpenWarm","e.vic_open_warm"),
    ("vicFjord",   "e.vic_fjord"),
    ("vicIsthmus", "e.vic_isthmus"),
    ("vicBadl",    "e.vic_badl"),
    ("vicBadlOth", "e.vic_badl_oth"),
    ("vicMushB",   "e.vic_mush_b"),
    ("vicFlower",  "e.vic_flower"),
    ("vicCherry",  "e.vic_cherry"),
    ("vicMtg",     "e.vic_mtg"),
    ("mtgCompM",   "e.mtg_comp_m"),
    ("chyCompM",   "e.chy_comp_m"),
]

def newscore(r, terms):
    s = 0.0
    for t in terms:
        v = r.get(t["name"])
        if v is None or miss(v): continue
        s += t["d"] * (v - t["mu"]) / t["sd"]
    return s

def recal_fit(rows, labels, hard_min=None):
    pos = [r for r in rows if labels.get(r["_seed"], 0) >= 1]
    neg = [r for r in rows if r["_seed"] in labels and labels[r["_seed"]] <= 0
           and (hard_min is None or r.get("score", 0) >= hard_min)]
    terms = []
    for name, expr in FEATURES:
        vals = [r[name] for r in rows if name in r and not miss(r[name])]
        if len(vals) < 5: continue
        mu = statistics.fmean(vals); sd = statistics.pstdev(vals)
        if sd <= 0: continue
        v1 = [r[name] for r in pos if name in r and not miss(r[name])]
        v0 = [r[name] for r in neg if name in r and not miss(r[name])]
        if len(v1) < 2 or len(v0) < 2: continue
        m1, m0 = statistics.fmean(v1), statistics.fmean(v0)
        s1 = statistics.stdev(v1); s0 = statistics.stdev(v0)
        sp = math.sqrt(((len(v1)-1)*s1*s1 + (len(v0)-1)*s0*s0) / max(len(v1)+len(v0)-2, 1))
        if sp <= 0: continue
        d = (m1 - m0) / sp
        terms.append({"name": name, "expr": expr, "d": d, "mu": mu, "sd": sd,
                      "m1": m1, "m0": m0, "n1": len(v1), "n0": len(v0)})
    terms.sort(key=lambda t: -abs(t["d"]))
    return terms, pos, neg

def cmd_recal(files, opts):
    if not files:
        print("recal needs a corpus file"); return 1
    rows = dedupe_file(load_rows(files))
    if not rows:
        print("no rows parsed"); return 1

    # --todo N [--out todo.csv] : labeling worksheet (top scorers + random tail)
    if "--todo" in opts:
        n = int(opts["--todo"]); path = opts.get("--out", "todo.csv")
        srt = sorted(rows, key=lambda r: -(r.get("score") if r.get("score") is not None else -1e18))
        n_top = min(len(srt), (2 * n) // 3)
        sel = srt[:n_top]
        rest = srt[n_top:]
        k = min(n - n_top, len(rest))
        if k > 0:
            sel = sel + random.Random(20260705).sample(rest, k)
        random.Random(7).shuffle(sel)  # score order hidden while judging
        with open(path, "w") as f:
            f.write("seed,x,z,oldscore,label\n")
            for r in sel:
                f.write('%s,%d,%d,%.2f,\n' % (r["_seed"], int(r.get("x", 0)),
                                              int(r.get("z", 0)), r.get("score", float("nan"))))
        print("wrote %d rows to %s" % (len(sel), path))
        return 0

    labels_path = opts.get("--labels")
    if labels_path is None and len(files) > 1 and files[1].endswith(".csv"):
        labels_path = files[1]
    if labels_path is None:
        print("recal fit needs labels:  recal CORPUS --labels labels.csv"); return 1
    hard_min = float(opts["--hardneg"]) if "--hardneg" in opts else None
    labels = load_labels(labels_path)
    matched = sum(1 for s in labels if any(r["_seed"] == s for r in rows))
    terms, pos, neg = recal_fit(rows, labels, hard_min)
    print("rows: %d (deduped by seed) | labeled rows matched: %d (%d magic-like, %d other)"
          % (len(rows), matched, len(pos), len(neg)))
    if len(pos) < 5:
        print("\nWARNING: fewer than 5 positive labels -- weights are provisional.")

    print("\nseparation table (Cohen's d; positive = magic-like rows score higher):")
    print('  %-14s %7s %12s %12s %4s %4s' % ("feature", "d", "mean(magic)", "mean(other)", "n1", "n0"))
    for t in terms:
        print('  %-14s %+7.2f %12.4g %12.4g %4d %4d'
              % (t["name"], t["d"], t["m1"], t["m0"], t["n1"], t["n0"]))

    kept = [t for t in terms if abs(t["d"]) >= MIN_D]
    print("\n// ---- paste into compute_score() in probe.cpp (recalibrated) ----")
    print("    const Term terms[] = {")
    for t in kept:
        print('        { %s,  %.6g,  %.6g,  %+.2f }, // %s'
              % (t["expr"], t["mu"], t["sd"], t["d"], t["name"]))
    print("    };")
    print("\n(%d terms with |d| >= %g)" % (len(kept), MIN_D))

    lab = [(r, labels[r["_seed"]]) for r in rows if r["_seed"] in labels]
    lab.sort(key=lambda rl: -newscore(rl[0], kept))
    print("\nre-ranked labeled rows (new | old | label | seed | x z):")
    for r, l in lab[:40]:
        print('  %8.2f %8.2f  %d  %s  (%d, %d)'
              % (newscore(r, kept), r.get("score", float("nan")), l, r["_seed"],
                 int(r.get("x", 0)), int(r.get("z", 0))))

    # --canary S (repeatable): refuse to bless a Term table that buries a seed
    # you care about. Simulates the full compute_score with the proposed terms
    # (not just the raw term sum), since that's what actually ships.
    canaries = opts.get("--canary", [])
    if canaries and kept:
        new_terms = [(t["name"], t["mu"], t["sd"], t["d"]) for t in kept]
        score_rows(rows, terms=new_terms)
        new_rank = {r["_seed"]: i + 1 for i, r in enumerate(ranked(rows))}
        score_rows(rows)  # back to shipped LEGACY
        old_rank = {r["_seed"]: i + 1 for i, r in enumerate(ranked(rows))}
        ok = True
        print("\ncanary check (rank under shipped terms -> proposed terms):")
        for cs in canaries:
            cs = norm_seed(cs)
            if cs not in new_rank:
                print("  %s: not in corpus (skipped)" % cs); continue
            o, nn = old_rank.get(cs, 10 ** 9), new_rank[cs]
            bad = nn > max(50, 3 * o)
            ok = ok and not bad
            print("  %s: rank %d -> %d  %s" % (cs, o, nn, "OK" if not bad else "<<< BURIED"))
        print("canary verdict: %s" % ("PASS -- safe to paste (then update LEGACY here + `check`)"
              if ok else "WARN -- do not paste; gather more labels (positives especially) first"))
    return 0

# ==========================================================================
# Logistic regression head-to-head (formerly fit_logistic.py)
# ==========================================================================
def cmd_logistic(files, opts):
    labels_path = opts.get("--labels")
    if not labels_path:
        print("logistic needs --labels labels.csv"); return 1
    k = int(opts.get("--k", 5)); l2 = float(opts.get("--l2", 2.0))
    iters = int(opts.get("--iters", 2500))
    COLS = []
    for c, _, _, _ in LEGACY:
        if c not in COLS: COLS.append(c)
    EXTRA_COLS = [
        "qualN","qualArea","scQ90","mtnCompM","mtnComps","lowCompM","lowComps",
        "gorgeFrac","peakDensity","darkCoreFrac","maxDark896Near","oceanDtHl",
        "oceanSpiral","vicMush","vicWarmOc",
        "hMin","w896hMeanMin","w1664hMeanMin","vcross","rcross","wAbs",
        "ridge","valley","eroFmin","w896eroMin","w1664eroMin",
        "w896mtnMax","w1664mtnMax","w896hStdMax","w1664hStdMax",
        "bioMtOg","bioTg","bioMdw","bioChy","bioDrk","bioPln","bioOcn",
        "coreRadiusCells","blobAreaBlocks2","maxY",
    ]
    for c in EXTRA_COLS:
        if c not in COLS: COLS.append(c)

    def sigmoid(z):
        if z > 40: return 1.0
        if z < -40: return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    rows = load_rows(files)
    score_rows(rows)
    corpus = ranked(rows)
    labels_d = load_labels(labels_path)
    labeled = [r for r in corpus if r["_seed"] in labels_d]
    labeled.sort(key=lambda r: int(r["_seed"]))
    npos = sum(1 for r in labeled if labels_d[r["_seed"]] >= 1)
    print("logistic: %d corpus rows | %d labeled (%d positives) | %d features"
          % (len(corpus), len(labeled), npos, len(COLS)))
    if len(labeled) < 10:
        print("too few labeled rows to fit"); return 1

    stats = []
    for c in COLS:
        vs = [g(r, c) for r in corpus]
        vs = [v for v in vs if v is not None]
        mu = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / max(1, len(vs) - 1)) or 1.0
        stats.append((mu, sd))

    def feat(r):
        out = []
        for j, c in enumerate(COLS):
            v = g(r, c)
            mu, sd = stats[j]
            out.append(0.0 if v is None else (v - mu) / sd)
        return out

    Xl = [feat(r) for r in labeled]
    yl = [1.0 if labels_d[r["_seed"]] >= 1 else 0.0 for r in labeled]

    def train(idxs):
        d = len(COLS)
        w = [0.0] * d; b = 0.0
        n = len(idxs)
        for it in range(iters):
            lr = 0.2 / (1.0 + it / 1500.0)
            gw = [0.0] * d; gb = 0.0
            for t in idxs:
                xi, yi = Xl[t], yl[t]
                e = sigmoid(b + math.fsum(wj * xj for wj, xj in zip(w, xi))) - yi
                for j in range(d):
                    if xi[j]: gw[j] += e * xi[j]
                gb += e
            for j in range(d):
                w[j] -= lr * (gw[j] + l2 * w[j]) / n
            b -= lr * gb / n
        return w, b

    def ranks_under(w):
        scored = []
        for r in corpus:
            x = feat(r)
            scored.append((math.fsum(wj * xj for wj, xj in zip(w, x)), r["_seed"]))
        scored.sort(key=lambda t: -t[0])
        return {s: n + 1 for n, (_, s) in enumerate(scored)}

    Ks = (20, 50, 100, 500)
    agg = {K: 0 for K in Ks}; tot = 0
    for f in range(k):
        test = [t for t in range(len(labeled)) if t % k == f]
        w, b = train([t for t in range(len(labeled)) if t % k != f])
        rk = ranks_under(w)
        tp = [labeled[t]["_seed"] for t in test if yl[t] >= 1.0]
        tot += len(tp)
        line = "  fold %d: %2d held-out positives |" % (f + 1, len(tp))
        for K in Ks:
            rec = sum(1 for s in tp if rk[s] <= K)
            agg[K] += rec
            line += " @%d %d" % (K, rec)
        print(line)
    print("  logistic held-out overall: " +
          "  ".join("recall@%d %.2f" % (K, agg[K] / max(1, tot)) for K in Ks) +
          "  (n=%d)" % tot)
    print("  compare against `seedlab kfold` on the same file/labels/folds.")

    w, b = train(range(len(labeled)))
    order = sorted(range(len(COLS)), key=lambda j: -abs(w[j]))
    print("\n  full-data model (bias %+.3f); top |weights|:" % b)
    for j in order[:15]:
        print("    %+8.3f  %s" % (w[j], COLS[j]))
    print("\n  (porting into compute_score is ~20 lines if and only if the held-out")
    print("   numbers above clearly beat kfold's. Dot product, standardized inputs.)")
    return 0

# ==========================================================================
# Corpus stats (new) + audit workflow (new)
# ==========================================================================
def fmt_big(x):
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if x >= scale:
            return "%.2f%s" % (x / scale, unit)
    return "%d" % x

def cmd_stats(files, opts):
    rows = load_rows(files)
    if not rows:
        print("no rows"); return 1
    uniq = dedupe_file(rows)
    print("rows: %d | unique seeds: %d" % (len(rows), len(uniq)))
    sc = sorted(r["score"] for r in uniq if r.get("score") is not None and r["score"] > -998)
    if sc:
        n = len(sc)
        def q(p): return sc[min(n - 1, max(0, int(p * (n - 1))))]
        print("score: min %.2f | p01 %.2f p05 %.2f p25 %.2f | median %.2f | p75 %.2f p90 %.2f p99 %.2f | max %.2f"
              % (sc[0], q(.01), q(.05), q(.25), q(.5), q(.75), q(.9), q(.99), sc[-1]))
        neg = sum(1 for v in sc if v < 0)
        print("negative-score share: %d/%d = %.1f%%" % (neg, n, 100.0 * neg / n))
        for t in (0, 2, 4, 6, 8):
            c = sum(1 for v in sc if v >= t)
            print("  score >= %2d: %7d (%.3f%%)" % (t, c, 100.0 * c / n))
    areas = sorted(v for v in (g(r, "blobAreaBlocks2") for r in uniq) if v is not None)
    if areas:
        na = len(areas)
        print("blob area (blocks^2): min %s | median %s | max %s" %
              (fmt_big(areas[0]), fmt_big(areas[na // 2]), fmt_big(areas[-1])))
    edge = sum(1 for r in uniq if r.get("edgeClipped") is not None and r["edgeClipped"] >= 1)
    print("edge-clipped blobs: %d" % edge)
    xs = [int(r["x"]) for r in uniq if "x" in r]
    zs = [int(r["z"]) for r in uniq if "z" in r]
    if xs:
        print("headline x range [%d, %d] | z range [%d, %d]" % (min(xs), max(xs), min(zs), max(zs)))
    searched = float(opts.get("--searched", 0))
    if searched > 0 and sc:
        print("rates over %s searched seeds:" % fmt_big(searched))
        for t in (4, 6, 7, 8):
            c = sum(1 for v in sc if v >= t)
            if c:
                per = searched / c
                print("  score >= %d: ~1 per %s seeds (~%.1f h between finds at 20M seeds/s)"
                      % (t, fmt_big(per), per / 20e6 / 3600.0))
    return 0

def _parse_bands(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out

def _wilson(k2, n2, z=1.96):
    if n2 == 0: return 0.0, 0.0, 0.0
    p = k2 / n2
    d = 1 + z * z / n2
    c = (p + z * z / (2 * n2)) / d
    h = z * math.sqrt(p * (1 - p) / n2 + z * z / (4 * n2 * n2)) / d
    return p, c - h, c + h

def cmd_audit(files, opts):
    sub = opts.get("_sub")
    if sub == "sample":
        rows = dedupe_file(load_rows(files))
        rows = [r for r in rows if r.get("score") is not None and r["score"] > -998]
        rows.sort(key=lambda r: -r["score"])
        bands = _parse_bands(opts.get("--bands", "1-20,21-50,51-100,101-200,201-400"))
        n = int(opts.get("--n", 10))
        prefix = opts.get("--out", "audit")
        rng = random.Random(20260909)
        entries = []
        eid = 0
        for lo, hi in bands:
            pool = rows[lo - 1:hi]
            for r in rng.sample(pool, min(n, len(pool))):
                eid += 1
                entries.append((eid, r, "%d-%d" % (lo, hi)))
        with open(prefix + "_key.csv", "w") as f:  # SECRET: id -> band
            f.write("id,seed,band\n")
            for eid, r, band in entries:
                f.write("%d,%s,%s\n" % (eid, r["_seed"], band))
        shuffled = entries[:]
        rng.shuffle(shuffled)
        with open(prefix + "_labels.csv", "w") as f:  # fill this one blind
            f.write("id,seed,x,z,label\n")
            for eid, r, band in shuffled:
                f.write('%d,%s,%d,%d,\n' % (eid, r["_seed"], int(r.get("x", 0)), int(r.get("z", 0))))
        print("wrote %d blind rows: %s_labels.csv (label these) + %s_key.csv (keep secret)"
              % (len(entries), prefix, prefix))
        print("then:  seedlab.py audit tally --key %s_key.csv --labels %s_labels.csv" % (prefix, prefix))
        return 0
    if sub == "tally":
        keyp = opts.get("--key", "audit_key.csv")
        labp = opts.get("--labels", "audit_labels.csv")
        band_of = {}
        for line in open(keyp):
            if line.startswith("id,"): continue
            parts = line.strip().split(",")
            if len(parts) >= 3: band_of[parts[0]] = parts[2]
        per = {}
        for line in open(labp):
            line = line.split("#")[0].strip()
            if not line or line.startswith("id,"): continue
            parts = line.split(",")
            if len(parts) < 5 or parts[4] == "": continue
            band = band_of.get(parts[0])
            if band is None: continue
            try: lab = int(parts[4])
            except ValueError: continue
            k2, n2 = per.get(band, (0, 0))
            per[band] = (k2 + (1 if lab >= 1 else 0), n2 + 1)
        print("audit tally (gem = label >= 1; Wilson 95%% CI):")
        for band in sorted(per.keys(), key=lambda b: int(b.split("-")[0])):
            k2, n2 = per[band]
            p, lo, hi = _wilson(k2, n2)
            print("  ranks %-9s %2d/%2d = %5.1f%%   CI [%4.1f%%, %4.1f%%]" %
                  (band, k2, n2, 100 * p, 100 * lo, 100 * hi))
        print("\nreading: if bands past ~rank 50 sit near 0%%, trust the score and stop scrolling;")
        print("if a deep band still pays out, widen your nightly review to that rank.")
        return 0
    print("audit subcommands: sample, tally"); return 1

def cmd_positives(files, opts):
    if not files:
        print("positives needs a labels.csv"); return 1
    lab = load_labels(files[0])
    mn = int(opts.get("--min", 1))
    seeds = sorted(int(s) for s, l in lab.items() if l >= mn)
    for s in seeds:
        print(s)
    print("%d seeds with label >= %d" % (len(seeds), mn), file=sys.stderr)
    return 0

# ==========================================================================
# showtop / consolidate / cxx2py
# ==========================================================================
SHOWTOP_DEF_COLS = ["score","seed","x","z","blobAreaBlocks2","coreRadiusCells","hMax","hMin",
    "bw1664Dh","bw1664hStd","bw1664Hi","bw1664Lo","bw1664Ocn","bw1664Dark",
    "bw3200Sc","peakDensity","highLargest","high","low","vicMush","vicWarmOc"]

def cmd_showtop(files, opts):
    top = int(opts.get("--top", 25))
    cols = opts.get("--cols", "").split(",") if opts.get("--cols") else list(SHOWTOP_DEF_COLS)
    searched = float(opts.get("--searched", 0))
    rows = load_rows(files)
    def score_of(r):
        v = r.get("score")
        return v if v is not None else float("-inf")
    rows.sort(key=score_of, reverse=True)
    if searched > 0 and rows:
        n = len(rows)
        base = searched / n
        for idx, r in enumerate(rows):
            r["_rarity"] = "?" if score_of(r) == float("-inf") else "1:" + fmt_big(base / ((idx + 1) / n))
        cols = cols + ["_rarity"]
    INT_COLS = {"x","z","anchorX","anchorZ","blobAreaBlocks2","coreRadiusCells","maxY",
                "edgeClipped","gpuH","w896eroX","w896eroZ","bw896X","bw896Z","bw1664X","bw1664Z",
                "bw3200X","bw3200Z"}
    def cell(r, c):
        if c == "seed": return r.get("_seed", "?")
        if c == "_rarity": return r.get("_rarity", "?")
        v = r.get(c)
        if v is None: return "?"
        if c in INT_COLS: return str(int(v))
        return ("%.4f" % v).rstrip("0").rstrip(".")
    print(" ".join("rarity" if c == "_rarity" else c for c in cols))
    for r in rows[:top]:
        print(" ".join(cell(r, c) for c in cols))
    print("(%d rows parsed)" % len(rows), file=sys.stderr)
    return 0

def _is_int_tok(t):
    if t[:1] in ('+', '-'):
        t = t[1:]
    return t.isdigit() and len(t) > 0

def _parse_seed_line(line):
    toks = line.split('#', 1)[0].split()
    if not toks:
        return None
    v = None
    for i in range(len(toks) - 2):
        if _is_int_tok(toks[i]) and _is_int_tok(toks[i+1]) and _is_int_tok(toks[i+2]):
            v = int(toks[i]); break
    if v is None:
        for t in toks:
            if _is_int_tok(t):
                v = int(t); break
    if v is None:
        return None
    U64 = 1 << 64; I64 = 1 << 63
    if v < -I64 or v > U64 - 1:
        return None
    v %= U64
    if v >= I64:
        v -= U64
    return v

def cmd_consolidate(files, opts):
    out_path = opts.get("-o", "seed_atlas.txt")
    first_seen = bool(opts.get("--first-seen"))
    expanded = []
    for p in files:
        if os.path.isdir(p):
            expanded.extend(sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith('.txt')))
        else:
            expanded.append(p)
    out_abs = os.path.abspath(out_path)
    expanded = [f for f in expanded if os.path.isfile(f) and os.path.abspath(f) != out_abs]
    if not expanded:
        print('no input files found', file=sys.stderr); return 1
    seen = {}
    pre_existing = 0
    if os.path.isfile(out_path):
        with open(out_path, errors='replace') as fp:
            for line in fp:
                v = _parse_seed_line(line)
                if v is None: continue
                if v not in seen:
                    seen[v] = out_path
                    pre_existing += 1
    print('%-44s %9s %9s' % ('file', 'parsed', 'new'))
    if pre_existing:
        print('%-44s %9d %9s' % ('<existing atlas>', pre_existing, '-'))
    for path in expanded:
        parsed = new = 0
        with open(path, errors='replace') as fp:
            for line in fp:
                v = _parse_seed_line(line)
                if v is None: continue
                parsed += 1
                if v not in seen:
                    seen[v] = path
                    new += 1
        print('%-44s %9d %9d' % (os.path.basename(path), parsed, new))
    seeds = list(seen) if first_seen else sorted(seen)
    with open(out_path, 'w') as fp:
        for v in seeds:
            fp.write('%d\n' % v)
    added = len(seeds) - pre_existing
    neg = sum(1 for v in seeds if v < 0)
    print('\n%d unique seeds across %d files -> %s (%d pre-existing, %d newly added)'
          % (len(seeds), len(expanded), out_path, pre_existing, added))
    if seeds and not first_seen:
        print('signed-negative: %d (%.1f%%) | range [%d, %d]'
              % (neg, 100.0 * neg / len(seeds), seeds[0], seeds[-1]))
    print('ready for:  ./mountain_rangefinder --seeds %s ...' % out_path)
    return 0

def cmd_cxx2py(files, opts):
    for line in sys.stdin:
        m = re.search(r'\{\s*[^,]+,\s*([-\d.eE]+),\s*([-\d.eE]+),\s*([+-][\d.eE]+)\s*\},\s*//\s*(\S+)', line)
        if m:
            print('    ("%s",%s,%s,%s),' % (m.group(4), m.group(1), m.group(2), m.group(3)))
    return 0

# ==========================================================================
# Dispatch
# ==========================================================================
COMMANDS = {
    "stats": cmd_stats, "showtop": cmd_showtop, "watchlist": cmd_watchlist,
    "check": cmd_check, "explain": cmd_explain, "rerank": cmd_rerank,
    "kfold": cmd_kfold, "jitter": cmd_jitter, "softmin": cmd_softmin,
    "recal": cmd_recal, "logistic": cmd_logistic, "lens": cmd_lens,
    "calib": cmd_calib, "audit": cmd_audit, "positives": cmd_positives,
    "consolidate": cmd_consolidate, "cxx2py": cmd_cxx2py,
}

def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd = argv[0]
    argv = argv[1:]
    VALUE_OPTS = {"--seed","--labels","--k","--trials","--frac","--gem","--l2",
                  "--iters","--top","--cols","--searched","--min","--todo","--out",
                  "--bands","--n","--key","--hardneg","-o"}
    MULTI_OPTS = {"--try", "--canary"}
    files, opts = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in VALUE_OPTS:
            if i + 1 >= len(argv):
                print("missing value for %s" % a); return 1
            opts[a] = argv[i + 1]; i += 2
        elif a in MULTI_OPTS:
            if i + 1 >= len(argv):
                print("missing value for %s" % a); return 1
            opts.setdefault(a, []).append(argv[i + 1]); i += 2
        elif a == "--first-seen":
            opts[a] = True; i += 1
        else:
            files.append(a); i += 1
    if cmd == "audit" and files:
        opts["_sub"] = files[0]
        files = files[1:]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(__doc__); return 1
    return fn(files, opts)

if __name__ == "__main__":
    sys.exit(main())
