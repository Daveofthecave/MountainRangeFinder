#!/usr/bin/env python3
"""consolidate_seedlists.py -- squash every seed list into one deduped master file.

Handles every format any version of the pipeline (or Cubiomes Viewer) has
ever produced:
  - plain "one seed per line" lists
  - seed-first output rows      (seed x z ...)
  - score-first output rows     (score seed x z ...)
  - probe rows                  (seed x z [stats...])
  - '#' comments and blank lines
  - signed AND unsigned 64-bit spellings of the same seed (normalized signed)

The output file (default: cordillera.txt, or seed_atlas.txt via -o) is
treated as a persistent atlas: any seeds already present in it are kept
and never overwritten. New seeds found in the input files are appended
(in --first-seen order, or merged and sorted otherwise) and the combined
set is written back.

Usage:
  python3 consolidate_seedlists.py <dir-or-files...> [-o seed_atlas.txt] [--first-seen]

Example (put seed .txt files in the same directory and run):
  python3 consolidate_seedlists.py . -o seed_atlas.txt --first-seen
"""

import os
import sys

U64 = 1 << 64
I64 = 1 << 63

def is_int_tok(t):
    if t[:1] in ('+', '-'):
        t = t[1:]
    return t.isdigit() and len(t) > 0

def parse_seed(line):
    # strip comments (full-line and trailing) and blank lines
    toks = line.split('#', 1)[0].split()
    if not toks:
        return None
    # Prefer the first token of the first run of >= 3 consecutive integer
    # tokens ("seed x z ..." / "score seed x z ..."), matching how
    # --verify / --list-seeds locate seeds in full output rows.
    for i in range(len(toks) - 2):
        if is_int_tok(toks[i]) and is_int_tok(toks[i+1]) and is_int_tok(toks[i+2]):
            v = int(toks[i])
            break
    else:
        v = None
        for t in toks:                      # plain "one seed per line" fallback
            if is_int_tok(t):
                v = int(t)
                break
        if v is None:
            return None
    if v < -I64 or v > U64 - 1:             # not a plausible 64-bit seed
        return None
    v %= U64                                # wrap into unsigned 64-bit...
    if v >= I64:
        v -= U64                            # ...then normalize to signed
    return v

def main():
    args = sys.argv[1:]
    if not args or '-h' in args or '--help' in args:
        print(__doc__)
        return 0
    out_path = 'cordillera.txt'
    first_seen = False
    inputs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == '-o':
            out_path = args[i + 1]; i += 2
        elif a == '--first-seen':
            first_seen = True; i += 1
        else:
            inputs.append(a); i += 1

    files = []
    for p in inputs:
        if os.path.isdir(p):
            files.extend(sorted(os.path.join(p, f)
                                for f in os.listdir(p) if f.endswith('.txt')))
        else:
            files.append(p)

    # Never slurp a stale master copy of the output back into itself.
    out_abs = os.path.abspath(out_path)
    files = [f for f in files
             if os.path.isfile(f) and os.path.abspath(f) != out_abs]
    if not files:
        print('no input files found', file=sys.stderr)
        return 1

    seen = {}                               # seed -> first source

    # Pre-load any existing atlas entries so they are preserved (never
    # overwritten) when we write the combined result back out.
    pre_existing = 0
    if os.path.isfile(out_path):
        with open(out_path, errors='replace') as fp:
            for line in fp:
                v = parse_seed(line)
                if v is None:
                    continue
                if v not in seen:
                    seen[v] = out_path
                    pre_existing += 1

    print('%-44s %9s %9s' % ('file', 'parsed', 'new'))
    if pre_existing:
        print('%-44s %9d %9s' % ('<existing atlas>', pre_existing, '-'))
    for path in files:
        parsed = 0
        new = 0
        with open(path, errors='replace') as fp:
            for line in fp:
                v = parse_seed(line)
                if v is None:
                    continue
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
    print('\n%d unique seeds across %d files -> %s '
          '(%d pre-existing, %d newly added)'
          % (len(seeds), len(files), out_path, pre_existing, added))
    if seeds and not first_seen:
        print('signed-negative: %d (%.1f%%) | range [%d, %d]'
              % (neg, 100.0 * neg / len(seeds), seeds[0], seeds[-1]))
    print('ready for:  ./mountain_rangefinder --seeds %s ...' % out_path)
    print("or paste straight into Cubiomes Viewer's seed list.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
