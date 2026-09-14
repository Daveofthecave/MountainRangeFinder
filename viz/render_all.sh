#!/bin/sh
# viz/render_all.sh -- render every explainer GIF (fields cache in viz/cache/).
set -e
cd "$(dirname "$0")/.."          # repo root
for s in viz/make_g*.py; do
    echo "== $s"
    python3 "$s"
done
echo "done -- GIFs in viz/out/"
