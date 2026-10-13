#!/usr/bin/env bash
set -euo pipefail

for seed in 167 39 193; do
  python -m chs_match.train --dataset LastFM --seed "$seed" --skip-existing
done

for seed in 84 61 146; do
  python -m chs_match.train --dataset IMDB --seed "$seed" --skip-existing
done

for seed in 315 366 88; do
  python -m chs_match.train --dataset DBLP --seed "$seed" --skip-existing
done
