#!/usr/bin/env bash
set -euo pipefail

run_dataset () {
  local dataset="$1"; shift
  local seeds=("$@")

  for seed in "${seeds[@]}"; do
    python -m chs_match.train --dataset "$dataset" --seed "$seed" --skip-existing
    python -m chs_match.train --dataset "$dataset" --seed "$seed" --skip-existing --normalization row_softmax
  done
}

run_dataset LastFM 167 39 193
run_dataset IMDB   84 61 146
run_dataset DBLP   315 366 88
