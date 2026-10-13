#!/usr/bin/env bash
# Download the original datasets via PyG and construct local CHS-Match benchmarks.
# No benchmark tensors are committed to this repository.
set -euo pipefail

dataset="${1:-all}"
case "$dataset" in
  LastFM|IMDB|DBLP) datasets=("$dataset") ;;
  all) datasets=(LastFM IMDB DBLP) ;;
  *)
    echo "Usage: bash scripts/build_benchmark.sh [LastFM|IMDB|DBLP|all]" >&2
    exit 2
    ;;
esac

echo "Preparing source datasets through PyTorch Geometric..."
python preprocess/01_prepare_source_graphs.py --dataset "$dataset"

for item in "${datasets[@]}"; do
  echo "Constructing node-disjoint benchmark for $item"
  python preprocess/02_search_node_disjoint_partition.py --dataset "$item"
  python preprocess/03_generate_node_disjoint_pairs.py --dataset "$item"
  python preprocess/04_build_multi_positive_ground_truth.py \
    --dataset "$item" --split all --output-dir benchmark
done

echo "Finished. Local benchmark files are under benchmark/*.pt (ignored by Git)."
echo "For 2-hop/3-hop variants, see preprocess/README.md."
