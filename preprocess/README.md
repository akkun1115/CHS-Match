# Benchmark construction from PyTorch Geometric sources

**The repository does not redistribute dataset or benchmark `.pt` files.**
This pipeline downloads the source graphs via PyTorch Geometric and builds
CHS-Match's node-disjoint training, validation, and test query–target pairs.

For the whole construction pipeline, run from the repository root:

```bash
bash scripts/build_benchmark.sh all
```

For one dataset:

```bash
bash scripts/build_benchmark.sh LastFM
```

**Important:** Partition search and multi-positive ground-truth construction
can be resource-intensive and take significant time. The benchmark generation
code follows the documented procedure but was **not fully rerun as part of
preparing this public repository**; byte-identical reproduction of the exact
paper benchmark tensors is not promised. Upstream download and use conditions
remain the user's responsibility; see [the dataset notice](../benchmark/DATA_NOTICE.md).

## Individual steps

Steps 1–4 must be completed in order. Steps 5a and 5b are optional,
independent branches that build receptive-field variants from 1-hop data.

### 1. Prepare source graphs

```bash
python preprocess/01_prepare_source_graphs.py --dataset all
```

Downloads PyG source datasets to local `raw/` and saves processed
source graphs under local `data/`.

### 2. Search for source-node-disjoint partitions

```bash
python preprocess/02_search_node_disjoint_partition.py --dataset LastFM
python preprocess/02_search_node_disjoint_partition.py --dataset IMDB
python preprocess/02_search_node_disjoint_partition.py --dataset DBLP
```

Saves `build/partitions/`. Train, validation, and test source-node sets are
disjoint.

### 3. Construct query–target pairs

```bash
python preprocess/03_generate_node_disjoint_pairs.py --dataset LastFM
python preprocess/03_generate_node_disjoint_pairs.py --dataset IMDB
python preprocess/03_generate_node_disjoint_pairs.py --dataset DBLP
```

Saves intermediate query–target pairs to `build/pairs/`.

### 4. Construct multi-positive ground-truth matching matrices

```bash
python preprocess/04_build_multi_positive_ground_truth.py --dataset all --split all --output-dir benchmark
```

Saves the 1-hop training/validation/test files to local `benchmark/*.pt`,
where `python -m chs_match.train` expects to read them by default. This
step may be computationally expensive.

### 5a–5b. Optional receptive-field variants

```bash
python preprocess/05a_build_2hop_benchmark.py --input-dir benchmark --output-dir benchmark/2hop
python preprocess/05b_build_3hop_benchmark.py --input-dir benchmark --output-dir benchmark/3hop
```

These take the locally generated 1-hop tensors, keep matching labels, and
build the respective hop-specific graph representations.

## Notes

- Generated data under `raw/`, `data/`, `build/` and `benchmark/*.pt` must
  **not** be committed to the repository.
- The wrapper `scripts/build_benchmark.sh` is the preferred entry point.
- Each individual script supports `--help` for options; existing output files
  may require an explicit `--overwrite` in scripts that support it.
