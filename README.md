# CHS-Match

Official research artifact for **CHS-Match: Coupled Representation Learning for Heterogeneous Subgraph Matching**.

This repository is prepared to accompany the revised manuscript and contains:

- the CHS-Match training/evaluation implementation;
- scripts to construct node-disjoint LastFM, IMDB, and DBLP benchmark instances **locally** from PyTorch Geometric source datasets (the benchmark data files are not redistributed here);
- benchmark-construction and multi-positive ground-truth scripts;
- deterministic 2-hop/3-hop meta-path preprocessing scripts;
- shell scripts for the main CHS-Match runs and the row-softmax vs. partial-Sinkhorn controlled comparison.

## Reproducibility scope and AI-assisted repository preparation

This repository focuses on reproducing **CHS-Match**, including its model, training/evaluation procedure, and the node-disjoint benchmark used for the main comparison. The third-party baseline implementations are **not** distributed here.

**Disclosure:** AI-assisted programming tools were used to reorganize and simplify the research code for release. This is a refactored publication package, **not** a verbatim copy of the scripts used during the experiments: development-only options were removed, code was split into modules, and the command-line interface was simplified. The authors are responsible for the released code. No full experimental GPU rerun was performed as part of preparing this repository, and we do **not** claim numerical identity with the original research script. Differences may arise from implementation details, dependency versions, or nondeterministic operations. The published accuracy values have not been independently regenerated from this release package; please report any discrepancies. The provided benchmark-generation scripts and paper settings are intended to facilitate independent checks.

We recommend comparing the three-run mean and standard deviation with Table 3 rather than expecting bit-for-bit identical floating-point outputs across different software and hardware configurations. The timings reported by the simple training entry point are wall-clock timings and are **not directly comparable** with the CUDA-event GPU-only measurements in the paper's Table 4.

## Repository layout

```text
CHS-Match/
├── chs_match/
│   ├── model.py          # CHS-Match model and matching loss
│   ├── train.py          # compact training/evaluation entry point
│   └── data.py           # benchmark loader and packed batching
├── benchmark/            # dataset provenance and format docs; generate .pt files locally
├── preprocess/           # numbered benchmark-construction pipeline
├── scripts/
│   ├── build_benchmark.sh   # construct data locally from PyG
│   ├── run_main.sh
│   └── run_normalization_comparison.sh
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

The publication code intentionally omits development-only alternatives that are not needed to reproduce the paper. The adopted sum semantic reduction, soft semantic alignment, log-parameterized temperature, packed matching implementation, embedding input projection, LayerNorm, and reported dropout settings are fixed in `chs_match/model.py`.

## Environment

Install PyTorch for your CUDA/CPU environment first, then:

```bash
pip install -r requirements.txt
```

Training is intended for an NVIDIA GPU. The original datasets are downloaded locally by PyTorch Geometric; generated benchmark files are not part of the GitHub repository.

For reference, the GitHub Actions **CPU smoke tests** use Python 3.11, PyTorch 2.5.1 (CPU), and PyTorch Geometric 2.6.1. They exercise the model with locally generated, small synthetic test cases. No third-party benchmark files are needed for the tests. This establishes basic package functionality on that environment, **not** numerical reproduction of the full GPU experiments. The PyTorch/PyG versions used for the original paper runs have not been established by this check.

## Generate the benchmarks locally

The repository **does not include or redistribute** the original or derived LastFM, IMDB, or DBLP benchmark tensors. To construct the node-disjoint benchmark on your computer, install the dependencies and run:

```bash
bash scripts/build_benchmark.sh all
```

To generate only one dataset, use e.g. `bash scripts/build_benchmark.sh LastFM`. This downloads source datasets through PyTorch Geometric, then constructs node-disjoint partitions, query/target pairs, and multi-positive matching matrices. **This preparation can take substantial CPU time and storage**, especially partition search and exact correspondence construction. The generated `benchmark/*.pt` files stay on your computer and are ignored by Git.

The generator follows the documented research procedure but may not yield byte-identical tensors to those used in the paper, depending on source versions and environment. **No claim of exact recreation of the manuscript's particular benchmark instances is made.** See [construction instructions](preprocess/README.md), [format](benchmark/README.md), and [provenance](benchmark/DATA_NOTICE.md). Review applicable upstream data terms before using or redistributing datasets.

## Main experiments

The paper uses 1-hop data, hidden dimension 64, four heads, Adam with learning rate `1e-3`, batch size 32, 500 epochs, `lambda_mm=0.2`, and partial Sinkhorn with `K=5`.

| Dataset | Layers | Seeds |
| --- | ---: | --- |
| LastFM | 2 | 167, 39, 193 |
| IMDB | 3 | 84, 61, 146 |
| DBLP | 5 | 315, 366, 88 |

Run all nine main experiments:

```bash
bash scripts/run_main.sh
```

A single main run is intentionally short:

```bash
python -m chs_match.train --dataset LastFM --seed 167
```

The formal settings above are defaults, including the dataset-specific layer depth.

After all nine runs, summarize the results and compare them with the three CHS-Match entries in the paper's Table 3:

```bash
python scripts/summarize_results.py --results-dir results --require-complete
```

For a quick functional check **without claiming reproduction of the paper's accuracy**, install the optional development dependencies and run:

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

Each training run now writes its results to a configuration-specific JSON filename and refuses to overwrite an existing result. Use a new `--output-dir` (or explicitly move/remove the previous result) to rerun exactly the same configuration.

## Reported analyses

Only settings needed by the paper or response analyses are exposed:

```bash
# Different layer depth / receptive-field experiment
python -m chs_match.train --dataset IMDB --seed 84 --layers 4

# 2-hop data generated under benchmark/2hop
python -m chs_match.train --dataset IMDB --seed 84 --hop 2

# Matching-loss sensitivity
python -m chs_match.train --dataset DBLP --seed 315 --lambda-mm 0.5

# Sinkhorn-iteration sensitivity
python -m chs_match.train --dataset LastFM --seed 167 --sinkhorn-iters 3

# Ablations reported in the paper
python -m chs_match.train --dataset LastFM --seed 167 --ablation no_cgna
python -m chs_match.train --dataset LastFM --seed 167 --ablation no_casa
python -m chs_match.train --dataset LastFM --seed 167 --ablation no_both
```

Row-wise softmax is retained only because it is the comparison method reported in the controlled normalization experiment:

```bash
bash scripts/run_normalization_comparison.sh
```

or for one run:

```bash
python -m chs_match.train --dataset LastFM --seed 167 --normalization row_softmax
```

All results are written as JSON under `results/`.

## Local benchmark files and preprocessing

After you run `bash scripts/build_benchmark.sh all`, the local `benchmark/*.pt` files contain train/validation/test query–target instances with multi-positive correspondence labels. These files are intentionally excluded from version control.

The preprocessing scripts are:

```text
preprocess/01_prepare_source_graphs.py
preprocess/02_search_node_disjoint_partition.py
preprocess/03_generate_node_disjoint_pairs.py
preprocess/04_build_multi_positive_ground_truth.py
preprocess/05a_build_2hop_benchmark.py
preprocess/05b_build_3hop_benchmark.py
```

See [preprocess/README.md](preprocess/README.md) for the full step-by-step pipeline. To generate the optional receptive-field variants **after** the 1-hop files exist:

```bash
python preprocess/05a_build_2hop_benchmark.py --input-dir benchmark --output-dir benchmark/2hop
python preprocess/05b_build_3hop_benchmark.py --input-dir benchmark --output-dir benchmark/3hop
```

## Dataset provenance

The benchmark-construction procedure uses the LastFM, IMDB, and DBLP heterogeneous graph datasets obtained through PyTorch Geometric (`torch_geometric.datasets.LastFM`, `IMDB`, and `DBLP`). Derived benchmark files, if generated locally, are not covered by the repository's MIT code license. See [`benchmark/DATA_NOTICE.md`](benchmark/DATA_NOTICE.md) for details.

## Data availability

Repository URL:

```text
https://github.com/akkun1115/CHS-Match
```

## License

The original CHS-Match source code and scripts are released under the MIT License; see [`LICENSE`](LICENSE).

Generated benchmark data files are **not covered by the MIT License** because they are derived from third-party source datasets and are not redistributed in this repository. Use of those files remains subject to the applicable terms of the underlying datasets. See [`benchmark/DATA_NOTICE.md`](benchmark/DATA_NOTICE.md).
