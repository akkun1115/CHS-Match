# Local node-disjoint benchmark

**No LastFM, IMDB or DBLP dataset tensors are distributed in this repository.**
The data and generated benchmark files should be obtained/constructed locally using
the [preprocessing pipeline](../preprocess/README.md). From the repository root:

```bash
bash scripts/build_benchmark.sh all
```

This will download the source data using PyTorch Geometric and create
`benchmark/lastfm_{train,val,test}.pt`,
`benchmark/imdb_{train,val,test}.pt`, and
`benchmark/dblp_{train,val,test}.pt` locally. They are ignored by Git.

Each generated `*.pt` file stores:

```text
[targets, queries, matching_matrices]
```

- `targets`: list of PyG `HeteroData` target graphs
- `queries`: list of PyG `HeteroData` query graphs
- `matching_matrices`: list of dictionaries keyed by node type; each tensor is a multi-positive query-to-target correspondence matrix

For reference, the **original manuscript experiments** used these split sizes:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| LastFM | 1,023 | 219 | 219 |
| IMDB | 836 | 179 | 179 |
| DBLP | 861 | 184 | 185 |

The construction scripts implement the intended generation procedure, but exact
identity with the original benchmark tensors is **not guaranteed** and has not
been established by a full regeneration check.

For optional 2-hop / 3-hop variants, first generate the 1-hop benchmark, then:

```bash
python preprocess/05a_build_2hop_benchmark.py --input-dir benchmark --output-dir benchmark/2hop
python preprocess/05b_build_3hop_benchmark.py --input-dir benchmark --output-dir benchmark/3hop
```

## Source provenance and permissions

See [DATA_NOTICE.md](DATA_NOTICE.md). The source data is obtained through
PyTorch Geometric, not shipped with this code. Any locally generated files
are outside the code's MIT license and must be used in accordance with the
applicable source-data terms.
