# Benchmark data provenance and reuse notice

## Source datasets

The CHS-Match benchmark-construction scripts use heterogeneous graph datasets
obtained locally using the PyTorch Geometric (PyG) dataset loaders.
**Neither the original datasets nor their derived benchmark tensors are
redistributed in this repository.** The loaders and associated datasets are documented in PyG v2.5.0:

| Dataset | PyG loader | PyG documentation |
| --- | --- | --- |
| LastFM | `torch_geometric.datasets.LastFM` | [LastFM](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.LastFM.html) |
| IMDB | `torch_geometric.datasets.IMDB` | [IMDB](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.IMDB.html) |
| DBLP | `torch_geometric.datasets.DBLP` | [DBLP](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.DBLP.html) |

PyG identifies all three source datasets as collected for the following work:

> Xinyu Fu, Jiani Zhang, Ziqiao Meng, and Irwin King.
> *MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous
> Graph Embedding.* The Web Conference (WWW), 2020.
> https://arxiv.org/abs/2002.01680

Original MAGNN repository and dataset download references:
https://github.com/cynricfu/MAGNN

The CHS-Match project builds node-disjoint query/target graph pairs and
correspondence labels from those source graphs. When generated locally, the `*.pt` files contain derived benchmark instances
rather than unchanged copies of the PyG downloads.
Construction scripts and documentation are under `preprocess/`.

## License and permissions

The repository's MIT License applies to the original CHS-Match source code
and scripts, **not** to the third-party source datasets or locally derived
benchmark files. The PyG library's software license should not be interpreted as a
separate grant of rights to redistribute dataset content.

The PyG documentation and MAGNN dataset links provide provenance, but do not
by themselves establish unrestricted redistribution rights. Dataset files are
therefore not included in the repository.
Users should check the applicable upstream dataset terms and obtain
permissions where required before redistributing dataset content.

No additional rights to third-party data are granted by this repository.
