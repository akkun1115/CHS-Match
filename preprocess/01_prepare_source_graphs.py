"""Step 1: prepare the source heterogeneous graphs.

The source datasets are loaded through PyTorch Geometric and converted to the
node-feature representation used by the CHS-Match benchmark. By default, the
prepared graphs are written under data/.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch_geometric.datasets as pyg_datasets


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_scalar_categories(data, low: int = 0, high: int = 9) -> None:
    for node_type in data.node_types:
        x = torch.empty((data[node_type].num_nodes, 1), dtype=torch.float32)
        for i in range(data[node_type].num_nodes):
            x[i, 0] = random.randint(low, high)
        data[node_type].x = x


def prepare_lastfm(raw_root: Path):
    seed_all(42)
    data = pyg_datasets.LastFM(root=str(raw_root / "LastFM"))[0]
    random_scalar_categories(data)
    return data


def prepare_imdb(raw_root: Path):
    seed_all(42)
    data = pyg_datasets.IMDB(root=str(raw_root / "IMDB"))[0]
    random_scalar_categories(data)
    return data


def prepare_dblp(raw_root: Path):
    seed_all(42)
    data = pyg_datasets.DBLP(root=str(raw_root / "DBLP"))[0]

    # Paper categories are the conference IDs used in the benchmark preparation.
    conference_edge_type = ("paper", "to", "conference")
    data["paper"].x = (
        data[conference_edge_type].edge_index[1]
        .unsqueeze(-1)
        .to(dtype=torch.float32)
    )

    # Author and term categories use the same seeded categorical assignment
    # as the benchmark preparation.
    for node_type in ("author", "term"):
        x = torch.empty((data[node_type].num_nodes, 1), dtype=torch.float32)
        for i in range(data[node_type].num_nodes):
            x[i, 0] = random.randint(0, 9)
        data[node_type].x = x

    # Conference identity has already been transferred to paper.x.
    del data["conference"]
    del data[("conference", "to", "paper")]
    del data[("paper", "to", "conference")]
    return data


PREPARERS = {
    "LastFM": prepare_lastfm,
    "IMDB": prepare_imdb,
    "DBLP": prepare_dblp,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PyG source graphs for benchmark construction.")
    parser.add_argument(
        "--dataset",
        choices=["LastFM", "IMDB", "DBLP", "all"],
        default="all",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = PREPARERS if args.dataset == "all" else (args.dataset,)

    for name in datasets:
        out = args.output_dir / f"{name.lower()}_origin.pt"
        if out.exists() and not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. Pass --overwrite to replace it."
            )
        graph = PREPARERS[name](args.raw_root)
        torch.save(graph, out)
        print(f"saved {out}: {graph}")


if __name__ == "__main__":
    main()
