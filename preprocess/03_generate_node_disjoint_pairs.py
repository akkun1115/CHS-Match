"""Step 3: generate query-target pairs from a fixed source partition.

This script uses the source graphs prepared in Step 1 and the fixed partition
created in Step 2. It does not search or modify the partition.

By default, generated train/validation/test pairs are written under build/pairs/
using the benchmark tuple format [targets, queries, matching_matrices].
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import tqdm
from torch_geometric.data import HeteroData


CONFIG = {
    "LastFM": {
        "origin_file": "lastfm_origin.pt",
        "target_down": 50,
        "target_over": 100,
        "target_walks": 100,
        "target_walk_length": 5,
        "restart_prob": 0.2,
        "total_pair_count": 1461,
    },
    "IMDB": {
        "origin_file": "imdb_origin.pt",
        "target_down": 60,
        "target_over": 110,
        "target_walks": 200,
        "target_walk_length": 5,
        "restart_prob": 0.8,
        "total_pair_count": 1194,
    },
    "DBLP": {
        "origin_file": "dblp_origin.pt",
        "target_down": 70,
        "target_over": 110,
        "target_walks": 100,
        "target_walk_length": 5,
        "restart_prob": 0.3,
        "total_pair_count": 1230,
    },
}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def partition_type_metric(partition):
    """Return the type-balance metric written by Step 2."""
    return {
        "name": "max_type_prop_deviation",
        "value": partition.get("max_type_prop_deviation"),
    }


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_total_nodes(graph: HeteroData):
    return sum(
        graph[nt].num_nodes
        for nt in graph.node_types
    )


def induced_source_subgraph(
    origin: HeteroData,
    selected_ids: Dict[str, torch.Tensor],
):
    graph = HeteroData()

    subset = {
        nt: selected_ids.get(
            nt,
            torch.tensor(
                [],
                dtype=torch.long,
            ),
        )
        for nt in origin.node_types
    }

    for nt in origin.node_types:
        ids = subset[nt]

        if ids.numel() == 0:
            continue

        graph[nt].x = origin[nt].x[ids]
        graph[nt].orig_id = ids.clone()

    for etype in origin.edge_types:
        src, _, tgt = etype

        if (
            subset[src].numel() == 0
            or subset[tgt].numel() == 0
        ):
            continue

        e = origin[etype].edge_index

        src_map = torch.full(
            (origin[src].num_nodes,),
            -1,
            dtype=torch.long,
        )
        src_map[subset[src]] = torch.arange(
            subset[src].numel()
        )

        tgt_map = torch.full(
            (origin[tgt].num_nodes,),
            -1,
            dtype=torch.long,
        )
        tgt_map[subset[tgt]] = torch.arange(
            subset[tgt].numel()
        )

        ns = src_map[e[0]]
        nd = tgt_map[e[1]]

        valid = (
            (ns >= 0)
            & (nd >= 0)
        )

        if valid.any():
            graph[etype].edge_index = torch.stack(
                [
                    ns[valid],
                    nd[valid],
                ],
                dim=0,
            )

    return graph


def sample_induced_rw_subgraph(
    graph: HeteroData,
    start_node_type: str,
    num_nodes_s: int,
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    node_idx: Optional[int] = None,
):
    if node_idx is None:
        node_idx = random.randint(
            0,
            graph[start_node_type].num_nodes - 1,
        )

    subset = {
        nt: torch.tensor(
            [],
            dtype=torch.long,
        )
        for nt in graph.node_types
    }

    subset[start_node_type] = torch.tensor(
        [node_idx],
        dtype=torch.long,
    )

    total_nodes_collected = 1
    use_fixed_start = True

    for _ in range(num_walks):
        if (
            use_fixed_start
            or random.random() < restart_prob
        ):
            current_type = start_node_type
            current_node = node_idx
            use_fixed_start = False
        else:
            available = [
                (nt, int(x))
                for nt in subset
                for x in subset[nt].tolist()
            ]

            if not available:
                break

            current_type, current_node = (
                random.choice(available)
            )

        for _ in range(walk_length):
            neighbors = []

            for etype in graph.edge_types:
                src, _, tgt = etype
                e = graph[etype].edge_index

                if current_type == src:
                    vals = e[
                        1,
                        e[0] == current_node,
                    ].tolist()

                    neighbors += [
                        (tgt, int(x))
                        for x in vals
                    ]

                elif current_type == tgt:
                    vals = e[
                        0,
                        e[1] == current_node,
                    ].tolist()

                    neighbors += [
                        (src, int(x))
                        for x in vals
                    ]

            if not neighbors:
                break

            next_type, current_node = (
                random.choice(neighbors)
            )

            subset[next_type] = torch.cat(
                [
                    subset[next_type],
                    torch.tensor(
                        [current_node],
                        dtype=torch.long,
                    ),
                ]
            ).unique()

            current_type = next_type

            total_nodes_collected = sum(
                x.numel()
                for x in subset.values()
            )

            if (
                total_nodes_collected
                >= num_nodes_s
            ):
                break

        if (
            total_nodes_collected
            >= num_nodes_s
        ):
            break

    subgraph = HeteroData()

    mapping = {
        nt: None
        for nt in graph.node_types
    }

    for nt in graph.node_types:
        ids = subset[nt]

        if ids.numel() > 0:
            subgraph[nt].x = graph[nt].x[ids]
            mapping[nt] = ids

            if "orig_id" in graph[nt]:
                subgraph[nt].orig_id = (
                    graph[nt].orig_id[ids]
                    .detach()
                    .cpu()
                    .clone()
                )

    for etype in graph.edge_types:
        src, _, tgt = etype

        if (
            subset[src].numel() == 0
            or subset[tgt].numel() == 0
        ):
            continue

        e = graph[etype].edge_index

        src_map = torch.full(
            (graph[src].num_nodes,),
            -1,
            dtype=torch.long,
        )
        src_map[subset[src]] = torch.arange(
            subset[src].numel()
        )

        tgt_map = torch.full(
            (graph[tgt].num_nodes,),
            -1,
            dtype=torch.long,
        )
        tgt_map[subset[tgt]] = torch.arange(
            subset[tgt].numel()
        )

        ns = src_map[e[0]]
        nd = tgt_map[e[1]]

        valid = (
            (ns >= 0)
            & (nd >= 0)
        )

        if valid.any():
            subgraph[etype].edge_index = torch.stack(
                [
                    ns[valid],
                    nd[valid],
                ],
                dim=0,
            )

    return subgraph, mapping


def make_mm(
    target,
    query,
    q_to_t,
):
    mm = {}

    for nt in query.node_types:
        mat = torch.zeros(
            query[nt].num_nodes,
            target[nt].num_nodes,
            dtype=torch.float32,
        )

        for qid, tid in enumerate(
            q_to_t[nt].tolist()
        ):
            mat[qid, int(tid)] = 1.0

        mm[nt] = mat

    return mm


def target_signature(graph):
    return tuple(
        (
            nt,
            tuple(
                sorted(
                    int(x)
                    for x in graph[nt]
                    .orig_id
                    .tolist()
                )
            ),
        )
        for nt in graph.node_types
    )


def allocate_counts(
    total,
    train_frac,
    val_frac,
    test_frac,
):
    n_train = round(
        total * train_frac
    )
    n_val = round(
        total * val_frac
    )
    n_test = (
        total
        - n_train
        - n_val
    )

    return {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }


def generate_pairs(
    source_split,
    cfg,
    desired,
    seed,
    query_ratio,
    query_walks,
    query_walk_length,
    max_attempt_multiplier,
):
    seed_all(seed)

    targets = []
    queries = []
    mms = []

    seen_targets = set()

    attempts = 0
    max_attempts = max(
        desired * max_attempt_multiplier,
        1000,
    )

    pbar = tqdm.tqdm(
        total=desired,
        desc=f"Generate {desired} pairs",
    )

    while (
        len(targets) < desired
        and attempts < max_attempts
    ):
        attempts += 1

        target_size = random.randint(
            cfg["target_down"],
            cfg["target_over"],
        )

        start_type = random.choice(
            source_split.node_types
        )

        target, t_map = (
            sample_induced_rw_subgraph(
                source_split,
                start_type,
                target_size,
                cfg["target_walks"],
                cfg["target_walk_length"],
                cfg["restart_prob"],
            )
        )

        n_t = count_total_nodes(
            target
        )

        if not (
            cfg["target_down"]
            <= n_t
            <= cfg["target_over"]
        ):
            continue

        if (
            set(target.node_types)
            != set(source_split.node_types)
        ):
            continue

        if any(
            value is None
            for value in t_map.values()
        ):
            continue

        sig = target_signature(
            target
        )

        if sig in seen_targets:
            continue

        q_target = max(
            1,
            int(query_ratio * n_t),
        )

        q_start = random.choice(
            target.node_types
        )

        query, q_to_t = (
            sample_induced_rw_subgraph(
                target,
                q_start,
                q_target,
                query_walks,
                query_walk_length,
                cfg["restart_prob"],
            )
        )

        if any(
            value is None
            for value in q_to_t.values()
        ):
            continue

        n_q = count_total_nodes(
            query
        )

        if not (
            0.20 * n_t
            <= n_q
            <= 0.60 * n_t
        ):
            continue

        if (
            set(query.node_types)
            != set(target.node_types)
        ):
            continue

        mm = make_mm(
            target,
            query,
            q_to_t,
        )

        seen_targets.add(sig)

        targets.append(target)
        queries.append(query)
        mms.append(mm)

        pbar.update(1)

    pbar.close()

    if len(targets) < desired:
        raise RuntimeError(
            f"Generated only "
            f"{len(targets)}/{desired} "
            f"pairs after {attempts} attempts"
        )

    return [
        targets,
        queries,
        mms,
    ], attempts


def source_nodes(data):
    result = set()

    targets, queries, _ = data

    for collection in [
        targets,
        queries,
    ]:
        for graph in collection:
            for nt in graph.node_types:
                result.update(
                    (
                        nt,
                        int(x),
                    )
                    for x in graph[nt]
                    .orig_id
                    .tolist()
                )

    return result


def run(
    root,
    output_dir,
    dataset_name,
    args,
):
    cfg = CONFIG[
        dataset_name
    ]

    origin = torch_load(
        root
        / "data"
        / cfg["origin_file"]
    )

    if args.partition_path is not None:
        partition_path = args.partition_path
        if not partition_path.is_absolute():
            partition_path = (root / partition_path).resolve()
    else:
        partition_path = (
            root
            / "build"
            / "partitions"
            / f"{dataset_name.lower()}_partition.pt"
        )

    if not partition_path.exists():
        raise FileNotFoundError(
            f"Partition file not found: {partition_path}"
        )

    partition = torch_load(
        partition_path
    )

    if partition.get("dataset") not in (None, dataset_name):
        raise RuntimeError(
            f"Partition dataset mismatch: "
            f"requested={dataset_name}, "
            f"partition={partition.get('dataset')}"
        )

    node_ids = partition[
        "node_ids"
    ]

    source_graphs = {
        split: induced_source_subgraph(
            origin,
            node_ids[split],
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    }

    pair_counts = allocate_counts(
        cfg["total_pair_count"],
        partition["split_fractions"]["train"],
        partition["split_fractions"]["val"],
        partition["split_fractions"]["test"],
    )

    if args.train_pairs is not None:
        pair_counts["train"] = (
            args.train_pairs
        )
    if args.val_pairs is not None:
        pair_counts["val"] = (
            args.val_pairs
        )
    if args.test_pairs is not None:
        pair_counts["test"] = (
            args.test_pairs
        )

    print("=" * 90)
    print(dataset_name)
    print("=" * 90)
    print(
        "Fixed partition:",
        partition_path,
    )
    print(
        "edge-cut ratio:",
        partition[
            "edge_cut_ratio"
        ],
    )
    type_metric = partition_type_metric(partition)

    print(
        f"{type_metric['name']}:",
        type_metric["value"],
    )
    print(
        "pair counts:",
        pair_counts,
    )

    structural_metrics = partition.get("structural_metrics")
    if structural_metrics is not None:
        print("structural metrics:")
        for split in ["train", "val", "test"]:
            if split in structural_metrics:
                m = structural_metrics[split]
                print(
                    f"  {split}: "
                    f"LCC={m.get('lcc_fraction')}, "
                    f"nonisolated={m.get('nonisolated_fraction')}, "
                    f"internal={m.get('internal_neighbor_fraction')}, "
                    f"full_usable={m.get('full_range_usable_node_fraction')}"
                )

    datasets = {}
    attempts = {}

    split_seed_offset = {
        "train": 0,
        "val": 100000,
        "test": 200000,
    }

    for split in [
        "train",
        "val",
        "test",
    ]:
        print()
        print(
            f"Generate {split}"
        )

        data, n_attempts = (
            generate_pairs(
                source_graphs[split],
                cfg,
                pair_counts[split],
                args.generation_seed
                + split_seed_offset[split],
                args.query_ratio,
                args.query_walks,
                args.query_walk_length,
                args.max_attempt_multiplier,
            )
        )

        datasets[split] = data
        attempts[split] = n_attempts

    # Strict source-node overlap check.
    sets = {
        split: source_nodes(
            datasets[split]
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    }

    overlap = {
        "train_val":
            len(
                sets["train"]
                & sets["val"]
            ),
        "train_test":
            len(
                sets["train"]
                & sets["test"]
            ),
        "val_test":
            len(
                sets["val"]
                & sets["test"]
            ),
    }

    assert all(
        x == 0
        for x in overlap.values()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        path = (
            output_dir
            / f"{dataset_name.lower()}_{split}.pt"
        )

        torch.save(
            datasets[split],
            path,
        )

        saved[split] = str(path)

    report = {
        "dataset":
            dataset_name,
        "partition_file":
            str(partition_path),
        "partition": {
            "trial":
                partition["trial"],
            "seed":
                partition["seed"],
            "edge_cut_ratio":
                partition["edge_cut_ratio"],
            "selection_method":
                partition.get("selection_method"),
            "type_metric_name":
                type_metric["name"],
            "type_metric_value":
                type_metric["value"],
            "max_allowed_type_prop_deviation":
                partition.get("max_allowed_type_prop_deviation"),
            "split_fractions":
                partition[
                    "split_fractions"
                ],
        },
        "target_sampler": {
            "size_min":
                cfg["target_down"],
            "size_max":
                cfg["target_over"],
            "num_walks":
                cfg["target_walks"],
            "walk_length":
                cfg["target_walk_length"],
            "restart_prob":
                cfg["restart_prob"],
        },
        "query_sampler": {
            "query_ratio":
                args.query_ratio,
            "num_walks":
                args.query_walks,
            "walk_length":
                args.query_walk_length,
            "restart_prob":
                cfg["restart_prob"],
            "accepted_query_target_ratio":
                [0.20, 0.60],
            "role":
                "node-disjoint benchmark construction setting",
        },
        "pair_counts":
            pair_counts,
        "generation_attempts":
            attempts,
        "source_node_overlap":
            overlap,
        "files":
            saved,
    }

    report_path = (
        output_dir
        / f"{dataset_name.lower()}_node_disjoint_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "STRICT NODE-DISJOINT:",
        overlap,
    )
    print(
        "Report:",
        report_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate node-disjoint query-target pairs from a fixed partition.")

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "LastFM",
            "IMDB",
            "DBLP",
        ],
        required=True,
    )

    parser.add_argument(
        "--partition-path",
        type=Path,
        default=None,
        help=(
            "Explicit partition .pt file. "
            "Relative paths are resolved from --project-root. "
            "If omitted, build/partitions/<dataset>_partition.pt is used."
        ),
    )

    parser.add_argument(
        "--generation-seed",
        type=int,
        default=424242,
    )

    parser.add_argument(
        "--query-ratio",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--query-walks",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--query-walk-length",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--train-pairs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--val-pairs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--test-pairs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.project_root.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "build" / "pairs"
    )

    run(
        root,
        output_dir,
        args.dataset,
        args,
    )


if __name__ == "__main__":
    main()
