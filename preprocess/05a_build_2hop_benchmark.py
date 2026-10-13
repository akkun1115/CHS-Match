"""Step 5a: build the optional 2-hop receptive-field benchmark variant.

This transformation starts from the 1-hop benchmark, preserves the original
edges and matching matrices, and adds deterministic length-2 meta-path edge
channels. It is independent of Step 5b. By default, it reads build/benchmark/
and writes build/benchmark/2hop/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


DATASETS = ("LastFM", "IMDB", "DBLP")
SPLITS = ("train", "val", "test")


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unpack_dataset(obj):
    if (
        isinstance(obj, (list, tuple))
        and len(obj) == 3
        and isinstance(obj[0], (list, tuple))
        and isinstance(obj[1], (list, tuple))
        and isinstance(obj[2], (list, tuple))
    ):
        targets, queries, mms = obj
        if not (len(targets) == len(queries) == len(mms)):
            raise RuntimeError(
                "Dataset length mismatch: "
                f"targets={len(targets)}, queries={len(queries)}, mms={len(mms)}"
            )
        return list(targets), list(queries), list(mms)

    raise TypeError(
        "Expected benchmark format [targets, queries, matching_matrices]."
    )


def edge_type_to_string(edge_type: Tuple[str, str, str]) -> str:
    return f"{edge_type[0]}::{edge_type[1]}::{edge_type[2]}"


def add_two_hop_edges(
    data: HeteroData,
) -> Tuple[HeteroData, dict]:
    """Add 2-hop meta-path edges using the benchmark construction semantics.

    Deterministic ordering used to construct the released benchmark:
      edge_type1 -> edge_type2 -> edge i in edge_type1 -> matching edge j in edge_type2

    This implementation preserves that ordering and multiplicity, but uses an
    index from second-hop source node -> edge positions instead of repeatedly
    scanning the whole edge list.
    """
    new_data = data.clone()
    original_edge_types = list(data.edge_types)

    added_edge_count = 0
    added_edge_types = []
    added_self_edges = 0

    for edge_type1 in original_edge_types:
        src_type, rel1, mid_type = edge_type1
        edge_index1 = data[edge_type1].edge_index.detach().cpu()

        src1 = edge_index1[0].tolist()
        dst1 = edge_index1[1].tolist()

        for edge_type2 in original_edge_types:
            mid_type2, rel2, dst_type = edge_type2

            if mid_type != mid_type2:
                continue

            edge_index2 = data[edge_type2].edge_index.detach().cpu()
            src2 = edge_index2[0].tolist()
            dst2 = edge_index2[1].tolist()

            # Preserve original edge2 order within each intermediate node.
            by_second_hop_source: Dict[int, List[int]] = defaultdict(list)
            for j, node_id in enumerate(src2):
                by_second_hop_source[int(node_id)].append(j)

            out_src: List[int] = []
            out_dst: List[int] = []

            for i, intermediate in enumerate(dst1):
                for j in by_second_hop_source.get(int(intermediate), ()):
                    s = int(src1[i])
                    d = int(dst2[j])
                    out_src.append(s)
                    out_dst.append(d)
                    if src_type == dst_type and s == d:
                        added_self_edges += 1

            if not out_src:
                continue

            two_hop_edge_index = torch.tensor(
                [out_src, out_dst],
                dtype=torch.long,
            )

            new_edge_type = (
                src_type,
                f"{rel1}_{mid_type}_{rel2}",
                dst_type,
            )

            # The benchmark construction uses assignment here. If an exotic relation-name
            # collision occurs, later construction overwrites earlier one.
            new_data[new_edge_type].edge_index = two_hop_edge_index

            added_edge_count += int(two_hop_edge_index.size(1))
            added_edge_types.append(new_edge_type)

    return new_data, {
        "added_two_hop_edges": added_edge_count,
        "added_two_hop_edge_type_occurrences": len(added_edge_types),
        "added_two_hop_self_edges": added_self_edges,
        "added_two_hop_edge_types": [
            edge_type_to_string(x)
            for x in added_edge_types
        ],
    }


def tensors_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    return a == b


def validate_original_graph_preserved(
    source: HeteroData,
    transformed: HeteroData,
):
    if tuple(source.node_types) != tuple(transformed.node_types):
        raise RuntimeError("Node types changed during 2-hop preprocessing.")

    for ntype in source.node_types:
        src_store = source[ntype]
        out_store = transformed[ntype]

        if set(src_store.keys()) != set(out_store.keys()):
            raise RuntimeError(
                f"Node attributes changed for {ntype}: "
                f"{set(src_store.keys())} vs {set(out_store.keys())}"
            )

        for key in src_store.keys():
            if not tensors_equal(src_store[key], out_store[key]):
                raise RuntimeError(
                    f"Node attribute changed: {ntype}.{key}"
                )

    for etype in source.edge_types:
        if etype not in transformed.edge_types:
            raise RuntimeError(
                f"Original edge type disappeared: {etype}"
            )

        src_store = source[etype]
        out_store = transformed[etype]

        for key in src_store.keys():
            if key not in out_store:
                raise RuntimeError(
                    f"Original edge attribute disappeared: {etype}.{key}"
                )
            if not tensors_equal(src_store[key], out_store[key]):
                raise RuntimeError(
                    f"Original edge attribute changed: {etype}.{key}"
                )


def validate_mm_unchanged(source_matching_matrices, output_matching_matrices):
    if len(source_matching_matrices) != len(output_matching_matrices):
        raise RuntimeError("Matching-matrix list length changed.")

    for pair_idx, (source_mm, output_mm) in enumerate(zip(source_matching_matrices, output_matching_matrices)):
        if type(source_mm) is not type(output_mm):
            raise RuntimeError(
                f"MM type changed at pair {pair_idx}: "
                f"{type(source_mm)} vs {type(output_mm)}"
            )

        if isinstance(source_mm, dict):
            if set(source_mm.keys()) != set(output_mm.keys()):
                raise RuntimeError(
                    f"MM node-type keys changed at pair {pair_idx}"
                )
            for ntype in source_mm:
                if not torch.equal(
                    source_mm[ntype].detach().cpu(),
                    output_mm[ntype].detach().cpu(),
                ):
                    raise RuntimeError(
                        f"MM changed at pair {pair_idx}, node type {ntype}"
                    )
        elif isinstance(source_mm, torch.Tensor):
            if not torch.equal(
                source_mm.detach().cpu(),
                output_mm.detach().cpu(),
            ):
                raise RuntimeError(
                    f"MM changed at pair {pair_idx}"
                )
        else:
            if source_mm != output_mm:
                raise RuntimeError(
                    f"MM changed at pair {pair_idx}"
                )


def collect_edge_schema(graphs: Iterable[HeteroData]) -> List[str]:
    schema = set()
    for graph in graphs:
        for etype in graph.edge_types:
            schema.add(edge_type_to_string(etype))
    return sorted(schema)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def copy_ground_truth_sidecars(
    input_dir: Path,
    output_dir: Path,
    dataset: str,
    split: str,
):
    stem = f"{dataset.lower()}_{split}"
    copied = []

    for suffix in (
        "_summary.json",
        "_summary.txt",
        "_pair_details.csv",
    ):
        src = input_dir / f"{stem}{suffix}"
        dst = output_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(str(dst))

    return copied


def process_one(
    input_dir: Path,
    output_dir: Path,
    dataset: str,
    split: str,
    overwrite: bool,
):
    input_path = input_dir / f"{dataset.lower()}_{split}.pt"
    output_path = output_dir / f"{dataset.lower()}_{split}.pt"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --overwrite only if you intentionally want to rebuild it."
        )

    obj = torch_load(input_path)
    targets, queries, matching_matrices = unpack_dataset(obj)

    start = time.perf_counter()

    new_targets = []
    new_queries = []

    target_added_edges = 0
    query_added_edges = 0
    target_added_self_edges = 0
    query_added_self_edges = 0

    pbar = tqdm(
        range(len(targets)),
        desc=f"{dataset}/{split} 2hop",
        unit="pair",
        dynamic_ncols=True,
    )

    for i in pbar:
        new_target, t_stat = add_two_hop_edges(targets[i])
        new_query, q_stat = add_two_hop_edges(queries[i])

        # Full per-pair preservation checks.
        validate_original_graph_preserved(targets[i], new_target)
        validate_original_graph_preserved(queries[i], new_query)

        new_targets.append(new_target)
        new_queries.append(new_query)

        target_added_edges += t_stat["added_two_hop_edges"]
        query_added_edges += q_stat["added_two_hop_edges"]
        target_added_self_edges += t_stat["added_two_hop_self_edges"]
        query_added_self_edges += q_stat["added_two_hop_self_edges"]

        pbar.set_postfix(
            t2hop=target_added_edges,
            q2hop=query_added_edges,
        )

    # Ground-truth matching matrices are intentionally reused, not recomputed.
    output_matching_matrices = matching_matrices
    validate_mm_unchanged(matching_matrices, output_matching_matrices)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        [new_targets, new_queries, output_matching_matrices],
        output_path,
    )

    copied_sidecars = copy_ground_truth_sidecars(
        input_dir,
        output_dir,
        dataset,
        split,
    )

    elapsed = time.perf_counter() - start

    summary = {
        "dataset": dataset,
        "split": split,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_file": str(input_path),
        "output_file": str(output_path),
        "pairs": len(targets),
        "hop": 2,
        "keep_original_1hop_edges": True,
        "keep_duplicate_two_hop_edges": True,
        "keep_two_hop_self_return_edges": True,
        "save_intermediates_attribute": False,
        "matching_matrices_reused_without_modification": True,
        "target_added_two_hop_edges": target_added_edges,
        "query_added_two_hop_edges": query_added_edges,
        "target_added_two_hop_self_edges": target_added_self_edges,
        "query_added_two_hop_self_edges": query_added_self_edges,
        "target_edge_schema": collect_edge_schema(new_targets),
        "query_edge_schema": collect_edge_schema(new_queries),
        "copied_ground_truth_sidecars": copied_sidecars,
        "elapsed_sec": elapsed,
    }

    per_split_summary = (
        output_dir / f"{dataset.lower()}_{split}_2hop_preprocess.json"
    )
    atomic_json_save(summary, per_split_summary)

    print()
    print("=" * 100)
    print(f"{dataset} / {split} DONE")
    print("input :", input_path)
    print("output:", output_path)
    print("pairs :", len(targets))
    print("target added 2hop edges:", f"{target_added_edges:,}")
    print("query  added 2hop edges:", f"{query_added_edges:,}")
    print("elapsed:", f"{elapsed:.2f}s")
    print("summary:", per_split_summary)
    print("=" * 100)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build the deterministic 2-hop benchmark variant.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="all",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("build/benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/benchmark/2hop"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    splits = SPLITS if args.split == "all" else (args.split,)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for dataset in datasets:
        for split in splits:
            summary = process_one(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                dataset=dataset,
                split=split,
                overwrite=args.overwrite,
            )
            all_summaries.append(summary)

            # Refresh the global manifest after every completed split.
            atomic_json_save(
                {
                    "created_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "source_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "hop": 2,
                    "semantics": {
                        "original_1hop_edges_preserved": True,
                        "duplicate_two_hop_edges_preserved": True,
                        "two_hop_self_return_edges_preserved": True,
                        "ground_truth_reused_without_recomputation": True,
                        "intermediate_path_attributes_saved": False,
                    },
                    "completed": all_summaries,
                },
                args.output_dir / "preprocess_summary.json",
            )

    print()
    print("ALL REQUESTED 2-HOP PREPROCESSING COMPLETED")
    print("manifest:", args.output_dir / "preprocess_summary.json")


if __name__ == "__main__":
    main()
