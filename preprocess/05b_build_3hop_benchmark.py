"""Step 5b: build the optional 3-hop receptive-field benchmark variant.

This transformation starts directly from the 1-hop benchmark, preserves the
original edges and matching matrices, and adds deterministic length-2 and
length-3 meta-path edge channels. It does not depend on Step 5a. By default,
it reads build/benchmark/ and writes build/benchmark/3hop/.
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


def build_source_index(edge_index: torch.Tensor) -> Dict[int, List[int]]:
    """Map source-node ID -> edge positions while preserving original order."""
    source = edge_index[0].tolist()
    index: Dict[int, List[int]] = defaultdict(list)
    for pos, node_id in enumerate(source):
        index[int(node_id)].append(pos)
    return index


def add_multihop_edges(
    data: HeteroData,
) -> Tuple[HeteroData, dict]:
    """Add 2-hop and 3-hop meta-path edges using the benchmark construction semantics.

    Deterministic loop order for 2-hop:
        edge_type1 -> edge_type2 -> edge i -> matching edge j

    Deterministic loop order for 3-hop:
        edge_type1 -> edge_type2 -> edge_type3
        -> edge i -> matching edge j -> matching edge k

    Multiplicity and order are preserved. The only optimization is indexing
    second/third-hop edges by their source node rather than rescanning the
    entire edge list for every path prefix.
    """
    new_data = data.clone()
    original_edge_types = list(data.edge_types)

    # Cache CPU edge tensors and source-node indexes once per original relation.
    edge_cache = {}
    source_index_cache = {}

    for etype in original_edge_types:
        ei = data[etype].edge_index.detach().cpu()
        edge_cache[etype] = ei
        source_index_cache[etype] = build_source_index(ei)

    two_hop_generated_edges = 0
    three_hop_generated_edges = 0
    two_hop_self_edges = 0
    three_hop_self_edges = 0

    two_hop_relation_occurrences = 0
    three_hop_relation_occurrences = 0
    two_hop_relation_collisions = 0
    three_hop_relation_collisions = 0

    two_hop_types_seen = set()
    three_hop_types_seen = set()

    for edge_type1 in original_edge_types:
        src_type1, rel1, mid_type1 = edge_type1
        edge_index1 = edge_cache[edge_type1]

        src1 = edge_index1[0].tolist()
        dst1 = edge_index1[1].tolist()

        for edge_type2 in original_edge_types:
            mid_type2, rel2, mid_type3 = edge_type2

            if mid_type1 != mid_type2:
                continue

            edge_index2 = edge_cache[edge_type2]
            dst2 = edge_index2[1].tolist()
            by_second_source = source_index_cache[edge_type2]

            # -------------------------------------------------------------
            # 2-hop construction
            # -------------------------------------------------------------
            out2_src: List[int] = []
            out2_dst: List[int] = []

            for i, intermediate1 in enumerate(dst1):
                for j in by_second_source.get(int(intermediate1), ()):
                    s = int(src1[i])
                    d = int(dst2[j])
                    out2_src.append(s)
                    out2_dst.append(d)

                    if src_type1 == mid_type3 and s == d:
                        two_hop_self_edges += 1

            if out2_src:
                two_hop_edge_index = torch.tensor(
                    [out2_src, out2_dst],
                    dtype=torch.long,
                )

                two_hop_type = (
                    src_type1,
                    f"{rel1}_{mid_type1}_{rel2}",
                    mid_type3,
                )

                # The benchmark construction uses assignment here; preserve that behavior.
                if two_hop_type in two_hop_types_seen:
                    two_hop_relation_collisions += 1
                two_hop_types_seen.add(two_hop_type)

                new_data[two_hop_type].edge_index = two_hop_edge_index

                two_hop_generated_edges += int(two_hop_edge_index.size(1))
                two_hop_relation_occurrences += 1

            # -------------------------------------------------------------
            # 3-hop construction
            # -------------------------------------------------------------
            for edge_type3 in original_edge_types:
                mid_type4, rel3, dst_type = edge_type3

                if mid_type3 != mid_type4:
                    continue

                edge_index3 = edge_cache[edge_type3]
                dst3 = edge_index3[1].tolist()
                by_third_source = source_index_cache[edge_type3]

                out3_src: List[int] = []
                out3_dst: List[int] = []

                for i, intermediate1 in enumerate(dst1):
                    for j in by_second_source.get(int(intermediate1), ()):
                        intermediate2 = int(dst2[j])

                        for k in by_third_source.get(intermediate2, ()):
                            s = int(src1[i])
                            d = int(dst3[k])
                            out3_src.append(s)
                            out3_dst.append(d)

                            if src_type1 == dst_type and s == d:
                                three_hop_self_edges += 1

                if not out3_src:
                    continue

                three_hop_edge_index = torch.tensor(
                    [out3_src, out3_dst],
                    dtype=torch.long,
                )

                three_hop_type = (
                    src_type1,
                    f"{rel1}_{mid_type1}_{rel2}_{mid_type3}_{rel3}",
                    dst_type,
                )

                # The benchmark construction also uses assignment here.
                if three_hop_type in three_hop_types_seen:
                    three_hop_relation_collisions += 1
                three_hop_types_seen.add(three_hop_type)

                new_data[three_hop_type].edge_index = three_hop_edge_index

                three_hop_generated_edges += int(three_hop_edge_index.size(1))
                three_hop_relation_occurrences += 1

    return new_data, {
        "generated_two_hop_edges": two_hop_generated_edges,
        "generated_three_hop_edges": three_hop_generated_edges,
        "generated_two_hop_self_edges": two_hop_self_edges,
        "generated_three_hop_self_edges": three_hop_self_edges,
        "two_hop_relation_occurrences": two_hop_relation_occurrences,
        "three_hop_relation_occurrences": three_hop_relation_occurrences,
        "two_hop_relation_collisions": two_hop_relation_collisions,
        "three_hop_relation_collisions": three_hop_relation_collisions,
        "two_hop_edge_types": sorted(
            edge_type_to_string(x)
            for x in two_hop_types_seen
        ),
        "three_hop_edge_types": sorted(
            edge_type_to_string(x)
            for x in three_hop_types_seen
        ),
    }


def tensors_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    return a == b


def validate_original_graph_preserved(
    source: HeteroData,
    transformed: HeteroData,
):
    """Verify that every original node/edge attribute is unchanged."""
    if tuple(source.node_types) != tuple(transformed.node_types):
        raise RuntimeError("Node types changed during 3-hop preprocessing.")

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

    target_2hop_edges = 0
    target_3hop_edges = 0
    query_2hop_edges = 0
    query_3hop_edges = 0

    target_2hop_self_edges = 0
    target_3hop_self_edges = 0
    query_2hop_self_edges = 0
    query_3hop_self_edges = 0

    target_2hop_collisions = 0
    target_3hop_collisions = 0
    query_2hop_collisions = 0
    query_3hop_collisions = 0

    pbar = tqdm(
        range(len(targets)),
        desc=f"{dataset}/{split} 3hop",
        unit="pair",
        dynamic_ncols=True,
    )

    for i in pbar:
        new_target, t_stat = add_multihop_edges(
            targets[i]
        )
        new_query, q_stat = add_multihop_edges(
            queries[i]
        )

        validate_original_graph_preserved(
            targets[i],
            new_target,
        )
        validate_original_graph_preserved(
            queries[i],
            new_query,
        )

        new_targets.append(new_target)
        new_queries.append(new_query)

        target_2hop_edges += t_stat["generated_two_hop_edges"]
        target_3hop_edges += t_stat["generated_three_hop_edges"]
        query_2hop_edges += q_stat["generated_two_hop_edges"]
        query_3hop_edges += q_stat["generated_three_hop_edges"]

        target_2hop_self_edges += t_stat["generated_two_hop_self_edges"]
        target_3hop_self_edges += t_stat["generated_three_hop_self_edges"]
        query_2hop_self_edges += q_stat["generated_two_hop_self_edges"]
        query_3hop_self_edges += q_stat["generated_three_hop_self_edges"]

        target_2hop_collisions += t_stat["two_hop_relation_collisions"]
        target_3hop_collisions += t_stat["three_hop_relation_collisions"]
        query_2hop_collisions += q_stat["two_hop_relation_collisions"]
        query_3hop_collisions += q_stat["three_hop_relation_collisions"]

        pbar.set_postfix(
            t2=target_2hop_edges,
            t3=target_3hop_edges,
            q2=query_2hop_edges,
            q3=query_3hop_edges,
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
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_file": str(input_path),
        "output_file": str(output_path),
        "pairs": len(targets),
        "hop": 3,
        "construction_source": "original_1hop_edges_only",
        "keep_original_1hop_edges": True,
        "include_two_hop_edges": True,
        "include_three_hop_edges": True,
        "keep_duplicate_path_edges": True,
        "keep_two_hop_self_return_edges": True,
        "keep_three_hop_self_return_edges": True,
        "save_intermediates_attribute": False,
        "matching_matrices_reused_without_modification": True,
        "target_generated_two_hop_edges": target_2hop_edges,
        "target_generated_three_hop_edges": target_3hop_edges,
        "query_generated_two_hop_edges": query_2hop_edges,
        "query_generated_three_hop_edges": query_3hop_edges,
        "target_generated_two_hop_self_edges": target_2hop_self_edges,
        "target_generated_three_hop_self_edges": target_3hop_self_edges,
        "query_generated_two_hop_self_edges": query_2hop_self_edges,
        "query_generated_three_hop_self_edges": query_3hop_self_edges,
        "target_two_hop_relation_collisions": target_2hop_collisions,
        "target_three_hop_relation_collisions": target_3hop_collisions,
        "query_two_hop_relation_collisions": query_2hop_collisions,
        "query_three_hop_relation_collisions": query_3hop_collisions,
        "target_edge_schema": collect_edge_schema(new_targets),
        "query_edge_schema": collect_edge_schema(new_queries),
        "copied_ground_truth_sidecars": copied_sidecars,
        "elapsed_sec": elapsed,
    }

    per_split_summary = (
        output_dir / f"{dataset.lower()}_{split}_3hop_preprocess.json"
    )
    atomic_json_save(summary, per_split_summary)

    print()
    print("=" * 100)
    print(f"{dataset} / {split} DONE")
    print("input :", input_path)
    print("output:", output_path)
    print("pairs :", len(targets))
    print("target generated 2hop edges:", f"{target_2hop_edges:,}")
    print("target generated 3hop edges:", f"{target_3hop_edges:,}")
    print("query  generated 2hop edges:", f"{query_2hop_edges:,}")
    print("query  generated 3hop edges:", f"{query_3hop_edges:,}")
    print("elapsed:", f"{elapsed:.2f}s")
    print("summary:", per_split_summary)
    print("=" * 100)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build the deterministic 3-hop benchmark variant.")

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
        help=(
            "Use the original 1-hop benchmark as input. "
            "Do not point this at the saved 2-hop directory if you want "
            "the released 3-hop construction semantics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/benchmark/3hop"),
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

            # Refresh manifest after every completed split.
            atomic_json_save(
                {
                    "created_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "source_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "hop": 3,
                    "semantics": {
                        "constructed_directly_from_original_1hop_data": True,
                        "original_1hop_edges_preserved": True,
                        "two_hop_edges_included": True,
                        "three_hop_edges_included": True,
                        "duplicate_path_edges_preserved": True,
                        "two_hop_self_return_edges_preserved": True,
                        "three_hop_self_return_edges_preserved": True,
                        "ground_truth_reused_without_recomputation": True,
                        "intermediate_path_attributes_saved": False,
                    },
                    "completed": all_summaries,
                },
                args.output_dir / "preprocess_summary.json",
            )

    print()
    print("ALL REQUESTED 3-HOP PREPROCESSING COMPLETED")
    print("manifest:", args.output_dir / "preprocess_summary.json")


if __name__ == "__main__":
    main()
