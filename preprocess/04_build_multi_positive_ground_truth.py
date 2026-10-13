"""Step 4: build multi-positive ground-truth correspondence matrices.

For each query-target pair, this script expands the generation-time mapping to
the set of admissible node-level correspondences under the benchmark matching
semantics. The output keeps the released benchmark tuple format.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import shutil
import statistics
import time
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


SPLITS = ("train", "val", "test")
DATASETS = ("LastFM", "IMDB", "DBLP")


# =============================================================================
# I/O
# =============================================================================

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


# =============================================================================
# Graph conversion / exact matching semantics
# =============================================================================

def feature_signature(x: torch.Tensor) -> Tuple:
    vals = x.detach().cpu().reshape(-1).tolist()
    return tuple(vals)


def hetero_to_colored_digraph(data: HeteroData) -> nx.DiGraph:
    g = nx.DiGraph()

    for ntype in data.node_types:
        x = data[ntype].x
        for nid in range(data[ntype].num_nodes):
            g.add_node(
                (str(ntype), int(nid)),
                node_type=str(ntype),
                feature=feature_signature(x[nid]),
                anchor=False,
            )

    relation_lists = defaultdict(list)

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        e = data[etype].edge_index.detach().cpu()

        rel_sig = (str(src_type), str(rel), str(dst_type))

        for src, dst in e.t().tolist():
            u = (str(src_type), int(src))
            v = (str(dst_type), int(dst))
            relation_lists[(u, v)].append(rel_sig)

    for (u, v), rels in relation_lists.items():
        rels = tuple(sorted(rels, key=repr))
        g.add_edge(u, v, rels=rels)

    return g


def node_match(a: dict, b: dict) -> bool:
    return (
        a["node_type"] == b["node_type"]
        and a["feature"] == b["feature"]
    )


def anchored_node_match(a: dict, b: dict) -> bool:
    return (
        a["node_type"] == b["node_type"]
        and a["feature"] == b["feature"]
        and bool(a.get("anchor", False)) == bool(b.get("anchor", False))
    )


def edge_match(a: dict, b: dict) -> bool:
    return a.get("rels", ()) == b.get("rels", ())


def make_empty_mm(target: HeteroData, query: HeteroData) -> Dict[str, torch.Tensor]:
    return {
        ntype: torch.zeros(
            query[ntype].num_nodes,
            target[ntype].num_nodes,
            dtype=torch.float32,
        )
        for ntype in query.node_types
    }


def seed_with_original_mm(
    exact_mm: Dict[str, torch.Tensor],
    seed_mm: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Keep the known generation-time correspondence in every stored GT.

    The query generator samples the query from its target and seed_mm stores that
    known q->target mapping. If exact search times out, the stored GT is an
    under-approximation consisting of:
        original positives + additional exact positives found before timeout.
    """
    for ntype in seed_mm:
        exact_mm[ntype][seed_mm[ntype] > 0] = 1.0
    return exact_mm


def validate_shapes(
    target: HeteroData,
    query: HeteroData,
    seed_mm: Dict[str, torch.Tensor],
):
    for ntype in query.node_types:
        if ntype not in seed_mm:
            raise RuntimeError(f"Seed matching matrix missing node type {ntype}")

        expected = (
            query[ntype].num_nodes,
            target[ntype].num_nodes,
        )
        if tuple(seed_mm[ntype].shape) != expected:
            raise RuntimeError(
                f"seed_mm[{ntype}] shape={tuple(seed_mm[ntype].shape)} "
                f"expected={expected}"
            )


def validate_exact_mm(
    exact_mm: Dict[str, torch.Tensor],
    seed_mm: Dict[str, torch.Tensor],
):
    # Every query row must have at least one exact target.
    for ntype, mat in exact_mm.items():
        row_counts = mat.sum(dim=1)
        if not torch.all(row_counts >= 1):
            bad = torch.where(row_counts < 1)[0].tolist()[:20]
            raise RuntimeError(
                f"Exact GT has rows without a valid match for {ntype}: {bad}"
            )

    # Generation-time answer must remain accepted.
    for ntype in seed_mm:
        old_positive = seed_mm[ntype] > 0
        missing = old_positive & ~(exact_mm[ntype] > 0)
        if missing.any():
            idx = missing.nonzero(as_tuple=False)[:20].tolist()
            raise RuntimeError(
                f"Original generated mapping is not contained in exact GT "
                f"for {ntype}. First missing positions: {idx}"
            )


def multigt_stats(exact_mm: Dict[str, torch.Tensor]) -> dict:
    total_rows = 0
    multi_rows = 0
    total_positive_positions = 0
    max_row_answers = 0

    for mat in exact_mm.values():
        row_counts = mat.sum(dim=1).to(torch.long)
        total_rows += int(row_counts.numel())
        multi_rows += int((row_counts > 1).sum().item())
        total_positive_positions += int(row_counts.sum().item())
        if row_counts.numel():
            max_row_answers = max(
                max_row_answers,
                int(row_counts.max().item()),
            )

    return {
        "query_rows": total_rows,
        "multi_answer_rows": multi_rows,
        "positive_positions": total_positive_positions,
        "max_answers_per_row": max_row_answers,
    }


# =============================================================================
# Per-pair timeout
# =============================================================================

@contextmanager
def pair_timeout(seconds: float):
    """Linux/Unix timeout for a single pair. 0 disables it."""
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(
            f"Pair exceeded timeout of {seconds:.1f}s."
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# Exact solver A: enumerate all full solutions
# =============================================================================

def solve_pair_enumerate(
    target: HeteroData,
    query: HeteroData,
    seed_mm: Dict[str, torch.Tensor],
    max_solutions: int = 0,
    switch_after: int = 0,
    heartbeat_every: int = 0,
    heartbeat_prefix: str = "",
):
    validate_shapes(target, query, seed_mm)

    gt = hetero_to_colored_digraph(target)
    gq = hetero_to_colored_digraph(query)

    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        gt,
        gq,
        node_match=node_match,
        edge_match=edge_match,
    )

    exact_mm = seed_with_original_mm(
        make_empty_mm(target, query),
        seed_mm,
    )

    solution_count = 0
    iterator = matcher.subgraph_isomorphisms_iter()
    started = time.perf_counter()

    try:
        for mapping_t_to_q in iterator:
            solution_count += 1

            q_to_t = {
                q_node: t_node
                for t_node, q_node in mapping_t_to_q.items()
            }

            if len(q_to_t) != gq.number_of_nodes():
                raise RuntimeError(
                    "Exact matcher returned an incomplete query mapping."
                )

            for q_node, t_node in q_to_t.items():
                q_type, q_id = q_node
                t_type, t_id = t_node

                if q_type != t_type:
                    raise RuntimeError(
                        f"Internal type mismatch: {q_node} -> {t_node}"
                    )

                exact_mm[q_type][q_id, t_id] = 1.0

            if heartbeat_every > 0 and solution_count % heartbeat_every == 0:
                tqdm.write(
                    f"{heartbeat_prefix} enumerated={solution_count:,} "
                    f"elapsed={time.perf_counter() - started:.1f}s"
                )

            if switch_after > 0 and solution_count >= switch_after:
                try:
                    next(iterator)
                except StopIteration:
                    break

                stat = multigt_stats(exact_mm)
                stat.update({
                    "status": "switch_to_union",
                    "ground_truth_complete": False,
                    "solver_used": "enumerate",
                    "solution_count": solution_count,
                    "solution_count_exact": False,
                    "candidate_checks": 0,
                    "enumerated_before_switch": solution_count,
                    "timed_out": False,
                })
                return exact_mm, stat

            if max_solutions > 0 and solution_count >= max_solutions:
                try:
                    next(iterator)
                except StopIteration:
                    break
                raise RuntimeError(
                    f"More than --max-solutions={max_solutions} exact mappings "
                    "exist for this pair. Ground truth would be incomplete."
                )

    except TimeoutError:
        stat = multigt_stats(exact_mm)
        stat.update({
            "status": "partial_timeout",
            "ground_truth_complete": False,
            "solver_used": "enumerate",
            "solution_count": solution_count,
            "solution_count_exact": False,
            "candidate_checks": 0,
            "enumerated_before_switch": 0,
            "timed_out": True,
        })
        return exact_mm, stat

    if solution_count == 0:
        raise RuntimeError(
            "No exact induced matching found. The generated query should be "
            "an induced subgraph of its target."
        )

    validate_exact_mm(exact_mm, seed_mm)

    stat = multigt_stats(exact_mm)
    stat.update({
        "status": "exact",
        "ground_truth_complete": True,
        "solver_used": "enumerate",
        "solution_count": solution_count,
        "solution_count_exact": True,
        "candidate_checks": 0,
        "enumerated_before_switch": 0,
        "timed_out": False,
    })
    return exact_mm, stat


# =============================================================================
# Exact solver B: exact union by anchored existence checks
# =============================================================================

def candidate_targets_for_query_node(
    gt: nx.DiGraph,
    gq: nx.DiGraph,
    q_node,
):
    q_attr = gq.nodes[q_node]
    return [
        t_node
        for t_node, t_attr in gt.nodes(data=True)
        if node_match(t_attr, q_attr)
    ]


def exists_with_forced_pair(
    gt: nx.DiGraph,
    gq: nx.DiGraph,
    t_node,
    q_node,
) -> bool:
    # Force exactly q_node <-> t_node by marking only those two nodes.
    gt.nodes[t_node]["anchor"] = True
    gq.nodes[q_node]["anchor"] = True

    try:
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            gt,
            gq,
            node_match=anchored_node_match,
            edge_match=edge_match,
        )
        return matcher.subgraph_is_isomorphic()
    finally:
        gt.nodes[t_node]["anchor"] = False
        gq.nodes[q_node]["anchor"] = False


def solve_pair_union(
    target: HeteroData,
    query: HeteroData,
    seed_mm: Dict[str, torch.Tensor],
    heartbeat_every: int = 0,
    heartbeat_prefix: str = "",
    enumerated_before_switch: int = 0,
    initial_mm: Optional[Dict[str, torch.Tensor]] = None,
):
    validate_shapes(target, query, seed_mm)

    gt = hetero_to_colored_digraph(target)
    gq = hetero_to_colored_digraph(query)

    if initial_mm is None:
        exact_mm = make_empty_mm(target, query)
    else:
        exact_mm = {
            k: v.detach().cpu().clone()
            for k, v in initial_mm.items()
        }

    exact_mm = seed_with_original_mm(exact_mm, seed_mm)

    checks = 0
    started = time.perf_counter()

    try:
        for q_node in gq.nodes:
            q_type, q_id = q_node
            candidates = candidate_targets_for_query_node(gt, gq, q_node)

            if not candidates:
                raise RuntimeError(
                    f"No type/feature-compatible target candidate for {q_node}"
                )

            for t_node in candidates:
                _, t_id = t_node

                # Already known valid either from the original generated mapping
                # or from the enumerate phase before an auto switch.
                if exact_mm[q_type][q_id, t_id] > 0:
                    continue

                checks += 1

                if exists_with_forced_pair(
                    gt,
                    gq,
                    t_node,
                    q_node,
                ):
                    exact_mm[q_type][q_id, t_id] = 1.0

                if heartbeat_every > 0 and checks % heartbeat_every == 0:
                    tqdm.write(
                        f"{heartbeat_prefix} candidate_checks={checks:,} "
                        f"elapsed={time.perf_counter() - started:.1f}s"
                    )

    except TimeoutError:
        stat = multigt_stats(exact_mm)
        stat.update({
            "status": "partial_timeout",
            "ground_truth_complete": False,
            "solver_used": "union",
            "solution_count": -1,
            "solution_count_exact": False,
            "candidate_checks": checks,
            "enumerated_before_switch": enumerated_before_switch,
            "timed_out": True,
        })
        return exact_mm, stat

    validate_exact_mm(exact_mm, seed_mm)

    stat = multigt_stats(exact_mm)
    stat.update({
        "status": "exact",
        "ground_truth_complete": True,
        "solver_used": "union",
        "solution_count": -1,
        "solution_count_exact": False,
        "candidate_checks": checks,
        "enumerated_before_switch": enumerated_before_switch,
        "timed_out": False,
    })
    return exact_mm, stat


def solve_pair(
    target: HeteroData,
    query: HeteroData,
    seed_mm: Dict[str, torch.Tensor],
    solver_mode: str,
    max_solutions: int,
    auto_switch_solutions: int,
    heartbeat_every: int,
    heartbeat_prefix: str,
):
    if solver_mode == "enumerate":
        return solve_pair_enumerate(
            target,
            query,
            seed_mm,
            max_solutions=max_solutions,
            switch_after=0,
            heartbeat_every=heartbeat_every,
            heartbeat_prefix=heartbeat_prefix,
        )

    if solver_mode == "union":
        return solve_pair_union(
            target,
            query,
            seed_mm,
            heartbeat_every=heartbeat_every,
            heartbeat_prefix=heartbeat_prefix,
        )

    if solver_mode != "auto":
        raise ValueError(solver_mode)

    partial_mm, stat = solve_pair_enumerate(
        target,
        query,
        seed_mm,
        max_solutions=max_solutions,
        switch_after=auto_switch_solutions,
        heartbeat_every=heartbeat_every,
        heartbeat_prefix=heartbeat_prefix,
    )

    if stat["status"] in ("exact", "partial_timeout"):
        return partial_mm, stat

    if stat["status"] != "switch_to_union":
        raise RuntimeError(f"Unexpected solver status: {stat['status']}")

    tqdm.write(
        f"{heartbeat_prefix} reached "
        f"{stat['enumerated_before_switch']:,} full solutions; "
        "switching to union solver while retaining positives already found."
    )

    return solve_pair_union(
        target,
        query,
        seed_mm,
        heartbeat_every=heartbeat_every,
        heartbeat_prefix=heartbeat_prefix,
        enumerated_before_switch=stat["enumerated_before_switch"],
        initial_mm=partial_mm,
    )


# =============================================================================
# Checkpoint / progress helpers
# =============================================================================

def percentile(values, q: float):
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".partial.pt")


def save_checkpoint(path: Path, result_mms, rows, next_index: int):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "result_mms": result_mms,
            "rows": rows,
            "next_index": next_index,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path):
    obj = torch_load(path)

    if "result_mms" not in obj:
        raise RuntimeError(
            "Checkpoint is from an older incompatible version. "
            "Use a fresh --output-dir or remove the old .partial.pt."
        )

    return (
        obj["result_mms"],
        obj["rows"],
        int(obj["next_index"]),
    )


DETAIL_FIELDS = [
    "dataset",
    "split",
    "pair_index",
    "status",
    "ground_truth_complete",
    "solver_used",
    "solution_count",
    "solution_count_exact",
    "candidate_checks",
    "enumerated_before_switch",
    "timed_out",
    "query_rows",
    "multi_answer_rows",
    "positive_positions",
    "max_answers_per_row",
    "elapsed_sec",
]


def atomic_write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def split_artifact_paths(output_path: Path):
    stem = output_path.stem
    parent = output_path.parent
    return {
        "details_csv": parent / f"{stem}_pair_details.csv",
        "summary_json": parent / f"{stem}_summary.json",
        "summary_txt": parent / f"{stem}_summary.txt",
    }


def summarize_rows(
    dataset: str,
    split: str,
    input_path: Path,
    output_path: Path,
    input_pairs: int,
    attempted_pairs: int,
    rows,
    solver_mode: str,
    auto_switch_solutions: int,
    pair_timeout_sec: float,
    wall_elapsed_sec: float,
    completed: bool,
):
    exact_rows = [r for r in rows if bool(r["ground_truth_complete"])]
    partial_rows = [r for r in rows if not bool(r["ground_truth_complete"])]
    timeout_rows = [r for r in rows if bool(r["timed_out"])]

    total_query_rows = sum(int(r["query_rows"]) for r in rows)
    total_multi_rows = sum(int(r["multi_answer_rows"]) for r in rows)
    total_positive_positions = sum(int(r["positive_positions"]) for r in rows)

    elapsed_values = [float(r["elapsed_sec"]) for r in rows]
    avg_pair_sec = statistics.mean(elapsed_values) if elapsed_values else 0.0
    median_pair_sec = statistics.median(elapsed_values) if elapsed_values else 0.0
    p95_pair_sec = percentile(elapsed_values, 0.95)
    max_pair_sec = max(elapsed_values) if elapsed_values else 0.0

    slowest_pair_index = -1
    if rows:
        slowest = max(rows, key=lambda r: float(r["elapsed_sec"]))
        slowest_pair_index = int(slowest["pair_index"])

    multi_pairs = sum(int(r["multi_answer_rows"]) > 0 for r in rows)
    union_pairs = sum(r["solver_used"] == "union" for r in rows)
    switched_pairs = sum(
        int(r["enumerated_before_switch"]) > 0
        for r in rows
    )

    max_answers = max(
        [int(r["max_answers_per_row"]) for r in rows] or [0]
    )

    return {
        "dataset": dataset,
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_pairs": input_pairs,
        "attempted_pairs": attempted_pairs,
        "processed_pairs_so_far": len(rows),
        "completed": completed,
        "exact_pairs": len(exact_rows),
        "partial_pairs": len(partial_rows),
        "timeout_partial_pairs": len(timeout_rows),
        "partial_original_pair_indices": [
            int(r["pair_index"]) for r in partial_rows
        ],
        "solver_mode_requested": solver_mode,
        "auto_switch_solutions": auto_switch_solutions,
        "pair_timeout_sec": pair_timeout_sec,
        "timeout_policy": (
            "retain under-approximate multi-GT: original generated positives "
            "+ additional exact positives found before timeout"
            if pair_timeout_sec > 0
            else "disabled"
        ),
        "matching_semantics": (
            "injective induced heterogeneous subgraph isomorphism; "
            "exact node type + exact node feature/category + exact directed "
            "edge/relation signature"
        ),
        "accuracy_definition": (
            "Top-1 is correct iff predicted target node is positive in "
            "the stored multi-hot ground-truth row"
        ),
        "pairs_with_at_least_one_multi_answer_row": multi_pairs,
        "pair_multi_answer_fraction": (
            multi_pairs / len(rows) if rows else 0.0
        ),
        "total_query_rows": total_query_rows,
        "multi_answer_rows": total_multi_rows,
        "multi_answer_row_fraction": (
            total_multi_rows / total_query_rows
            if total_query_rows
            else 0.0
        ),
        "mean_stored_valid_targets_per_query_row": (
            total_positive_positions / total_query_rows
            if total_query_rows
            else 0.0
        ),
        "max_stored_valid_targets_per_query_row": max_answers,
        "pairs_solved_by_union_mode": union_pairs,
        "pairs_auto_switched_to_union": switched_pairs,
        "average_pair_sec": avg_pair_sec,
        "median_pair_sec": median_pair_sec,
        "p95_pair_sec": p95_pair_sec,
        "max_pair_sec": max_pair_sec,
        "slowest_pair_index": slowest_pair_index,
        "wall_elapsed_sec_this_invocation": wall_elapsed_sec,
    }


def summary_to_text(summary: dict) -> str:
    lines = [
        "=" * 100,
        f"{summary['dataset']} / {summary['split']} MULTI-GT SUMMARY",
        "=" * 100,
        f"input_file                    : {summary['input_file']}",
        f"output_file                   : {summary['output_file']}",
        f"processed                     : {summary['processed_pairs_so_far']}/{summary['attempted_pairs']}",
        f"completed                     : {summary['completed']}",
        f"exact_pairs                   : {summary['exact_pairs']}",
        f"partial_pairs                 : {summary['partial_pairs']}",
        f"timeout_partial_pairs         : {summary['timeout_partial_pairs']}",
        f"partial_original_pair_indices : {summary['partial_original_pair_indices']}",
        f"multi_answer_pair_fraction    : {summary['pair_multi_answer_fraction']:.6f}",
        f"multi_answer_row_fraction     : {summary['multi_answer_row_fraction']:.6f}",
        f"mean_stored_valid_targets/row : {summary['mean_stored_valid_targets_per_query_row']:.6f}",
        f"max_stored_valid_targets/row  : {summary['max_stored_valid_targets_per_query_row']}",
        f"average_pair_sec              : {summary['average_pair_sec']:.6f}",
        f"median_pair_sec               : {summary['median_pair_sec']:.6f}",
        f"p95_pair_sec                  : {summary['p95_pair_sec']:.6f}",
        f"max_pair_sec                  : {summary['max_pair_sec']:.6f}",
        f"slowest_pair_index            : {summary['slowest_pair_index']}",
        f"solver_mode                   : {summary['solver_mode_requested']}",
        f"auto_switched_pairs           : {summary['pairs_auto_switched_to_union']}",
        f"union_pairs                   : {summary['pairs_solved_by_union_mode']}",
        f"pair_timeout_sec              : {summary['pair_timeout_sec']}",
        "=" * 100,
    ]
    return "\\n".join(lines) + "\\n"


def write_split_progress_artifacts(output_path: Path, summary: dict, rows):
    paths = split_artifact_paths(output_path)
    atomic_write_csv(paths["details_csv"], rows, DETAIL_FIELDS)
    atomic_write_json(paths["summary_json"], summary)
    atomic_write_text(paths["summary_txt"], summary_to_text(summary))


def archive_existing_split_artifacts(output_path: Path):
    candidates = [output_path, checkpoint_path_for(output_path)]
    candidates.extend(split_artifact_paths(output_path).values())

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_path.parent / "archive" / f"{output_path.stem}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for p in existing:
        shutil.copy2(p, archive_dir / p.name)


def rebuild_aggregate_artifacts(output_dir: Path):
    # Aggregate from all persistent per-split files, including previous runs.
    detail_files = sorted(
        p for p in output_dir.glob("*_pair_details.csv")
        if p.name != "construction_pair_details.csv"
    )

    all_rows = []
    for path in detail_files:
        with path.open("r", newline="", encoding="utf-8") as f:
            all_rows.extend(csv.DictReader(f))

    atomic_write_csv(
        output_dir / "construction_pair_details.csv",
        all_rows,
        DETAIL_FIELDS,
    )

    summary_files = sorted(
        p for p in output_dir.glob("*_summary.json")
        if p.name != "construction_summaries.json"
    )

    summaries = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in summary_files
    ]

    atomic_write_json(
        output_dir / "construction_summaries.json",
        summaries,
    )


# =============================================================================
# File processing
# =============================================================================

def process_file(
    input_path: Path,
    output_path: Path,
    dataset: str,
    split: str,
    max_pairs: int,
    max_solutions: int,
    solver_mode: str,
    auto_switch_solutions: int,
    heartbeat_every: int,
    slow_pair_sec: float,
    pair_timeout_sec: float,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
):
    if output_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite intentionally."
        )

    if overwrite and not resume:
        archive_existing_split_artifacts(output_path)

    obj = torch_load(input_path)
    targets, queries, seed_mms = unpack_dataset(obj)

    n_total = len(targets)
    n_process = n_total if max_pairs <= 0 else min(max_pairs, n_total)

    partial_path = checkpoint_path_for(output_path)

    if resume and partial_path.exists():
        result_mms, rows, start_index = load_checkpoint(partial_path)

        if len(result_mms) != start_index or len(rows) != start_index:
            raise RuntimeError(
                "Checkpoint inconsistent: expected one result/MM per processed pair."
            )

        tqdm.write(
            f"[resume] {dataset}/{split}: restarting at pair "
            f"{start_index + 1}/{n_process}"
        )
    else:
        result_mms = []
        rows = []
        start_index = 0

    print()
    print("=" * 100)
    print(f"{dataset} / {split}")
    print("input :", input_path)
    print("output:", output_path)
    print(f"pairs : {n_process}/{n_total}")
    print("solver:", solver_mode)
    if solver_mode == "auto":
        print("auto switch after full solutions:", auto_switch_solutions)
    print(
        "pair timeout:",
        (
            f"{pair_timeout_sec:.1f}s -> retain partial GT and continue"
            if pair_timeout_sec > 0
            else "disabled"
        ),
    )
    print("checkpoint:", partial_path)
    print("=" * 100)

    file_start = time.perf_counter()

    # A persistent summary exists immediately, not only after the split ends.
    progress_summary = summarize_rows(
        dataset, split, input_path, output_path,
        n_total, n_process, rows,
        solver_mode, auto_switch_solutions,
        pair_timeout_sec, 0.0, False,
    )
    write_split_progress_artifacts(output_path, progress_summary, rows)
    rebuild_aggregate_artifacts(output_path.parent)

    pbar = tqdm(
        range(start_index, n_process),
        total=n_process,
        initial=start_index,
        desc=f"{dataset}/{split}",
        unit="pair",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    try:
        for i in pbar:
            start = time.perf_counter()
            prefix = f"[{dataset}/{split} pair {i + 1}/{n_process}]"

            with pair_timeout(pair_timeout_sec):
                result_mm, stat = solve_pair(
                    targets[i],
                    queries[i],
                    seed_mms[i],
                    solver_mode=solver_mode,
                    max_solutions=max_solutions,
                    auto_switch_solutions=auto_switch_solutions,
                    heartbeat_every=heartbeat_every,
                    heartbeat_prefix=prefix,
                )

            elapsed = time.perf_counter() - start
            result_mms.append(result_mm)

            row = {
                "dataset": dataset,
                "split": split,
                "pair_index": i,
                **stat,
                "elapsed_sec": elapsed,
            }
            rows.append(row)

            if stat["timed_out"]:
                tqdm.write(
                    f"{prefix} TIMEOUT after {elapsed:.2f}s -> saved PARTIAL GT "
                    "(original answer retained + additional discovered positives)"
                )

            elapsed_values = [float(r["elapsed_sec"]) for r in rows]
            avg_pair = statistics.mean(elapsed_values)
            partial_count = sum(
                not bool(r["ground_truth_complete"])
                for r in rows
            )
            multi_frac = (
                sum(int(r["multi_answer_rows"]) > 0 for r in rows)
                / len(rows)
            )

            sol_display = (
                str(stat["solution_count"])
                if stat["solution_count_exact"]
                else ("partial" if stat["timed_out"] else "union")
            )

            pbar.set_postfix(
                last=f"{elapsed:.2f}s",
                avg=f"{avg_pair:.2f}s",
                sol=sol_display,
                multi=f"{multi_frac:.1%}",
                mode=stat["solver_used"],
                partial=partial_count,
            )

            if slow_pair_sec > 0 and elapsed >= slow_pair_sec:
                tqdm.write(
                    f"{prefix} SLOW PAIR: {elapsed:.2f}s, "
                    f"status={stat['status']}, "
                    f"solver={stat['solver_used']}, "
                    f"multi_rows={stat['multi_answer_rows']}/{stat['query_rows']}, "
                    f"candidate_checks={stat['candidate_checks']:,}, "
                    f"enumerated_before_switch={stat['enumerated_before_switch']:,}"
                )

            # Persist CSV + JSON + TXT after every pair.
            progress_summary = summarize_rows(
                dataset, split, input_path, output_path,
                n_total, n_process, rows,
                solver_mode, auto_switch_solutions,
                pair_timeout_sec,
                time.perf_counter() - file_start,
                False,
            )
            write_split_progress_artifacts(output_path, progress_summary, rows)
            rebuild_aggregate_artifacts(output_path.parent)

            # Save resumable tensor state periodically and always after timeout.
            if (
                stat["timed_out"]
                or (
                    checkpoint_every > 0
                    and (i + 1) % checkpoint_every == 0
                )
            ):
                save_checkpoint(
                    partial_path,
                    result_mms,
                    rows,
                    next_index=i + 1,
                )

    except BaseException:
        next_index = len(result_mms)

        save_checkpoint(
            partial_path,
            result_mms,
            rows,
            next_index=next_index,
        )

        progress_summary = summarize_rows(
            dataset, split, input_path, output_path,
            n_total, n_process, rows,
            solver_mode, auto_switch_solutions,
            pair_timeout_sec,
            time.perf_counter() - file_start,
            False,
        )
        write_split_progress_artifacts(output_path, progress_summary, rows)
        rebuild_aggregate_artifacts(output_path.parent)

        tqdm.write(
            f"[checkpoint saved] {partial_path}; "
            f"next pair={next_index + 1}, stored={len(result_mms)}"
        )
        raise
    finally:
        pbar.close()

    if len(result_mms) != n_process or len(rows) != n_process:
        raise RuntimeError(
            "Every attempted pair must have exactly one stored GT and status row."
        )

    # Pair order/count stay unchanged, including timed-out partial pairs.
    torch.save(
        [
            targets[:n_process],
            queries[:n_process],
            result_mms,
        ],
        output_path,
    )

    if partial_path.exists():
        partial_path.unlink()

    final_summary = summarize_rows(
        dataset, split, input_path, output_path,
        n_total, n_process, rows,
        solver_mode, auto_switch_solutions,
        pair_timeout_sec,
        time.perf_counter() - file_start,
        True,
    )
    write_split_progress_artifacts(output_path, final_summary, rows)
    rebuild_aggregate_artifacts(output_path.parent)

    print()
    print(summary_to_text(final_summary), end="")

    return final_summary, rows


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build multi-positive ground-truth correspondence matrices.")

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
        default=Path("build/pairs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/benchmark"),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Pilot: process only first N pairs. 0 = all.",
    )

    parser.add_argument(
        "--solver-mode",
        choices=["enumerate", "union", "auto"],
        default="auto",
        help=(
            "enumerate=all full solutions; union=exact multi-GT directly; "
            "auto=enumerate easy pairs and switch on combinatorial explosion."
        ),
    )
    parser.add_argument(
        "--auto-switch-solutions",
        type=int,
        default=500,
        help=(
            "In auto mode, switch a pair to exact union solver if at least "
            "this many full solutions are found. Default: 500."
        ),
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=0,
        help=(
            "Hard safety cap for enumerate mode. 0 = unlimited. "
            "Exceeding the cap aborts instead of saving incomplete GT."
        ),
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=500,
        help=(
            "During a single slow pair, print progress every N full solutions "
            "or candidate checks. 0 disables heartbeat."
        ),
    )
    parser.add_argument(
        "--slow-pair-sec",
        type=float,
        default=5.0,
        help="Print a warning for pairs taking at least this many seconds.",
    )
    parser.add_argument(
        "--pair-timeout",
        type=float,
        default=0.0,
        help=(
            "If one pair exceeds this many seconds, save the multi-GT "
            "found so far (always including the original generated answer), "
            "mark it partial, and continue. 0 disables timeout."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save resumable partial state every N completed pairs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the .partial.pt checkpoint if present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    splits = SPLITS if args.split == "all" else (args.split,)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (dataset, split)
        for dataset in datasets
        for split in splits
    ]

    for job_idx, (dataset, split) in enumerate(jobs, start=1):
        print()
        print("#" * 100)
        print(f"JOB {job_idx}/{len(jobs)}: {dataset} / {split}")
        print("#" * 100)

        in_path = (
            args.input_dir
            / f"{dataset.lower()}_{split}.pt"
        )

        if not in_path.exists():
            raise FileNotFoundError(in_path)

        suffix = ""
        if args.max_pairs > 0:
            suffix = f"_pilot{args.max_pairs}"

        out_path = (
            args.output_dir
            / f"{dataset.lower()}_{split}{suffix}.pt"
        )

        summary, rows = process_file(
            input_path=in_path,
            output_path=out_path,
            dataset=dataset,
            split=split,
            max_pairs=args.max_pairs,
            max_solutions=args.max_solutions,
            solver_mode=args.solver_mode,
            auto_switch_solutions=args.auto_switch_solutions,
            heartbeat_every=args.heartbeat_every,
            slow_pair_sec=args.slow_pair_sec,
            pair_timeout_sec=args.pair_timeout,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            overwrite=args.overwrite,
        )

        # process_file already persists per-split artifacts after every pair.
        # Rebuild aggregate files from ALL existing split files so a later run
        # never erases previous dataset/split results.
        rebuild_aggregate_artifacts(args.output_dir)

    print()
    print("=" * 100)
    print("DONE")
    print("aggregate summaries:", args.output_dir / "construction_summaries.json")
    print("aggregate details  :", args.output_dir / "construction_pair_details.csv")
    print("Per-split *_summary.json / *_summary.txt / *_pair_details.csv persist independently.")
    print("=" * 100)


if __name__ == "__main__":
    main()
