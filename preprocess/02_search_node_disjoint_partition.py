"""Step 2: search a strict source-node-disjoint 70/15/15 partition.

The search preserves per-node-type split ratios and favors partitions whose
train/validation/test source subgraphs retain usable heterogeneous structure.
Model accuracy is never used during partition selection.

By default, selected partitions are written under build/partitions/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


CONFIG = {
    "LastFM": {
        "origin_file": "lastfm_origin.pt",
        "target_min": 50,
        "target_max": 100,
    },
    "IMDB": {
        "origin_file": "imdb_origin.pt",
        "target_min": 60,
        "target_max": 110,
    },
    "DBLP": {
        "origin_file": "dblp_origin.pt",
        "target_min": 70,
        "target_max": 110,
    },
}

SPLITS = ("train", "val", "test")
SPLIT_ID = {"train": 0, "val": 1, "test": 2}
ID_SPLIT = {0: "train", 1: "val", 2: "test"}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_global_index(origin: HeteroData):
    offsets = {}
    inverse = []
    type_global_ids = {}
    cursor = 0

    for ntype in origin.node_types:
        offsets[ntype] = cursor
        ids = list(range(cursor, cursor + origin[ntype].num_nodes))
        type_global_ids[ntype] = ids

        for local_id in range(origin[ntype].num_nodes):
            inverse.append((ntype, local_id))

        cursor += origin[ntype].num_nodes

    return offsets, inverse, type_global_ids


def build_weighted_adjacency(origin: HeteroData, offsets):
    """
    adj[u][v] is the number of STORED directed edges between u and v.
    The map is made symmetric only for partition analysis/optimization.
    """
    n = sum(origin[nt].num_nodes for nt in origin.node_types)
    adj = [defaultdict(int) for _ in range(n)]
    total_stored_edges = 0

    for etype in origin.edge_types:
        src_type, _, dst_type = etype
        edge_index = origin[etype].edge_index.detach().cpu()

        so = offsets[src_type]
        do = offsets[dst_type]

        for s, d in edge_index.t().tolist():
            u = so + int(s)
            v = do + int(d)

            total_stored_edges += 1

            if u == v:
                continue

            adj[u][v] += 1
            adj[v][u] += 1

    adj = [dict(x) for x in adj]
    degree = [sum(x.values()) for x in adj]

    return adj, degree, total_stored_edges


def quota_counts(n, train_frac, val_frac, test_frac):
    fracs = [train_frac, val_frac, test_frac]
    raw = [n * x for x in fracs]
    base = [math.floor(x) for x in raw]

    remainder = n - sum(base)

    order = sorted(
        range(3),
        key=lambda i: raw[i] - base[i],
        reverse=True,
    )

    for i in order[:remainder]:
        base[i] += 1

    return {
        "train": base[0],
        "val": base[1],
        "test": base[2],
    }


def initialize_assignment(
    origin,
    type_global_ids,
    rng,
    train_frac,
    val_frac,
    test_frac,
):
    n_total = sum(origin[nt].num_nodes for nt in origin.node_types)

    assignment = [-1] * n_total

    buckets = {
        nt: {split: [] for split in SPLITS}
        for nt in origin.node_types
    }

    quotas = {}

    for nt in origin.node_types:
        ids = list(type_global_ids[nt])
        rng.shuffle(ids)

        q = quota_counts(
            len(ids),
            train_frac,
            val_frac,
            test_frac,
        )
        quotas[nt] = q

        a = q["train"]
        b = a + q["val"]

        parts = {
            "train": ids[:a],
            "val": ids[a:b],
            "test": ids[b:],
        }

        for split, gids in parts.items():
            sid = SPLIT_ID[split]
            buckets[nt][split] = gids

            for gid in gids:
                assignment[gid] = sid

    return assignment, buckets, quotas


# ---------------------------------------------------------------------------
# Fast balanced-boundary optimization used only to produce good candidates
# ---------------------------------------------------------------------------

def initial_boundary_volume(assignment, adj, degree):
    boundary = [0, 0, 0]
    volume = [0, 0, 0]

    for u, sid in enumerate(assignment):
        volume[sid] += degree[u]

    for u, neighbors in enumerate(adj):
        for v, weight in neighbors.items():
            if u >= v:
                continue

            su = assignment[u]
            sv = assignment[v]

            if su != sv:
                boundary[su] += weight
                boundary[sv] += weight

    return boundary, volume


def boundary_ratios(boundary, volume):
    return [
        boundary[i] / volume[i] if volume[i] > 0 else float("inf")
        for i in range(3)
    ]


def balanced_boundary_score(boundary, volume):
    ratios = boundary_ratios(boundary, volume)
    return max(ratios) + 0.25 * (sum(ratios) / 3.0)


def proposed_swap_state(
    a,
    b,
    assignment,
    adj,
    degree,
    boundary,
    volume,
):
    sa = assignment[a]
    sb = assignment[b]

    new_boundary = boundary.copy()
    new_volume = volume.copy()

    new_volume[sa] += degree[b] - degree[a]
    new_volume[sb] += degree[a] - degree[b]

    affected = {}

    for v, w in adj[a].items():
        key = (a, v) if a < v else (v, a)
        affected[key] = w

    for v, w in adj[b].items():
        key = (b, v) if b < v else (v, b)
        affected[key] = w

    def new_label(node):
        if node == a:
            return sb
        if node == b:
            return sa
        return assignment[node]

    for (u, v), weight in affected.items():
        old_u = assignment[u]
        old_v = assignment[v]
        new_u = new_label(u)
        new_v = new_label(v)

        if old_u != old_v:
            new_boundary[old_u] -= weight
            new_boundary[old_v] -= weight

        if new_u != new_v:
            new_boundary[new_u] += weight
            new_boundary[new_v] += weight

    return new_boundary, new_volume


def optimize_candidate(
    origin,
    type_global_ids,
    adj,
    degree,
    train_frac,
    val_frac,
    test_frac,
    seed,
    steps,
):
    rng = random.Random(seed)

    assignment, buckets, quotas = initialize_assignment(
        origin,
        type_global_ids,
        rng,
        train_frac,
        val_frac,
        test_frac,
    )

    boundary, volume = initial_boundary_volume(
        assignment,
        adj,
        degree,
    )

    score = balanced_boundary_score(boundary, volume)
    node_types = list(origin.node_types)
    accepted = 0

    for _ in range(steps):
        nt = rng.choice(node_types)

        available = [
            split
            for split in SPLITS
            if buckets[nt][split]
        ]

        if len(available) < 2:
            continue

        s1, s2 = rng.sample(available, 2)

        l1 = buckets[nt][s1]
        l2 = buckets[nt][s2]

        i = rng.randrange(len(l1))
        j = rng.randrange(len(l2))

        a = l1[i]
        b = l2[j]

        if assignment[a] == assignment[b]:
            continue

        new_boundary, new_volume = proposed_swap_state(
            a,
            b,
            assignment,
            adj,
            degree,
            boundary,
            volume,
        )

        new_score = balanced_boundary_score(
            new_boundary,
            new_volume,
        )

        if new_score < score:
            assignment[a], assignment[b] = (
                assignment[b],
                assignment[a],
            )
            l1[i], l2[j] = b, a

            boundary = new_boundary
            volume = new_volume
            score = new_score
            accepted += 1

    return {
        "assignment": assignment,
        "buckets": buckets,
        "quotas": quotas,
        "boundary": boundary,
        "volume": volume,
        "boundary_ratios": boundary_ratios(boundary, volume),
        "balanced_boundary_score": score,
        "accepted_swaps": accepted,
    }


# ---------------------------------------------------------------------------
# Structural evaluation: this decides which candidate is actually selected
# ---------------------------------------------------------------------------

def analyze_split_structure(
    split_id,
    assignment,
    adj,
    inverse,
    node_types,
    target_min,
    target_max,
):
    nodes = [
        u
        for u, sid in enumerate(assignment)
        if sid == split_id
    ]

    node_set = set(nodes)
    n = len(nodes)

    internal_degree = {}
    nonisolated = 0
    internal_incident_weight = 0
    total_incident_weight = 0

    for u in nodes:
        internal = 0
        total = 0

        for v, w in adj[u].items():
            total += w

            if assignment[v] == split_id:
                internal += w

        internal_degree[u] = internal
        internal_incident_weight += internal
        total_incident_weight += total

        if internal > 0:
            nonisolated += 1

    nonisolated_fraction = (
        nonisolated / n if n else 0.0
    )

    internal_neighbor_fraction = (
        internal_incident_weight / total_incident_weight
        if total_incident_weight > 0
        else 0.0
    )

    avg_internal_degree = (
        internal_incident_weight / n
        if n
        else 0.0
    )

    # Connected components in the induced split graph.
    visited = set()
    component_records = []

    for start in nodes:
        if start in visited:
            continue

        q = deque([start])
        visited.add(start)

        comp = []
        comp_types = set()

        while q:
            u = q.popleft()
            comp.append(u)

            nt, _ = inverse[u]
            comp_types.add(nt)

            for v in adj[u]:
                if (
                    v in node_set
                    and v not in visited
                ):
                    visited.add(v)
                    q.append(v)

        component_records.append(
            {
                "size": len(comp),
                "types": comp_types,
            }
        )

    component_records.sort(
        key=lambda x: x["size"],
        reverse=True,
    )

    lcc_size = (
        component_records[0]["size"]
        if component_records
        else 0
    )

    lcc_fraction = (
        lcc_size / n
        if n
        else 0.0
    )

    required_types = set(node_types)

    min_range_usable_nodes = sum(
        rec["size"]
        for rec in component_records
        if (
            rec["size"] >= target_min
            and rec["types"] == required_types
        )
    )

    full_range_usable_nodes = sum(
        rec["size"]
        for rec in component_records
        if (
            rec["size"] >= target_max
            and rec["types"] == required_types
        )
    )

    min_range_usable_fraction = (
        min_range_usable_nodes / n
        if n
        else 0.0
    )

    full_range_usable_fraction = (
        full_range_usable_nodes / n
        if n
        else 0.0
    )

    eligible_min_components = sum(
        1
        for rec in component_records
        if (
            rec["size"] >= target_min
            and rec["types"] == required_types
        )
    )

    eligible_full_components = sum(
        1
        for rec in component_records
        if (
            rec["size"] >= target_max
            and rec["types"] == required_types
        )
    )

    return {
        "num_nodes": n,
        "num_components": len(component_records),
        "largest_component_nodes": lcc_size,
        "lcc_fraction": lcc_fraction,
        "nonisolated_fraction": nonisolated_fraction,
        "avg_internal_degree": avg_internal_degree,
        "internal_neighbor_fraction": internal_neighbor_fraction,
        "min_range_usable_node_fraction":
            min_range_usable_fraction,
        "full_range_usable_node_fraction":
            full_range_usable_fraction,
        "eligible_components_target_min":
            eligible_min_components,
        "eligible_components_target_max":
            eligible_full_components,
    }


def global_edge_cut_ratio(boundary, total_stored_edges):
    cut = sum(boundary) / 2.0

    return (
        cut / total_stored_edges
        if total_stored_edges > 0
        else 0.0
    )


def candidate_structural_key(split_metrics, global_cut):
    """
    Lexicographic key. Larger is better.

    Edge-cut is LAST, not first.
    """
    vals = list(split_metrics.values())

    worst_full_range = min(
        x["full_range_usable_node_fraction"]
        for x in vals
    )

    worst_min_range = min(
        x["min_range_usable_node_fraction"]
        for x in vals
    )

    worst_nonisolated = min(
        x["nonisolated_fraction"]
        for x in vals
    )

    worst_internal_neighbor = min(
        x["internal_neighbor_fraction"]
        for x in vals
    )

    worst_lcc = min(
        x["lcc_fraction"]
        for x in vals
    )

    worst_avg_internal_degree = min(
        x["avg_internal_degree"]
        for x in vals
    )

    # Final tie-break: lower edge cut is better.
    return (
        worst_full_range,
        worst_min_range,
        worst_nonisolated,
        worst_internal_neighbor,
        worst_lcc,
        worst_avg_internal_degree,
        -global_cut,
    )


def serialize_from_buckets(buckets, inverse):
    result = {
        split: {}
        for split in SPLITS
    }

    for nt, by_split in buckets.items():
        for split, gids in by_split.items():
            local_ids = sorted(
                inverse[gid][1]
                for gid in gids
            )

            result[split][nt] = torch.tensor(
                local_ids,
                dtype=torch.long,
            )

    return result


def type_counts_from_buckets(buckets):
    return {
        split: {
            nt: len(buckets[nt][split])
            for nt in buckets
        }
        for split in SPLITS
    }


def node_counts_from_buckets(buckets):
    return {
        split: sum(
            len(buckets[nt][split])
            for nt in buckets
        )
        for split in SPLITS
    }


def type_props_from_counts(type_counts, node_counts):
    return {
        split: {
            nt:
                type_counts[split][nt]
                / node_counts[split]
            for nt in type_counts[split]
        }
        for split in SPLITS
    }


def max_prop_deviation(origin, type_props):
    total = sum(
        origin[nt].num_nodes
        for nt in origin.node_types
    )

    global_props = {
        nt:
            origin[nt].num_nodes / total
        for nt in origin.node_types
    }

    max_dev = 0.0

    for split in SPLITS:
        for nt in origin.node_types:
            max_dev = max(
                max_dev,
                abs(
                    type_props[split][nt]
                    - global_props[nt]
                ),
            )

    return max_dev, global_props


def main():
    parser = argparse.ArgumentParser(description="Search a structurally usable source-node-disjoint partition.")

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

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260916)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not math.isclose(
        args.train_frac
        + args.val_frac
        + args.test_frac,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "split fractions must sum to 1"
        )

    cfg = CONFIG[args.dataset]

    root = args.project_root.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "build"
        / "partitions"
    )

    origin = torch_load(
        root
        / "data"
        / cfg["origin_file"]
    )

    offsets, inverse, type_global_ids = (
        make_global_index(origin)
    )

    adj, degree, total_stored_edges = (
        build_weighted_adjacency(
            origin,
            offsets,
        )
    )

    print("=" * 90)
    print(args.dataset)
    print("=" * 90)
    print(
        "Target size range:",
        cfg["target_min"],
        "-",
        cfg["target_max"],
    )
    print(
        "Selection priority:",
        "usable heterogeneous components -> nonisolated -> internal structure -> edge cut",
    )
    print(
        "Restarts / steps:",
        args.restarts,
        "/",
        args.steps,
    )

    candidates = []

    for restart in range(args.restarts):
        restart_seed = (
            args.seed
            + restart * 1000003
        )

        candidate = optimize_candidate(
            origin,
            type_global_ids,
            adj,
            degree,
            args.train_frac,
            args.val_frac,
            args.test_frac,
            restart_seed,
            args.steps,
        )

        split_metrics = {}

        for sid in range(3):
            split_metrics[ID_SPLIT[sid]] = (
                analyze_split_structure(
                    sid,
                    candidate["assignment"],
                    adj,
                    inverse,
                    origin.node_types,
                    cfg["target_min"],
                    cfg["target_max"],
                )
            )

        global_cut = global_edge_cut_ratio(
            candidate["boundary"],
            total_stored_edges,
        )

        structural_key = candidate_structural_key(
            split_metrics,
            global_cut,
        )

        record = {
            "restart": restart,
            "seed": restart_seed,
            "candidate": candidate,
            "split_metrics": split_metrics,
            "global_edge_cut_ratio": global_cut,
            "structural_key": structural_key,
        }

        candidates.append(record)

        print()
        print(
            f"restart={restart} "
            f"edge_cut={global_cut:.6f}"
        )

        for split in SPLITS:
            m = split_metrics[split]

            print(
                f"  {split}: "
                f"full_usable={m['full_range_usable_node_fraction']:.4f}, "
                f"min_usable={m['min_range_usable_node_fraction']:.4f}, "
                f"nonisolated={m['nonisolated_fraction']:.4f}, "
                f"internal={m['internal_neighbor_fraction']:.4f}, "
                f"LCC={m['lcc_fraction']:.4f}, "
                f"avg_deg={m['avg_internal_degree']:.3f}"
            )

    # Structural metrics decide; edge-cut is only the last tie-break.
    best = max(
        candidates,
        key=lambda x: x["structural_key"],
    )

    buckets = best["candidate"]["buckets"]

    type_counts = type_counts_from_buckets(
        buckets
    )

    node_counts = node_counts_from_buckets(
        buckets
    )

    type_props = type_props_from_counts(
        type_counts,
        node_counts,
    )

    max_dev, global_props = max_prop_deviation(
        origin,
        type_props,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_ids = serialize_from_buckets(
        buckets,
        inverse,
    )

    partition_path = (
        output_dir
        / f"{args.dataset.lower()}_partition.pt"
    )

    torch.save(
        {
            "dataset": args.dataset,
            "selection_method":
                "exact per-type stratification + structural-usability-first selection",
            "split_fractions": {
                "train": args.train_frac,
                "val": args.val_frac,
                "test": args.test_frac,
            },
            "trial": best["restart"],
            "seed": best["seed"],
            "edge_cut_ratio":
                best["global_edge_cut_ratio"],
            "max_type_prop_deviation":
                max_dev,
            "structural_metrics":
                best["split_metrics"],
            "node_ids": node_ids,
        },
        partition_path,
    )

    report_candidates = []

    for rec in candidates:
        report_candidates.append(
            {
                "restart": rec["restart"],
                "seed": rec["seed"],
                "global_edge_cut_ratio":
                    rec["global_edge_cut_ratio"],
                "split_metrics":
                    rec["split_metrics"],
                "structural_key":
                    list(rec["structural_key"]),
                "accepted_swaps":
                    rec["candidate"]["accepted_swaps"],
            }
        )

    report = {
        "dataset":
            args.dataset,
        "selection_method":
            "exact per-type stratification + structural-usability-first selection",
        "selection_priority": [
            "maximize worst-split full_range_usable_node_fraction",
            "maximize worst-split min_range_usable_node_fraction",
            "maximize worst-split nonisolated_fraction",
            "maximize worst-split internal_neighbor_fraction",
            "maximize worst-split lcc_fraction",
            "maximize worst-split avg_internal_degree",
            "minimize global edge-cut ratio",
        ],
        "target_size_range": {
            "min": cfg["target_min"],
            "max": cfg["target_max"],
        },
        "restarts":
            args.restarts,
        "steps_per_restart":
            args.steps,
        "selected_restart":
            best["restart"],
        "selected_seed":
            best["seed"],
        "split_fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "global_type_proportions":
            global_props,
        "node_counts":
            node_counts,
        "node_type_counts":
            type_counts,
        "node_type_proportions":
            type_props,
        "max_type_prop_deviation":
            max_dev,
        "global_edge_cut_ratio":
            best["global_edge_cut_ratio"],
        "structural_metrics":
            best["split_metrics"],
        "candidates":
            report_candidates,
        "partition_file":
            str(partition_path),
    }

    report_path = (
        output_dir
        / f"{args.dataset.lower()}_partition_report.json"
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
    print("=" * 90)
    print("SELECTED STRUCTURALLY USABLE PARTITION")
    print("=" * 90)
    print(
        "restart / seed:",
        best["restart"],
        "/",
        best["seed"],
    )
    print(
        "global edge-cut:",
        f"{best['global_edge_cut_ratio']:.6f}",
    )
    print(
        "max type proportion deviation:",
        f"{max_dev:.6f}",
    )

    for split in SPLITS:
        print(
            split,
            best["split_metrics"][split],
        )

    print("Saved:", partition_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
