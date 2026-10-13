"""Training/evaluation entry point for the CHS-Match model.

The CLI exposes only settings needed to reproduce analyses reported in the
paper: dataset/seed, layer depth and hop size, lambda_mm, Sinkhorn iterations,
the normalization comparison, and the CGNA/CASA ablations. All other adopted
implementation choices are fixed in chs_match.model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch_geometric
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from chs_match.data import CHSMatchDataset, collate_packed
from chs_match.model import (
    CHSMatch,
    CHSMatchLoss,
    HEADS,
    HIDDEN_DIM,
    MIN_CATEGORIES,
)
def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_LAYERS = {
    "LastFM": 2,
    "IMDB": 3,
    "DBLP": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate CHS-Match on locally generated benchmark files."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["LastFM", "IMDB", "DBLP"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--layers",
        "--num-layer",
        dest="layers",
        type=int,
        default=None,
        help="Model depth. Defaults to the paper setting for each dataset.",
    )
    parser.add_argument(
        "--hop",
        "--data-hop",
        dest="hop",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Neighborhood hop size; used to choose the default data directory.",
    )
    parser.add_argument("--lambda-mm", type=float, default=0.2)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument(
        "--normalization",
        choices=["partial_sinkhorn", "row_softmax"],
        default="partial_sinkhorn",
        help=(
            "partial_sinkhorn is CHS-Match; row_softmax is retained only for "
            "the controlled normalization comparison in the paper."
        ),
    )
    parser.add_argument(
        "--ablation",
        choices=["full", "no_cgna", "no_casa", "no_both"],
        default="full",
        help="Ablation configurations reported in the paper.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a configuration if its result JSON already exists.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override benchmark directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    if args.layers is None:
        args.layers = DEFAULT_LAYERS[args.dataset]
    if args.layers < 1:
        parser.error("--layers must be >= 1")
    if args.sinkhorn_iters < 1:
        parser.error("--sinkhorn-iters must be >= 1")
    if not 0.0 <= args.lambda_mm <= 1.0:
        parser.error("--lambda-mm must be in [0,1]")

    if args.data_dir is None:
        args.data_dir = (
            Path("benchmark")
            if args.hop == 1
            else Path("benchmark") / f"{args.hop}hop"
        )
    return args


def infer_category_cardinalities(
    train_set,
    node_types: Tuple[str, ...],
) -> Dict[str, int]:
    """Infer per-node-type categorical vocabulary sizes from TRAIN only."""
    inferred = {node_type: 0 for node_type in node_types}

    for target, query, _ in train_set:
        for graph in (target, query):
            for node_type in graph.node_types:
                x = graph[node_type].x
                if x.numel() == 0:
                    continue
                if x.dim() == 1:
                    count = int(x.max().item()) + 1
                elif x.dim() == 2 and x.size(1) == 1:
                    count = int(x.max().item()) + 1
                elif x.dim() == 2:
                    count = int(x.size(1))
                else:
                    raise RuntimeError(
                        f"Unsupported feature shape {tuple(x.shape)} "
                        f"for node type {node_type}."
                    )
                inferred[node_type] = max(inferred[node_type], count)

    return {
        node_type: max(MIN_CATEGORIES, inferred[node_type])
        for node_type in node_types
    }


def collect_metadata(train_set, val_set, test_set):
    """Use the union of TRAIN relations and reject unseen VAL/TEST relations."""
    first_target = train_set[0][0]
    node_types = tuple(first_target.node_types)
    train_edge_types = set()

    for target, query, _ in train_set:
        if tuple(target.node_types) != node_types:
            raise RuntimeError("TRAIN target node types are inconsistent.")
        if tuple(query.node_types) != node_types:
            raise RuntimeError("TRAIN query node types are inconsistent.")
        train_edge_types.update(tuple(x) for x in target.edge_types)
        train_edge_types.update(tuple(x) for x in query.edge_types)

    def validate(name, dataset):
        unseen = set()
        for target, query, _ in dataset:
            for graph in (target, query):
                if tuple(graph.node_types) != node_types:
                    raise RuntimeError(f"{name} node types are inconsistent.")
                unseen.update(
                    tuple(edge_type)
                    for edge_type in graph.edge_types
                    if tuple(edge_type) not in train_edge_types
                )
        if unseen:
            raise RuntimeError(
                f"{name} contains relation/meta-path types unseen in TRAIN: "
                f"{sorted(unseen)}"
            )

    validate("VAL", val_set)
    validate("TEST", test_set)
    return node_types, tuple(sorted(train_edge_types))


def move_batch(batch, device):
    target, query, packed_gt, layouts = batch
    target = target.to(device, non_blocking=True)
    query = query.to(device, non_blocking=True)
    packed_gt = {
        key: value.to(device, non_blocking=True)
        for key, value in packed_gt.items()
    }
    layouts = {
        node_type: {
            key: (
                value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in layout.items()
        }
        for node_type, layout in layouts.items()
    }
    return target, query, packed_gt, layouts


def top1_counts(
    prediction: Dict[str, Tensor],
    packed_gt: Dict[str, Tensor],
    layouts: Dict[str, dict],
) -> Tuple[Tensor, int]:
    correct: Optional[Tensor] = None
    total = 0

    for node_type, pred in prediction.items():
        gt = packed_gt[node_type]
        layout = layouts[node_type]
        masked_pred = torch.where(
            layout["t_active"].unsqueeze(1),
            pred,
            torch.full_like(pred, -1.0),
        )
        pred_index = masked_pred.argmax(dim=2)
        hit = (
            torch.gather(gt, 2, pred_index.unsqueeze(2))
            .squeeze(2)
            .bool()
            & layout["q_active"]
        )
        count = hit.sum()
        correct = count if correct is None else correct + count
        total += int(layout["total_q"])

    if correct is None:
        correct = torch.zeros(
            (),
            device=next(iter(prediction.values())).device,
            dtype=torch.long,
        )
    return correct, total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum: Optional[Tensor] = None
    correct_sum: Optional[Tensor] = None
    total_nodes = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            target, query, gt, layouts = move_batch(batch, device)
            predictions = model(target, query, layouts)
            loss = criterion(predictions, gt, layouts, query)

            detached = loss.detach()
            loss_sum = (
                detached
                if loss_sum is None
                else loss_sum + detached
            )

            correct, count = top1_counts(
                predictions[-1], gt, layouts
            )
            correct_sum = (
                correct
                if correct_sum is None
                else correct_sum + correct
            )
            total_nodes += count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started

    mean_loss = (
        float((loss_sum / max(len(loader), 1)).item())
        if loss_sum is not None
        else 0.0
    )
    accuracy = (
        float((correct_sum.float() / total_nodes).item())
        if correct_sum is not None and total_nodes
        else 0.0
    )
    return mean_loss, accuracy, seconds


def clone_state(model) -> Dict[str, Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def result_path(args) -> Path:
    """Return a stable, configuration-specific result path.

    The hash prevents silently overwriting results from sensitivity runs
    differing in lambda, iterations, optimizer settings, or input data.
    """
    variant = (
        "sinkhorn"
        if args.normalization == "partial_sinkhorn"
        else "row-softmax"
    )
    if args.ablation != "full":
        variant += f"_{args.ablation}"

    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hop": args.hop,
        "layers": args.layers,
        "normalization": args.normalization,
        "sinkhorn_iters": args.sinkhorn_iters,
        "lambda_mm": args.lambda_mm,
        "ablation": args.ablation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "data_dir": str(args.data_dir.resolve()),
        "device": args.device,
        "hidden_dim": HIDDEN_DIM,
        "heads": HEADS,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    name = (
        f"{args.dataset.lower()}_seed{args.seed}_"
        f"{args.hop}hop_{args.layers}layer_{variant}_{run_id}.json"
    )
    return args.output_dir / name


def main():
    args = parse_args()
    output = result_path(args)
    if output.exists():
        if args.skip_existing:
            print(f"skipped (existing result): {output}")
            return
        raise FileExistsError(
            f"Result already exists: {output}. "
            "Choose a different --output-dir to preserve the previous run, "
            "or pass --skip-existing to reuse the result."
        )
    fix_random_seed(args.seed)
    device = torch.device(args.device)

    paths = {
        split: args.data_dir / f"{args.dataset.lower()}_{split}.pt"
        for split in ("train", "val", "test")
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    train_set = CHSMatchDataset(str(paths["train"]))
    val_set = CHSMatchDataset(str(paths["val"]))
    test_set = CHSMatchDataset(str(paths["test"]))

    node_types, edge_types = collect_metadata(
        train_set, val_set, test_set
    )
    categories = infer_category_cardinalities(
        train_set, node_types
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_packed,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_packed,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_packed,
        num_workers=0,
        pin_memory=True,
    )

    model = CHSMatch(
        metadata=(list(node_types), list(edge_types)),
        num_layers=args.layers,
        num_categories_dict=categories,
        normalization=args.normalization,
        sinkhorn_iters=args.sinkhorn_iters,
        ablation=args.ablation,
    ).to(device)

    criterion = CHSMatchLoss(
        lambda_mm=args.lambda_mm
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
    )

    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    train_times = []

    progress = tqdm(range(args.epochs))
    for epoch in progress:
        model.train()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        train_loss_tensor: Optional[Tensor] = None
        for batch in train_loader:
            target, query, gt, layouts = move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(target, query, layouts)
            loss = criterion(
                predictions, gt, layouts, query
            )
            loss.backward()
            optimizer.step()

            detached = loss.detach()
            train_loss_tensor = (
                detached
                if train_loss_tensor is None
                else train_loss_tensor + detached
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_times.append(
            time.perf_counter() - started
        )

        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        if (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = clone_state(model)

        train_loss = (
            float(
                (
                    train_loss_tensor
                    / max(len(train_loader), 1)
                ).item()
            )
            if train_loss_tensor is not None
            else 0.0
        )
        progress.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_loss=f"{val_loss:.5f}",
            val_acc=f"{val_acc:.5f}",
            best=f"{best_val_acc:.5f}",
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected.")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_seconds = evaluate(
        model, test_loader, criterion, device
    )

    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hop": args.hop,
        "layers": args.layers,
        "normalization": args.normalization,
        "sinkhorn_iters": (
            args.sinkhorn_iters
            if args.normalization == "partial_sinkhorn"
            else None
        ),
        "lambda_mm": args.lambda_mm,
        "ablation": args.ablation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": HIDDEN_DIM,
        "heads": HEADS,
        "learning_rate": args.lr,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_geometric": str(torch_geometric.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
        },
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_seconds_per_epoch_mean": statistics.mean(train_times),
        "test_seconds": test_seconds,
        "trainable_parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
