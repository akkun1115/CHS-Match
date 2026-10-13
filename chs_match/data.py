"""Dataset loading and packed mini-batch collation for CHS-Match."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch.
        return torch.load(path, map_location="cpu")


class CHSMatchDataset(Dataset):
    """Benchmark format: [targets, queries, matching_matrices]."""

    def __init__(self, path: str):
        self.targets, self.queries, self.matching = _torch_load(path)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.targets[index],
            self.queries[index],
            self.matching[index],
        )


def collate_packed(samples):
    """Collate graph pairs and multi-positive GT into padded pair blocks."""
    targets, queries, labels = map(list, zip(*samples))
    target = Batch.from_data_list(targets)
    query = Batch.from_data_list(queries)

    packed_gt: Dict[str, Tensor] = {}
    layouts: Dict[str, dict] = {}
    batch_size = len(labels)

    for node_type in labels[0]:
        q_counts_list = [
            int(label[node_type].size(0))
            for label in labels
        ]
        t_counts_list = [
            int(label[node_type].size(1))
            for label in labels
        ]
        if any(q > t for q, t in zip(q_counts_list, t_counts_list)):
            raise RuntimeError(
                f"Injective matching requires Q<=T for node type {node_type}."
            )

        max_q = max(q_counts_list, default=0)
        max_t = max(t_counts_list, default=0)

        gt = torch.zeros(
            (batch_size, max_q, max_t),
            dtype=torch.bool,
        )
        for batch_index, label in enumerate(labels):
            local = label[node_type]
            qn, tn = local.shape
            gt[batch_index, :qn, :tn] = local > 0
        packed_gt[node_type] = gt

        q_counts = torch.tensor(q_counts_list, dtype=torch.long)
        t_counts = torch.tensor(t_counts_list, dtype=torch.long)
        q_offsets = torch.cumsum(
            torch.cat([q_counts.new_zeros(1), q_counts[:-1]]),
            dim=0,
        )
        t_offsets = torch.cumsum(
            torch.cat([t_counts.new_zeros(1), t_counts[:-1]]),
            dim=0,
        )

        q_pos = torch.arange(max_q)
        t_pos = torch.arange(max_t)
        q_active = q_pos.view(1, -1) < q_counts.view(-1, 1)
        t_active = t_pos.view(1, -1) < t_counts.view(-1, 1)
        q_index = q_offsets.view(-1, 1) + q_pos.view(1, -1)
        t_index = t_offsets.view(-1, 1) + t_pos.view(1, -1)

        if sum(q_counts_list) > 0:
            q_index = q_index.clamp_max(sum(q_counts_list) - 1)
        if sum(t_counts_list) > 0:
            t_index = t_index.clamp_max(sum(t_counts_list) - 1)

        real_active = q_active.unsqueeze(2) & t_active.unsqueeze(1)
        row_pos = torch.arange(max_t).view(1, max_t, 1)
        col_pos = torch.arange(max_t).view(1, 1, max_t)
        q_counts_3d = q_counts.view(batch_size, 1, 1)
        t_counts_3d = t_counts.view(batch_size, 1, 1)
        active_rows = row_pos < t_counts_3d
        active_cols = col_pos < t_counts_3d
        square_active = active_rows & active_cols

        layouts[node_type] = {
            "batch_size": batch_size,
            "q_counts": q_counts,
            "t_counts": t_counts,
            "q_index": q_index,
            "t_index": t_index,
            "q_active": q_active,
            "t_active": t_active,
            "real_active": real_active,
            "square_active": square_active,
            "dummy_mask": square_active & (row_pos >= q_counts_3d),
            "active_row_2d": active_rows.squeeze(2),
            "active_col_2d": active_cols.squeeze(1),
            "max_q": max_q,
            "max_t": max_t,
            "total_q": sum(q_counts_list),
            "total_t": sum(t_counts_list),
        }

    return target, query, packed_gt, layouts
