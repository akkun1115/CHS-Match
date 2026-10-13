"""Core CHS-Match model used in the paper.

This publication version intentionally keeps only the adopted CHS-Match
configuration and the variants that are actually reported in the manuscript:
- partial Sinkhorn normalization (default), with row-wise softmax retained only
  for the controlled normalization comparison;
- soft target-to-query semantic alignment;
- sum reduction in cross-attentive semantic aggregation;
- log-parameterized learnable temperature;
- packed per-pair matching storage;
- CGNA/CASA ablations reported in the ablation study.

Development-only switches from the experimental code are deliberately omitted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax


HIDDEN_DIM = 64
HEADS = 4
MLP_DROPOUT = 0.1
GAT_DROPOUT = 0.1
MIN_CATEGORIES = 10


class HeteroProjectLayer(nn.Module):
    """Type-specific categorical embedding followed by LayerNorm and ELU."""

    def __init__(
        self,
        node_types: List[str],
        h_dim: int,
        num_categories_dict: Dict[str, int],
    ):
        super().__init__()
        self.project = nn.ModuleDict({
            node_type: nn.Embedding(num_categories_dict[node_type], h_dim)
            for node_type in node_types
        })
        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(h_dim)
            for node_type in node_types
        })

    @staticmethod
    def _category_ids(x: Tensor) -> Tensor:
        if x.dim() == 1:
            raw = x
        elif x.dim() == 2 and x.size(1) == 1:
            raw = x.squeeze(1)
        elif x.dim() == 2:
            raw = x.argmax(dim=1)
        else:
            raise RuntimeError(
                "Expected [N], [N,1] categorical IDs or [N,C] one-hot input; "
                f"got shape {tuple(x.shape)}."
            )
        return raw.long()

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            node_type: F.elu(
                self.norm[node_type](
                    self.project[node_type](self._category_ids(x))
                )
            )
            for node_type, x in x_dict.items()
        }


def block_diagonal_mm_from_packed(
    packed_mm: Tensor,
    rhs: Tensor,
    layout: dict,
) -> Tensor:
    """Compute independent local M_i @ rhs_i products in one padded batch."""
    if rhs.size(0) == 0:
        return rhs.new_zeros((0,) + rhs.shape[1:])

    trailing_shape = rhs.shape[1:]
    rhs_flat = rhs.reshape(rhs.size(0), -1)
    packed_rhs = rhs_flat[layout["t_index"]]
    packed_rhs = torch.where(
        layout["t_active"].unsqueeze(-1),
        packed_rhs,
        torch.zeros((), dtype=rhs.dtype, device=rhs.device),
    )
    packed_out = torch.bmm(packed_mm, packed_rhs)
    out_flat = packed_out[layout["q_active"]]
    return out_flat.reshape((-1,) + trailing_shape)


def batched_cosine_blocks(
    q_emb: Tensor,
    t_emb: Tensor,
    layout: dict,
) -> Tensor:
    """Compute cosine scores directly in [B,Qmax,Tmax] pair blocks."""
    batch_size = layout["batch_size"]
    max_q = layout["max_q"]
    max_t = layout["max_t"]
    if batch_size == 0 or max_q == 0 or max_t == 0:
        return q_emb.new_zeros((batch_size, max_q, max_t))

    q_pack = q_emb[layout["q_index"]]
    t_pack = t_emb[layout["t_index"]]
    dot = torch.bmm(q_pack, t_pack.transpose(1, 2))
    q_norm = torch.norm(q_pack, dim=2, keepdim=True)
    t_norm = torch.norm(t_pack, dim=2).unsqueeze(1)
    norm = q_norm * t_norm
    score = torch.where(
        norm != 0,
        dot / (norm + 1e-9),
        -torch.ones_like(dot),
    )
    return torch.where(
        layout["real_active"],
        score,
        torch.zeros_like(score),
    )


class MatchingMatrix(nn.Module):
    """Packed matching matrix with learnable log-temperature.

    Partial Sinkhorn is the adopted model. Row-wise softmax is kept only because
    it is the comparison method reported in the normalization study.
    """

    def __init__(
        self,
        normalization: str = "partial_sinkhorn",
        sinkhorn_iters: int = 5,
    ):
        super().__init__()
        if normalization not in {"partial_sinkhorn", "row_softmax"}:
            raise ValueError(normalization)
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1")
        self.normalization = normalization
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.log_temperature = nn.Parameter(
            torch.tensor([math.log(0.5)], dtype=torch.float32)
        )

    def _temperature(self) -> Tensor:
        return torch.exp(self.log_temperature)

    def _row_softmax(self, packed_scores: Tensor, layout: dict) -> Tensor:
        active = layout["real_active"]
        logits = torch.where(
            active,
            packed_scores / self._temperature(),
            torch.full_like(packed_scores, -1e9),
        )
        assignment = F.softmax(logits, dim=2)
        return torch.where(
            active,
            assignment,
            torch.zeros_like(assignment),
        )

    def _partial_sinkhorn(self, packed_scores: Tensor, layout: dict) -> Tensor:
        """Log-domain partial Sinkhorn with explicit dummy query rows."""
        batch_size = layout["batch_size"]
        max_q = layout["max_q"]
        max_t = layout["max_t"]
        if batch_size == 0 or max_t == 0:
            return packed_scores.new_zeros((batch_size, max_q, max_t))

        scaled = packed_scores / self._temperature()
        neg_value = torch.finfo(scaled.dtype).min
        kernel = scaled.new_full((batch_size, max_t, max_t), neg_value)

        real_active = layout["real_active"]
        if max_q > 0:
            kernel[:, :max_q, :] = torch.where(
                real_active,
                scaled,
                torch.full_like(scaled, neg_value),
            )

        active = layout["square_active"]
        kernel = torch.where(
            layout["dummy_mask"],
            torch.zeros_like(kernel),
            kernel,
        )
        kernel = torch.where(
            active,
            kernel,
            torch.full_like(kernel, neg_value),
        )

        log_u = kernel.new_zeros((batch_size, max_t))
        log_v = kernel.new_zeros((batch_size, max_t))
        for _ in range(self.sinkhorn_iters):
            row_lse = torch.logsumexp(
                kernel + log_v.unsqueeze(1),
                dim=2,
            )
            log_u = torch.where(
                layout["active_row_2d"],
                -row_lse,
                torch.zeros_like(row_lse),
            )

            col_lse = torch.logsumexp(
                kernel + log_u.unsqueeze(2),
                dim=1,
            )
            log_v = torch.where(
                layout["active_col_2d"],
                -col_lse,
                torch.zeros_like(col_lse),
            )

        assignment = torch.exp(
            kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
        )
        assignment = torch.where(
            active,
            assignment,
            torch.zeros_like(assignment),
        )
        real_assignment = assignment[:, :max_q, :]
        return torch.where(
            real_active,
            real_assignment,
            torch.zeros_like(real_assignment),
        )

    def forward(
        self,
        t_emb: Dict[str, Tensor],
        q_emb: Dict[str, Tensor],
        layouts: Dict[str, dict],
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for node_type in t_emb:
            scores = batched_cosine_blocks(
                q_emb[node_type],
                t_emb[node_type],
                layouts[node_type],
            )
            if self.normalization == "partial_sinkhorn":
                out[node_type] = self._partial_sinkhorn(
                    scores,
                    layouts[node_type],
                )
            else:
                out[node_type] = self._row_softmax(
                    scores,
                    layouts[node_type],
                )
        return out


class ResidualMLP(nn.Module):
    """Two-layer MLP used before the residual update."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.norm1 = nn.LayerNorm(2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(MLP_DROPOUT)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.elu(self.norm1(self.fc1(x))))
        x = self.dropout(F.elu(self.norm2(self.fc2(x))))
        return x


def han_semantic_aggregation(
    xs: List[Tensor],
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:
    if len(xs) == 0:
        return None, None

    out = torch.stack(xs)
    if out.numel() == 0:
        return out.view(0, out.size(-1)), None

    attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
    attn = F.softmax(attn_score, dim=0)
    out = torch.sum(
        attn.view(len(xs), 1, 1) * out,
        dim=0,
    )
    return out, attn


def paired_han_style_aggregation(
    xs_q: List[Tensor],
    xs_kv: List[Tensor],
    q_lin: nn.Module,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:
    if len(xs_kv) == 0:
        return None, None

    out_q = torch.stack(xs_q)
    out_kv = torch.stack(xs_kv)
    if out_kv.numel() == 0:
        return out_kv.view(0, out_kv.size(-1)), None

    q_repr = q_lin(out_q).mean(dim=1)
    k_repr = k_lin(out_kv).mean(dim=1)
    if q_repr.size(0) == k_repr.size(0):
        attn_score = (q_repr * k_repr).sum(dim=-1)
    else:
        attn_score = torch.matmul(q_repr, k_repr.T).mean(dim=0)

    attn = F.softmax(attn_score, dim=0)
    attn_for_weight = attn.view(-1, 1, 1)
    return (
        (attn_for_weight * out_kv).sum(dim=0),
        attn_for_weight,
    )


def cross_semantic_aggregation(
    xs_q: List[Tensor],
    xs_kv: List[Tensor],
    q_lin: nn.Module,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:
    """CASA cross-attention with the adopted sum reduction."""
    if len(xs_kv) == 0:
        return None, None

    z = torch.stack(xs_kv, dim=0)
    q_in = torch.stack(xs_q, dim=0)
    if z.numel() == 0:
        return z.view(0, z.size(-1)), None

    q_repr = q_lin(q_in)
    k_repr = k_lin(z)
    score = torch.einsum("bnd,and->ab", q_repr, k_repr)
    attn = F.softmax(score, dim=0)
    y = torch.einsum("ab,and->bnd", attn, z)
    return y.sum(dim=0), attn


class CHSHeteroConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = HEADS,
        negative_slope: float = 0.2,
        dropout: float = GAT_DROPOUT,
        ablation: str = "full",
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {
                node_type: in_channels
                for node_type in metadata[0]
            }
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads})."
            )
        if ablation not in {"full", "no_cgna", "no_casa", "no_both"}:
            raise ValueError(ablation)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.enable_cgna = ablation not in {"no_cgna", "no_both"}
        self.enable_casa = ablation not in {"no_casa", "no_both"}

        self.k_lin_self = nn.Linear(out_channels, out_channels)
        self.q_lin_self = nn.Linear(out_channels, out_channels)
        self.k_lin_cross = nn.Linear(out_channels, out_channels)
        self.q_lin_cross = nn.Linear(out_channels, out_channels)

        self.k_han_style_self = nn.Linear(out_channels, out_channels)
        self.q_han_style_self = nn.Linear(out_channels, out_channels)
        self.k_han_style_cross = nn.Linear(out_channels, out_channels)
        self.q_han_style_cross = nn.Linear(out_channels, out_channels)

        self.k = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.empty(1, out_channels))

        self.proj = nn.ModuleDict({
            node_type: Linear(channels, out_channels)
            for node_type, channels in self.in_channels.items()
        })

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            key = "__".join(edge_type)
            self.lin_src[key] = nn.Parameter(
                torch.empty(1, heads, dim)
            )
            self.lin_dst[key] = nn.Parameter(
                torch.empty(1, heads, dim)
            )

        self.gamma_logit = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin_self.reset_parameters()
        self.q_lin_self.reset_parameters()
        self.k_lin_cross.reset_parameters()
        self.q_lin_cross.reset_parameters()
        self.k_han_style_self.reset_parameters()
        self.q_han_style_self.reset_parameters()
        self.k_han_style_cross.reset_parameters()
        self.q_han_style_cross.reset_parameters()
        self.k.reset_parameters()
        glorot(self.q)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        cross_dict: Optional[Dict[NodeType, List[Tensor]]],
        counterpart_dict: Dict[NodeType, Tensor],
        packed_mm: Dict[NodeType, Tensor],
        target_conv: bool,
        matching_layouts: Dict[str, dict],
    ) -> Tuple[
        Dict[NodeType, Tensor],
        Dict[NodeType, List[Tensor]],
    ]:
        h = self.heads
        d = self.out_channels // self.heads

        x_node_dict = {
            node_type: self.proj[node_type](x).view(-1, h, d)
            for node_type, x in x_dict.items()
        }
        counterpart_node_dict: Dict[NodeType, Tensor] = {}
        relation_outputs: Dict[NodeType, List[Tensor]] = {
            node_type: []
            for node_type in x_dict
        }

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_key = "__".join(edge_type)

            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]

            if not target_conv and self.enable_cgna:
                for node_type in (src_type, dst_type):
                    if node_type not in counterpart_node_dict:
                        counterpart = block_diagonal_mm_from_packed(
                            packed_mm[node_type],
                            counterpart_dict[node_type],
                            matching_layouts[node_type],
                        )
                        counterpart_node_dict[node_type] = (
                            self.proj[node_type](counterpart)
                            .view(-1, h, d)
                        )

                x_src = counterpart_node_dict[src_type]
                x_dst = counterpart_node_dict[dst_type]

            alpha_src = (
                x_src * self.lin_src[edge_key]
            ).sum(dim=-1)
            alpha_dst = (
                x_dst * self.lin_dst[edge_key]
            ).sum(dim=-1)

            out = self.propagate(
                edge_index,
                x=(x_src, x_dst),
                alpha=(alpha_src, alpha_dst),
            )
            relation_outputs[dst_type].append(F.relu(out))

        aggregated: Dict[NodeType, Tensor] = {}
        for node_type, outs in relation_outputs.items():
            out_han, _ = han_semantic_aggregation(
                outs,
                self.q,
                self.k,
            )

            if not self.enable_casa:
                aggregated[node_type] = out_han
                continue

            if cross_dict is None:
                aligned_semantics = outs
                out_cross, _ = cross_semantic_aggregation(
                    aligned_semantics,
                    outs,
                    self.q_lin_self,
                    self.k_lin_self,
                )
                out_han_style, _ = paired_han_style_aggregation(
                    aligned_semantics,
                    outs,
                    self.q_han_style_self,
                    self.k_han_style_self,
                )
            else:
                cross_t = torch.stack(
                    cross_dict[node_type],
                    dim=0,
                )
                target_semantics = cross_t.permute(1, 0, 2)
                aligned_q = block_diagonal_mm_from_packed(
                    packed_mm[node_type],
                    target_semantics,
                    matching_layouts[node_type],
                )
                aligned_t = aligned_q.permute(1, 0, 2)
                aligned_semantics = [
                    aligned_t[e]
                    for e in range(aligned_t.size(0))
                ]

                out_cross, _ = cross_semantic_aggregation(
                    aligned_semantics,
                    outs,
                    self.q_lin_cross,
                    self.k_lin_cross,
                )
                out_han_style, _ = paired_han_style_aggregation(
                    aligned_semantics,
                    outs,
                    self.q_han_style_cross,
                    self.k_han_style_cross,
                )

            gamma = torch.sigmoid(self.gamma_logit)
            z_han_style = out_han + out_han_style
            aggregated[node_type] = (
                gamma * out_cross
                + (1.0 - gamma) * z_han_style
            )

        return aggregated, relation_outputs

    def message(
        self,
        x_j: Tensor,
        alpha_i: Tensor,
        alpha_j: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        alpha = F.leaky_relu(
            alpha_j + alpha_i,
            self.negative_slope,
        )
        alpha = softmax(
            alpha,
            index,
            ptr,
            size_i,
        )
        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training,
        )
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class CHSLayer(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        h_dim: int,
        heads: int,
        normalization: str,
        sinkhorn_iters: int,
        ablation: str,
    ):
        super().__init__()

        self.matching = MatchingMatrix(
            normalization=normalization,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.mlp = nn.ModuleDict({
            node_type: ResidualMLP(h_dim)
            for node_type in metadata[0]
        })
        self.conv = CHSHeteroConv(
            in_channels=h_dim,
            out_channels=h_dim,
            metadata=metadata,
            heads=heads,
            dropout=GAT_DROPOUT,
            ablation=ablation,
        )

    def forward(
        self,
        h_t: Dict[str, Tensor],
        h_q: Dict[str, Tensor],
        packed_mm: Dict[str, Tensor],
        target_edge_index_dict: Dict[EdgeType, Adj],
        query_edge_index_dict: Dict[EdgeType, Adj],
        matching_layouts: Dict[str, dict],
    ) -> Tuple[
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Tensor],
    ]:
        h_t_conved, semantic_t = self.conv(
            h_t,
            target_edge_index_dict,
            cross_dict=None,
            counterpart_dict=h_q,
            packed_mm=packed_mm,
            target_conv=True,
            matching_layouts=matching_layouts,
        )
        h_q_conved, _ = self.conv(
            h_q,
            query_edge_index_dict,
            cross_dict=semantic_t,
            counterpart_dict=h_t,
            packed_mm=packed_mm,
            target_conv=False,
            matching_layouts=matching_layouts,
        )

        new_h_t = {
            node_type: (
                self.mlp[node_type](h_t_conved[node_type])
                + h_t[node_type]
            )
            for node_type in h_t
        }
        new_h_q = {
            node_type: (
                self.mlp[node_type](h_q_conved[node_type])
                + h_q[node_type]
            )
            for node_type in h_q
        }

        new_packed_mm = self.matching(
            new_h_t,
            new_h_q,
            matching_layouts,
        )
        return new_h_t, new_h_q, new_packed_mm


class CHSMatch(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        num_layers: int,
        num_categories_dict: Dict[str, int],
        normalization: str = "partial_sinkhorn",
        sinkhorn_iters: int = 5,
        ablation: str = "full",
        h_dim: int = HIDDEN_DIM,
        heads: int = HEADS,
    ):
        super().__init__()

        self.projection = HeteroProjectLayer(
            list(metadata[0]),
            h_dim,
            num_categories_dict,
        )
        self.matching = MatchingMatrix(
            normalization=normalization,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.layers = nn.ModuleList([
            CHSLayer(
                metadata=metadata,
                h_dim=h_dim,
                heads=heads,
                normalization=normalization,
                sinkhorn_iters=sinkhorn_iters,
                ablation=ablation,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        target,
        query,
        matching_layouts: Dict[str, dict],
    ) -> List[Dict[str, Tensor]]:
        t_dict = {
            node_type: target[node_type].x
            for node_type in target.node_types
        }
        q_dict = {
            node_type: query[node_type].x
            for node_type in query.node_types
        }

        h_t = self.projection(t_dict)
        h_q = self.projection(q_dict)

        target_edges = {
            edge_type: target[edge_type].edge_index
            for edge_type in target.edge_types
        }
        query_edges = {
            edge_type: query[edge_type].edge_index
            for edge_type in query.edge_types
        }

        packed_mm = self.matching(
            h_t,
            h_q,
            matching_layouts,
        )
        matching_matrices: List[Dict[str, Tensor]] = []

        for layer in self.layers:
            h_t, h_q, packed_mm = layer(
                h_t,
                h_q,
                packed_mm,
                target_edges,
                query_edges,
                matching_layouts,
            )
            matching_matrices.append(packed_mm)

        return matching_matrices


class CHSMatchLoss(nn.Module):
    """Matching loss used by the published CHS-Match configuration."""

    def __init__(self, lambda_mm: float = 0.2):
        super().__init__()
        self.lambda_mm = float(lambda_mm)

    @staticmethod
    def _layer_loss(
        pred: Tensor,
        gt: Tensor,
        q_active: Tensor,
        t_active: Tensor,
    ) -> Tensor:
        active = (
            q_active.unsqueeze(2)
            & t_active.unsqueeze(1)
        )
        active_f = active.to(pred.dtype)
        gt_f = gt.to(pred.dtype)

        matched = (
            pred * gt_f * active_f
        ).sum(dim=2)
        nonmatched = (
            pred
            * (gt_f == 0).to(pred.dtype)
            * active_f
        ).sum(dim=2)

        u = 1.0 - (matched - nonmatched)
        u = u[q_active]
        return (
            torch.mean(u * u)
            if u.numel()
            else pred.new_zeros(())
        )

    def forward(
        self,
        matching_matrices: List[Dict[str, Tensor]],
        packed_gt: Dict[str, Tensor],
        matching_layouts: Dict[str, dict],
        query,
    ) -> Tensor:
        total_nodes = sum(
            query[node_type].x.size(0)
            for node_type in packed_gt
        )
        loss_total = next(
            iter(packed_gt.values())
        ).new_zeros(())

        for node_type, gt in packed_gt.items():
            layout = matching_layouts[node_type]
            preds = [
                layer[node_type]
                for layer in matching_matrices
            ]

            final_loss = self._layer_loss(
                preds[-1],
                gt,
                layout["q_active"],
                layout["t_active"],
            )

            if len(preds) > 1:
                intermediate_loss = torch.stack([
                    self._layer_loss(
                        pred,
                        gt,
                        layout["q_active"],
                        layout["t_active"],
                    )
                    for pred in preds[:-1]
                ]).mean()

                node_type_loss = (
                    self.lambda_mm * intermediate_loss
                    + (1.0 - self.lambda_mm) * final_loss
                )
            else:
                node_type_loss = final_loss

            loss_total = (
                loss_total
                + node_type_loss
                * (
                    query[node_type].x.size(0)
                    / total_nodes
                )
            )

        return loss_total
