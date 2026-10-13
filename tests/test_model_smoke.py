"""Small CPU functional tests, not a claim of reproducing paper results."""
import torch
from torch_geometric.data import HeteroData

from chs_match.data import collate_packed
from chs_match.model import CHSMatch, CHSMatchLoss


EDGE_TYPE = ("node", "rel", "node")


def make_pair(query_nodes: int, target_nodes: int):
    assert 0 < query_nodes <= target_nodes

    def make_graph(num_nodes: int):
        graph = HeteroData()
        graph["node"].x = (
            torch.arange(num_nodes, dtype=torch.float).remainder(3).view(-1, 1)
        )
        src = torch.arange(num_nodes, dtype=torch.long)
        dst = (src + 1).remainder(num_nodes)
        graph[EDGE_TYPE].edge_index = torch.stack([src, dst], dim=0)
        return graph

    target = make_graph(target_nodes)
    query = make_graph(query_nodes)
    ground_truth = {
        "node": torch.eye(query_nodes, target_nodes, dtype=torch.bool)
    }
    return target, query, ground_truth


def test_forward_backward_for_both_normalizations():
    torch.manual_seed(123)
    target, query, gt, layouts = collate_packed(
        [make_pair(2, 3), make_pair(1, 4)]
    )
    for normalization in ("partial_sinkhorn", "row_softmax"):
        model = CHSMatch(
            metadata=(["node"], [EDGE_TYPE]),
            num_layers=2,
            num_categories_dict={"node": 3},
            normalization=normalization,
            sinkhorn_iters=5,
            h_dim=16,
            heads=4,
        )
        prediction = model(target, query, layouts)
        assert len(prediction) == 2
        weights = prediction[-1]["node"]
        assert weights.shape == (2, 2, 4)
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()
        assert (weights[~layouts["node"]["real_active"]] == 0).all()

        loss = CHSMatchLoss(lambda_mm=0.2)(
            prediction, gt, layouts, query
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
        )


def test_normalization_does_not_assign_to_padded_targets():
    target, query, _, layouts = collate_packed(
        [make_pair(2, 3), make_pair(1, 4)]
    )
    model = CHSMatch(
        metadata=(["node"], [EDGE_TYPE]),
        num_layers=1,
        num_categories_dict={"node": 3},
        normalization="row_softmax",
        h_dim=16,
        heads=4,
    )
    model.eval()
    with torch.no_grad():
        weight = model(target, query, layouts)[-1]["node"]
    active = layouts["node"]["q_active"]
    row_sums = weight.sum(dim=2)
    torch.testing.assert_close(row_sums[active], torch.ones_like(row_sums[active]))
    assert (weight[~layouts["node"]["real_active"]] == 0).all()
