"""Exercise the benchmark loader and CHS-Match on locally created toy data.

Does not require redistributing upstream LastFM, IMDB or DBLP tensors.
"""
import pytest
import torch
from torch_geometric.data import HeteroData

from chs_match.data import CHSMatchDataset, collate_packed
from chs_match.model import CHSMatch, CHSMatchLoss


EDGE = ("node", "edge", "node")


def make_pair(query_nodes, target_nodes):
    def graph(count):
        item = HeteroData()
        item["node"].x = torch.arange(count, dtype=torch.float32).remainder(5).view(-1, 1)
        src = torch.arange(count, dtype=torch.long)
        dst = (src + 1).remainder(count)
        item[EDGE].edge_index = torch.stack([src, dst])
        return item

    gt = {"node": torch.eye(query_nodes, target_nodes, dtype=torch.bool)}
    return graph(target_nodes), graph(query_nodes), gt


@pytest.mark.parametrize("query_nodes,target_nodes", [(1, 2), (2, 3), (3, 5)])
def test_locally_created_benchmark_format_and_train_step(tmp_path, query_nodes, target_nodes):
    pair = make_pair(query_nodes, target_nodes)
    local_file = tmp_path / "local_generated_benchmark.pt"
    torch.save([[pair[0]], [pair[1]], [pair[2]]], local_file)
    dataset = CHSMatchDataset(str(local_file))
    assert len(dataset) == 1

    target, query, gt, layouts = collate_packed([dataset[0]])
    model = CHSMatch(
        metadata=(["node"], [EDGE]),
        num_layers=1,
        num_categories_dict={"node": 5},
        normalization="partial_sinkhorn",
        sinkhorn_iters=5,
        h_dim=16,
        heads=4,
    )
    predictions = model(target, query, layouts)
    loss = CHSMatchLoss(0.2)(predictions, gt, layouts, query)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
