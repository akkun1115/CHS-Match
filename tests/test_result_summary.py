"""Tests for the CHS-Match main-run result summarizer (no GPU required)."""
import json

from scripts.summarize_results import MAIN_RUNS, summarize


def make_result(dataset, seed):
    return {
        "dataset": dataset,
        "seed": seed,
        "layers": MAIN_RUNS[dataset]["layers"],
        "hop": 1,
        "normalization": "partial_sinkhorn",
        "sinkhorn_iters": 5,
        "ablation": "full",
        "lambda_mm": 0.2,
        "epochs": 500,
        "batch_size": 32,
        "hidden_dim": 64,
        "heads": 4,
        "learning_rate": 1e-3,
        "test_acc": MAIN_RUNS[dataset]["paper"][0] / 100.0,
    }


def test_summary_requires_all_nine_main_runs(tmp_path, capsys):
    for dataset, config in MAIN_RUNS.items():
        for seed in config["seeds"]:
            (tmp_path / f"{dataset}-{seed}.json").write_text(
                json.dumps(make_result(dataset, seed)), encoding="utf-8"
            )
    assert summarize(tmp_path, require_complete=True) == 0
    captured = capsys.readouterr().out
    assert "LastFM" in captured and "IMDB" in captured and "DBLP" in captured

    (tmp_path / "DBLP-88.json").unlink()
    assert summarize(tmp_path, require_complete=True) == 2


def test_summary_refuses_ambiguous_duplicate_runs(tmp_path):
    result = make_result("LastFM", 167)
    for file in ("first.json", "second.json"):
        (tmp_path / file).write_text(json.dumps(result), encoding="utf-8")
    assert summarize(tmp_path) == 2
