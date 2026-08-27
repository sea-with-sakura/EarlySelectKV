import json

import torch

from analysis.es_infllmv2.es_infllmv2 import (
    ESInfLLMV2Config,
    _block_mask_to_token_mask,
    _forced_block_mask,
    config_from_sparse_config,
)
from analysis.es_infllmv2.summarize_compatibility import summarize


def test_infllmv2_sparse_config_is_loaded_from_checkpoint_values():
    cfg = config_from_sparse_config(
        {
            "kernel_size": 48,
            "kernel_stride": 12,
            "init_blocks": 2,
            "block_size": 32,
            "window_size": 1024,
            "topk": 80,
            "dense_len": 4096,
        },
        mode="oracle_sparse",
    )
    assert cfg.mode == "oracle_sparse"
    assert cfg.kernel_size == 48
    assert cfg.kernel_stride == 12
    assert cfg.init_blocks == 2
    assert cfg.block_size == 32
    assert cfg.window_size == 1024
    assert cfg.remote_topk == 80
    assert cfg.dense_len == 4096


def test_infllmv2_forced_mask_keeps_initial_and_causal_local_blocks():
    cfg = ESInfLLMV2Config(block_size=64, window_size=128, init_blocks=1)
    mask = _forced_block_mask(
        batch_size=1,
        kv_heads=1,
        q_len=1,
        kv_seq_len=256,
        query_positions=torch.tensor([[130]]),
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 1, 4)
    assert mask.flatten().tolist() == [True, True, True, False]


def test_infllmv2_block_mask_expands_to_gqa_token_mask():
    blocks = torch.tensor([[[[True, False, True]]]])
    tokens = _block_mask_to_token_mask(
        blocks,
        kv_seq_len=5,
        block_size=2,
        num_key_value_groups=2,
    )
    assert tokens.shape == (1, 2, 1, 5)
    assert tokens[0, 0, 0].tolist() == [True, True, False, False, True]
    assert torch.equal(tokens[:, 0], tokens[:, 1])


def test_infllmv2_summary_uses_archived_scores_and_sparse_threshold(tmp_path):
    dataset = "gov_report"
    rank_dir = tmp_path / "rank256" / dataset
    exact_dir = tmp_path / "exact_wq" / dataset
    rank_dir.mkdir(parents=True)
    exact_dir.mkdir(parents=True)
    rank_summary = {
        "sample_count": 3,
        "sparse_config": {"dense_len": 8},
        "modes": {
            "oracle_sparse": {"score": 25.21, "metric": "rouge_score"},
            "es_sparse": {"score": 24.68, "metric": "rouge_score"},
        },
    }
    exact_summary = {"modes": {"es_sparse": {"score": 25.91, "metric": "rouge_score"}}}
    (rank_dir / "summary.json").write_text(json.dumps(rank_summary), encoding="utf-8")
    (exact_dir / "summary.json").write_text(json.dumps(exact_summary), encoding="utf-8")
    predictions = [{"input_tokens": 7}, {"input_tokens": 8}, {"input_tokens": 9}]
    (rank_dir / f"{dataset}_oracle_sparse.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )

    result = summarize(tmp_path, [dataset])

    assert result["rows"][0]["sparse_count"] == 2
    assert result["mean_exact_wq_minus_real_q"] == 0.7
    assert result["mean_rank256_minus_real_q"] == -0.53
