import torch
import torch.nn as nn
from litgpt.config import Config
from litgpt.model import GPT

import pipeline.litgpt_prefill as litgpt_prefill
from pipeline.litgpt_prefill import (
    _build_causal_mask_for_input_pos,
    _iter_chunk_ranges,
    prefill_last_token_logits_layerwise,
    setup_standard_kv_cache,
)
from pipeline.lookaheadkv.runtime import LookaheadKVConfig, LookaheadKVRuntime
from pipeline.rocketkv.runtime import RocketKVConfig, RocketKVRuntime


def test_iter_chunk_ranges_avoids_singleton_tail():
    assert list(_iter_chunk_ranges(4097, 4096)) == [(0, 4095), (4095, 4097)]
    assert list(_iter_chunk_ranges(8193, 4096)) == [(0, 4096), (4096, 8191), (8191, 8193)]
    assert list(_iter_chunk_ranges(2, 1)) == [(0, 1), (1, 2)]


def test_standard_chunked_prefill_avoids_singleton_tail(monkeypatch):
    chunk_lengths = []

    class DummyModel:
        transformer = type("Transformer", (), {"ln_f": nn.Identity()})()
        config = type("Config", (), {"final_logit_softcapping": None})()
        lm_head = nn.Identity()

    def fake_forward_prefill_chunk(base_model, *, idx, input_pos, input_pos_maxp1):
        del base_model, input_pos, input_pos_maxp1
        chunk_lengths.append(int(idx.size(1)))
        return idx.unsqueeze(-1).to(dtype=torch.float32)

    monkeypatch.setattr(litgpt_prefill, "_forward_prefill_chunk", fake_forward_prefill_chunk)
    idx = torch.arange(9).view(1, -1)

    logits = litgpt_prefill.prefill_last_token_logits(
        DummyModel(),
        idx,
        torch.arange(9),
        input_pos_maxp1=9,
        chunk_size=4,
    )

    assert chunk_lengths == [4, 3, 2]
    assert logits.item() == 8.0


def test_prefill_chunk_mask_uses_lower_right_causal_bias():
    input_pos = torch.arange(3, 7)

    mask = _build_causal_mask_for_input_pos(input_pos, input_pos_maxp1=7)

    expected = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True],
        ]
    )
    actual = mask._materialize() if hasattr(mask, "_materialize") else mask[0, 0]
    assert torch.equal(actual, expected)


def test_prefill_chunk_mask_handles_non_contiguous_positions():
    input_pos = torch.tensor([0, 2, 4])

    mask = _build_causal_mask_for_input_pos(input_pos, input_pos_maxp1=5)

    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(mask[0, 0], expected)


def test_layerwise_chunked_prefill_matches_litgpt_last_token_logits():
    torch.manual_seed(23)
    cfg = Config(
        block_size=16,
        vocab_size=64,
        padded_vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        rotary_percentage=1.0,
    )
    model = GPT(cfg).eval()
    idx = torch.randint(0, 64, (1, 9))
    input_pos = torch.arange(9)

    setup_standard_kv_cache(
        model,
        logical_max_seq_length=9,
        cache_max_seq_length=9,
        device=torch.device("cpu"),
        build_mask=True,
    )
    expected = model(idx, input_pos, input_pos_maxp1=9)[:, -1:, :].detach()
    model.clear_kv_cache()

    actual = prefill_last_token_logits_layerwise(
        model,
        idx,
        input_pos,
        input_pos_maxp1=9,
        chunk_size=4,
    ).detach()

    torch.testing.assert_close(actual, expected)


def test_public_sparse_runtimes_compress_single_token_prefill_tail():
    torch.manual_seed(31)
    q = torch.randn(1, 8, 1, 4)
    k = torch.randn(1, 2, 17, 4)
    v = torch.randn(1, 2, 17, 4)
    common = {
        "prompt_len": 17,
        "prompt_budget": 8,
        "topk_budget": 4,
        "compression_ratio": 2.0,
        "window_size": 4,
        "kernel_size": 3,
        "skip_layers": 0,
        "attention_heads": 8,
        "kv_heads": 2,
        "routing_mode": "hsa",
    }
    runtimes = [
        RocketKVRuntime(RocketKVConfig(**common)),
        LookaheadKVRuntime(LookaheadKVConfig(**common)),
    ]

    for runtime in runtimes:
        runtime.maybe_compress_prefill(
            layer_idx=0,
            attn_module=None,
            q=q,
            k=k,
            v=v,
            input_pos_maxp1=17,
        )

        assert runtime.prompt_k_by_layer[0].shape == (1, 2, runtime.active_prompt_len, 4)
        assert runtime.prompt_v_by_layer[0].shape == (1, 2, runtime.active_prompt_len, 4)
