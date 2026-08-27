from types import SimpleNamespace

from pipeline.config_utils import load_json_with_extends
from pipeline.lookaheadkv.inference import _build_runtime
from pipeline.lookaheadkv.runtime import DEFAULT_DECODE_LOCAL_WINDOW_SIZE


def _dummy_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            n_head=8,
            n_query_groups=2,
        )
    )


def test_earlyselectkv_in_mid_keeps_original_hsa_routing():
    params = {
        "method": "earlyselectkv_in_mid",
        "token_budget": 256,
        "earlyselectkv_in_mid": {
            "lookahead_source": "in_mid",
            "lookahead_mid_until_layer": 20,
            "window_size": 32,
        },
    }
    runtime = _build_runtime(_dummy_model(), params, prompt_len=1024, max_new_tokens=64)

    assert runtime.cfg.routing_mode == "hsa"
    assert runtime.cfg.lookahead_source == "in_mid"
    assert not runtime.uses_local_route


def test_earlyselectkv_in_mid_local_uses_hsa_local_routing_budget():
    params = {
        "method": "earlyselectkv_in_mid_local",
        "token_budget": 256,
        "earlyselectkv_in_mid": {
            "lookahead_source": "in_mid",
            "lookahead_mid_until_layer": 20,
            "window_size": 32,
        },
        "earlyselectkv_in_mid_local": {
            "decode_local_budget": None,
        },
    }
    runtime = _build_runtime(_dummy_model(), params, prompt_len=1024, max_new_tokens=64)

    assert runtime.cfg.routing_mode == "hsa_local"
    assert runtime.cfg.lookahead_source == "in_mid"
    assert runtime.uses_local_route

    topk_budget = int(runtime.cfg.topk_budget)
    expected_local = min(DEFAULT_DECODE_LOCAL_WINDOW_SIZE, max(16, topk_budget // 8))
    assert runtime._resolve_decode_local_budget(key_len=512) == expected_local


def test_earlyselectkv_in_mid_local_budget_matches_hsa_local():
    common = {
        "token_budget": 512,
        "earlyselectkv_in_mid": {
            "lookahead_source": "in_mid",
            "lookahead_mid_until_layer": 20,
            "window_size": 64,
        },
        "earlyselectkv_in_mid_local": {
            "decode_local_budget": None,
        },
        "earlyselectkv_hsa_local": {
            "window_size": 64,
        },
    }
    in_mid = _build_runtime(
        _dummy_model(),
        {"method": "earlyselectkv_in_mid_local", **common},
        prompt_len=2048,
        max_new_tokens=128,
    )
    hsa_local = _build_runtime(
        _dummy_model(),
        {"method": "earlyselectkv_hsa_local", **common},
        prompt_len=2048,
        max_new_tokens=128,
    )

    assert in_mid.cfg.routing_mode == hsa_local.cfg.routing_mode == "hsa_local"
    assert in_mid.cfg.topk_budget == hsa_local.cfg.topk_budget
    assert in_mid._resolve_decode_local_budget(1024) == hsa_local._resolve_decode_local_budget(1024)


def test_earlyselectkv_in_mid_local_keeps_prefill_window_separate_from_fixed_decode_local_cap():
    params = {
        "method": "earlyselectkv_in_mid_local",
        "token_budget": 512,
        "earlyselectkv_in_mid": {
            "lookahead_source": "in_mid",
            "lookahead_mid_until_layer": 20,
            "window_size": 128,
        },
        "earlyselectkv_in_mid_local": {"decode_local_budget": None},
    }

    runtime = _build_runtime(_dummy_model(), params, prompt_len=2048, max_new_tokens=128)

    assert runtime.cfg.window_size == 128
    assert runtime._resolve_decode_local_budget(key_len=1024) == 32


def test_aliases_read_earlyselectkv_config():
    for method in ["lookaheadkv_hsa_local", "EarlySelectKV_hsa_local"]:
        params = {
            "method": method,
            "token_budget": 512,
            "earlyselectkv_hsa_local": {
                "window_size": 64,
                "decode_local_budget": None,
            },
        }

        runtime = _build_runtime(_dummy_model(), params, prompt_len=2048, max_new_tokens=128)

        assert runtime.cfg.routing_mode == "hsa_local"
        assert runtime.cfg.window_size == 64


def test_low_rank_proxy_method_uses_svd_projection():
    params = load_json_with_extends("config/pipeline_config/longbench/mistral-7b-instruct-v0.2.json")[
        "pipeline_params"
    ]
    params["method"] = "earlyselectkv_mid_svd_r256"
    params["token_budget"] = 512

    runtime = _build_runtime(_dummy_model(), params, prompt_len=2048, max_new_tokens=128)

    assert runtime.cfg.lookahead_query_mode == "svd_wq"
    assert runtime.cfg.lookahead_source == "mid"
    assert runtime.cfg.lookahead_svd_path.endswith("wq_svd_rank256.pth")
