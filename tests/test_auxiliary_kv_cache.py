import torch

from pipeline.auxiliary_kv_cache import ChunkSummaryCache
from pipeline.lookaheadkv.runtime import LookaheadKVConfig, LookaheadKVRuntime
from pipeline.rocketkv.runtime import RocketKVConfig, RocketKVRuntime
from pipeline.runtime_decode_utils import compress_standard_prompt_kv


def test_chunk_summary_builds_and_updates_min_max_pages():
    k = torch.tensor(
        [[[[1.0, 4.0], [-2.0, 3.0], [5.0, -1.0], [0.0, 7.0], [2.0, 2.0]]]]
    )
    cache = ChunkSummaryCache(chunk_size=2)
    cache.build_layer(0, k, capacity_len=8)

    summary = cache.get(0, key_len=8)
    assert summary is not None
    expected_min = torch.tensor([[-2.0, 3.0], [0.0, -1.0], [2.0, 2.0], [float("inf"), float("inf")]])
    expected_max = torch.tensor([[1.0, 4.0], [5.0, 7.0], [2.0, 2.0], [-float("inf"), -float("inf")]])
    torch.testing.assert_close(summary[0, 0, :, :2], expected_min)
    torch.testing.assert_close(summary[0, 0, :, 2:], expected_max)

    cache.update_layer(0, cache_input_pos=torch.tensor([5]), k_update=torch.tensor([[[[-3.0, 10.0]]]]))
    summary = cache.get(0, key_len=6)
    assert summary is not None
    torch.testing.assert_close(summary[0, 0, 2, :2], torch.tensor([-3.0, 2.0]))
    torch.testing.assert_close(summary[0, 0, 2, 2:], torch.tensor([2.0, 10.0]))

    cache.update_layer(0, cache_input_pos=torch.tensor([6]), k_update=torch.tensor([[[[9.0, -4.0]]]]))
    summary = cache.get(0, key_len=7)
    assert summary is not None
    torch.testing.assert_close(summary[0, 0, 3, :2], torch.tensor([9.0, -4.0]))
    torch.testing.assert_close(summary[0, 0, 3, 2:], torch.tensor([9.0, -4.0]))


def test_chunk_size_one_summary_tracks_tokens_without_page_minmax():
    k = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    cache = ChunkSummaryCache(chunk_size=1)
    cache.build_layer(0, k, capacity_len=5)

    summary = cache.get(0, key_len=5)
    assert summary is not None
    torch.testing.assert_close(summary[:, :, :3, :], k)
    torch.testing.assert_close(summary[:, :, 3:, :], torch.zeros(1, 1, 2, 4))

    update = torch.tensor([[[[100.0, 101.0, 102.0, 103.0]]]])
    cache.update_layer(0, cache_input_pos=torch.tensor([3]), k_update=update)
    summary = cache.get(0, key_len=4)
    assert summary is not None
    torch.testing.assert_close(summary[:, :, 3:4, :], update)


def test_prompt_compression_score_chunking_matches_full_observation_window():
    torch.manual_seed(5)
    q = torch.randn(1, 4, 6, 8)
    k = torch.randn(1, 2, 13, 8)
    v = torch.randn(1, 2, 13, 8)

    full = compress_standard_prompt_kv(
        q=q,
        k=k,
        v=v,
        key_len=13,
        active_prompt_len=7,
        window_size=4,
        kernel_size=3,
        attention_heads=4,
        kv_heads=2,
        score_chunk_size=64,
    )
    chunked = compress_standard_prompt_kv(
        q=q,
        k=k,
        v=v,
        key_len=13,
        active_prompt_len=7,
        window_size=4,
        kernel_size=3,
        attention_heads=4,
        kv_heads=2,
        score_chunk_size=1,
    )

    torch.testing.assert_close(chunked.prompt_k, full.prompt_k)
    torch.testing.assert_close(chunked.prompt_v, full.prompt_v)
    assert torch.equal(chunked.keep_idx, full.keep_idx)


def test_rocketkv_hsa_aux_route_matches_online_summary_fallback():
    torch.manual_seed(7)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 11, 8)
    cfg = RocketKVConfig(
        prompt_len=11,
        prompt_budget=11,
        topk_budget=4,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="hsa",
    )

    runtime_with_aux = RocketKVRuntime(cfg)
    runtime_with_aux.aux_cache.build_layer(0, k, capacity_len=11)
    aux_idx = runtime_with_aux._select_hsa_indices(layer_idx=0, q=q, k=k)

    runtime_online = RocketKVRuntime(cfg)
    online_idx = runtime_online._select_hsa_indices(layer_idx=0, q=q, k=k)

    assert torch.equal(aux_idx, online_idx)


def test_lookahead_hsa_aux_route_matches_online_summary_fallback():
    torch.manual_seed(11)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 13, 8)
    cfg = LookaheadKVConfig(
        prompt_len=13,
        prompt_budget=13,
        topk_budget=5,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="hsa",
    )

    runtime_with_aux = LookaheadKVRuntime(cfg)
    runtime_with_aux.aux_cache.build_layer(0, k, capacity_len=13)
    aux_route = runtime_with_aux._compute_hsa_route(layer_idx=0, q=q, k=k)

    runtime_online = LookaheadKVRuntime(cfg)
    online_route = runtime_online._compute_hsa_route(layer_idx=0, q=q, k=k)

    assert torch.equal(aux_route["indices"], online_route["indices"])


def test_lookahead_local_does_not_maintain_page_summary_cache():
    cfg = LookaheadKVConfig(
        prompt_len=13,
        prompt_budget=13,
        topk_budget=5,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="chunk1_local",
    )

    assert not LookaheadKVRuntime(cfg).uses_aux_cache


def test_lookahead_hsa_local_maintains_page_summary_cache():
    cfg = LookaheadKVConfig(
        prompt_len=13,
        prompt_budget=13,
        topk_budget=5,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="hsa_local",
    )

    runtime = LookaheadKVRuntime(cfg)
    assert runtime.uses_aux_cache
    assert runtime.chunk_size == 2


def test_lookahead_local_route_appends_local_tail_outside_prefix_route():
    torch.manual_seed(13)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 10, 8)
    cfg = LookaheadKVConfig(
        prompt_len=10,
        prompt_budget=10,
        topk_budget=5,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="hsa_local",
        decode_local_budget=3,
    )
    runtime = LookaheadKVRuntime(cfg)

    route = runtime._compute_local_route(q=q, k=k, route_chunk_size=runtime.chunk_size)
    indices = route["indices"]

    assert indices.shape == (1, 2, 1, 5)
    torch.testing.assert_close(indices[..., -3:], torch.tensor([[[[7, 8, 9]], [[7, 8, 9]]]]))
    assert torch.all(indices[..., :-3] < 7)


def test_lookahead_hsa_local_aux_route_matches_online_prefix_with_partial_page():
    torch.manual_seed(17)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 13, 8)
    k[:, :, 11:, :] = 1000.0
    cfg = LookaheadKVConfig(
        prompt_len=13,
        prompt_budget=13,
        topk_budget=5,
        compression_ratio=4.0,
        window_size=4,
        kernel_size=3,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        routing_mode="hsa_local",
        decode_local_budget=2,
    )

    runtime_with_aux = LookaheadKVRuntime(cfg)
    runtime_with_aux.aux_cache.build_layer(0, k, capacity_len=13)
    aux_route = runtime_with_aux._compute_local_route(
        layer_idx=0,
        q=q,
        k=k,
        route_chunk_size=runtime_with_aux.chunk_size,
    )

    runtime_online = LookaheadKVRuntime(cfg)
    online_route = runtime_online._compute_local_route(
        layer_idx=0,
        q=q,
        k=k,
        route_chunk_size=runtime_online.chunk_size,
    )

    assert torch.equal(aux_route["indices"], online_route["indices"])
    assert torch.all(aux_route["indices"][..., :-2] < 11)
    torch.testing.assert_close(aux_route["indices"][..., -2:], torch.tensor([[[[11, 12]], [[11, 12]]]]))
