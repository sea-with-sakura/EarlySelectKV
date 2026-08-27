from types import SimpleNamespace

import torch

from pipeline.lookahead_loki.runtime import LookaheadLokiConfig, LookaheadLokiRuntime


class _DummyLookaheadLokiRuntime(LookaheadLokiRuntime):
    def _approx_route_scores(self, *, attn_self, layer_idx, q, k):
        del attn_self, layer_idx, q
        scores = torch.arange(k.size(2), device=k.device, dtype=torch.float32)
        return scores.view(1, 1, 1, k.size(2)).expand(k.size(0), self.cfg.kv_heads, 1, -1)


def test_lookahead_loki_grouped_exact_output():
    cfg = LookaheadLokiConfig(
        prompt_len=4,
        token_budget=3,
        rank=2,
        skip_layers=0,
        attention_heads=4,
        kv_heads=2,
        attention_scores_scalar=None,
        head_size=8,
        pca_dir=None,
        use_real_q_fallback=True,
    )
    runtime = _DummyLookaheadLokiRuntime(cfg)
    attn = SimpleNamespace(
        mscale=1.0,
        config=SimpleNamespace(attention_logit_softcapping=None),
    )
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    cos = torch.empty(0)
    sin = torch.empty(0)

    out = runtime.compute_attention_output(
        attn_self=attn,
        layer_idx=1,
        q=q,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        rope_n_elem=0,
        rope_interleave=False,
    )

    assert out is not None
    assert out.shape == (1, 1, 4, 8)
