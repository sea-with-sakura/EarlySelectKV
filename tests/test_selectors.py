import torch

from pipeline.decode_attention import grouped_decode_attention
from pipeline.quest.runtime import QuestConfig, QuestRuntime
from pipeline.topk.runtime import TopKConfig, TopKRuntime


class _MaterializedMask:
    def __init__(self, value: torch.Tensor):
        self.value = value

    def _materialize(self) -> torch.Tensor:
        return self.value


def test_grouped_decode_attention_normalizes_batched_mask():
    torch.manual_seed(17)
    config = type(
        "Config",
        (),
        {
            "attention_scores_scalar": None,
            "head_size": 3,
            "attention_logit_softcapping": None,
        },
    )()
    attention = type("Attention", (), {"config": config, "mscale": 1.0})()
    q = torch.randn(2, 4, 2, 3)
    k = torch.randn(2, 2, 3, 3)
    v = torch.randn(2, 2, 3, 3)
    batched_mask = torch.tensor(
        [
            [[True, False, True], [False, True, True]],
            [[False, True, True], [True, True, False]],
        ]
    )
    canonical_mask = batched_mask.unsqueeze(1).unsqueeze(2)

    expected = grouped_decode_attention(attention, q, k, v, canonical_mask)
    actual = grouped_decode_attention(attention, q, k, v, _MaterializedMask(batched_mask))

    torch.testing.assert_close(actual, expected)


def test_exact_topk_selects_largest_dot_products():
    runtime = TopKRuntime(
        TopKConfig(
            prompt_len=4,
            topk_budget=2,
            group_shared_within_kv_group=True,
            skip_layers=0,
            attention_heads=1,
            kv_heads=1,
            attention_scores_scalar=None,
            head_size=2,
        )
    )
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[0.1, 0.0], [0.9, 0.0], [0.3, 0.0], [0.8, 0.0]]]])

    selection = runtime.select_attention_indices(layer_idx=0, q=q, k=k)

    assert selection is not None
    assert selection.indices.tolist() == [[[[1, 3]]]]


def test_quest_always_includes_the_last_partial_page():
    runtime = QuestRuntime(
        QuestConfig(
            prompt_len=5,
            token_budget=4,
            page_size=2,
            skip_layers=0,
            attention_heads=1,
            kv_heads=1,
            min_select_pages=1,
            include_last_page=True,
            group_shared_within_kv_group=True,
        )
    )
    q = torch.tensor([[[[1.0]]]])
    k = torch.tensor([[[[0.1], [0.1], [0.9], [0.8], [0.0]]]])

    selection = runtime.select_attention_indices(layer_idx=0, q=q, k=k)

    assert selection is not None
    assert selection.indices.tolist() == [[[[2, 3, 4]]]]
