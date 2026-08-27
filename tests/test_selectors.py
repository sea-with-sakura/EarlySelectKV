import torch

from pipeline.quest.runtime import QuestConfig, QuestRuntime
from pipeline.topk.runtime import TopKConfig, TopKRuntime


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
