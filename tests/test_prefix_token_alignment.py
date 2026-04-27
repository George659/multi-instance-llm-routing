"""任务 3：RAG 共享前缀 token 对齐合同测试。"""

from __future__ import annotations

from collections import defaultdict

from src.workload.generators import (
    VLLM_PREFIX_BLOCK_SIZE,
    generate_workload,
    render_messages_to_token_ids,
)
from tests.conftest import DummyTokenizer


def _rag_only_config(num_requests: int = 120) -> dict:
    return {
        "workload": {
            "num_requests": num_requests,
            "mix": {"chat": 0.0, "rag": 1.0, "agent": 0.0},
            "chat": {
                "prompt_tokens_min": 60,
                "prompt_tokens_max": 80,
                "max_tokens_min": 50,
                "max_tokens_max": 70,
                "min_tokens": None,
            },
            "rag": {
                "prompt_tokens_min": 180,
                "prompt_tokens_max": 180,
                "max_tokens_min": 100,
                "max_tokens_max": 100,
                "min_tokens": None,
                "shared_prefix_ratio": 1.0,
                "shared_prefix_tokens": 32,
                "num_prefix_groups": 4,
            },
            "agent": {
                "prompt_tokens_min": 100,
                "prompt_tokens_max": 140,
                "max_tokens_min": 140,
                "max_tokens_max": 180,
                "min_tokens": 120,
                "num_task_templates": 4,
            },
        }
    }


def test_same_prefix_group_requests_share_identical_token_prefix() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_rag_only_config(), tokenizer, seed=1234)

    groups: dict[int, list] = defaultdict(list)
    for request in requests:
        assert request.metadata.prefix_group_id is not None
        groups[request.metadata.prefix_group_id].append(request)

    # 文档要求同 group 的大量请求共享前 N 个 token；这里保证每组至少能采到多条。
    assert all(len(group_requests) >= 2 for group_requests in groups.values())

    for group_requests in groups.values():
        reference_ids = render_messages_to_token_ids(tokenizer, group_requests[0].messages)
        shared_prefix_tokens = group_requests[0].metadata.shared_prefix_tokens

        for request in group_requests[1:]:
            current_ids = render_messages_to_token_ids(tokenizer, request.messages)
            assert current_ids[:shared_prefix_tokens] == reference_ids[:shared_prefix_tokens]


def test_different_prefix_groups_do_not_share_the_same_group_prefix() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_rag_only_config(), tokenizer, seed=4321)

    first_request_by_group = {}
    for request in requests:
        group_id = request.metadata.prefix_group_id
        assert group_id is not None
        first_request_by_group.setdefault(group_id, request)

    group_ids = sorted(first_request_by_group)
    assert len(group_ids) >= 2

    left = first_request_by_group[group_ids[0]]
    right = first_request_by_group[group_ids[1]]
    shared_prefix_tokens = min(
        left.metadata.shared_prefix_tokens,
        right.metadata.shared_prefix_tokens,
    )

    left_ids = render_messages_to_token_ids(tokenizer, left.messages)
    right_ids = render_messages_to_token_ids(tokenizer, right.messages)
    assert left_ids[:shared_prefix_tokens] != right_ids[:shared_prefix_tokens]


def test_shared_prefix_token_count_is_block_aligned() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_rag_only_config(num_requests=32), tokenizer, seed=9)

    for request in requests:
        assert request.metadata.shared_prefix_tokens % VLLM_PREFIX_BLOCK_SIZE == 0
        assert request.metadata.shared_prefix_tokens >= VLLM_PREFIX_BLOCK_SIZE
