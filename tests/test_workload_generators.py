"""任务 3：三类 workload generator 的通用行为测试。"""

from __future__ import annotations

from collections import Counter

from src.workload.generators import generate_workload
from tests.conftest import DummyTokenizer


def _mixed_workload_config(num_requests: int = 24) -> dict:
    return {
        "workload": {
            "num_requests": num_requests,
            "mix": {"chat": 0.4, "rag": 0.35, "agent": 0.25},
            "chat": {
                "prompt_tokens_min": 60,
                "prompt_tokens_max": 80,
                "max_tokens_min": 50,
                "max_tokens_max": 70,
                "min_tokens": None,
            },
            "rag": {
                "prompt_tokens_min": 140,
                "prompt_tokens_max": 180,
                "max_tokens_min": 90,
                "max_tokens_max": 110,
                "min_tokens": None,
                "shared_prefix_ratio": 0.6,
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


def _single_type_config(workload_type: str, num_requests: int = 8) -> dict:
    config = _mixed_workload_config(num_requests=num_requests)
    config["workload"]["mix"] = {
        "chat": 1.0 if workload_type == "chat" else 0.0,
        "rag": 1.0 if workload_type == "rag" else 0.0,
        "agent": 1.0 if workload_type == "agent" else 0.0,
    }
    return config


def test_generate_workload_returns_expected_count_and_ids() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_mixed_workload_config(num_requests=18), tokenizer, seed=7)

    assert len(requests) == 18
    assert requests[0].request_id == "req-000000"
    assert requests[-1].request_id == "req-000017"
    assert len({request.request_id for request in requests}) == 18


def test_generate_workload_covers_all_three_types_with_mixed_config() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_mixed_workload_config(num_requests=40), tokenizer, seed=42)
    counts = Counter(request.workload_type for request in requests)

    assert counts["chat"] > 0
    assert counts["rag"] > 0
    assert counts["agent"] > 0


def test_generated_requests_have_consistent_basic_metadata() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_mixed_workload_config(num_requests=20), tokenizer, seed=11)

    for request in requests:
        assert request.metadata.prompt_tokens_target > 0
        assert request.metadata.prompt_tokens_actual > 0
        assert request.metadata.arrival_time_offset == 0.0
        assert request.max_tokens > 0
        assert request.min_tokens is None or request.min_tokens <= request.max_tokens
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[1]["role"] == "user"


def test_chat_requests_do_not_use_shared_prefix() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_single_type_config("chat"), tokenizer, seed=101)

    assert requests
    assert all(request.workload_type == "chat" for request in requests)
    assert all(request.metadata.prefix_group_id is None for request in requests)
    assert all(request.metadata.shared_prefix_tokens == 0 for request in requests)
    assert all(request.min_tokens is None for request in requests)


def test_rag_requests_can_emit_shared_prefix_groups() -> None:
    tokenizer = DummyTokenizer()
    config = _single_type_config("rag", num_requests=24)
    config["workload"]["rag"]["shared_prefix_ratio"] = 1.0
    requests = generate_workload(config, tokenizer, seed=202)

    assert requests
    assert all(request.workload_type == "rag" for request in requests)
    assert all(request.metadata.prefix_group_id is not None for request in requests)
    assert all(request.metadata.shared_prefix_tokens > 0 for request in requests)


def test_agent_requests_respect_min_tokens_configuration() -> None:
    tokenizer = DummyTokenizer()
    requests = generate_workload(_single_type_config("agent"), tokenizer, seed=303)

    assert requests
    assert all(request.workload_type == "agent" for request in requests)
    assert all(request.min_tokens == 120 for request in requests)
    for request in requests:
        assert request.min_tokens is not None
        assert request.max_tokens >= request.min_tokens


def test_generation_is_deterministic_for_same_seed() -> None:
    tokenizer_a = DummyTokenizer()
    tokenizer_b = DummyTokenizer()
    config = _mixed_workload_config(num_requests=12)

    run_a = generate_workload(config, tokenizer_a, seed=999)
    run_b = generate_workload(config, tokenizer_b, seed=999)

    snapshot_a = [
        (
            request.request_id,
            request.workload_type,
            request.max_tokens,
            request.min_tokens,
            request.metadata.prompt_tokens_target,
            request.metadata.prompt_tokens_actual,
            request.metadata.prefix_group_id,
            request.metadata.shared_prefix_tokens,
            request.messages,
        )
        for request in run_a
    ]
    snapshot_b = [
        (
            request.request_id,
            request.workload_type,
            request.max_tokens,
            request.min_tokens,
            request.metadata.prompt_tokens_target,
            request.metadata.prompt_tokens_actual,
            request.metadata.prefix_group_id,
            request.metadata.shared_prefix_tokens,
            request.messages,
        )
        for request in run_b
    ]

    assert snapshot_a == snapshot_b
