"""任务 7：runner 边界行为测试。"""

from __future__ import annotations

import asyncio

from src.benchmark.runner import _execute_batch
from src.workload.schema import Request, RequestMetadata, RequestResult


class DummyClient:
    async def send(
        self,
        request: Request,
        *,
        scheduled_at: float | None = None,
        stream: bool = True,
        include_usage: bool = True,
    ) -> RequestResult:
        del stream, include_usage
        result = RequestResult.from_request(request)
        result.scheduled_at = scheduled_at
        result.send_at = scheduled_at
        result.first_token_at = scheduled_at
        result.end_at = scheduled_at
        result.prompt_tokens = request.metadata.prompt_tokens_actual
        result.completion_tokens = 1
        result.total_tokens = result.prompt_tokens + 1
        result.token_arrival_times = [scheduled_at] if scheduled_at is not None else []
        return result


def _build_request(request_id: str, arrival_time_offset: float) -> Request:
    return Request(
        request_id=request_id,
        workload_type="chat",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        max_tokens=16,
        min_tokens=None,
        temperature=0.0,
        metadata=RequestMetadata(
            prompt_tokens_target=8,
            prompt_tokens_actual=8,
            prefix_group_id=None,
            shared_prefix_tokens=0,
            arrival_time_offset=arrival_time_offset,
            seed=1,
        ),
    )


def test_execute_batch_marks_pending_requests_as_timeout_failures() -> None:
    requests = [
        _build_request("req-000000", arrival_time_offset=0.0),
        _build_request("req-000001", arrival_time_offset=5.0),
    ]

    results = asyncio.run(
        _execute_batch(
            client=DummyClient(),
            requests=requests,
            run_duration_limit_sec=0.05,
            stream=True,
            include_usage=True,
        )
    )

    assert len(results) == 2
    assert [result.request_id for result in results] == ["req-000000", "req-000001"]

    success_result = results[0]
    timeout_result = results[1]

    assert success_result.status == "success"
    assert timeout_result.status == "failed"
    assert timeout_result.error_type == "timeout"
    assert timeout_result.error_message is not None
    assert timeout_result.scheduled_at is not None
    assert timeout_result.end_at is not None
