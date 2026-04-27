"""Step 2 的异步 streaming benchmark client。

这个 client 的职责非常单一：

1. 把 `Request` 转成 OpenAI-compatible payload；
2. 通过 HTTP streaming 方式发送请求；
3. 逐个解析 SSE chunk，并记录时间戳、文本内容和 usage；
4. 无论成功还是失败，都返回一个结构化的 `RequestResult`。

这里刻意不绑定 vLLM 内部实现，只假设目标服务兼容 OpenAI chat completions
的 streaming 接口。后续把 `api_base` 从本地 vLLM 换成 Router 地址时，
benchmark 侧代码不需要改。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from src.workload.schema import ErrorType, Request, RequestResult


class StreamingChatClient:
    """基于 `httpx.AsyncClient` 的异步 streaming client。"""

    def __init__(self, api_base: str, path: str, timeout: float, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.timeout = timeout
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            timeout=httpx.Timeout(timeout),
        )

    async def __aenter__(self) -> "StreamingChatClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """显式关闭底层 HTTP 连接池。"""

        await self._client.aclose()

    async def send(
        self,
        request: Request,
        *,
        scheduled_at: float | None = None,
        stream: bool = True,
        include_usage: bool = True,
    ) -> RequestResult:
        """发送一个 streaming chat 请求，并返回结构化结果。

        参数说明：
        - `scheduled_at` 使用 `time.perf_counter()` 同源时间；若不传，则默认记为 send 前时刻。
        - `stream` / `include_usage` 默认与 Step 2 benchmark 配置保持一致。
        """

        payload = request.to_openai_payload(
            self.model,
            stream=stream,
            include_usage=include_usage,
        )

        result = RequestResult.from_request(request)
        result.scheduled_at = scheduled_at

        if result.scheduled_at is None:
            result.scheduled_at = time.perf_counter()

        try:
            result.send_at = time.perf_counter()
            async with self._client.stream("POST", self.path, json=payload) as response:
                response.raise_for_status()
                await self._consume_stream(response, result)
        except httpx.TimeoutException as exc:
            return self._build_failed_result(
                request,
                result,
                error_type="timeout",
                error_message=str(exc),
            )
        except httpx.HTTPStatusError as exc:
            return self._build_failed_result(
                request,
                result,
                error_type="http_error",
                error_message=self._format_http_error(exc),
            )
        except httpx.RequestError as exc:
            return self._build_failed_result(
                request,
                result,
                error_type="connection_error",
                error_message=str(exc),
            )
        except ValueError as exc:
            return self._build_failed_result(
                request,
                result,
                error_type="parse_error",
                error_message=str(exc),
            )

        if result.end_at is None:
            result.end_at = time.perf_counter()

        return result

    async def _consume_stream(
        self,
        response: httpx.Response,
        result: RequestResult,
    ) -> None:
        """逐行消费 SSE 响应。"""

        collected_chunks: list[str] = []
        saw_done = False

        async for raw_line in response.aiter_lines():
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue

            observed_at = time.perf_counter()
            data_str = raw_line[5:].strip()
            if not data_str:
                continue

            if data_str == "[DONE]":
                result.end_at = observed_at
                saw_done = True
                break

            payload = self._parse_sse_json(data_str)
            self._update_result_from_chunk(
                payload=payload,
                observed_at=observed_at,
                result=result,
                collected_chunks=collected_chunks,
            )

        if not saw_done and result.end_at is None:
            # 有些兼容实现可能直接在最后断流，不显式发送 [DONE]。
            result.end_at = time.perf_counter()

        if collected_chunks:
            result.generated_text = "".join(collected_chunks)

    def _update_result_from_chunk(
        self,
        *,
        payload: dict[str, Any],
        observed_at: float,
        result: RequestResult,
        collected_chunks: list[str],
    ) -> None:
        """从单个 SSE JSON chunk 中提取内容与 usage。"""

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise ValueError("invalid SSE choice payload")

            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if result.first_token_at is None:
                        result.first_token_at = observed_at
                    result.token_arrival_times.append(observed_at)
                    collected_chunks.append(content)
                    result.generated_text = "".join(collected_chunks)

        usage = payload.get("usage")
        if isinstance(usage, dict):
            result.prompt_tokens = self._optional_usage_int(usage.get("prompt_tokens"))
            result.completion_tokens = self._optional_usage_int(
                usage.get("completion_tokens")
            )
            result.total_tokens = self._optional_usage_int(usage.get("total_tokens"))

    @staticmethod
    def _parse_sse_json(data_str: str) -> dict[str, Any]:
        """解析单条 `data:` JSON。"""

        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse SSE JSON chunk: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("SSE payload must be a JSON object")
        return payload

    @staticmethod
    def _optional_usage_int(value: Any) -> int | None:
        """把 usage 字段安全转成整数。"""

        if value is None:
            return None
        return int(value)

    @staticmethod
    def _format_http_error(exc: httpx.HTTPStatusError) -> str:
        """尽量把 HTTP 错误信息格式化得可读一些。"""

        status_code = exc.response.status_code
        response_text = exc.response.text.strip()
        if response_text:
            return f"HTTP {status_code}: {response_text}"
        return f"HTTP {status_code}: {exc}"

    @staticmethod
    def _build_failed_result(
        request: Request,
        partial_result: RequestResult,
        *,
        error_type: ErrorType,
        error_message: str,
    ) -> RequestResult:
        """把部分结果转成失败结果，同时保留已记录的时间戳和元数据。"""

        failed = RequestResult.from_request(
            request,
            status="failed",
            error_type=error_type,
            error_message=error_message,
        )

        failed.scheduled_at = partial_result.scheduled_at
        failed.send_at = partial_result.send_at
        failed.first_token_at = partial_result.first_token_at
        failed.end_at = partial_result.end_at or time.perf_counter()
        failed.token_arrival_times = list(partial_result.token_arrival_times)
        failed.prompt_tokens = partial_result.prompt_tokens
        failed.completion_tokens = partial_result.completion_tokens
        failed.total_tokens = partial_result.total_tokens
        failed.generated_text = partial_result.generated_text
        return failed


__all__ = ["StreamingChatClient"]
