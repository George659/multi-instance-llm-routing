"""Step 2 的核心数据结构。

这个文件主要解决两件事：

1. 用统一的 `Request` 结构描述 workload generator 生成出来的请求。
2. 用统一的 `RequestResult` 结构描述 benchmark 运行后采集到的结果。

这样做的好处是：

- workload 生成层只负责“造什么请求”；
- client 层只负责“怎么把请求发出去”；
- metrics / runner 层只负责“怎么消费结果做统计”。

几层之间通过 schema 解耦，后续即使把请求发给本地 vLLM、远端 Router，
或者别的 OpenAI-compatible 服务，数据结构都不用重新设计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Literal


WorkloadType = Literal["chat", "rag", "agent"]
RequestStatus = Literal["success", "failed"]
ErrorType = Literal["timeout", "http_error", "parse_error", "connection_error"]
ChatMessage = dict[str, Any]


@dataclass(slots=True)
class RequestMetadata:
    """请求的辅助元数据。

    这部分信息不会直接原样发给模型服务端，但对实验分析非常重要。

    例如：
    - 生成器希望 prompt 有多少 token
    - 这个请求是否属于某个共享前缀组
    - 这个请求计划在 benchmark 开始后的第几秒发送

    后续 CSV / summary 统计时会大量依赖这些字段做分组分析。
    """

    # 生成器期望的 prompt token 数，通常来自配置采样结果。
    prompt_tokens_target: int
    # prompt 经过最终拼接和 tokenization 后的真实 token 数。
    prompt_tokens_actual: int
    # RAG 场景下的共享前缀分组编号；如果没有共享前缀则为 None。
    prefix_group_id: int | None
    # 共享前缀的真实 token 数。这个值应该来自 token-level 拼接后的结果。
    shared_prefix_tokens: int
    # 相对 benchmark 开始时间的发送偏移量，单位秒。
    arrival_time_offset: float
    # 当前请求对应的随机种子，方便复现实验。
    seed: int

    def __post_init__(self) -> None:
        """在 dataclass 初始化后做基础合法性校验。

        这里的校验目标不是做复杂业务逻辑，而是尽早拦住明显不合理的数据，
        避免后面 generator / runner / metrics 在更深层的位置才报错。
        """
        if self.prompt_tokens_target <= 0:
            raise ValueError("prompt_tokens_target must be positive")
        if self.prompt_tokens_actual <= 0:
            raise ValueError("prompt_tokens_actual must be positive")
        if self.shared_prefix_tokens < 0:
            raise ValueError("shared_prefix_tokens cannot be negative")
        if self.arrival_time_offset < 0:
            raise ValueError("arrival_time_offset cannot be negative")
        if self.prefix_group_id is None and self.shared_prefix_tokens != 0:
            raise ValueError(
                "shared_prefix_tokens must be 0 when prefix_group_id is None"
            )
        if self.prefix_group_id is not None and self.shared_prefix_tokens <= 0:
            raise ValueError(
                "shared_prefix_tokens must be positive when prefix_group_id is set"
            )

    @property
    def has_shared_prefix(self) -> bool:
        """是否带共享前缀。

        这里不单独存布尔字段，而是从 `prefix_group_id` 推导，避免状态不一致。
        """

        return self.prefix_group_id is not None

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典，方便后续写日志、写 JSON 或拼接 CSV 行。"""

        data = asdict(self)
        data["has_shared_prefix"] = self.has_shared_prefix
        return data


@dataclass(slots=True)
class Request:
    """标准化的 benchmark 请求对象。

    这是 workload generator 产出的“中间表示”：

    - 上游 generator 负责构造 `messages`、长度控制和 metadata
    - 下游 client 再把它转换成 HTTP 请求 payload 发给服务端

    这样我们在本地代码里就不用到处直接传裸字典，字段语义会更稳定。
    """

    request_id: str
    workload_type: WorkloadType
    messages: list[ChatMessage]
    max_tokens: int
    min_tokens: int | None
    temperature: float
    metadata: RequestMetadata

    def __post_init__(self) -> None:
        """校验请求本体字段与 metadata 是否一致。

        这里有一个关键点：
        `Request` 里已经有 `max_tokens/min_tokens`，
        但 metadata 里不再重复存这一对字段。

        这样设计是为了明确“单一事实来源”：
        - 请求发送时，直接读 `Request` 顶层字段更方便
        - prompt 长度、共享前缀等分析字段放进 metadata 更方便

        这样可以避免 `max_tokens/min_tokens` 在两个地方重复存储后产生同步问题。
        """
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")

    def to_openai_payload(
        self,
        model: str,
        *,
        stream: bool = True,
        include_usage: bool = True,
    ) -> dict[str, Any]:
        """把内部请求对象转换成 OpenAI-compatible 的请求体。

        这个方法主要给后续 `client.py` 用。

        转换规则和 Step 2 文档保持一致：
        - `model/messages/max_tokens/temperature` 必带
        - `min_tokens` 只有在非 None 时才带上
        - 开启 streaming 时附带 `stream_options.include_usage`

        说明：
        - `min_tokens` 是 vLLM OpenAI-compatible 扩展里常见的字段；
        - 使用前应确认当前 vLLM 版本支持该参数。

        这样 client 不需要自己重新拼字段，只要调用这个方法即可。
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if self.min_tokens is not None:
            payload["min_tokens"] = self.min_tokens
        if stream:
            payload["stream_options"] = {"include_usage": include_usage}
        return payload


@dataclass(slots=True)
class RequestResult:
    """单个请求在 benchmark 中的观测结果。

    这个类是 runner/client/metrics 之间共享的结果载体。

    它既包含：
    - 请求成功或失败的状态
    - streaming 过程中记录下来的关键时间戳
    - usage 返回的 token 数
    - 从原始 Request 继承过来的实验元数据

    也提供一些常用派生指标，避免别的模块重复计算。
    """

    request_id: str
    workload_type: WorkloadType
    status: RequestStatus
    error_type: ErrorType | None = None
    error_message: str | None = None

    scheduled_at: float | None = None
    send_at: float | None = None
    first_token_at: float | None = None
    end_at: float | None = None
    token_arrival_times: list[float] = field(default_factory=list)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generated_text: str | None = None

    prompt_tokens_target: int | None = None
    prompt_tokens_actual: int | None = None
    max_tokens_target: int | None = None
    min_tokens_target: int | None = None
    prefix_group_id: int | None = None
    shared_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        """校验结果对象的最基本约束。"""
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.status == "success" and self.error_type is not None:
            raise ValueError("error_type must be None when status is success")
        if self.status == "failed" and self.error_type is None:
            raise ValueError("error_type is required when status is failed")
        if self.shared_prefix_tokens < 0:
            raise ValueError("shared_prefix_tokens cannot be negative")
        if self.prefix_group_id is None and self.shared_prefix_tokens != 0:
            raise ValueError(
                "shared_prefix_tokens must be 0 when prefix_group_id is None"
            )
        if self.prefix_group_id is not None and self.shared_prefix_tokens <= 0:
            raise ValueError(
                "shared_prefix_tokens must be positive when prefix_group_id is set"
            )

    @property
    def has_shared_prefix(self) -> bool:
        """结果是否对应共享前缀请求。"""

        return self.prefix_group_id is not None

    @property
    def queue_delay_sec(self) -> float | None:
        """客户端排队延迟。

        含义是：请求原计划发送时间到真正开始发送之间差了多少秒。
        如果调度器或本地事件循环有堆积，这个值会变大。
        """
        if self.scheduled_at is None or self.send_at is None:
            return None
        return self.send_at - self.scheduled_at

    @property
    def ttft_sec(self) -> float | None:
        """TTFT（Time To First Token）。

        含义是：从 HTTP 请求真正发出，到收到第一个非空输出 token 的时间。
        这是 LLM serving 里最核心的交互延迟指标之一。
        """
        if self.send_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.send_at

    @property
    def e2e_latency_sec(self) -> float | None:
        """端到端延迟。

        含义是：从请求发出，到整个流式响应结束的总耗时。
        """
        if self.send_at is None or self.end_at is None:
            return None
        return self.end_at - self.send_at

    @property
    def tpot_list(self) -> list[float]:
        """根据每个 token chunk 的到达时间，计算相邻 token 间隔列表。

        TPOT 常被理解为 token-per-output-time 的细粒度观测基础。
        这里只返回相邻时间差，后续可以继续算 mean / p95 等统计量。
        """
        if len(self.token_arrival_times) < 2:
            return []
        return [
            current - previous
            for previous, current in zip(
                self.token_arrival_times, self.token_arrival_times[1:]
            )
        ]

    @property
    def mean_tpot_sec(self) -> float | None:
        """TPOT 均值。

        如果输出 token 太少，不足以形成时间差列表，就返回 None。
        """
        values = self.tpot_list
        if not values:
            return None
        return mean(values)

    @property
    def p95_tpot_sec(self) -> float | None:
        """TPOT 的 P95。

        这里基于相邻 token 到达间隔做 nearest-rank 百分位估计。
        如果样本为空，则返回 None。
        """

        values = sorted(self.tpot_list)
        if not values:
            return None
        rank = max(0, int(len(values) * 0.95) - 1)
        return values[rank]

    @property
    def num_output_tokens_observed(self) -> int:
        """基于本地观测到的流式 chunk 数，估算输出 token 数。

        注意这里是 fallback 指标，不一定等于服务端最终 usage 中的
        `completion_tokens`。但如果 usage 缺失，它至少能提供一个可用的
        本地近似观测值，方便后续分析。
        """
        return len(self.token_arrival_times)

    @classmethod
    def from_request(
        cls,
        request: Request,
        *,
        status: RequestStatus = "success",
        error_type: ErrorType | None = None,
        error_message: str | None = None,
    ) -> "RequestResult":
        """从 `Request` 快速初始化一个 `RequestResult`。

        这个方法适合给 client / runner 使用：
        请求一发出去，就先根据原始请求创建一个结果对象骨架，
        把后续分析需要的 metadata 先复制进来；
        然后再在请求执行过程中不断补齐时间戳、usage、错误信息等字段。
        """

        return cls(
            request_id=request.request_id,
            workload_type=request.workload_type,
            status=status,
            error_type=error_type,
            error_message=error_message,
            prompt_tokens_target=request.metadata.prompt_tokens_target,
            prompt_tokens_actual=request.metadata.prompt_tokens_actual,
            max_tokens_target=request.max_tokens,
            min_tokens_target=request.min_tokens,
            prefix_group_id=request.metadata.prefix_group_id,
            shared_prefix_tokens=request.metadata.shared_prefix_tokens,
        )

    def to_dict(
        self,
        *,
        include_token_arrivals: bool = True,
        include_text: bool = False,
    ) -> dict[str, Any]:
        """转成可序列化字典，方便写 CSV / JSON。

        这里除了原始字段，还会额外补几个派生指标：
        - `queue_delay_sec`
        - `ttft_sec`
        - `e2e_latency_sec`
        - `mean_tpot_sec`
        - `p95_tpot_sec`
        - `num_output_tokens_observed`

        `token_arrival_times` 可能比较长，所以保留一个开关，允许调用方在
        输出到 CSV 时把它排除掉，避免结果文件过大。
        `generated_text` 同理，默认不输出，避免结果文件过于臃肿。
        """

        data = asdict(self)
        data["has_shared_prefix"] = self.has_shared_prefix
        data["queue_delay_sec"] = self.queue_delay_sec
        data["ttft_sec"] = self.ttft_sec
        data["e2e_latency_sec"] = self.e2e_latency_sec
        data["mean_tpot_sec"] = self.mean_tpot_sec
        data["p95_tpot_sec"] = self.p95_tpot_sec
        data["num_output_tokens_observed"] = self.num_output_tokens_observed
        if not include_token_arrivals:
            data.pop("token_arrival_times", None)
        if not include_text:
            data.pop("generated_text", None)
        return data


__all__ = [
    "ChatMessage",
    "ErrorType",
    "Request",
    "RequestMetadata",
    "RequestResult",
    "RequestStatus",
    "WorkloadType",
]
