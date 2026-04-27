"""Step 2 的请求到达模式生成器。

这个文件负责根据 benchmark 配置生成每个请求相对于 run 开始时间的发送偏移。

Step 2 文档要求支持三种 arrival pattern：

1. constant：固定间隔到达
2. poisson：指数分布间隔到达
3. burst：突发到达，但平均速率要和其他模式可比

核心设计点：

- 对外统一用 `generate_arrival_offsets()` 入口；
- 三种模式都暴露“等价平均速率” `effective_rate`，方便跨模式对比；
- burst 模式必须校验 `burst_size / burst_interval_sec ~= effective_rate`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
import random
from typing import Any, Literal, cast


ArrivalMode = Literal["constant", "poisson", "burst"]
BURST_ACTIVE_WINDOW_RATIO = 0.1
BURST_RATE_TOLERANCE = 0.01


def generate_arrival_offsets(
    config: Mapping[str, Any] | Any,
    *,
    num_requests: int,
    seed: int,
) -> list[float]:
    """统一入口：根据配置生成 arrival offsets。

    参数说明：
    - `config` 可以直接传 `arrival` 段，也可以传完整 benchmark 配置；
    - `num_requests` 是本次 run 需要生成的请求数；
    - `seed` 用于控制 poisson / burst 的可复现性。

    返回值：
    - 长度为 `num_requests` 的单调非降浮点列表；
    - 每个值表示“从 run 开始起，第几个请求应该在第几秒发送”。
    """

    arrival_cfg = _unwrap_arrival_config(config)
    mode = cast(ArrivalMode, arrival_cfg["mode"])
    rng = random.Random(seed)

    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if num_requests == 0:
        return []

    if mode == "constant":
        request_rate = _positive_float(arrival_cfg["request_rate"], "request_rate")
        return generate_constant_arrival(num_requests=num_requests, request_rate=request_rate)

    if mode == "poisson":
        request_rate = _positive_float(arrival_cfg["request_rate"], "request_rate")
        return generate_poisson_arrival(
            num_requests=num_requests,
            request_rate=request_rate,
            rng=rng,
        )

    if mode == "burst":
        burst_cfg = _require_mapping(arrival_cfg.get("burst"), "arrival.burst")
        effective_rate = _positive_float(burst_cfg["effective_rate"], "burst.effective_rate")
        burst_size = _positive_int(burst_cfg["burst_size"], "burst.burst_size")
        burst_interval_sec = _positive_float(
            burst_cfg["burst_interval_sec"],
            "burst.burst_interval_sec",
        )
        return generate_burst_arrival(
            num_requests=num_requests,
            effective_rate=effective_rate,
            burst_size=burst_size,
            burst_interval_sec=burst_interval_sec,
            rng=rng,
        )

    raise ValueError(f"unsupported arrival mode: {mode}")


def generate_constant_arrival(*, num_requests: int, request_rate: float) -> list[float]:
    """生成固定间隔到达序列。

    文档定义：
    `interval = 1 / request_rate`
    `arrival_offsets = [0, interval, 2 * interval, ...]`
    """

    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    request_rate = _positive_float(request_rate, "request_rate")

    interval = 1.0 / request_rate
    return [index * interval for index in range(num_requests)]


def generate_poisson_arrival(
    *,
    num_requests: int,
    request_rate: float,
    rng: random.Random,
) -> list[float]:
    """生成 poisson 到达序列。

    这里按指数分布采样 inter-arrival time，然后做累加。
    第一条请求允许在 0 附近立即到达，因此偏移从第一段指数间隔开始累计。
    """

    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    request_rate = _positive_float(request_rate, "request_rate")

    offsets: list[float] = []
    current = 0.0
    for _ in range(num_requests):
        current += rng.expovariate(request_rate)
        offsets.append(current)
    return offsets


def generate_burst_arrival(
    *,
    num_requests: int,
    effective_rate: float,
    burst_size: int,
    burst_interval_sec: float,
    rng: random.Random,
) -> list[float]:
    """生成 burst 到达序列。

    burst 模式的设计重点是“平均速率可比”：

    - `effective_rate` 是期望平均 req/s；
    - `burst_size / burst_interval_sec` 必须与它一致；
    - burst 内部请求会被压缩到一个很短的时间窗里，形成短时突发。

    文档推荐：
    `intra_burst_gap = (burst_interval_sec * 0.1) / burst_size`
    """

    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")

    effective_rate = _positive_float(effective_rate, "effective_rate")
    burst_size = _positive_int(burst_size, "burst_size")
    burst_interval_sec = _positive_float(burst_interval_sec, "burst_interval_sec")

    expected_avg_rate = burst_size / burst_interval_sec
    relative_error = abs(expected_avg_rate - effective_rate) / effective_rate
    if relative_error > BURST_RATE_TOLERANCE:
        raise ValueError(
            "burst average rate mismatch: "
            f"expected {expected_avg_rate:.6f} req/s from burst_size / burst_interval_sec, "
            f"but effective_rate={effective_rate:.6f}"
        )

    intra_burst_gap = (burst_interval_sec * BURST_ACTIVE_WINDOW_RATIO) / burst_size
    offsets: list[float] = []

    for request_index in range(num_requests):
        burst_index = request_index // burst_size
        position_in_burst = request_index % burst_size

        burst_start = burst_index * burst_interval_sec

        # 在 burst 内部保留稳定顺序，同时加入极小随机抖动，避免所有请求严格等距
        # 到达时过于“人工”。抖动上限远小于 gap，不会破坏单调性。
        base_offset = burst_start + position_in_burst * intra_burst_gap
        jitter = 0.0
        if position_in_burst > 0:
            jitter = rng.uniform(0.0, intra_burst_gap * 0.01)
        offsets.append(base_offset + jitter)

    offsets.sort()
    return offsets


def infer_effective_rate(config: Mapping[str, Any] | Any) -> float:
    """从 arrival 配置中推导等价平均速率。"""

    arrival_cfg = _unwrap_arrival_config(config)
    mode = cast(ArrivalMode, arrival_cfg["mode"])

    if mode in {"constant", "poisson"}:
        return _positive_float(arrival_cfg["request_rate"], "request_rate")

    if mode == "burst":
        burst_cfg = _require_mapping(arrival_cfg.get("burst"), "arrival.burst")
        return _positive_float(burst_cfg["effective_rate"], "burst.effective_rate")

    raise ValueError(f"unsupported arrival mode: {mode}")


def compute_average_rate(offsets: list[float]) -> float:
    """根据生成出来的 offsets 粗略估计平均速率。

    这个函数主要服务于单元测试和调试，不用于正式 benchmark 统计。
    """

    if not offsets:
        return 0.0
    if len(offsets) == 1:
        return math.inf

    duration = offsets[-1] - offsets[0]
    if duration <= 0:
        return math.inf
    return (len(offsets) - 1) / duration


def _unwrap_arrival_config(config: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """把不同形态的 benchmark / arrival 配置统一成 arrival 段。"""

    if isinstance(config, Mapping):
        if "benchmark" in config and isinstance(config["benchmark"], Mapping):
            return _unwrap_arrival_config(config["benchmark"])
        if "arrival" in config and isinstance(config["arrival"], Mapping):
            return config["arrival"]
        return config

    if is_dataclass(config) and not isinstance(config, type):
        data = cast(dict[str, Any], asdict(cast(Any, config)))
        return _unwrap_arrival_config(data)

    if hasattr(config, "benchmark"):
        return _unwrap_arrival_config(getattr(config, "benchmark"))
    if hasattr(config, "arrival"):
        return _unwrap_arrival_config(getattr(config, "arrival"))
    if hasattr(config, "__dict__"):
        return _unwrap_arrival_config(dict(vars(config)))

    raise TypeError("unsupported arrival config type")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """确保某个配置字段是映射对象。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    """把配置值转成正浮点数。"""

    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    """把配置值转成正整数。"""

    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


__all__ = [
    "ArrivalMode",
    "BURST_ACTIVE_WINDOW_RATIO",
    "BURST_RATE_TOLERANCE",
    "compute_average_rate",
    "generate_arrival_offsets",
    "generate_burst_arrival",
    "generate_constant_arrival",
    "generate_poisson_arrival",
    "infer_effective_rate",
]
