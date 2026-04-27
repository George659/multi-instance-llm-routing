"""Step 2 的 benchmark 指标计算工具。

这个文件只做“计算”，不负责：

- 发送请求
- 读写 CSV / JSON
- 管理实验生命周期

这样 runner 可以把 `RequestResult` 列表交给这里，拿到：

1. 每请求的结构化行数据
2. summary 统计结果
3. 按 workload_type / prefix_group 的分组统计
4. P95 / P99 的可靠性标记
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from src.workload.schema import RequestResult


def build_result_rows(
    results: list[RequestResult],
    *,
    run_id: str,
    arrival_mode: str,
    effective_rate: float,
    model: str,
) -> list[dict[str, Any]]:
    """把请求结果转换成适合写 CSV 的行字典。"""

    rows: list[dict[str, Any]] = []
    for result in results:
        row = result.to_dict(include_token_arrivals=False, include_text=False)
        row["run_id"] = run_id
        row["arrival_mode"] = arrival_mode
        row["effective_rate"] = effective_rate
        row["model"] = model
        rows.append(row)
    return rows


def build_summary(
    results: list[RequestResult],
    *,
    run_id: str,
    workload_mix: dict[str, float],
    arrival_mode: str,
    effective_rate: float,
    measured_requests: int,
    min_samples_for_p95: int,
    min_samples_for_p99: int,
) -> dict[str, Any]:
    """生成 Step 2 文档要求的 summary 结构。"""

    success_results = [result for result in results if result.status == "success"]
    failed_results = [result for result in results if result.status == "failed"]

    reliability = compute_reliability(
        success_count=len(success_results),
        min_samples_for_p95=min_samples_for_p95,
        min_samples_for_p99=min_samples_for_p99,
    )

    return {
        "run_id": run_id,
        "config_summary": {
            "workload_mix": workload_mix,
            "arrival_mode": arrival_mode,
            "effective_rate": effective_rate,
            "measured_requests": measured_requests,
        },
        "sample_counts": {
            "total": len(results),
            "success": len(success_results),
            "failed": len(failed_results),
        },
        "reliability": reliability,
        "ttft": summarize_latency_metric(
            [result.ttft_sec for result in success_results],
            reliability=reliability,
        ),
        "e2e_latency": summarize_latency_metric(
            [result.e2e_latency_sec for result in success_results],
            reliability=reliability,
        ),
        "tpot": summarize_tpot_metric(success_results),
        "throughput": summarize_throughput(success_results),
        "by_workload_type": summarize_by_workload_type(
            success_results,
            reliability=reliability,
        ),
        "by_prefix_group": summarize_by_prefix_group(
            success_results,
            reliability=reliability,
        ),
    }


def compute_reliability(
    *,
    success_count: int,
    min_samples_for_p95: int,
    min_samples_for_p99: int,
) -> dict[str, Any]:
    """计算 P95 / P99 是否可靠。"""

    p95_reliable = success_count >= min_samples_for_p95
    p99_reliable = success_count >= min_samples_for_p99

    warnings: list[str] = []
    if not p95_reliable:
        warnings.append(
            "P95 estimates are based on "
            f"{success_count} samples; consider this in interpretation "
            f"if min_samples_for_p95={min_samples_for_p95}."
        )
    if not p99_reliable:
        warnings.append(
            "P99 estimates are based on "
            f"{success_count} samples; consider this in interpretation "
            f"if min_samples_for_p99={min_samples_for_p99}."
        )

    return {
        "p95_reliable": p95_reliable,
        "p99_reliable": p99_reliable,
        "warning": " ".join(warnings) if warnings else None,
    }


def summarize_latency_metric(
    values: list[float | None],
    *,
    reliability: dict[str, Any],
) -> dict[str, float | None]:
    """汇总 latency 类指标。"""

    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return {
            "mean_sec": None,
            "p50_sec": None,
            "p90_sec": None,
            "p95_sec": None,
            "p99_sec": None,
        }

    p95_value = percentile(clean_values, 95) if reliability["p95_reliable"] else None
    p99_value = percentile(clean_values, 99) if reliability["p99_reliable"] else None

    return {
        "mean_sec": mean(clean_values),
        "p50_sec": percentile(clean_values, 50),
        "p90_sec": percentile(clean_values, 90),
        "p95_sec": p95_value,
        "p99_sec": p99_value,
    }


def summarize_tpot_metric(results: list[RequestResult]) -> dict[str, float | None]:
    """汇总 TPOT 指标。

    这里按所有成功请求的 inter-token latencies 合并后做全局分布统计。
    """

    values: list[float] = []
    for result in results:
        values.extend(result.tpot_list)

    if not values:
        return {
            "mean_sec": None,
            "p50_sec": None,
            "p95_sec": None,
        }

    return {
        "mean_sec": mean(values),
        "p50_sec": percentile(values, 50),
        "p95_sec": percentile(values, 95),
    }


def summarize_throughput(results: list[RequestResult]) -> dict[str, float | None]:
    """计算 request / input token / output token 吞吐。"""

    if not results:
        return {
            "request_per_sec": None,
            "input_tokens_per_sec": None,
            "output_tokens_per_sec": None,
        }

    send_times = [result.send_at for result in results if result.send_at is not None]
    end_times = [result.end_at for result in results if result.end_at is not None]
    if not send_times or not end_times:
        return {
            "request_per_sec": None,
            "input_tokens_per_sec": None,
            "output_tokens_per_sec": None,
        }

    wall_clock_sec = max(end_times) - min(send_times)
    if wall_clock_sec <= 0:
        return {
            "request_per_sec": math.inf,
            "input_tokens_per_sec": math.inf,
            "output_tokens_per_sec": math.inf,
        }

    input_tokens = sum((result.prompt_tokens or 0) for result in results)
    output_tokens = sum(
        (
            result.completion_tokens
            if result.completion_tokens is not None
            else result.num_output_tokens_observed
        )
        for result in results
    )

    return {
        "request_per_sec": len(results) / wall_clock_sec,
        "input_tokens_per_sec": input_tokens / wall_clock_sec,
        "output_tokens_per_sec": output_tokens / wall_clock_sec,
    }


def summarize_by_workload_type(
    results: list[RequestResult],
    *,
    reliability: dict[str, Any],
) -> dict[str, Any]:
    """按 workload_type 分组汇总。"""

    grouped: dict[str, list[RequestResult]] = {}
    for result in results:
        grouped.setdefault(result.workload_type, []).append(result)

    return {
        workload_type: summarize_group_metrics(group_results, reliability=reliability)
        for workload_type, group_results in grouped.items()
    }


def summarize_by_prefix_group(
    results: list[RequestResult],
    *,
    reliability: dict[str, Any],
) -> dict[str, Any]:
    """按 prefix_group_id 分组汇总。"""

    grouped: dict[str, list[RequestResult]] = {}
    for result in results:
        key = (
            str(result.prefix_group_id)
            if result.prefix_group_id is not None
            else "no_shared_prefix"
        )
        grouped.setdefault(key, []).append(result)

    return {
        group_name: summarize_group_metrics(group_results, reliability=reliability)
        for group_name, group_results in grouped.items()
    }


def summarize_group_metrics(
    results: list[RequestResult],
    *,
    reliability: dict[str, Any],
) -> dict[str, Any]:
    """给单个分组生成一份紧凑 summary。"""

    return {
        "sample_count": len(results),
        "ttft": summarize_latency_metric(
            [result.ttft_sec for result in results],
            reliability=reliability,
        ),
        "e2e_latency": summarize_latency_metric(
            [result.e2e_latency_sec for result in results],
            reliability=reliability,
        ),
        "tpot": summarize_tpot_metric(results),
        "throughput": summarize_throughput(results),
    }


def percentile(values: list[float], q: float) -> float:
    """计算线性插值百分位数。

    这里不用额外依赖 numpy，方便 Step 2 先保持依赖面简单。
    """

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (q / 100.0)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


__all__ = [
    "build_result_rows",
    "build_summary",
    "compute_reliability",
    "percentile",
    "summarize_by_prefix_group",
    "summarize_by_workload_type",
    "summarize_group_metrics",
    "summarize_latency_metric",
    "summarize_throughput",
    "summarize_tpot_metric",
]
