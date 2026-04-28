"""Step 2 的 benchmark runner。

这个模块把前面已经实现的几个组件真正串起来：

- 读取配置
- 生成 run_id
- 写环境快照
- 加载 tokenizer
- 生成 workload 与 arrival schedule
- 执行 warmup 和 measured requests
- 写结果 CSV 与 summary JSON

设计上，runner 尽量只做“流程编排”，不把具体逻辑重新写一遍：

- 请求发送复用 `StreamingChatClient`
- 指标汇总复用 `metrics.py`
- 环境快照复用 `env_snapshot.py`
- run_id 复用 `run_id.py`
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import csv
import json
from pathlib import Path
import time
from typing import Any

import yaml

from src.benchmark.client import StreamingChatClient
from src.benchmark.env_snapshot import write_env_snapshot
from src.benchmark.metrics import build_result_rows, build_summary
from src.utils.run_id import generate_run_id
from src.workload.arrival import generate_arrival_offsets, infer_effective_rate
from src.workload.generators import generate_workload
from src.workload.schema import Request, RequestResult


async def run_benchmark(
    *,
    model_config: dict[str, Any],
    workload_config: dict[str, Any],
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    """执行一次完整 Step 2 benchmark run。"""

    model_section = _require_mapping(model_config.get("model"), "model")
    workload_section = _require_mapping(workload_config.get("workload"), "workload")
    benchmark_section = _require_mapping(benchmark_config.get("benchmark"), "benchmark")

    run_id = generate_run_id(
        model_config=model_config,
        workload_config=workload_config,
        benchmark_config=benchmark_config,
    )

    output_cfg = _require_mapping(benchmark_section.get("output"), "benchmark.output")
    env_snapshot_path = _resolve_output_path(output_cfg["env_snapshot_json"], run_id)
    result_csv_path = _resolve_output_path(output_cfg["result_csv"], run_id)
    summary_json_path = _resolve_output_path(output_cfg["summary_json"], run_id)

    # runner 启动后的第一件事：把环境与完整配置落盘。
    write_env_snapshot(
        output_path=env_snapshot_path,
        run_id=run_id,
        model_config=model_config,
        workload_config=workload_config,
        benchmark_config=benchmark_config,
    )

    tokenizer = _load_tokenizer(model_section["tokenizer_name"])

    warmup_requests = int(benchmark_section["warmup_requests"])
    measured_requests = int(benchmark_section["measured_requests"])
    total_requests = warmup_requests + measured_requests
    max_run_duration_sec = float(benchmark_section["max_run_duration_sec"])
    benchmark_seed = int(benchmark_section["seed"])

    request_cfg = _require_mapping(benchmark_section.get("request"), "benchmark.request")
    arrival_mode = str(_require_mapping(benchmark_section["arrival"], "benchmark.arrival")["mode"])
    effective_rate = infer_effective_rate(benchmark_section)

    prepared_workload_config = _clone_mapping(workload_config)
    prepared_workload_config["workload"]["num_requests"] = total_requests

    all_requests = generate_workload(
        prepared_workload_config,
        tokenizer,
        seed=int(workload_section.get("seed", benchmark_seed)),
    )
    arrival_offsets = generate_arrival_offsets(
        benchmark_section,
        num_requests=total_requests,
        seed=benchmark_seed,
    )
    _attach_arrival_offsets(all_requests, arrival_offsets)

    warmup_batch = all_requests[:warmup_requests]
    measured_batch = all_requests[warmup_requests:]

    async with StreamingChatClient(
        api_base=str(model_section["api_base"]),
        path=str(model_section["chat_completions_path"]),
        timeout=float(model_section["timeout_seconds"]),
        model=str(model_section["name"]),
    ) as client:
        if warmup_batch:
            print(f"[runner] starting warmup: {len(warmup_batch)} requests")
            await _execute_batch(
                client=client,
                requests=warmup_batch,
                run_duration_limit_sec=max_run_duration_sec,
                stream=bool(request_cfg.get("stream", True)),
                include_usage=bool(request_cfg.get("include_usage", True)),
            )

        print(f"[runner] starting measured run: {len(measured_batch)} requests")
        measured_results = await _execute_batch(
            client=client,
            requests=measured_batch,
            run_duration_limit_sec=max_run_duration_sec,
            stream=bool(request_cfg.get("stream", True)),
            include_usage=bool(request_cfg.get("include_usage", True)),
        )

    rows = build_result_rows(
        measured_results,
        run_id=run_id,
        arrival_mode=arrival_mode,
        effective_rate=effective_rate,
        model=str(model_section["name"]),
    )
    _write_csv(result_csv_path, rows)

    summary = build_summary(
        measured_results,
        run_id=run_id,
        workload_mix={
            "chat": float(workload_section["mix"]["chat"]),
            "rag": float(workload_section["mix"]["rag"]),
            "agent": float(workload_section["mix"]["agent"]),
        },
        arrival_mode=arrival_mode,
        effective_rate=effective_rate,
        measured_requests=measured_requests,
        min_samples_for_p95=int(benchmark_section["min_samples_for_p95"]),
        min_samples_for_p99=int(benchmark_section["min_samples_for_p99"]),
    )
    _write_json(summary_json_path, summary)
    _print_summary(summary, result_csv_path=result_csv_path, summary_json_path=summary_json_path)

    return {
        "run_id": run_id,
        "result_csv": str(result_csv_path),
        "summary_json": str(summary_json_path),
        "env_snapshot_json": str(env_snapshot_path),
        "summary": summary,
    }


async def _execute_batch(
    *,
    client: StreamingChatClient,
    requests: list[Request],
    run_duration_limit_sec: float,
    stream: bool,
    include_usage: bool,
) -> list[RequestResult]:
    """按 arrival schedule 执行一批请求。

    这里采用 `asyncio.create_task` 并发调度：
    - 每个请求根据自身 `arrival_time_offset` 睡到目标时间再发送；
    - 整个 batch 再受一个 wall-clock 超时保护。
    """

    if not requests:
        return []

    run_start_perf = time.perf_counter()
    task_to_request = {
        asyncio.create_task(
            _scheduled_send(
                client=client,
                request=request,
                run_start_perf=run_start_perf,
                stream=stream,
                include_usage=include_usage,
            )
        ): request
        for request in requests
    }
    tasks = list(task_to_request.keys())

    done, pending = await asyncio.wait(tasks, timeout=run_duration_limit_sec)

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    results: list[RequestResult] = []
    for task in done:
        task_result = task.result()
        results.append(task_result)
    for task in pending:
        request = task_to_request[task]
        results.append(
            _build_timeout_result(
                request,
                run_start_perf=run_start_perf,
                run_duration_limit_sec=run_duration_limit_sec,
            )
        )

    results.sort(key=lambda result: result.request_id)
    return results


async def _scheduled_send(
    *,
    client: StreamingChatClient,
    request: Request,
    run_start_perf: float,
    stream: bool,
    include_usage: bool,
) -> RequestResult:
    """等待到目标偏移后真正发送请求。"""

    scheduled_at = run_start_perf + request.metadata.arrival_time_offset
    delay = scheduled_at - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    return await client.send(
        request,
        scheduled_at=scheduled_at,
        stream=stream,
        include_usage=include_usage,
    )


def _build_timeout_result(
    request: Request,
    *,
    run_start_perf: float,
    run_duration_limit_sec: float,
) -> RequestResult:
    """把超时未完成的请求转成失败结果，保证结果集完整。"""

    result = RequestResult.from_request(
        request,
        status="failed",
        error_type="timeout",
        error_message=(
            "request did not complete before max_run_duration_sec elapsed"
        ),
    )
    result.scheduled_at = run_start_perf + request.metadata.arrival_time_offset
    result.end_at = run_start_perf + run_duration_limit_sec
    return result


def _attach_arrival_offsets(requests: list[Request], offsets: list[float]) -> None:
    """把 arrival offsets 写回每个请求的 metadata。"""

    if len(requests) != len(offsets):
        raise ValueError("requests and offsets must have the same length")

    for request, offset in zip(requests, offsets):
        request.metadata.arrival_time_offset = offset


def _load_tokenizer(tokenizer_name: str) -> Any:
    """按配置加载 tokenizer。"""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required to load the tokenizer for benchmark runner"
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {config_path}")
    return data


def _resolve_output_path(template: Any, run_id: str) -> Path:
    """把带 `{run_id}` 的输出模板解析成真实路径。"""

    resolved = str(template).format(run_id=run_id)
    return Path(resolved)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """把结果行写成 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """把 summary 写成 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(
    summary: dict[str, Any],
    *,
    result_csv_path: Path,
    summary_json_path: Path,
) -> None:
    """向 stdout 打印简短 run 结果。"""

    sample_counts = summary["sample_counts"]
    reliability = summary["reliability"]
    ttft = summary["ttft"]
    e2e = summary["e2e_latency"]
    throughput = summary["throughput"]

    print(f"[runner] run_id={summary['run_id']}")
    print(
        "[runner] samples="
        f"{sample_counts['success']} success / {sample_counts['failed']} failed "
        f"(total={sample_counts['total']})"
    )
    print(
        "[runner] ttft_mean="
        f"{ttft['mean_sec']} p95={ttft['p95_sec']} p99={ttft['p99_sec']}"
    )
    print(
        "[runner] e2e_mean="
        f"{e2e['mean_sec']} request_per_sec={throughput['request_per_sec']}"
    )
    if reliability.get("warning"):
        print(f"[runner] warning={reliability['warning']}")
    print(f"[runner] result_csv={result_csv_path}")
    print(f"[runner] summary_json={summary_json_path}")


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """确保某个配置段是字典。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _clone_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """做一个足够简单稳定的深拷贝。"""

    return json.loads(json.dumps(data))


def _apply_overrides(
    benchmark_config: dict[str, Any],
    overrides: list[str] | None,
) -> dict[str, Any]:
    """把 CLI 传入的 `key=value` 覆盖项应用到 benchmark 配置。

    约定：
    - 路径默认相对于 `benchmark` 段，例如 `warmup_requests=5`
    - 也允许显式写 `benchmark.arrival.request_rate=1.0`
    """

    if not overrides:
        return benchmark_config

    updated = _clone_mapping(benchmark_config)
    benchmark_section = updated.get("benchmark")
    if not isinstance(benchmark_section, dict):
        raise ValueError("benchmark config must contain a top-level 'benchmark' mapping")

    for item in overrides:
        key_path, raw_value = _parse_override_item(item)
        value = yaml.safe_load(raw_value)
        normalized_path = (
            key_path.split(".")[1:] if key_path.startswith("benchmark.") else key_path.split(".")
        )
        _set_nested_value(benchmark_section, normalized_path, value)

    return updated


def _parse_override_item(item: str) -> tuple[str, str]:
    """解析单条 `a.b.c=value` 覆盖项。"""

    if "=" not in item:
        raise ValueError(f"invalid override (expected key=value): {item}")
    key_path, raw_value = item.split("=", 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"invalid override with empty key: {item}")
    return key_path, raw_value.strip()


def _set_nested_value(target: dict[str, Any], path: list[str], value: Any) -> None:
    """按点路径写入嵌套字典。"""

    if not path:
        raise ValueError("override path cannot be empty")

    current: dict[str, Any] = target
    for segment in path[:-1]:
        existing = current.get(segment)
        if existing is None:
            current[segment] = {}
            existing = current[segment]
        if not isinstance(existing, dict):
            raise ValueError(
                f"cannot apply override through non-mapping segment: {'.'.join(path)}"
            )
        current = existing

    current[path[-1]] = value


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description="Run Step 2 benchmark harness")
    parser.add_argument("--model-config", required=True, help="Path to model.yaml")
    parser.add_argument("--workload-config", required=True, help="Path to workload.yaml")
    parser.add_argument(
        "--benchmark-config",
        required=True,
        help="Path to benchmark.yaml",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override benchmark config with key=value. Paths are relative to the "
            "benchmark section, e.g. warmup_requests=5 or arrival.request_rate=1.0"
        ),
    )
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    model_config = _load_yaml(args.model_config)
    workload_config = _load_yaml(args.workload_config)
    benchmark_config = _load_yaml(args.benchmark_config)
    benchmark_config = _apply_overrides(benchmark_config, args.override)

    await run_benchmark(
        model_config=model_config,
        workload_config=workload_config,
        benchmark_config=benchmark_config,
    )


def main() -> None:
    """CLI 入口。"""

    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
