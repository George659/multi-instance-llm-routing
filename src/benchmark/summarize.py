"""Step 2 的离线 summary 脚本。

这个脚本读取 runner 输出的 `*_results.csv`，重建 `RequestResult` 列表，
然后复用 `metrics.py` 的统计逻辑重新生成 `summary.json`。

支持两种常见用法：

1. 显式指定某个 CSV：
   `python -m src.benchmark.summarize --input experiments/results/<run_id>_results.csv`
2. 自动选择最近一次 run：
   `python -m src.benchmark.summarize --latest`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

from src.benchmark.metrics import build_summary
from src.workload.schema import ErrorType, RequestResult, RequestStatus, WorkloadType


DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_LOGS_DIR = Path("experiments/logs")
DEFAULT_MIN_SAMPLES_FOR_P95 = 200
DEFAULT_MIN_SAMPLES_FOR_P99 = 1000


def summarize_results_csv(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    logs_dir: str | Path = DEFAULT_LOGS_DIR,
) -> dict[str, Any]:
    """从结果 CSV 生成 summary，并可选写回 JSON 文件。"""

    csv_path = Path(input_path)
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"results CSV is empty: {csv_path}")

    run_id = _require_single_value(rows, "run_id")
    arrival_mode = _require_single_value(rows, "arrival_mode")
    effective_rate = _parse_required_float(_require_single_value(rows, "effective_rate"))

    env_snapshot = _load_env_snapshot(run_id=run_id, logs_dir=logs_dir)
    workload_mix = _extract_workload_mix(env_snapshot, rows)
    measured_requests = _extract_measured_requests(env_snapshot, rows)
    min_samples_for_p95 = _extract_min_samples(
        env_snapshot,
        key="min_samples_for_p95",
        default=DEFAULT_MIN_SAMPLES_FOR_P95,
    )
    min_samples_for_p99 = _extract_min_samples(
        env_snapshot,
        key="min_samples_for_p99",
        default=DEFAULT_MIN_SAMPLES_FOR_P99,
    )

    results = [_row_to_request_result(row) for row in rows]
    summary = build_summary(
        results,
        run_id=run_id,
        workload_mix=workload_mix,
        arrival_mode=arrival_mode,
        effective_rate=effective_rate,
        measured_requests=measured_requests,
        min_samples_for_p95=min_samples_for_p95,
        min_samples_for_p99=min_samples_for_p99,
    )

    resolved_output = Path(output_path) if output_path is not None else _default_summary_path(csv_path)
    _write_json(resolved_output, summary)
    _print_summary(summary, input_path=csv_path, output_path=resolved_output)
    return summary


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def _require_single_value(rows: list[dict[str, str]], field_name: str) -> str:
    values = {row.get(field_name, "") for row in rows}
    values.discard("")
    if not values:
        raise ValueError(f"missing required field in CSV: {field_name}")
    if len(values) != 1:
        raise ValueError(f"expected a single {field_name} in CSV, got: {sorted(values)}")
    return next(iter(values))


def _row_to_request_result(row: dict[str, str]) -> RequestResult:
    status = _parse_request_status(row.get("status"))
    error_type = _parse_error_type(row.get("error_type")) if status == "failed" else None

    return RequestResult(
        request_id=_required_str(row.get("request_id"), "request_id"),
        workload_type=_parse_workload_type(row.get("workload_type")),
        status=status,
        error_type=error_type,
        error_message=_optional_str(row.get("error_message")),
        scheduled_at=_optional_float(row.get("scheduled_at")),
        send_at=_optional_float(row.get("send_at")),
        first_token_at=_optional_float(row.get("first_token_at")),
        end_at=_optional_float(row.get("end_at")),
        token_arrival_times=[],
        prompt_tokens=_optional_int(row.get("prompt_tokens")),
        completion_tokens=_optional_int(row.get("completion_tokens")),
        total_tokens=_optional_int(row.get("total_tokens")),
        generated_text=None,
        prompt_tokens_target=_optional_int(row.get("prompt_tokens_target")),
        prompt_tokens_actual=_optional_int(row.get("prompt_tokens_actual")),
        max_tokens_target=_optional_int(row.get("max_tokens_target")),
        min_tokens_target=_optional_int(row.get("min_tokens_target")),
        prefix_group_id=_optional_int(row.get("prefix_group_id")),
        shared_prefix_tokens=_optional_int(row.get("shared_prefix_tokens")) or 0,
    )


def _load_env_snapshot(run_id: str, logs_dir: str | Path) -> dict[str, Any] | None:
    env_path = Path(logs_dir) / f"{run_id}_env.json"
    if not env_path.exists():
        return None

    payload = json.loads(env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"env snapshot must be a JSON object: {env_path}")
    return payload


def _extract_workload_mix(
    env_snapshot: dict[str, Any] | None,
    rows: list[dict[str, str]],
) -> dict[str, float]:
    if env_snapshot is not None:
        mix = (
            env_snapshot.get("configs", {})
            .get("workload", {})
            .get("workload", {})
            .get("mix")
        )
        if isinstance(mix, dict):
            return {
                "chat": float(mix.get("chat", 0.0)),
                "rag": float(mix.get("rag", 0.0)),
                "agent": float(mix.get("agent", 0.0)),
            }

    counts = {"chat": 0, "rag": 0, "agent": 0}
    for row in rows:
        workload_type = row.get("workload_type")
        if workload_type in counts:
            counts[workload_type] += 1

    total = sum(counts.values())
    if total <= 0:
        return {"chat": 0.0, "rag": 0.0, "agent": 0.0}
    return {key: value / total for key, value in counts.items()}


def _extract_measured_requests(
    env_snapshot: dict[str, Any] | None,
    rows: list[dict[str, str]],
) -> int:
    if env_snapshot is not None:
        value = (
            env_snapshot.get("configs", {})
            .get("benchmark", {})
            .get("benchmark", {})
            .get("measured_requests")
        )
        if value is not None:
            return int(value)
    return len(rows)


def _extract_min_samples(
    env_snapshot: dict[str, Any] | None,
    *,
    key: str,
    default: int,
) -> int:
    if env_snapshot is not None:
        value = (
            env_snapshot.get("configs", {})
            .get("benchmark", {})
            .get("benchmark", {})
            .get(key)
        )
        if value is not None:
            return int(value)
    return default


def _default_summary_path(csv_path: Path) -> Path:
    if csv_path.name.endswith("_results.csv"):
        return csv_path.with_name(csv_path.name.replace("_results.csv", "_summary.json"))
    return csv_path.with_suffix(".summary.json")


def _find_latest_results_csv(results_dir: str | Path) -> Path:
    candidates = sorted(Path(results_dir).glob("*_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"no results CSV files found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(
    summary: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
) -> None:
    sample_counts = summary["sample_counts"]
    ttft = summary["ttft"]
    e2e = summary["e2e_latency"]
    throughput = summary["throughput"]
    reliability = summary["reliability"]

    print(f"[summarize] input={input_path}")
    print(f"[summarize] run_id={summary['run_id']}")
    print(
        "[summarize] samples="
        f"{sample_counts['success']} success / {sample_counts['failed']} failed "
        f"(total={sample_counts['total']})"
    )
    print(
        "[summarize] ttft_mean="
        f"{ttft['mean_sec']} p95={ttft['p95_sec']} p99={ttft['p99_sec']}"
    )
    print(
        "[summarize] e2e_mean="
        f"{e2e['mean_sec']} request_per_sec={throughput['request_per_sec']}"
    )
    if reliability.get("warning"):
        print(f"[summarize] warning={reliability['warning']}")
    print(f"[summarize] output={output_path}")


def _required_str(value: str | None, field_name: str) -> str:
    text = _optional_str(value)
    if text is None:
        raise ValueError(f"missing required field: {field_name}")
    return text


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_int(value: str | None) -> int | None:
    text = _optional_str(value)
    if text is None:
        return None
    return int(text)


def _optional_float(value: str | None) -> float | None:
    text = _optional_str(value)
    if text is None:
        return None
    return float(text)


def _parse_required_float(value: str) -> float:
    return float(value)


def _parse_workload_type(value: str | None) -> WorkloadType:
    text = _required_str(value, "workload_type")
    allowed = {"chat", "rag", "agent"}
    if text not in allowed:
        raise ValueError(f"invalid workload_type: {text}")
    return cast(WorkloadType, text)


def _parse_request_status(value: str | None) -> RequestStatus:
    text = _required_str(value, "status")
    allowed = {"success", "failed"}
    if text not in allowed:
        raise ValueError(f"invalid status: {text}")
    return cast(RequestStatus, text)


def _parse_error_type(value: str | None) -> ErrorType:
    text = _required_str(value, "error_type")
    allowed = {"timeout", "http_error", "parse_error", "connection_error"}
    if text not in allowed:
        raise ValueError(f"invalid error_type: {text}")
    return cast(ErrorType, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step 2 benchmark summary JSON from results CSV.")
    parser.add_argument("--input", help="Path to a *_results.csv file.")
    parser.add_argument("--output", help="Path to write summary JSON. Defaults to sibling *_summary.json.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically summarize the most recent *_results.csv under experiments/results.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory used by --latest. Default: experiments/results",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help="Directory containing <run_id>_env.json files. Default: experiments/logs",
    )
    args = parser.parse_args()

    if args.latest:
        input_path = _find_latest_results_csv(args.results_dir)
    elif args.input:
        input_path = Path(args.input)
    else:
        parser.error("either --input or --latest is required")
        return

    summarize_results_csv(
        input_path,
        output_path=args.output,
        logs_dir=args.logs_dir,
    )


if __name__ == "__main__":
    main()
