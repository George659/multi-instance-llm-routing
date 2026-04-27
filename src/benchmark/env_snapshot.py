"""Step 2 的环境快照采集工具。

这个模块负责在 benchmark run 启动时，把“本次实验运行环境”完整记录下来。

记录这些信息的意义在于：

- 后续数字变了时，可以快速判断是代码版本变了、配置变了，还是环境变了；
- run 结果和环境快照一一对应，便于复现实验；
- 即使后续切换本地 / 云上环境，也能保留一致的快照结构。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any, cast

from src.utils.run_id import get_git_short_hash, is_git_dirty


def build_env_snapshot(
    *,
    run_id: str,
    model_config: Mapping[str, Any] | Any,
    workload_config: Mapping[str, Any] | Any,
    benchmark_config: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """采集一份完整的环境快照字典。"""

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": collect_git_info(),
        "python": platform.python_version(),
        "packages": collect_package_versions(["vllm", "torch", "transformers", "httpx"]),
        "cuda": collect_cuda_info(),
        "system": collect_system_info(),
        "configs": {
            "model": normalize_config(model_config),
            "workload": normalize_config(workload_config),
            "benchmark": normalize_config(benchmark_config),
        },
    }


def write_env_snapshot(
    *,
    output_path: str | Path,
    run_id: str,
    model_config: Mapping[str, Any] | Any,
    workload_config: Mapping[str, Any] | Any,
    benchmark_config: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """采集环境快照并写入 JSON 文件。"""

    snapshot = build_env_snapshot(
        run_id=run_id,
        model_config=model_config,
        workload_config=workload_config,
        benchmark_config=benchmark_config,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def collect_git_info() -> dict[str, Any]:
    """采集 git commit / branch / dirty 状态。"""

    commit = get_git_short_hash()
    try:
        branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
        if not branch:
            branch = "unknown"
    except (OSError, subprocess.SubprocessError):
        branch = "unknown"

    return {
        "commit": commit,
        "branch": branch,
        "dirty": is_git_dirty(),
    }


def collect_package_versions(package_names: list[str]) -> dict[str, str | None]:
    """采集关键依赖包版本。

    若某个包尚未安装，则返回 None，而不是抛异常。
    """

    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as metadata  # type: ignore

    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def collect_cuda_info() -> dict[str, Any]:
    """采集 CUDA / GPU 信息。

    这里优先走 PyTorch，因为 Step 2 后续本来也依赖 torch/vLLM 环境。
    如果 torch 不存在或 GPU 不可用，就回退到尽量保守的空值结构。
    """

    try:
        import torch
    except ImportError:
        return {
            "version": None,
            "driver": _query_nvidia_smi_driver_version(),
            "device_name": None,
            "device_count": 0,
            "compute_capability": None,
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else None

    compute_capability = None
    if device_count > 0:
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"

    return {
        "version": getattr(torch.version, "cuda", None),
        "driver": _query_nvidia_smi_driver_version(),
        "device_name": device_name,
        "device_count": device_count,
        "compute_capability": compute_capability,
    }


def collect_system_info() -> dict[str, str]:
    """采集基础系统信息。"""

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


def normalize_config(config: Mapping[str, Any] | Any) -> Any:
    """把配置对象转成可序列化的稳定结构。"""

    if isinstance(config, Mapping):
        return {str(key): normalize_config(value) for key, value in sorted(config.items())}

    if is_dataclass(config) and not isinstance(config, type):
        return normalize_config(cast(dict[str, Any], asdict(cast(Any, config))))

    if isinstance(config, (list, tuple)):
        return [normalize_config(item) for item in config]

    if hasattr(config, "__dict__") and not isinstance(config, type):
        return normalize_config(vars(config))

    return config


def _query_nvidia_smi_driver_version() -> str | None:
    """尽量通过 `nvidia-smi` 获取 driver version。"""

    try:
        output = _run_git_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
    except (OSError, subprocess.SubprocessError):
        return None

    first_line = output.strip().splitlines()
    return first_line[0].strip() if first_line else None


def _run_git_command(args: list[str]) -> str:
    """运行一个外部命令并返回 stdout。

    虽然名字沿用了 git 工具模块里的习惯，这里也顺手用于 `nvidia-smi`。
    """

    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


__all__ = [
    "build_env_snapshot",
    "collect_cuda_info",
    "collect_git_info",
    "collect_package_versions",
    "collect_system_info",
    "normalize_config",
    "write_env_snapshot",
]
