"""Step 2 的复合 run_id 生成工具。

文档要求的 run_id 结构是：

`{timestamp}_{git_short_hash}_{config_hash}`

示例：
`20260425_143022_a3f9b21_cfg7e2d`

这样设计的目的很直接：

- `timestamp` 告诉我们实验发生在什么时候；
- `git_short_hash` 告诉我们是哪个代码版本；
- `config_hash` 告诉我们是哪个配置组合。

后续无论是写结果文件名，还是做 env snapshot，runner 都可以直接复用这里的函数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import subprocess
from typing import Any, cast


def generate_run_id(
    *,
    model_config: Mapping[str, Any] | Any,
    workload_config: Mapping[str, Any] | Any,
    benchmark_config: Mapping[str, Any] | Any,
    timestamp: datetime | None = None,
    git_short_hash: str | None = None,
) -> str:
    """生成符合 Step 2 规范的复合 run_id。"""

    timestamp_part = format_run_timestamp(timestamp)
    git_part = git_short_hash or get_git_short_hash()
    config_part = compute_config_hash(
        model_config=model_config,
        workload_config=workload_config,
        benchmark_config=benchmark_config,
    )
    return f"{timestamp_part}_{git_part}_{config_part}"


def format_run_timestamp(timestamp: datetime | None = None) -> str:
    """把时间戳格式化成 `YYYYMMDD_HHMMSS`。"""

    current = timestamp or datetime.now(timezone.utc)
    return current.strftime("%Y%m%d_%H%M%S")


def get_git_short_hash() -> str:
    """获取当前仓库的短 commit hash。

    如果 working tree 是 dirty 状态，会在短 hash 后加 `-dirty`。
    如果当前目录不是 git 仓库，或者 git 命令不可用，则回退到 `nogit`。
    """

    try:
        short_hash = _run_git_command(["git", "rev-parse", "--short", "HEAD"]).strip()
        if not short_hash:
            return "nogit"

        if is_git_dirty():
            return f"{short_hash}-dirty"
        return short_hash
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def is_git_dirty() -> bool:
    """判断当前仓库 working tree 是否存在未提交修改。"""

    try:
        output = _run_git_command(["git", "status", "--porcelain"]).strip()
        return bool(output)
    except (OSError, subprocess.SubprocessError):
        return False


def compute_config_hash(
    *,
    model_config: Mapping[str, Any] | Any,
    workload_config: Mapping[str, Any] | Any,
    benchmark_config: Mapping[str, Any] | Any,
) -> str:
    """计算三个配置拼接后的短哈希。

    文档里建议用三个 yaml 内容拼接后的 SHA256 前 8 字节。
    这里我们做等价实现：把配置对象规范化成稳定 JSON，再拼接求 SHA256，
    最后取前 8 个十六进制字符，并加 `cfg` 前缀。
    """

    normalized_payload = {
        "model": _normalize_config(model_config),
        "workload": _normalize_config(workload_config),
        "benchmark": _normalize_config(benchmark_config),
    }
    encoded = _stable_json_dumps(normalized_payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:8]
    return f"cfg{digest}"


def _normalize_config(config: Mapping[str, Any] | Any) -> Any:
    """把不同形态的配置对象转换成稳定、可序列化的结构。"""

    if isinstance(config, Mapping):
        return {str(key): _normalize_config(value) for key, value in sorted(config.items())}

    if is_dataclass(config) and not isinstance(config, type):
        return _normalize_config(cast(dict[str, Any], asdict(cast(Any, config))))

    if isinstance(config, (list, tuple)):
        return [_normalize_config(item) for item in config]

    if hasattr(config, "__dict__") and not isinstance(config, type):
        return _normalize_config(vars(config))

    return config


def _stable_json_dumps(value: Any) -> str:
    """生成稳定 JSON 字符串，用于计算配置哈希。"""

    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run_git_command(args: list[str]) -> str:
    """执行 git 命令并返回 stdout。"""

    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


__all__ = [
    "compute_config_hash",
    "format_run_timestamp",
    "generate_run_id",
    "get_git_short_hash",
    "is_git_dirty",
]
