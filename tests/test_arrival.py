"""任务 4：arrival pattern 生成逻辑测试。"""

from __future__ import annotations

import math
import random

import pytest

from src.workload.arrival import (
    BURST_ACTIVE_WINDOW_RATIO,
    BURST_RATE_TOLERANCE,
    compute_average_rate,
    generate_arrival_offsets,
    generate_burst_arrival,
    generate_constant_arrival,
    generate_poisson_arrival,
    infer_effective_rate,
)


def test_constant_arrival_has_expected_fixed_interval() -> None:
    offsets = generate_constant_arrival(num_requests=5, request_rate=2.0)

    assert offsets == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_generate_arrival_offsets_dispatches_constant_mode() -> None:
    config = {
        "benchmark": {
            "arrival": {
                "mode": "constant",
                "request_rate": 4.0,
            }
        }
    }

    offsets = generate_arrival_offsets(config, num_requests=4, seed=1)
    assert offsets == [0.0, 0.25, 0.5, 0.75]
    assert infer_effective_rate(config) == 4.0


def test_poisson_arrival_is_monotonic_and_average_rate_converges() -> None:
    rate = 3.0
    offsets = generate_poisson_arrival(
        num_requests=20_000,
        request_rate=rate,
        rng=random.Random(123),
    )

    assert len(offsets) == 20_000
    assert all(left < right for left, right in zip(offsets, offsets[1:]))

    observed_rate = compute_average_rate(offsets)
    relative_error = abs(observed_rate - rate) / rate
    # poisson 有随机波动，所以这里用一个温和但足够收敛的阈值。
    assert relative_error < 0.05


def test_generate_arrival_offsets_dispatches_poisson_mode_deterministically() -> None:
    config = {
        "arrival": {
            "mode": "poisson",
            "request_rate": 2.5,
        }
    }

    run_a = generate_arrival_offsets(config, num_requests=10, seed=77)
    run_b = generate_arrival_offsets(config, num_requests=10, seed=77)

    assert run_a == run_b
    assert infer_effective_rate(config) == 2.5


def test_burst_arrival_matches_effective_rate_over_burst_horizon() -> None:
    effective_rate = 2.0
    burst_size = 20
    burst_interval_sec = 10.0
    num_requests = 200  # 整数个 burst，便于按完整 horizon 计算平均速率

    offsets = generate_burst_arrival(
        num_requests=num_requests,
        effective_rate=effective_rate,
        burst_size=burst_size,
        burst_interval_sec=burst_interval_sec,
        rng=random.Random(99),
    )

    assert len(offsets) == num_requests
    assert all(left < right for left, right in zip(offsets, offsets[1:]))

    intra_burst_gap = (burst_interval_sec * BURST_ACTIVE_WINDOW_RATIO) / burst_size
    assert math.isclose(offsets[1] - offsets[0], intra_burst_gap, rel_tol=0.02)

    # 对 burst 模式，平均速率应该按完整 burst horizon 计算，而不是只看活跃窗口。
    num_bursts = num_requests // burst_size
    total_horizon = num_bursts * burst_interval_sec
    observed_rate = num_requests / total_horizon
    relative_error = abs(observed_rate - effective_rate) / effective_rate
    assert relative_error <= BURST_RATE_TOLERANCE


def test_generate_arrival_offsets_dispatches_burst_mode() -> None:
    config = {
        "benchmark": {
            "arrival": {
                "mode": "burst",
                "request_rate": 2.0,
                "burst": {
                    "effective_rate": 2.0,
                    "burst_size": 20,
                    "burst_interval_sec": 10.0,
                },
            }
        }
    }

    offsets = generate_arrival_offsets(config, num_requests=25, seed=5)
    assert len(offsets) == 25
    assert offsets[0] == 0.0
    assert offsets[20] >= 10.0
    assert infer_effective_rate(config) == 2.0


def test_burst_arrival_rejects_mismatched_effective_rate() -> None:
    with pytest.raises(ValueError, match="burst average rate mismatch"):
        generate_burst_arrival(
            num_requests=10,
            effective_rate=3.0,
            burst_size=20,
            burst_interval_sec=10.0,
            rng=random.Random(1),
        )


def test_generate_arrival_offsets_returns_empty_for_zero_requests() -> None:
    config = {"arrival": {"mode": "constant", "request_rate": 2.0}}
    assert generate_arrival_offsets(config, num_requests=0, seed=1) == []
