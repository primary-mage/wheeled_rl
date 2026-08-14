#!/usr/bin/env python3
"""Validate a P1-P6 MuJoCo acceptance-suite result for global promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "p1": {"stationary_speed_rms": 0.10, "vx_rmse": 0.12, "yaw_rmse": 0.20},
    "p2": {"stationary_speed_rms": 0.12, "vx_rmse": 0.20, "yaw_rmse": 0.25},
    "p3": {"stationary_speed_rms": 0.14, "vx_rmse": 0.28, "yaw_rmse": 0.30},
    "p4a": {"stationary_speed_rms": 0.15, "vx_rmse": 0.35, "yaw_rmse": 0.32},
    "p4b": {"stationary_speed_rms": 0.16, "vx_rmse": 0.42, "yaw_rmse": 0.34},
    "p4c": {"stationary_speed_rms": 0.18, "vx_rmse": 0.50, "yaw_rmse": 0.36},
    "p5": {"stationary_speed_rms": 0.18, "vx_rmse": 0.52, "yaw_rmse": 0.35},
    "p6": {"stationary_speed_rms": 0.20, "vx_rmse": 0.55, "yaw_rmse": 0.40},
}
COMMON_LIMITS = {"falls": 0, "min_base_height": 0.18, "max_abs_roll": 0.45, "max_abs_pitch": 0.45}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--phase", choices=tuple(THRESHOLDS), required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    if metrics.get("phase") != args.phase:
        print(f"phase mismatch: file={metrics.get('phase')!r}, expected={args.phase!r}")
        return 1

    failures: list[str] = []
    for key, maximum in THRESHOLDS[args.phase].items():
        value = float(metrics[key])
        if value > maximum:
            failures.append(f"{key}={value:.4f} > {maximum:.4f}")
    if int(metrics["falls"]) > COMMON_LIMITS["falls"]:
        failures.append(f"falls={metrics['falls']} > 0")
    for key in ("min_base_height",):
        value = float(metrics[key])
        if value < COMMON_LIMITS[key]:
            failures.append(f"{key}={value:.4f} < {COMMON_LIMITS[key]:.4f}")
    for key in ("max_abs_roll", "max_abs_pitch"):
        value = float(metrics[key])
        if value > COMMON_LIMITS[key]:
            failures.append(f"{key}={value:.4f} > {COMMON_LIMITS[key]:.4f}")

    if failures:
        print("gate failed: " + "; ".join(failures))
        return 1
    print(f"gate passed for {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
