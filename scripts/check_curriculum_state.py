"""Exit successfully when enough parallel environments reached the final curriculum level."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state_path", type=Path)
    parser.add_argument("--expected-num-envs", type=int, required=True)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--min-max-fraction", type=float, default=0.90)
    args = parser.parse_args()

    if not args.state_path.is_file():
        print(f"curriculum state not found: {args.state_path}")
        return 1

    state = torch.load(args.state_path, map_location="cpu")
    levels = state["level"] if isinstance(state, dict) else state
    levels = torch.as_tensor(levels).flatten()
    if levels.numel() != args.expected_num_envs:
        print(f"expected {args.expected_num_envs} levels, found {levels.numel()}")
        return 1

    max_level = args.num_levels - 1
    fraction = (levels == max_level).float().mean().item()
    mean_level = levels.float().mean().item()
    print(f"mean_level={mean_level:.3f} max_level_fraction={fraction:.3f}")
    return 0 if fraction >= args.min_max_fraction else 1


if __name__ == "__main__":
    raise SystemExit(main())
