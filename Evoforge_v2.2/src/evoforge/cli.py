from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .app import App
from .config import WorldConfig
from .world import World


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoforge")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Launch the interactive simulation.")
    run.add_argument("--width", type=int, default=1400)
    run.add_argument("--height", type=int, default=900)
    run.add_argument("--agents", type=int, default=220)
    run.add_argument("--plants", type=int, default=700)
    run.add_argument("--seed", type=int)
    run.add_argument("--load", type=Path)

    bench = sub.add_parser("benchmark", help="Run the simulation headlessly.")
    bench.add_argument("--steps", type=int, default=5000)
    bench.add_argument("--width", type=int, default=1400)
    bench.add_argument("--height", type=int, default=900)
    bench.add_argument("--agents", type=int, default=500)
    bench.add_argument("--plants", type=int, default=1500)
    bench.add_argument("--seed", type=int, default=1)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = WorldConfig(
        width=args.width,
        height=args.height,
        initial_agents=args.agents,
        initial_plants=args.plants,
    )

    if args.command == "run":
        App(config, seed=args.seed, load_path=args.load).run()
        return 0

    world = World(config, seed=args.seed)
    started = time.perf_counter()

    for _ in range(args.steps):
        world.step()

    elapsed = time.perf_counter() - started
    stats = world.stats()
    stats.update(
        {
            "elapsed_seconds": elapsed,
            "steps_per_second": args.steps / elapsed if elapsed else 0.0,
        }
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
