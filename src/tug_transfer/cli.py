from __future__ import annotations

import argparse

from tug_transfer.config import load_config
from tug_transfer.experiments import (
    run_fogstar_experiment,
    run_spmt_experiment,
    run_tug_experiment,
)
from tug_transfer.figures import run_paper_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tug-transfer",
        description="Run dense TUG and gait-transfer experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("tug", "spmt", "fogstar", "figures"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            required=True,
            help=f"Path to the {command} YAML configuration file.",
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "tug":
        result = run_tug_experiment(config)
    elif args.command == "spmt":
        result = run_spmt_experiment(config)
    elif args.command == "fogstar":
        result = run_fogstar_experiment(config)
    elif args.command == "figures":
        result = run_paper_figures(config)
    else:
        raise ValueError(args.command)
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
