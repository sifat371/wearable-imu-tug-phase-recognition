from __future__ import annotations

import argparse

from tug_transfer.config import load_config
from tug_transfer.experiments import run_fogstar_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_fogstar_experiment(load_config(args.config))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
