from __future__ import annotations

import argparse

from tug_transfer.config import load_config
from tug_transfer.figures import run_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready paper figures.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = run_paper_figures(load_config(args.config))
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
