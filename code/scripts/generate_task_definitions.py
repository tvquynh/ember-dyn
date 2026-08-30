"""Install the synchronized EMBER-Dyn T1-T15 task definitions.

The adjacent JSON template is the reader-facing protocol specification used by
the manuscript and baseline runner. This utility copies it without changing task
IDs, names, metrics, cohort rules, or validation/test discipline.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_IDS = [f"T{i}" for i in range(1, 16)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "task_definitions_template.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "task_definitions.json",
    )
    args = parser.parse_args()

    with args.template.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    if list(tasks) != EXPECTED_IDS:
        raise ValueError(f"Expected ordered task IDs {EXPECTED_IDS}, got {list(tasks)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.template, args.output)
    print(f"Wrote synchronized definitions to {args.output}")


if __name__ == "__main__":
    main()
