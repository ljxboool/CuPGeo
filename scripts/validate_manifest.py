#!/usr/bin/env python3
from __future__ import annotations

import argparse
from c3tta.data.manifest import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mode", choices=("source", "target_adapt"), default="source")
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    errors = validate_manifest(args.csv, args.mode, args.check_paths, args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {args.csv} ({args.mode})")


if __name__ == "__main__":
    main()
