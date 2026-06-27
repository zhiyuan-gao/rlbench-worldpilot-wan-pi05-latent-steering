from __future__ import annotations

import argparse
import json
from pathlib import Path

from .latent_cache import latent_path_for_record, load_latents
from .sample_index import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WAN latent cache coverage and tensor shapes.")
    parser.add_argument("--sample-index-path", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.sample_index_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    missing = []
    shapes = {}
    for row in rows:
        path = latent_path_for_record(args.cache_root, row)
        if not path.exists():
            missing.append(path.as_posix())
            continue
        latents = load_latents(args.cache_root, row)
        shapes[str(tuple(latents.shape))] = shapes.get(str(tuple(latents.shape)), 0) + 1
    result = {
        "sample_index_path": args.sample_index_path.as_posix(),
        "cache_root": args.cache_root.as_posix(),
        "checked": len(rows),
        "missing": len(missing),
        "shape_counts": shapes,
        "missing_examples": missing[:8],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

