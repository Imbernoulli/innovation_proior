#!/usr/bin/env python3
"""Split methods into batches for answer-rewrite subagents."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_JSON = ROOT / "methods.json"
TASK_MAP = ROOT / "tools" / "method_primary_task.json"
BATCH_DIR = ROOT / "tools" / "rewrite_batches"
BATCH_SIZE = 10


def main():
    methods = json.loads(METHODS_JSON.read_text(encoding="utf-8"))
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()

    items = [{"slug": e["slug"], "task": task_map.get(e["slug"])} for e in methods]
    total = len(items)
    num_batches = math.ceil(total / BATCH_SIZE)
    for i in range(num_batches):
        batch = items[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        (BATCH_DIR / f"batch_{i:03d}.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Split {total} methods into {num_batches} rewrite batches")


if __name__ == "__main__":
    main()
