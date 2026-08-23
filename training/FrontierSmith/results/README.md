# FrontierSmith reproduction results

`reproduction_results.csv` is the small summary matrix recovered from
`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/reproduction_results.csv`.

The raw `outputs/` tree is intentionally not committed: it contains large `samples.jsonl`, rollout
JSONL, workspaces, logs, and some unreadable runtime directories. Use this CSV as the light index of
what had been aggregated.

## Snapshot

- Rows: 22
- Rows with `exists=True`: 19
- Common coverage for completed rows: 182 problems / 910 scored samples (`n_samples=5`)
- Invalid or incomplete rows are preserved with their `stage` labels, especially
  `trained_eval_invalid_port_collision` and `exists=False`.

## Selected rows

| label | stage | FrontierCS best@5 mean | ALE-Bench best@5 mean | complete problems |
|---|---|---:|---:|---:|
| qwen35_9b_base | base_eval_reference | 2.6976 | 400.8188 | 182 |
| qwen35_9b_base_model | base_eval_reference | 0 | 286.5 | 182 |
| qwen35_9b_mixed_no_thinking | trained_eval_reference | 0 | 354.9387 | 182 |
| qwen35_9b_mixed_thinking | trained_eval_reference | 7.4128 | 606.1506 | 182 |
| qwen35_9b_mixed_step20_thinking | trained_eval | 0 | 383.1549 | 182 |
| qwen3_8b | base_eval | 0 | 639.4286 | 182 |
| qwen3_8b_mixed_public_step25 | trained_eval_fallback | 0 | 736.5447 | 182 |

The CSV also records model paths, summary JSON paths, served model tags, partial/error counts, and
sample JSONL source paths from the original scratch run. Those paths are provenance only unless the
same GPFS workspace still exists.
