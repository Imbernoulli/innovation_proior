# Evidence snapshot: 2026-08-15

This report is review-only. No running-job file was edited and no Slurm command
that changes state was run.

## Engine configuration and occupancy

* FCS/ALE job `12427130`: `logs/cce-rlv10_base_s20-s0p1-12427130.out:12`
  records `max_num_seqs: 128`; `:20` records
  `enable_prefix_caching=False`; `:30` records FLASH_ATTN. The eval client
  announces `concurrency=96` at `:104`, then the engine reaches 96 running,
  zero waiting at `:114`. The early 40.6 tok/s point at `:111` follows the
  explicit JIT warnings at `:105-110`, so it is not evidence of judge scoring.
  Later, the same job reaches 88.2% KV at `:826`; low initial KV is not a
  global explanation of that run.
* Research job `12426793`: `logs/cc-eval-9b-res-anchor_loraIM-s1-12426793.out:12`
  records `max_num_seqs: 64`, `:19` records prefix caching off, and `:99`
  announces client `concurrency=32`. It reaches 32 running at `:104`. With
  80 of 320 samples still outstanding (`:796`), it falls to 4 running,
  zero waiting and 6.5% KV (`:813-815`).

The original evaluator serializes a request and score in one worker at
`scripts/eval_qwen35_base_vllm_request.py:514-575`, and its coupled executor
is at `:1102-1110`. FCS scoring synchronously submits and polls
(`.cache/Frontier-CS-official/src/frontier_cs/runner/algorithmic_local.py:154-203`,
`:295-326`); Research scoring calls `subprocess.run` in
`scripts/frontiercs_research_eval.py:545-548` and
`scripts/frontiercs_research_cpu_eval.py:442-445`.

## Exact aggregate timing calculation

The following command was run against the displayed JSONL files, selecting
only records with `error == null` and defining serial score wait as
`total_seconds - generation_seconds`:

```sh
jq -s 'map(select(.error == null)) as $r |
  ($r | map(.total_seconds) | add) as $total |
  ($r | map(.total_seconds - .generation_seconds) | add) as $score |
  {n: ($r|length), total_worker_seconds: $total,
   score_wait_worker_seconds: $score, score_wait_share: ($score/$total),
   no_overlap_speedup_upper_bound: (1/(1-($score/$total)))}' FILE
```

* Completed FCS/ALE shard
  `outputs/cc_eval_rlv10_base_s20_thinking_32k_both_vllm/shard_0/samples.jsonl`:
  `n=451`, `total_worker_seconds=288345.3358018398`,
  `score_wait_worker_seconds=4846.4038116931915`, share
  `0.016807637266669007`, and unchanged-generation arithmetic `1.0170949632x`.
  This rules out scoring serialization as the dominant *aggregate* time in
  that completed shard, despite its real ability to produce short underfeed
  intervals.
* Research JSONL was still growing when read. The snapshot of
  `outputs/cc_eval_anchor_loraIM_research_thinking_32k_vllm/shard_1/samples.jsonl`
  was `n=290`, `total_worker_seconds=62543.70718169212`,
  `score_wait_worker_seconds=19316.651049613953`, share
  `0.3088504330819128`, unchanged-generation arithmetic `1.4468648291x`.
  It is a snapshot, not a completed-run claim.

Raw fields used by the calculation occur in each JSONL record beginning at
line 1. The FCS/ALE scheduler log identifies that output location at
`logs/cce-rlv10_base_s20-s0p1-12427130.out:3`.

## CPU-allocation observation

Read-only `scontrol show node della-l07g2` returned `CPUTot=48` and
`Gres=gpu:a100:4`; the node therefore has 12 CPU cores per A100 by arithmetic.
Read-only `sacct -X -j 12427130,12426793` reported `AllocCPUS=8` and
`ReqCPUS=8` for both eval jobs (the latter was still RUNNING at the snapshot).
The two launchers each currently request 8 at
`slurm/cc_eval_thinking_both_autopart.sh:28` and
`slurm/cc_eval_research_autopart.sh:32`.

## Validation of new artifacts

* `git apply --check` passed for both patch files.
* `python -m py_compile scripts/eval_qwen35_base_vllm_request_pipelined.py`
  passed.
* Its CPU-only smoke test printed:
  `PIPELINE_SMOKE_PASS: all 12 generations completed before the first slow score finished`.
