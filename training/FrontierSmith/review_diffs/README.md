# Review-only changes for pipelined LLM evaluation

These are intentionally not applied to live Slurm or server scripts.

* `enable_opt_in_prefix_caching.patch` leaves the existing default unchanged
  (`ENABLE_PREFIX_CACHING=0`) and accepts only `0` or `1`.  A new MLS-Bench or
  Research job can opt in by exporting `ENABLE_PREFIX_CACHING=1`; do not opt in
  for comparability-sensitive FCS/ALE reruns without first running a seeded
  output-hash canary.
* `request_12_cpus_per_gpu.patch` changes only the two evaluation Slurm CPU
  requests from 8 to 12.  It makes no changes to judging or sampling.
* The new executable is `scripts/eval_qwen35_base_vllm_request_pipelined.py`.
  Substitute it for `scripts/eval_qwen35_base_vllm_request.py` in an as-yet
  unsubmitted copy of an eval launcher, retaining all of the existing request
  arguments and adding `--score-concurrency 6` (or an explicitly chosen
  smaller judge-safe value).  It refuses iterative FrontierCS because that
  protocol depends on judge feedback for subsequent requests.
