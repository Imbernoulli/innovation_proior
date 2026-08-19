# Changelog — toolformer

## 2026-08-18 — obs-fix
- **reasoning.md, answer.md, train_answer.md all stated the finetune's
  compute protocol and checkpoint choice as an already-completed run**:
  "ZeRO-3 with BF16 on eight A100 40GB GPUs; I train up to 2k steps, check
  held-out CCNet perplexity every 500 steps, and keep the checkpoint with
  the best perplexity" (and the matching clauses in answer.md/train_answer.md)
  read as a narrator-run training outcome — a specific checkpoint already
  identified as best — when at proposal time this training has not happened
  (obs_scan `run_num` flag: "8xA100 training protocol + checkpoint selection
  claimed done").
- Rewrote all three to plan/decision-rule voice, keeping every design detail
  (hardware, batch size, LR, warmup, step budget, eval cadence): the compute
  plan is stated as a plan ("the engineering plan is ..."), and checkpoint
  choice is reframed as a selection *rule* — keep whichever checkpoint scores
  lowest on held-out CCNet perplexity, checked every 500 steps, rather than
  automatically the final one — motivated by the risk that finetuning on a
  self-generated, tool-heavy corpus could regress on ordinary text partway
  through. No numbers invented or removed; no result (an actual observed
  best-checkpoint value) is claimed anywhere.
- context.md carries no matching claim; left untouched.
