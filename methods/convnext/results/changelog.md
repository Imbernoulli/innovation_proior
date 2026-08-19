# Changelog

## 2026-08-18 — obs-fix(epistemic)
The source_value_audit obs-scan (obs_scan_v3.jsonl, V-verdict) flagged reasoning.md as a
narrated multi-rung ablation roadmap written as completed ImageNet-1K training runs — e.g.
"This alone moves the ResNet-50 from 76.13 to 78.82", "the experiment confirms the first
half: depthwise alone drops accuracy to 78.28", "kernel size 3 gives 79.92. Kernel 5 gives
80.35. Kernel 7 gives 80.57" — plus a matching full results table in answer.md and two
result-as-accomplished-fact sentences in train_answer.md. A single-turn proposal has not
run its own experiments yet, real historical numbers or not.

Rewrote all three channels:
- `reasoning.md`: every narrator-run training outcome (recipe retrain, stage-ratio,
  patchify stem, depthwise-alone, width increase, inverted bottleneck, moved-up depthwise,
  the 5-point kernel sweep, GELU, activation sparsity, norm sparsity, LayerNorm swap,
  separate downsampling divergence-and-fix, and the entire ResNet-200/Swin-B-regime
  re-run) is now hypothesis -> matched-budget test DESIGN -> PREDICTION -> decision rule.
  The training-instability "wall" for bare separate downsampling is reframed from an
  encountered failure into an anticipated risk that is preemptively designed around
  (boundary LayerNorms added before ever training the bare variant). Kept untouched: all
  on-page deterministic computation (the depthwise-vs-dense MAC count, the channel-vs-
  spatial 1x1-vs-7x7 MAC count, the stem downsampling grid arithmetic, the tensor shape
  trace, the LayerScale near-identity numerical check) — these are desk checks, not
  claimed training runs. The closing paragraph now states the ConvNeXt landing as a
  falsifiable prediction contingent on the matched-budget ladder clearing, not an
  accomplished result.
- `answer.md`: replaced the 18-row completed-ablation results table (top-1/GFLOPs at every
  step, including the Swin-T/Swin-B reference comparison) with the same step sequence
  reframed as the validation PLAN — one design axis per step, matched-budget protocol,
  explicit decision rule, run once per compute regime. Final block spec, stage configs,
  code, and training hyperparameters are architecture/recipe description, not results, and
  are unchanged.
- `train_answer.md`: "The result ... is surprising: the residual ConvNet closes the
  controlled gap ... without ever adding self-attention" -> "The prediction ... is a
  surprising one: that the residual ConvNet closes the controlled gap ..."; "ConvNeXt thus
  demonstrates that ... a ConvNet can match or exceed ..." -> "ConvNeXt is proposed on the
  claim that ... the matched-budget validation chain ... is what would decide whether that
  claim holds." Mechanism/design content and code (verbatim from answer.md) unchanged.

`context.md` left untouched — none of the removed numbers appear there as pre-existing
facts; the only numbers context.md carries are the ~4.5/~15 GFLOPs compute-regime bands,
which are retained as given context in the answer.md validation-plan rewrite.

This is a genuine multi-rung improvement ladder (12 design steps re-run at two compute
regimes), not a single ablation decision — flagged for trajectory-track conversion to
carry the real runs and their results.
