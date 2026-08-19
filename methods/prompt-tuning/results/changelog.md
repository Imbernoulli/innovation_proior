# prompt-tuning changelog

## 2026-08-18 — epistemic correction (svfix observation removal)
- `results/reasoning.md` (span-corruption base-model paragraph): an earlier svfix pass had the
  narrator "go run it and look," reporting a Small-beats-Base/Large/XL anomaly, a three-runs-per-size
  noise check, and a mid-size 0%/copying/empty-string failure "stable across the three runs" — all
  own-experiment observations a single-turn proposal has not yet earned (the method's own results
  belong only in trajectory observation turns). Rewrote the paragraph in predictive voice: kept the
  hypothesis (span-corruption base should be a bad frozen prompting target), the concrete failure
  mode as a PREDICTION to be matched or broken by a real run, and the noise-vs-anomaly control
  (repeat before trusting a reversal) as experiment design rather than a reported result. Removed all
  claimed numbers and run counts. Downstream discriminating experiment (three frozen bases: raw /
  sentinel-prepend / LM-adapted, predicted ordering LM-adapt ≫ sentinel-prepend ≈ raw) was untouched
  by the original svfix diff and is unchanged. `answer.md` / `train_answer.md` had no svfix diff for
  this method, so no violation there. Landing (LM-adapt once, then prompt-tune) unaffected — it was
  already stated as a plan pending the discriminating experiment, not a reported outcome.
