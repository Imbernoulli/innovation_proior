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
  by the original svfix diff and is unchanged. Landing (LM-adapt once, then prompt-tune) unaffected —
  it was already stated as a plan pending the discriminating experiment, not a reported outcome.
  (Note: the original entry here claimed "`answer.md` / `train_answer.md` had no svfix diff for this
  method, so no violation there" — that inference was wrong; see the follow-up entry below. Absence
  of an svfix diff does not mean the pre-existing content in those files was already clean.)

## 2026-08-18 — repair pass: violation persisted in answer.md / train_answer.md
A verifier rejected the prior entry above: `results/train_answer.md` (never touched by any svfix
commit, so out of the strict svfix-diff scope, but carrying the identical violation class in its
pre-existing prose) still reported the base-model failure as an observed fact ("that is exactly what
happens... stable across runs rather than noise") and stated the three-base discriminating result as
already known ("the LM adaptation is what fixes the cause"). `results/answer.md` carried a milder
version of the same two claims. Re-reading both files against the now-corrected `reasoning.md` showed
the violation was systemic, not confined to the two flagged sentences: the "central finding" framing
of the XXL scale claim, and the initialization- and prompt-length-knob paragraphs, all stated the
method's own not-yet-run results as settled fact in both files. Rewrote every such passage in both
files to predictive/hypothesis voice (mirroring `reasoning.md`'s already-fixed phrasing: "I expect /
should / the prediction is / a claim the sweep is built to test"), keeping every hypothesis, the
concrete failure-mode prediction, the three-base discriminating-experiment design and its predicted
ordering (LM-adapt ≫ sentinel-prepend ≈ raw), the init and length-knob predictions, and the landing
(LM-adapt once, then prompt-tune) exactly as before. No numbers, code, algebra, or non-violating
content removed or added. `python3 tools/lint_inframe.py` — zero hits for prompt-tuning; no
out-of-frame source-attribution leaks introduced.
