# changelog — missforest

## 2026-08-18 (obs-fix: self-supplied observation, R -> V)
`obs_scan_v3.jsonl` flagged (`abl_shows`) a sentence in `reasoning.md`'s `m_try` discussion:
"(I'll keep in mind this small experiment shows the case against large `m_try` is
decorrelation and cost, *not* a raw-accuracy cliff — on this table large `m_try` was
marginally the most accurate — so the argument for `sqrt(p)` is genuinely a balance...)".
That sentence sat downstream of a larger passage narrating a completed run: "On a 9-column
synthetic table (so `sqrt(p) ≈ 3`) I ran the full imputer at three settings of `m_try` and
read off the final NRMSE: m_try=1 -> 0.820, m_try=3 -> 0.746, m_try=8 -> 0.732."

Adjudicated as V, not desk-scale: this isn't a hand-traceable micro-example — it requires
actually running the full iterative missForest imputer (multiple ~100-tree random forests,
round-robin sweeps to convergence) at three hyperparameter settings and reading off NRMSE to
three significant figures, i.e. a genuine (if small) ablation of the method's own key
hyperparameter, which is exactly the "our ablations show..." pattern the rule bans regardless
of dataset size.

Fixed by rewriting the passage in `reasoning.md` to keep the three-point design space
(`m_try ∈ {1, floor(sqrt(p)), p}`, matched table/`n_tree`/stopping rule) as a controlled-test
DESIGN, turn each point into an explicit PREDICTION grounded in Breiman's strength/correlation
bound (m_try=1 worst by a wide margin; sqrt(p) captures most of the gain; m_try=p only a small
further gain at higher split-search cost), and state the DECISION RULE (ship sqrt(p) if it
lands close to the m_try≈p result while m_try=1 is clearly worse — captures the bulk of the
achievable accuracy without paying the full-search price). All NRMSE numbers removed; the
mechanism argument (strength/correlation tradeoff, why `sqrt(p)` is a balance point and not a
"bigger is worse" claim) is preserved. `answer.md`/`train_answer.md` already stated the
`m_try = floor(sqrt(p))` default via the Breiman-bound mechanism argument with no narrator-run
numbers — no changes needed there.

Verification: `python3 tools/lint_inframe.py | grep methods/missforest/` — no hits.
`experiments/source_value_audit/obs_scan_v3.jsonl` is a stale, untracked snapshot that
predates this batch; the `tools/obs_scan.py` present in this checkout writes a differently-
formatted `obs_scan_hits.jsonl` (pattern names `V1_*`, not `abl_shows`) and does not
regenerate `obs_scan_v3.jsonl` at all — confirmed against `lookahead`/`chain-of-thought`/`awq`,
whose already-fixed passages are still absent from their live files despite still showing as
flagged in the stale `obs_scan_v3.jsonl`. Verified directly instead: grepped the flagged quote
and all NRMSE/`m_try` numeric strings across `reasoning.md`/`answer.md`/`train_answer.md` —
no hits.
