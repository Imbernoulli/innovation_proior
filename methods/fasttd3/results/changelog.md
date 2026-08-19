# Changelog

## 2026-08-18 — svfix(W3_ancestors_only)
Decisive-step fix: the "no architecture" bet (drop LayerNorm/residual stabilizers,
bet on data diversity taming the deadly triad) was staged as a falsifiable prediction
in reasoning.md but never closed the loop with an observed result — it just said
"Pending that test, I'll go with..." The primary paper (src/neurips_2025.tex, the
Architecture paragraph) already reports the actual ablation outcome: residual paths
+ LayerNorm "tend to slow down training without significant gains," attributed to
data diversity from parallel sim + large batch reducing effective off-policyness.
Rewrote the decisive-step paragraph so it reports that real observed result (slower
training, no significant return gain) as the outcome of the stated test, instead of
leaving the prediction unconfirmed. Touched two downstream summary sentences for
consistency (no longer call the no-stabilizers bet "still pending" after it's
reported checked). No factual errors found; no change to the landing (final method
or code). See notes/sources.md for the grounding quote.

## 2026-08-18 — svfix(epistemic)
Reverted the above svfix(W3_ancestors_only) edit to reasoning.md. A single-turn
method unit is a proposal: at that point in the frame, this method's own
experiments have not happened yet, so the narrator cannot claim to have run the
architecture ablation and report its outcome — even though the outcome quoted was
a real number from the primary paper's own reported ablation, attributing it to
the proposal's first-person voice as something "I run" / "I check directly" /
"I've already checked" puts the method's own result in the narrator's mouth before
it exists in the frame. Restored the three touched passages (LayerNorm/residual
decisive step, the verified-vs-open summary, and the causal-chain closer) to
hypothesis + falsifiable prediction + decision rule ("Pending that test, I'll
go with...", "the central empirical bet, stated so a run can falsify it"), with
no reported observation and no numbers. reasoning.md is now byte-identical to
its pre-svfix(W3_ancestors_only) state. The real ablation number this fix removed
belongs in a trajectory observation turn, not in proposal-voice reasoning.md.
