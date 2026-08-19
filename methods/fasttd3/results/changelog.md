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
