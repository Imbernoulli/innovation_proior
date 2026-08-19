# changelog — acon

## 2026-08-18 (svfix repair pass, D_candidate)
Prior fix commit `8979662da` ("svfix(D_candidate): bpe — Sennrich subword-nmt GH issue #19...")
bundled this method's decisive-step grounding together with an unrelated `bpe` fix in a single
commit, violating the ONLY-touches-SLUG rule. Content was verified correct by an independent
reviewer (quote checked against `acon_arxiv_v2_cvpr.tex`, decisive step depends on the
39.4/37.2/36.3/34.8 meta-ACON granularity-ablation numbers and the pooling-vs-no-pooling
mechanism, no filler/leak/lint hits, answer.md unchanged) — only the commit scope was flagged.
This entry accompanies a follow-up commit that touches `methods/acon/` alone, separating the
acon fix's provenance record from the bpe commit going forward. No content in reasoning.md,
answer.md, or train_answer.md changed in this pass.

Grounding recap (unchanged from the original fix): `notes/sources.md` Source 1 — the primary
paper's own "Design space in meta-ACON" ablation table (identical in both the Sept 2020 preprint
and the Apr 2021 CVPR camera-ready), which shows pixel-wise routing (finest granularity, most
information, zero extra params) losing to layer-wise (37.2% vs 36.3% top-1 error) because
`σ(x_{c,h,w})` has no spatial pooling to inform the per-pixel gate, while channel-wise wins
outright (34.8%) by pooling over space *per channel* before deciding — grounds the granularity
choice at `methods/acon/results/reasoning.md` line 79 in a real, checkable ablation rather than
an SE-module analogy asserted a priori.

## 2026-08-18 (epistemic correction pass)
The grounding above was correct on facts but written in the wrong voice for a single-turn
PROPOSAL: `reasoning.md` had the narrator *run* the granularity ablation in-line ("I try all
three on ShuffleNetV2 0.5×... gets to 36.3%... gets only to 37.2%... The design that actually
wins is channel-wise, at 34.8%"), i.e. reporting the method's own experimental outcome as
something already observed, which this frame does not allow — the method's own results belong
only in a separate trajectory-observation turn, not the proposal. Fixed by rewriting the passage
to keep the three-way granularity question, the naive "finer must be better" instinct, each
design's a-priori prediction (pixel-wise risks a no-pooling noisy gate; layer-wise pools but
conflates channels; channel-wise pools per-channel), the matched-conditions experiment design
(same ShuffleNetV2 0.5× backbone, same budget, same ReLU reference point), and an explicit
decision rule (ship whichever granularity comes out lowest under that comparison) — while
removing the claimed observation and the 39.4/37.2/36.3/34.8 numbers from proposal voice. The
numbers and the mechanism stay truthfully recorded above in this changelog and in
`notes/sources.md`; this unit now needs a trajectory-observation turn to supply them as an
actual result. No changes to answer.md or train_answer.md were needed (svfix diff touched only
reasoning.md).
