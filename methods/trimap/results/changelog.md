# trimap changelog

## 2026-08-20 — svfix repair (W3_ancestors_only): ground the decisive step in the PaCMAP structural finding

A prior verifier rejected a `sound_as_is` claim on this method: `notes/explainer_capture.md` already
recorded that the PaCMAP re-analysis (arXiv:2012.04456) is the source of the decisive step's
mechanism ("global structure comes from PCA init, not the triplet loss"), reconstructed in-frame, but
`results/reasoning.md` never actually depended on that material — the step ran solely on the trace's
own S-curve wall experiment (0/11132 violations, curved vs. flattened tie) plus an unconnected
first-principles assertion about what a PCA start buys.

Checked for an independent route through the primary TriMap paper first (`refs/trimap_v1_1803.pdf`):
it uses PCA only as a 50-D pre-reduction step before the kNN search, never as the embedding init, and
its own stated source of global-structure preservation is the **random triplets**, not PCA init — so
no primary-derivable substitute exists.

Fixed by surfacing the PaCMAP dependency explicitly, in-frame, using only its non-empirical content
(the structural claim that a triplet only carries far-field information when *both* compared points
are far from the reference point — never an empirical/ablation outcome, which would postdate TriMap
and can't be self-narrated as an observed result per the project's hard rule):
1. After the S-curve/flattened-line tie (para "But wait — let me check..."), added a generalization:
   the tie isn't a coincidence of the one example — every neighbor triplet has `j` near `i` by
   construction, so none of them has the "both far from `i`" shape needed to carry far-field
   information; the whole neighbor-triplet set is structurally blind to global arrangement.
2. Tightened the "triplets only sharpen locally" claim in the PCA-init paragraph to rest on the
   trace's own already-derived force-magnitude fact (per-triplet pull/push shrinking toward zero as a
   triplet saturates) instead of an unconnected assertion that "there's essentially no work left."

Landing (PCA init + saturating triplet refinement, the code) is unchanged — this was a grounding fix
at the decisive step, not a factual correction. Full source record: `notes/sources.md` (quote, file,
line numbers, and what was deliberately excluded and why). Lint clean
(`tools/lint_inframe.py`, categories A_paren/B_meta/E_paperref/C_rsn_header — no hits);
`grep -i "the paper|the authors|arxiv|et al\."` — no new hits; `tools/obs_scan.py` shows no new hits
introduced by this edit (the one pre-existing `trimap` hit, about the 2000-random-projections check,
is unrelated and already guarded). Diff limited to `results/reasoning.md`.
