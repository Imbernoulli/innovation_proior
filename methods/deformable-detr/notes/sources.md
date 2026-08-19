# Sources — deformable-detr (svfix W3_ancestors_only)

## Decisive step identified
Predicting sampling offsets Δp_mqk AND attention weights A_mqk directly from the
query z_q via one linear projection — dropping query-key dot-product attention
entirely (reasoning.md, single-scale deformable attention paragraph, right after
Eq. 2). This is the design that makes the module both cheap (no need to touch any
key content to get a weight) and the "deformable-conv analogue."

## Defect found
The trace justified dropping the dot product with a fabricated logical necessity
("circularity": weight depends on location, location is a free query-predicted
parameter, so "there's no fixed key set sitting there to compare against" /
"costs exactly what I was trying to avoid"). This is not actually true: the
offsets, once predicted, DO fix a location; bilinear-sampling content there for a
dot-product weight is no more expensive in FLOPs than the sampling the value
branch W'_m x(p_q+Δp) already pays for. The primary paper (src/src/3-method.tex
L64-76) states the design (Eq. 2) with zero justification for why weights are
query-only rather than query-key — it is silent on this exact fork, so the trace's
"circularity" framing is invented, not sourced from the primary, and (per my own
recomputation above) not even correct as physics/complexity.

## Source found (web hunt, OpenReview)
type: self-account (paper's own authors, official rebuttal comment thread)
venue: OpenReview, ICLR 2021 Paper1041 (Deformable DETR forum id gZ9hCDWe6ke)
retrieval: `https://api.openreview.net/notes/search?term=Deformable%20DETR&content=all&group=all&source=all&limit=100`
   (forum-filtered to gZ9hCDWe6ke; per-note ids below)
local file: refs/self_accounts/openreview_iclr2021_paper1041_carion_thread.txt

Public_Comment (note id x1VT5henOtF), title "Is deformable attention an attention
mechanism?", from Nicolas Carion — DETR's own first author — posted as a public
reviewer/commenter on this exact fork:
> "in the proposed formulation, the $A_{mqk}$ could very well be computed as a
> dot-product as well (between the query and each of the sampled point), making
> it a 'true' attention mechanism. Have you tried such thing?"

Official_Comment (note id WoBeM7mA97O), "Reply to Nicolas Carion (1/2)", the
Deformable DETR authors' answer, A#1:
> "Yes, we tried using dot-product to obtain the attention weight in some early
> experiments, where K=1 and other design choices are very similar with the
> default setting of Deformable DETR. It achieves on par performance (AP
> difference <0.5%) compared with that of linear projection. However, we
> experimentally found that using dot-product results ~25% slower speed than
> that of linear projection. Therefore, we choose to obtain the attention weight
> by linear projection for efficiency."
> "In terms of speed, using dot-product has the same computational complexity as
> the linear projection. The inefficiency of the dot-product may be related to
> the implementation... the dot-product requires additional random memory access
> for sampling key features and the batch matrix-matrix product... Meanwhile,
> for linear projection, we only need to compute a matrix multiplication between
> the query features... and the weights of linear projection..."

This confirms: (a) dot-product attention on the sampled points is architecturally
available and was actually tried (no logical circularity — my fabricated
framing was wrong); (b) it is an EMPIRICAL fork (accuracy parity, ~25% slower,
same FLOP-order but worse memory-access pattern), not a derivable one; (c) the
authors' own words note complexity is identical, the gap is implementation
(gather + batched matmul vs one dense matmul).

## Fix applied
Per the empirical-decisive-step rule, the method's own observed numbers
(AP diff <0.5%, ~25% slower) may NOT appear inline in reasoning.md as a claimed
self-run observation. Rewrote the paragraph as hypothesis (content-based weight
vs. query-only weight) -> why complexity alone can't decide (sampling already
paid for by the value branch either way) -> matched-test design (K=1, same
offsets, one arm dot-products against the sampled content, the other reads the
weight off the same linear projection) tracking both accuracy and wall-clock
throughput, flagging the real, checkable candidate cost (gather + batched matmul
vs. one dense matmul) -> decision rule (ship whichever wins on speed without an
accuracy hit). Landing (linear-projection-only, no dot product) is unchanged —
matches both the primary's Eq. 2 and the authors' own account of what shipped.
