# Sources — decisive-step sourcing check (svfix, W3_primary_plus_ancestors)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines 15-37: factorizing the standard convolution
`D_K^2 M N D_F^2` into a depthwise (spatial-only) pass `D_K^2 M D_F^2` plus a
pointwise (channel-mix-only) `1x1` pass `M N D_F^2`, landing on the reduction
ratio `1/N + 1/D_K^2` (~8.8x at the primary's own representative layer). TRIAGE
flagged class D and noted the self-account (Google Research blog) and the
Hugging Face explainer are on disk but unused.

## Quality-gate verdict: SOUND_AS_IS — no rewrite made

### (a) Genuinely derived on the page, not asserted
This is not a hindsight restatement of the finished design; it is checkable
arithmetic the trace works out itself, including a self-administered failure
check on the "obvious" first move:
- reasoning.md tries the half-measure first ("Let me try keeping only the
  spatial part first") — depthwise alone — then tests it on a worked toy
  case (4x4 input, M=2 channels, one 3x3 filter each, later mixed to N=3):
  it hand-verifies the depthwise output stays 2 channels not 3, and that
  zeroing input channel 0 leaves depthwise channel 1 bit-for-bit unchanged
  while channel 0 changes. That is a concrete, checkable reason depthwise
  alone cannot do the whole job ("this pass filters space but cannot create
  a single new channel mixture"), which is what forces the pointwise step —
  not an asserted design choice.
- It then verifies pointwise supplies exactly the missing capability on the
  same toy case (`G[0,0,0] = sum_m Ghat[0,0,m] Kpw[m,0]`, i.e. every output
  channel is a combination of all M depthwise channels).
- The `1/N + 1/D_K^2` cancellation is checked two independent ways: symbolic
  algebra AND raw multiply-add counts on the primary's own representative
  layer (`D_K=3, M=N=512, D_F=14`), both landing on `0.11306`.
- The `alpha` (width) and `rho` (resolution) scaling laws are each derived
  symbolically and then cross-checked against raw counts at `alpha=0.75`
  and `rho=10/14`, not just asserted.
No step here states an experimental/empirical outcome as logically
necessary, and the think-voice never self-supplies a training-run result —
it is pure, checkable algebra plus a hand-traced toy example and repeated
raw-count cross-checks, which is exactly what the quality gate calls a
legitimate derivation (not the empirical-outcome case the gate warns about).

### (b) Backed directly by the primary paper's own text and numbers
The primary source states the exact obstacle the trace's toy example
reconstructs, almost word for word:

> "Depthwise convolution is extremely efficient relative to standard
> convolution. However it only filters input channels, it does not combine
> them to create new features. So an additional layer that computes a
> linear combination of the output of depthwise convolution via 1x1
> convolution is needed in order to generate these new features."
(`methods/mobilenet/src/main.tex` line 138, and identically
`refs/primary/source/main.tex` line 138 — this is arXiv 1704.04861,
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications", Sec. 3.1.)

The primary's Table 3 also carries the exact numeric example the trace
independently re-derives and matches to the reported precision: full
462M/2.36M, depthwise-separable 52.3M/0.27M, alpha=0.75 -> 29.6M/0.15M,
rho=0.714 -> 15.1M/0.15M mult-adds/params (`refs/primary/1704.04861.pdf`,
Table 3; extracted text checked with `pdftotext -layout`). The trace's
`8.8x` figure matches the primary's own "8 to 9 times less computation"
claim (same section). No factual error found — the trace's stride-1 call on
the final `1024` block (reasoning.md line 65) already matches the official
TF-Slim `CONV_DEFS` and the primary's own retained `7x7` map (a stride typo in
the primary's Table 1 that an earlier pass on this method had already
corrected; unchanged here).

Condition (b) is satisfied twice over: by the primary's own explicit
sentence, and by the trace's own honest, re-checkable arithmetic — either
alone would suffice.

## Why the self-account and explainer were checked and deliberately left out
Both were read in full (`refs/self_accounts/google_research_blog_mobilenets.html`,
`refs/explainers/huggingface_course_mobilenet.mdx`) before concluding
sound_as_is, not skipped because TRIAGE class was D:
- **Google Research blog** (Howard & Zhu, June 2017) is a release
  announcement, not a retrospective: deployment framing, a checkpoint table
  (MACs/params/accuracy per width x resolution variant), a pointer back to
  the primary "for technical details," and acknowledgements. It contains zero
  discussion of the depthwise/pointwise split, no failed-attempt narrative,
  no obstacle language distinct from what the primary already states. There
  is nothing here that would deepen or change the decisive step; citing it
  would be a name-drop, not a grounding.
- **Hugging Face course explainer** is a third-party pedagogical restatement
  ("imagine a sponge," "a tiny dot") of the same depthwise/pointwise split,
  at lower rigor than the trace's own toy-example verification, and mixes in
  MobileNetV2-era vocabulary ("channel-wise linear bottleneck layers") that
  postdates this method (source_matrix.md already flags this cross-version
  contamination). Grafting it in would both add nothing the primary/trace
  doesn't already establish and risk pulling in an anachronistic V2 concept.
Threading either source into the decisive step would be exactly the
wave-2 mistake the fix prompt warns against: a bolted-on citation the
reasoning doesn't need, added only to pad provenance count.

## Search performed
- Re-read reasoning.md in full, notes/synthesis.md, notes/source_matrix.md,
  notes/discovery_synthesis.md.
- `grep` of the primary's `.tex` source (both `src/` and `refs/primary/source/`
  copies) for the depthwise-cannot-mix sentence — found verbatim, Sec. 3.1.
- `pdftotext -layout` extraction of `refs/primary/1704.04861.pdf` to confirm
  Table 3's numbers against the trace's own recomputation.
- Full read of both on-disk self-account/explainer files (see above) — ruled
  out as adding decisive-step content, not just checked for existence.
- `grep -i mobile SELF_ACCOUNT_SOURCES.md` at repo root: no entry for this
  method (one unrelated McClintock "mobile controlling elements" hit only) —
  the Google Research blog self-account on disk was located independently by
  an earlier pass, not sourced from this registry.

## Conclusion
outcome = sound_as_is. No rewrite to reasoning.md, answer.md, or
train_answer.md. No new source grafted onto the decisive step.
