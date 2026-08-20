# Changelog

## 2026-08-19 — svfix(W3_primary_plus_ancestors) verification pass
- Quality-gate review of the decisive step TRIAGE flagged (class D): the
  factorization of a standard convolution (`D_K^2 M N D_F^2`) into a
  depthwise spatial pass (`D_K^2 M D_F^2`) plus a pointwise `1x1` channel-mix
  pass (`M N D_F^2`), landing on the `1/N + 1/D_K^2` reduction ratio.
- Confirmed genuinely derived, not asserted: reasoning.md tries depthwise
  alone first, then tests it on a worked toy case (4x4 input, M=2 channels,
  N=3 mix target) and hand-verifies it cannot create a new channel
  combination (zeroing input channel 0 leaves depthwise channel 1
  bit-for-bit unchanged) — a concrete, checkable reason that forces the
  pointwise step, not a stated design choice. The cost-reduction cancellation
  is then checked two independent ways (symbolic algebra and raw
  multiply-add counts on the primary's own `D_K=3,M=N=512,D_F=14` layer, both
  landing on `0.11306`), and the `alpha`/`rho` scaling laws are each
  cross-checked against raw counts the same way.
- Cross-checked against the primary source already on disk
  (`src/main.tex` line 138, arXiv 1704.04861 Sec. 3.1): "Depthwise
  convolution is extremely efficient relative to standard convolution.
  However it only filters input channels, it does not combine them to
  create new features. So an additional layer ... via 1x1 convolution is
  needed in order to generate these new features." This is the primary's
  own statement of the exact obstacle the trace's toy example reconstructs.
  Table 3's numeric example (462M/2.36M -> 52.3M/0.27M -> 29.6M/0.15M ->
  15.1M/0.15M) matches the trace's independently recomputed numbers to the
  reported precision, and the primary's own "8 to 9 times less computation"
  matches the trace's `8.8x`.
- Checked both on-disk non-primary sources TRIAGE flagged as unused (Google
  Research blog self-account, Hugging Face course explainer) in full before
  deciding not to graft either in: the blog is a release announcement
  (checkpoint table, deployment framing, pointer back to the primary) with no
  discussion of the depthwise/pointwise split or any failed-attempt content
  beyond what the primary already states; the HF explainer is a lower-rigor
  pedagogical restatement of the same split that also mixes in
  MobileNetV2-era vocabulary ("channel-wise linear bottleneck layers"),
  which postdates this method. Neither adds anything the primary/trace does
  not already establish; grafting either in would be a decorative citation,
  not a grounding.
- No factual errors found. The trace's stride-1 call on the final `1024`
  block (reasoning.md line 65) already matches the official TF-Slim
  `CONV_DEFS` and the retained `7x7` map; left unchanged.
- No rewrite to reasoning.md, answer.md, or train_answer.md; no new source
  grafted. See `notes/sources.md` for the full write-up and quotes.
