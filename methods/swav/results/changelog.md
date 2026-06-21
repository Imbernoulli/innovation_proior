# SwAV review/fix changelog

- `methods/swav/results/context.md:17` tightened the pre-method OT scaffold: normalized row/column marginals are `1/K` and `1/B`, and only after multiplying by `B` do columns become ordinary per-sample codes.
- `methods/swav/results/reasoning.md:34` corrected the derivation narrative for normalized OT mass versus post-rescale assignment mass.
- `methods/swav/results/reasoning.md:40` fixed the feasible-set notation from `Q` to `𝒬`.
- `methods/swav/results/reasoning.md:57` added the distributed Sinkhorn detail that the official implementation all-reduces total/row sums and returns codes after `Q *= B`.
- `methods/swav/results/reasoning.md:59` removed past-tense hindsight around hard-vs-soft assignment and recast it as in-frame reconstruction.
- `methods/swav/results/reasoning.md:63` clarified that equipartition is `1/K` normalized mass, equivalent to `B/K` assignment weight after rescaling.
- `methods/swav/results/reasoning.md:75` corrected prototype normalization timing to match the reference loop: normalize vectors before use on the next iteration.
- `methods/swav/results/reasoning.md:80` replaced the simplified local Sinkhorn/training snippet with code faithful to the official distributed path: `distributed_sinkhorn(out, args)`, all-reduce fallback, queue-only assignment augmentation, `np.sum(args.nmb_crops)`, `F.log_softmax`, and `freeze_prototypes_niters`.
- `methods/swav/results/answer.md:23` made the same OT mass correction in the final method statement.
- `methods/swav/results/answer.md:31` corrected prototype normalization wording to avoid paper-column/PyTorch-row ambiguity.
- `methods/swav/results/answer.md:43` added distributed helpers used by the final code artifact.
- `methods/swav/results/answer.md:79` replaced local `sinkhorn(scores, ...)` with `distributed_sinkhorn(out, args)` mirroring the canonical implementation.
- `methods/swav/results/answer.md:97` aligned the training loop with the official reference implementation: epoch-indexed LR schedule, assignment-only queue, full-res crops for codes, `nmb_crops`, and prototype freeze naming.
- `methods/swav/results/.codex_review.json:2` replaced the stale rate-limit review state with the completed manual audit metadata and explicitly recorded that the independent strict-check gate was unavailable.
- `methods/swav/notes/source_matrix.md:1` added the source-by-source evidence matrix covering primary, supplement, ancestors, explainer, author-side thesis, and canonical code checkout.
- `methods/swav/notes/discovery_synthesis.md:1` added audit synthesis notes for math signs/constants, implementation faithfulness, leak/scaffold review, and fixes made.
- `methods/swav/refs/self_accounts/search_log.md:1` documented the author self-account search and the thesis/talk sources found.
