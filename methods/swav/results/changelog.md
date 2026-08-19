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

## 2026-08-17 — source-value recheck
- `results/reasoning.md`: the decisive reframe (code need not be dataset-level truth, only consistent across views) now runs through the author-side retrospective's own augmentation ablation on the clustering predecessor — 12.5% intra-class-edge purity with random-resized cropping, 3.5% with a fixed-size random-location crop, 0.9%/0.6% with the central crop only, flips worth nothing (12.5/11.1/12.4) — plus the observation that only one of k-means' two outputs (assignment, not centroids) was ever used. No factual corrections were needed; landing and code unchanged.

## 2026-08-18 — epistemic correction (svfix recheck audit)
- `results/reasoning.md`: the DeepCluster augmentation-ablation paragraph added by the 2026-08-17 recheck was voiced as the narrator's own live experiment ("I ablated... I get 12.5%"), which reads as an own-method observation claim. Reframed to attributed prior-work voice ("that work's own ablation... reports"), matching how DeepCluster is treated in third person elsewhere in the same file. Numbers kept unchanged — they are DeepCluster's own recorded result (Caron thesis §3.4, Table 3.4), genuinely predating SwAV, not SwAV's own result. No claimed observation was removed; only the misleading first-person "I ran this" framing was fixed.
