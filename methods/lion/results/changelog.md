# Changelog

- 2026-08-18 `methods/lion/results/reasoning.md` two-buffer-to-one-buffer collapse paragraph: replaced a
  fabricated numerical claim ("sweeping β to best-fit the recorded m ... mean-squared error is ~10% of
  the variance of m") with an actual algebraic substitution of `m_t = interp(g_t, v_{t-1}, 0.9)` into
  `v_t = interp(g_t, m_t, 1.1)`, which yields the exact closed form `v_t = 0.99·v_{t-1} + 0.01·g_t` —
  reproducing the primary's own stated equivalence (release.tex line 528: "two interp functions with
  constants ~0.9 and ~1.1 is equivalent to one with ~0.99") with real, checkable math instead of an
  invented curve-fit statistic that does not actually reproduce (verified numerically: the fabricated
  ~10% residual figure was wrong by roughly an order of magnitude). See `notes/sources.md`.
- 2026-08-18 `methods/lion/results/reasoning.md` "do I need both constants" ablation paragraph: replaced
  a fabricated sign-disagreement count ("disagrees on the sign about 12 of 40 steps ... about 8 of 40")
  with the primary paper's own published ablation numbers (Table `tab:multi`, release.tex lines 933-940:
  Ablation_0.9/Ablation_0.99 vs. Lion vs. AdamW on ViT-S/16 and ViT-B/16 ImageNet), restaged as
  hypothesis -> discriminating test -> observed record numbers -> interpretation. See `notes/sources.md`.
- 2026-08-18 `methods/lion/results/reasoning.md` closing recap sentence: changed "behave to a good
  approximation like a single ≈0.99 momentum EMA" to "algebraically fold ... into a single ≈0.99 momentum
  EMA" to match the corrected (exact, not approximate) derivation above.
- 2026-08-18 `methods/lion/notes/sources.md` added: documents the self-account search trail (none found;
  OpenReview forum for this paper has no reviews, no Google Research blog post, no relevant GitHub issue
  discussion) and the primary-source passages (release.tex) now grounding the decisive step.
