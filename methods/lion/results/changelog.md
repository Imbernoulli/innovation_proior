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
- 2026-08-18 epistemic fix `methods/lion/results/reasoning.md` "do I need both constants" ablation
  paragraph: the prior svfix pass (above) replaced a fabricated sign-disagreement count with the primary
  paper's own published ablation table numbers, but framed them as an experiment the narrator ran and
  observed ("So run it. I train ... The recorded top-1 numbers: ..."). Those numbers are Lion's own
  ablation results — this is a single-turn proposal frame in which the method's own experiments have not
  happened yet, so the narrator cannot report having run them. Rewrote to keep the hypothesis (is one EMA
  enough, or is the decoupling load-bearing), the discriminating-experiment design (train both degenerate
  single-constant versions alongside AdamW and the two-constant rule, matched ViT-S/16/ViT-B/16 setup and
  tuning budget), each hypothesis's prediction, and the decision rule ("whichever pattern survives at both
  scales is what decides it") — with no claimed observation and no numbers. The landing keeps both
  constants provisionally (matching what the actual paper does), but that choice is now stated as a
  decision rule pending the run, not a reported result; this unit needs a trajectory-conversion pass to
  carry the real observation.
- 2026-08-18 epistemic fix, repair pass `methods/lion/results/reasoning.md` closing "Recapping the chain
  that got me here" paragraph: the prior epistemic-fix pass above rewrote the ablation paragraph's body
  but missed an echo of the same claim later in the same file — the recap sentence still asserted the
  optimizer's two constants were "ablated to confirm both constants are needed," a completed-past-tense
  claim of a confirmed observation that directly contradicted the paragraph just above it (now framed as
  an undone, decision-rule-pending test). Reworded to "pending the ablation that decides whether both
  constants earn their place," removing the claimed confirmation while keeping the recap's factual content
  (two decoupled constants, what each separates) intact. Also un-mixed this method's edits out of commit
  `146584f77`, which carried them alongside an unrelated label-smoothing fix via a shared git index;
  reverted that accidental inclusion and recommitted lion standalone so the commit touches only this
  method.
