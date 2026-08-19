# electra changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (adversarial-vs-MLE generator section, end of the REINFORCE derivation):
  the prior svfix pass (`c4979a4af`, citing an Appendix F ablation) had the narrator claim to have
  actually built the REINFORCE generator and run it head-to-head against the MLE generator at
  matched size, then report the outcome — "it lands at 58% masked-LM accuracy where the
  identical-sized MLE generator gets 65%", entropy-collapsed samples, and "it never catches up to
  D trained against the plain-MLE generator." Real published numbers or not, at this point in the
  frame the method's own experiments have not happened yet, so reporting an observed outcome is out
  of scope for a single-turn proposal.
- Fix: removed the claimed run and both numbers (58%, 65%) and the "it never catches up" outcome
  sentence. Kept everything else the prior svfix pass added or that already existed: the
  matched-size experiment design (same 256-hidden generator either way, so the comparison isn't
  confounded by capacity), both hypotheses (REINFORCE's action-space size makes it
  sample-inefficient vs. MLE's dense gradient; adversarial generators tend to collapse to
  low-entropy, low-diversity output), each hypothesis's prediction (a real measurable MLM-accuracy
  gap; less diverse samples that make worse negatives for D), and an explicit decision rule: train
  both generators to convergence at matched size, compare MLM accuracy and sample diversity, check
  whether D trained on REINFORCE's negatives ever catches up to D trained on MLE's — whichever
  proposal mechanism produces the stronger D is the one that ships.
- The landing (training G by plain MLE) is now carried by a stated prediction/working bet rather
  than a resolved observation — that is expected per the epistemic-fix rule, not a gap to paper
  over. This unit needs a trajectory-observation turn to actually run the matched-size REINFORCE-
  vs-MLE comparison and supply the real numbers.
- No changes to `answer.md` or `train_answer.md` were needed — the svfix diff (`c4979a4af`) touched
  only `reasoning.md`, and neither of those files carries the removed numbers (their own REINFORCE
  mentions are pre-existing, hedged design statements without self-reported experimental results,
  out of scope for this pass).
