# Soft Actor-Critic changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the restricted-policy-class passage: the trace previously justified the
  KL projection only by "I can't represent an arbitrary energy-based density with my Gaussian
  actor". It now runs through the real wall, taken from Haarnoja's UC Berkeley PhD dissertation
  (`refs/self_accounts/haarnoja_thesis.pdf`, §1 chapter summary; quote in `notes/source_matrix.md`,
  a self-account the earlier pass wrongly recorded as non-existent): the energy-based optimal policy
  in soft Q-learning is a consequence of assuming the policy class is unrestricted, so substituting
  a Gaussian into soft Q-learning does not converge to the optimal solution — the fixed point is
  wrong, not merely approximate — which is what forces the improvement statement to be defined
  relative to Π and hence the restricted-class projection proof.
- Checked and NOT changed: "empirically it doesn't even beat DDPG from scratch" is faithful to
  `src/v1801/main_no_comments.tex:109` and to the thesis. The choice of the information projection
  remains justified by the partition function dropping out of the gradient, which matches the
  thesis's own "in principle we could choose any projection ... convenient" remark.
- Landing, proofs and code unchanged.
