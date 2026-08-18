# Spectral Signatures changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the amplification half of the decisive step: the trace previously disposed
  of it in one clause ("I only need the learned features to enlarge the poison-clean mean gap enough
  that the covariance bump becomes visible"). That clause sits exactly where the real work stalled.
  The passage now runs through the documented attempt and its failure, taken from the fully written
  but commented-out subsection in the primary's own LaTeX source
  (`src/theory.tex:85-168` "Why do learned representations increase the signal?" and its cut
  appendix `src/supplementary-material.tex:113-205`; quotes in `notes/source_matrix.md`): the
  conjecture under assumptions (A1)/(A2) that could not be closed — the source literally reads
  "??? TODO" and "We cannot provide any rigorous justification that such a separation will occur" —
  followed by the fallback that did survive, an existence construction (margin ⇒ a linear functional
  that isolates the trigger, shifted ReLUs make a channel that is exactly zero on clean images, spare
  overparametrized filters double it per layer ⇒ 2^l growth), plus the honest scope ("backpropagation
  will in general not learn anything remotely similar to the parameters described here") and the
  consequence for an attacker (break the margin, not the gap).
- No factual errors found. The Σ_F = (1−ε)Σ_D + εΣ_W + ε(1−ε)ΔΔᵀ decomposition, the
  ⟨v,Δ⟩² ≥ ‖Δ‖² − σ²/(ε(1−ε)) bound, and the honest Chebyshev margin (already correctly caveating
  the over-clean "6σ²/ε" constant) all re-derive correctly. Landing and code unchanged.
