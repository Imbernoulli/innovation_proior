# Changelog — awq

## 2026-08-18 — svfix(D_candidate)
- **Fixed fabrication in results/reasoning.md (decisive-step scale-vs-error table).**
  The trace previously described a synthetic "200k trials on random 8-weight
  groups" Monte Carlo experiment with invented error/Δ'/Δ numbers, presented
  as something the scientist actually ran. None of those numbers correspond
  to src/ or any saved source. Replaced with the real experiment on
  OPT-6.7B recorded in src/text/3_approach.tex §3.2 and src/figure_text/tab_scale_study.tex:
  the actual measured proportion of groups with Δ'≠Δ, average Δ'/Δ, average
  (Δ'/Δ)(1/s), and WikiText-2 PPL for s ∈ {1, 1.25, 1.5, 2, 4}. This also
  fixes the trace's closing recap paragraph, which cited the fabricated
  "climb to 3×" figure — now cites the real measured value (~1.2× at s=4).
- **Grounded two previously unsourced design choices with author self-account
  (GitHub issue replies from Ji Lin / tonylins, AWQ first author, on
  mit-han-lab/llm-awq):** (1) why the search-space activation statistic is
  the per-channel *average* magnitude rather than the max — src/text/3_approach.tex
  doesn't explain this, only the code does it silently; issue #58 has Lin's
  stated rationale. (2) that combining weight magnitude into the scale
  search (on top of activation magnitude) was tried and dropped because it
  didn't move the loss — issue #110. Sources logged in notes/sources.md;
  saved locally at refs/github_issues_scale_design.txt.
- No change to the landing (method + code): the scale-search formula,
  α-grid-search procedure, and quantize_block implementation are unchanged.
