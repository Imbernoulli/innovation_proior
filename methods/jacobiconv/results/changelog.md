# Changelog

## 2026-08-18 — svfix(D_candidate)

Fixed the "GPR-GNN numbers" cited in `reasoning.md` (intro and closing paragraphs), `answer.md`, and
`train_answer.md`. They read Texas 0.9065, Cornell 0.8705, Cora 0.8890, Citeseer 0.8020, "Texas std
across seeds ranges from 0.0262 to 0.0471" — none of that matches the GPRGNN column of the primary
paper's own Table 2 (checked against `refs/jacobiconv_wang_zhang_icml2022.pdf`/`.txt`, arXiv
2205.11172v2): actual values are Texas 92.95±1.31, Cornell 91.37±1.81, Cora 88.57±0.69, Citeseer
80.12±0.83. Cornell was off by ~4.3 points, Texas by ~2.3 — both well outside the paper's own reported
95% CIs, so this was not reproduction noise. Also checked the original GPR-GNN paper (arXiv
2006.07988) directly: it reports no per-seed std at all, so the "std ... ranges from 0.0262 to 0.0471"
claim had no traceable source in either paper. Replaced with the primary paper's actual GPRGNN row and
its actual per-dataset 95% CIs across all three files; the substantive point (heterophilic datasets
Texas/Cornell carry a much wider interval than homophilic Cora/Citeseer, evidence of ill-conditioned
optimization) still holds under the corrected numbers. No search of primary-external self-account
material for the decisive Jacobi-vs-Chebyshev step turned up anything beyond the primary paper itself
(see `notes/sources.md` for the full search log); the decisive step's mechanism was independently
verified against the primary source's own text and is unchanged.
