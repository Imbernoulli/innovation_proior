# refs/ provenance — primary, antecedents, analysis (all retrieved this run)

All files below were downloaded from arXiv this run and read (full text / OCR via pdftotext).
The previous attempt left refs/ empty (memory-based); this run grounds every claim in retrieved sources.

## PRIMARY (read fully)
- `refs/barnard-steinerberger-1903.08731.pdf` — R. Barnard & S. Steinerberger, "Three Convolution
  Inequalities on the Real Line with Connections to Additive Combinatorics", arXiv:1903.08731 (J. Number
  Theory 207, 2020). LaTeX source extracted to `src/1903.08731/convolutiofinal.tex` and read in full
  (intro + all three inequalities + both proofs + the arcsine-distribution example construction + bibliography).
  Source of: the average problem (w = chi_[-1/2,1/2]), the 0.91 upper bound and its Hardy–Littlewood /
  Si-function proof, the 0.8 lower bound (Gaussian-times-linear example), the min-autocorrelation
  Theorem 2 (<= 0.411) with the inf sin(x)/x = -0.217234 constant, and the Sidon-set lineage table.

## PRINCIPLED ANALYSIS / METHOD (read fully) — the variational/spectral/LP method reconstructed
- `refs/dedios-madrid-2106.13873.pdf` — J. de Dios Pont & J. Madrid, "On classical inequalities for
  autocorrelations and autoconvolutions", arXiv:2106.13873. LaTeX source extracted to `src/2106.13873/`
  (intro.tex, existence.tex, approximation.tex, computational.tex, main.tex) and read in full.
  Source of the entire reconstructed method: the L^{1:2} Rayleigh quotient; AM–GM lambda-relaxation with
  H_lambda / B_lambda norms; existence + compact-support via Euler–Lagrange and a <= 2||w||_1^2/C_opt,R^2;
  the discretization error 0 <= c_lambda - c_{lambda,delta} <= 16 delta^2/(pi^2 c_lambda lambda^2) (optimal
  Poincaré + Young + regularity ||f'||_2 <= 4/(c_lambda lambda^{3/2})); the rank-one whitening
  A_lambda = sqrt(lambda)Id + b_lambda|1><1|, M_lambda = 2 A_lambda^-1 K_w A_lambda^-1, power method;
  the support-length sweep for f>=0; the triangular cell kernel
  tw(s) = delta^-2 int_{s-delta}^{s+delta} w(t)(delta-|t-s|) dt; the conjectured fixed-point iteration;
  and Table 1 numeric brackets 0.8055809..0.8055896 (indicator) and 0.7152474..0.7152576 (Gaussian).

## ANTECEDENTS (best-effort)
- `refs/cloninger-steinerberger-1403.7988.pdf` — A. Cloninger & S. Steinerberger, "On Suprema of
  Autoconvolutions with an Application to Sidon sets", arXiv:1403.7988 (Proc. AMS 145, 2017). The
  computational relaxation that pushed the MAX constant c_max >= 1.28 (cost exponential in discretization,
  non-smooth extremizers) — the cautionary baseline contrasting with the tractable average problem.
- `refs/matolcsi-vinuesa-0907.1379.pdf` — M. Matolcsi & C. Vinuesa, "Improved bounds on the supremum of
  autoconvolutions", arXiv:0907.1379 (JMAA 372, 2010). c_max in [1.2748, 1.5098]; disproves the
  Schinzel–Schmidt conjecture; the LP/Fourier lower-bound lineage (Yu, Martin–O'Bryant) and the explicit
  upper-bound example.
- Martin & O'Bryant "The symmetric subset problem in continuous Ramsey theory", Experiment. Math. 16
  (2007) — `refs/martin-obryant-symmetric-subset-math0210041.pdf` (arXiv:math/0210041). The discrete
  B_h[g] / generalized-Sidon variational framing antecedent.
- Discrete origins (best-effort, noted as gaps): Cilleruelo–Ruzsa–Vinuesa (Adv. Math. 2010) connecting
  c_max to B_h[g] asymptotics is cited via the primary and via 1403.7988/2106.13873; not separately PDF'd.

## RELATED EXISTENCE / DUAL ANALYSIS (best-effort)
- `refs/madrid-ramos-2003.06962.pdf` — J. Madrid & J. P. G. Ramos, "On optimal autocorrelation inequalities
  on the real line", arXiv:2003.06962 (CPAA 20(1), 2021). LaTeX in `src/2003.06962/JoaoJose.tex`. Improves
  the average upper bound 0.91 -> 0.864 via the sharp Hausdorff–Young (Beckner) inequality in a dual Fourier
  formulation; Gaussian-mean (8a/27pi)^{1/4}; conjectures existence/compact-support of extremizers (the
  conjecture that 2106.13873 then proves).

## Anti-pattern (context only, NOT reconstructed in reasoning.md)
AlphaEvolve / ThetaEvolve searched-bound improvements to a numeric autoconvolution constant are a one-sided
test-function search (a better lower bound on a sup), with no matching upper bound and no certificate. The
reconstructed method is the opposite, principled move: a two-sided rigorous bracket via the variational
reduction + spectral relaxation + delta^2 discretization certificate. AlphaEvolve is treated only as a
generic "search-improved construction" baseline in context.md, and never appears in reasoning.md.

## Canonical code
`code/suprema-autocorrelations/` is the authors' repo (github.com/jaumededios/suprema-autocorrelations),
cloned this run. The public repo exposes only README + LICENSE (the Section-3 notebook was never committed),
so the final code in results/ is reconstructed faithfully from the paper's explicit Section 3.2–3.3 algorithm
and smoke-tested: on a coarse grid (delta=0.02) it returns 0.80536 for the indicator, converging to the
paper's 0.80558. Flag: notebook source not retrievable; code grounded on the paper's algorithm, not copied.
