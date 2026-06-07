# Source provenance and gaps

## Three-source bottom line — all obtained and read this run
1. PRIMARY (foundational equations): the foundational MPC sources are Richalet, Rault, Testud &
   Papon 1978 (IDCOM / model predictive heuristic control, Automatica 14(5):413-428), Cutler &
   Ramaker 1980 (DMC, Joint Automatic Control Conf.), and the canonical survey García, Prett &
   Morari 1989 "Model Predictive Control: Theory and Practice—a Survey" (Automatica 25(3):335-348).
   ALL THREE are paywalled Elsevier/Automatica with no author or arXiv copy located after a
   budgeted search (researchgate/academia/semanticscholar = abstract-only dead ends). GAP FLAGGED.
   Faithful reproduction used instead: Qin & Badgwell 2003, "A survey of industrial model
   predictive control technology", Control Eng. Practice 11:733-764 (open PDF from CMU CEPAC),
   which reproduces the IDCOM FIR model + reference trajectory, the DMC step-response model and
   dynamic matrix, QDMC-as-QP, and the LQG-can't-handle-constraints argument verbatim in
   equations, and traces the genealogy LQG -> IDCOM/DMC -> QDMC. refs/qin_badgwell_survey.pdf
   URL: https://cepac.cheme.cmu.edu/pasilectures/darciodolak/Review_article_2.pdf
2. BACKGROUND: LQR/LQG + discrete Riccati (Kalman 1960; Kwakernaak & Sivan 1972) — re-used as
   known (the existing methods/lqr/ deliverable covers the derivation; not re-derived here per
   the brief). QP background. The discrete batch/condensed LQ construction and the DARE are in
   the book Ch 8 (read, refs/borrelli_bemporad_morari_book.pdf pp.164-167).
3. THIRD-PARTY EXPLAINER: Borrelli, Bemporad & Morari, "Predictive Control for Linear and Hybrid
   Systems" (Cambridge 2017), open author PDF. Read Ch 8 (batch/condensed QP, DARE), Ch 11
   (constrained 2-norm QP, condensed eq. 11.31 and sparse eq. 11.30, prediction matrices S^x,
   S^u, H/F/Y eq. 8.8-8.10), Ch 12 (RHC idea 12.1, implementation, loss of feasibility Ex 12.1,
   loss of stability Ex 12.2, persistent feasibility via control-invariant terminal set Thm 12.1
   / Lemma 12.2, stability Thm 12.2 with telescoping proof 12.19-12.23, terminal cost = LQR
   value matrix eq. 12.24-12.26). refs/borrelli_bemporad_morari_book.pdf
   URL: https://cse.lab.imtlucca.it/~bemporad/publications/papers/BBMbook.pdf

## Canonical code
pyMPC (forgi86), OSQP-based linear constrained MPC.
URL: https://github.com/forgi86/pyMPC  (code/pyMPC/, core mpc_no_slack.py)
The reasoning.md / answer.md controller mirrors its sparse-banded OSQP formulation: decision
vector (x_0..x_N, u_0..u_{N-1}), dynamics as equality constraints, box + slew-rate (Δu)
constraints, first input applied, warm-started re-solve. Verified end-to-end in closed loop
(code/verify_mpc_closed_loop.py): point-mass setpoint task converges to target and respects the
input box, slew-rate bound, and position limit (residual sub-1e-3 overshoots are OSQP's default
tolerance, not formulation errors).

## Unsourced facts
None. Every equation traces to the book or the Qin-Badgwell survey; every empirical premise
(actuator saturation, economic optimum at constraints, short-horizon infeasibility/divergence)
is sourced (Prett & Gillette 1980 via Qin-Badgwell; book Ex 12.1/12.2). The three foundational
1978/1980/1989 primaries themselves are paywalled — their equations are taken from the faithful
Qin-Badgwell reproduction, gap flagged above.
