# Antecedents & gaps

## Sources used
- **Self-account (primary backbone):** Dantzig SOL 87-5 (1987) + Reminiscences (1982/83). Read in full.
- **Primary formulation:** Dantzig 1951, "Maximization of a linear function of variables subject to linear inequalities," in Koopmans (ed.) *Activity Analysis of Production and Allocation*, ch. 21. **GAP: no free full-text PDF found** (Wiley/Cowles monograph, not digitized openly). MITIGATED: the SOL 87-5 retrospective reproduces Dantzig's own algebraic formulation (problem eq. 1; pricing s = argmin[cⱼ−(πA.ⱼ+π₀)]; the convergence theorem and proof), so the primary algebra is grounded in Dantzig's own text.
- **Canonical code:** scipy `_linprog_simplex.py` (full-tableau Dantzig simplex; pricing, min-ratio test, pivot, Bland's rule, Phase 1/2). Read in full.

## Antecedents (best-effort, web-grounded)
- **Leontief input-output (1936 "Quantitative input and output relations…"):** matrix of interindustry flows; system of linear equations, one equation per industry's product distribution; steady-state; one-to-one process↔item. Dantzig's model is the dynamic, many-activity generalization. (Wikipedia Input–output model; New World Encyclopedia.)
- **Fourier–Motzkin elimination (Fourier 1826/27; Motzkin PhD 1936):** eliminate a variable from a system of linear inequalities by pairing positive- and negative-coefficient inequalities → project the polyhedron to lower dimension. First algorithm ever used for LP feasibility; can derive LP duality. Dantzig cites Fourier 1824 as a prior special-case proposer of edge descent. (Wikipedia Fourier–Motzkin elimination.)
- **von Neumann game theory / duality:** Minimax theorem (vN 1928, "Zur Theorie der Gesellschaftsspiele"); *Theory of Games* (vN–Morgenstern 1944). At the Oct 1947 meeting vN conjectured LP ≡ games and outlined Farkas' Lemma + duality to Dantzig. Forerunner of the LP Duality Theorem. (Dantzig's own account.)
- **Kantorovich 1939:** independent LP proposal (different method), neglected in USSR, unknown to Dantzig until ~1959.

## Gaps / notes
- 1951 primary not retrieved as PDF; primary algebra grounded via Dantzig's own 1987 reproduction instead — flagged here per skill's "note gaps" instruction.
- Klee–Minty worst case, Khachiyan ellipsoid (1979), Karmarkar (1984) are POSTERIOR to the 1947 discovery and excluded from the in-frame trace (the 1987 retrospective mentions them but the reasoning ends at the discovery).
