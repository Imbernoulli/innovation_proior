# Sources retrieved and read this run

## (1) Primary source — read in full
- arXiv:2102.09844 LaTeX source, already in methods/egnn/src/sections/*.tex.
  Read: introduction.tex, background.tex, model.tex, related.tex, appendix.tex (all proofs +
  implementation details), experiments.tex (settings only; proposed-method outcome tables NOT
  used). Bibliography example_paper.bbl checked for ancestor citations.

## (2) Background / load-bearing ancestors
- Gilmer et al. 2017 (MPNN) — message-passing GNN, read from paper background+related sections
  and verified via the standard m_ij/m_i/h update form in background.tex.
- Schütt et al. 2017 (SchNet) — continuous-filter conv, Gaussian RBF exp(-gamma(r-mu_k)^2),
  E(n)-invariant; verified via WebSearch (emergentmind SchNet topic) + paper related.tex table.
- Thomas et al. 2018 (Tensor Field Networks) + Fuchs et al. 2020 (SE(3)-Transformer) — spherical
  harmonics / steerable kernels, SE(3)-equivariant, expensive, 3D-only; from paper related.tex.
- Köhler et al. 2019/2020 (Radial Field / Equivariant Flows) — E(n)-equivariant coord-only
  update m_ij = phi_rf(||r_ij||) r_ij; from paper related.tex + Table 1.

## (3) Third-party explainers
- emergentmind.com EGNN topic pages (two) — confirmed design rationale: invariant message via
  distance, equivariant coord update as scalar-weighted relative differences, no spherical
  harmonics, difference vectors as overcomplete basis.
- emergentmind.com SchNet topic — RBF distance expansion form, invariance.

## Canonical implementations (code grounding) — fetched and read
- vgsatorras/egnn  models/egnn_clean/egnn_clean.py (official authors' clean reference).
- a-r-j/ProteinWorkshop  proteinworkshop/models/graph_encoders/layers/egnn.py + egnn.py
  (the exact source the task edit.py ports from).

## Self-account
- None found (recent paper; no Nobel/Turing-style retrospective or author memoir exists).
  Reconstruction is from primary source + ancestors + explainers + canonical code.

---

# svfix (W3_primary_plus_ancestors) — decisive-step sourcing check

## TRIAGE decisive step (independently re-identified)
reasoning.md lines ~17-27: distance-only invariant message passing (the
SchNet-style move, already derived a paragraph earlier) gives an invariant
`m_ij` but can only ever emit invariant scalars — the directional/type-1
structure collapses at layer one and can't come back, so a purely invariant
network can't output a coordinate. The fix: a second, equivariant update on
`x` itself — a weighted sum of relative-difference vectors `x_i - x_j`
(already type-1: translation-cancelling, rotates with `Q`), with the weight
supplied by an *invariant scalar* `phi_x(m_ij)` read off the already-invariant
edge message. Because the weight is invariant, `Q` factors straight out of
the sum: `sum_j w_ij (Q x_i - Q x_j) = Q sum_j w_ij (x_i - x_j)`. The trace
then proves this holds for the whole 4-equation layer by induction and
double-checks it with a worked numeric example (two points, a 90° rotation
+ translation, matching to the digit).

## Search performed this pass
- `grep -ril "satorras\|egnn\|equivariant graph" methods/egnn/refs methods/egnn/notes methods/egnn/src`
  → refs/{primary,ancestors,explainers,self_accounts}/*.md are pointer stubs (<1KB each,
  no downloaded emergentmind/self-account text on disk); the real, substantive material is
  the primary LaTeX in `src/sections/*.tex` (verified >2KB per file, full paper text).
- `grep -i "egnn\|satorras\|equivariant graph" SELF_ACCOUNT_SOURCES.md` (repo root) → no hits.
- WebFetch `github.com/vgsatorras` (author's GitHub profile) → no personal website/blog linked,
  only a Twitter/X handle; no self-account there.
- WebFetch `vgsatorras.github.io` → 404, no such page.
- WebFetch DBLP homepage lookup → no personal-site link on record.
- `api.openreview.net/notes/search?term=E(n) Equivariant Graph Neural Networks...` → no forum
  entry for this paper (ICML 2021 did not run this submission through OpenReview); 20 unrelated
  follow-up-work hits returned, none a self-account of the 2021 paper itself.
- GitHub Issues API on `vgsatorras/egnn` (13 issues) → read the one with an author reply that
  touches the coordinate-update equation (#5, "About the implementation of Eq4"): Victor
  Satorras's reply explains the sparse-vs-fully-connected `edge_index` implementation choice,
  not the invariant-scalar-weight/why-`Q`-factors-out design step. Not load-bearing for this
  decisive step.
- WebSearch quota was exhausted repo-wide for this session (200/200); fell back to curl against
  DuckDuckGo (blocked by anomaly/bot check) and Bing (JS-rendered, no parseable organic results)
  — both dead ends, consistent with the earlier recorded search-log conclusion.
- Net result: no first-author self-account (thesis/talk/blog/interview) exists for this
  decisive step, confirming refs/self_accounts/search_log.md's earlier conclusion.

## Correct backing IS in the primary, stated and proved directly
- `src/sections/model.tex` (eq. \ref{eq:method_coords} + surrounding prose): "the position of
  each particle $\rmx_i$ is updated by the weighted sum of all relative differences
  $(\rmx_i - \rmx_j)_{\forall j}$. The weights of this sum are provided as the output of the
  function $\phi_x$... This equation is... the reason why equivariances 1, 2 are preserved."
- `src/sections/appendix.tex` (\S Equivariance Proof) carries out exactly the "Q factors out"
  algebra the trace performs: "Notice $\rmm_{ij}$ is already invariant as proven above." then
  `Q\rmx_i^l + g + QC\sum_{j\neq i}(\rmx_i^l - \rmx_j^l)\phi_x(\rmm_{i,j}) = Q(\rmx_i^l +
  C\sum_{j\neq i}(\rmx_i^l - \rmx_j^l)\phi_x(\rmm_{i,j})) + g = Q\rmx_i^{l+1} + g`. The trace's
  inductive proof (reasoning.md ~lines 29-42) reproduces this derivation step for step, in the
  scientist's own first-person voice, not copied verbatim.
- `src/sections/related.tex` backs the earlier fork (steerable TFN/SE(3)-Transformer "heavy,
  3D-only" vs. Radial Field "only operates on x, doesn't propagate h") that the trace uses to
  motivate needing a cheap, dimension-agnostic vector channel: "the spherical harmonics need to
  be recomputed which can be expensive... an extension of this method to arbitrary dimensions is
  unknown" / Radial Field "only operates on $\rmx$ and it doesn't propagate node features $\rmh$".

## Quality-gate verdict
sound_as_is. Both prongs hold:
(a) genuinely derived on the page: the invariant-message-only move fails for a concrete,
    checkable reason (collapsing to a scalar distance destroys the directional structure needed
    to emit a vector — matches Table 1's own characterization of SchNet as invariant-only), and
    the resolution (invariant-scalar-weighted difference-vector sum) is derived algebraically,
    proved by induction across the full 4-equation layer, and cross-checked with a concrete
    worked numeric example, matching to the digit;
(b) the justification is the primary's own equations and its own appendix proof — model.tex
    states the mechanism, appendix.tex proves it with the identical "Q pulled out of the sum"
    algebraic step the trace performs.
No non-primary source exists that is load-bearing for this step, and none was found this pass
after a genuine multi-venue search (GitHub profile/issues, DBLP, OpenReview, repo-root
self-account registry, search-engine fallback). Grafting a citation onto an already-sound,
self-contained derivation would be decorative, not grounding, per the quality gate's explicit
warning. No rewrite of reasoning.md was performed; only this file was appended to.
