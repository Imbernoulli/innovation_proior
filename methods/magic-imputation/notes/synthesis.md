# MAGIC synthesis (grounded)

## Sources (three-source)
1. PRIMARY: van Dijk et al. 2018, "Recovering Gene Interactions from Single-Cell Data Using Data Diffusion", Cell 174:716. Open-access full text PMC6771278 (STAR Methods extracted in full via WebFetch).
2. BACKGROUND: scRNA-seq dropout/sparsity; manifold assumption; kNN graphs + Gaussian/adaptive affinity kernels; Markov transition matrices; diffusion maps (Coifman & Lafon 2006, ACHA 21:5) — heat kernel, eigenvalues, diffusion distance, powering M -> lambda^t. Diffusion-maps formulas verified via search/PNAS-ACHA.
3. EXPLAINER: MAGIC docs (magic.readthedocs.io) + the magic-impute repo README/tutorial. Parameter semantics (knn=5, decay=1, t default 3 / 'auto', knn_max=3*knn).

## Pre-method empirical facts (-> context Background; NEVER fabricated)
- scRNA-seq captures only ~5-15% of each cell's transcriptome (small random sample). Dropout = an expressed gene not detected (recorded 0), esp. lowly-expressed genes. Counts are sparse and noisy.
- Manifold assumption: cell phenotypes lie on a low-dimensional manifold embedded in high-dim gene space; gene-gene regulatory relationships restrict cells to a lower-dim subspace. The manifold is representable by a kNN graph (nodes=cells, edges=most-similar cells).

## Algorithm (grounded in STAR Methods + graphtools + magic.py)
Preprocessing: filter empty cells/genes; library-size normalize each cell (divide by row sum, x median library size); sqrt transform; PCA to ~70% variance (20-100 PCs, default n_pca=100).
1. Distances: Euclidean in PCA space.
2. Affinity kernel (adaptive bandwidth, alpha-decay generalization of Gaussian):
   - per-cell bandwidth sigma_i = distance to knn-th neighbor (graphtools: distances[i, knn-1] * bandwidth_scale). Paper calls it ka-th NN.
   - A(i,j) = exp(-(d(i,j)/sigma_i)^decay). Paper's Gaussian = exp(-(d/sigma)^2) (decay=2). MAGIC default decay (alpha) = 1 in current code; decay=None gives a hard kNN (no decay). knn_max = 3*knn caps neighbors (paper: k = 3*ka).
3. Symmetrize: paper A <- A + A'; graphtools default "+" = (A + A^T)/2 (equivalent after row-normalization).
4. Markov: M = D^{-1} A, row-stochastic (normalize l1, axis=1). M(i,j)=A(i,j)/sum_k A(i,k).
5. Diffuse / impute: X_imputed = M^t X. Dimputed(i,j) = sum_k M^t(i,k) D(k,j) = t-step random-walk weighted average over neighborhood.
6. Select t: paper STAR uses R^2(data_t, data_{t-1}) = 1 - SSE/SST, stop after it drops below 0.05 change (pick 2nd t after). Released code: Procrustes disparity between successive iterates, threshold 0.001, t_max=20 ('auto'). Both = "stop when successive diffused matrices stop changing." Fixed default t=3.
7. Rescale (paper STAR; optional, off by default in py): per gene, max of imputed set to 99th percentile of original: Drescaled(:,j) = Dimputed(:,j) * percentile(D(:,j),0.99)/max(Dimputed(:,j)).

## Spectral view (Coifman-Lafon)
M (or its conjugate symmetric form D^{-1/2} A D^{-1/2}) has eigenvalues in [0,1], lambda_1=1 (stationary), trivial constant eigenvector. M^t has eigenvalues lambda^t. Powering shrinks every lambda<1 toward 0, fast for small lambda (noise / high-frequency graph modes), slow for lambda near 1 (smooth manifold directions). So M^t is a low-pass filter on the graph: keeps signal supported on the slow manifold modes, kills noise. Diffusion distance / diffusion-map embedding Psi_t(x_i) = [lambda_l^t psi_l(i)].

## Design decisions -> why
- Adaptive (per-cell) bandwidth: density varies; fixed sigma over-connects dense regions, disconnects sparse ones. sigma_i = dist to ka-th NN equalizes effective neighbor count.
- ka small, k=3ka: ka must be as small as possible to keep graph connected (locality); k caps nonzero entries for sparsity/speed; beyond ~3 sigma the Gaussian is ~0 anyway.
- Symmetrize: random-walk affinity is asymmetric (different local densities); additive symmetrization "pulls in outliers" and denoises; needed so the operator has the nice [0,1] real spectrum.
- Row-stochastic M not symmetric A: want a probabilistic averaging operator (rows sum to 1) so imputation = convex combination -> stays in data range, no rescaling of magnitude from the operator itself.
- Power M (diffusion) instead of single hop: single-hop average still trusts noisy edges; t-step walk reinforces edges supported by many paths (true neighbors) and washes out spurious shortcuts; spectrally = sharper low-pass.
- Choosing t by convergence of successive iterates: too small = under-denoised, too large = collapses toward stationary distribution (everything -> global mean, loses biology). The plateau in change marks the transition from the fast noise-removal regime to the slow over-smoothing regime.
- PCA preconditioning: distances in raw sparse gene space are dominated by dropout noise; PCA gives diffusion a cleaner geometry; ~70% variance keeps signal, drops noise dims.
- sqrt + libsize norm: libsize removes depth confound; sqrt variance-stabilizes count data (Poisson-ish) so Euclidean distance is meaningful.

## In-frame: never name the paper. Method may be called MAGIC in answer.md.

## svfix audit addendum (2026-08-18, TRACK=W3_notes_unclear)
Quality-gate check: the decisive step in reasoning.md (powering the row-stochastic Markov
operator M^t X as a spectral low-pass filter; picking t at the elbow of per-step R^2 change)
is genuinely DERIVED on the page via the trace's own honest spectral computation (Markov
matrix eigenvalues in [0,1], M^t shrinks each mode by lambda^t) AND that derivation is backed
almost verbatim by the primary paper's own STAR Methods. Verdict: sound_as_is. No rewrite
made to reasoning.md/answer.md/train_answer.md/code.

PROVENANCE BUG FOUND AND FIXED: refs/magic_pmc.html on disk was NOT the MAGIC paper — it was
a wrong PMC article ("Prioritizing uncharacterized genes in the search for glioma biomarkers",
Towner & Wren, CNS Oncol 2014), i.e. a bad download that silently landed at the wrong PMC id.
refs/magic_biorxiv.pdf was also broken: it is a Cloudflare "Just a moment..." challenge page
saved with a .pdf extension, not the preprint. This means the synthesis.md claim above
("STAR Methods extracted in full via WebFetch") could not previously be verified from any
file on disk, which is exactly the W3_notes_unclear flag.

Fix applied: re-fetched the correct primary source, PMC6771278 / pmc.ncbi.nlm.nih.gov/articles/PMC6771278/
(van Dijk et al. 2018, Cell 174:716, "Recovering gene interactions from single-cell data using
data diffusion" — confirmed via <meta name="citation_title"> in the saved HTML) and overwrote
refs/magic_pmc.html with the real article; also saved a plain-text extraction refs/magic_pmc.txt.
Both now contain, verbatim, the STAR Methods passages that ground the decisive step, e.g.:
- low-pass framing: "Powering M has the effect of low-pass filtering the eigenvalues of the
  Markov transition matrix... diminishes the importance of noise dimensions with near-zero
  explanatory power." (refs/magic_pmc.txt, STAR Methods, "Markov affinity based graph diffusion")
- elbow/R^2 stopping rule: "we reason that the decay has approximately converged after it has
  gone below 0.05, i.e., less than 5% change from the previous t" (refs/magic_pmc.txt, STAR
  Methods, "Diffusion time for Markov Affinity Matrix") — matches reasoning.md's "(R^2 says <5%
  change from the previous t)" essentially verbatim.
- spurious-edge / powering rationale for "why not one hop": "spurious edges would have
  similarity in the raw data, but these would not be supported by shared neighbors... Powering
  M has the effect of low-pass filtering..." (same section) — backs reasoning.md's "trust an
  edge in proportion to how robustly connected the two cells are... through many independent
  paths" passage.

STILL BROKEN, not fixed this pass (not load-bearing for the decisive step, so left as a flag
rather than blocking this fix): refs/magic_biorxiv.pdf (Cloudflare challenge page, not the
preprint; biorxiv.org rate-limited retries with HTTP 429 during this session) and
refs/diffusion_maps_coifman_lafon_2006.pdf (actually a ScienceDirect/Elsevier access-gate HTML
page, not the PDF — TRIAGE correctly treated Coifman-Lafon as background-math corroboration
only, never as the decisive-step source, so this doesn't affect the sound_as_is verdict, but a
future pass should refetch both from an open-access mirror).
