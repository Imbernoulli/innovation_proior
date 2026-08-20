# Changelog

## 2026-08-19 — svfix(W3_primary_only)
Fixed an inverted-direction logic error in the Gotsman–Linial bridge exposition. The
trace (and `train_answer.md`) said a large $s(g,x)$ means a vertex "has almost all $n$
neighbors of its own color" (high monochromatic degree); the correct relation is
$\deg_H(x)=n-s(g,x)$, so a large $s(g,x)$ means the vertex is nearly *isolated* in its
own color class (low monochromatic degree). Corrected the direction so that small $s(f)$
is now shown to force $\Gamma(H)$ *small* everywhere, which is what the independently
proved graph theorem ($\Gamma(H)\ge\sqrt n$ always) rules out — the actual mechanism of
the bridge. Grounded against `refs/resolution_survey.pdf` (already on disk), which states
$\deg_G(x)=n-s(g,x)$ explicitly, and against the primary source's own statement of the
equivalence. No change to the landing (final theorem, construction, or constant $C=4$);
the decisive spectral step ($A_n^2=nI$) was reviewed and left as-is (sound_as_is — it is
genuinely derived on the page from a checkable failure of the plain 0/1 adjacency
matrix, matching Huang's primary construction).

Files touched: `results/reasoning.md` (one paragraph), `results/train_answer.md` (one
sentence/its surrounding clause), `notes/sources.md` (new).
