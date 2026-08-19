# Sources — decisive-step sourcing check (svfix, W3_ancestors_only)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines 11-45: estimate the Pareto front's Minkowski exponent p
from a single "central" point on the normalized first front — the point
closest to the diagonal (1,...,1), excluding the corners — via the
closed-form heuristic p = log(M)/log(1/mean(C)) (derived by assuming C sits
at the manifold's symmetric center, so M·c̄^p = 1), then use that p to build
a proximity(1/||A||_p) x diversity(greedy scaled-gap) survival score that
replaces NSGA-II's L1 crowding distance.

## Quality-gate verdict: SOUND_AS_IS — no rewrite made

### (a) Genuinely derived on the page, not asserted
The whole passage (reasoning.md paragraphs 12-15, 17, 29, 39) is real,
checkable math worked by hand, not a hindsight-toned restatement:
- Rejects the "obvious" honest move (nonlinear LM fit over the whole front,
  GFM's route) for a concrete, checkable reason: cost
  O(G'.M^2.(M+N)), re-paid every generation — this number is the actual
  complexity stated for GFM-MOEA in the source (see below), reused
  correctly.
- Argues corner points carry zero information about curvature (Σ 1^p+0+...
  = 1 for *any* p) — a first-principles, checkable fact — which is *why* the
  central point is chosen, stated *before* the numeric robustness check, not
  bolted on after.
- Derives p = log(M)/log(1/mean(C)) in closed form from the symmetric-center
  assumption (M.c̄^p=1 => p = -log M/log c̄).
- Verifies the closed form exactly by hand on 5 known cases (flat M=2/3,
  sphere M=2/3, convex p=0.5) — I recomputed all 5 independently, all
  correct to the stated precision.
- Quantifies how the estimate degrades as the chosen point drifts off the
  diagonal (45/40/35/30 deg on a true p=2 circle -> p_est 2.000/1.978/
  1.915/1.818) and explains the mechanism directly ("plugging the 35 deg
  estimate p=1.915 back into the true manifold equation gives
  a_1^p+a_2^p=1.027, not 1, so the symmetric substitution is the source of
  the error") — I recomputed this too (cos/sin at 30/35/40 deg, logs): every
  number matches to the stated precision.
- Traces the greedy scaled-gap selection by hand on a constructed 5-point
  front (2 corners + center + interior point + its near-duplicate) and gets
  scores [inf, inf, 2.00, 1.00, 0.52] with the near-duplicate dropped last —
  I re-ran this by hand (L1 distances, tie-break-by-first-index exactly as
  the eventual code's `np.argmax`/`argpartition` would do) and reproduced
  every intermediate number and the final ordering exactly.
No step in this region states an empirical/experimental outcome as
logically necessary, and no step self-supplies an "I ran it and got X"
observation — it is pure, checkable algebra plus hand-traced toy examples,
which is exactly what the quality gate calls a legitimate derivation.

### (b) Backed by the trace's own honest computation, and *also* by the
### primary-adjacent source — but grafting the source would be redundant
`refs/agemoea2.pdf` (Panichella, *An Improved Pareto Front Modeling
Algorithm for Large-scale Many-Objective Optimization*, GECCO 2022 — the
paper notes/synthesis.md already established as standing in for the
paywalled 2019 AGE-MOEA primary, cf. its Algorithm 1 / Eq. 2 / Eq. 4 / Eq. 5
being the source for the rest of the trace's equations) has, in
Sec. 2.2 "Limitations of Existing Modeling Methods" (`refs/agemoea2.txt`
around line 218-235), the *same* phenomenon documented with a different
worked instance:

> "the accuracy of computing/approximating the value ? strongly depends on
> how close the chosen reference point is to the theoretical center of the
> front." (OCR renders the glyph "p" as "?"; `refs/agemoea2.txt` lines
> 230-231)

with a numeric table for three points A, B, C on a true p=2 circle (p_est =
0.495, 1.628, 1.305 respectively) and "[o]nly if we apply Equation 4 to the
theoretical middle point C* = (1/sqrt2, 1/sqrt2), we obtain the correct
value p=2" (same file, lines 232-235). This is literally the trace's own
40/35/30-degree drift check, restated with a different set of three points.

I deliberately did **not** thread this into reasoning.md, for two reasons:
1. **Anachronism.** This section is AGE-MOEA-II's (2022) own retrospective
   analysis of AGE-MOEA's (2019) accuracy limitation — three years after
   the discovery this trace reconstructs. notes/synthesis.md already
   forbids AGE-MOEA-II appearing anywhere in-frame ("is itself a later
   work"); citing this passage inside the first-person present-tense
   reasoning would be exactly the "later work showed" hindsight leak the
   fix procedure bans, not a fix.
2. **Redundant graft.** Even set aside the anachronism, the trace already
   derives and quantifies the identical phenomenon on its own, correctly,
   with a different concrete example. Bolting on AGE-MOEA-II's A/B/C numbers
   on top would be exactly the wave-2 mistake called out in the fix prompt
   ("a blog stat that was just the primary's own number restated, and
   overclaimed independence; the verifier killed it") — a citation the
   reasoning does not need, added only to pad provenance.
I record it here only as independent confirmation that the trace's
self-derived numeric behavior matches the real, documented behavior of the
actual AGE-MOEA formula (i.e. this is not a fabricated toy example — the
central-point sensitivity it demonstrates is real and well-known for this
exact algorithm), which is what condition (b) of the quality gate asks for.

## Search performed (before concluding sound_as_is)
- `grep -ril` for corner/neighbor/normalization rationale across
  `methods/agemoea/{refs,src,notes,code}` — code/pymoo_age.py and the 5
  PlatEMO .m files (AGEMOEA.m, EnvironmentalSelection.m, SurvivalScore.m,
  FindCornerSolutions.m, Point2LineDistance.m) contain *no* design-rationale
  comments at all — only implementation notes ("let's round", "approximate
  p (norm)", "prevent already selected to be reselected", "avoid numpy
  underflow"). No documented obstacle/failed-attempt for the "2 nearest
  selected neighbors" choice or the normalization fallback exists in either
  canonical implementation; the TRIAGE hint's suggested angle ("thread in
  the author-code obstacle... why pymoo picks 2-nearest-neighbor gap") does
  not pan out — there is nothing there to thread in.
- `grep -i agemoea\|panichella SELF_ACCOUNT_SOURCES.md` at repo root -> no
  entry.
- Checked all 5 other refs/*.pdf (try1.pdf, acm_gw.pdf,
  agemoea_gecco2019.pdf, agemoea_orig.pdf, agemoea_preprint.pdf) with
  `pdftotext -layout` -> all extract to 0 bytes of text (paywall/Cloudflare
  stub captures, as notes/synthesis.md already logged); confirmed, not
  re-fetchable through this method's normal means.
- `curl api.github.com/search/issues?q=repo:anyoptimization/pymoo+AGEMOEA`
  and `...+age-moea+in:title,body` -> 5 and 7 hits respectively, all bug
  reports (numba install issues, division-by-zero edge cases) triaged by
  the pymoo maintainer (blankjul), none from Panichella, none discussing
  design rationale for the central-point heuristic, corner detection, or
  the 2-neighbor greedy fill.
- WebSearch quota was exhausted this session before a thesis/interview/
  retrospective search for Panichella could be run; the on-disk material
  (author-authored code x2, cross-validated; author's own 2022 restatement
  with an equivalent worked example) already gives condition (b) two ways
  over, so this was not chased further.

## Conclusion
outcome = sound_as_is. No rewrite to reasoning.md, answer.md, or
train_answer.md. No new file threaded into the trace's citations.
