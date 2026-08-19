# Sources — decisive-step sourcing check (svfix, W3_ancestors_only)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines 1-11: replace Pareto dominance (a partial order that goes
blind once nearly every candidate is mutually non-dominated at high M) with
a total-order scalar. The trace first tries PBI (`g^pbi = d1 + theta*d2`)
and rejects it for two concrete, checkable reasons worked out on the page:
(1) theta is a single fixed knob with no universal setting across problems
and objective counts; (2) PBI's off-line term d2 is a *Euclidean* distance
that is contaminated by convergence (the trace computes it explicitly: a 5x
radial scaling of a fixed-direction solution moves d2 from 1.414 to 7.071,
i.e. scales by exactly 5, while the angle stays fixed at 18.43 -> 18.43).
This motivates measuring diversity by the angle instead of the Euclidean
off-line distance, landing on APD = (1 + penalty(angle)) * ||f'||.

## Search performed
- `ls refs/ src/ code/` -> refs/rvea-primary.pdf (Surrey open-research copy)
  + refs/rvea-checked.pdf (Honda soft-computing.de cross-check copy, same
  paper); code/pymoo_rvea.py + code/xavier_rvea.py (third-party canonical
  implementations, correctly used only for equation cross-validation per
  notes/synthesis.md — never cited inside reasoning.md itself, confirmed by
  `grep -in "pymoo\|xavier\|github" results/reasoning.md` -> no hits).
- `grep -i rvea SELF_ACCOUNT_SOURCES.md` -> no entry.
- Web/venue check for a first-person account (the track's premise, since
  notes/synthesis.md already logs "no author self-account / discovery
  memoir found"): "RVEA Ran Cheng thesis reference vector", "Ran Cheng
  Surrey PhD thesis many-objective", "RVEA Yaochu Jin retrospective",
  "RVEA GitHub issue Cheng why angle penalized distance", "Cheng Jin
  Olhofer Sendhoff interview many-objective optimization". No first-author
  thesis chapter, interview, talk transcript, or GitHub issue thread
  discussing the *design rationale* surfaced (IEEE TEC 2016 predates
  open-access preprint servers commonly used for this kind of paper; no
  e-print, no OpenReview record for a non-ML-conference IEEE TEC journal
  article). Confirms the notes/synthesis.md finding: nothing beyond the
  primary exists for this method's origin story.
- Since TRACK=W3_ancestors_only and material_on_disk=true, also checked
  whether an ANCESTOR primary (already on disk elsewhere in this repo, for
  a sibling method) supplies grounding the RVEA primary itself lacks:
  `methods/moead/refs/moead-primary.txt` (Zhang & Li 2007, the actual PBI
  paper) does independently carry the theta-fragility point ("these
  benefits come with a price. One has to set the value of the penalty
  factor. It is well-known that a too large or too small penalty factor
  will worsen the performance of a penalty method.") — but this is
  redundant with what is already directly in the RVEA primary itself (see
  below), so grafting a Zhang & Li citation on top would be exactly the
  "bolted-on citation the reasoning doesn't need" the quality gate warns
  against. Not added.

## The obstacle IS stated directly in the primary (not merely implied)
`refs/rvea-primary.pdf` (Cheng, Jin, Olhofer, Sendhoff, *A Reference Vector
Guided Evolutionary Algorithm for Many-Objective Optimization*, IEEE TEC
20(5), 2016), extracted via `pdftotext -layout` to `/tmp/rvea-primary.txt`
for verification, Sec III-C:

> "the penalty item θ in PBI is a fixed parameter... As pointed out in
> [70], there is no unique setting for the parameter θ in PBI that works
> well on different types of problems with different numbers of
> objectives."
> — matches reasoning.md paragraph 7 exactly ("no value of theta that
> works well across different problems and different numbers of
> objectives").

> "no matter what the exact distance a candidate solution is from the
> ideal point, the angle between the candidate solution and a reference
> vector is constant. Second, angles can be more easily normalized into
> the same range, e.g., [0, 1]."
> — the same "angle is invariant to distance, distance is not" point the
> trace derives numerically in paragraph 10 (5x-scaling check: d2 scales
> by 5, angle unchanged at 18.43 degrees).

The trace's NSGA-III normalization counterexample (paragraph 33,
`f'_1=(0.1,2)`, `f'_2=(1,10)` -> `(0.1,0.2)`, `(1,1)`) is *the paper's own
worked example*, reproduced verbatim (Sec III-D): "Given two translated
objective vectors, f'1 = (0.1, 2) and f'2 = (1, 10)... after objective
normalization, the two vectors become f'1 = (0.1, 0.2) and f'2 = (1, 1)...
the difference between the two vectors has been substantially changed, from
||f'2-f'1|| = 8.0 to ||f'2-f'1|| = 1.2." — the trace independently adds the
angle computation on top of this same pair (2.85 deg -> 18.43 deg, a 6.5x
change) as its own honest, verified extension of the paper's example. The
"adapt vectors, not every generation" design (paragraph 41) is also directly
attributed correctly: the primary states "as pointed out by Giagkiozis et
al. in [72], the reference vector adaptation strategy should not be
employed very frequently during the search process."

## Independent verification of the trace's own worked numerics
Recomputed every numeric example in the decisive-step region (paragraphs
7-11, 17-21, 31, 33, 35-39) by hand: PBI d1/d2 at (3,1) and its 5x scaling
(3,1)->(15,5); angle invariance at the same two points; the gamma_0=gamma_1
=12.5deg / gamma_2=77.5deg three-vector example and the theta/gamma ratios
0.958 and 0.155; the early/late-generation APD flip (A=(5,1) vs B=(3.5,2.6),
t=10 vs t=399, t_max=400); the Hadamard reference-vector-adaptation example
(10x-stretched front, 45deg -> 5.7deg). All values check out exactly (to
the precision stated) — no arithmetic error to correct.

## Quality-gate verdict
sound_as_is. Both prongs of the gate hold:
(a) the decisive step is genuinely derived on the page — PBI's theta and
    Euclidean-d2 both fail for concrete, checkable reasons (a fixed
    unwinnable knob; a distance that scales 5x under pure radial motion
    while direction is unchanged, computed explicitly), and angle replaces
    distance because it demonstrably does not have that flaw;
(b) the obstacle/justification really is in the primary, stated in close
    to the same words the trace uses, including the paper's own worked
    normalization counterexample which the trace reproduces exactly before
    extending it with its own (verified-correct) angle computation.
No non-primary source exists that the decisive step actually needs — the
one plausible ancestor candidate (Zhang & Li 2007's own theta caveat) says
nothing the RVEA primary doesn't already say more specifically (with the
exact citation [70] the trace's phrasing tracks), so grafting it on would be
decorative, not grounding. No rewrite of reasoning.md was performed; only
this file was added.
