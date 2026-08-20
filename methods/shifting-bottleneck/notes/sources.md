# Sources — decisive-step sourcing check (svfix, W3_primary_only)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines ~111-176: replace Balas's yes/no *criticality* test with a
computable *degree* of bottleneck — the bottleneck machine is
`argmax` over unsequenced `k` of `ell(k, M_0)`, the optimal `L_max` of the
`1|r_j|L_max` single-machine subproblem built from heads/tails via two
longest-path passes.

## Re-verification of TRIAGE's finding
Re-read reasoning.md in full, notes/synthesis.md, and `refs/` (3 PDFs).
TRIAGE (class D) reports every derivation step maps 1:1 onto ABZ 1988
sections, and that neither non-primary PDF forces this step (columbia-bb
gives only an *alternate B&B lower bound* for the one-machine subproblem —
a different part of the method; jss-ip-sbp-dd-2024 only motivates the
project via LP-relaxation weakness, used as context-only material per
notes/synthesis.md's own instruction). Extracted
`refs/adams-balas-zawack-1988.pdf` with `pdftotext -layout` and confirmed
this directly: the primary source itself gives the criticality-is-not-
operational argument almost verbatim to what reasoning.md reconstructs.

`refs/adams-balas-zawack-1988.txt` (p.393, Section "The Shifting Bottleneck
Approach"):

> "This definition certainly makes sense in view of the known fact (Balas
> 1969) that any schedule better than the one associated with S uses a
> selection in which at least one arc of every longest path in Ds is
> reversed. While appealing and theoretically justified, this notion is
> however not sufficiently operational for our purposes: it simply
> partitions the set of machines into critical and noncritical ones without
> offering means of distinguishing between degrees to which a machine
> constitutes a bottleneck. In order to prioritize the machines, we need a
> concept that expresses the bottleneck quality as a matter of degree
> rather than a yes or no property. This quality could be measured, for
> instance, by the marginal utility of the machine in reducing the
> makespan, were it not for the practical difficulty of assessing the
> latter. Instead, we use as a measure of the bottleneck quality of machine
> k the value of an optimal solution to a certain one-machine scheduling
> problem on machine k."
> — refs/adams-balas-zawack-1988.txt, lines 185-201 (Section 2)

reasoning.md's decisive passage (lines 111-121) runs through exactly this
chain, unprompted and in the narrator's own words: criticality is
theoretically justified (cites Balas's arc-reversal fact) but is a yes/no
label that "doesn't tell me which critical machine is *more* of a
bottleneck"; the ideal fix ("marginal effect on the makespan") is named and
rejected as impractical to compute; the resolution ("So I need a computable
proxy. And I already built the right object: the single-machine
subproblem.") lands on the same proxy the primary uses. Checked
`refs/columbia-bb-1rjLmax.pdf` and `refs/jss-ip-sbp-dd-2024.pdf` for any
independent contribution to *this* step — neither mentions criticality,
bottleneck ranking, or marginal utility; both are off-topic to this
specific step (confirming TRIAGE), so grafting either in would be a
name-drop, not grounding.

## Beyond the primary: reasoning.md's own honest computation
The trace does not stop at restating the primary's motivating paragraph —
it goes on to *verify* the criterion is a real, non-arbitrary measure by
working a self-built 3-job/3-machine instance by hand (lines 41-52, 90-99,
132-187): computes heads/tails by two longest-path passes, hand-enumerates
all six orderings of machine 0 to confirm `B(0,∅)=11` (`ell=1`) is optimal,
re-derives the `L_max`/lateness identity numerically, drops the bottleneck
order into the graph and recomputes the longest path to confirm the
makespan rises by exactly `ell=1` as predicted, and brute-forces all 216
machine orderings of the tiny instance to confirm `max_k B(k,∅)=11` is not
just a valid relaxation bound but tight on this instance. This is
deterministic combinatorial arithmetic worked on the page (not a claimed
ML-training observation), so it is legitimate self-derivation, not
fake-derivation or self-supplied observation.

## Quality-gate verdict: sound_as_is
(a) genuinely derived on the page: the obvious first move (Balas
    criticality) is set against a concrete, checkable limitation stated
    directly in the primary (yes/no, not a degree) and the resolution
    (subproblem optimum as computable proxy) follows from that limitation,
    then is independently verified by hand computation and brute force on
    a worked instance.
(b) the justification is in the primary (quoted above, near-verbatim) *and*
    is additionally the trace's own honest, checkable arithmetic.

No non-primary source forces this step; the two non-primary PDFs on disk
are genuinely off-topic to it (alternate B&B lower bound; LP-relaxation
motivation, both elsewhere/context per notes/synthesis.md). Grafting either
in here would be exactly the wave-2 mistake this fix pass is warned
against. No rewrite made. In the course of re-verifying the worked example
by hand (op0/op3/op7 permutation check, lines 132-151), found one
imprecise phrase — "It's the only one that keeps op0... early" — where a
tied second permutation (op3,op0,op7) also reaches `B=11`; this is loose
phrasing around a hand-checked tie, not a wrong constant/sign/formula (the
value `B=11`, `ell=1` and every downstream use of it are correct), so it
does not meet the fix-trigger bar for "a factual error" and was left
alone per the no-padding/no-unneeded-edits rule.

## Commit-scope note (added during rvea repair pass, 2026-08-20)
This file was originally added alongside methods/rvea/notes/sources.md in
commit 439357c64, whose commit message described only the rvea decisive
step and did not disclose that the same commit also carried this
92-line shifting-bottleneck write-up — an independent verifier flagged
that as an undisclosed scope violation (two methods bundled into one
commit, one of them unmentioned). This content itself was independently
re-verified and is unchanged (see quality-gate verdict above); this note
only re-files it under its own accurately-scoped commit so the audit
trail attributes shifting-bottleneck's sourcing check to its own commit
rather than rvea's.
