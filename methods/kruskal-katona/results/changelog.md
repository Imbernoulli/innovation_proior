# Changelog

## 2026-08-20 — svfix(W3_notes_unclear) repair pass

**Factual error fixed.** The "stable ⊉ colex" counterexample at the decisive step (reasoning.md,
the paragraph beginning "Now I'd love to say 'the only stable families are the colex initial
segments...'") claimed the 2-uniform family $\{1,3\},\{2,3\}$ is stable ("there's no other shift
available"). This is false: applying $S_3$ (i=3, not i=2) to $\{2,3\}$ gives $\{2,3\}\setminus\{3\}\cup\{1\}=\{1,2\}$,
which is *not* already in the family, so the shift is unblocked and fires — $S_3(\{\{1,3\},\{2,3\}\})
=\{\{1,3\},\{1,2\}\}\neq\{\{1,3\},\{2,3\}\}$. The family is therefore not stable at all (verified
programmatically against the shift definition already stated earlier in the same file, line 27).

Replaced with a correct, minimal witness: $\{1,3\},\{1,4\}$. Both sets already contain $1$, so the
shift's defining precondition ($1\notin F$) fails for both before any blocking check is even needed —
the family is stable trivially. It is not the colex initial segment ($\{1,2\},\{1,3\}$), and, as an
extra checkable fact, it even ties the colex segment's shadow size (both shadows have size 3:
$\{1\},\{3\},\{4\}$ vs $\{1\},\{2\},\{3\}$) — sharpening the point that shifting can converge on a
different-shaped stable family rather than uniquely forcing colex. Verified against the shift
definition and against brute-force shadow computation.

Propagated the same fix to `results/train_answer.md` (identical claim in the shift/stable paragraph).
`results/answer.md` does not state this specific counterexample (it states the theorem/proof directly
without the "test that hope" narrative beat), so no change needed there. No change to code (the
python block only implements cascade/colex/shadow functions, not the shift operator).

No external source added at this step: notes/synthesis.md and refs/frankl-1984-shifting.{pdf,txt}
(the primary, P. Frankl, "A new short proof for the Kruskal-Katona theorem," Discrete Math. 48 (1984)
327-329) plus refs/das-kruskal-katona-lecture.txt (a faithful exposition of that same Frankl proof,
with the matching remark "If the only stable families were the initial segments of the colexicographic
order ... Sadly, this is not the case") corroborate that stable does not imply colex, but neither
source gives a specific witnessing example — so the counterexample itself is necessarily the trace's
own on-page computation. A prior fix attempt apparently grafted the Das "Sadly, this is not the case"
sentence onto this passage as a citation; an independent verifier correctly rejected that as decorative
(the trace's own — but broken — example did not depend on it). This pass does not reinstate any such
citation; it corrects the on-page computation itself so the self-derived counterexample is actually
true.
