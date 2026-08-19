# Sources — decisive-step sourcing check (svfix, W3_ancestors_only)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines ~21-33: the method of multipliers's augmented Lagrangian
Lρ(x,z,y)=f(x)+g(z)+yᵀ(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖² is robust but, when minimized
*jointly* over (x,z), the expanded quadratic penalty produces a cross term
ρ(Ax)ᵀ(Bz) that binds the two blocks together — the same coupling failure
already worked out earlier in the trace for the separable dual-decomposition
case (∑_{i≠j}(Aᵢxᵢ)ᵀ(Aⱼxⱼ)). The fix: replace the joint (x,z) minimization
with one Gauss-Seidel sweep — minimize over x with z frozen, then over z with
the new x frozen, then dual-update with step ρ — which turns the bilinear
cross term into a linear (hence decoupled) term in each subproblem.

## Search performed
- `grep -ril "gauss-seidel\|joint minimization\|alternating direction" methods/admm/refs methods/admm/notes` → both refs/boyd-admm.pdf and refs/parikh-prox.pdf; boyd-admm.pdf is the direct hit (parikh-prox.pdf covers proximal operators, not the joint-vs-alternating point).
- `grep -i admm SELF_ACCOUNT_SOURCES.md` → no entry for admm.
- Web: "ADMM Boyd retrospective", "alternating direction method of multipliers self-account", "Gabay Mercier 1976 ADMM history", "Glowinski Marroco 1975 alternating direction", "Eckstein Bertsekas ADMM history interview" — no first-author self-account, thesis, or interview transcript surfaced beyond citation trails that lead back to the same Boyd et al. survey and its own bibliography (Glowinski-Marroco 1975 and Gabay-Mercier 1976 are 1970s French-language numerical-analysis papers with no accessible retrospective; Eckstein's 1989 MIT PhD thesis on Douglas-Rachford/ADMM is cited by the survey but is a technical derivation, not a first-person account of *why* the joint-solve obstacle motivated the alternating fix — that exact explanation already lives, explicitly, in the primary itself).

## Correct backing IS in the primary, stated directly (not merely implied)
`refs/boyd-admm.pdf` (Boyd, Parikh, Chu, Peleato, Eckstein, *Distributed
Optimization and Statistical Learning via ADMM*, §3.1, p.14) states the
joint-vs-alternating obstacle and its fix in essentially the same terms the
trace derives:

> "Here the augmented Lagrangian is minimized jointly with respect to
> the two primal variables. In ADMM, on the other hand, x and z are
> updated in an alternating or sequential fashion... ADMM can be viewed
> as a version of the method of multipliers where a single Gauss-Seidel
> pass... over x and z is used instead of the usual joint minimization.
> Separating the minimization over x and z into two steps is precisely
> what allows for decomposition when f or g are separable."
> — refs/boyd-admm.pdf, p.14 (§3.1), extracted via `pdftotext -layout`

The trace's rewrite (freezing z turns the cross term linear, restoring
separability) is the algebraic mechanism *underneath* this same sentence —
it is a faithful, first-person re-derivation of the primary's own claim, not
a name-drop and not a distinct external source.

## Quality-gate verdict
sound_as_is. Both prongs of the gate hold:
(a) the decisive step is genuinely derived on the page: the obvious first
    move (joint (x,z) minimization of the augmented Lagrangian, i.e. the
    unmodified method of multipliers applied to the two-block problem)
    fails for a concrete, checkable reason — expanding the squared penalty
    produces a cross term ρ(Ax)ᵀ(Bz) that mixes the blocks, worked out
    explicitly on the page (mirroring the earlier ∑_{i≠j} cross-term
    computation for dual decomposition) — and the resolution (freeze one
    block, alternate) follows directly from that failure being isolated to
    the joint step;
(b) the justification really is in the primary: boyd-admm.pdf §3.1 states,
    in essentially the same words, that ADMM = method-of-multipliers with
    one Gauss-Seidel pass replacing the joint minimization, and that this
    separation is precisely what restores decomposability.
No non-primary source exists to graft on top of a step the primary already
derives directly — grafting one (e.g. a Gabay-Mercier citation with nothing
new to add) would be decorative, not grounding, per the quality gate's
explicit warning against bolting citations onto sound traces. No rewrite of
reasoning.md was performed; only this file was added.
