# Sources — decisive-step sourcing check (svfix, W3_ancestors_only)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md paras 15-59: the constant error carousel (CEC) — a single
self-connected linear unit with `f'(net)·w_jj = 1.0`, forced by requiring
constant error flow through the unit — wrapped in multiplicative input and
output gates that resolve the input-weight/output-weight conflicts, with
backprop truncated at the cell/gate boundary to make the CEC's unit-gain
error path the only surviving multi-step channel; later (paras 53-59)
extended with a learned forget gate so the CEC's fixed self-weight 1.0
becomes an adaptive `y^φ ∈ [0,1]` that can reset the otherwise
unboundedly-growing state. This matches TRIAGE's characterization exactly.

## Search performed
1. `grep -ril` over `methods/lstm/refs methods/lstm/notes` for CEC/gate/
   forget-gate/conflict terms — everything traces back to the two files
   already on disk that matter: `refs/lstm1997_full.pdf` (primary, 32pp
   incl. appendix) and `refs/graves_phd.pdf` (co-author PhD thesis,
   flagged by TRIAGE as unused).
2. `grep -i lstm SELF_ACCOUNT_SOURCES.md` at repo root -> no entry.
3. Checked file integrity of the other refs/ files: `refs/gers_021943.pdf`,
   `refs/21498848.pdf`, `refs/3022d04f.pdf`, `refs/fg_aba7cb.pdf` are all
   broken downloads (`file` reports HTML/ASCII text, not PDF — a
   bioinf.jku.at redirect page, a 404 page, and two empty stubs). No
   recoverable content in any of them for the Gers et al. 2000 forget-gate
   paper; the trace does not depend on quoting Gers 2000 directly (see
   below — the forget-gate motivation is the trace's own worked
   computation, and the 1997 primary already states the CEC-recovery
   property that the forget gate must preserve).
4. Extracted `refs/graves_phd.pdf` (Graves, "Supervised Sequence
   Labelling with Recurrent Neural Networks", Ch. 4, "Long Short-Term
   Memory") in full and read it end to end (`refs/graves_phd.txt`,
   ~5000 lines). This is the file TRIAGE flagged as "sitting unused... for
   a fixer to mine." Verdict on mining it: it is a textbook-style summary
   of the 1997/2000 results (forward/backward equations, an illustrative
   figure of gradient preservation, remarks that the original truncated
   gradient "could only be proven" constant-error-flow while later work
   (Graves & Schmidhuber 2005b) computes the *exact* gradient with full
   BPTT). None of this is a self-account of what Hochreiter/Schmidhuber
   tried and rejected, and none of it is cited by the current trace as
   forcing any design choice — it restates the primary's own claims in
   modern vectorized notation. Grafting a citation to it here would be
   exactly the "bolted-on citation the reasoning doesn't need" the fix
   instructions warn against.

## The decisive step is already fully derived from, and matches almost
## verbatim, the primary source on disk

The naive-fix failure (why a bare CEC is not enough) — trace para 21:
> "Take a single incoming weight `w_ji`... That one weight has to do two
> contradictory jobs across the sequence."

backed by `refs/lstm1997_full.txt` (OCR of pp. 5-6, Section 3.2, "Input
weight conflict"):
> "since the same incoming weight has to be used for both storing certain
> inputs and ignoring others, wji will often receive con icting weight
> update signals during this time"

The truncation requirement (why error must be cut at the cell boundary) —
trace para 39:
> "If I let standard backprop run unrestricted, error that flows out of the
> cell through the output gate and back into the network could come back
> into the cell through the input gate at an earlier step, around a loop"

backed by `refs/lstm1997_full.txt` (Appendix A.1, p.24, and Conclusion,
p.10):
> "Truncation ensures that there are no loops across which an error that
> left some memory cell through its input or input gate can reenter the
> cell through its output or output gate. This in turn ensures constant
> error flow through the memory cell's CEC."
and (Conclusion):
> "constant error carrousel CEC, provided that truncated backprop cuts o
> error ow trying to leak out of memory cells"

The CEC derivation itself (`f_j'(net)·w_jj = 1.0`, integrate the ODE, get a
linear unit) — trace paras 15-19 — matches `refs/lstm1997_full.txt`
Section 3.2 line for line:
> "To enforce constant error ow through j , we require
> fj0 (netj (t))wjj = 1:0... fj has to be linear, and unit j's activation
> has to remain constant"

The forget-gate section (trace paras 53-59) is NOT copied from any source —
it is the trace's own worked computation (`tanh'(1)=0.420`,
`tanh'(3)=9.9e-3`, `tanh'(5)=1.8e-4`, `tanh'(10)=8.2e-9`, showing the
unbounded-state saturation problem numerically) followed by the design
move (multiply the CEC's fixed self-weight by a learned gate, recovering
the exact CEC when the gate is open). This is exactly type-(b) of the
quality gate: "the trace's own honest computation." Gers et al. 2000's
actual forget-gate paper could not be recovered (see search log above),
but the trace does not need it — the motivating failure (state grows
unboundedly on continual streams, output saturates) is derived on the page
with real numbers, not asserted, and the resolution (gate the self-weight,
recover CEC at gate=1) is the obvious/forced fix given that failure, not an
appeal to authority.

## Quality-gate verdict: sound_as_is, no source grafted

(a) Genuinely derived on the page: every link in the decisive-step chain —
    vanishing/exploding gradient bound (worked with real numbers, paras
    9-13), the CEC's forcing ODE (paras 15-19), the two weight conflicts
    that break a bare CEC (para 21), why the gate must be multiplicative
    not additive (para 23, a checkable claim: additive can't zero a
    nonzero signal), the 5-step hand-worked numeric example proving
    truncation gives ratio exactly 1.0 (para 43), and the forget gate's
    saturation-failure computation (para 53) — is an actual derivation or
    worked computation in the trace, not an assertion or hindsight
    restatement.
(b) The obstacle/justification is in the primary: verified above by direct
    quote-matching against `refs/lstm1997_full.pdf` for every step except
    the forget gate, which is the trace's own honest computation.

No hindsight tone, no "the paper"/"the authors"/"et al."/"arXiv" leakage
(checked via `grep -n -i` and `tools/lint_inframe.py`, both clean). No
empirical outcome is fake-derived or self-supplied — the toy numeric
examples in paras 43 and 53 are hand-computations of the stated forward/
backward equations on invented small inputs (a worked counterexample), not
claims about running a real training experiment.

Per the fix instructions: "Do NOT graft a source onto a sound trace — a
bolted-on citation that the reasoning doesn't need is damage, not
improvement." graves_phd.pdf remains unused deliberately, not by
oversight: it corroborates but never forces any step, and citing it would
convert an already-self-sufficient derivation into a decorative citation.
