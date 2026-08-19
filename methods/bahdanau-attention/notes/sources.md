# Sources — decisive-step sourcing check (svfix repair pass, W3_ancestors_only)

## Decisive step
reasoning.md ~line 21: hard argmax alignment fails because argmax is discrete/zero-gradient
almost everywhere → backprop can't reach the alignment scorer → the classical MT fallback
(latent-variable alignment + EM) is a bolted-on estimation procedure, not joint end-to-end
training → resolve by relaxing the hard pick to a soft, differentiable weighted sum (softmax
over alignment energies), so gradient flows through the alignment MLP.

## Repair history
A prior attempt's `sound_evidence.backing_quote` cited supp.tex L71-73 (the `W_a∈R^{n×n},
U_a∈R^{n×2n}, v_a∈R^n` weight-matrix shape declarations) as backing for the
differentiability/argmax claim. An independent verifier correctly rejected this: the shape
declarations say nothing about gradients or discreteness. The verifier grepped main.tex +
supp.tex for the literal strings "argmax", "differentiable", "discrete" and found nothing,
and suggested searching non-primary sources (Luong 2015, Xu et al. 2015 "Show, Attend and
Tell") for an explicit hard-vs-soft-attention gradient argument.

## Correct backing IS in the primary — different vocabulary than the verifier grepped for
`src/main.tex` L226-231 states the soft-vs-latent design decision and its gradient
justification directly, using "latent variable" / "backpropagated" rather than
"argmax"/"differentiable"/"discrete" (why the earlier keyword grep missed it):

> "Note that unlike in traditional machine translation, the alignment is not considered
> to be a latent variable. Instead, the alignment model directly computes a soft
> alignment, which allows the gradient of the cost function to be backpropagated
> through. This gradient can be used to train the alignment model as well as the
> whole translation model jointly."
> — src/main.tex, L226–231

This is the resolution half of the decisive step (soft alignment → differentiable →
joint backprop training, replacing a latent-variable/EM treatment). The other half —
that a hard argmax pick has zero gradient almost everywhere and is undefined at ties —
is not asserted by the paper in those words, but it is a directly checkable mathematical
fact about the argmax function (a piecewise-constant step function), not an empirical
claim requiring an external citation; it is the trace's own honest, verifiable computation,
exactly like a worked counterexample computed on the page.

## Quality-gate verdict
sound_as_is. Both prongs of the gate hold:
(a) the decisive step is genuinely derived on the page: the obvious first move (hard
    argmax alignment) fails for a concrete, checkable reason (argmax's derivative is
    zero a.e. / undefined at ties) and the resolution (soft, weighted-sum alignment)
    follows directly from that failure;
(b) the justification for the resolution — soft alignment vs. latent-variable EM,
    gradient backprop through the alignment model — really is stated in the primary
    (main.tex L226-231), not merely implied.
No source grafting performed; no rewrite of reasoning.md was needed. This file records
the corrected primary-source citation so a future auditor does not need to re-derive it.

## Search effort logged (for completeness, per procedure step 2)
- `grep -ril` over refs/ src/ notes/ for alignment/differentiable/argmax/latent terms.
- `grep -i bahdanau SELF_ACCOUNT_SOURCES.md` at repo root — no entry.
- No refs/ directory exists for this method; src/ holds only the primary's own LaTeX
  source (main.tex, supp.tex, search.bbl/tex) — no author self-account or survey
  material is on disk. Given the quality-gate outcome (sound_as_is, justification
  already in the primary), no further external search was required by the procedure.
