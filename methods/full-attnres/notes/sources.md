# Sources — decisive-step sourcing check (svfix, W3_primary_plus_ancestors)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md lines 11-19: put residual / Highway / Hyper-Connections into one
algebraic picture — a lower-triangular depth-mixing matrix `M` where
`h_l = sum_i M_{i->l} v_i` — then use the RANK of `M`'s off-diagonal blocks
as the invariant that pins each baseline to a fixed-width cross-depth state
(residual/Highway = rank 1, Hyper-Connections = rank `m`), which is the
reason a fixed-width recurrent depth-mix cannot be widened away and full
softmax attention over depth is needed to escape the cap.

## Quality-gate verdict: SOUND_AS_IS — no rewrite made

### (a) Genuinely derived on the page, not asserted
The block-rank passage (reasoning.md paragraphs 11-17) is real, checkable
linear algebra worked by hand, including a genuine self-caught bug, not a
hindsight-toned restatement:
- Sets up `M_{i->l} = beta_i^T A^x_{i+1->l} alpha_l` for Hyper-Connections
  and predicts block rank `m` for `m=2`.
- Computes it wrong first: "I formed the product `prod_j A_j` by multiplying
  the `A_j` in an arbitrary order for each `(i,l)` pair" -> measured block
  rank comes out 6, not 2 — a worked counterexample that doesn't close, the
  exact "quantity that doesn't close" the gate asks for, not a staged doubt.
- Diagnoses the real reason (arbitrary order is "not a transition operator";
  a genuine depth transition must compose as the same descending product
  `A_{l-1} A_{l-2} ... A_{i+1}`), fixes it, and re-measures: block rank drops
  to 2, 4, 1 for `m` = 2, 4, 1 respectively — the corrected computation now
  matches the claimed `m`-semiseparable bound.
- Only after the corrected, verified rank invariant is established does the
  trace move to "the score between a destination and a source cannot factor
  through a fixed-width state" -> softmax over depth. The forcing reason
  (fixed block rank = fixed state width, checked and self-corrected on the
  page) precedes and drives the resolution; nothing is asserted first and
  justified after.

### (b) Backed by the primary source's own derivation of the identical framework
`refs/primary/attnres_primary_fulltext.txt` Section 6.2 "Residual Connections
as Structured Matrices" derives the SAME depth-mixing-matrix / semiseparable-
rank framework, with the same object names and the same conclusion per
baseline:

> "The semiseparable rank of M [8] offers a unified lens for comparing them."
(line 999)

> "Since the cumulative products factor through scalar gates, M is
1-semiseparable [8], the same rank as the standard residual but with
input-dependent weights." (Highway; lines 1065-1066)

> "Mi→l = βi⊤ A×i+1→l αl, ... where A×i→j := prod_{k=i+1}^{j} Ak. The m × m
transitions render M m-semiseparable [8]." (Hyper-Connections, Eq. 10;
lines 1076-1078)

The primary's `A×_{i→j} := prod_{k=i+1}^{j} A_k` is exactly the descending,
properly-ordered product the trace discovers it needs after its first
(wrong, arbitrary-order) attempt — i.e. the trace's self-caught bug and fix
land on precisely the operator the primary states, not a different one.
Primary also explicitly frames this in attention terms for the reader
("αl plays the role of a query issued by layer l, βi serves as a key...",
lines ~1121-1124), matching the trace's own move from "separable bilinear
kernel" to depth-wise softmax attention.

This is condition (b) satisfied twice over: the numeric result is both the
trace's own honest, self-correcting computation AND matches the primary's
stated closed-form claim line for line.

### Why no source was grafted in
TRIAGE flagged `refs/explainers/explainers.md` (DataCamp + silkeplessers
blog, both >2KB) as on-disk-but-uncited. Reread both in full: DataCamp
covers the linear-vs-softmax duality, RMSNorm-on-keys rationale, zero-init,
and Full/Block AttnRes cost — all already present in the primary and already
covered elsewhere in reasoning.md (paragraphs 27-46), not at the block-rank
step. silkeplessers is explicitly "high-level; equations/ablations are in
the primary source" (its own words) and adds no rank/matrix content at all.
Neither explainer discusses the semiseparable-rank invariant, the block-rank
computation, or the ordering bug that is the actual decisive step here —
threading either in would be a bolted-on citation the reasoning does not
need, exactly the wave-2 mistake the fix prompt warns against ("a blog stat
that was just the primary's own number restated, and overclaimed
independence; the verifier killed it"). Left uncited, correctly.

`refs/ancestors/{denseformer,hyperconnections,muddformer}.*` are the primary
papers of prior methods (DenseFormer, Hyper-Connections, MUDDFormer), not
self-accounts; they're already load-bearing earlier in reasoning.md
(paragraphs 9-10, the survey of baselines) and hyperconnections.txt is
where the `H_l = H_{l-1} A_l + f(...) beta_{l-1}^T` recurrence itself is
correctly cited from. None of the three carries a self-account (interview,
thesis, retrospective) about the rank framework — that framework is native
to the AttnRes primary's own Discussion section, not inherited from any
ancestor.

`refs/self_accounts/` is empty. `grep -i "attnres\|attention residual\|kimi"
SELF_ACCOUNT_SOURCES.md` at repo root -> no entry. The primary is a
March-2026 Kimi Team technical report (arXiv:2603.15031, github.com/
MoonshotAI/Attention-Residuals) — a multi-author corporate report, not a
single-first-author academic paper with a thesis/Nobel-lecture lineage, and
current enough that no retrospective/interview about it plausibly exists
yet. Given the decisive step already passes the quality gate on (a)+(b)
from material already on disk, no further external search was warranted
(the procedure gates step 2 behind "only continue... if the gate says fix").

## Conclusion
outcome = sound_as_is. No rewrite to reasoning.md, answer.md, or
train_answer.md. No new file threaded into the trace's citations. No
factual errors found in the decisive step or its surrounding derivation.

## Commit-scope note (added during glow repair pass, 2026-08-20)
This file was originally added alongside methods/glow/results/reasoning.md
in commit b4e16ab0b, whose commit message described only the glow
decisive step and did not disclose that the same commit also carried this
104-line full-attnres write-up — an independent verifier flagged that as
an undisclosed scope violation (two methods bundled into one commit, one
of them unmentioned). This content itself was independently re-verified
and is unchanged (see quality-gate verdict above); this note only
re-files it under its own accurately-scoped commit so the audit trail
attributes full-attnres's sourcing check to its own commit rather than
glow's.
