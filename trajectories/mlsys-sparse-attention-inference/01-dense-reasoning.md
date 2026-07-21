The module is the whole point, but it drops into a frozen model, and with the budget removed that
model's attention is just full scaled-dot-product attention — so the reference I have to climb toward is
exactly that: the unmodified attention forward, every query reading every key. The task hands me one
`SparseAttention` instance monkey-patched into every attention layer of a 1.5B instruct model, replacing
the attention forward and nothing else — no retraining, no weight edits, no growing the sequence. For each
query token the module decides which preceding keys get non-zero attention, and the kept fraction of
`(q, k)` pairs — the density — has to stay at or under `0.25` (plus `0.02` slack) averaged over all 24
layers, or the harness aborts. Before I argue about *how* to spend a quarter of the attention matrix, I
need to know what the full matrix scores, because the sparse rungs only make sense as fractions of that.
So the first rung is the oracle: spend the whole matrix, report density `1.0`, and let the harness skip
the budget check for this one baseline.

The layer computes `out_t = Σ_j softmax_j(q_t · k_j / √D) v_j`; causality restricts the allowed set for
query `t` to `{0, …, t}`, the lower triangle, so stacked it is `softmax(QKᵀ/√D + M)V` with `M` the
`−∞`-above-the-diagonal bias. That is precisely the pretrained Qwen attention. The one thing I must not
touch is the scale, and the number says why: `D = 128`, each logit is a 128-term dot product, and if the
trained coordinates are roughly zero-mean and uncorrelated the sum has standard deviation
`√128 ≈ 11.3`. Logits spread over ±11 drive the softmax into saturation — a gap of 22 is a weight ratio of
`e²² ≈ 3.6·10⁹`, a near one-hot spike with no blending. Dividing by `√D` pulls the spread back to unit
scale, the regime the softmax is responsive in and the regime the model was *trained* under, so its
learned `q, k` magnitudes are calibrated to exactly this normalization. Touch the scale or insert a mask
and I am no longer measuring this model's ceiling; the three numbers this rung produces stop being a valid
target. The oracle's discipline is restraint: reproduce the trained computation exactly and read off what
it scores.

The cost is why the budget exists. `QKᵀ` is `N×N` per head per layer; at `N = 8192` that is `67M` entries
per head, `288` such matrices across 12 heads and 24 layers, and the score matmul alone is
`N²·D ≈ 8.6` billion multiply-adds per head. But the binding constraint is memory traffic, not FLOP: every
one of those `N²` logits is written out, read back for the softmax, and the full K/V streamed through for
the weighted sum, so wall-clock is dominated by moving the `O(N²)` matrix across the memory hierarchy.
That is the quadratic that pins long-context inference — doubling the context quadruples the attention
work — and exactly why dense attention can be the oracle but never the deployed module.

I still have a choice of *how* to realize the full matrix, and it matters only because a careless
realization would OOM or silently change the graded numbers. The explicit textbook form — materialize
`QKᵀ`, add the causal bias, softmax, times `V` — pays the full quadratic memory tax and courts fp16
overflow on logits that reach the tens before scaling, forcing an fp32 upcast that doubles the transient
memory again. Hand-rolling a tiled flash pass is memory-sane but re-derives the online-softmax
bookkeeping, and any drift in how I accumulate the running statistics makes the oracle differ subtly from
the model's own attention — the one thing the reference must not do. PyTorch's fused
`scaled_dot_product_attention` *is* that tiled algorithm, kernel-verified, building the causal bias
internally when I pass `is_causal=True`; it gives the identical mathematical result, never materializes the
`N²` matrix in HBM, and does the reduction stably. So the oracle hands `q, k, v` straight to fused SDPA
with `attn_mask=None` — not a shortcut that changes the answer, the memory-wise sane way to realize the
identical computation.

The rest is settled upstream, and redoing any of it would corrupt the reference. The loop replicates GQA
before the module is called (I receive 12 heads on both Q and K/V), applies RoPE to `q, k` before they
reach me (so I must not re-rotate, or I would score in a positional frame the model never saw), and passes
`is_causal=True` and the default scale. Everything that makes this the *pretrained model's* attention is
fixed above me, and my edit is the empty one: forward the tensors and return. `self.last_density` is `1.0`
literally — a dense causal forward keeps every admissible pair, so a full lower-triangular mask would sum
to exactly `N(N+1)/2` over that same denominator, and counting a phantom mask would only invite a
floating-point disagreement at the boundary. The harness recognizes this rung through `ALLOW_DENSE_FLAG=1`,
which forwards `--allow-dense` so `enforce_budget` skips the `0.25 + 0.02` ceiling for the oracle alone;
that asymmetry is what makes the oracle a *reference* rather than a competitor — it is not satisfying the
constraint, it is defining the quality the constrained rungs are measured against.

Now what the rung must produce, because that is the reason to run it. It fixes three target numbers, and
the three environments probe different failure modes, so I predict the shape. On `niah_8k` the task plants
one fact — a needle, "the special magic number is …" — at a random position in a long synthetic haystack
and asks a query that targets it. Full attention lets the question query's `q` score directly against the
needle key; that single large logit dominates the softmax and the needle value is routed forward in one
hop, regardless of whether the needle sits at position 200 or 7000. Because dense keeps every key the
needle is always in the kept set, so I expect retrieval essentially perfect, near `1.0`. This is the
cleanest discriminator on the whole ladder precisely because the answer lives in exactly one position —
there is no partial credit for being *near* the needle, so a sparse pattern that fails to cover the
needle's block falls off a cliff here while the oracle does not. The failure mode a sparse rung faces is
sharp: if its mask does not include the needle's position, that dominant logit is masked to `−∞` before
the softmax sees it, the weight is renormalized over filler, and the needle value never enters the output.

The two QA environments behave oppositely, and I have to hold the distinction firmly or I will misread
every sparse rung later. On `longbench_qasper` and `longbench_multifieldqa_en` the metric is token-overlap
F1, and these are genuinely hard for a 1.5B instruct model even with full attention — the model's own
competence, not attention coverage, is the binding limit. A small model reading a long paper and asked to
extract a specific answer will paraphrase, miss, or hedge, and F1 punishes all of that. So F1 folds
together two different things — did the attention reach the evidence, and could the 1.5B turn that evidence
into the reference phrasing — and at the oracle the first is guaranteed, which makes its QA F1 almost
purely a readout of the second, answer-formatting competence, capping it well below `1.0`. I must treat
that modest number as the *realistic* ceiling: when a sparse rung comes in below it, part of the shortfall
is lost evidence (which sparsity controls) and part is this same formatting cap (which sparsity cannot
fix), and only the *movement* relative to the oracle isolates the sparsity effect. And unlike NIAH these
degrade gracefully — the evidence is distributed across the document, so a mask covering *some* of the
relevant spans still composes a partial answer and keeps part of the F1. NIAH is a position-coverage
cliff; the QA tasks are a distributed-evidence slope. I do not know the three numbers yet, but I am
confident of the shape: NIAH near the top, the two QA scores low and graded.

One premise underlies the whole ladder, and if it were false the budget would be hopeless: attention in a
trained model is empirically *concentrated* — for a given query the softmax puts almost all its mass on a
handful of keys and leaves a long near-zero tail. A query at position `t` spreading weight uniformly over
its `t+1` admissible keys would have entropy `log(t+1) ≈ 9` nats at `t ≈ 8000`; trained attention sits
nowhere near that, with an effective number of attended keys — `exp(entropy)` — of a few tens, not
thousands. That gap is why a `0.25` budget is even plausible: keeping a quarter of the causal pairs means
keeping about `0.25·(t+1) ≈ 2000` of a late query's `~8000` admissible keys, vastly more than the softmax
leans on. So the budget is generous relative to the true support; the difficulty is never its *size* but
its *location* — the mass-carrying keys move with the query, and a method that spends 2000 keys in the
wrong place keeps a quarter of the matrix and still misses the handful that mattered. The oracle has the
support for free; every sparse rung is a bet about where it is.

The contract measures density against `N(N+1)/2`, not `N²`, and that is what makes density an honest
number rather than a lever. The upper triangle is not "kept at weight zero" — it is structurally forbidden
by causality, never a pair the model could spend attention on. Dividing by the full `N²` would let a
causal method report roughly half its true density and sneak a mask that is actually at budget under the
ceiling by a factor of two. Normalizing by the admissible count reports the fraction of *reachable* pairs
kept, which is what the budget constrains, and it is why the oracle reads exactly `1.0` and not `0.5`.
Every sparse rung reports this same admissible-fraction, and I will trust the *measured* count over any
closed-form estimate whenever the two disagree at the diagonal or the sequence boundary.

So the oracle's three numbers set the bar, and the gap to each sparse rung is the signal that tells me
whether the missing quality is a coverage problem — fix it with anchors and windows — or a routing problem
— fix it by making the pattern depend on the query. The falsifiable expectation I set up for the first
sparse rung: under the same budget, a static content-blind pattern should claw back a usable share of the
QA F1, because distributed evidence is partly reachable by any reasonable local-plus-anchor mask, but it
should *collapse* on NIAH from the oracle's near-perfect retrieval toward chance, because nothing about a
fixed mask routes to a needle whose position it did not anticipate. If that NIAH collapse happens while the
QA numbers hold, the diagnosis is confirmed and it points the ladder straight at query-adaptivity; if a
static mask somehow keeps NIAH near the ceiling, my whole reading of the static-versus-adaptive split is
wrong. The distilled oracle module — the empty edit that hands `q, k, v` to fused SDPA and reports density
`1.0` — is in the answer.
