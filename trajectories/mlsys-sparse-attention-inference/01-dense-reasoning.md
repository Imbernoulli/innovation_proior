The module is the whole point, but it drops into a frozen model, and with the budget removed that
model's attention is just full scaled-dot-product attention — so the reference I have to climb toward is
exactly that: the unmodified attention forward, every query reading every key. The task hands me one
`SparseAttention` instance monkey-patched into every attention layer of a 1.5B instruct model, replacing
the attention forward and nothing else — no retraining, no fine-tuning, no weight edits, no growing the
sequence. For each query token the module gets to decide which of the preceding keys receive non-zero
attention, and the kept fraction of `(q, k)` pairs — the density — has to stay at or under `0.25` (plus a
small `0.02` slack) averaged across all 24 layers, or the harness aborts the run outright. Before I spend
a single forward arguing about *how* to spend a quarter of the attention matrix, I want to know what the
full matrix can do, because the sparse rungs only make sense as fractions of that. So the first rung is
the oracle: spend the whole attention matrix, report the true density of `1.0`, and let the harness skip
the budget check for this one baseline.

Let me write down what the layer actually computes so I am sure the oracle is the genuine reference and
not some near-copy. Each query `q_t` scores against every key it is allowed to see, the scores go
through a softmax, and the output is the softmax-weighted blend of the values:
`out_t = Σ_j softmax_j(q_t · k_j / √D) v_j`. Causality forbids reading the future, so for this LLM the
allowed set for query `t` is `{0, …, t}` — the lower triangle. Stack it and it is
`softmax(QKᵀ/√D + M)V` with `M` the additive `−∞`-above-the-diagonal causal bias. That is precisely the
computation the pretrained Qwen attention performs; nothing about the oracle deviates from it. The scale
defaults to `1/√D` because the dot product of two `D`-dimensional vectors has standard deviation `√D`,
and dividing by `√D` puts the logit scale back at unity so the softmax sits in its responsive region —
the same scaling the model was trained under, which I must not change if this is to be a faithful upper
bound. The moment I touch the scale or the mask I am no longer measuring the model's own ceiling; I am
measuring some other model's, and the three numbers this rung produces would stop being a valid target.
So the discipline of the oracle is restraint: reproduce the trained computation exactly, change nothing,
and read off what it scores.

The cost is the reason the budget exists in the first place, and I want to be precise about it because
the entire ladder is a reaction to this one term. The score matrix `QKᵀ` is `(N × N)` per head per layer.
At the 8K context this benchmark evaluates, that is on the order of tens of millions of entries per head
before the softmax even runs, and the model carries 12 heads across 24 layers, so the full forward
forms and softmaxes hundreds of such matrices. Forming and softmaxing all of it is `O(N²D)` compute and
`O(N²)` memory traffic — and crucially the memory traffic is the binding constraint, because every one of
those `N²` logits has to be written, read back for the softmax, and the full key/value tensors streamed
through for the weighted sum. That is the quadratic that pins long-context inference: doubling the context
quadruples the attention work, and the KV reads alone scale with `N` per query, so the matrix grows
faster than any hardware budget can absorb. This is exactly why dense attention can be the *oracle* but
never the *deployed module* — the whole task is to get the patched module under `0.25` density without
losing the quality this full computation buys, and I cannot even state what "losing quality" means until
I have measured the quality the full computation buys. So the oracle has to run first.

The scaffold makes the implementation almost trivial, and that is the point — the oracle should be the
*least* clever rung. I have to be careful here about what the fixed loop already does for me, because if
I redo any of it inside the module I corrupt the reference. The loop replicates GQA before the module is
called: the backbone has 12 query heads and 2 KV heads, but the harness applies the key/value
replication so I receive 12 heads on both Q and K/V and never touch the grouping myself. It applies RoPE
to `q` and `k` before they reach me, so the rotary position information is already baked into the vectors
I score — I must not re-rotate. It passes `is_causal=True` and the default `scale`. So everything that
makes this the *pretrained model's* attention — the head layout, the positional encoding, the causal
direction, the trained scale — is settled upstream, and my edit is the empty one: hand `q, k, v`
straight to PyTorch's fused scaled-dot-product attention with `attn_mask=None` and `is_causal` forwarded,
which builds the lower-triangular `−∞` bias internally and dispatches to an optimized kernel — the same
math as the explicit three-step form, far less memory written out, identical output. Using the fused
kernel rather than a hand-written `QKᵀ → softmax → V` is not a shortcut that changes the answer; it is
the numerically and memory-wise sane way to realize the identical computation, and it keeps the oracle
from gratuitously OOM-ing on the very `N²` matrix that motivates the rest of the work.

The only bookkeeping the contract demands is `self.last_density`, and here it is honest and constant: a
dense causal forward attends to every admissible `(q, k)` pair, so the kept fraction over the
`N(N+1)/2` causal denominator is exactly `1.0`. I set it to `1.0` literally rather than counting a mask,
because there is no mask — nothing is dropped, and counting a phantom mask would only invite a rounding
disagreement with the contract's denominator. The harness recognizes this rung through the
`ALLOW_DENSE_FLAG=1` env var, which forwards `--allow-dense` to the loop so `enforce_budget` skips the
`0.25 + 0.02` ceiling for the oracle alone; every other rung will be aborted if its aggregated mean
density crosses that line, so the oracle is structurally the only baseline permitted to report `1.0`.
That asymmetry is deliberate and it is what makes the oracle a *reference* rather than a *competitor*: it
is not trying to satisfy the constraint, it is defining the quality the constrained rungs are measured
against.

Now reason about what this rung must produce, because that is the entire reason to run it. The oracle
fixes three target numbers — one per environment — that every sparse rung is graded against, and the
three environments are chosen to probe different failure modes, so I should predict how the ceiling lands
on each. On `niah_8k`, retrieval at 8K context: the needle is a single fact buried in a long synthetic
haystack, and full attention can route any query directly to the key that holds it in one hop, so I
expect the oracle to retrieve essentially perfectly. This is the cleanest discriminator on the whole
ladder, because the answer lives in exactly one position, so a sparse pattern that fails to cover the
needle's block falls off a cliff here while the oracle does not — there is no partial credit for being
*near* the needle. On `longbench_qasper`, scientific-paper QA, and `longbench_multifieldqa_en`,
multi-field long-document QA, the metric is F1 against reference answers, and these are genuinely hard for
a 1.5B instruct model even with full attention — the model's own competence caps the score well below
1.0, so the oracle's F1 here is modest and is the *realistic* ceiling, not a perfect one. I have to hold
that distinction firmly, because if I forgot it I would read a low sparse F1 as a sparsity failure when
it is partly just the model's ceiling. The two QA tasks are not a retrieval cliff; they degrade more
gracefully, because the evidence is distributed across the document rather than hidden in one position,
so a sparse pattern that covers *some* of the relevant spans can still answer partially.

That split is the diagnosis I carry into the ladder, and it is worth naming the precise tension the
sparse rungs inherit, because the oracle is what makes the tension legible. A sparse method gets a
density budget `ρ = 0.25`: it may keep a quarter of the causal pairs. The question is *recall* — of the
keys that dense attention actually put weight on, how many does the kept set retain? NIAH is the
position-coverage test: did the sparse mask happen to include the one block the needle sits in? The QA
tasks are the distributed-evidence test: did the sparse mask cover enough of the document's relevant
spans to compose an answer? A static pattern will pass or fail NIAH largely by luck of where the needle
lands relative to its fixed window and anchors, and will lose graded F1 on the QA tasks in proportion to
how much relevant context it drops. A content-adaptive pattern should do better on NIAH precisely because
it can *choose* the needle's block from the query, and better on QA because it can route to the spans the
query is actually asking about. The whole reason the oracle keeps *everything* is that with full density
recall is trivially total — every high-weight key is, by construction, kept — and that is exactly the
property a sparse method has to approximate cheaply: preserve the high-recall retrieval tokens (the ones
dense attention would have weighted) while it is free to drop the long tail of near-zero-weight keys that
the softmax was going to ignore anyway.

So the oracle's three numbers set the bar, and the gap between them and each sparse rung's numbers is
exactly the signal that tells me whether the missing quality is a coverage problem (fix it with anchors
and windows) or a routing problem (fix it by making the pattern depend on the query). I expect the sparse
rungs to recover most of the QA F1 but to struggle on NIAH until the selection becomes query-aware — that
is the hypothesis the next three rungs will test against these measured ceilings. Concretely, the
falsifiable expectation I am setting up for the very first sparse rung: under the same density budget, a
static content-blind pattern should claw back a usable share of the oracle's QA F1, but it should
*collapse* on NIAH from the oracle's near-perfect retrieval toward chance — because nothing about a fixed
mask routes to a needle whose position it did not anticipate. If that NIAH collapse happens while the QA
numbers hold, the diagnosis is confirmed and it points the ladder straight at query-adaptivity; if a
static mask somehow keeps NIAH near the ceiling, my whole reading of the static-vs-adaptive split is
wrong and I would have to rethink. The distilled oracle module — the empty edit that hands `q, k, v` to
fused SDPA and reports density `1.0` — is in the answer.
