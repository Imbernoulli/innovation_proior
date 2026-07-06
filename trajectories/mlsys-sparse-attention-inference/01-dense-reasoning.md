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
computation the pretrained Qwen attention performs; nothing about the oracle deviates from it. Now the
scale is not a decoration I can adjust, and I want to check that with an actual number rather than
asserting it. The head dimension here is `D = 128` (the backbone's hidden width divided over its query
heads), and each logit is a dot product `q_t · k_j = Σ_{d=1}^{128} q_{t,d} k_{j,d}`. If the trained
coordinates are roughly zero-mean with `O(1)` per-coordinate magnitude and are not strongly correlated
across the 128 dimensions, that sum has variance on the order of `D` and therefore a standard deviation
of `√D = √128 ≈ 11.3`. Logits with a spread of ±11 drive the softmax straight into saturation: a gap of
22 between two logits is a weight ratio of `e²² ≈ 3.6·10⁹`, so the distribution collapses to a near
one-hot spike and the model can no longer blend. Dividing by `√D` pulls that spread back to unit scale,
which is exactly the regime the softmax is responsive in — and it is the regime the model was *trained*
under, so its learned `q, k` magnitudes are calibrated to precisely this normalization. The moment I
touch the scale or insert a mask I am no longer measuring this model's ceiling; I am measuring some other
model's, and the three numbers this rung produces would stop being a valid target. So the discipline of
the oracle is restraint: reproduce the trained computation exactly, change nothing, and read off what it
scores.

The cost is the reason the budget exists in the first place, and I want to be precise about it in real
units because the entire ladder is a reaction to this one term. The score matrix `QKᵀ` is `(N × N)` per
head per layer. At the 8K context this benchmark evaluates, `N = 8192`, so `N² ≈ 67.1` million entries
per head before the softmax even runs, and the causal triangle the density is measured against holds
`N(N+1)/2 ≈ 33.6` million admissible pairs. The model carries 12 query heads across 24 layers, so a full
forward forms and softmaxes 288 such matrices. If I materialize one layer's logits for all 12 heads in
float32 that is `12 · 67.1M · 4 bytes ≈ 3.2 GB` of scores — for a single layer, transient — and the
compute for the score matmul alone is `N²·D ≈ 8.6` billion multiply-adds per head, which across the 288
head-layers and counting both the `QKᵀ` and the `·V` matmuls comes to roughly `10` TFLOP of attention
arithmetic for one 8K forward, on top of the MLP work. And the binding constraint is not the FLOP but the
memory traffic: every one of those `N²` logits has to be written out, read back for the softmax, and the
full key/value tensors streamed through for the weighted sum, so the wall-clock is dominated by moving
the `O(N²)` matrix across the memory hierarchy. That is the quadratic that pins long-context inference —
doubling the context quadruples the attention work and the KV reads scale with `N` per query — and it is
exactly why dense attention can be the *oracle* but never the *deployed module*. The whole task is to get
the patched module under `0.25` density without losing the quality this full computation buys, and I
cannot even state what "losing quality" means until I have measured the quality the full computation
buys. So the oracle has to run first.

Given that I am going to run the full matrix, I still have a choice about *how* to realize it, and it is
worth being deliberate because a careless realization would either blow up memory or, worse, silently
change the numbers the whole ladder is graded against. Three ways are on the table. The first is the
explicit textbook form — form `QKᵀ`, add the causal bias, softmax, multiply by `V` — written out in
PyTorch ops. That materializes the `3.2 GB`-per-layer logit tensor to HBM and reads it back, which is the
traffic I just called the binding cost, and doing the exponential in fp16 courts overflow on logits that
can legitimately reach the tens before scaling; I would have to upcast to fp32 for stability, doubling the
transient memory again. It works, but it pays the full quadratic memory tax and adds a precision hazard
for no benefit. The second is to hand-roll a tiled, flash-style pass myself — stream K/V blocks, keep a
running max and normalizer, never write the full matrix. That is the memory-sane algorithm, but writing
it by hand means re-deriving the online-softmax bookkeeping, and any drift in how I accumulate the running
statistics would make the oracle's outputs differ subtly from the model's own attention — which is the
one thing the reference must not do. The third is PyTorch's fused `scaled_dot_product_attention`, which
*is* that tiled algorithm, kernel-verified, dispatching to an optimized backend that builds the causal
`−∞` bias internally when I pass `is_causal=True`. It gives the identical mathematical result as the
explicit form, never materializes the `N²` matrix in HBM, and does the reduction in a numerically stable
way I do not have to re-audit. So the oracle hands `q, k, v` straight to fused SDPA with `attn_mask=None`
and `is_causal` forwarded. Using the fused kernel is not a shortcut that changes the answer; it is the
memory-wise sane way to realize the identical computation, and it keeps the oracle from gratuitously
OOM-ing on the very `N²` matrix that motivates the rest of the work.

The scaffold makes the rest of the implementation almost trivial, and that is the point — the oracle
should be the *least* clever rung. I have to be careful here about what the fixed loop already does for
me, because if I redo any of it inside the module I corrupt the reference. The loop replicates GQA before
the module is called: the backbone has 12 query heads and 2 KV heads, but the harness applies the
key/value replication so I receive 12 heads on both Q and K/V and never touch the grouping myself. It
applies RoPE to `q` and `k` before they reach me, so the rotary position information is already baked into
the vectors I score — I must not re-rotate, or I would be applying the rotation twice and scoring in a
positional frame the model never saw. It passes `is_causal=True` and the default `scale`. So everything
that makes this the *pretrained model's* attention — the head layout, the positional encoding, the causal
direction, the trained scale — is settled upstream, and my edit is the empty one: forward the tensors to
fused SDPA and return. Every place I might be tempted to "help" is a place I would introduce a deviation
from the reference, so the correct oracle is defined as much by what I refuse to add as by the single line
I keep.

The only bookkeeping the contract demands is `self.last_density`, and here it is honest and constant, but
I want to *check* it rather than assert it. A dense causal forward attends to every admissible `(q, k)`
pair. If I built the full lower-triangular boolean mask and counted it, `mask.sum()` would be exactly
`N(N+1)/2`, and the contract's causal denominator is that same `N(N+1)/2`, so the kept fraction is
`N(N+1)/2 ÷ N(N+1)/2 = 1.0` identically, with no rounding slack. That is why I set `last_density = 1.0`
literally rather than counting a phantom mask — there is no mask, nothing is dropped, and counting one
would only invite a floating-point disagreement with the denominator at the boundary. The harness
recognizes this rung through the `ALLOW_DENSE_FLAG=1` env var, which forwards `--allow-dense` to the loop
so `enforce_budget` skips the `0.25 + 0.02` ceiling for the oracle alone; the aggregation the harness runs
is a mean over the 24 layers, so any other rung whose 24-layer mean density crosses `0.27` is aborted and
scores nothing, while the oracle reports `1.0` at every layer and is waved through. That asymmetry is
deliberate and it is what makes the oracle a *reference* rather than a *competitor*: it is not trying to
satisfy the constraint, it is defining the quality the constrained rungs are measured against.

Now reason about what this rung must produce, because that is the entire reason to run it. The oracle
fixes three target numbers — one per environment — that every sparse rung is graded against, and the
three environments are chosen to probe different failure modes, so I should predict how the ceiling lands
on each and be honest about which predictions are near-certain and which are only expectations. Take
`niah_8k` first, retrieval at 8K context. The task plants a single fact — a needle, something like "the
special magic number is …" — at a random position in a long synthetic haystack, and asks a query that
targets it. Full attention lets that query's `q` vector score directly against the needle key's `k`; if
the model can align them at all, that one logit dominates the softmax and its value is routed forward in
a single hop, regardless of whether the needle sits at position 200 or 7000. Because dense keeps *every*
key, the needle is always in the kept set — recall of the one high-weight key is trivially total — so I
expect the oracle to retrieve essentially perfectly, near `1.0`. This is the cleanest discriminator on the
whole ladder precisely because the answer lives in exactly one position: there is no partial credit for
being *near* the needle, so a sparse pattern that fails to cover the needle's block falls off a cliff here
while the oracle does not.

To make the one-hop claim concrete rather than a slogan, trace it. The haystack is thousands of filler
sentences; somewhere among them sits "the special magic number is 7429387," and the prompt ends with
"what is the special magic number?" The query token for that question produces a `q` whose direction was
shaped, during pretraining, to align with keys that encode "magic number is …". Against the needle key
that alignment is large; against the thousands of filler keys it is small and roughly unstructured. In
the dense forward that single large logit sits in the query's row, survives the softmax as the dominant
weight, and pulls the needle value — carrying the token "7429387" — into the output in one step, with no
intermediate hop required. Because the oracle drops nothing, this works no matter where the needle was
planted, so retrieval accuracy should be near `1.0`. The failure mode a sparse rung faces is now sharp:
if its mask does not include the needle's position, that dominant logit is masked to `−∞` before the
softmax ever sees it, the weight is renormalized over filler, and the needle value never enters the
output. There is no graceful version of missing the needle.

The two QA environments behave oppositely, and I have to hold the distinction firmly or I will misread
every sparse rung later. On `longbench_qasper`, scientific-paper QA, and `longbench_multifieldqa_en`,
multi-field long-document QA, the metric is token-overlap F1 against reference answers, and these are
genuinely hard for a 1.5B instruct model even with full attention — the model's own competence, not its
attention coverage, is the binding limit. A small model reading a long scientific paper and asked to
extract a specific answer will often paraphrase, miss, or hedge, and F1 punishes all of that. So the
oracle's F1 on these will be *modest*, well below `1.0`, and — this is the load-bearing point — that
modest number is the *realistic* ceiling, not a perfect one. If I forgot this I would later read a low
sparse F1 as a sparsity failure when it is partly just the model's ceiling. It is worth being concrete about what token-overlap F1 rewards, because it changes how I read
the gap later. F1 scores the overlap between the model's answer tokens and the reference answer tokens —
precision against recall of the answer string. A model that finds the right passage but paraphrases it,
adds a hedging preamble, or returns a superset of the answer is penalized on precision even when its
retrieval was perfect; a model that finds only part of the evidence is penalized on recall. So the score
folds together two very different things — *did the attention reach the evidence* and *could the 1.5B turn
that evidence into the reference phrasing* — and at the oracle the first is guaranteed, which means the
oracle's QA F1 is almost purely a readout of the second, the model's own answer-formatting competence.
That is precisely why it caps well below `1.0` and why I must treat it as the realistic ceiling: when a
sparse rung comes in below it, part of the shortfall is lost evidence (the thing sparsity controls) and
part is just this same formatting cap (which sparsity cannot fix), and only the *movement* relative to the
oracle isolates the sparsity effect. And unlike NIAH these two
degrade gracefully: the evidence is distributed across the document rather than hidden in one position, so
a sparse pattern that covers *some* of the relevant spans can still compose a partial answer and keep part
of the F1. NIAH is a position-coverage cliff; the QA tasks are a distributed-evidence slope. I do not yet
know the exact three numbers — the feedback table will tell me — but I am confident of the *shape*:
NIAH near the top, the two QA scores low and graded.

There is one more thing the oracle establishes that is not a number but is the premise the whole ladder
rests on, and it is worth stating now because if it were false the budget would be hopeless. Attention in
a trained model is empirically *concentrated*: for a given query the softmax deposits almost all of its
mass on a small handful of keys and leaves a long tail of near-zero weights. I can bound the extreme case
to see how much room that leaves. If a query at position `t` spread its weight uniformly over all `t+1`
admissible keys, its attention entropy would be `log(t+1)`, which at `t ≈ 8000` is about `9` nats — a
maximally diffuse distribution over thousands of keys. Trained attention sits nowhere near that: the
effective number of attended keys, `exp(entropy)`, is typically a few tens, not thousands. That gap is the
entire reason a `0.25` budget is even plausible. Keeping a quarter of the causal pairs means keeping, for
a late query, about `0.25·(t+1) ≈ 2000` of its `~8000` admissible keys — vastly more than the few tens the
softmax actually leans on. So the budget is not tight relative to the *true support* of the attention; it
is generous. The difficulty is never the *size* of the support but its *location*: the mass-carrying keys
move with the query, and a method that spends its 2000-key budget in the wrong place keeps a quarter of the
matrix and still misses the handful of keys that mattered. The oracle, keeping everything, has the support
for free; every sparse rung is a bet about where that support is, and the three numbers this rung fixes are
what those bets are scored against.

I should also pin down why the contract measures density against `N(N+1)/2` and not `N²`, because that
choice is what makes "density" an honest number rather than a lever. The upper triangle is not "kept at
weight zero" — it is *structurally forbidden* by causality, it never was a pair the model could spend
attention on. Dividing kept pairs by the full `N²` would let a causal method report roughly half its true
density and sneak a mask that is actually at budget under the ceiling by a factor of two. Normalizing by
the admissible count `N(N+1)/2` reports the fraction of *reachable* pairs kept, which is the quantity the
budget is trying to constrain, and it is why the oracle's honest reading is exactly `1.0` and not `0.5`.
Every sparse rung will have to report this same admissible-fraction, and I will trust the *measured* count
over any closed-form estimate whenever the two disagree at the diagonal or the sequence boundary.

That split is the diagnosis I carry into the ladder, and it is worth naming the precise tension the sparse
rungs inherit, because the oracle is what makes the tension legible. A sparse method gets a density budget
`ρ = 0.25`: it may keep a quarter of the causal pairs. The real question is *recall* — of the keys that
dense attention actually put weight on, how many does the kept set retain? NIAH is the position-coverage
test: did the sparse mask happen to include the one block the needle sits in? The QA tasks are the
distributed-evidence test: did the mask cover enough of the document's relevant spans to answer? A static
pattern — one whose kept set depends only on `N` and not on the query — will pass or fail NIAH largely by
luck of where the needle lands relative to its fixed window and anchors, and will lose graded F1 on the QA
tasks in proportion to how much relevant context it drops. A content-adaptive pattern should do better on
NIAH precisely because it can *choose* the needle's block from the query, and better on QA because it can
route to the spans the query is actually asking about. The reason the oracle keeps *everything* is that
with full density recall is trivially total — every high-weight key is, by construction, kept — and that
is exactly the property a sparse method has to approximate cheaply: preserve the high-recall retrieval
tokens the softmax was going to weight, while it is free to drop the long tail of near-zero-weight keys
the softmax was going to ignore anyway.

So the oracle's three numbers set the bar, and the gap between them and each sparse rung's numbers is
exactly the signal that tells me whether the missing quality is a coverage problem — fix it with anchors
and windows — or a routing problem — fix it by making the pattern depend on the query. Concretely, the
falsifiable expectation I am setting up for the very first sparse rung: under the same density budget, a
static content-blind pattern should claw back a usable share of the oracle's QA F1, because distributed
evidence is partly reachable by any reasonable local-plus-anchor mask, but it should *collapse* on NIAH
from the oracle's near-perfect retrieval toward chance — because nothing about a fixed mask routes to a
needle whose position it did not anticipate. If that NIAH collapse happens while the QA numbers hold, the
diagnosis is confirmed and it points the ladder straight at query-adaptivity; if a static mask somehow
keeps NIAH near the ceiling, my whole reading of the static-versus-adaptive split is wrong and I would
have to rethink from the ground up. The distilled oracle module — the empty edit that hands `q, k, v` to
fused SDPA and reports density `1.0` — is in the answer.
