I want a single architecture that simulates a finite-state semiautomaton end to end: read the whole
symbol stream `σ_{1:T}`, emit the whole state trajectory `q_{1:T}`, one prediction per position, trained
by per-token cross-entropy. The harness fixes everything else — the three environments, the online data
stream, AdamW, the early stop — so the only object I design is the `nn.Module` that `build_model` returns,
mapping `input_ids ∈ {0,…,alphabet_size−1}^{B×T}` to logits `[B, T, num_states]`. The question is what
inductive bias to put in that module first, and I want to start from the architecture whose *failure* will
teach me the most about the task, not the one I expect to win. So let me reason about what the task actually
demands and where a plausible starting architecture will crack.

The background hands me three families to choose from, and I should weigh them against each other before I
commit, because this first choice sets the frame for everything after. A recurrent encoder–decoder maintains a
hidden state updated serially at each step; a stacked-convolution model (ByteNet-style) relates positions
`D` apart in `O(log_k D)` layers; self-attention routes every position to every other in one `softmax(QKᵀ/√d_k)V`
op with `O(1)` path length. The natural simulator of a semiautomaton is the recurrent one: its hidden state
is literally a learned state register, and `h_t = f(h_{t-1}, x_t)` is exactly the transition shape
`q_t = δ(q_{t-1}, σ_t)`. If I only cared about being *right*, I would reach for an RNN immediately. But the
whole point of this benchmark — and the reason it is interesting — is the gap Liu et al. 2022 opened: a
*non-recurrent* network does not get to unroll `T` serial steps; its expressive power is capped by its depth,
and depth is a structural choice I make once. Their theorems say a depth-`O(log T)` attention stack can
simulate *any* semiautomaton, a *constant* depth suffices for *solvable* ones via Krohn–Rhodes factorization,
but *non-solvable* ones — the smallest generating `A_5` — provably need more than constant depth unless
`TC^0 = NC^1`. The three environments are chosen to straddle exactly that line.

The convolutional middle ground I can dismiss on the same axis: a stacked conv spans the `T=40` prefix in
`O(log_k T)` layers, a fixed constant-in-`T` stack that sits in the same bounded-composition-stage class a
non-solvable group defeats — and its local-then-hierarchical mixing is a weaker router than attention's
single-hop all-to-all, so a failure would confound "too shallow" with "could not even route the prefix."
Attention's `O(1)`-path-length router isolates any failure as a pure depth/composition failure, which is
exactly what I want from a probe.

So the sharp move is to begin at the *shallowest* credible attention model — one layer — and watch precisely
which environments it clears and which it cannot, because the pattern of failure is the complexity boundary
made empirical. Opening instead with a deep six-layer stack would be backwards: it fuses "does depth help"
with "how much depth," reads a single opaque number, and yields no diagnosis. The value of this first probe is
the *shape* of the failure — which environments a single composition stage solves and which it abandons — and
only the minimal-depth model exposes that shape without confound.

That fixes the starting point: a shallow GPT-2-style causal Transformer with a single encoder layer,
`d_model=128`, 4 heads. Let me justify each piece against the task rather than treating it as a given, because
I have to be sure the model can in principle express the easy environments before I read its numbers on the
hard one.

Begin with the embedding. The input is a sequence of symbol ids; I look each up in a learned token embedding
of width `d_model`. But self-attention is permutation-equivariant — `softmax(QKᵀ/√d_k)V` is built entirely
from dot products and weighted sums over the *set* of positions, with no term that knows which position is
which. For a semiautomaton that is fatal on its own: the state at step `t` is the *ordered* application of
`σ_1, σ_2, …, σ_t`, and the same multiset of symbols in a different order gives a different state (group
multiplication does not commute in general). Let me make that concrete on the environment I am about to
score, because it is the crux of why order must be injected. In `grid_world` the walk starts in the middle
and takes `±1` steps with reflecting/clamping walls. Take the two streams `[R,R,R,R,R,L]` and `[R,R,R,R,L,R]`
— the *same* multiset of five `R` and one `L`. Starting from the middle state 4 on `{0,…,8}`: the first gives
`4→5→6→7→8→8→7` (the fifth `R` clamps at the wall, then `L` steps back to 7); the second gives
`4→5→6→7→8→7→8` (final state 8). Same symbols, different order, different final state — 7 versus 8. A
permutation-invariant model literally cannot separate these, so I *must* inject order: a learned position
embedding `pos_emb` added to the token embedding — one learned vector per absolute index `0…seq_len−1`. With sequence length fixed at 40 and no extrapolation required, a learned absolute table is a
perfectly adequate order signal; it is also the most flexible, since the model can learn whatever
position-to-position relation the task needs rather than being handed a sinusoidal prior. Adding rather than
concatenating keeps the downstream matrices at width `d_model`; a learned linear over the sum can already
separate content from position into different subspaces if it wants to, so concatenation buys nothing but
width.

Now the attention itself, and the one detail that is load-bearing for this task: the causal mask. I register
an upper-triangular boolean mask and pass `is_causal=True`, so position `t`
attends only to positions `≤ t`. This is not decoration — it is the correct structural prior for a
semiautomaton. The true state `q_t` is a function of `σ_{1:t}` and *nothing after*; a symbol arriving at step
`t+3` cannot influence `q_t`. If I let attention see the future, the model could in principle exploit
spurious future correlations on the training stream and I would be giving it information the real transition
function never has. Masking to the causal prefix bakes in "the state at `t` depends only on the prefix up to
`t`," which is exactly true and which strictly shrinks the hypothesis space to the correct one. So causal
attention here is the analog of the autoregressive constraint in language modeling, but its justification is
the semiautomaton's own definition, not a generation requirement.

Four heads is a comfortable default, not a law: at `d_model=128` each head gets a `d_k=32` subspace, plenty
to embed a scalar count or a recency score, and the redundancy lets the optimizer find *some* head that lands
on the right pattern rather than betting everything on one. It is not what decides whether `random_dfa`
clears — heads add parallel routing within one stage, and the wall is about the *number of stages*.

Pre-norm (`norm_first=True`) keeps a clean identity path for the gradient and trains shallow and deep stacks
alike without warmup gymnastics — which matters because I will be sweeping *depth* later and want the
optimization to stay comparable. The 4× GELU MLP is the layer's only per-position nonlinear compute (attention
mixes *across* positions but is otherwise a weighted linear combination of values), so it needs the width to
carve the per-symbol features the head reads. The final linear head maps `d_model → num_states`. So the
forward pass is: embed token + position, one causal self-attention + MLP block, project to state logits —
`[B, T] → [B, T, num_states]`, the constant-depth (depth-1) probe I wanted.

The model is small — the single encoder block is roughly 0.2M parameters (attention ≈ 65k, MLP ≈ 131k),
essentially independent of the environment, with the tiny alphabets, state counts, and a `40·128` position
table adding little. Its capacity lives almost entirely in one mixing-plus-MLP stage, so whatever it cannot
do, it cannot do for lack of *stages*, not width. And the compute is trivial at batch 64 over 8000–12000
online steps against a ~1800 s per-env budget, so wall time is not the binding constraint here and will
comfortably absorb the deeper models to come.

The optimizer config is AdamW with `lr=3e-4, wd=1e-4, beta1=0.9, beta2=0.999`, the standard GPT-2-style
recipe. The harness already provides gradient clipping at 1.0 and a fresh online batch every step, so I do
not have to worry about overfitting — the test stream is sampled the same way as training, and the only
thing that can go wrong is *underfitting the function class*. That is the whole experiment: with no
overfitting confound, whatever this model fails to reach, it fails to reach because a depth-1 attention model
*cannot express it*, not because of a data or regularization artifact. That is precisely why the benchmark
trains online — to make the capacity question clean.

Now let me predict, per environment, where this lands, because the prediction *is* the diagnosis I carry into
the next attempt. Take `memory_unit` first. Its transition is `noop` keeps the state, `write(j)` jumps to
state `j`. The current state is entirely determined by the *most recent write* in the prefix — everything
before the last write is irrelevant. On `[write(3), noop, noop, write(5), noop]` the true
states are `3,3,3,5,5`, and at each position `t` the answer is "the target of the last `write` at or before
`t`." That is a single attention pattern away: give every `write(j)` key a large "recency" component so that
position `t`'s query, restricted to its causal prefix, puts almost all its softmax mass on the *most recent*
non-`noop` position, and let the value at that position carry the write target `j`; the head copies it and
the linear head reads it out as state `j`. One layer, one head, is more than enough, and there is no ordered
composition to perform — only a lookup of the latest write. I expect `memory_unit` to hit essentially 1.0 —
this is the constant-function semigroup, one of the Krohn–Rhodes primes, and depth-1 attention is provably
sufficient. If even this fails I have an optimization bug, not a capacity wall.

`grid_world` is the first genuine test. `Grid_9` is a position on a line, alphabet `{L, R}`, with reflecting
(clamping) walls; the state is the clamped prefix sum of `±1` steps starting from the middle. Without walls
this is pure prefix-sum-of-counts — the net displacement — which Liu et al. 2022 show has an `O(1)`-depth
self-attention construction (uniform attention to the prefix computes the average step, hence the count,
hence the position). The wrinkle is the *clamping*, and my order-dependence trace above already shows why it
bites: `[R,R,R,R,R,L]` and `[R,R,R,R,L,R]` have identical net displacement `+4` yet land on states 7 and 8,
because the first walk *hit the wall and stuck* for a step while the second did not. So the state is *not* a
function of the net count alone once a wall is touched — it depends on the running maximum excursion of the
prefix, a second aggregation on top of the sum. A single attention layer can compute one such aggregation
cleanly (the sum), and its MLP can apply a fixed pointwise clamp to that sum; but a fixed clamp of the count
cannot recover the two different answers above, because they share the same count. Folding in "did the walk
ever hit a wall, and when did it last do so" needs the model to track the prefix's running extremum
*jointly* with its sum, which is more than one mixing step reliably delivers. So my honest expectation is that
depth-1 gets *most* of the way but not all of it — the clamping interacts across the whole prefix in a way a
single mixing step renders only approximately — so I expect `grid_world` to land high but visibly short of
1.0. That shortfall, if it appears, is the signal that one mixing step is not quite enough recurrence-depth
even for this nominally constant-depth environment, and it foreshadows the real wall.

I can even reason about *where* on the sequence that error should concentrate. A prefix of `t` symmetric `±1`
steps starting at the middle state 4 of a nine-cell line has expected displacement 0 and standard deviation
`√t`, so it first has an appreciable chance of reaching a wall (distance 4 from the middle) once `√t ≳ 4`,
i.e. around `t ≈ 16`. For short prefixes `t < 16` the walk almost never touches a wall, the clamp never fires,
and the state *is* the plain prefix sum — which one mixing step computes exactly — so early positions should
be near-perfect; it is the later positions, where wall contact becomes common and the "did we clamp, and when"
history matters, that a single stage should mispredict. So the expectation is not a uniform error but an error
that grows with position, concentrated in the back half of each sequence.

`random_dfa` is the wall itself. A random `δ` on `|Q|=60` states with `|Σ|=8` almost surely generates a
transformation semigroup containing `S_5`, which is non-solvable. There is an arithmetic coincidence worth
naming: `60 = |A_5|`, the order of the smallest non-solvable group. So the 60 states are exactly the size of
`A_5`'s regular representation, and a handful of the eight generators acting as permutations of those 60
elements can realize the full `A_5` action on itself — non-solvability is not incidental to this environment,
it is native to the state count. By Liu et al. 2022 there is no constant-depth attention shortcut for it
unless `TC^0 = NC^1`. A depth-1 model is the most constant-depth thing there is, so I expect it to be *far*
from solving `random_dfa`. Chance for 60 states is `1/60 ≈ 0.017`, so anything a shallow model reads off
above a few percent is short-range structure it can genuinely exploit — the first one or two positions, whose
state is a direct lookup of `δ(q_0, σ_1)` with no composition, plus frequently-revisited states — but the
long-prefix compositions that dominate positions 10–40 are exactly what one stage cannot compute. So I expect
low double-digit per-token accuracy, well above the `1/60` floor but nowhere near solved. Crucially, because
the aggregate is the *geometric mean* across the three environments, a near-floor `random_dfa` pins the whole
task score near that smallest factor regardless of how clean `memory_unit` and `grid_world` are — well below
the arithmetic mean. So the headline number will be dominated by the `random_dfa` failure, and that is exactly
the quantity to move next.

So I commit to the one-layer causal Transformer as a probe. The falsifiable claim is the *ordering*: the
solvable environments clear and the non-solvable one does not, and the gap between them is the complexity
boundary. If `random_dfa` is anywhere near solved by one layer, the whole Liu-et-al. story is wrong. If
instead it collapses while the other two are clean, the next move must *buy effective depth* — more sequential
composition per token — without simply paying for it in parameters and wall time.
