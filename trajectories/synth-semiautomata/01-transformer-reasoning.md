I want a single architecture that simulates a finite-state semiautomaton end to end: read the whole
symbol stream `σ_{1:T}`, emit the whole state trajectory `q_{1:T}`, one prediction per position, trained
by per-token cross-entropy. The harness fixes everything else — the three environments, the online data
stream, AdamW, the early stop — so the only object I design is the `nn.Module` that `build_model` returns,
mapping `input_ids ∈ {0,…,alphabet_size−1}^{B×T}` to logits `[B, T, num_states]`. The question is what
inductive bias to put in that module first, and I want to start from the architecture whose *failure* will
teach me the most about the task, not the one I expect to win. So let me reason about what the task actually
demands and where a plausible starting architecture will crack.

The background hands me three families to choose from, and I should weigh them against each other before I
commit, because the first rung sets the frame for the whole climb. A recurrent encoder–decoder maintains a
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

The convolutional option is worth pausing on rather than dismissing, because it is the tempting middle
ground: parallel like attention, cheaper than full pairwise mixing. But it buys me the wrong resource. A
stacked conv relates two positions only after enough layers that their receptive fields overlap — with
kernel `k` and dilation, spanning the full `T=40` prefix takes `O(log_k T)` layers, i.e. a handful even for
small `k`. That is a fixed, *constant-in-`T`* stack again, so on the complexity axis a conv model sits in the
same class as a shallow attention stack: it has a bounded number of composition stages, and Krohn–Rhodes says
that is precisely what a non-solvable group defeats. Worse, a conv's mixing is local-then-hierarchical, which
is a *weaker* router than attention's single-hop all-to-all, so it would confound "did the model fail because
it lacked depth" with "did it fail because it could not even route the prefix." Attention gives me the clean
`O(1)`-path-length router, which means any failure I read off is a *depth/composition* failure and nothing
else. That isolation is what makes attention the right probe, so the conv branch is out.

So the sharp scientific move is to begin at the *shallowest* credible attention model — one layer — and watch
precisely which environments it clears and which it cannot, because the pattern of failure is the complexity
boundary made empirical. A one-layer model is the cleanest probe of "what does a constant-depth shortcut
actually buy here." I could instead open with a deep six-layer stack that I expect to do better, but that
would be scientifically backwards: it fuses "does depth help" with "how much depth," reads a single opaque
number, and hands the next rung no diagnosis. The whole value of rung one is the *shape* of the failure —
which environments a single composition stage solves and which it abandons — and only the minimal-depth model
exposes that shape without confound. I am deliberately spending the first rung on a probe, not a contender.

That fixes the starting point: the scaffold default, a shallow GPT-2-style causal Transformer with a single
encoder layer, `d_model=128`, 4 heads. Let me justify each piece against the task rather than treating it as
a given, because I have to be sure the model can in principle express the easy environments before I read its
numbers on the hard one.

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
permutation-invariant model literally cannot separate these, so I *must* inject order. The scaffold uses a
learned position embedding `pos_emb` added to the token embedding — one learned vector per absolute index
`0…seq_len−1`. With sequence length fixed at 40 and no extrapolation required, a learned absolute table is a
perfectly adequate order signal; it is also the most flexible, since the model can learn whatever
position-to-position relation the task needs rather than being handed a sinusoidal prior. Adding rather than
concatenating keeps the downstream matrices at width `d_model`; a learned linear over the sum can already
separate content from position into different subspaces if it wants to, so concatenation buys nothing but
width.

Now the attention itself, and the one detail in the scaffold that is load-bearing for this task: the causal
mask. The model registers an upper-triangular boolean mask and passes `is_causal=True`, so position `t`
attends only to positions `≤ t`. This is not decoration — it is the correct structural prior for a
semiautomaton. The true state `q_t` is a function of `σ_{1:t}` and *nothing after*; a symbol arriving at step
`t+3` cannot influence `q_t`. If I let attention see the future, the model could in principle exploit
spurious future correlations on the training stream and I would be giving it information the real transition
function never has. Masking to the causal prefix bakes in "the state at `t` depends only on the prefix up to
`t`," which is exactly true and which strictly shrinks the hypothesis space to the correct one. So causal
attention here is the analog of the autoregressive constraint in language modeling, but its justification is
the semiautomaton's own definition, not a generation requirement.

The head count deserves its own arithmetic, since 4 heads is a choice and not a law. With `d_model=128`
and `n_heads=4` each head works in a `d_k = 128/4 = 32`-dimensional subspace, and the `1/√d_k = 1/√32 ≈
0.177` scaling in `softmax(QKᵀ/√d_k)` keeps the pre-softmax logits from saturating for unit-scale keys and
queries. Four heads is comfortably more than the minimum: `memory_unit` needs exactly one head (the
copy-from-last-write pattern), and even `grid_world` at its most demanding wants only a small number of
distinct aggregations — one head to accumulate the signed step count, perhaps another to track where the
walk turned — so four gives redundancy and lets the optimizer find *some* head that lands on the right
pattern rather than betting everything on a single one. A width-32 subspace is also plenty to embed a
scalar count or a recency score, so I am not starved for per-head capacity. There is no reason to widen or
narrow this for the probe; the head count is not what will decide whether `random_dfa` clears, since heads
add parallel routing within one stage and the wall is about the *number of stages*.

Let me actually pin down the recency head numerically, because "attend to the most recent write" is only a
solution if softmax can make it sharp enough. Suppose position `t`'s query encodes the scalar `t` and each
key at position `s` encodes `s`, so the pre-softmax score for attending from `t` to `s ≤ t` is proportional
to `s` (recent positions score higher). Then within the causal prefix the softmax weight on the most recent
non-`noop` position `s*` versus the next-most-recent `s' < s*` is `exp(α·s*)/(exp(α·s*)+exp(α·s')+…)`; with a
learnable temperature `α` and a gap `s*−s' ≥ 1`, even `α = 5` gives a weight ratio `exp(5) ≈ 148` to 1
between the last write and the one before it, so the head puts ~99% of its mass on the correct source. The
`noop` positions are excluded not by masking but by the value they carry — a `noop` value can be trained to
contribute the *identity* (carry the current state forward), so it does not matter if a little mass leaks
onto them. This confirms the copy head is not just topologically available but numerically sharp at a
temperature the model can reach, so `memory_unit → 1.0` is a firm prediction, not a hopeful one.

The scaffold uses `norm_first=True` (pre-norm) and a GELU MLP of width `4·d_model`. Pre-norm is the
modern, stable ordering — `x + Sublayer(LayerNorm(x))` keeps a clean identity path for the gradient and
trains shallow and deep stacks alike without warmup gymnastics, which matters because I will be sweeping
*depth* across the ladder and want the optimization to stay comparable as I do. The 4× MLP is the only
per-position nonlinear compute the layer has (attention mixes *across* positions but is otherwise a
weighted linear combination of values), so it needs enough width to carve the per-symbol features the head
will read; 4× is the usual capacity knee. The final `head` is a linear `d_model → num_states`, producing the
per-position logits the harness scores. So the forward pass is: embed token + position, one causal
self-attention + MLP block, project to state logits — `[B, T] → [B, T, num_states]`. Let me sanity-check the
shapes end to end, because a silent broadcast bug would masquerade as a capacity result. `input_ids` is
`[B, 40]`; `token_emb` maps it to `[B, 40, 128]`; `pos = arange(40)` broadcast to `[B, 40]` and `pos_emb`
gives `[B, 40, 128]`; the sum stays `[B, 40, 128]`; the encoder with a `[40, 40]` causal mask preserves
`[B, 40, 128]`; the head projects the last axis `128 → num_states`, so `[B, 40, num_states]`. That is the
contract exactly, and it is the constant-depth (depth-1) probe I wanted.

It is worth counting the parameters, because "constant depth, small model" is the property I am claiming and
I should know its size. The single encoder layer dominates: the attention `in_proj` is `3·d_model² = 3·128²
≈ 49k`, its output projection `d_model² ≈ 16k`, and the MLP two linears `2·(128·512) ≈ 131k`, so roughly
0.2M in the block, essentially independent of the environment. The embeddings and head scale only with the
tiny alphabets and state counts — `memory_unit` has 9 symbols and 8 states, `grid_world` 2 symbols and 9
states, `random_dfa` 8 symbols and 60 states — plus a `40·128 ≈ 5k` position table, none of which moves the
total off ~0.2M. So this is a small model whose capacity lives almost entirely in one mixing-plus-MLP stage;
whatever it cannot do, it cannot do for lack of *stages*, not for lack of width. The attention cost per
forward is `O(T²·d_model) = 40²·128 ≈ 0.2M` multiply-adds for the score matrix and the MLP is
`O(T·d_model·4d_model) = 40·128·512 ≈ 2.6M`, trivial at batch 64; with 8000–12000 online steps and a
~30-minute (1800 s) per-env budget, a depth-1 model finishes with enormous headroom, which also tells me the
budget is not the binding constraint on this rung and will comfortably absorb the deeper models to come.

The optimizer config is AdamW with `lr=3e-4, wd=1e-4, beta1=0.9, beta2=0.999`, the standard GPT-2-style
recipe. The harness already provides gradient clipping at 1.0 and a fresh online batch every step, so I do
not have to worry about overfitting — the test stream is sampled the same way as training, and the only
thing that can go wrong is *underfitting the function class*. That is the whole experiment: with no
overfitting confound, whatever this model fails to reach, it fails to reach because a depth-1 attention model
*cannot express it*, not because of a data or regularization artifact. That is precisely why the benchmark
trains online — to make the capacity question clean.

Now let me predict, per environment, where this lands, because the prediction *is* the diagnosis I will hand
to the next rung. Take `memory_unit` first. Its transition is `noop` keeps the state, `write(j)` jumps to
state `j`. The current state is entirely determined by the *most recent write* in the prefix — everything
before the last write is irrelevant. Let me trace it: on `[write(3), noop, noop, write(5), noop]` the true
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

I can even reason about *where* on the sequence that grid_world error should concentrate, which sharpens the
falsifiable prediction. A prefix of `t` symmetric `±1` steps starting at the middle state 4 of a
nine-cell line has expected displacement 0 and standard deviation `√t`, so it first has an appreciable
chance of reaching a wall (distance 4 from the middle) once `√t ≳ 4`, i.e. around `t ≈ 16`. For short
prefixes `t < 16` the walk almost never touches a wall, the clamp never fires, and the state *is* the plain
prefix sum — which one mixing step computes exactly — so early positions should be near-perfect. It is the
later positions `t ≳ 16..40`, where wall contact becomes common and the "did we clamp, and when" history
starts to matter, that a single stage should mispredict. So my mechanistic expectation is not a uniform
error but an error that grows with position, concentrated in the back half of each sequence — and if the
number comes back short of 1.0, that positional profile is the signature confirming it is the clamping
history, not some generic underfit, that a depth-1 model cannot capture.

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
the aggregate score is the *geometric mean* across the three environments, a near-floor `random_dfa` will
drag the whole task score down regardless of how clean `memory_unit` and `grid_world` are — the geometric
mean of something like `{1, ~0.9, ~0.2}` is pinned near the smallest factor, roughly `(1·0.9·0.2)^{1/3} ≈
0.56`, in the mid-0.5s and far below the arithmetic mean near `0.7`. So the headline number this baseline reports will be dominated by its
`random_dfa` failure, and that is exactly the quantity the ladder has to move.

So here is what I am committing to and what I expect to read back. The model is the literal scaffold default
— one causal Transformer encoder layer, learned token + absolute position embeddings, causal mask, GELU MLP,
linear state head, AdamW at `3e-4`. It is the cleanest constant-depth probe of the benchmark. I expect
`memory_unit ≈ 1.0` (a single copy-from-last-write head solves it exactly), `grid_world` high but possibly
short of 1.0 (prefix sum is easy, the wall-clamping nonlinearity may not fully close in one mixing step, as
the order-dependence trace shows), and `random_dfa` poor (no constant-depth shortcut exists for a
non-solvable group, so a depth-1 model is structurally incapable). The falsifiable claim is the *ordering*:
solvable environments clear, the non-solvable one does not, and the gap between them is the complexity
boundary. If `random_dfa` is anywhere near solved by one layer, the whole Liu-et-al. story — and the rest of
this ladder — is wrong. If instead it collapses while the other two are clean, the diagnosis for the next
rung writes itself: a constant-depth shortcut is provably unavailable on the hard environment, so the next
move must *buy effective depth* — more sequential computation per token — without simply paying for it in
parameters and wall time. That is the thread the rest of the climb pulls on. The full scaffold module is in
the answer.
