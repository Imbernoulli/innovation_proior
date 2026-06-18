I want a single architecture that simulates a finite-state semiautomaton end to end: read the whole
symbol stream `σ_{1:T}`, emit the whole state trajectory `q_{1:T}`, one prediction per position, trained
by per-token cross-entropy. The harness fixes everything else — the three environments, the online data
stream, AdamW, the early stop — so the only object I design is the `nn.Module` that `build_model` returns,
mapping `input_ids ∈ {0,…,alphabet_size−1}^{B×T}` to logits `[B, T, num_states]`. The question is what
inductive bias to put in that module first, and I want to start from the architecture whose *failure* will
teach me the most about the task, not the one I expect to win. So let me reason about what the task actually
demands and where a plausible starting architecture will crack.

The natural simulator of a semiautomaton is a recurrent net: its hidden state is literally a learned state
register, and `h_t = f(h_{t-1}, x_t)` is exactly the transition shape `q_t = δ(q_{t-1}, σ_t)`. If I only
cared about being *right*, I would reach for an RNN immediately. But the whole point of this benchmark — and
the reason it is interesting — is the gap Liu et al. 2022 opened: a *non-recurrent* network does not get to
unroll `T` serial steps; its expressive power is capped by its depth, and depth is a structural choice I make
once. Their theorems say a depth-`O(log T)` attention stack can simulate *any* semiautomaton, a *constant*
depth suffices for *solvable* ones via Krohn–Rhodes factorization, but *non-solvable* ones — the smallest
generating `A_5` — provably need more than constant depth unless `TC^0 = NC^1`. The three environments are
chosen to straddle exactly that line. So the sharp scientific move is to begin at the *shallowest* credible
attention model — one layer — and watch precisely which environments it clears and which it cannot, because
the pattern of failure is the complexity boundary made empirical. A one-layer model is the cleanest probe of
"what does a constant-depth shortcut actually buy here."

That fixes the starting point: the scaffold default, a shallow GPT-2-style causal Transformer with a single
encoder layer, `d_model=128`, 4 heads. Let me justify each piece against the task rather than treating it as
a given, because I have to be sure the model can in principle express the easy environments before I read its
numbers on the hard one.

Begin with the embedding. The input is a sequence of symbol ids; I look each up in a learned token embedding
of width `d_model`. But self-attention is permutation-equivariant — `softmax(QKᵀ/√d_k)V` is built entirely
from dot products and weighted sums over the *set* of positions, with no term that knows which position is
which. For a semiautomaton that is fatal on its own: the state at step `t` is the *ordered* application of
`σ_1, σ_2, …, σ_t`, and the same multiset of symbols in a different order gives a different state (group
multiplication does not commute in general). So I must inject order. The scaffold uses a learned position
embedding `pos_emb` added to the token embedding — one learned vector per absolute index `0…seq_len−1`. With
sequence length fixed at 40 and no extrapolation required, a learned absolute table is a perfectly adequate
order signal; it is also the most flexible, since the model can learn whatever position-to-position relation
the task needs rather than being handed a sinusoidal prior. Adding rather than concatenating keeps the
downstream matrices at width `d_model`; a learned linear over the sum can already separate content from
position into different subspaces if it wants to, so concatenation buys nothing but width.

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

The scaffold uses `norm_first=True` (pre-norm) and a GELU MLP of width `4·d_model`. Pre-norm is the
modern, stable ordering — `x + Sublayer(LayerNorm(x))` keeps a clean identity path for the gradient and
trains shallow and deep stacks alike without warmup gymnastics, which matters because I will be sweeping
*depth* across the ladder and want the optimization to stay comparable as I do. The 4× MLP is the only
per-position nonlinear compute the layer has (attention mixes *across* positions but is otherwise a
weighted linear combination of values), so it needs enough width to carve the per-symbol features the head
will read; 4× is the usual capacity knee. The final `head` is a linear `d_model → num_states`, producing the
per-position logits the harness scores. So the forward pass is: embed token + position, one causal
self-attention + MLP block, project to state logits — `[B, T] → [B, T, num_states]`. That is a faithful
fill of the contract, and it is the constant-depth (depth-1) probe I wanted.

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
before the last write is irrelevant. That is a single attention pattern away: a head that learns to attend
from position `t` back to the most recent non-`noop` symbol and copy its write target is an exact,
constant-depth solution. One layer, one head, is more than enough. I expect `memory_unit` to hit essentially
1.0 — this is the constant-function semigroup, one of the Krohn–Rhodes primes, and depth-1 attention is
provably sufficient. If even this fails I have an optimization bug, not a capacity wall.

`grid_world` is the first genuine test. `Grid_9` is a position on a line, alphabet `{L, R}`, with reflecting
(clamping) walls; the state is the clamped prefix sum of `±1` steps starting from the middle. Without walls
this is pure parity-of-counts — a prefix sum — which Liu et al. 2022 Thm 3 show has an `O(1)`-depth
self-attention construction (uniform attention to the prefix computes the average step, hence the count).
The wrinkle is the *clamping*: once the walk hits a wall it sticks until it turns around, so the map from
"net displacement" to "state" is nonlinear (a saturation). A single attention layer can compute the prefix
sum cleanly; whether one layer plus one MLP can also fold in the clamping nonlinearity *and* keep it accurate
across all 9 states for every prefix length up to 40 is the open question. My honest expectation is that
depth-1 gets *most* of the way but not all of it — the clamping interacts across the whole prefix in a way a
single mixing step may render only approximately — so I expect `grid_world` to land high but visibly short of
1.0. That shortfall, if it appears, is the signal that one mixing step is not quite enough recurrence-depth
even for this nominally constant-depth environment, and it foreshadows the real wall.

`random_dfa` is the wall itself. A random `δ` on `|Q|=60` states with `|Σ|=8` almost surely generates a
transformation semigroup containing `S_5`, which is non-solvable. By Liu et al. 2022 Thm 4 there is no
constant-depth attention shortcut for it unless `TC^0 = NC^1`. A depth-1 model is the most constant-depth
thing there is, so I expect it to be *far* from solving `random_dfa` — somewhere in the low double digits of
per-token accuracy, the analog of the ~7.9% depth-1 row in their Figure 8 (here a bit higher because 40
steps and a fixed table give some short-range structure a shallow model can exploit, e.g. the first few
positions and frequently-revisited states). Crucially, because the aggregate score is the *geometric mean*
across the three environments, a near-zero `random_dfa` will drag the whole task score down regardless of how
clean `memory_unit` and `grid_world` are. So the headline number this baseline reports will be dominated by
its `random_dfa` failure, and that is exactly the quantity the ladder has to move.

So here is what I am committing to and what I expect to read back. The model is the literal scaffold default
— one causal Transformer encoder layer, learned token + absolute position embeddings, causal mask, GELU MLP,
linear state head, AdamW at `3e-4`. It is the cleanest constant-depth probe of the benchmark. I expect
`memory_unit ≈ 1.0` (a single copy-from-last-write head solves it exactly), `grid_world` high but possibly
short of 1.0 (prefix sum is easy, the wall-clamping nonlinearity may not fully close in one mixing step), and
`random_dfa` poor (no constant-depth shortcut exists for a non-solvable group, so a depth-1 model is
structurally incapable). The falsifiable claim is the *ordering*: solvable environments clear, the
non-solvable one does not, and the gap between them is the complexity boundary. If `random_dfa` is anywhere
near solved by one layer, the whole Liu-et-al. story — and the rest of this ladder — is wrong. If instead it
collapses while the other two are clean, the diagnosis for the next rung writes itself: a constant-depth
shortcut is provably unavailable on the hard environment, so the next move must *buy effective depth* — more
sequential computation per token — without simply paying for it in parameters and wall time. That is the
thread the rest of the climb pulls on. The full scaffold module is in the answer.
