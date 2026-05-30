Let me start from the thing that actually hurts when I try to train and deploy a large autoregressive language model. Two architectures are on the table, and each fails me on exactly the axis where the other succeeds. The Transformer (Vaswani et al. 2017) trains beautifully — every position's representation is computed in parallel, so a GPU gets the big dense matrix multiplies it wants — but its per-position output `Attn(Q,K,V)_t = Σ_{i=1}^{T} e^{q_tᵀk_i} v_i / Σ_{i=1}^{T} e^{q_tᵀk_i}` compares token `t` against every token `i`, so a layer costs `O(T²d)` time and `O(T²+Td)` memory, and at decode time I have to keep a key/value cache that grows with the context. Quadratic in sequence length, unbounded memory at inference. The recurrent net is the mirror image: an LSTM carries a fixed state, `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`, `h_t = o_t ⊙ σ(c_t)`, so inference is `O(d)` memory and `O(Td)` total — exactly the deployment profile I want — but the update reads `h_{t-1}` before it can produce `h_t`, a strict chain of `T` sequential steps per layer that the hardware cannot parallelize across time, and on top of that the repeated multiplication through the recurrence vanishes or explodes gradients (Hochreiter 1998), which has historically kept these models shallow and small.

So the goal is sharp: I want a model that *trains* with the time-parallelism of a Transformer and *runs* with the constant-memory linear-time recurrence of an RNN — the same network, readable two ways. And I want it without paying the usual tax of the linear-attention crowd: an approximation. Linformer projects keys/values to low rank, Performer replaces the softmax kernel with random features — both trade away some of attention's exactness for the scaling, and that compromise tends to bite harder the larger the model gets.

What can I build from? The promising line is Linear Transformers (Katharopoulos et al. 2020). The trick there is to notice that if I replace `e^{q_tᵀk_i}` by a *factorizable* kernel `φ(q_t)ᵀφ(k_i)`, then the numerator `Σ_{i≤t} φ(q_t)ᵀφ(k_i) v_i = φ(q_t)ᵀ (Σ_{i≤t} φ(k_i) v_iᵀ)` — and that inner sum `S_t = Σ_{i≤t} φ(k_i) v_iᵀ` is just a running state I can update incrementally, `S_t = S_{t-1} + φ(k_t) v_tᵀ`. That's the whole game: pull the query out of the sum, and the sum over the past becomes a recurrence. Linear time, RNN-form inference. But there's the catch I just named — `φ` is an *approximation* of the softmax kernel, and the state `S_t` is a `d×d` matrix (an outer product accumulated), which is a heavier state than I'd like, and the choice of `φ` is a modeling knob that affects quality. The structural idea — "make the sum over the past a running state" — is exactly right. I want to keep that and lose the approximation.

Now look at the Attention Free Transformer (Zhai et al. 2021), which goes at the dot product itself. AFT writes `Attn⁺(W,K,V)_t = Σ_{i=1}^{t} e^{w_{t,i}+k_i} ⊙ v_i / Σ_{i=1}^{t} e^{w_{t,i}+k_i}`. Stare at this for a second. There is *no* `q_tᵀk_i` here at all. The thing that decides how much position `i` contributes to position `t` is `e^{w_{t,i}+k_i}`: a learned pairwise position bias `w_{t,i}` plus the key `k_i`, exponentiated, and then it multiplies `v_i` *element-wise* (the `⊙`), not through a dot product. So the "attention weight" of `i` for `t` is a scalar-per-channel built from a position term and a content term (the key), and the values are combined as an element-wise weighted average. This is genuinely attention-like — it's still a normalized weighted average of values — but the quadratic `QKᵀ` interaction is gone. That feels like the right primitive: a content-modulated, position-weighted average, with no token-token dot product.

But AFT as written doesn't solve my problem yet, and I should see exactly why. The weight matrix `w_{t,i}` is an arbitrary function of the *absolute pair* `(t,i)` — it's a full `T×T` table of learned scalars. That's `O(T²)` parameters to store, it's tied to a maximum length, and crucially it gives me *no* recurrence: because `w_{t,i}` can be anything for each `(t,i)`, the sum `Σ_{i≤t} e^{w_{t,i}+k_i} v_i` for position `t` shares nothing reusable with the sum for position `t+1` — change `t` and every term's weight changes arbitrarily. So I can't accumulate a running state the way the Linear-Transformer trick wants. The arbitrariness of `w_{t,i}` is the obstacle.

What would I have to constrain about `w_{t,i}` so that the sum becomes a recurrence? The Linear-Transformer accumulation worked because moving from `t` to `t+1` only *added* a term — the old terms' contributions didn't change. Here the old terms' weights *do* change with `t`, through `w_{t,i}`. The only way the per-term weight can change with `t` in a way that's reusable is if it changes by a *fixed multiplicative factor per step*. That means `w_{t,i}` must depend on `t` and `i` only through their *gap* `t-i`, and must be *linear* in that gap, so that exponentiating turns "one more step of distance" into "multiply by a constant." So set

`w_{t,i} = -(t-i) w`,

where — and this is the move that also kills the `O(T²)` storage — `w` is not a matrix at all but a single per-*channel* vector, `w ∈ (ℝ_{≥0})^d`. Each feature channel gets its own decay rate; the bias for the pair `(t,i)` is just that channel's decay times how far back `i` is from `t`. I require `w ≥ 0` so that `e^{w_{t,i}} = e^{-(t-i)w} ≤ 1` and the weights *decay* as I go backward in time — older tokens count less, smoothly, at a per-channel rate. This is a different flavor of attention from the Transformer's: instead of token-to-token dot-product scores, the "where to attend" is governed by *which channels decay slowly versus fast* — channel-directed attention. A channel with `w≈0` keeps the whole history; a channel with large `w` is essentially local.

Substitute back. The weighted average at position `t` becomes `Σ_{i} e^{-(t-i)w + k_i} ⊙ v_i / Σ_{i} e^{-(t-i)w + k_i}`. Now check the recurrence I was hoping for. Define the numerator and denominator as accumulators

`a_t = Σ_{i≤t} e^{-(t-i)w} e^{k_i} ⊙ v_i`,   `b_t = Σ_{i≤t} e^{-(t-i)w} e^{k_i}`.

Going from `t-1` to `t`: every term already in `a_{t-1}` had distance `(t-1)-i`, and now its distance is `t-i`, one larger, so its weight picks up exactly one more factor of `e^{-w}`; and the new token `t` enters at distance zero, weight `e^{k_t}`. So

`a_t = e^{-w} ⊙ a_{t-1} + e^{k_t} ⊙ v_t`,   `b_t = e^{-w} ⊙ b_{t-1} + e^{k_t}`.

There it is — the running state. Two `d`-vectors, `a` and `b`, the numerator and denominator of an exponentially-weighted moving average of the values, updated in `O(d)` per step. No `d×d` matrix like Linear Transformers; no approximation of softmax — this *is* exactly the normalized weighted average AFT defined, I've just constrained the weights so the sum telescopes. The output at `t` is `a_t / b_t`, element-wise.

Let me make sure the unrolling is right before I trust it. `b_1 = e^{-w}·0 + e^{k_1} = e^{k_1}`, which is the `i=1` term at distance 0 from `t=1`. `b_2 = e^{-w} e^{k_1} + e^{k_2}` — the `i=1` term now at distance 1 carries `e^{-w}`, the `i=2` term at distance 0 carries 1. That matches `Σ_{i≤2} e^{-(2-i)w}e^{k_i}` exactly. Good, the newest token has no decay and each older one accumulates one extra `e^{-w}` per step back. The recurrence is faithful to the explicit sum.

Now a subtlety I have to handle, and it's where the design earns its keep. If I literally use `a_t/b_t` as the output for position `t`, the *current* token `t` is weighted by `e^{k_t}` with decay factor `e^0 = 1` — the same footing as the most recent past token. But "the token I'm sitting on right now" usually deserves special, distinct treatment from "the token one step back," and folding it into the same decaying geometric law makes the model unable to distinguish present from immediate past. Worse, the decay `w` is shared by the present token and all the past, so there's no separate knob for "how much does *now* matter" — `w` would have to do double duty and could degenerate. So I pull the current token out of the decaying sum and give it its own weight: a per-channel "bonus" vector `u` (call it the *time-first* weight) replacing the decay term for the current position only. The output becomes

`wkv_t = [ Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} ⊙ v_i + e^{u + k_t} ⊙ v_t ] / [ Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u + k_t} ]`,

where the sum now runs only over the *past* `i ≤ t-1`, indexed relative to `t-1`, and the current token gets `e^{u+k_t}` instead of a decayed weight. In recurrence form this is clean: keep `a, b` as the decayed *past-only* accumulators, and at each step form the output by adding the bonus-weighted current token on top of them before normalizing:

`wkv_t = (a_{t-1} + e^{u+k_t} ⊙ v_t) / (b_{t-1} + e^{u+k_t})`,
then advance the state with the *regular* (non-bonus) weight so the current token decays normally for *future* steps:
`a_t = e^{-w} ⊙ a_{t-1} + e^{k_t} ⊙ v_t`,   `b_t = e^{-w} ⊙ b_{t-1} + e^{k_t}`,   with `a_0 = b_0 = 0`.

So the ordering per step is: read the past state, add the current token with its special bonus `u`, emit the output, *then* fold the current token into the state with its ordinary decay weight `e^{k_t}` so that for the next position it has become "the past" and decays like everything else. The present is privileged exactly once, when it is the present.

I should sanity-check that this thing can't blow up numerically, because `e^{k_t}` is an exponential of an unbounded learned quantity — for large positive `k_t` it overflows in float. The fix is the standard log-sum-exp / running-max trick, done online. Keep, alongside `a` and `b`, a running maximum exponent `p_t` and store `a, b` *divided* by `e^{p_t}` so the stored magnitudes stay `O(1)`. Initialize `a'_1 = v_1, b'_1 = 1, p_1 = k_1` (factoring `e^{k_1}` out). Then at each step, to form the output I compare the past's exponent scale `p_{t-1}` against the current token's `u+k_t`, take `q = max(p_{t-1}, u+k_t)`, and compute

`wkv_t = (e^{p_{t-1}-q} a'_{t-1} + e^{u+k_t-q} v_t) / (e^{p_{t-1}-q} b'_{t-1} + e^{u+k_t-q})`,

where now both exponentials have argument `≤ 0`, so nothing overflows. To advance the state I take `q' = max(p_{t-1} - w, k_t)`, and
`a'_t = e^{p_{t-1}-w-q'} a'_{t-1} + e^{k_t-q'} v_t`, `b'_t = e^{p_{t-1}-w-q'} b'_{t-1} + e^{k_t-q'}`, `p_t = q'`.
Same two-term update, just rescaled by the new running max, with `w` (the per-step decay) absorbed into the past's exponent. In exact arithmetic `p_t` is pure bookkeeping and could be dropped, leaving `a, b` as the only state — but in float I carry it, so the inference state per layer is the small set `{x_t, a'_t, b'_t, p_t}`, each a `d`-vector: `O(d)` memory, constant in context length. That's the RNN profile I wanted, exactly.

Now, the other direction: training. The whole point was that this should *also* be parallel. The expensive parts of the layer are the linear projections that produce the keys, values, and the gate — those are per-token matrix multiplies `O(BTd²)`, completely independent across time, so they run as one big batched GEMM exactly like a Transformer's `W_Q,W_K,W_V`. The only sequential part left is the `wkv` accumulation itself, a scan over `T` — but it's element-wise, tiny (on the order of `d` work per step, not `d²`), and parallelizes trivially over the batch and the channel dimensions. So I write that scan as a single custom CUDA kernel: it iterates over time once, at each step computing the output from the running max and the `a,b` accumulators, then updating them with the current token — "output using the past, accumulate the current token, advance the state" — and everything *around* it is the parallel matmul stack. Training reads the model as a parallel network with one cheap scan; inference reads the same weights as a recurrence. One architecture, two views. (For pathologically long sequences I could even parallelize the scan itself with a parallel-prefix method, but the linear scan is already cheap.)

I have the core operator. Now what produces the `k_t`, `v_t`, and the gate at each position? In a Transformer these come from linear projections of the token. I'll do that too, but with one twist that's cheap and surprisingly useful. Instead of projecting `x_t` alone, I project a *per-channel linear interpolation* between the current token and the previous one:

`r_t = W_r (μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1})`, and the same form for `k_t = W_k(…)`, `v_t = W_v(…)`, each with its own learned mixing vector `μ`.

Why blend in `x_{t-1}`? Because the `wkv` operator mixes information *across* distant positions through the decay, but each token's own key/value is otherwise computed from a single position; giving every projection a small, learnable window over the current and immediately-previous token lets the model build features that depend on a two-token context — the kind of "what just came right before" signal that's needed for, say, completing a pattern from the token before. It's a width-2 local mix, almost free, and the per-channel `μ` lets each channel choose how much of the past token to fold in. Mechanically it's just a shift of the sequence by one and a blend — `nn.ZeroPad2d((0,0,1,-1))` to produce `x_{t-1}`, then the interpolation. I'll call this token shift, and I use it for the time-mixing projections `r,k,v` and, separately, for the channel-mixing projections below.

Two more pieces complete the time-mixing block. First, the receptance gate. I named the projection `r` "receptance" because its job is to decide, per channel, *how much of the accumulated context the position is willing to receive*. The raw `wkv_t` is a weighted average of past values; I gate it with `σ(r_t)` — a sigmoid in `[0,1]` per channel — so the model can suppress or admit the aggregated signal channel by channel, much like an LSTM's output gate decides how much of the cell state to expose. Then a final linear projection mixes the gated result back into the residual stream:

`o_t = W_o (σ(r_t) ⊙ wkv_t)`.

That's the entire time-mixing sub-block: token-shift the input, project to `r,k,v`, run the `wkv` recurrence over `k,v` with decay `w` and bonus `u`, gate by `σ(r)`, project out. The four letters R, W, K, V name its pieces — Receptance, the time-decay Weight, Key, Value.

Now the position-wise piece. The time-mixing block moves information across time but, channel-wise, it's mostly linear in the values (a weighted average gated by a sigmoid); the network needs a genuine per-position nonlinear transform too, the analog of a Transformer's feed-forward sub-layer. So I add a channel-mixing block: token-shift the input again into `k'` and `r'` projections, expand `k'` through a nonlinearity, project back, and gate. For the nonlinearity I use squared ReLU, `max(k', 0)²` (So et al. 2021, Primer) — it zeroes negatives like ReLU but grows quadratically on the positive side, giving a sharper, higher-contrast activation than plain ReLU that was found to improve these position-wise maps. Gate again by the receptance:

`o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`.

Here `W'_k` expands to a wider hidden width, the squared-ReLU acts, `W'_v` projects back, and `σ(r'_t)` gates the whole thing. It's a gated, squared-ReLU MLP applied independently per position — the across-channel mixing partner to the across-time mixing of the `wkv` block.

A block is then these two sub-blocks, each wrapped as a pre-normalized residual: `x ← x + TimeMix(LN(x))`, `x ← x + ChannelMix(LN(x))`. Residuals and LayerNorm (Ba et al. 2016) are what let me stack this deep without the gradient dying, which matters because I want to scale this to billions of parameters and the recurrent past has burned everyone on gradient stability before.

That gradient-stability worry deserves more than a hope, because the entire reason RNNs stayed shallow is that backpropagating through a long recurrence multiplies many Jacobians and the product vanishes or explodes. Does my `wkv` recurrence dodge that? Let me look at the gradient honestly. Simplify by dropping the token shift; then `wkv_T = Σ_t K^e_t ⊙ v_t / Σ_t K^e_t` with `K^e_t = e^{W_k x_t + w_{T,t}}` — that is literally a *weighted average* of the values, write it `E(v_t)` with the weights `K^e_t`. The gradient of the output with respect to the value projection is `∂(wkv_T)_i/∂(W_v)_{i,j} = E_i[(x_t)_j]`, an average of inputs, which is bounded by `max_t |(x_t)_j|` — and notice that bound *does not depend on `T` at all*. No growth with sequence length, so no explosion. The gradient with respect to the key projection works out to `∂(wkv_T)_i/∂(W_k)_{i,j} = cov_i((x_t)_j, (v_t)_i)`, a covariance under those same weights — also bounded, and it doesn't collapse to zero because the softmax-style weighting always has at least two non-degenerate terms (the bonus `u` term and the decayed `w` terms), so the "distribution" I'm averaging over never becomes a single point. The reason this is so much tamer than a vanilla RNN is structural: because the output is a *normalized weighted average*, its Jacobians come out as expectations and covariances of bounded quantities, not as a runaway product of weight matrices. The decay `w` controls *how far back* the average reaches — small `w` keeps the whole history, large `w` makes it local — but it controls the *averaging*, not a multiplicative gradient chain, so I get controllable memory without the exploding/vanishing pathology. That's why I can stack many of these layers.

A couple of initialization choices fall out of wanting this deep stack to start cleanly. I initialize most of the projection weights (`W_r, W_k, W_v`) to zero so the model begins from a near-identity, noise-free state and only the structured parts (the decays, the output/value projections) carry initial signal — an identity-mapping-style start (He et al. 2016) that keeps the information path clean through depth, no biases on the linear layers. The decay `w` itself I don't store directly as a non-negative number; I store an unconstrained real parameter and set the effective decay as its exponential, so non-negativity is automatic and the channel decays span a wide geometric range — initialized so early layers have many fast-decaying (local) channels and deeper layers have slow-decaying (long-memory) channels, with `w_i = -5 + 8·(i/(d-1))^{0.7 + 1.3·l/(L-1)}` across channel `i` and layer `l`. The bonus `u` I init with a small alternating zigzag pattern, `u_i = 0.5·(((i+1) mod 3) - 1) + log(0.3)`, to break symmetry across channels from the start. And the embedding: I noticed that with a standard normal init the embedding matrix barely moves in early training — the model gets stuck escaping its initial noisy embedding — so I initialize embeddings *tiny*, `U(±10⁻⁴)`, and put an extra LayerNorm right after the embedding. A tiny embedding plus a normalizer means a single small step changes the *direction* of the normalized embedding a lot, so the model escapes the initial state fast; it also makes post-LN training stable. Small but real convergence speedups.

So the whole model: an embedding (tiny init, extra LN), a stack of identical residual blocks each with a time-mixing sub-block (token shift → `r,k,v` → `wkv` recurrence with channel decay `w` and bonus `u` → `σ(r)` gate → out projection) and a channel-mixing sub-block (token shift → `r',k'` → squared-ReLU MLP → `σ(r')` gate), then a final LayerNorm and a linear head to vocabulary logits with cross-entropy. Linear in time, constant memory at inference, parallel at training, no approximation. Let me write it.

```python
import torch, torch.nn as nn

# ---- WKV: the linear-attention recurrence, decay w and bonus u per channel ----
# Training: run as a scan (one CUDA kernel); shown here as the reference recurrence.
def wkv(time_decay, time_first, k, v):
    # time_decay stored in log-space: effective per-step decay factor is exp(-exp(time_decay))
    # time_first is the bonus u (current-token weight)
    B, T, C = k.shape
    w = torch.exp(time_decay)            # w >= 0 automatically
    u = time_first
    y = torch.empty_like(v)
    a = torch.zeros(B, C, device=k.device)   # numerator state  Σ decayed e^{k} v
    b = torch.zeros(B, C, device=k.device)   # denominator state Σ decayed e^{k}
    p = torch.full((B, C), -1e38, device=k.device)  # running max exponent
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        # output for token t: past state + current token with bonus u (no decay)
        q  = torch.maximum(p, u + kt)
        e1 = torch.exp(p - q); e2 = torch.exp(u + kt - q)
        y[:, t] = (e1 * a + e2 * vt) / (e1 * b + e2)
        # advance state: fold current token in with ordinary weight (decays for the future)
        q2 = torch.maximum(p - w, kt)
        e1 = torch.exp(p - w - q2); e2 = torch.exp(kt - q2)
        a = e1 * a + e2 * vt
        b = e1 * b + e2
        p = q2
    return y

def token_shift(x):                      # produce x_{t-1}; pad one zero at front, drop last
    return nn.functional.pad(x, (0, 0, 1, -1))

class TimeMix(nn.Module):                 # across-time mixing: the R W K V block
    def __init__(self, d, layer_id, n_layers):
        super().__init__()
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))   # per-channel token-shift blends
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_v = nn.Parameter(torch.ones(1, 1, d))
        self.time_decay = nn.Parameter(torch.zeros(d))   # W (log-space decay)
        self.time_first = nn.Parameter(torch.zeros(d))   # u (bonus)
        self.key   = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)       # linear interp current/prev token
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k, v, r = self.key(xk), self.value(xv), self.receptance(xr)
        rwkv = torch.sigmoid(r) * wkv(self.time_decay, self.time_first, k, v)  # receptance gate
        return self.output(rwkv)

class ChannelMix(nn.Module):              # per-position transform: gated squared-ReLU MLP
    def __init__(self, d, layer_id, n_layers):
        super().__init__()
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))
        hidden = 4 * d
        self.key   = nn.Linear(d, hidden, bias=False)
        self.value = nn.Linear(hidden, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))        # squared ReLU
        return torch.sigmoid(self.receptance(xr)) * self.value(k)

class Block(nn.Module):
    def __init__(self, d, layer_id, n_layers):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = TimeMix(d, layer_id, n_layers)
        self.ffn = ChannelMix(d, layer_id, n_layers)
    def forward(self, x):
        x = x + self.att(self.ln1(x))     # pre-LN residual
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.ln_emb = nn.LayerNorm(d)                     # extra post-embedding LN (small init)
        self.blocks = nn.ModuleList(Block(d, i, n_layers) for i in range(n_layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.uniform_(self.emb.weight, -1e-4, 1e-4)    # tiny embedding init
    def forward(self, idx):
        x = self.ln_emb(self.emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x))
```

The causal chain, start to finish: the Transformer is quadratic and the RNN is sequential, so I want one network readable both ways. AFT already removes the `QKᵀ` dot product and reweights values by a position bias plus a key — but its bias is an arbitrary `T×T` matrix, which blocks any recurrence. Forcing that bias to be linear in the time gap and per-channel, `w_{t,i} = -(t-i)w`, makes the normalized weighted average telescope into a two-vector running state `a_t = e^{-w}a_{t-1}+e^{k_t}v_t`, `b_t = e^{-w}b_{t-1}+e^{k_t}` — exact, no approximation, `O(d)` memory. Pulling the current token out with a separate bonus `u` privileges the present without corrupting the decay law. The log-sum-exp running max keeps the exponentials safe. Because the expensive projections are per-token they batch into Transformer-style GEMMs, leaving only a cheap scan to run sequentially, so the same weights train in parallel and infer as a recurrence. The receptance gate `σ(r)` admits context per channel; the squared-ReLU channel-mix supplies the per-position nonlinearity; pre-LN residuals plus the bounded-by-construction gradients (the output is an average, so its Jacobians are bounded expectations and covariances, not a runaway product) let the whole thing stack deep and scale.
