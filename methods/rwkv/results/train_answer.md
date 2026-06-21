What hurts when I try to train and deploy a large autoregressive language model is that the two architectures on the table each fail me on exactly the axis where the other succeeds. The Transformer trains beautifully — every position's representation is computed in parallel, so the accelerator gets the big dense matrix multiplies it wants — but its per-position output $\text{Attn}(Q,K,V)_t = \sum_{i=1}^{T} e^{q_t^\top k_i} v_i / \sum_{i=1}^{T} e^{q_t^\top k_i}$ compares token $t$ against every token $i$, so a layer costs $O(T^2 d)$ time and $O(T^2 + Td)$ memory, and at decode time I must keep a key/value cache that grows with the context: quadratic in sequence length, unbounded memory at inference. The recurrent net is the mirror image — an LSTM carries a fixed state, $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$, $h_t = o_t \odot \sigma(c_t)$, so inference is $O(d)$ memory and $O(Td)$ total, exactly the deployment profile I want — but the update reads $h_{t-1}$ before it can produce $h_t$, a strict chain of $T$ sequential steps per layer that the hardware cannot parallelize across time, and the repeated multiplication through the recurrence vanishes or explodes gradients (Hochreiter 1998), which has kept these models shallow and small. The linear-attention family does get a recurrence — Linear Transformers (Katharopoulos et al. 2020) replace $e^{q_t^\top k_i}$ by a factorizable kernel $\varphi(q_t)^\top\varphi(k_i)$ so the numerator $\sum_{i\le t}\varphi(q_t)^\top\varphi(k_i)v_i = \varphi(q_t)^\top\big(\sum_{i\le t}\varphi(k_i)v_i^\top\big)$ accumulates into a running state $S_t = S_{t-1} + \varphi(k_t)v_t^\top$ — but only by *approximating* the softmax kernel through the chosen feature map $\varphi$, and at the cost of a heavier $d\times d$ matrix state. The goal is therefore sharp: one network that trains with the time-parallelism of a Transformer and runs with the constant-memory linear-time recurrence of an RNN, the same weights readable two ways, and without the linear-attention tax of an approximation.

I propose RWKV — Receptance Weighted Key Value. The right primitive to start from is the Attention Free Transformer (Zhai et al. 2021), which attacks the dot product itself: it computes $\sum_{i=1}^{t} e^{w_{t,i}+k_i}\odot v_i / \sum_{i=1}^{t} e^{w_{t,i}+k_i}$, with no $q_t^\top k_i$ anywhere. What decides how much position $i$ contributes to $t$ is $e^{w_{t,i}+k_i}$ — a learned pairwise position bias $w_{t,i}$ plus the key $k_i$, exponentiated — and it multiplies $v_i$ *element-wise* rather than through a dot product. This is genuinely attention-like, a normalized, content-modulated, position-weighted average of values, but with the quadratic $QK^\top$ interaction gone. The obstacle is that AFT's $w_{t,i}$ is an arbitrary function of the absolute pair $(t,i)$ — a full $T\times T$ table of learned scalars — so it costs $O(T^2)$ to store, is tied to a maximum length, and gives no recurrence at all: because every term's weight changes arbitrarily when I move from $t$ to $t+1$, the sum for position $t$ shares nothing reusable with the sum for $t+1$. The Linear-Transformer accumulation worked precisely because stepping forward only *added* a term and left the old terms' contributions unchanged; here the old weights change. The only way a per-term weight can change with $t$ in a reusable way is if it changes by a *fixed multiplicative factor per step*, which forces $w_{t,i}$ to depend on $t$ and $i$ only through their gap $t-i$ and to be *linear* in that gap, so that exponentiating turns one more step of distance into multiplication by a constant. So I set

$$w_{t,i} = -(t-i)\,w,\qquad w \in (\mathbb{R}_{\ge 0})^d,$$

where the move that also kills the $O(T^2)$ storage is that $w$ is not a matrix but a single per-*channel* vector: each feature channel gets its own decay rate, and the bias for $(t,i)$ is just that channel's decay times how far back $i$ sits. I require $w \ge 0$ so that $e^{w_{t,i}} = e^{-(t-i)w} \le 1$ and the weights decay smoothly as I go backward in time. This is a different flavor of attention — instead of token-to-token scores, "where to attend" is governed by which channels decay slowly versus fast: a channel with $w \approx 0$ keeps the whole history, a channel with large $w$ is essentially local.

Substituting back, define the numerator and denominator accumulators $a_t = \sum_{i\le t} e^{-(t-i)w}e^{k_i}\odot v_i$ and $b_t = \sum_{i\le t} e^{-(t-i)w}e^{k_i}$. Going from $t-1$ to $t$, every term already in $a_{t-1}$ had distance $(t-1)-i$ and now has distance $t-i$, one larger, so it picks up exactly one more factor of $e^{-w}$; and the new token enters at distance zero with weight $e^{k_t}$. Hence

$$a_t = e^{-w}\odot a_{t-1} + e^{k_t}\odot v_t,\qquad b_t = e^{-w}\odot b_{t-1} + e^{k_t}.$$

There is the running state: two $d$-vectors, the numerator and denominator of an exponentially-weighted moving average of the values, updated in $O(d)$ per step — no $d\times d$ matrix, and no approximation, since this *is* exactly AFT's normalized weighted average, only with the weights constrained so the sum telescopes. Unrolling confirms it: $b_1 = e^{k_1}$, $b_2 = e^{-w}e^{k_1} + e^{k_2}$ — the $i{=}1$ term at distance 1 carries $e^{-w}$, the $i{=}2$ term at distance 0 carries 1 — matching $\sum_{i\le 2}e^{-(2-i)w}e^{k_i}$ exactly.

One subtlety earns the design its keep. If I emit $a_t/b_t$ directly, the *current* token sits at decay factor $e^0 = 1$, on the same footing as the most recent past token, and the model cannot distinguish "the token I am on now" from "the token one step back"; worse, the single decay $w$ would have to govern both the present and all of the past and could degenerate. So I pull the current token out of the decaying sum and give it its own per-channel "bonus" weight $u$, the time-first weight, replacing the decay term for the current position only:

$$\text{wkv}_t = \frac{\sum_{i<t} e^{-(t-1-i)w+k_i}\odot v_i + e^{u+k_t}\odot v_t}{\sum_{i<t} e^{-(t-1-i)w+k_i} + e^{u+k_t}}.$$

In recurrence form this stays clean: keep $a,b$ as the decayed past-only accumulators, and at each step read the past, add the current token at bonus $u$, emit, *then* fold the current token into the state with its ordinary weight so it decays normally for future steps,

$$\text{wkv}_t = \frac{a_{t-1} + e^{u+k_t}\odot v_t}{b_{t-1} + e^{u+k_t}},\qquad a_t = e^{-w}\odot a_{t-1} + e^{k_t}\odot v_t,\quad b_t = e^{-w}\odot b_{t-1} + e^{k_t},$$

with $a_0=b_0=0$. The present is privileged exactly once, when it is the present. Because $e^{k_t}$ is the exponential of an unbounded learned quantity, I carry the state through a running log-sum-exp rescaling: alongside $a,b$ I keep a running maximum exponent $p$ and store $a,b$ divided by $e^p$ so their magnitudes stay $O(1)$. To emit I take $q = \max(p_{t-1}, u+k_t)$ and form $(e^{p_{t-1}-q}a'_{t-1} + e^{u+k_t-q}v_t)/(e^{p_{t-1}-q}b'_{t-1} + e^{u+k_t-q})$, both exponents now $\le 0$ so nothing overflows; to advance I take $q' = \max(p_{t-1}-w, k_t)$ and rescale both accumulators by it, absorbing the per-step decay into the past's exponent. The inference state per layer is then the small set $\{x_t, a'_t, b'_t, p_t\}$, each a $d$-vector: $O(d)$ memory, constant in context length — the RNN profile, exactly.

Training reads the same network the other way. The expensive parts of the layer are the linear projections that produce the keys, values, receptance and gate — per-token matrix multiplies $O(BTd^2)$, completely independent across time, so they batch into one big GEMM like a Transformer's $W_Q, W_K, W_V$. The only sequential piece is the $\text{wkv}$ accumulation, a scan over $T$, but it is element-wise and tiny — order $d$ work per step, not $d^2$ — and parallelizes over batch and channels; I write it as a single custom CUDA kernel that, per step, emits the output from the running max and the $a,b$ accumulators and then updates them with the current token. One architecture, two views.

The rest of the block fills in what produces $k_t, v_t$ and the gate, with one cheap twist: rather than project $x_t$ alone, I project a per-channel linear interpolation between the current and previous token, $r_t = W_r(\mu_r \odot x_t + (1-\mu_r)\odot x_{t-1})$, and likewise for $k,v$, each with its own learned mixing vector $\mu$. The $\text{wkv}$ operator mixes information across distant positions through the decay, but each token's own key/value is otherwise computed from a single position; this width-2 local mix — implemented as a one-step sequence shift, $x_{t-1}$ via padding one zero at the front and dropping the last — lets every channel fold in as much of the immediately-preceding token as it wants, giving features that depend on a two-token context almost for free. I call this token shift, and use it for the time-mixing $r,k,v$ projections and again for the channel-mixing projections. Then the receptance gate: I named $r$ "receptance" because its job is to decide per channel how much of the accumulated context the position is willing to receive. The raw $\text{wkv}_t$ is gated by $\sigma(r_t) \in [0,1]$ — like an LSTM's output gate deciding how much cell state to expose — and a final projection mixes the gated result back into the residual stream, $o_t = W_o(\sigma(r_t)\odot \text{wkv}_t)$. The four letters R, W, K, V name the pieces: Receptance, the time-decay Weight, Key, Value.

The time-mixing block moves information across time but is, channel-wise, mostly a linear average gated by a sigmoid, so I add a position-wise nonlinear transform — the analog of a Transformer's feed-forward sub-layer. The channel-mixing block token-shifts the input into $k'$ and $r'$ projections, expands $k'$ to a wider hidden width, applies squared ReLU $\max(k',0)^2$ (So et al. 2021, Primer) — zeroing negatives like ReLU but growing quadratically on the positive side, a sharper, higher-contrast activation that improves these maps — projects back, and gates: $o'_t = \sigma(r'_t)\odot(W'_v\cdot \max(k'_t,0)^2)$. A block is these two sub-blocks, each a pre-normalized residual, $x \leftarrow x + \text{TimeMix}(\text{LN}(x))$, $x \leftarrow x + \text{ChannelMix}(\text{LN}(x))$; residuals and LayerNorm (Ba et al. 2016) are what let me stack this deep.

That depth deserves more than a hope, because the reason RNNs stayed shallow is that backpropagating through a long recurrence multiplies many Jacobians into a vanishing or exploding product. Here the recurrence dodges it structurally. Dropping the token shift, $\text{wkv}_T = \sum_t K^e_t \odot v_t / \sum_t K^e_t$ with $K^e_t = e^{W_k x_t + w_{T,t}}$ is literally a weighted average $\mathbb{E}(v_t)$ of the values. So $\partial(\text{wkv}_T)_i/\partial(W_v)_{i,j} = \mathbb{E}_i[(x_t)_j]$, an average of inputs bounded by $\max_t|(x_t)_j|$ — a bound that does not depend on $T$, so no explosion — and $\partial(\text{wkv}_T)_i/\partial(W_k)_{i,j} = \text{cov}_i((x_t)_j,(v_t)_i)$, a covariance under those same weights, also bounded and never collapsing to zero because the weighting always has at least two non-degenerate terms (the bonus $u$ term and the decayed $w$ terms). Because the output is a *normalized weighted average*, its Jacobians come out as bounded expectations and covariances rather than a runaway product of weight matrices; the decay $w$ controls how far back the average reaches, not a multiplicative gradient chain, so I get controllable memory without the vanishing/exploding pathology. A few initialization choices follow from wanting a clean deep start: I zero-initialize $W_r, W_k, W_v$ so the model begins near identity with only the structured parts carrying signal (an identity-mapping start, He et al. 2016); I store the decay as an unconstrained real parameter and set the effective decay as its exponential so $w \ge 0$ is automatic; and the embedding I initialize tiny, $U(\pm 10^{-4})$, with an extra LayerNorm right after it, because a standard normal embedding barely moves early in training — a tiny embedding plus a normalizer means a single small step changes the *direction* of the normalized embedding a lot, so the model escapes its initial state fast and post-LN training stays stable. The whole model: a small-init, extra-LN embedding, a stack of identical residual blocks each with the time-mixing sub-block (token shift → $r,k,v$ → $\text{wkv}$ recurrence with channel decay $w$ and bonus $u$ → $\sigma(r)$ gate → out projection) and the channel-mixing sub-block (token shift → $r',k'$ → squared-ReLU MLP → $\sigma(r')$ gate), then a final LayerNorm and a linear head to vocabulary logits under cross-entropy. Linear in time, constant memory at inference, parallel at training, no approximation.

```python
import torch, torch.nn as nn

def wkv(time_decay, time_first, k, v):
    # time_decay in log-space: effective per-step decay = exp(-exp(time_decay)); time_first = bonus u
    B, T, C = k.shape
    w, u = torch.exp(time_decay), time_first
    y = torch.empty_like(v)
    a = torch.zeros(B, C, device=k.device)            # numerator state
    b = torch.zeros(B, C, device=k.device)            # denominator state
    p = torch.full((B, C), -1e38, device=k.device)    # running max exponent
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        q  = torch.maximum(p, u + kt)                  # output: past + current(bonus u)
        e1, e2 = torch.exp(p - q), torch.exp(u + kt - q)
        y[:, t] = (e1 * a + e2 * vt) / (e1 * b + e2)
        q2 = torch.maximum(p - w, kt)                  # advance state (current decays for future)
        e1, e2 = torch.exp(p - w - q2), torch.exp(kt - q2)
        a, b, p = e1 * a + e2 * vt, e1 * b + e2, q2
    return y

def token_shift(x):
    return nn.functional.pad(x, (0, 0, 1, -1))          # x_{t-1}

class TimeMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_v = nn.Parameter(torch.ones(1, 1, d))
        self.time_decay = nn.Parameter(torch.zeros(d))  # W
        self.time_first = nn.Parameter(torch.zeros(d))  # u
        self.key   = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k, v, r = self.key(xk), self.value(xv), self.receptance(xr)
        rwkv = torch.sigmoid(r) * wkv(self.time_decay, self.time_first, k, v)
        return self.output(rwkv)

class ChannelMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))
        self.key   = nn.Linear(d, 4 * d, bias=False)
        self.value = nn.Linear(4 * d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))      # squared ReLU
        return torch.sigmoid(self.receptance(xr)) * self.value(k)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att, self.ffn = TimeMix(d), ChannelMix(d)
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.ln_emb = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.uniform_(self.emb.weight, -1e-4, 1e-4)
    def forward(self, idx):
        x = self.ln_emb(self.emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x))
```
