OK, let me think this through from scratch. Everyone around me is in a parameter-count arms race — 175B, then 280B, then 540B — on the working belief that more parameters is the road to better models. And there's real evidence for it: scale up a dense Transformer language model and it not only gets lower loss, it starts doing tasks from a few in-context examples. So the reflex is "bigger." But I want to step back and ask what I'm actually paying for, because I think the field is optimizing the wrong cost.

There are two costs to a language model and they get conflated. One is training it: you pay it once. The other is *running* it: every single query in deployment streams the weights through the accelerator and does the arithmetic, and a deployed model answers maybe billions of queries over its life. So the lifetime cost is dominated, by orders of magnitude, by inference, not training. Yet every scaling result I know optimizes the *training* budget.

Look at where the field's wisdom sits. Kaplan and colleagues fit power laws for Transformer LMs — loss falls predictably with compute — and concluded that when you get more compute you should spend most of it making the model *bigger* and relatively little on more data. Then Hoffmann and colleagues — the Chinchilla analysis — redid this carefully, matching the learning-rate schedule to the number of tokens, and found a quite different balance. Fitting the loss over model size $N$ and token count $D$ at fixed compute, where to good approximation $C \approx 6ND$, the minimum sits where you grow $N$ and $D$ *together*, roughly $N_{\text{opt}}\propto C^{1/2}$ and $D_{\text{opt}}\propto C^{1/2}$. The punchline everyone took away: the fashionable giant models were over-parameterized and under-trained; a 70B model on ~1.4T tokens beats a 540B model. Chinchilla's recipe is "for this training budget, here is the $(N,D)$ that minimizes loss."

But notice what Chinchilla is minimizing: *training* compute. It finds the $(N,D)$ that gets the lowest loss for a fixed number of training FLOPs, and then it stops — at the training-optimal token count for that $N$. That's the right thing to minimize if your cost is the one-time training run. It is the wrong thing to minimize if your cost is serving the model forever.

Let me make the trade-off explicit, because the whole project turns on it. Suppose I fix a target quality $L^\*$ — a loss, equivalently a benchmark level — and ask: which model should I deploy to hit $L^\*$? Training-compute-optimality says pick the $(N,D)$ on the compute-optimal frontier that reaches $L^\*$, and that frontier balances $N$ against $D$. But the inference cost of a deployed dense model scales with $N$ — every query pays roughly $2N$ FLOPs per token, independent of how many tokens I trained on. $D$ is paid once, at training; $N$ is paid every query, forever. So if I'm optimizing the thing that actually dominates lifetime cost, I should be pushing $N$ *down* and compensating by pushing $D$ *up*, well past the point where that's optimal for the training budget. A bigger model might reach $L^\*$ with less *training* compute, but a smaller model trained on more tokens reaches the same $L^\*$ and is cheaper *every time it runs*.

Is there headroom to actually do that — to keep shrinking $N$ and paying with more $D$? That's an empirical question about whether a small model's loss is still falling when I push it past its Chinchilla-optimal token count. And here's the fact that decides it: take a 7B model and just keep training it. Its loss is still going *down* after 1T tokens — far beyond the few-hundred-billion tokens any training-compute-optimal rule would hand a model that size. The curve hasn't flattened. So the small model has not saturated; the extra data is still buying quality. That's the opening. Nobody chasing the *training* frontier has any reason to sit at a 7B model and feed it a trillion-plus tokens, because per training FLOP that's wasteful. But per *inference* FLOP it's exactly right: I'm trading cheap one-time training tokens for a permanently smaller, faster-to-serve model.

So the objective flips. I'm not going to ask "what's the biggest model I can train" or even "what's the compute-optimal model for my training budget." I'm going to ask: across a range of *inference* budgets — a model that fits on one GPU, a model for a bigger server — what is the best quality I can reach by taking a relatively small $N$ and training it on far more tokens than compute-optimality would assign? Build a family, 7B up to ~65B, each trained on ~1–1.4T tokens, deliberately over the training-optimal token count. And as a constraint I'll impose on myself: use only publicly available data, so the whole thing can be released openly — which no one has managed to do competitively yet.

Now, training a small model on a trillion-plus tokens is a *long* run. That changes what I need from the architecture. I'm not going to invent a new architecture; the dense decoder Transformer is the right substrate and I don't want to gamble a trillion-token run on something unproven. But I do want every refinement that (a) makes a deep Transformer train stably over a very long horizon, and (b) doesn't cost me parameters or FLOPs I'd rather spend elsewhere. Let me go through the architecture sub-layer by sub-layer and decide what to keep and what to change, justifying each change by what actually breaks without it.

Start with normalization. The original Transformer is post-norm: each sub-layer computes its function, adds the residual, *then* normalizes — $\text{LN}(x + \text{Sublayer}(x))$. The trouble with that for deep stacks is that the normalization sits *on* the residual path. The residual connection is supposed to be a clean identity highway that lets gradients flow from the top of the stack to the bottom undisturbed; that's the whole point of residuals. If I normalize after the add, every layer's LayerNorm is squeezing the signal on that highway, and the gradients to early layers get attenuated and the optimization gets touchy — you need careful warmup and you still get instability as depth grows. Move the normalization to the *input* of each sub-layer instead — $x + \text{Sublayer}(\text{LN}(x))$ — and the residual path is a bare sum of sub-layer outputs, an unobstructed identity from input to output. Gradients flow straight down it. Deep models become dramatically easier to optimize, and for a long run where stability is everything, that's not optional. So: pre-normalization, normalize the input of each sub-layer.

Now *which* normalizer. LayerNorm does two operations: it subtracts the mean (re-centering) and divides by the standard deviation (re-scaling), then applies a learned gain and bias. Do I need both halves? The re-scaling is clearly doing real work — it's what makes the layer invariant to the scale of its input, which gives a kind of implicit learning-rate adaptation and keeps activations in a sane range. But the re-centering? What does subtracting the mean buy me? In a residual network with learned linear projections everywhere, a constant shift in the mean of an activation vector is something the next linear layer's weights can already absorb — there's nothing the mean-subtraction does that the rest of the network can't compensate for. If that's right, re-centering is dead weight: it costs a mean computation and a bias parameter and contributes little. So drop it. Normalize by the root-mean-square alone:

$$\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j} x_j^2 + \epsilon}}\, g_i,$$

with a learned gain $g$ and no bias, no mean subtraction. This is scale-invariant but not shift-invariant — and I've just argued shift-invariance is the dispensable half. The payoff is concrete: no mean to compute, no bias to store, fewer reductions over the feature dimension, so it's cheaper per call, and over hundreds of billions of tokens that adds up. I'll compute the RMS in fp32 for numerical safety and cast back. So RMSNorm on the input of each sub-layer.

Next, the feed-forward sub-layer. The standard one is $W_2\,\text{ReLU}(W_1 x)$, two matrices, with the hidden width $4d$. Can I get more out of the FFN for the same budget? There's a family of gated alternatives. A gated linear unit computes a component-wise product of two linear projections of the input, one of them passed through a nonlinearity: $\big(\phi(W_1 x)\big) \odot \big(W_3 x\big)$, then projected out by $W_2$. The intuition is that the gate $W_3 x$ multiplicatively modulates the activated branch — a data-dependent, per-coordinate gate rather than a fixed pointwise nonlinearity — which is strictly more expressive than a plain activation. Which $\phi$? Sweeping the choices, Swish/SiLU — $\text{SiLU}(z) = z\,\sigma(z)$ — comes out best on perplexity at matched compute; smooth and non-monotonic, it beats ReLU and GELU gates in this slot. So the FFN becomes

$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x)\odot (W_3 x)\big).$$

But now there are *three* matrices, $W_1, W_2, W_3$, not two. If I keep the hidden width at $4d$ I've inflated the FFN's parameters and FLOPs by 50%, which is not a free improvement — it's just a bigger FFN, and I should compare at *equal* budget. To hold parameters and compute fixed against the two-matrix $4d$ baseline, shrink the hidden width to $\tfrac23$ of it: $4d \to \tfrac23\cdot 4d = \tfrac83 d$. Three matrices at width $\tfrac23\cdot 4d$ cost about the same as two matrices at width $4d$. So the gating buys expressiveness for free, paid for by a narrower hidden layer. (In code I'll round that $\tfrac23\cdot 4d$ up to a convenient multiple, like 256, for hardware efficiency.)

Now position. The original Transformer adds an absolute position embedding to the token embedding once, at the bottom — the model sees "token $x$ at absolute slot $m$." Two things bother me about that for what I want. First, it's absolute: the model has to *learn* that relative offsets are what matter, rather than having relative position built in. Second, it's injected once and has to propagate up through the stack. What I actually want is for the attention dot product between a query at position $m$ and a key at position $n$ to depend on $x_m$, $x_n$, and the *relative* offset $m-n$ — nothing else about absolute position. Can I find position-dependent maps $f_q(x_m,m)$ and $f_k(x_n,n)$ such that

$$\langle f_q(x_m,m),\, f_k(x_n,n)\rangle = g(x_m, x_n,\, m-n)?$$

Let me try the smallest case, $d=2$, and look for $f$ as multiplication by a position-dependent matrix. If I rotate a 2D vector by an angle proportional to its position — $f(x,m) = R(m\theta)\,x$ where $R$ is the 2×2 rotation by angle $m\theta$ — then the inner product is $(R(m\theta) q)^\top (R(n\theta) k) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n-m)\theta\big)\,k$, because rotations compose by adding angles and $R(\alpha)^\top = R(-\alpha)$. So the absolute angles $m\theta, n\theta$ cancel and only the *difference* $(n-m)\theta$ survives. That's exactly the relative-position property I asked for, and it falls straight out of the group property of rotations.

Lift to $d$ dimensions by pairing up coordinates and rotating each 2D pair by its own frequency: a block-diagonal rotation $R^d_{\Theta,m}$ with blocks $R(m\theta_i)$, $i = 1,\dots,d/2$. What frequencies? Borrow the geometric schedule from sinusoidal embeddings: $\theta_i = 10000^{-2(i-1)/d}$. The low-index pairs rotate fast (short angular wavelength — they encode fine, short-range position), the high-index pairs rotate slowly (long wavelength — coarse, long-range). The spread of frequencies gives the attention a natural *decay* with relative distance: as $|m-n|$ grows, the fast components dephase and the dot product's positional contribution averages down, so far-apart tokens couple less by default. And because this is a rotation applied directly to the query and key *inside* the attention, I apply it at every layer, to $q$ and $k$ (not to the values — values aren't compared by a dot product), instead of adding a vector once at the bottom. I never materialize the big block-diagonal matrix; rotating 2D pairs is just an elementwise complex multiply.

So the architecture is settled: pre-norm with RMSNorm on each sub-layer's input; SwiGLU FFN at hidden width $\tfrac23\cdot 4d$; RoPE on the queries and keys at every layer; otherwise a standard causal dense decoder. Three changes, each earning its place — stability, free FFN expressiveness, built-in relative position — and none of them gambling the architecture itself.

Optimization for the long run. AdamW — Adam with decoupled weight decay, so the decay acts as true $L_2$ regularization rather than getting tangled into the adaptive denominator. I'll set $\beta_1 = 0.9$ and, notably, $\beta_2 = 0.95$ rather than the usual $0.999$. Why pull $\beta_2$ down? $\beta_2$ controls how long a memory the second-moment estimate keeps; $0.999$ averages the squared gradients over ~1000 steps. With very large batches the gradient statistics shift over training, and a too-long second-moment memory reacts sluggishly and lets the effective step size mis-track, which shows up as instability and loss spikes over a long run. Shortening the memory to $0.95$ (~20 steps) keeps the variance estimate responsive and the training stable. Weight decay $0.1$, gradient clipping at global norm $1.0$ to kill the occasional exploding-gradient spike, 2000 warmup steps so the adaptive moments settle before the learning rate is full, and a cosine decay of the learning rate down to 10% of its peak. Batch size 4M tokens. Tokenize with byte-pair encoding (SentencePiece), splitting every number into individual digits and falling back to bytes for unknown characters so nothing is out-of-vocabulary.

There's a practical wall I'll hit if I'm naive about it: a trillion-token run on a dense multi-billion-parameter model is enormously expensive, and the two things that limit throughput are memory and the cost of the attention. Take attention first. The standard implementation forms the full $n\times n$ score matrix and stores it for the backward pass — that's $O(n^2)$ memory per head and a lot of wasted work, because in a causal language model half of those scores are masked out (a query can't attend to future keys). So I'll use a memory-efficient causal attention that never materializes the full score matrix and never even computes the upper-triangular masked scores, with a fused backward. That cuts both the memory and the runtime of the most expensive sub-layer.

Memory more broadly: the backward pass needs activations, and storing every activation across all layers blows the memory budget. Activation checkpointing is the trade — keep only a few activations and recompute the rest during backward. But recomputation costs FLOPs, so I want to be selective: save the activations that are *expensive* to recompute, namely the outputs of the big linear layers (the projections and FFN matmuls), and recompute the cheap pointwise stuff. To control exactly what's saved versus recomputed I write the transformer block's backward by hand rather than leaning on autograd, which would save everything. And because the model is sharded across many GPUs with tensor and sequence parallelism, each block incurs `all_reduce` communication to stitch the shards back together; I overlap that communication with computation so the network traffic hides behind the matmuls instead of stalling them.

Let me write it, grounded in how a real decoder layer is actually wired. RMSNorm first — normalize by RMS, scale by a learned gain, do the reduction in fp32:

```python
import math
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    # Pre-norm normalizer on each sub-layer's INPUT. Re-scaling only:
    # divide by RMS, no mean subtraction, no bias -> cheaper than LayerNorm,
    # and the dropped re-centering is the dispensable half.
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)   # RMS reduction in fp32
        return output * self.weight
```

RoPE next. Precompute, for every position, the per-pair rotation as a complex phase $e^{i m\theta_i}$ with $\theta_i = 10000^{-2(i-1)/d}$; then applying the rotation is viewing $q$ and $k$ as $d/2$ complex pairs and multiplying by the phase — that's the block-diagonal rotation done as an elementwise complex product:

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # theta_i = theta^(-2i/dim) for pair i; angle at position m is m*theta_i.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()          # [pos, dim/2] angles m*theta_i
    return torch.polar(torch.ones_like(freqs), freqs)   # unit-modulus phases e^{i m theta_i}


def apply_rotary_emb(xq, xk, freqs_cis):
    # rotate each 2D coordinate-pair of q,k by its position angle == relative
    # position lands in <q,k>; values are untouched.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Attention: project to $q,k,v$, rotate $q$ and $k$, cache keys and values for incremental decoding, scaled-dot-product with a causal mask, project out. (The model-parallel linear layers shard these projections across GPUs; here they're plain linears for clarity.)

```python
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)   # relative position into q,k

        # KV cache so each decode step reuses past keys/values.
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask                      # causal mask (upper-tri = -inf)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

The SwiGLU feed-forward — three matrices, hidden width $\tfrac23\cdot 4d$ rounded up to `multiple_of`, gate the SiLU branch by the linear branch:

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)                 # 4d -> (2/3)*4d: keep params fixed
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)     # activated branch
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)     # gate branch
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)     # project out

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))      # SwiGLU
```

The block is pure pre-norm residual — normalize, sub-layer, add — and the stack embeds tokens, runs the blocks, applies a final norm, and projects to logits with no learned position embedding anywhere (RoPE carries position):

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)  # norm INPUT
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(i, args) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def forward(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h).float()
```

The causal chain, end to end: a deployed language model pays its inference cost — which scales with parameter count $N$ — on every one of billions of queries, so lifetime cost is dominated by inference, not the one-time training; yet the established scaling recipes minimize *training* compute and stop at the training-optimal token count. Since inference cost is set by $N$ alone while training tokens $D$ are paid once, the right move for a fixed target quality is to push $N$ down and pay with more $D$ — viable precisely because a small model's loss is still falling well past 1T tokens, so it hasn't saturated. Training a small model on a trillion-plus tokens is a long, expensive run, which demands an architecture that stays stable and trains efficiently over that horizon: pre-normalization with RMSNorm on each sub-layer's input for clean gradient flow at low cost; a SwiGLU FFN narrowed to $\tfrac23\cdot 4d$ for free gating expressiveness at fixed budget; RoPE rotating queries and keys at every layer so relative position falls directly out of the attention dot product with natural distance decay; AdamW with $\beta_2$ pulled to $0.95$, gradient clipping, warmup and cosine decay for long-run stability; and memory-efficient causal attention plus selective activation checkpointing and overlapped tensor-parallel communication so the trillion-token run is affordable — yielding a family of small, openly-trainable models that are cheapest where it counts, at inference.
