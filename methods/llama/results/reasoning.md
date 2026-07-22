OK, let me think this through from scratch. Everyone around me is in a parameter-count arms race — 175B, then 280B, then 540B — on the working belief that more parameters is the road to better models. And there's real evidence for it: scale up a dense Transformer language model and it not only gets lower loss, it starts doing tasks from a few in-context examples. So the reflex is "bigger." But I want to step back and ask what I'm actually paying for, because I think the field is optimizing the wrong cost.

There are two costs to a language model and they get conflated. One is training it: you pay it once. The other is *running* it: every single query in deployment streams the weights through the accelerator and does the arithmetic, and a deployed model answers maybe billions of queries over its life. Once the model is served at scale, the repeated inference cost can dominate the one-time training run. Yet every scaling result I know optimizes the *training* budget.

Look at where the field's wisdom sits. Kaplan and colleagues fit power laws for Transformer LMs — loss falls predictably with compute — and concluded that when you get more compute you should spend most of it making the model *bigger* and relatively little on more data. Then Hoffmann and colleagues — the Chinchilla analysis — redid this carefully, matching the learning-rate schedule to the number of tokens, and found a quite different balance. Fitting the loss over model size $N$ and token count $D$ at fixed compute, where to good approximation $C \approx 6ND$, the minimum sits where you grow $N$ and $D$ *together*, roughly $N_{\text{opt}}\propto C^{1/2}$ and $D_{\text{opt}}\propto C^{1/2}$. The punchline everyone took away: the fashionable giant models were over-parameterized and under-trained; a 70B model on ~1.4T tokens beats a 540B model. Chinchilla's recipe is "for this training budget, here is the $(N,D)$ that minimizes loss."

But notice what Chinchilla is minimizing: *training* compute. It finds the $(N,D)$ that gets the lowest loss for a fixed number of training FLOPs, and then it stops — at the training-optimal token count for that $N$. That's the right thing to minimize if your cost is the one-time training run. It is the wrong thing to minimize if your cost is serving the model forever.

Suppose I fix a target quality $L^\*$ — a loss, equivalently a benchmark level — and ask: which model should I deploy to hit $L^\*$? Training-compute-optimality says pick the $(N,D)$ on the compute-optimal frontier that reaches $L^\*$, and that frontier balances $N$ against $D$. But the inference cost of a deployed dense model scales with $N$ — every query pays roughly $2N$ FLOPs per token, independent of how many tokens I trained on. $D$ is paid once, at training; $N$ is paid every query, forever. So if I'm optimizing the thing that actually dominates lifetime cost, I should be pushing $N$ *down* and compensating by pushing $D$ *up*, well past the point where that's optimal for the training budget. A bigger model might reach $L^\*$ with less *training* compute, but a smaller model trained on more tokens reaches the same $L^\*$ and is cheaper *every time it runs*.

Let me put numbers on this before I trust it, because the trade is not obviously a win — over-training a small model *costs* extra training FLOPs, and I need to know whether the inference savings actually pay that back. Take two candidates that I'll suppose reach the same $L^\*$: a 13B trained at roughly its training-optimal allocation, say $D\approx 260$B tokens, against a 7B I deliberately over-train to $D\approx 1.4$T. Training compute is $6ND$. The 13B costs $6\cdot 13\text{e}9\cdot 260\text{e}9 \approx 2.0\times 10^{22}$ FLOPs; the over-trained 7B costs $6\cdot 7\text{e}9\cdot 1.4\text{e}12 \approx 5.9\times 10^{22}$ — about $2.9\times$ *more* to train. So at the training counter the small model looks like a bad deal, exactly as compute-optimality says. Now inference: $2N$ FLOPs per generated token, so the 7B costs $14\text{e}9$ vs the 13B's $26\text{e}9$ per token — $0.54\times$, nearly half. The extra training I spent on the small model is $5.9\text{e}22 - 2.0\text{e}22 = 3.9\times 10^{22}$ FLOPs; each served token saves $2(13\text{e}9 - 7\text{e}9) = 1.2\times 10^{10}$ FLOPs. Break-even is $3.9\text{e}22 / 1.2\text{e}10 \approx 3.2\times 10^{12}$ tokens served. So once the model serves more than ~3 trillion tokens over its life, the smaller one is cheaper *overall* — and a model deployed at scale blows past 3T tokens of generation quickly. The arithmetic confirms the direction: pay the one-time training premium, win on every query forever.

That argument assumed I *can* keep shrinking $N$ and recover quality with more $D$ — that the smaller model still reaches $L^\*$ at all. That's the load-bearing empirical assumption, and it's the one piece here I can't settle with arithmetic: it depends on whether a small model's loss is still falling when pushed well past its Chinchilla-optimal token count, or whether it has flattened out. If a 7B's training loss is still descending after 1T tokens — far beyond the few-hundred-billion any training-compute-optimal rule would assign it — then the extra data is still buying quality and the headroom is real; if the curve has saturated, the whole trade collapses because the small model never reaches $L^\*$. I can't plot that curve from my desk, so I'll treat it as the central hypothesis to verify on the actual loss curve, and I'd want to *watch* loss-vs-tokens during the run and be ready to abandon the plan if it plateaus early. My expectation, from how slowly these curves bend, is that a 7B is nowhere near saturated at a few-hundred-billion tokens — but I'm flagging it as expectation, not established fact. If it holds, the opening is exactly what no one chasing the *training* frontier has reason to exploit: feeding a trillion-plus tokens to a 7B is wasteful per training FLOP but right per *inference* FLOP — trading cheap one-time training tokens for a permanently smaller, faster-to-serve model.

So the objective flips. I'm not going to ask "what's the biggest model I can train" or even "what's the compute-optimal model for my training budget." I'm going to ask: across a range of *inference* budgets — a model that fits on one GPU, a model for a bigger server — what is the best quality I can reach by taking a relatively small $N$ and training it on far more tokens than compute-optimality would assign? Build a family, 7B up to ~65B, each trained on ~1–1.4T tokens, deliberately over the training-optimal token count. And as a constraint I'll impose on myself: use only publicly available data, so the whole thing can be released openly — which no one has managed to do competitively yet.

Now, training a small model on a trillion-plus tokens is a *long* run. That changes what I need from the architecture. I'm not going to invent a new architecture; the dense decoder Transformer is the right substrate and I don't want to gamble a trillion-token run on something unproven. But I do want every refinement that (a) makes a deep Transformer train stably over a very long horizon, and (b) doesn't cost me parameters or FLOPs I'd rather spend elsewhere. The first pressure is normalization.

The original Transformer is post-norm: each sub-layer computes its function, adds the residual, *then* normalizes — $\text{LN}(x + \text{Sublayer}(x))$. The trouble with that for deep stacks is that the normalization sits *on* the residual path. The residual connection is supposed to be a clean identity highway that lets gradients flow from the top of the stack to the bottom undisturbed; that's the whole point of residuals. If I normalize after the add, every layer's LayerNorm is squeezing the signal on that highway, and the gradients to early layers get attenuated and the optimization gets touchy — you need careful warmup and you still get instability as depth grows. Move the normalization to the *input* of each sub-layer instead — $x + \text{Sublayer}(\text{LN}(x))$ — and the residual path is a bare sum of sub-layer outputs, an unobstructed identity from input to output. Gradients flow straight down it. Deep models become dramatically easier to optimize, and for a long run where stability is everything, that's not optional. So: pre-normalization, normalize the input of each sub-layer.

Now *which* normalizer. LayerNorm does two operations: it subtracts the mean (re-centering) and divides by the standard deviation (re-scaling), then applies a learned gain and bias. Do I need both halves? The re-scaling is clearly doing real work — it's what makes the layer invariant to the scale of its input, which gives a kind of implicit learning-rate adaptation and keeps activations in a sane range. But the re-centering? What does subtracting the mean buy me? In a residual network with learned linear projections everywhere, a constant shift in the mean of an activation vector is something the next linear layer's weights can already absorb — there's nothing the mean-subtraction does that the rest of the network can't compensate for. If that's right, re-centering is dead weight: it costs a mean computation and a bias parameter and contributes little. So drop it. Normalize by the root-mean-square alone:

$$\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j} x_j^2 + \epsilon}}\, g_i,$$

with a learned gain $g$ and no bias, no mean subtraction. This is scale-invariant but not shift-invariant — and I've just argued shift-invariance is the dispensable half. The payoff is concrete: no mean to compute, no bias to store, fewer reductions over the feature dimension, so it's cheaper per call, and over hundreds of billions of tokens that adds up. I'll compute the RMS in fp32 for numerical safety and cast back. So RMSNorm on the input of each sub-layer.

Next, the feed-forward sub-layer. The standard one is $W_2\,\text{ReLU}(W_1 x)$, two matrices, with the hidden width $4d$. Can I get more out of the FFN for the same budget? There's a family of gated alternatives. A gated linear unit computes a component-wise product of two linear projections of the input, one of them passed through a nonlinearity: $\big(\phi(W_1 x)\big) \odot \big(W_3 x\big)$, then projected out by $W_2$. The intuition is that the gate $W_3 x$ multiplicatively modulates the activated branch — a data-dependent, per-coordinate gate rather than a fixed pointwise nonlinearity — which is strictly more expressive than a plain activation. Which $\phi$? Sweeping the choices, Swish/SiLU — $\text{SiLU}(z) = z\,\sigma(z)$ — comes out best on perplexity at matched compute; smooth and non-monotonic, it beats ReLU and GELU gates in this slot. So the FFN becomes

$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x)\odot (W_3 x)\big).$$

But now there are *three* matrices, $W_1, W_2, W_3$, not two. Let me count parameters so I compare at *equal* budget rather than fooling myself. The baseline FFN at hidden width $h$ has $W_1\in\mathbb{R}^{d\times h}$ and $W_2\in\mathbb{R}^{h\times d}$, i.e. $2hd$ parameters; at $h=4d$ that's $8d^2$. The GLU has $W_1, W_3\in\mathbb{R}^{d\times h'}$ and $W_2\in\mathbb{R}^{h'\times d}$, i.e. $3h'd$. If I keep $h'=4d$ I get $12d^2$ — a 50% inflation, which is not a free improvement, it's just a bigger FFN. To match $3h'd = 8d^2 = 2\cdot 4d\cdot d$ I need $h' = \tfrac23\cdot 4d = \tfrac83 d$. Let me check the factor concretely at $d=4096$: baseline $8d^2 = 134{,}217{,}728$; the matched hidden width is $h'=\tfrac23\cdot 4\cdot 4096 = 10922$, and $3\cdot 10922\cdot 4096 = 134{,}209{,}536$ — within $0.006\%$ of the baseline, the small gap just from rounding $h'$ to an integer. So the $\tfrac23$ factor genuinely holds parameters fixed, and the gating buys expressiveness paid for by a narrower hidden layer rather than by a bigger budget. (In code I'll round that $\tfrac23\cdot 4d$ up to a convenient multiple, like 256, for hardware efficiency.)

Now position. The original Transformer adds an absolute position embedding to the token embedding once, at the bottom — the model sees "token $x$ at absolute slot $m$." Two things bother me about that for what I want. First, it's absolute: the model has to *learn* that relative offsets are what matter, rather than having relative position built in. Second, it's injected once and has to propagate up through the stack. What I actually want is for the attention dot product between a query at position $m$ and a key at position $n$ to depend on $x_m$, $x_n$, and the signed *relative* offset $n-m$ — nothing else about absolute position. Can I find position-dependent maps $f_q(x_m,m)$ and $f_k(x_n,n)$ such that

$$\langle f_q(x_m,m),\, f_k(x_n,n)\rangle = g(x_m, x_n,\, n-m)?$$

Let me try the smallest case, $d=2$, and look for $f$ as multiplication by a position-dependent matrix. If I rotate a 2D vector by an angle proportional to its position — $f(x,m) = R(m\theta)\,x$ where $R$ is the 2×2 rotation by angle $m\theta$ — then the inner product is $(R(m\theta) q)^\top (R(n\theta) k) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n-m)\theta\big)\,k$, because rotations compose by adding angles and $R(\alpha)^\top = R(-\alpha)$. If that algebra is right, the absolute angles $m\theta, n\theta$ should cancel and only the *difference* $(n-m)\theta$ should survive — which is the relative-position property I wanted. Let me not take the cancellation on faith; let me put numbers through it. Take $\theta = 0.5$, $q = (1, 2)$, $k = (-1, 0.5)$, and positions $m=3$, $n=7$. Rotating $q$ by $m\theta = 1.5$ rad and $k$ by $n\theta = 3.5$ rad and dotting gives $\langle R(1.5)q,\, R(3.5)k\rangle = -2.27324$. Computing the right-hand side instead, $q^\top R((n-m)\theta) k = q^\top R(2.0)\, k = -2.27324$ — same value. And the real test of "relative": shift *both* positions by 100, to $m=103, n=107$, keeping the offset $n-m=4$ fixed. The dot product is again $-2.27324$, unchanged. So the absolute position has genuinely dropped out and only the offset $n-m$ remains — the cancellation is real, not just a hopeful rearrangement, and it falls out of the group property of rotations.

Lift to $d$ dimensions by pairing up coordinates and rotating each 2D pair by its own frequency: a block-diagonal rotation $R^d_{\Theta,m}$ with blocks $R(m\theta_i)$, $i = 1,\dots,d/2$. What frequencies? Borrow the geometric schedule from sinusoidal embeddings: $\theta_i = 10000^{-2(i-1)/d}$. The low-index pairs rotate fast (short angular wavelength — they encode fine, short-range position), the high-index pairs rotate slowly (long wavelength — coarse, long-range). The spread of frequencies gives the attention a distance-sensitive phase structure: as $|n-m|$ grows, the fast components dephase, so the dot product has a built-in way to attenuate far-apart interactions instead of treating all offsets as unrelated learned IDs. And because this is a rotation applied directly to the query and key *inside* the attention, I apply it at every layer, to $q$ and $k$ (not to the values — values aren't compared by a dot product), instead of adding a vector once at the bottom. I never materialize the big block-diagonal matrix; rotating 2D pairs is just an elementwise complex multiply.

So the architecture is settled: pre-norm with RMSNorm on each sub-layer's input; SwiGLU FFN at hidden width $\tfrac23\cdot 4d$; RoPE on the queries and keys at every layer; otherwise a standard causal dense decoder. Three changes, each earning its place — stability, free FFN expressiveness, built-in relative position — and none of them gambling the architecture itself.

Optimization for the long run. AdamW — Adam with decoupled weight decay, so the decay acts as true $L_2$ regularization rather than getting tangled into the adaptive denominator. I'll set $\beta_1 = 0.9$ and, notably, $\beta_2 = 0.95$ rather than the usual $0.999$. Why pull $\beta_2$ down? $\beta_2$ controls how long a memory the second-moment estimate keeps; $0.999$ averages the squared gradients over ~1000 steps. With very large batches the gradient statistics shift over training, and a too-long second-moment memory reacts sluggishly and lets the effective step size mis-track, which shows up as instability and loss spikes over a long run. Shortening the memory to $0.95$ (~20 steps) keeps the variance estimate responsive and the training stable. Weight decay $0.1$, gradient clipping at global norm $1.0$ to kill the occasional exploding-gradient spike, 2000 warmup steps so the adaptive moments settle before the learning rate is full, and a cosine decay of the learning rate down to 10% of its peak. Batch size 4M tokens. Tokenize with byte-pair encoding (SentencePiece), splitting every number into individual digits and falling back to bytes for unknown characters so nothing is out-of-vocabulary.

There's a practical wall I'll hit if I'm naive about it: a trillion-token run on a dense multi-billion-parameter model is enormously expensive, and the two things that limit throughput are memory and the cost of the attention. Take attention first. The standard implementation forms the full $n\times n$ score matrix and stores it for the backward pass — that's $O(n^2)$ memory per head and a lot of wasted work, because in a causal language model half of those scores are masked out (a query can't attend to future keys). So I'll use a memory-efficient causal attention that never materializes the full score matrix and never even computes the upper-triangular masked scores, with a fused backward. That cuts both the memory and the runtime of the most expensive sub-layer.

Memory more broadly: the backward pass needs activations, and storing every activation across all layers blows the memory budget. Activation checkpointing is the trade — keep only a few activations and recompute the rest during backward. But recomputation costs FLOPs, so I want to be selective: save the activations that are *expensive* to recompute, namely the outputs of the big linear layers (the projections and FFN matmuls), and recompute the cheap pointwise stuff. To control exactly what's saved versus recomputed I write the transformer block's backward by hand rather than leaning on autograd, which would save everything. And because the model is sharded across many GPUs with tensor and sequence parallelism, each block incurs `all_reduce` communication to stitch the shards back together; I overlap that communication with computation so the network traffic hides behind the matmuls instead of stalling them.

Let me write it, grounded in how a real decoder layer is actually wired. RMSNorm first — normalize by RMS, scale by a learned gain, do the reduction in fp32:

```python
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


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
    # zero-based pair j uses theta_j = theta^(-2j/dim); angle at position m is m*theta_j.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()          # [pos, dim/2] angles m*theta_i
    return torch.polar(torch.ones_like(freqs), freqs)   # unit-modulus phases e^{i m theta_i}


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    # rotate each 2D coordinate-pair of q,k by its position angle == relative
    # position lands in <q,k>; values are untouched.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Attention: project to $q,k,v$ with tensor-parallel linears, rotate $q$ and $k$, cache keys and values for incremental decoding, scaled-dot-product with a causal mask, project out through a row-parallel linear.

```python
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads
        self.wq = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False,
            gather_output=False, init_method=lambda x: x
        )
        self.wk = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False,
            gather_output=False, init_method=lambda x: x
        )
        self.wv = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False,
            gather_output=False, init_method=lambda x: x
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, bias=False,
            input_is_parallel=True, init_method=lambda x: x
        )
        self.cache_k = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim
        ).cuda()
        self.cache_v = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim
        ).cuda()

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

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
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

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
        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    @torch.inference_mode()
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
        return self.output(h[:, -1, :]).float()
```
