The field has been running a parameter-count arms race — 175B, then 280B, then 540B — on the working belief that more parameters is the road to better models, and there is real evidence behind it: scale up a dense decoder Transformer and it not only reaches lower loss, it starts solving new tasks from a handful of in-context examples. But that reflex optimizes the wrong cost. A language model has two costs and they get conflated. Training it is paid once. *Running* it is paid on every query, and a deployed model answers something like billions of queries over its life, so at scale the repeated inference cost can dominate the one-time training run. Yet every scaling result I know optimizes the training budget. Kaplan and colleagues fit power laws for Transformer LMs and concluded that extra compute should mostly buy a bigger model and comparatively little more data. Hoffmann and colleagues — the Chinchilla analysis — redid this carefully, matching the learning-rate schedule to the token count, and found a different balance: fitting the loss over model size $N$ and tokens $D$ at fixed compute $C \approx 6ND$, the minimum sits where you grow $N$ and $D$ together, $N_{\text{opt}}\propto C^{1/2}$, $D_{\text{opt}}\propto C^{1/2}$, so a 70B model on ~1.4T tokens beats a 540B one. The lesson taken away was "you were over-parameterized and under-trained." But notice what Chinchilla minimizes: training compute. It finds the $(N,D)$ reaching the lowest loss for a fixed training FLOP budget and then stops, at the training-optimal token count for that $N$. That is the right thing to minimize if your cost is the one-time run, and the wrong thing if your cost is serving the model forever.

So fix a target quality $L^\*$ and ask which model to deploy to reach it. The inference cost of a dense model scales with $N$ alone — roughly $2N$ FLOPs per generated token, independent of how many tokens it was trained on. $D$ is paid once at training; $N$ is paid on every query, forever. If I am optimizing the cost that actually dominates lifetime spend, I should push $N$ *down* and compensate by pushing $D$ *up*, well past the point that is optimal for the training budget: a bigger model might reach $L^\*$ with less training compute, but a smaller model trained on more tokens reaches the same $L^\*$ and is cheaper every single time it runs. The only question is whether there is headroom to keep shrinking $N$ and paying with more $D$ — whether a small model's loss is still falling when pushed past its Chinchilla-optimal token count. The fact that decides it: take a 7B model and just keep training it, and its loss is still going down after 1T tokens, far beyond the few-hundred-billion any training-compute-optimal rule would assign it. The curve has not flattened, so the small model has not saturated and the extra data is still buying quality. Nobody chasing the training frontier has a reason to feed a 7B model a trillion-plus tokens, because per training FLOP that is wasteful — but per inference FLOP it is exactly right.

I propose LLaMA: a family of dense decoder Transformers, 7B up to 65B parameters, each trained on far more tokens (1.0–1.4T) than compute-optimality would assign, deliberately over the training-optimal token count, and — as a self-imposed constraint so the result can be released openly — on publicly available data alone, which no competitive model had yet managed. Training a small model on a trillion-plus tokens is a long run, and that is what dictates the architecture. I do not want to gamble a trillion-token run on an unproven design, so the substrate stays a standard causal dense decoder; I add only the refinements that make a deep Transformer train stably over a very long horizon without costing parameters or FLOPs I would rather spend elsewhere. Three changes, each earning its place.

The first is normalization placement. The original Transformer is post-norm — each sub-layer computes its function, adds the residual, *then* normalizes, $\text{LN}(x + \text{Sublayer}(x))$ — which puts the normalization directly on the residual path. The residual connection is supposed to be a clean identity highway carrying gradients from the top of the stack to the bottom undisturbed; normalizing after the add squeezes the signal on that highway, attenuates gradients to early layers, and makes deep optimization touchy. Move the norm to the *input* of each sub-layer, $x + \text{Sublayer}(\text{LN}(x))$, and the residual path becomes a bare sum of sub-layer outputs — an unobstructed identity — so gradients flow straight down it and deep models become far easier to optimize. For a long run where stability is everything, pre-normalization is not optional. Then comes the question of *which* normalizer. LayerNorm does two things: it subtracts the mean (re-centering) and divides by the standard deviation (re-scaling). The re-scaling is doing real work — it makes the layer invariant to the scale of its input, which gives an implicit learning-rate adaptation and keeps activations in range. But the re-centering buys little: in a residual network full of learned linear projections, a constant shift in an activation's mean is something the next linear layer's weights can already absorb, so mean-subtraction is dead weight costing a reduction and a bias parameter. Drop it and normalize by the root-mean-square alone,

$$\bar{x}_i = \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_{j} x_j^2 + \epsilon}}\; g_i,$$

with a learned gain $g$ and no bias. This RMSNorm is scale-invariant but not shift-invariant — and shift-invariance is exactly the dispensable half. It is cheaper per call (no mean, no bias, fewer reductions), which compounds over hundreds of billions of tokens; I compute the RMS in fp32 for numerical safety and cast back.

The second change is the feed-forward sub-layer. The standard FFN is $W_2\,\text{ReLU}(W_1 x)$, two matrices at hidden width $4d$. A gated linear unit instead forms a component-wise product of two linear projections of the input, one passed through a nonlinearity, then projected out: the activated branch is multiplicatively modulated by a data-dependent, per-coordinate gate, which is strictly more expressive than a fixed pointwise nonlinearity. Sweeping the activation, Swish/SiLU — $\text{SiLU}(z) = z\,\sigma(z)$, smooth and non-monotonic — comes out best on perplexity at matched compute, beating ReLU and GELU gates in this slot, so the FFN becomes

$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x)\odot (W_3 x)\big).$$

This SwiGLU has three matrices, not two. Keeping the hidden width at $4d$ would inflate the FFN's parameters and FLOPs by 50%, which is not a free improvement — it is just a bigger FFN, and the honest comparison is at equal budget. So shrink the hidden width to $\tfrac23$ of $4d$, i.e. $4d \to \tfrac23\cdot 4d = \tfrac83 d$: three matrices at width $\tfrac23\cdot 4d$ cost about what two matrices at width $4d$ cost, so the gating buys expressiveness for free, paid for by a narrower hidden layer. In code that $\tfrac23\cdot 4d$ is rounded up to a convenient multiple (256) for hardware efficiency.

The third change is position. The original Transformer adds an absolute position embedding to the token embedding once at the bottom, so the model must learn that relative offsets are what matter and must propagate that signal up the stack. What I actually want is for the attention dot product between a query at position $m$ and a key at position $n$ to depend on $x_m$, $x_n$, and the signed relative offset $n-m$ — nothing else about absolute position. So I look for position-dependent maps $f_q(x_m,m)$, $f_k(x_n,n)$ with

$$\langle f_q(x_m,m),\, f_k(x_n,n)\rangle = g(x_m, x_n,\, n-m).$$

Try the smallest case, $d=2$, with $f$ a position-dependent matrix multiply: rotate a 2D vector by an angle proportional to its position, $f(x,m) = R(m\theta)\,x$. Then $(R(m\theta) q)^\top (R(n\theta) k) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n-m)\theta\big)\,k$, because rotations compose by adding angles and $R(\alpha)^\top = R(-\alpha)$. The absolute angles cancel and only the difference $(n-m)\theta$ survives — exactly the relative-position property, falling straight out of the group structure of rotations. Lift to $d$ dimensions by pairing coordinates and rotating each 2D pair by its own frequency: a block-diagonal rotation with blocks $R(m\theta_i)$, $i=1,\dots,d/2$, using the geometric schedule $\theta_i = 10000^{-2(i-1)/d}$. The low-index pairs rotate fast (fine, short-range position), the high-index pairs slowly (coarse, long-range); the spread of frequencies gives attention a distance-sensitive phase structure, so as $|n-m|$ grows the fast components dephase and the dot product attenuates far-apart interactions instead of treating every offset as an unrelated learned ID. Because this is a rotation applied directly to the query and key inside the attention, I apply RoPE at every layer to $q$ and $k$ (not the values — values are not compared by a dot product) rather than adding a vector once at the bottom, and I never materialize the block-diagonal matrix: rotating 2D pairs is just an elementwise complex multiply by the precomputed phase $e^{i m\theta_i}$.

Everything else is the standard causal dense decoder. For the long run the optimizer is AdamW — Adam with decoupled weight decay so the decay acts as true $L_2$ regularization rather than being tangled into the adaptive denominator — with $\beta_1 = 0.9$ and, notably, $\beta_2 = 0.95$ rather than the usual $0.999$. $\beta_2$ sets the memory of the second-moment estimate; $0.999$ averages squared gradients over ~1000 steps, and with very large batches the gradient statistics drift over training, so a too-long memory reacts sluggishly, lets the effective step size mis-track, and produces loss spikes. Shortening the memory to $0.95$ (~20 steps) keeps the variance estimate responsive and the run stable. Weight decay is $0.1$, gradient clipping is at global norm $1.0$ to kill exploding-gradient spikes, there are 2000 warmup steps so the moments settle before the learning rate is full, the learning rate cosine-decays to 10% of its peak, the batch is 4M tokens, and tokenization is byte-pair encoding via SentencePiece with numbers split into individual digits and a byte fallback so nothing is out-of-vocabulary. Making the trillion-token run affordable also needs care with the two throughput limiters, memory and attention cost. The standard attention forms the full $n\times n$ score matrix and stores it for the backward pass — $O(n^2)$ memory per head, and in a causal model half those scores are masked-out future positions anyway — so I use a memory-efficient causal attention that never materializes the full matrix and never computes the upper-triangular masked scores, with a fused backward. For memory more broadly I use selective activation checkpointing: save the activations that are expensive to recompute (the outputs of the big linear and FFN matmuls) and recompute the cheap pointwise work, controlling exactly what is saved by writing the transformer block's backward by hand instead of letting autograd save everything. And since the model is sharded with tensor and sequence parallelism, each block's `all_reduce` communication is overlapped with computation so the traffic hides behind the matmuls. The result is a family of small, openly-trainable models — 6.7B ($d{=}4096$, 32 heads, 32 layers), 13.0B, 32.5B, 65.2B — that are cheapest exactly where it counts, at inference.

```python
import math
from dataclasses import dataclass
from typing import Tuple
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
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : start_pos + seqlen].transpose(1, 2)
        values = self.cache_v[:bsz, : start_pos + seqlen].transpose(1, 2)
        xq = xq.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
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
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


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
