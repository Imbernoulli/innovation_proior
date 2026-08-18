The thing that keeps bothering me when I look at attention maps from a trained Transformer is how much
mass goes nowhere useful. I give the model a long context, ask it to pull one fact out of the middle, and
when I plot the attention weights over the answer span, the model does put the *most* weight there — but
it's a minority of the total. The rest is smeared, thinly, across hundreds of irrelevant tokens. Token by
token each of those weights is tiny, but there are hundreds of them, and they sum to a real fraction of
the probability mass. The output `o_m = sum_n a_{m,n} v_n` is therefore a weighted average that is part
signal and part a low-level average of junk. I want to know whether that junk is *forced* by the
operator or whether it's something I can remove.

Let me pin down exactly why it's forced. The attention weights come from a softmax:
`a_{m,n} = exp(s_{m,n}) / sum_n' exp(s_{m,n'})`, with `s_{m,n} = q_m^T k_n / sqrt(d)`. Every term `exp(·)`
is strictly positive. So no matter how small I drive a logit `s_{m,n}`, the weight `a_{m,n}` is never
zero — it can approach zero only as `s_{m,n} -> -infinity`, and finite logits never get there. With a
context of length `N`, even if every irrelevant token gets a weight of order `1/N`, the *aggregate*
irrelevant mass is order `(N - k)/N` where `k` is the number of genuinely relevant tokens — for large `N`
and small `k`, that's most of the mass. The softmax is a probability distribution; it is constitutionally
incapable of assigning exactly zero, and it has no mechanism to take mass back *off* a position once
assigned. So the noise floor is not a training artifact I can anneal away; it's baked into the shape of
the operator. A single softmax cannot subtract.

That word — *subtract* — is the hint. If the operator can only ever add positive mass, then the only way
to get cancellation is to introduce a second quantity and take a difference. Where have I seen "cancel a
common floor by subtracting two correlated signals" work? A differential amplifier: it takes two inputs,
rejects whatever voltage they have in common (the common-mode noise the environment dumps onto both
wires), and amplifies only their difference. Noise-cancelling headphones: estimate the ambient noise with
a second microphone and subtract it, so the floor goes away and the music survives. The shared structure
is always the same — the noise is *common* to the two channels, the signal *differs* between them, so the
difference keeps the signal and kills the noise. Can I build attention the same way?

Here is the bet I want to make precise. Suppose I form *two* attention maps over the same context,
`A1 = softmax(s1)` and `A2 = softmax(s2)`, from two different query/key projections. Both are
strictly-positive distributions, both have the smeared noise floor over the irrelevant tokens — that floor
is the *common-mode* part. If the two projections are trained so that their *signal* lands in the same
place (the relevant tokens) but their *noise floors are correlated*, then the difference `A1 - lambda*A2`,
for a suitable scalar `lambda`, cancels the common floor and leaves the signal standing. The relevant
tokens, where `A1` is much larger than `lambda*A2`, survive; the irrelevant tokens, where `A1` and
`lambda*A2` are both small and roughly equal, cancel toward zero — and *can go negative*, which a single
softmax could never do, but which is exactly what "subtract the floor" requires.

So the candidate operator is
`DiffAttn = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V`,
where `Q = [Q1; Q2]`, `K = [K1; K2]` are the query and key each split into two halves, and `lambda` is a
learnable scalar. Before I invest in the parameter bookkeeping, I should actually subtract two softmaxes
on a concrete example and see what comes out, because the analogy could be hiding a sign or scale problem.

Take a context of length `N = 200`, two relevant positions `{0, 1}`, and a shared content-driven "floor"
`noise[n] ~ N(0, 1)` that both halves see. Let the first map spike hard on the relevant tokens
(`s1[n] = noise[n] + 6` on relevant, `noise[n]` elsewhere) and let the second map reproduce the floor on
the tail but stay near zero on the signal (`s2[n] = noise[n] - 20` on relevant, `noise[n]` elsewhere).
With `lambda = 0.8` I get, summing weights:

```
single softmax A1:  relevant mass = 0.9072   irrelevant mass = 0.0928
diff A1 - 0.8*A2:   relevant mass = 0.9072   irrelevant mass = -0.7072
```

The relevant mass survives intact — good — but the irrelevant mass didn't cancel toward zero, it
*overshot* to a large negative `-0.71`, and 198 weights went negative. That's not the clean cancellation
I pictured. The reason is concrete: `A1` here is already so concentrated that its floor over the tail is
tiny (`0.093` total), whereas `A2` spreads its full unit of mass across those same 198 tokens, so
`0.8 * A2` over the tail is `0.8` — an order of magnitude bigger than the floor I wanted to remove. With a
fixed scalar `lambda`, subtracting `lambda * A2` only cancels `A1`'s floor when the two floors are
*comparable in magnitude per token*. So the operator is not magic; the cancellation is conditional on the
two maps actually matching on the tail. I need to find the regime where that holds before I trust it.

Where would the floors match? Precisely when `A1` is *not* already concentrated — the noisy,
lost-in-the-middle regime that motivated this whole exercise. Let me redo the check with a weak, realistic
spike (`s1[n] = noise[n] + 2` on relevant) so a single softmax leaks most of its mass, and pick `lambda`
to match the tail-floor scale rather than guessing `0.8`:

```
A1 alone:        relevant mass = 0.2127   irrelevant mass = 0.7873   (79% of mass is leaked noise)
A1 tail = 0.7873,  A2 tail = 1.0000  ->  matched lambda = 0.7873
diff A1 - 0.787*A2:  relevant mass = 0.2127   irrelevant mass = 0.0000
```

Now it works the way I claimed: the irrelevant mass cancels to `0.0000` while the full relevant mass is
kept, and the noise's share of the total weight drops from `0.787` to essentially zero. So the operator
does cancel the floor — but only when `lambda` is tuned to the floor's scale, and it helps most exactly
when the single softmax is leaking most, which is the case I care about. Two things fell out of this that
I didn't have before: (i) the right `lambda` is content-dependent and lands near `0.79` here, so a sane
operating point for `lambda` is somewhere below 1 and not a free-for-all; (ii) the resulting weights
genuinely go negative and no longer sum to one, so this is a *signed* attention pattern — the operator can
now push an irrelevant value vector's contribution to zero, which a single positive softmax structurally
cannot. The analogy survives contact with arithmetic, with the caveat that `lambda` has to be calibrated.

Now the problems, in order. First, `lambda`. If I just make `lambda` a free scalar parameter the optimizer
can drive it anywhere, and at the start of training a badly-scaled `lambda` makes the two large positive
maps subtract into something with a wild magnitude and sign, and the gradients are a mess. I want `lambda`
to start in a sane place and be *re-parameterized* so the learnable part is well-conditioned. The trick I
like: don't learn `lambda` directly; learn it as a difference of two exponentials of dot products, plus a
constant init. Concretely
`lambda = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init`,
with `lambda_q1, lambda_k1, lambda_q2, lambda_k2` learnable vectors of dimension `head_dim` initialized
small (normal, std 0.1). My intuition is that at init the two exponentials nearly cancel so `lambda`
starts at `lambda_init`, but "nearly" is doing work there — `exp` is convex, so `E[exp(q.k)] > exp(E[q.k])`,
and I should check the residual isn't large. With `head_dim = 64` and 20,000 random draws of the four
vectors, I measure `exp(q1.k1) - exp(q2.k2)`:

```
mean = 0.0012,  std = 0.1129
```

So at init `lambda = lambda_init + 0.0012 ± 0.11` — the mean offset is negligible (the two convex terms
have the same distribution and their biases cancel), and the spread of `~0.11` per head is small relative
to `lambda_init ~ 0.2-0.8`. Good: `lambda` really does start at the constant I set, with a small
per-head jitter, not at some `exp`-inflated value. The `exp` keeps each term positive and gives
multiplicative, well-scaled gradients; expressing `lambda` as a *difference* of two such terms lets it
move both up and down around `lambda_init` smoothly. So the constant `lambda_init` sets the operating
point and the learnable vectors provide calibrated, signed adjustment.

What should `lambda_init` be, and should it depend on depth? Think about what `lambda` controls: how much
of the second map is subtracted, i.e. how aggressively the floor is cancelled. Early layers are doing
broad, diffuse mixing — I don't want to cancel hard there, or I starve the upper layers of context. Deep
layers are doing focused retrieval — there I want strong cancellation. So I want `lambda_init` to *grow*
with depth, starting modest and rising. A schedule that does this smoothly:
`lambda_init = 0.8 - 0.6 * exp(-0.3 * (l - 1))` for layer index `l = 1, 2, ...`. Let me tabulate it
rather than read off only the endpoints:

```
l =  1     2     3     5     8     12    24
     0.20  0.36  0.47  0.62  0.73  0.78  0.80
```

It starts at `0.8 - 0.6 = 0.20`, rises monotonically toward `0.8`, with most of the climb in the first
few layers (already `0.62` by layer 5) and then a long flat tail. That's the shape I argued for: gentle
cancellation early, strong cancellation deep. And the ceiling `0.8` sits right where my floor-matching
example wanted `lambda` to be (`~0.79` in the noisy regime) — so the schedule parks the deep layers at the
operating point where the cancellation came out clean and lets the four learnable vectors fine-tune around
it. It's a fixed schedule with no extra parameters.

Second problem: parameter and FLOP budget. I've doubled the queries and keys — `Q1, Q2, K1, K2` — so a
naive version uses twice the q/k projection size and computes two full attention maps, which would make
this not a fair comparison to a vanilla Transformer. I fix this by *halving the head dimension*. In a
standard `h`-head model each head has dimension `d = d_model / h`. In differential attention I keep the
same number of "logical" heads but give each one a head dimension of `d/2 = d_model / h / 2`, and use
`2h` query/key sub-heads of that smaller dimension (the two halves `Q1, Q2` and `K1, K2`), while the value
side uses `h` heads of dimension `2 * (d/2) = d` — i.e. the value head spans both halves. Total q/k width
is `2h * (d/2) = h*d = d_model`, exactly the vanilla q/k width; total v width is `h * d = d_model`,
exactly vanilla. So the projection matrices are the same size and the attention compute is matched. The
doubling is absorbed by the halving — differential attention costs what a vanilla Transformer costs.

Third problem, and it's subtle: scale and stability of the output across heads. After the subtraction the
per-head output statistics are different from a vanilla head — the signed weights and the cancellation
change the variance of `o_m` — and because `lambda` is learned per head, different heads end up with
different output scales, which makes the residual stream and the gradients inhomogeneous. The remedy is to
normalize each head's output independently before concatenating: apply a normalization (GroupNorm across
heads, equivalently a per-head RMSNorm over the `2*(d/2) = d`-dimensional head output) to bring every
head's output to a common scale. This is the same "normalize before you combine" discipline that makes
deep nets trainable, applied per head because the subtraction made the heads heterogeneous.

There's one more scale subtlety I have to get right or the magnitude is off. The subtraction
`A1 - lambda*A2` shrinks the overall gain of the operator relative to a single softmax. I can pin the
factor exactly in the worst case for gain loss — fully correlated maps, `A2 = A1` — where every weight
becomes `A1[n] - lambda*A1[n] = (1 - lambda)*A1[n]`. I checked this on the toy example with `lambda = 0.8`
and identical floors: every single weight, relevant and irrelevant alike, came out scaled by `0.2000`
(`0.00030 -> 0.00006`, `0.00013 -> 0.00003`, and the relevant mass `0.9072 -> 0.1814`, ratio `0.2000`).
So in the correlated regime the whole map is uniformly attenuated by `(1 - lambda)`. The realistic case is
in between — selective cancellation on the tail, near-full survival on the signal — but the gain still
scales with `(1 - lambda)`, so this is the factor I have to compensate. If I leave it uncorrected the
output magnitude depends on `lambda` and drifts as `lambda` learns, fighting the per-head normalization.
So after the per-head norm I rescale by the *fixed* constant
`(1 - lambda_init)` — fixed, not the learned `lambda`, so the gain compensation is a stable constant set
at init and the normalization handles the rest. This keeps the differential head's output gradient flow
aligned with what a vanilla head would have, which is what lets me reuse a vanilla Transformer's
hyperparameters and learning rate unchanged. So: subtract, per-head normalize, multiply by
`(1 - lambda_init)`.

Let me put the forward pass together carefully, because the shapes and the order matter. Project `x` to
`q, k, v` at full `d_model` width. Reshape `q` and `k` to `2h` sub-heads of dimension `d/2`, and `v` to
`h` heads of dimension `d`. Apply RoPE to `q` and `k` (position is orthogonal to all of this; I keep the
relative-position encoding the substrate uses). Scale `q` by `1/sqrt(d/2)`, form `att = q k^T`, add the
causal mask (`-inf` above the diagonal), softmax over the key axis. Now reshape the `2h` head maps into
`h` *pairs* — `(B, h, 2, T, T)` — so that head-pair `j` holds `A1 = att[:, j, 0]` and `A2 = att[:, j, 1]`.
Compute `lambda` from the four learnable vectors and `lambda_init`. Take the differential map
`att[:, j, 0] - lambda * att[:, j, 1]`, shape `(B, h, T, T)`. Multiply by `v` (the `h` heads of dimension
`d`) to get the per-head output `(B, h, T, d)`. Per-head normalize (RMSNorm over the `d` axis), rescale by
`(1 - lambda_init)`, concatenate heads back to `d_model`, and project out. Causal masking, RoPE, and the
output projection are exactly the vanilla ones; only the score-formation step changed — from one positive
softmax to a difference of two.

One worry remains that the numbers above don't settle: is this just a free parameter the optimizer will
zero out? The cancellation I verified assumed the second map had *learned* to reproduce the first's tail
floor while staying off the signal. There's nothing automatic about that — it's a configuration training
has to find. What makes me think it will: the two query/key halves see the *same* content `x`, so their
floor patterns over the irrelevant tail are structurally correlated to begin with — the common-mode part
is handed to the operator for free, and `A1 - lambda*A2` drives it toward zero whenever the halves track
each other there. On the relevant tokens the model has an incentive to make the halves *disagree* (first
spikes, second doesn't), so the difference stays large — and my noisy-regime check showed that when this
configuration is reached, the irrelevant mass cancels to `0.0000` while the relevant mass is untouched.
The trivial solution `lambda = 0` (back to a single softmax) is not where gradient descent should stop,
because that solution leaves all `0.79` of the leaked mass in place, and cancelling it produces a cleaner
value average that predicts the next token better — there is loss to be gained by moving off `lambda = 0`.
I can't prove training reaches the clean configuration from here, but the lever is direct (`lambda` plus
the two projections) and the gradient points the right way. And because the result is signed and can be
sparse, it can express "attend to these, actively ignore those" in a way a single softmax structurally
cannot.

The whole thing is a drop-in replacement for the score-formation step: same projections, same parameter
count, same FLOPs, same RoPE, same causal mask, same residual placement. The one change is that each
"head" is now two softmax sub-heads whose difference forms a signed attention pattern, with a depth-scaled
init, a per-head output norm, and a fixed gain compensation. Everything that made the single softmax leak
mass onto irrelevant context — the strictly-positive floor with no way to subtract — is exactly what the
difference of two softmaxes removes.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth):
    # depth: 0-based layer index. Paper schedule (1-based l): 0.8 - 0.6*exp(-0.3*(l-1)).
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class MultiheadDiffAttn(nn.Module):
    def __init__(self, d_model, n_heads, depth):
        super().__init__()
        self.n_heads = n_heads
        # Halve the head dim so the doubled q/k is parameter/FLOP matched to a vanilla head.
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        # Per-head normalization over the 2*head_dim head output.
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rope, attn_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, 2 * self.head_dim)

        q, k = rope(q, k)                       # relative position on the doubled q/k sub-heads
        q = q.transpose(1, 2)                   # (B, 2H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)                   # (B,  H, T, 2*head_dim)

        q = q * self.scaling
        att = torch.matmul(q, k.transpose(-1, -2))           # (B, 2H, T, T)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=att.dtype), 1)
        att = att + attn_mask
        att = F.softmax(att, dim=-1, dtype=torch.float32).type_as(att)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att.view(B, self.n_heads, 2, T, T)
        att = att[:, :, 0] - lambda_full * att[:, :, 1]      # (B, H, T, T), signed

        o = torch.matmul(att, v)                             # (B, H, T, 2*head_dim)
        o = self.subln(o)                                    # per-head normalization
        o = o * (1.0 - self.lambda_init)                     # fixed gain compensation
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * 2 * self.head_dim)
        return self.out_proj(o)
```
