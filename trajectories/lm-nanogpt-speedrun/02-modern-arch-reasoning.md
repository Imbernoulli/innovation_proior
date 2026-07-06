Muon got the body of the network learning fast — 22.3 minutes, 6200 steps, val_loss 3.2785, about 216 ms
a step. Before I reach for the next change I want to read that record properly, because the two numbers in
it point at two different levers and I should know which one I'm pulling. Wallclock ≈ steps × time-per-step,
and here that's 6200 × 216.33 ms ≈ 1.34M ms. Muon moved the *first* factor: by making each update
condition-blind and better-directed it cut the step count relative to the AdamW baseline, and it did so at
almost no per-step cost (the orthogonalization was ~1% of the step). But it left the *second* factor,
216 ms/step, essentially untouched — that time is the forward and backward matmuls of a twelve-block GPT-2,
plus the embedding lookup and the head. So there are two distinct ways to shave the 1.34M ms from here: make
the model reach 3.28 in fewer steps (a better function to fit, so Muon converges faster), or make each step
cheaper (fewer wasted FLOPs). With the optimizer now condition-blind and nearly free, the thing pinning
*both* is the same object: the *architecture* Muon is optimizing. The body is still a faithful GPT-2 block —
GELU MLPs, the original normalization placement, a fused QKV projection, head dim 64, and the conservative
defaults Karpathy inherited. Each of those was the right choice in 2019; by now several have better-behaved
replacements that should let the same Muon+AdamW recipe reach 3.28 in fewer steps, and one or two of them
also cut the per-step time. Let me go through the block piece by piece and ask, for each, which factor it
moves and by what mechanism.

There's a design question lurking before I start: how much do I change at once? I could be surgical — swap
one thing, measure, keep or revert — which gives clean attribution but is slow, and I've got a pile of
individually-small, individually-well-motivated deltas that don't interact destructively. Or I could go
deeper/wider instead (more layers, more width), which is the other classic way to fit faster — but that
*raises* step_avg, and in a wallclock race a change that buys fewer steps by making each step fatter is a
wash at best and usually a loss. So I'll rule out the scale-up branch and instead apply the accumulated
"modern transformer" deltas together, re-tuning the schedule around them, and accept that attribution gets
muddier. The bet is that each delta is a Pareto improvement (fewer steps or cheaper steps, never worse), so
stacking them compounds.

Start with the MLP nonlinearity. GELU is smooth and fine, but it saturates softly and its gradient is
small over a wide region. There's a cheaper, sharper alternative that's been reported to train language
models slightly better: squared ReLU, x ↦ ReLU(x)². It's exactly zero for negative inputs (hard
sparsity, like ReLU) but grows quadratically for positive inputs, so the active units have a larger,
position-dependent gradient — the activation is more selective and the gradient signal through the MLP is
stronger where it fires. Cost-wise this is free: the MLP is dominated by its two matmuls (768→3072 and
3072→768), and the pointwise nonlinearity between them is a rounding error either way — a relu-and-square is
if anything cheaper than the erf/tanh approximation GELU carries. So this moves the step-*count* factor (a
better-conditioned function to fit) at zero step-*time* cost. Worth swapping GELU → ReLU².

The more fashionable nonlinearity upgrade would be a gated MLP — SwiGLU or GeGLU — and I should say why I'm
not reaching for it, because it's the tempting choice and it's wrong for *this* race. A gated MLP replaces
the single up-projection with two, a gate and a value branch, and multiplies them: instead of one 768→3072
matrix it needs two, so the MLP forward becomes 2·(768×3072) + (3072×768) MACs against ReLU²'s (768×3072) +
(3072×768). That's a 50% fatter MLP forward (and the convention of shrinking the hidden width to 8/3× to
hold parameters constant just trades the FLOPs back for a narrower bottleneck). The MLP is roughly two-thirds
of the body's matmul time, so a 50% fatter MLP is a real bump in step_avg — exactly the factor I'm trying to
push *down*. Gated MLPs earn their keep on final-quality benchmarks with a fixed compute budget; in a
wallclock-to-3.28 sprint the extra matmul is a tax I can't justify when ReLU² gives most of the
conditioning benefit for free. So the nonlinearity change is ReLU², not a gate.

Now the attention QK path. Rotary already replaced the learned absolute positions, good. But there's a
stability issue Muon makes more visible: the queries and keys can grow in norm during training, and since
the attention logits are qᵀk, a few large-norm query/key dimensions can blow the logits up and make the
softmax peaky and the gradients spiky. Let me put a scale on this. The attention logit for a head is qᵀk =
Σ_{i=1..d} qᵢkᵢ over the head dimension d, divided by √d in the softmax. Model qᵢ, kᵢ as roughly independent
with per-coordinate scale σ; then each product qᵢkᵢ has magnitude ~σ² and the sum of d of them, being a sum
of ~independent terms, has standard deviation ~√d·σ². After the 1/√d softmax scaling the logit sits at ~σ².
The trouble is the σ² dependence: if training lets the q,k coordinate scale drift from σ = 1 up to σ = 2 —
easy, since nothing pins it — the logit scale goes from ~1 to ~4, the softmax over the window sharpens toward
a one-hot, and the gradient it passes back spikes. It's a positive-feedback runaway: sharper softmax →
larger effective gradient on the winning key → q,k grow further. QK-norm cuts the loop at the source. RMS-norm
q and k along the head dimension so each has unit RMS (norm √d exactly), and then qᵀk = √d·√d·cos(θ) = d·cos(θ),
a bounded cosine similarity times a fixed d — after the 1/√d it's √d·cos(θ), capped at √d = √128 ≈ 11.3 in
magnitude regardless of how the raw q,k norms drift. The logit can no longer run away because its scale is
now a geometric constant, not a learned magnitude. The fix is
to normalize the per-head query and key vectors before the dot product — RMS-norm q and k along the head
dimension so each has unit RMS and hence norm √d. Then the attention score is essentially d·cos(angle), a
cosine-like similarity at a controlled scale — the logits can't run away no matter how the raw q,k norms
drift, and training is steadier. "QK norm." Apply rotary first, then norm q and k. And notice this interacts
with a change I'm about to make below: I'm going to *widen* the heads to d = 128, which doubles the number
of terms in that qᵀk sum versus d = 64, so the un-normalized logits would be even larger and QK-norm matters
*more* at the wider head, not less. The two changes support each other.

While I'm in the attention, reconsider the fused QKV. The baseline packs Q, K, V into one 3·n_embd-wide
`c_attn` matrix. That was a tiny matmul-efficiency convenience, but it forces Q, K, V to share a single
weight matrix that the optimizer treats as one object — and with Muon orthogonalizing matrices, a stacked
[3n, n] block is awkward. This is exactly why the Muon step from the previous rung carries that special
"split grouped QKV" branch: it has to detect the 3n×n shape and orthogonalize each n×n slice separately,
because orthogonalizing the stacked matrix as one object would mix the Q, K, V row spaces, which is
meaningless. If instead I give Q, K, V their own `c_q`, `c_k`, `c_v` matrices, each is a clean n×n matrix
that Muon orthogonalizes directly, the three roles get independent updates, and the special-case branch
becomes unnecessary — the architecture now matches what the optimizer wanted all along. So split the fused
projection into three separate linear layers. (The matmul FLOPs are identical — three n×n matmuls total the
same as one 3n×n — so this is step-count-neutral on time and a small conditioning win.)

Heads. The baseline uses 12 heads of dim 64 at width 768. Head dim 64 is small; modern practice and some
of the rotary/attention-kernel work prefer wider heads. Going to head dim 128 means 6 heads at width 768 —
and I should check the arithmetic, since head count must divide the width cleanly: 768 / 128 = 6 exactly, and
768 / 6 = 128, so 6 heads of dim 128 tile the 768-wide residual with nothing left over. Wider heads give
each attention head a richer subspace to compute similarity in, and 128 is a friendly size for the attention
kernels (the SDPA path is tuned for head dims that are multiples of 64, and 128 keeps the per-head matmuls at
a shape the tensor cores like). So n_head 12 → 6 (head_dim 64 → 128).

Now normalization, and this is where I want to be careful. GPT-2 uses LayerNorm with a learned gain and
bias. But the gain/bias add parameters and a per-feature affine that, empirically, the network often
doesn't need once the block has skip connections — what's actually doing the work is the *normalization*
(rescaling to unit RMS), not the learned affine. RMSNorm without any learnable parameters — just
x / RMS(x) — is cheaper, has no extra parameters for the optimizer to chase, and works as well. There's a
second, quieter reason to want it parameter-free here: a learned per-feature gain is a 1-D vector, which
neither Muon (2-D only) nor a clean AdamW group really wants to babysit, and every such dangling vector is
one more thing to tune and one more axis along which the run can drift. Dropping the affine removes those
vectors entirely. The baseline already had a hand-rolled `rmsnorm`; I'll standardize on the parameter-free
`F.rms_norm(x, (x.size(-1),))` everywhere normalization appears (pre-attention, pre-MLP, on q/k, and the
final norm before the head).

Initialization of the projections. This one interacts with the skip connections and I want to reason it
out on paper rather than assert it. The block is x ← x + attn(norm(x)) and x ← x + mlp(norm(x)) — residual.
At initialization, what should each sublayer contribute? With standard random init, each of the 2·n_layer =
24 sublayers adds a roughly-independent random vector to the residual stream, so the stream's variance grows
additively with depth — after 24 additions it's ~24× the input variance unless something damps it. That's
exactly why the baseline carried a depth-dependent `1/√(2·n_layer)` = 1/√24 ≈ 0.204 factor on the attention
output: multiplying each sublayer's contribution by ~0.2 keeps the accumulated residual variance ~O(1)
instead of ~O(24). But there's a cleaner way to get residual variance under control: *zero-initialize the
output projection* of both the attention (`c_proj`) and the MLP (`c_proj`). Then at step zero every block's
attn and mlp output are exactly zero, the residual passes the input straight through, and the network starts
as a clean identity stack — residual variance is *exactly* the input variance at every depth, no growth at
all, because nothing is being added yet. The blocks have to *earn* their contribution from zero. This makes
the early dynamics far gentler (no random sublayer outputs fighting the residual stream) and, crucially, it
makes the `1/√(2·n_layer)` fudge redundant: the variance it was there to control is already controlled by
the zero init, so I can delete `attn_scale` entirely. One knob removed, gentler start, and the LR can be
pushed harder because the early steps aren't fighting random init.

Let me sanity-check the variance claim with actual numbers, since deleting a scale factor is the kind of
thing that quietly blows up if I'm wrong. Suppose the pre-attention RMSNorm hands each sublayer a unit-RMS
input, so var ≈ 1 going in. With standard random init, an attention or MLP sublayer outputs a vector whose
per-coordinate variance is some O(1) value v — say v ≈ 1 for the estimate. The residual stream is the running
sum x + sublayer₁ + sublayer₂ + …, and independent additions add variances, so after the 2·12 = 24 sublayers
the stream variance is ≈ 1 + 24v ≈ 25, i.e. its RMS has grown by 5× from input to output — the deep blocks
see a residual that's a factor of 5 hotter than the shallow ones, and the softmax/logit scales drift with
depth. The baseline's 1/√24 ≈ 0.204 factor multiplies each contribution's variance by 1/24, giving 1 + 24·v/24
≈ 2, RMS growth √2 — bounded, which is the whole point of the fudge. Zero-init does better than bounded: with
c_proj = 0 every sublayer outputs exactly 0 at step zero, so the sum is just x and the stream variance is
*exactly* 1 at every depth — no growth, no depth-dependent drift, and nothing for a scale factor to correct.
So dropping `attn_scale` isn't a leap of faith; it's arithmetically redundant once the projections start at
zero, and the identity-stack start is strictly gentler than the √2-growth the fudge allowed.

Last, a pure-efficiency detail that's free loss-wise but real on wallclock: the GPT-2 vocab is 50257
tokens, an ugly number for the head matmul and the embedding. The head computes (tokens × 768) @ (768 ×
vocab) for on the order of tens of thousands of tokens per step, and it's the single biggest matmul in the
model — so its tiling matters. Tensor-core matmuls run in tiles that are multiples of 128 along each
dimension; 50257 = 128·392 + 81, so a 50257-wide output leaves a ragged final tile of 81 columns that the
kernel still has to schedule as a full 128-wide tile with 47 lanes masked off — wasted work that recurs
across the whole matmul. Padding the vocabulary up to the nearest multiple of 128 — 50304 = 128·393 —
makes the logical shape match the hardware tile exactly. The extra 47 rows are never targets (no token id
ever selects them, so they never contribute to the loss), so they cost nothing in accuracy; they just make
the matmul land cleanly and run faster. This is the one delta in the bundle that moves step_avg down rather
than step count.

None of these is a new idea on its own — they're the accumulated "modern transformer" deltas (rotary was
already in; now ReLU², QK-norm, parameter-free RMSNorm, separated QKV, head dim 128, zero-init residual
projections, padded vocab). And they sort cleanly into the two levers I identified from the record: ReLU²,
wider heads, split QKV, QK-norm, and zero-init residuals all target the *step-count* factor by giving Muon a
better-conditioned, more stable function to fit; the padded vocab targets the *step-time* factor. The bet is
that together they make the same Muon+AdamW optimization reach 3.28 in noticeably fewer steps *and* at a
lower per-step time. The risk is exactly the attribution risk I accepted up front — stacking this many
changes makes it hard to know which paid off, and some of them (zero-init especially) may need the learning
rate re-tuned or they'll stall — so I'll re-tune the schedule (the warmdown and total iteration count) along
with this rather than assume the old schedule transfers.

If the mechanism is right, the falsifiable signature is two-pronged and I can state it against this record's
numbers: the step count should drop below 6200 (a better function to fit), and the step_avg should drop
below 216 ms (the padded vocab plus the wider, kernel-friendlier heads), so both factors of the wallclock
product move the same way — a compounding cut rather than a single lever. I'd expect the step count toward
roughly 5100 and the step time noticeably under 216 ms; if instead the step count falls but step_avg *rises*,
that would say the wider heads cost more than the padded vocab saved, and I'd reconsider head dim 128.

Here is the modernized block and config — the deltas against the prior Muon script.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)   # split QKV
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK norm
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # squared ReLU; ~1-2% better than GELU
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))   # no attn_scale: zero-init projections handle it
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

@dataclass
class GPTConfig:
    vocab_size : int = 50304   # 50257 padded up to a multiple of 128
    n_layer : int = 12
    n_head : int = 6           # head dim 128
    n_embd : int = 768
```

The chain: the Muon record splits into steps × step_avg, and the architecture pins both, so I attack both
at once. GELU → ReLU² and head dim 64 → 128 give a function that fits faster; QK-norm bounds the qᵀk logits
(and matters *more* at the wider head, where the dot product has twice the terms) and parameter-free RMSNorm
removes unused affine params; splitting QKV gives Muon three clean n×n matrices and retires its special-case
branch; zero-initializing the residual output projections starts the net as an identity stack with residual
variance exactly O(1), which lets the `1/√(2·n_layer)` ≈ 0.204 attn-scale be dropped and the early dynamics
gentle enough to keep the learning rate high; and padding the vocab to 50304 = 128·393 lines the fattest
matmul up with the tensor-core tile, cutting per-step time at zero loss cost. Together these should cut the
step count from 6200 toward ~5100 *and* pull step_avg under 216 ms at the same 3.28 bar.
