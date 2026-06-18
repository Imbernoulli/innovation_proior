Let me start from what actually hurts. I want one sequence model I can drop under a foundation model for any modality — text, audio, DNA — and the only thing that really works at scale right now is attention. Attention works because every position reads directly from every other position; nothing is summarized away, the routing is dense and depends on the actual content. But that is exactly the bill I keep paying. Training is quadratic in length because I materialize all L² pairwise interactions. Decoding keeps the whole past resident as a key/value cache, so generating token t costs work proportional to t and the memory grows without bound, and there is no notion of anything outside the window at all. There is a whole zoo of cheaper attention variants, but almost every one of them trades away the property that made attention good, and none has matched it across modalities at scale. So I don't want a cheaper approximation of attention. I want to name what attention is doing that the cheap models can't, and get *that* back cheaply.

Here is the reframing that makes the field legible. A sequence model compresses context into a state and then acts on the state. Attention is the degenerate case where the compression is the identity — it keeps everything, the cache *is* the uncompressed context — which is why it is both effective (nothing thrown away) and expensive (nothing thrown away). At the other end is a recurrent model with a fixed-size state: cheap, constant work per step, linear training, no growing cache — but only as good as whatever its bounded state held onto. So the whole efficiency-versus-quality axis is one number: how big is the state, and how well is it used. The escape isn't a bigger state; it's a *smarter* compression — keep what's relevant, drop what isn't. And what counts as relevant depends on the actual tokens. That word, content, is going to be the whole story.

So what is the most principled cheap model on the table? The structured state space models, S4 and its line. Let me make sure I really understand them, because I suspect the answer is hiding in a property they have. They are inspired by a continuous linear system: a one-dimensional input signal x(t) pushed through an N-dimensional latent state,

  h'(t) = A h(t) + B x(t),  y(t) = C h(t),

with A an N×N matrix, B an N×1 vector, C a 1×N vector. This is the Kalman-filter / linear-dynamical-system object, nothing new in itself. To run it on a discrete sequence I pick a step size Δ and discretize. With zero-order hold — hold the input constant across each step of length Δ and integrate the ODE exactly — the discrete transition is

  Ā = exp(Δ A),  B̄ = (Δ A)^{−1}(exp(Δ A) − I) · Δ B,

and the system becomes a linear recurrence h_t = Ā h_{t−1} + B̄ x_t, y_t = C h_t. Let me sanity-check that B̄ in the scalar case so I trust the formula. If A is a scalar a < 0, the ODE h' = a h + b x with x held at x_t over [0, Δ] integrates to h(Δ) = e^{aΔ} h(0) + (e^{aΔ} − 1)/a · b x_t. So Ā = e^{aΔ} and B̄ = (e^{aΔ} − 1)/a · b, which is exactly (ΔA)^{−1}(e^{ΔA} − 1) · Δ b once I multiply and divide by Δ. Good — the discretization is exact for piecewise-constant input.

Why drag the continuous story along at all instead of just learning a discrete transition? Two reasons earlier work convinces me of. One: there is a special A — the HiPPO matrix — for which the state becomes a near-optimal online summary of the input history, the coefficients of the input on a basis of orthogonal polynomials. That is genuine principled long-range memory, not a decay heuristic. Two: the Δ parametrization is a clean timescale knob, and discretization keeps the dynamics normalized. So the HiPPO A is the reason these things remember.

Now the part that decides efficiency, the part I need to stare at hardest. Freeze (Δ, A, B, C) to be the same at every step and unroll from h_0 = 0:

  y_0 = C B̄ x_0,
  y_1 = C Ā B̄ x_0 + C B̄ x_1,
  y_2 = C Ā² B̄ x_0 + C Ā B̄ x_1 + C B̄ x_2,

and in general y_t = Σ_k C Ā^k B̄ · x_{t−k}. That is a convolution of the input with a single fixed kernel

  K̄ = (C B̄, C Ā B̄, C Ā² B̄, …),  y = x ∗ K̄.

So a linear time-invariant SSM is literally a global convolution, and that is gold for training: with the whole sequence in hand I don't unroll a recurrence, I build one length-L kernel and convolve with an FFT, fully parallel, in O(L log L); at inference I switch back to the recurrence and pay constant work per step. (The kernel isn't trivial to form for the HiPPO A — it can't be stably diagonalized, so S4 writes it as normal-plus-low-rank and reduces to a diagonal-plus-low-rank computation through a truncated generating function, a Cauchy kernel, and the Woodbury identity, getting the kernel in O(N + L). And S4D then showed you can throw the low-rank piece away: a plain diagonal A works about as well, so A is just N numbers per channel. I'll keep diagonal A in my pocket.)

But look hard at the assumption I just leaned on. The unrolling collapsed into a *single* kernel reused at every position **only because (Ā, B̄, C) don't depend on t** — the same C Ā^k B̄ is the coefficient at lag k everywhere. The convolution view, the FFT, the whole efficiency story is bought with time-invariance. Constant dynamics.

And constant dynamics is exactly the disease. Let me feel it on the smallest tasks rather than argue abstractly. Take Copying: emit a remembered block after a fixed offset. The spacing between an input and where it reappears is constant, so the model only needs to know *time* — "whatever came in K steps ago, output it now." An LTI model nails this: a kernel that's all zeros except a spike at lag K, or a recurrence that just delays. No content reasoning at all. Now perturb it into Selective Copying: scatter the tokens-to-remember at *random* positions with noise tokens between them that must be ignored. Suddenly the gap between a relevant input and its output is not fixed — it depends on how many noise tokens happened to fall in between, which depends on the *content*. A static convolution kernel is doomed: its coefficients are fixed lags, and no single set of lags works when the spacing varies per example. The recurrence is doomed for the mirror reason: Ā, B̄ are the same every step, so the model literally cannot decide "this token is noise, don't write it" versus "this token is data, keep it" — the transition that ingests x_t is the same whether x_t is signal or garbage. Induction Heads is the same lesson dressed as associative recall: see "Harry Potter," and when "Harry" later recurs, produce "Potter" — recognize a token by content and act at a content-determined moment. Constant dynamics can't.

So the tension, stated precisely: the time-invariance I love in LTI SSMs is what gives them the FFT convolution and the cheap scaling, and it is *also* what makes them unable to select based on content. Those are not two problems; they are one property seen from two sides. Which means the fix is forced — I have to make the dynamics depend on the input. The model must be able to look at x_t and change how it treats x_t.

Where do I inject the input-dependence? I'd like the smallest change that does the job. The parameters are Δ, A, B, C. B controls how the current input gets *written* into the state; C controls how the state gets *read* into the output; Δ is the discretization step, the timescale. If I want the model to be able to ignore a token I want control over whether x_t enters the state — that's B — and how much the state persists or is overwritten, which lives in Ā = exp(ΔA) and so is reachable through Δ. Reading selectively is C. So let me make B, C, and Δ functions of the input at each position: a projection s_B(x) = Linear_N(x) gives a per-position B of shape (B, L, N), s_C(x) = Linear_N(x) likewise, and for Δ a projection s_Δ(x) feeding Δ = softplus(parameter + s_Δ(x)) of shape (B, L, D). I'll come back to the exact shape of s_Δ — there's something to learn there. The headline: these parameters now carry a length dimension. The model has gone from time-invariant to time-varying.

What about A — should it be selective too? Let me check whether it would even add anything. A only ever touches the computation through Ā = exp(Δ A). If Δ is already input-dependent, then Ā = exp(Δ_t A) is already input-dependent through Δ_t even with a static A. So making A itself a function of the input is a redundant second route to the same place — selectivity in Δ already induces selectivity in Ā, hence in B̄ and the whole transition. I'll keep A static; it's the principled HiPPO/diagonal memory matrix, and Δ is enough to make the dynamics content-dependent. Fewer parameters, nothing essential lost.

Now the wall I should have seen coming, and it's serious. The instant Ā_t, B̄_t, C_t vary with position, go back to the unrolling: the coefficient relating y_t to x_{t−k} is now C_t Ā_t Ā_{t−1} … Ā_{t−k+1} B̄_{t−k}, which depends on *t*, not just on the lag k. There is no single kernel anymore. The convolution form is gone. The FFT is gone. The entire O(L log L) story that justified using SSMs evaporates the moment I add the selection I need. This is the trade the whole field had been quietly avoiding — it is *why* every structured SSM stayed LTI. So I'm forced back onto the recurrence h_t = Ā_t h_{t−1} + B̄_t x_t, and the recurrence has two problems that are exactly why nobody wanted to be stuck with it.

Problem one: it's sequential — each h_t needs h_{t−1}, a disaster on a GPU next to a parallel convolution. Problem two: memory. The hidden state has shape batch B, length L, channels D (the SSM runs independently per channel), state size N — that's B·L·D·N — while the input and output are only B·L·D. So the latent state is a factor of N bigger than the data, and N is something like 16, maybe up to 100. If I naively materialize the discretized Ā, B̄ and the running states across the whole sequence I'm writing and reading B·L·D·N numbers to and from slow GPU memory, and that blows up. The convolution mode existed precisely to avoid ever forming this expanded state — it bypassed h and worked directly with a B·L·D kernel. By going back to the recurrence I've reopened the very memory hole the convolution was invented to plug.

Let me take the two one at a time, because I think neither is actually fatal — they only looked fatal.

The sequential problem first. The recurrence is sequential, yes, but it is *linear*. And there's a classical fact about first-order linear recurrences: they parallelize. Write h_t = a_t h_{t−1} + b_t — with diagonal A everything decouples into scalar recurrences per channel-and-state-coordinate, so a_t, b_t are just numbers, which is another reason diagonal A is nice. Define a binary operation on pairs (a, b):

  (a₁, b₁) • (a₂, b₂) = (a₂ a₁,  a₂ b₁ + b₂).

What does this do? Think of each pair as the affine map "h ↦ a h + b." Composing (a₁,b₁) then (a₂,b₂) gives h ↦ a₂(a₁ h + b₁) + b₂ = (a₂a₁) h + (a₂b₁ + b₂) — exactly (a₂a₁, a₂b₁ + b₂). So • is composition of affine maps, and composition is associative. Let me confirm directly so I'm not hand-waving. Left-to-right:

  [(a₁,b₁) • (a₂,b₂)] • (a₃,b₃) = (a₂a₁, a₂b₁+b₂) • (a₃,b₃) = (a₃a₂a₁,  a₃a₂b₁ + a₃b₂ + b₃).

Right-to-left:

  (a₁,b₁) • [(a₂,b₂) • (a₃,b₃)] = (a₁,b₁) • (a₃a₂, a₃b₂+b₃) = (a₃a₂a₁,  a₃a₂b₁ + a₃b₂ + b₃).

Same. Associative. And the running scan of these pairs reconstructs the recurrence: set each step's pair to (a_t, b_t) = (Ā_t, B̄_t x_t), and the prefix composition up to t has second coordinate exactly h_t (starting from h_0 = 0). So computing all the h_t is a *prefix scan* under an associative operator — and associative prefix scans have a work-efficient parallel algorithm with O(L) total work and O(log L) depth, the Blelloch up-sweep/down-sweep. The time-varying recurrence parallelizes after all. The thing that would have killed parallelism — a nonlinearity over time — isn't here: the recurrence is linear in h even though the coefficients vary, and linearity is all the scan needs. (This is the same parallel-scan trick S5 used to run a diagonal SSM as a recurrence; the difference is I run it with input-dependent coefficients, which is the whole point.)

And while I'm here, let me reconsider whether the recurrence is even a bad deal on FLOPs, because I'd assumed the convolution was strictly cheaper. The recurrent scan does O(B·L·D·N) work — linear in L. The FFT convolution does O(B·L·D·log L). So the convolution carries a log L factor and the scan carries an N factor. For long sequences and a not-too-large state dimension N the scan can do *fewer* FLOPs, with a lower constant. Abandoning the convolution isn't even clearly a loss on arithmetic — the real problem was never FLOPs, it was the sequentiality (now solved by the scan) and the memory.

Now the memory problem, the one that actually decides whether this is practical. The issue is materializing B·L·D·N numbers in slow HBM. But I just learned this lesson from the attention world: on a GPU everything except dense matmul is bottlenecked by memory bandwidth, not arithmetic. The fix that worked there — don't write the big intermediate to HBM at all; compute it in fast on-chip SRAM and move only the small things across the slow bus. Apply the same discipline. What's small? The inputs Δ, A, B, C and the output y are far smaller than the expanded state; concretely, they scale like O(B·L·D + B·L·N + D·N). What's big? The discretized Ā, B̄ and the running states, size B·L·D·N. So: load Δ, A, B, C from HBM into SRAM; *do the discretization there* — form Ā, B̄ on chip, never in HBM; run the parallel scan there, keeping the expanded states on chip; multiply by C and sum over the state dimension to collapse N away; and write back only y, size B·L·D, to HBM. The expanded B·L·D·N state is born and dies inside SRAM and never touches slow memory. That cuts the asymptotic HBM traffic by a factor on the order of N, and the reported kernel speedup over a standard scan is 20–40×. (If a sequence is too long to fit a chunk in SRAM, split it into chunks and carry the running scan state across — the scan composes, so that's free.)

There is a backward-pass corollary I have to handle or the memory win is fake. Backprop through the scan needs the intermediate states. If I store them I'm back to writing B·L·D·N to HBM — the blowup I just avoided. So I won't store them; I'll *recompute* them in the backward pass. The inputs Δ, A, B, C and the output gradient are already reloaded HBM→SRAM for the backward; given them, regenerating the intermediate states on chip is cheaper than reading the expanded states from HBM. This is the recomputation idea, again straight from the IO-aware attention playbook, and it lands the activation memory of the whole selective-SSM layer at roughly where an optimized attention implementation sits. So three classical tricks stacked — parallel scan for the sequential problem, kernel fusion so the expanded state stays in SRAM, recomputation so backward doesn't reintroduce the blowup — and together they make a non-LTI, input-dependent SSM actually faster than the LTI convolution it replaced while scaling linearly in L.

Now let me pin down the form of the Δ parametrization, because I left s_Δ and τ_Δ as placeholders, and I think being careful here teaches me something. Take the smallest possible case and see what selection-on-Δ actually *is*. Set N = 1, A = −1, B = 1, let Δ_t depend on the input through a linear map, and use softplus for positivity. The continuous system is h'(t) = −h(t) + x(t) — a leaky integrator, the simplest decaying memory. The step is Δ_t = softplus(parameter + Linear(x_t)), and since a constant bias inside softplus is absorbed into the linear layer's bias, write Δ_t = softplus(Linear(x_t)). Now discretize with ZOH and watch. Ā = exp(Δ A) = exp(−Δ) = exp(−softplus(Linear(x_t))). Recall softplus(z) = log(1 + e^z), so exp(−softplus(z)) = 1/(1 + e^z) = σ(−z) = 1 − σ(z). So

  Ā_t = 1 − σ(Linear(x_t)).

And B̄: with A = −1, B = 1, ZOH gives B̄ = (ΔA)^{−1}(exp(ΔA) − I)·ΔB = −(exp(−Δ) − 1) = 1 − exp(−Δ) = 1 − Ā. So

  B̄_t = 1 − Ā_t = σ(Linear(x_t)).

Write g_t = σ(Linear(x_t)). The recurrence h_t = Ā_t h_{t−1} + B̄_t x_t becomes

  h_t = (1 − g_t) h_{t−1} + g_t x_t.

That is *exactly* the classical gating equation of an RNN — the convex combination of "keep the old state" and "write the new input," with a sigmoid gate computed from the input. I didn't put a gate in by hand; it fell out of input-dependent discretization of the leaky integrator. So the heuristic gate that LSTMs and GRUs reached for is the special case of selective discretization, and that tells me the *right* way to parametrize Δ: the softplus and the input-linear form aren't arbitrary, they are what makes Δ behave as a principled, learnable timescale that generalizes gating. A large Δ_t (g_t → 1) means "this step matters — overwrite the old state with the current input"; a small Δ_t (g_t → 0) means "ignore this token, persist the state." Selecting via Δ literally *is* the gate deciding whether a token enters memory. That's the Selective Copying solution staring back: filler tokens get Δ → 0 and slide past, data tokens get a large Δ so g_t → 1 and they get written. At a boundary, the same mechanism can flush the previous state; in the exact gate case it overwrites with the boundary input, so a zero or learned boundary input is what makes it a literal zero reset.

This also shapes s_Δ. I want the step size to depend on the token at every position and still let different channels carry different timescales, but a full D-by-D projection from the expanded signal down to D step sizes is wasteful. The gate intuition says there should be shared low-dimensional evidence for "write" versus "keep," then a channelwise map turns that evidence into per-channel steps. The minimal construction broadcasts a one-dimensional decision across channels; the implementation generalizes that to a trainable low-rank projection. So I project to a small rank R and expand: r_t = Linear_R(x_t) with R much smaller than D, s_Δ(x_t) = W_Δ r_t, and Δ_t = softplus(s_Δ(x_t) + b_Δ). In the implementation R ≈ d_model/16. B and C, by contrast, I do want full width-N per position, because they control the finer-grained "which coordinates of the state to write or read," and that's genuinely per-coordinate. The bias b_Δ is the learned base timescale, so the code adds it exactly once, right before the softplus.

A couple of smaller parametrization choices, reasoned rather than asserted. Real versus complex state: prior SSMs leaned on complex-valued states, which help on smooth perceptual signals — audio, video — where oscillatory bases matter. But for discrete, information-dense data — text, DNA — real-valued states work fine and are simpler and more hardware-friendly. Since my target is exactly the discrete modalities where LTI SSMs failed, real is the default. Diagonal A, real: a natural initialization is A_n = −(n+1), a spread of decay rates, the S4D-Real scheme descended from HiPPO theory — though once data is plentiful and the model is selective, fairly arbitrary initializations work, because Δ-selection is doing the heavy lifting now. And I'll initialize the Δ bias so softplus of it lands in a sensible range, around [0.001, 0.1], so the model starts with a spread of memory horizons rather than all-forget or all-keep.

Now I need to wrap this selective SSM into an actual network block, and I'd like to learn from how existing SSM architectures are built and then simplify. The standard one is H3-shaped: take an SSM, flank it with two multiplicative gated branches, put a short local convolution in front (H3 frames it as a shift-SSM, cheap mixing of a few neighboring positions), and then *separately* interleave a standard MLP block after it, the way Transformers alternate attention and MLP. So a "layer" is really two blocks: a sequence-mixing block and a channel-mixing MLP — two different things to stack and tune. There's a cleaner idea from the attention side, the gated attention unit, which fused the attention sublayer and the MLP sublayer into one homogeneous block. Let me do the analogous fusion here so I have a single repeating block instead of an SSM-block-then-MLP-block alternation.

So the block: take the input of width D and project it up by an expansion factor E (I'll use E = 2) into *two* branches of width E·D. Call them x and z. The x branch is the main path — run it through a short causal depthwise conv1d (the cheap local mixing that H3's shift-SSM was doing), apply a SiLU activation, then through the selective SSM, where all the content-dependent sequence mixing happens. The z branch is a gate — pass it through SiLU and multiply it into the main path elementwise (the multiplicative gating, and using SiLU here makes the gated-MLP part behave like the SwiGLU unit modern Transformers like). Then project the E·D result back to D with an output projection, and wrap the whole block in a pre-norm residual. Compared to the H3 block I've replaced its first multiplicative gate with the activation; compared to an MLP block I've added the SSM to the main branch — so the one block is genuinely "MLP block plus a conv→SSM path," which is why it replaces both. Stack it homogeneously with norms and residuals, and that's the whole architecture. No attention, no separate MLP block.

Let me sanity-check the parameter budget so this is a fair comparison to a Transformer. Most of the block's parameters live in the linear projections: the input projection makes two width-E·D branches from width D, that's 2·E·D² weights; the output projection is E·D → D, another E·D²; so ≈ 3·E·D² per block. The SSM's own parameters — the small Δ projection, the per-channel A, the B/C projections — are tiny next to those. With E = 2 that's about 6D² per block. A Transformer "layer" is multi-head attention (≈ 4D²) plus an MLP (≈ 8D² at the usual 4× width), so ≈ 12D². Two of my blocks ≈ 12D² ≈ one Transformer layer — so the way to match capacity is two of these blocks per Transformer layer.

One more thing the selectivity buys, worth noting because it's a quiet failure of the old models: people have observed that many sequence models don't actually get better when given more context, even though a model with a perfect ignore mechanism could always discard irrelevant context. The reason, I think, is that an LTI model can't choose to *ignore* irrelevant history — its convolution kernel applies the same weights to everything, so junk far back keeps leaking in. A selective model can drive Δ small to ignore a transient input, or drive Δ large to flush the previous state and focus on the current input, so it has a concrete mechanism for using longer context without being forced to carry all of it. It can also keep stitched-together sequences from bleeding into each other by resetting at boundaries. That's a direct mechanistic consequence of being able to select, and the property I'd want to verify.

Let me write the core computation exactly, the way I'd actually implement the real, input-dependent B/C path as a readable check. One detail is easy to mix up: the exact ZOH B̄ is what gave the leaky-integrator gate above, but the canonical code keeps the exact state transition Ā = exp(ΔA) and uses the first-order input update B̄ ≈ ΔB, which is why the scan below forms Δ · B · u. So the gate derivation is the design justification for the Δ parameterization, while this is the literal arithmetic in the code. The fused kernel does this same arithmetic, just in SRAM with the scan. For an input x of shape (B, L, D) and state size N:

```python
import torch
import torch.nn.functional as F

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    # u:     (B, D, L)   the main-branch signal x going into the SSM
    # delta: (B, D, L)   the input-dependent step Δ (before softplus)
    # A:     (D, N)      static (diagonal) state matrix, A = -exp(A_log) < 0
    # B:     (B, N, L)   input-dependent input matrix  (selective)
    # C:     (B, N, L)   input-dependent output matrix (selective)
    # D:     (D,)        skip connection;  z: (B, D, L) the gate branch
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()       # add the learned base timescale once
    if delta_softplus:
        delta = F.softplus(delta)                           # Δ = softplus(b_Δ + low-rank Linear(x)) > 0
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    # Discretize; here materialized, in the fused kernel this lives in SRAM only.
    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))   # Ā = exp(Δ A),  (B, D, L, N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)      # B̄ x_t ≈ (Δ ⊙ B) x_t

    h = A.new_zeros((batch, dim, dstate))
    ys = []
    for t in range(u.shape[2]):                                    # linear recurrence == associative scan
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]               # h_t = Ā_t h_{t-1} + B̄_t x_t
        ys.append(torch.einsum('bdn,bn->bd', h, C[:, :, t]))      # y_t = C_t h_t  (selective read-out)
    y = torch.stack(ys, dim=2)                                     # (B, D, L)

    if D is not None:
        y = y + u * D[..., None]                                   # per-channel skip inside the SSM
    if z is not None:
        y = y * F.silu(z)                                          # multiplicative gate (the z branch)
    return y
```

The for-loop is conceptual — what the hardware-aware version really does is replace it with the parallel associative scan over the operator (a, b) • (a′, b′) = (a′a, a′b + b′), with the discretization, the scan, and the multiply-by-C fused into one kernel so deltaA, deltaB_u, and the states never leave SRAM, and recomputed in the backward pass. The arithmetic is identical; only the memory choreography differs.

And the block that wraps it, mirroring the standard implementation:

```python
import torch, torch.nn as nn, math
from einops import rearrange

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # in_proj makes the two branches: x (main) and z (gate), each width d_inner.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # short causal depthwise conv = the cheap local mixing (H3's shift-SSM role).
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        # x_proj produces the *input-dependent* Δ (low-rank), B, and C from the signal.
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)   # low-rank Δ -> per-channel steps
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        # A: static, diagonal, real; stored as log so A = -exp(A_log) < 0 (S4D-Real init).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))                   # per-channel skip
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):           # (B, L, D)
        b, l, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l two_d -> b two_d l")
        x, z = xz.chunk(2, dim=1)               # x: main branch, z: gate branch

        x = self.act(self.conv1d(x)[..., :l])   # causal local conv (truncate the pad) then SiLU

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        # weight only here; the Δ bias is folded into the scan via delta_bias (don't double-count it)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=l)  # Δ before softplus, per channel
        B  = rearrange(B, "(b l) n -> b n l", l=l)                        # selective B
        C  = rearrange(C, "(b l) n -> b n l", l=l)                        # selective C
        A  = -torch.exp(self.A_log.float())

        y = selective_scan_ref(x, dt, A, B, C, D=self.D.float(), z=z,
                               delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)
```

So the whole chain, traced once forward. Good sequence modeling is good compression of context into a state, and compression has to be content-dependent or it fails on Selective Copying and Induction Heads. The cheap principled model, the structured SSM, gets its efficiency from being time-invariant — but time-invariance is exactly what forbids content-dependence, because a single fixed convolution kernel can't model variable, content-determined spacing. So I make Δ, B, C depend on the input, which restores selection — and selecting via Δ turns out to be precisely the classical RNN gate, derived from discretizing a leaky integrator — and that immediately destroys the convolution form. I recover the lost efficiency by going back to the recurrence, noticing it is a first-order *linear* recurrence hence an associative parallel scan, keeping the N-times-larger expanded state inside fast SRAM via kernel fusion instead of writing it to HBM, and recomputing intermediate states in the backward pass instead of storing them. Wrapped into a single homogeneous block that fuses the SSM path with a gated MLP and a short local conv, stacked with norms and residuals, that is a selective state space model — linear-time in training, constant-state at inference, and content-aware.
