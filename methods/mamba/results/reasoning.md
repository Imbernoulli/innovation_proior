Let me start where it actually hurts. I want a sequence model that I can use as the backbone of a foundation model — language, audio, DNA, the lot — and right now the only thing that really works at scale is attention. Attention works because every position can look directly at every other position; nothing is lost, nothing is summarized, the routing is dense and content-dependent. But that very property is the bill I keep paying. Training is quadratic in the sequence length, because I form interactions between all L² pairs. And at inference, autoregressive decoding has to keep the entire past around — the KV cache — so generating token t costs work proportional to t, and memory grows without bound. There's also no notion of anything outside the window at all. People have built a zoo of efficient-attention variants to patch the quadratic cost, but almost every one of them trades away the thing that made attention good in the first place, and none has convincingly matched it across modalities at scale. So I don't want a cheaper approximation of attention. I want to ask what attention is doing that the cheap models can't, and get *that* back cheaply.

Here's the reframing that makes the whole landscape legible. A sequence model is a thing that compresses context into a state and then acts on that state. Attention is the degenerate case where the "compression" is the identity — it keeps everything, the KV cache *is* the uncompressed context. That's why it's both effective (nothing is thrown away) and expensive (nothing is thrown away). At the other extreme is a recurrent model with a fixed-size state h: it's cheap, constant work per step, linear training, no growing cache — but it's only as good as whatever its bounded state managed to hold onto. So the efficiency-versus-quality axis of this entire field is really one number: how big is the state, and how well is it used. Efficient models need a small state; effective models need a state that holds everything that matters. The way out of the dilemma isn't a bigger state — it's a *smarter* compression: keep what's relevant, drop what isn't. And "relevant" depends on the actual tokens, on content. That word — content — is going to be the whole story.

So what's the most principled cheap model on the table? The structured state space models, S4 and its descendants. Let me make sure I really understand them, because I think the answer is hiding in a property they have. Start from the continuous linear system that they're inspired by: a one-dimensional input signal x(t) is pushed through an N-dimensional latent state,

  h'(t) = A h(t) + B x(t),  y(t) = C h(t).

A is N×N, B is N×1, C is 1×N. This is the Kalman-filter / linear-dynamical-system object, nothing new in itself. To use it on a discrete sequence I pick a step size Δ and discretize. With zero-order hold — hold the input constant across each step of length Δ and integrate the linear ODE exactly — I get the discrete transition

  Ā = exp(Δ A),  B̄ = (Δ A)^{−1}(exp(Δ A) − I) · Δ B,

and then the system is just a linear recurrence,

  h_t = Ā h_{t−1} + B̄ x_t,  y_t = C h_t.

Let me sanity-check that B̄ formula in the scalar case so I trust it. If A is a scalar a < 0, the ODE h' = a h + b x with x held at x_t over [0, Δ] integrates to h(Δ) = e^{aΔ} h(0) + (e^{aΔ} − 1)/a · b x_t. So Ā = e^{aΔ} and B̄ = (e^{aΔ} − 1)/a · b = (e^{aΔ} − 1)(aΔ)^{−1} · Δ b. Yes — matches, with the (ΔA)^{−1}(exp(ΔA) − I)·ΔB form. Good, the discretization is exact for piecewise-constant input.

Why bother with the continuous story at all, instead of just learning a discrete transition directly? Two reasons that earlier work makes me believe. One: there's a special choice of A — the HiPPO matrix — for which the state becomes a near-optimal online summary of the input history, the coefficients of the input projected onto a basis of orthogonal polynomials. That's a genuinely principled long-range memory, not a heuristic. Two: the Δ parametrization gives a clean knob for the timescale, and discretization keeps the dynamics well-normalized. Fine. The HiPPO A is the reason these things have memory.

Now the part that matters for efficiency, and the part I need to stare at hardest. Suppose I freeze (Δ, A, B, C) to be the same at every time step. Then unroll the recurrence from h_0 = 0:

  y_0 = C B̄ x_0,
  y_1 = C Ā B̄ x_0 + C B̄ x_1,
  y_2 = C Ā² B̄ x_0 + C Ā B̄ x_1 + C B̄ x_2,

and in general y_t = Σ_{k} C Ā^k B̄ · x_{t−k}. That is a convolution of the input with a single fixed kernel

  K̄ = (C B̄, C Ā B̄, C Ā² B̄, …, C Ā^k B̄, …),  y = x ∗ K̄.

So an LTI — linear time-invariant — state space model is literally a global convolution. And this is gold for training: if I have the whole sequence in hand, I don't unroll a recurrence, I compute one length-L kernel and do the convolution with an FFT, fully parallel, in O(L log L). For inference I switch back to the recurrence and pay constant work per step. Best of both worlds. (The kernel itself isn't trivial to compute for the HiPPO A, because that A can't be stably diagonalized — S4 writes it as normal-plus-low-rank, reduces to a diagonal-plus-low-rank computation through a truncated generating function, a Cauchy kernel, and the Woodbury identity, getting the kernel in O(N + L). And S4D then showed you can throw away the low-rank piece entirely: a plain diagonal A works about as well, so A is just N numbers per channel. Good, I'll keep that simplification in my pocket — diagonal A.)

But look at the load-bearing assumption I just made, and underline it: the unrolling collapses into a *single* kernel reused at every position **only because (Ā, B̄, C) don't depend on t**. The same C Ā^k B̄ has to be the coefficient at lag k everywhere in the sequence. The convolution view, the FFT trick, the whole efficiency story — all of it is purchased with time-invariance. Constant dynamics.

And constant dynamics is exactly the disease. Let me make myself feel it on the smallest possible tasks instead of arguing in the abstract. Take the Copying task: emit a remembered block of tokens after a fixed offset. The spacing between an input and where it should reappear is constant, so the model only needs to know *time* — "whatever came in K steps ago, output it now." An LTI model nails this: build a convolution kernel that's all zeros except a spike at lag K, or equivalently a recurrence that just delays. No content reasoning at all. Now perturb it into Selective Copying: scatter the tokens-to-remember at *random* positions, with noise tokens sprinkled in between that must be ignored. Suddenly the gap between a relevant input and its output is not fixed — it depends on how many noise tokens happened to fall in between, which depends on the *content*. A static convolution kernel is doomed: its coefficients are fixed lags, and there's no single set of lags that works when the spacing varies per example. The recurrence is doomed for the mirror reason: Ā, B̄ are the same every step, so the model literally cannot decide "this token is noise, don't write it into the state" versus "this token is data, keep it." The transition that ingests x_t is the same transition whether x_t is signal or garbage. Induction Heads is the same lesson in associative-recall clothing: see "Harry Potter," and later when "Harry" recurs, produce "Potter" — that requires recognizing a token by content and acting at a content-determined moment. Constant dynamics can't.

So here's the tension stated precisely. The thing I love about LTI SSMs — time-invariance — is the thing that gives them the FFT-convolution and the cheap scaling. And it's *also* the thing that makes them unable to select based on content. Those aren't two separate problems; they're the same property seen from two sides. Which means the fix is forced: I have to make the dynamics depend on the input. The model has to be able to look at x_t and change how it treats x_t.

Let me figure out *where* to inject the input-dependence, because there are several parameters and I'd like the smallest change that does the job. The parameters are Δ, A, B, C. Intuitively: B controls how the current input x_t gets written into the state; C controls how the state gets read out into y_t; Δ is the discretization step, the timescale. If I want the model to be able to ignore a token, I want control over whether x_t enters the state — that's B — and how much the state should persist or be overwritten — that's tied to Ā = exp(ΔA), which I can move through Δ. And reading selectively is C. So let me make B, C, and Δ functions of the input. Concretely, project the input at each position: s_B(x) = Linear_N(x) gives a per-position B of shape (B, L, N); s_C(x) = Linear_N(x) likewise; and for Δ, s_Δ(x) feeding Δ = τ_Δ(parameter + s_Δ(x)) of shape (B, L, D). I'll come back to the exact form of s_Δ and τ_Δ — there's a reason to be careful there. The headline is: these parameters now carry a length dimension. The model has gone from time-invariant to time-varying.

What about A? Should A be selective too? Let me check whether it would even add anything. A only ever touches the computation through the discretization, Ā = exp(Δ A). If Δ is already input-dependent, then Ā = exp(Δ_t A) is already input-dependent through Δ_t, even with a static A. So making A itself a function of the input would be a redundant second route to the same place — selectivity in Δ already induces selectivity in Ā and (through B̄) in the whole transition. I'll keep A static; it's the principled HiPPO/diagonal memory matrix, and Δ is enough to make the dynamics content-dependent. Simpler, fewer parameters, and I lose nothing essential.

Now I hit the wall I should have seen coming, and it's a serious one. The moment Ā_t, B̄_t, C_t vary with position, go back to the unrolling: the coefficient relating y_t to x_{t−k} is now C_t Ā_t Ā_{t−1} … Ā_{t−k+1} B̄_{t−k}, which depends on *t*, not just on the lag k. There is no single kernel anymore. The convolution form is gone. The FFT trick is gone. The entire O(L log L) training story that justified using SSMs in the first place evaporates the instant I add the selection I need. This is the trade the whole field had been avoiding — it's *why* every structured SSM stayed LTI. So I'm forced back onto the recurrence, h_t = Ā_t h_{t−1} + B̄_t x_t. And the recurrence has two problems that are exactly why nobody wanted to be stuck with it.

Problem one: it's sequential. Each h_t needs h_{t−1}. On a GPU that's a disaster compared to a parallel convolution. Problem two: memory. The hidden state has shape — let me count — batch B, length L, channels D (the SSM runs independently per channel), and state size N. That's B·L·D·N. The input and output are only B·L·D. So the latent state is a factor of N bigger than the data, and N is something like 16, maybe up to 100. If I naively materialize the discretized parameters Ā, B̄ and the running states across the whole sequence, I'm writing and reading B·L·D·N numbers to and from GPU memory. That blows up. The reason the convolution mode existed at all was precisely to *avoid* ever forming this expanded state — it bypassed h and worked directly with a B·L·D kernel. By going back to the recurrence I've reopened the very memory hole the convolution was invented to plug.

Let me take the two problems one at a time, because I think neither is actually fatal — they just looked fatal.

The sequential problem first. The recurrence is sequential, yes, but it is *linear*. And there's a classical fact about first-order linear recurrences: they parallelize. Write h_t = a_t h_{t−1} + b_t (I'm treating the diagonal/elementwise case, so a_t, b_t are just numbers per channel-and-state-coordinate; with diagonal A everything decouples into scalar recurrences, which is another reason diagonal A is nice). Define a binary operation on pairs (a, b):

  (a₁, b₁) • (a₂, b₂) = (a₂ a₁,  a₂ b₁ + b₂).

Check what this does. If I think of each pair as the affine map "h ↦ a h + b," then composing the map (a₁,b₁) then (a₂,b₂) gives h ↦ a₂(a₁ h + b₁) + b₂ = (a₂a₁) h + (a₂b₁ + b₂) — exactly the pair (a₂a₁, a₂b₁ + b₂). So • is just function composition of affine maps, and function composition is associative. Let me confirm associativity directly so I'm not hand-waving. Take three pairs and compose left-to-right:

  [(a₁,b₁) • (a₂,b₂)] • (a₃,b₃) = (a₂a₁, a₂b₁+b₂) • (a₃,b₃) = (a₃a₂a₁,  a₃(a₂b₁+b₂) + b₃) = (a₃a₂a₁,  a₃a₂b₁ + a₃b₂ + b₃).

And right-to-left:

  (a₁,b₁) • [(a₂,b₂) • (a₃,b₃)] = (a₁,b₁) • (a₃a₂, a₃b₂+b₃) = (a₃a₂a₁,  a₃a₂b₁ + (a₃b₂+b₃)) = (a₃a₂a₁,  a₃a₂b₁ + a₃b₂ + b₃).

Same. Associative. And the running scan of these pairs reconstructs the recurrence: with each step's pair set to (a_t, b_t) = (Ā_t, B̄_t x_t), the prefix composition up to t has second coordinate exactly h_t (starting from h_0 = 0). So computing all the h_t is a *prefix scan* under an associative operator — and associative prefix scans have a work-efficient parallel algorithm with O(L) total work and O(log L) depth (the Blelloch up-sweep/down-sweep). The time-varying recurrence is parallelizable after all. The non-linearity over time that would have killed parallelism isn't there — the recurrence is linear in h even though the coefficients vary, and linearity is all the scan needs. (This is the same parallel-scan trick S5 used to run a diagonal SSM as a recurrence; the difference is I'm going to run it with input-dependent coefficients, which is the whole point.)

And while I'm here, let me reconsider whether the recurrence is even a bad deal on FLOPs, because I'd assumed the convolution was strictly cheaper. The recurrent scan does O(B·L·D·N) work — linear in L. The FFT convolution does O(B·L·D·log L). So the convolution carries a log L factor and the scan carries an N factor. For long sequences and a not-too-large state dimension N, the scan can actually do *fewer* FLOPs, and with a lower constant. So abandoning the convolution isn't even clearly a loss on arithmetic — the real problem was never FLOPs, it was the two things above: sequentiality (now solved by the scan) and memory.

Now the memory problem, which is the one that actually decides whether this is practical. The issue is materializing B·L·D·N numbers in slow GPU memory (HBM). But I just learned this lesson from the attention world: on a GPU, everything except dense matmul is bottlenecked by memory bandwidth, not by arithmetic. The fix that worked for attention — don't write the big intermediate to HBM at all; compute it in fast on-chip SRAM and only move the small things across the slow bus. So apply the same discipline here. What's small? The inputs Δ, A, B, C and the output y are all only on the order of B·L·D (plus the tiny A and the projections), size B·L·D and D·N. What's big? The discretized Ā, B̄ and the running states, size B·L·D·N. So: load Δ, A, B, C from HBM into SRAM; *do the discretization there* — form Ā, B̄ on chip, never in HBM; run the parallel scan there, keeping the expanded states on chip; multiply by C and sum over the state dimension to collapse N away; and write back only y, of size B·L·D, to HBM. The expanded B·L·D·N state is born and dies inside SRAM and never touches the slow memory. That cuts the HBM traffic by a factor of N — which in practice is the 20–40× that decides whether the thing is usable. (If the sequence is too long to fit a chunk in SRAM, split it into chunks and carry the running scan state from one chunk to the next — the scan composes, so that's free.)

There's a backward-pass corollary I have to handle or the memory win is fake. Backprop through the scan needs the intermediate states. If I store them, I'm back to writing B·L·D·N to HBM — the blowup I just avoided. So I won't store them; I'll *recompute* them in the backward pass. The inputs Δ, A, B, C are small and already have to be reloaded HBM→SRAM for the backward anyway; given them, regenerating the intermediate states on chip is cheap, and it's cheaper than the HBM reads I'd otherwise pay. This is the recomputation idea, again straight from the IO-aware attention playbook, and it lands the activation memory of the whole selective-SSM layer at roughly the same place as an optimized attention implementation. So the design is three classical tricks stacked — parallel scan (for the sequential problem), kernel fusion (so the expanded state stays in SRAM), recomputation (so backward doesn't reintroduce the blowup) — and together they make a non-LTI, input-dependent SSM actually faster than the LTI convolution it replaced, while scaling linearly in L.

Let me go back and pin down the form of the Δ parametrization, because I left τ_Δ and s_Δ as placeholders and I think there's something to learn by being careful. I'll do the smallest case and see what selection-on-Δ actually *is*. Set N = 1, A = −1, B = 1, let Δ_t depend on the input through a linear map, and use softplus for the positivity of Δ. The continuous system is h'(t) = −h(t) + x(t) — a leaky integrator, the simplest possible decaying memory. The step is Δ_t = softplus(parameter + Linear(x_t)), and since a constant bias inside softplus is just absorbed into the linear layer's bias, write Δ_t = softplus(Linear(x_t)). Now discretize with ZOH and watch what comes out. Ā = exp(Δ A) = exp(−Δ) = exp(−softplus(Linear(x_t))). Recall softplus(z) = log(1 + e^z), so exp(−softplus(z)) = 1/(1 + e^z) = σ(−z) = 1 − σ(z). So

  Ā_t = 1 − σ(Linear(x_t)).

And B̄: with A = −1, B = 1, ZOH gives B̄ = (ΔA)^{−1}(exp(ΔA) − I)·ΔB = −(exp(−Δ) − 1) = 1 − exp(−Δ) = 1 − Ā. So

  B̄_t = 1 − Ā_t = σ(Linear(x_t)).

Now write g_t = σ(Linear(x_t)). The recurrence h_t = Ā_t h_{t−1} + B̄_t x_t becomes

  h_t = (1 − g_t) h_{t−1} + g_t x_t.

That is *exactly* the classical gating equation of an RNN — the convex combination of "keep the old state" and "write the new input," with a sigmoid gate computed from the input. I didn't put a gate in by hand; it fell out of input-dependent discretization of the leaky integrator. So the heuristic gate that LSTMs and GRUs reached for is the special case of selective discretization. That's not just a cute identity — it tells me the *right* way to parametrize Δ. The softplus and the input-linear form aren't arbitrary; they're what makes Δ behave as a principled, learnable timescale that generalizes gating. A large Δ_t (g_t → 1) means "this step matters — overwrite the state with the current input," and a small Δ_t (g_t → 0) means "ignore this token, persist the state." Selecting via Δ literally *is* the gate that decides whether a token enters memory. That's the Selective Copying solution staring back at me: filler tokens get Δ → 0 and slide past; data tokens get large Δ, so g_t → 1 and they get written. And state reset at a boundary is just Δ → ∞.

This also tells me how to shape s_Δ. I want the step size to depend on the token at every position and still allow different channels to carry different time scales, but a full D-by-D projection from the expanded signal to D step sizes is wasteful. The gate intuition says there should be shared low-dimensional evidence for "write" or "keep," then a channelwise map can turn that evidence into per-channel steps. So I use a low-rank factorization: r_t = Linear_R(x_t) with R much smaller than D, s_Δ(x_t) = W_Δ r_t, and Δ_t = softplus(s_Δ(x_t) + b_Δ). A rank-1 broadcast gate is the minimal special case; the implementation uses dt_rank ≈ d_model/16. B and C, by contrast, I do want full width-N per position, because they control the finer-grained "which coordinates of the state to write/read," and that's genuinely per-coordinate. The bias b_Δ is the learned base timescale, so the code has to add it exactly once, right before the softplus.

A couple of smaller parametrization choices, reasoned rather than asserted. Real versus complex state: prior SSMs leaned on complex-valued states, which help on smooth perceptual signals (audio, video) where oscillatory bases matter. But for discrete, information-dense data — text, DNA — real-valued states work fine and are simpler and more hardware-friendly. Since my target is exactly the discrete modalities where LTI SSMs failed, I'll make real the default. Diagonal A, real: a natural initialization is A_n = −(n+1), a spread of decay rates, the S4D-Real scheme descended from the HiPPO theory — though once the data is plentiful and the model is selective, even fairly arbitrary initializations work, because Δ-selection is doing the heavy lifting now. And I'll initialize the Δ bias so that softplus of it lands in a sensible range like [0.001, 0.1], i.e. so the model starts with a spread of memory horizons rather than all-forget or all-keep.

Now I need to wrap this selective SSM into an actual network block, and I'd like to learn from how the existing SSM architectures are built — and then simplify. The standard SSM block is H3-shaped: take an SSM, flank it with two multiplicative gated branches, stick a short local convolution in front (H3 frames it as a shift-SSM, it just mixes a few neighboring positions cheaply), and then — separately — interleave a standard MLP block after it, the way Transformers alternate attention and MLP. So a "layer" is really two blocks: a sequence-mixing block and a channel-mixing MLP. That's two different things to stack and tune. There's a cleaner idea floating around from the attention side — the gated attention unit fused the attention sublayer and the MLP sublayer into one homogeneous block. Let me do the analogous fusion here, so I have a single repeating block instead of an SSM-block-then-MLP-block alternation.

So the block: take the input of width D and project it up by an expansion factor E (I'll use E = 2) into *two* branches of width E·D. Call them x and z. The x branch is the main path: run it through a short causal depthwise conv1d (the cheap local mixing that H3's shift-SSM was doing), apply a SiLU activation, then through the selective SSM — that's where all the content-dependent sequence mixing happens. The z branch is a gate: pass it through SiLU and multiply it into the main path elementwise (this is the multiplicative gating, and using SiLU here makes the gated-MLP part behave like the SwiGLU unit that modern Transformers like). Then project the E·D result back down to D with an output projection, and wrap the whole block in a pre-norm residual. Compared to the H3 block I've replaced its first multiplicative gate with the activation; compared to an MLP block I've added the SSM to the main branch — so the one block is genuinely "MLP block plus a conv→SSM path," which is why it can replace both. Stack this block homogeneously — same block over and over, with norm and residuals — and that's the whole architecture. No attention, no separate MLP block.

Let me sanity-check the parameter budget so this is a fair comparison to a Transformer. Most of the block's parameters are in the linear projections: the input projection produces two width-E·D branches from width D, that's 2·E·D² weights; the output projection is E·D → D, another E·D²; so ≈ 3·E·D² per block. The SSM's own parameters — the small Δ projection, the per-channel A, the B/C projections — are tiny next to those. With E = 2 that's about 6D² per block. A Transformer "layer" is multi-head attention (≈ 4D²) plus an MLP (≈ 8D² with the usual 4× width), so ≈ 12D². So two of my blocks ≈ 12D² ≈ one Transformer layer — the right way to match capacity is to use two Mamba blocks per Transformer layer.

One more thing the selectivity buys, worth noting because it's a quiet failure of the old models: people have observed that many sequence models don't actually get better when you give them more context, even though more context should never hurt. The reason, I think, is that an LTI model can't choose to *ignore* irrelevant history — its convolution kernel applies the same weights to everything, so junk far back keeps leaking in. A selective model can drive Δ → ∞ (or the gate to reset) and wipe extraneous history whenever it wants, so in principle its quality should improve monotonically with context length, and it can keep stitched-together sequences from bleeding into each other by resetting at boundaries. That's a property I'd want to verify, but it's a direct mechanistic consequence of being able to select.

Let me now write down the core computation exactly, the way I'd actually implement the recurrence as a reference. One detail is easy to mix up: the exact ZOH B̄ is what gave the leaky-integrator gate above, but the implementation keeps the exact state transition Ā = exp(ΔA) and uses the first-order input update B̄ ≈ ΔB. That is why the scan below forms Δ · B · u. The fused CUDA kernel does this same arithmetic, just in SRAM with the scan. For an input x of shape (B, L, D), and state size N:

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
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)                                  # Δ = softplus(delta_bias + low-rank Linear(x)) > 0
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    # Discretize, in this reference materialized; in the fused kernel this lives in SRAM.
    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))   # Ā = exp(Δ A),  shape (B, D, L, N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)      # B̄ x_t ≈ (Δ ⊙ B) x_t

    h = A.new_zeros((batch, dim, dstate))
    ys = []
    for t in range(u.shape[2]):                                    # the linear recurrence == an associative scan
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                # h_t = Ā_t h_{t-1} + B̄_t x_t
        ys.append(torch.einsum('bdn,bn->bd', h, C[:, :, t]))       # y_t = C_t h_t  (selective read-out)
    y = torch.stack(ys, dim=2)                                     # (B, D, L)

    if D is not None:
        y = y + u * D[..., None]                                   # skip / residual within the SSM
    if z is not None:
        y = y * F.silu(z)                                          # multiplicative gate (the z branch)
    return y
```

The for-loop is conceptual — what the hardware-aware version really does is replace it with the parallel associative scan over the operator (a, b) • (a′, b′) = (a′a, a′b + b′), with the discretization, the scan, and the multiply-by-C all fused into one kernel so deltaA and deltaB_u and the states never leave SRAM, and recomputed in the backward pass. The arithmetic is identical; only the memory choreography differs.

And the block that wraps it, mirroring the standard implementation:

```python
import torch, torch.nn as nn, math
from einops import rearrange

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
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

        # A: static, diagonal, real; stored as log so A = -exp(A_log) < 0 (S4D-Real init).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))                   # per-channel skip
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):           # (B, L, D)
        b, l, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l two_d -> b two_d l")
        x, z = xz.chunk(2, dim=1)               # x: main branch, z: gate branch

        x = self.act(self.conv1d(x)[..., :l])   # local conv then SiLU

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        # weight only; the Δ bias is folded into the scan via delta_bias (don't double-count it)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=l)  # Δ before softplus, per channel
        B  = rearrange(B, "(b l) n -> b n l", l=l)                        # selective B
        C  = rearrange(C, "(b l) n -> b n l", l=l)                        # selective C
        A  = -torch.exp(self.A_log.float())

        y = selective_scan_ref(x, dt, A, B, C, D=self.D.float(), z=z,
                               delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)
```

So the whole chain, traced once forward: I started from the fact that good sequence modeling is good compression of context into a state, and compression has to be content-dependent or it fails on tasks like Selective Copying and Induction Heads. The cheap principled model, the structured SSM, gets its efficiency from being time-invariant — but time-invariance is exactly what forbids content-dependence, because a single fixed convolution kernel can't model variable, content-determined spacing. So I make Δ, B, C depend on the input, which restores selection (and selecting via Δ turns out to be precisely the classical RNN gate, derived from discretizing a leaky integrator) — and that immediately destroys the convolution form. I recover the lost efficiency by going back to the recurrence and noticing it's a first-order *linear* recurrence, hence an associative parallel scan; by keeping the N-times-larger expanded state inside fast SRAM via kernel fusion instead of writing it to HBM; and by recomputing intermediate states in the backward pass instead of storing them. Wrapped into a single homogeneous block that fuses the SSM path with a gated MLP and a short local conv, stacked with norms and residuals, that's a selective state space model — linear-time, constant-state at inference, and content-aware.
