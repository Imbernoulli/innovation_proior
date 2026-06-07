# Hyena — synthesis notes (Phase 1.5)

## The pain point
Attention is the gold-standard sequence mixer but costs O(L²) in time and memory. This caps context (textbooks, long audio, gigapixel images). Subquadratic approximations (linear/low-rank/sparse attention, Linformer, Reformer, AFT, Performer) trade expressivity for speed and need hybridization with full attention to reach Transformer quality. Question: is there a subquadratic operator that matches attention quality *without* attention?

## Three properties of attention worth keeping (distilled via mechanistic-interp probes: recall, induction)
- (a) **Data control**: y = A(u) v, where A(u) = softmax(u Mq Mk^T u^T / √D) is a *linear operator whose entries are a nonlinear function of the input*. One block encodes a whole family of linear maps; indexes through it per input. This is hypothesized to drive in-context learning.
- (b) **Sublinear parameter scaling**: attention params (the projection matrices Mq,Mk,Mv ∈ R^{D×D}) are decoupled from L. Lets Transformers spend params elsewhere (FFNs).
- (c) **Unrestricted context**: any-to-any dependency, no locality restriction (only causal masking).

So the target: build a data-controlled operator with sublinear params and unrestricted context, subquadratically.

## Building blocks already on the table
- **Convolution** y_t = (h*u)_t = Σ_{n} h_{t-n} u_n; as matrix-vec it's a Toeplitz matrix S_h times u. Unrestricted context needs a filter as long as the sequence (global, not FIR-local).
- **Explicit (FIR) filters**: CNN style, store M tap values. Memory ∂y_t/∂u_{t-n}=h_n is exactly M; params scale with filter length → can't afford length-L filters. O(ML) time.
- **Implicit filters**: h_t = γ_θ(t), a parametric function of position. Decouples param count from filter length. Two flavors:
  - SSM (S4/HiPPO): h_t = C A^t B (+Dδ). Memory extent set by spectral radius of A, tunable. Sublinear params. But materializing the kernel needs structured/iterative numerical methods (low GPU util) and a specific algebraic form.
  - FFN-of-position (CKConv, FlexConv, romero2021ckconv): γ_θ = a small MLP mapping a positional encoding of t to h_t. Free-form, any filter; single forward pass; high util. **This is the parametrization Hyena adopts.**
- **FFTConv (Cooley-Tukey)**: a length-L convolution is naively O(L²), but via the DFT convolution theorem it's O(L log L): pad to ≥2L-1, FFT both, multiply pointwise (Ŝ_h = W^{-1} D_H W circulant), iFFT. Materializes no Toeplitz matrix. This is what makes long convolutions affordable.
- **Gating / element-wise multiplication**: cheap O(L) data-dependent nonlinearity. Element-wise product in time = convolution in frequency (Σ duality).

## Baselines / lineage (load-bearing ancestors)
- **Self-attention (Vaswani 2017)**: y=A(u)v, the data-control template. Gap: O(L²).
- **S4 / HiPPO (gu2021efficiently, gu2020hippo)**: long convolution via structured SSM, sublinear params, long memory. Gap: not data-controlled (it's a fixed LTI filter — same operator for every input), and the kernel needs a special algebraic parametrization (NPLR) and care to materialize; underperforms on associative recall.
- **AFT (zhai2021attention)**: data control via gating + (softmax or single explicit conv). Gap: explicit conv → local / limited memory.
- **GSS (mehta2022long)**: gating + one long SSM convolution. = Hyena order-1. Gap: a single gate+conv is a limited data-controlled operator.
- **H3 (dao2022hungry)**: motivated by GSS failing associative recall. Two gates + two convolutions (one shift SSM = short conv, one diagonal SSM = long conv): z=k⊙(φ*v), y=q⊙(ψ*z). Surrogate attention matrix A(q,k)=D_q S_ψ D_k S_φ. = Hyena order-2 with SSM filters. Gap: fixed to 3 projections / order 2, and SSM (not free-form) filters.
- **CKConv (romero2021ckconv)**: implicit FFN-parametrized continuous conv kernels, sine activations for high-freq content. Gap: not data-controlled; used as plain conv.
- **SGConv (li2022makes)**: shows decaying multi-scale long conv filters work well → motivates the decay window.
- **Butterfly / Monarch (dao2019learning, dao2022monarch)**: fast structured matvec via products of sparse factors; decomposition length ↔ expressivity. Hyena's order-N product of D·S factors is "data-controlled butterfly."

## The Hyena operator (the landing)
Order-N: take N+1 projections (x¹,…,x^N, v). Recurrence:
- z¹ = v
- z^{n+1}_t = x^n_t · (h^n * z^n)_t , n=1..N
- y = z^{N+1}
i.e. y = x^N ⊙ (h^N * (x^{N-1} ⊙ (h^{N-1} * ( … (x^1 ⊙ (h^1 * v)))))).
Matrix form: y = H(u)v = D_x^N S_h^N ··· D_x^1 S_h^1 v, where D_x^n=diag(x^n), S_h^n=Toeplitz(h^n). Data-controlled (entries are functions of u), unrestricted context (long convs), sublinear params (implicit filters), O(N L log L) time.

### Why each piece (design-decision → why)
- **Why long (length-L) convolutions, not short**: unrestricted context (property c) demands any-to-any reach; FIR of size M only reaches M back.
- **Why implicit (FFN-of-position) filters, not explicit taps**: explicit length-L filter = L params/channel → kills sublinear scaling (property b). h_t=γ_θ(t) decouples length from param count.
- **Why FFN over SSM for the filter**: free-form (can approximate S4, CKConv, SGConv, FNO kernels), one forward pass = high GPU util, no iterative kernel materialization. SSM needs structured matrices + careful conditioning.
- **Why FFTConv**: makes the length-L conv O(L log L) instead of O(L²), without materializing the L×L Toeplitz/circulant matrix.
- **Why gating (element-wise mult)**: cheap O(L) way to inject input dependence → makes the operator data-controlled (the D_x^n factors). Without it, a chain of convolutions collapses to one convolution (LTI). Appendix B: D_q and W^* don't commute; if they did, A = W^* D_q D_Ψ D_k D_Φ W = a single conv. The non-commutativity of the gate is what makes the chain a genuine nonlinear-in-u operator.
- **Why interleave N gates with N convs / order N**: short recurrences recover GSS (N=1) and H3 (N=2); higher order → richer data-controlled matrix (butterfly: decomposition length ↔ expressivity).
- **Why the decay Window(t)=exp(-αt)(+bias)**: regularizes filters to finite effective length; α varies across channels → multi-scale filter lengths at init; mirrors li2022makes (decaying filters help). Bias so filters aren't forced to zero. Synergy: long decaying filters + high-freq filters let the operator pick specific inputs at specific steps (like a shift+diagonal SSM in H3).
- **Why sine activations in the filter FFN (CKConv-style)**: NN's have a low-frequency (spectral) bias (basri2020frequency); sine activations let γ_θ represent high-frequency filter content. Frequency ω_a of the sine and number of positional bands K together set the spectral richness / smoothness of filters at init. Non-smooth (rich high-freq) init trains better.
- **Why complex-exponential positional encoding ρ_k(t)=e^{i2πkt/L}**: a Fourier-feature basis for the FFN input; K bands precondition the spectrum (cut-off ≈ 2K+1). Increasing ω_a is a cheaper way to enrich spectrum than increasing K (which widens the FFN).
- **Why a short explicit depthwise conv on the projections (size 3)**: provides a local shift/short-range mix on q/k/v-like projections before the long convs — plays the role of H3's shift-SSM (gives the operator immediate-neighbor access cheaply).
- **Why causality via pad-to-2L-1**: for autoregressive LM the output must depend only on the past. If each h^n is causal (lower-triangular Toeplitz), H is lower-triangular (product of lower-tri × diagonal). In practice with FFTConv, eval filter at t=0..L-1 and zero-pad input+filter to 2L-1 → linear (aperiodic) conv, no future leakage.
- **Complexity**: O(N D L (log L + D)): per order, FFTConv is D·L log L; projections/output are L·D².

### Algorithm (forward)
1. Projection: ẑ=Linear(u): R^D→R^{(N+1)D}; depthwise short conv; split into x¹..x^N, v.
2. HyenaFilter: t=PosEnc(L); ĥ=FFN(t); h=ĥ·Window(t); split h¹..h^N.
3. for n=1..N: v ← x^n ⊙ FFTConv(h^n, v). return y=v. (then out_proj)

## Eval settings (pre-method, no outcomes)
- Mechanistic synthetics: associative recall, majority/counting, ICL of linear functions, arithmetic. Difficulty via seq len (1k…131k) and vocab size (10..40). 2-layer width-64 models. AdamW, cosine, warmup.
- LM: WikiText103 (GPT2 tok, vocab 50257), The Pile (125M/355M/1.3B). Metric: perplexity, FLOPs.
- Downstream: SuperGLUE, LAMBADA. Vision: ImageNet-1k (swap attention in ViT), CIFAR-10 (seq + 2D).
- Baselines to compare: Transformer/FlashAttention, H3, GSS, AFT-conv, RWKV, GPTNeo; conv parametrizations Conv1d/FNO/H3-SSM/TransferFunc/CKConv.

## Code grounding
HazyResearch/safari src/models/sequence/hyena.py — HyenaOperator (in_proj (order+1)*d_model, short_filter depthwise Conv1d size 3, recurrence loop v=filter_fn(v*x_i)), HyenaFilter (PositionalEmbedding complex-exp, Sin activation, MLP, ExponentialModulation window), fftconv_ref (rfft pad 2*seqlen, pointwise mult, irfft, + u*D residual bias).
