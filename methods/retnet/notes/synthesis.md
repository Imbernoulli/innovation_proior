# RetNet synthesis notes

## Pain point / research question

Transformer training is parallel because teacher forcing exposes the whole sequence and self-attention
forms \(QK^\top\) for all positions at once. Autoregressive inference has the opposite profile:
each new query must interact with all previous keys, so decode work is \(O(N)\) per step and the
KV cache grows linearly with context length. RNNs have fixed state and \(O(1)\) per-step decode,
but their state dependency blocks time-axis parallel training. The target problem is the paper's
"impossible triangle": parallel training, \(O(1)\) inference, and Transformer-level quality.

## Load-bearing ancestors and contrasts

- Transformer: self-attention removes recurrent sequential dependence and makes each layer
  parallel over positions, but decoder self-attention still masks future positions and keeps all
  previous keys/values for generation.
- Linear Transformer: replaces \(\exp(q\cdot k)\) with a factored kernel
  \(\phi(q)^\top\phi(k)\), allowing a causal running state. It shows the attention-to-recurrence
  route, but the normalizer and feature-map choice weaken quality and position modeling.
- RoPE/xPos: rotate query/key representations so their inner product depends on relative
  position; xPos adds reciprocal magnitude factors. RetNet's diagonalized state matrix produces
  exactly this sort of query/key position factor.
- S4/H3/Hyena: state-space and long-convolution models give efficient long-range modeling, but
  the core kernels are not the same high-dimensional content-addressed \(QK\) interaction unless
  additional gates/hybrids are introduced.
- RWKV/AFT line: efficient recurrent or elementwise time-mixing paths give low-cost inference,
  but the state is channelwise/elementwise rather than a \(d_k\times d_v\) outer-product memory.

## Core derivation checks

Start from
\[
s_n=A s_{n-1}+K_n^\top v_n,\qquad
o_n=Q_n s_n=\sum_{m=1}^n Q_n A^{n-m}K_m^\top v_m .
\]
Make \(Q=XW_Q,K=XW_K\), diagonalize
\(A=\Lambda\mathrm{diag}(\gamma e^{i\theta})\Lambda^{-1}\), and absorb the basis into the
learned projections. The source equation becomes
\[
o_n=\sum_{m=1}^n(Q_n(\gamma e^{i\theta})^n)(K_m(\gamma e^{i\theta})^{-m})^\top v_m .
\]
With scalar \(\gamma\):
\[
o_n=\sum_{m=1}^n \gamma^{n-m}(Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m .
\]
The sign-sensitive point is the dagger: the key is written with \(e^{im\theta}\) before the
conjugate transpose, so the score contains the relative phase. TorchScale applies the same real
RoPE `theta_shift` to \(q\) and \(k\); the dot product supplies the transpose/conjugate effect.

Parallel form:
\[
Q=(XW_Q)\odot\Theta,\quad K=(XW_K)\odot\overline{\Theta},\quad
D_{nm}=\gamma^{n-m}\mathbf{1}_{n\ge m},\quad
\mathrm{Retention}(X)=(QK^\top\odot D)V.
\]

Recurrent form:
\[
S_n=\gamma S_{n-1}+K_n^\top V_n,\quad \mathrm{Retention}(X_n)=Q_n S_n.
\]
Unrolling gives exactly row \(n\) of the parallel form.

Chunkwise form with zero-based local indices: chunk \(i\) covers positions \(Bi,\ldots,B(i+1)-1\).
For a key at local \(j'\), store it in the chunk state with
\(\zeta_{j'}=\gamma^{B-1-j'}\). For a query at local \(j\) in the next chunk, read the previous
state with \(\xi_j=\gamma^{j+1}\). The exponents add:
\[
(B-1-j')+(j+1)=B+j-j',
\]
which is the true distance from that key to that query. This fixes the paper's overloaded local
index notation \(\zeta_{ij}, \xi_{ij}\).

## Canonical implementation mapping

Canonical code: `methods/retnet/code/torchscale` at commit
`4d1e0e82e5adf86dd424f1463192635b73fc8efc`.

- `torchscale/architecture/retnet.py`: `RetNetRelPos` builds RoPE angles and log decays
  \(\log(1-2^{-5-h})\). It has three branches: recurrent `(sin, cos), gamma`; parallel
  `(sin, cos), mask`; and chunkwise `(sin, cos), (inner_mask, cross_decay,
  query_inner_decay, value_inner_decay)`.
- `torchscale/component/multiscale_retention.py`: `k *= key_dim ** -0.5` implements
  \(QK^\top/\sqrt d\). Parallel path applies the normalized decay mask, then divides score
  rows by a clamped detached absolute row sum.
- Recurrent path is not just `gamma * prev + kv` in code: it carries `prev_key_value` plus a
  `scale` buffer, combines old/new terms with square-root scale factors, then sums over the
  key dimension. This is the stabilized version of the same normalized retention function.
- Chunkwise path normalizes inner scores, stores recurrent states divided by `kv_scale`, tracks
  `cross_scale`, and aligns inner/cross terms by `all_scale = maximum(inner_scale, cross_scale)`.
- The paper says GroupNorm per head; maintained TorchScale uses `RMSNorm(head_dim,
  elementwise_affine=False)` applied per head after the retention path. The retained property is
  scale-invariance of each head before final gating/projection.
- The maintained decoder uses RMSNorm pre-norm, GLU FFN, optional DeepNorm residual scaling, and
  padding to `recurrent_chunk_size` when chunkwise mode sees a non-multiple sequence length.

## Review findings applied

- Math: fixed/clarified the RoPE conjugate sign convention, chunk local-index notation, and
  "exponents add" wording. Kept the \(D_{nm}=0\) future case and \(D_{nm}=\gamma^{n-m}\) past
  case explicit.
- Code faithfulness: replaced illustrative `nn.GroupNorm` and bare recurrence with
  TorchScale-shaped RMSNorm, recurrent scale tracking, chunkwise relative-position masks, and
  scale alignment.
- Leak/scaffold: `context.md` keeps the target method unnamed and exposes only a generic token
  mixer slot. Tightened Transformer cost language so KV-cache inference memory is not confused
  with quadratic full-attention training memory.
- Voice: `reasoning.md` remains a first-person present-tense derivation with no markdown
  headers; `answer.md` is allowed to name RetNet as the final artifact.
