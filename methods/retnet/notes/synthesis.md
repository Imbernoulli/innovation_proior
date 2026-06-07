# RetNet synthesis notes

## Pain point / research question
Transformer trains in parallel (teacher-forced, all positions at once via QK^T) but its
autoregressive *inference* is bad: O(N) per step (each new token attends to all past keys),
and the KV cache grows linearly with sequence length -> O(N) memory, memory-bound decode.
RNNs invert this: O(1) per-step inference with fixed state, but the recurrence is sequential
so training does not parallelize over the time axis. Add the third corner: strong performance.
"Impossible triangle" = train parallelism + O(1) inference + Transformer-level performance, all at once.

## The three strands that fall short (baselines)
- **Linear attention (Katharopoulos et al. 2020, "Transformers are RNNs")**: replace exp(q·k) with
  phi(q)·phi(k); associativity lets the KV sum be reused -> O(1) recurrent inference, O(N) train.
  BUT: dividing by sum of phi(q)·phi(k) (the normalizer) is the issue; performance < Transformer,
  and it "struggles to encode position." The normalization is unstable / dilutes.
- **Return-to-RNN with element-wise ops (RWKV, Peng et al. 2023)**: gets O(1) inference and competitive
  performance via token-shift + exponential decay, but token-mixing is element-wise (channel-wise scalars)
  -> lower capacity; and the WKV recurrence is essentially sequential to train (element-wise, not matmul state).
- **Replace attention with SSM/long-conv (S4 Gu et al. 2021; H3; Hyena)**: O(1) or O(N log N), parallel via
  convolution / FFT, decent performance, but content-unaware mixing (the SSM kernel doesn't depend on the
  token content the way QK does). S4 = content-unaware special case.

## Core derivation (the heart)
Start from a *linear recurrence with a state matrix*:
  s_n = A s_{n-1} + k_n^T v_n   (s_n in R^{d x ...}, k_n row in R^{1xd}, v_n scalar/row)
  o_n = q_n s_n
Unroll: o_n = sum_{m<=n} q_n A^{n-m} k_m^T v_m.   <-- this is the bridge: a recurrence already
*equals* a weighted sum over the past, like attention, with weight q_n A^{n-m} k_m^T.

Make q,k content-aware: Q = X W_Q, K = X W_K.

Diagonalize A = Lambda (gamma e^{i theta}) Lambda^{-1}, gamma,theta in R^d.
A^{n-m} = Lambda (gamma e^{i theta})^{n-m} Lambda^{-1}. Absorb Lambda into W_Q, W_K:
  o_n = sum_{m<=n} q_n (gamma e^{i theta})^{n-m} k_m^T v_m
      = sum_{m<=n} [q_n (gamma e^{i theta})^n] [k_m (gamma e^{i theta})^{-m}]^T v_m.
The factor (gamma e^{i theta})^n on q and (...)^{-m} on k => xPos / rotary-like: relative
position appears as n-m. The magnitude part gamma^{n-m} = decay, the phase part e^{i theta (n-m)} = rotation (RoPE).

Simplify gamma to a *scalar* (per head): pull gamma^{n-m} out:
  o_n = sum_{m<=n} gamma^{n-m} (Q_n e^{in theta})(K_m e^{im theta})^dagger v_m.

### (a) Parallel form (training)
Q = (X W_Q) ⊙ Θ, K = (X W_K) ⊙ Θbar, V = X W_V, Θ_n = e^{i n theta}.
D_{nm} = gamma^{n-m} if n>=m else 0   (causal mask + decay in one matrix).
Retention(X) = (Q K^T ⊙ D) V.   No softmax. O(N^2 d) train, parallel over positions.

### (b) Recurrent form (inference) — O(1)/step
S_n = gamma S_{n-1} + K_n^T V_n   (S_n in R^{d_k x d_v}, outer product accumulation)
Retention(X_n) = Q_n S_n.
Equivalence proof: unroll S_n = sum_{m<=n} gamma^{n-m} K_m^T V_m. Then
Q_n S_n = sum_{m<=n} gamma^{n-m} Q_n K_m^T V_m = sum_{m<=n} (Q_n K_m^T) gamma^{n-m} V_m,
which is exactly row n of (QK^T ⊙ D)V. QED. (Θ absorbed into Q,K as content-aware rotation.)

### (c) Chunkwise form (long-seq training)
Chunk length B. Chunk i covers positions [Bi, B(i+1)).
Within-chunk index j = 0..B-1 (local). Global pos = Bi + j.
- Inner-chunk (parallel within chunk): (Q_{[i]} K_{[i]}^T ⊙ D) V_{[i]}, D the BxB decay-mask.
- Cross-chunk state R_i = state summarizing all chunks <= i, carried recurrently:
  R_i = K_{[i]}^T (V_{[i]} ⊙ zeta) + gamma^B R_{i-1},  zeta_{j} = gamma^{B-1-j}  (paper writes zeta_{ij}=gamma^{B-i-1}, i=local idx)
  Rationale: when R_i is read by a *later* chunk, the contribution of local key at position j inside chunk i
  must carry decay from j to the chunk boundary B-1, i.e. gamma^{(B-1)-j}; the remaining gamma to the reader
  is supplied by the reader's own offset and the gamma^B cross-chunk factor.
- Cross-chunk output for query at local position j in chunk i: (Q_{[i]} R_{i-1}) ⊙ xi, xi_{j} = gamma^{j+1}.
  Rationale: query at local pos j is global pos Bi+j. R_{i-1} holds keys from chunks < i, summed up to the
  boundary of chunk i-1 (local index B-1 of that chunk = global B i -1). Distance from that boundary to the
  query is (Bi+j) - (Bi-1) = j+1, so multiply by gamma^{j+1}. QED.
Total: Retention(X_{[i]}) = inner + cross. Complexity O(B^2 d + B d^2) per chunk -> O(N(B+d)d) linear in N.

## Gated multi-scale retention (MSR)
- h heads, each head uses a *different* gamma (multi-scale): gamma = 1 - 2^{-5-arange(0,h)}.
  Why multi-scale: a single decay forces one timescale; different gamma per head = different effective
  context windows / memory horizons, like multi-head gives different subspaces. Ablation: helps.
- head_i = Retention(X, gamma_i).
- GroupNorm over concatenated heads (SubLN/Magneto): because different gamma -> different output variance per
  head, normalize each head separately so variances are balanced. Scale-invariant property of GroupNorm is
  exploited for the normalization tricks below.
- Swish gate: MSR(X) = (swish(X W_G) ⊙ Y) W_O, Y = GroupNorm(Concat(heads)).
  Why gate: retention removed softmax -> lost a nonlinearity; the swish gate restores non-linearity / gating.
  Ablation: removing it hurts.

## Normalization tricks (numerical stability, exploiting GroupNorm scale-invariance)
GroupNorm(alpha * head) = GroupNorm(head), so we can rescale retention scores freely.
1. QK^T / sqrt(d)  (like attention scaling).
2. Replace D with Dtilde_{nm} = D_{nm} / sqrt(sum_i D_{ni})  (row-normalize the decay mask).
3. R = QK^T ⊙ D, normalize Rtilde_{nm} = R_{nm} / max(|sum_i R_{ni}|, 1). Then Retention = Rtilde V.
These keep forward/backward numerics bounded without changing the result.

## Architecture / params
- Block: Y = MSR(LN(X)) + X ; X' = FFN(LN(Y)) + Y. FFN = gelu(X W1) W2.
- DeepNorm init for stability.
- Param allocation: W_Q,W_K in R^{dxd}; W_V,W_G in R^{dx2d}; W_O in R^{2dxd} (value dim 2x q/k);
  FFN intermediate 2d (vs 4d in Transformer) to match param count. head_dim 256 (q/k), 512 (v).
- gamma in experiments: 1 - exp(linspace(log 1/32, log 1/512, h)).

## Canonical impl mapping (torchscale)
- multiscale_retention.py: parallel_forward (QK^T * mask, normalize by abs row-sum clamp, @ v, groupnorm),
  recurrent_forward (kv = decay*prev + k*v, output = sum(q*kv)), chunk_recurrent_forward (inner qk*mask@v +
  cross (q*query_inner_decay)@kv_recurrent, with scale alignment). theta_shift = rotary (rotate_every_two).
  group_norm = RMSNorm elementwise_affine=False. gate = silu(g)*output, then out_proj.
- retnet.py: RetNetRelPos builds (sin,cos) for rotation + decay mask; angle=1/(10000^...) rotary base;
  decay=log(1-2^{-5-arange(h)}). DecoderLayer = retention + FFN(GLU) with RMSNorm pre-norm, deepnorm alpha.

## In-frame rule
Do NOT name RetNet/the paper in context.md/reasoning.md as an artifact. May name "retention"/"RetNet"
as the thing being built in answer.md. Ancestors (Katharopoulos 2020, Gu 2021/S4, RWKV/Peng 2023,
Su 2021/RoPE, xPos) cited freely.
