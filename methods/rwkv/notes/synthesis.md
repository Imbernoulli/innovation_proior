# RWKV synthesis (grounded)

## Verified facts
- arXiv 2305.13048, "RWKV: Reinventing RNNs for the Transformer Era", EMNLP 2023 Findings.
- Canonical impl: BlinkDL/RWKV-LM, RWKV-v4/src/model.py + cuda/wkv_cuda.cu; minimal ref Johan Wind blog.

## Pain point
- Transformer self-attention: time O(T^2 d), space O(T^2 + Td). Quadratic in seq length T. KV cache grows linearly with T at inference. (Table 1, line 204.)
- RNN (LSTM/GRU): linear in T, O(d) inference state, but recurrence h_t=f(h_{t-1},x_t) is sequential in time -> cannot parallelize training; vanishing gradients. (Sec 2.1, eqs lstm-1..6.)
- Goal: parallel training like Transformer + O(Td) time / O(d) inference like RNN, WITHOUT approximation (no low-rank like Linformer/Performer).

## Lineage / ancestors
- Vaswani 2017 attention: Attn(Q,K,V)_t = sum_i e^{q_t·k_i} v_i / sum_i e^{q_t·k_i}. Quadratic because every q_t·k_i pair.
- Linear Transformers (Katharopoulos 2020): replace exp(q·k) with feature map phi(q)·phi(k), factorize -> O(Td^2). RWKV cites for linear scaling idea.
- AFT (Attention Free Transformer, Zhai 2021): Attn+(W,K,V)_t = sum_{i<=t} e^{w_{t,i}+k_i} ⊙ v_i / sum_{i<=t} e^{w_{t,i}+k_i}. NO q·k dot product. w_{t,i} is a learned T×T matrix of scalar pairwise position biases. Key move: token interaction via additive position bias + key, element-wise with v. But w_{t,i} as full T×T matrix is still O(T^2) storage and not recurrent.
- QRNN (Bradbury 2017): RNN factors into linear blocks W,U + recurrent block; data dependence on prev timestep blocks parallelization.

## The RWKV move (derivation order)
1. Start from AFT. The blocker to making AFT an RNN: w_{t,i} is an arbitrary T×T matrix; depends on absolute pair (t,i). To get a recurrence I need w_{t,i} to depend only on the RELATIVE gap (t-i).
2. Set w_{t,i} = -(t-i) w, where w ∈ (R_{>=0})^d is a per-CHANNEL decay vector (not a matrix). Non-negative so e^{w_{t,i}} <= 1, weights decay backward in time. This is "channel-directed" attention: decay rate differs per feature channel. (eq line 368.)
3. Plug in: the explicit WKV (eq nom-denom, line 455):
   wkv_t = [ sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} ⊙ v_i  +  e^{u + k_t} ⊙ v_t ]
           / [ sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i}  +  e^{u + k_t} ]
   - The current token t is pulled OUT of the decaying sum and given a separate "bonus" weight u (time_first) instead of decay. Reason: with pure decay the current token would be weighted same as immediate past; u lets the model attend specially to "now" and prevents degradation of W. (line 461.)
   - Note exponent uses (t-1-i): past tokens indexed relative to t-1.
4. Recurrence (Appendix sec:appendix-rnn, lines 1076-1080):
   a_0=b_0=0
   wkv_t = (a_{t-1} + e^{u+k_t} ⊙ v_t) / (b_{t-1} + e^{u+k_t})
   a_t = e^{-w} ⊙ a_{t-1} + e^{k_t} ⊙ v_t
   b_t = e^{-w} ⊙ b_{t-1} + e^{k_t}
   - a = numerator state (weighted sum of v), b = denominator (normalizer). Two d-vectors = O(d) state. Verify: a_{t-1} = sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i  (each step older term multiplied by e^{-w} once more). Yes: at step t-1, a_{t-1}=e^{-w}a_{t-2}+e^{k_{t-1}}v_{t-1}; unrolling gives sum_{i} e^{-((t-1)-i)w} e^{k_i} v_i. Matches numerator. Current token uses u not decay -> added separately in wkv_t, NOT folded into a_{t-1} until next step.
   - Crucial ordering: wkv_t uses a_{t-1},b_{t-1} (past only, with bonus on current), THEN a_t,b_t fold the current token in with regular e^{k_t} (no bonus) for future steps. Confirmed by CUDA kernel: "output using past -> accumulate current k,v -> advance state".

## Numerical stability (Appendix lines 1091-1104, CUDA kernel)
- e^{k_t} overflows. Store shared exponent p_t = running max. a'_t, b'_t are a_t,b_t divided by e^{p_t}.
- q := max(p_{t-1}, u+k_t); wkv_t = (e^{p_{t-1}-q} a'_{t-1} + e^{u+k_t-q} v_t)/(e^{p_{t-1}-q} b'_{t-1} + e^{u+k_t-q}).
- q' := max(p_{t-1}-w, k_t); a'_t = e^{p_{t-1}-w-q'} a'_{t-1} + e^{k_t-q'} v_t; b'_t similarly; p_t = q'.
- This is the online-softmax / log-sum-exp running-max trick.

## Token shift (linear interpolation), Sec 3.1.1, eqs time-mix1..3, channel_mix1..2
- r_t = W_r (mu_r ⊙ x_t + (1-mu_r) ⊙ x_{t-1}); same form for k,v (time) and r',k' (channel).
- mu per-channel learned. Implemented as nn.ZeroPad2d((0,0,1,-1)) = shift sequence by one (x_{t-1}).
- Why: cheap way to mix current+previous token, gives each projection access to a 2-token window; acts like a learnable 1D conv of width 2 / induction-like local mixing.

## Output gating, Sec 3.1.3
- Time-mix: o_t = W_o (sigmoid(r_t) ⊙ wkv_t). sigmoid(r) = "receptance" gate; r is the Receptance vector that gates how much past info (wkv) is received.
- Channel-mix: o'_t = sigmoid(r'_t) ⊙ (W'_v · max(k'_t,0)^2). squared ReLU activation (Primer, So 2021). Channel-mix = position-wise FFN analog: k' is hidden (expanded), squared-ReLU nonlinearity, W'_v projects back, sigmoid(r') gates.

## R,W,K,V meaning
- R Receptance: receiver/gate for past info. W Weight: positional time-decay (trainable). K Key: like attention key. V Value: like attention value.

## Decay parameterization (grounding nuance, code)
- In code the stored time_decay parameter is log-space: actual w = exp(time_decay), and state decay factor = exp(-w). Johan Wind: num = exp(-exp(decay))*last_num + ... So time_decay is unconstrained real, w=exp(time_decay)>=0 automatically (guarantees non-negativity). Init: w_i = -5 + 8*(i/(d-1))^{0.7+1.3l/(L-1)} (this is the time_decay param value; line 1126). u_i = 0.5*((( i+1) mod 3) - 1) + log(0.3) (line 1128, zigzag bonus init).

## Complexity (Sec 3.2-3.3)
- Time-parallel training: linear projections W_{r,k,v,o} are O(BTd^2) matmuls, fully parallel over time (each token independent). WKV is a serial scan O(BTd) but tiny compute, done by custom CUDA kernel; parallel over batch+channel.
- Time-sequential inference: RNN form, O(d) state per layer (4DL without p, 5DL with numerical-stability p; line 1114), O(Td) total, constant per-step.

## Gradient stability (Appendix sec:appendix-gradstab, lines 1168-1198)
- Claim: with bounded inputs and fixed params, gradients wrt W_k, W_v uniformly bounded in T (no explosion); decay w controls contribution decay (no vanish unless desired).
- Simplify: ignore token shift, wkv_T = sum K^e_t ⊙ v_t / sum K^e_t = E(v_t), a weighted average (E = average over weights K^e_t = e^{W_k x_t + w_{T,t}}).
- d(wkv_T)_i/d(W_v)_{i,j} = E_i[(x_t)_j], bounded by max_t |(x_t)_j|, independent of T.
- d(wkv_T)_i/d(W_k)_{i,j} = cov_i((x_t)_j, (v_t)_i) (a covariance under weights) — bounded, and softmax has >=2 nonzero terms (u and w) so cov doesn't degenerate to 0.
- Intuition: wkv is a softmax-weighted average -> Jacobian is an expectation/covariance, naturally bounded, unlike raw RNN product-of-Jacobians that explode/vanish.

## Architecture (Sec 3, Implementation)
- Stacked residual blocks; each = time-mix sub-block + channel-mix sub-block. Pre-LN: x = x + TimeMix(LN1(x)); x = x + ChannelMix(LN2(x)).
- Embedding (small init U(±1e-4) + extra LayerNorm), then L blocks, then final LayerNorm + linear head -> logits, cross-entropy.
- d = 4s convention (hidden = 4× embedding? Actually s=embedding dim, d=hidden=4s in init notes line 1119). Channel-mix hidden is wider (4×).
- Small init emb: embeddings ~ N(0,1e-4) tiny so the model escapes the noisy initial embedding fast; extra LN after embedding. Allows post-LN training too.

## Design-decision -> why
- AFT over linear-attn feature maps: AFT's additive e^{w+k} form has no q·k, so dropping the query entirely and making w relative-position gives an exact recurrence with no approximation (vs Performer/Linformer which approximate softmax). "without approximation" is the selling point.
- w per-channel vector not matrix: turns O(T^2) bias matrix into O(d) decay -> enables recurrence + linear cost.
- bonus u for current token: pure decay would treat "now" like the most-recent past; u decouples current-token weight from the decay law, preventing W degeneration.
- sigmoid(r) receptance gate: lets each channel decide how much accumulated context to admit (gating, like LSTM output gate but on the wkv average).
- squared ReLU in channel-mix: from Primer; smoother high-end than ReLU, found to help.
- token shift (interp w/ x_{t-1}): cheap 2-token local mixing, gives induction-head-like ability, lets r/k/v see a small window.
- Pre-LN + small-init-emb + zero-init most W: identity-like init + clean gradient path for deep stacks.

## Scaffold (pre-method) -> final code correspondence
- Generic autoregressive LM harness: token embedding, stack of N residual blocks each with two sublayers (a token-mixing op + a position-wise op) wrapped pre-LN, final LN + linear head, cross-entropy. The two sublayer bodies are the empty slots.
- Final code fills: TimeMix (token shift, r/k/v projections, WKV recurrence, sigmoid-r gate, out proj) and ChannelMix (token shift, k/r projections, squared-ReLU, value proj, sigmoid-r gate).
