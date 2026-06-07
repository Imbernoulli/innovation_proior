# Synthesis — Linear Attention ("Transformers are RNNs")

## The pain point (problem)
- Softmax self-attention: V'_i = softmax(QKᵀ/√D) V. Building QKᵀ is an N×N matrix → O(N²) time AND O(N²) memory (the full attention matrix must be stored to backprop through it).
- Two consequences: (a) training/forward is quadratic in sequence length N → context is capped; (b) autoregressive generation is brutal: per step i, attention must look back over all i previous positions → O(i) per step, O(N²) over a full sequence of length N, even with a KV cache. Generating a long sequence (e.g. an image pixel-by-pixel, 3072 pixels for CIFAR) is excruciating.
- Goal: an attention that is O(N) time and memory, supports causal masking at O(N), and — the prize — costs O(1) per step at generation time.

## Load-bearing ancestors (all from main text; substantive)
1. **Vaswani et al. 2017 (softmax self-attention).** The object being optimized. sim(q,k)=exp(qᵀk/√D); softmax row-normalizes. Scaling 1/√D keeps the dot-product variance ~O(1) so softmax doesn't saturate. The N×N matrix is the bottleneck.
2. **Autoregressive Transformer decoding.** Training is parallel (teacher forcing, all positions at once, masked). Inference is sequential; per-step cost grows with context because every new query re-attends to all past keys. Not constant per step.
3. **Sparse Transformer, Child et al. 2019.** Sparse factorizations of the attention matrix → O(N√N). Restricts which positions attend to which. Reduces train cost but does not give O(1)-per-step generation.
4. **Reformer, Kitaev et al. 2020.** LSH buckets similar queries/keys → O(N log N) by doing fewer dot products. But LSH forces keys ≡ queries (shared), so it can't be used where keys ≠ queries (general decoding); and it still doesn't speed up autoregressive inference per step. Hashing also injects noise.
5. **Tsai et al. 2019 (kernel view of attention).** Attention = kernel smoother: V'_i = Σ_j k(Q_i,K_j)V_j / Σ_j k(Q_i,K_j). softmax is one choice of kernel (exponential). Polynomial / RBF kernels work comparably. This reframes attention so the only requirement on sim is non-negativity. KEY enabling idea: attention does not *need* softmax; any non-negative similarity defines a valid attention.
6. **Linearized softmax for large-vocab classification (Blanc & Rippel 2017; Rawat et al. 2019; also Goodman 2001, Morin 2005, Mnih 2009 hierarchical/sampled softmax).** softmax over many categories is the classic bottleneck; people approximated exp(·) with a dot product of feature maps to sample efficiently. Inspiration: replace exp(qᵀk) by φ(q)ᵀφ(k).
7. **RNNs / LSTM (Hochreiter & Schmidhuber 1997).** A model that carries a fixed-size hidden state, updates it per step at O(1), predicts. The thing Transformers were supposed to have replaced. The "aha" target: causal linear attention literally becomes one.

## The derivation chain (insight → form)
1. Generalize: V'_i = Σ_j sim(Q_i,K_j)V_j / Σ_j sim(Q_i,K_j). Equals softmax attention when sim(q,k)=exp(qᵀk/√D). Only constraint for it to be a valid attention: sim ≥ 0.
2. Pick sim a kernel with feature map: sim(q,k)=φ(q)ᵀφ(k) ≥ 0 (φ into a non-negative space). Then
   V'_i = Σ_j φ(Q_i)ᵀφ(K_j) V_j / Σ_j φ(Q_i)ᵀφ(K_j).
3. **Associativity.** φ(Q_i)ᵀφ(K_j) is a scalar; pull φ(Q_i)ᵀ out of the sum:
   V'_i = φ(Q_i)ᵀ (Σ_j φ(K_j) V_jᵀ) / (φ(Q_i)ᵀ Σ_j φ(K_j)).
   The two sums Σ_j φ(K_j)V_jᵀ (C×M) and Σ_j φ(K_j) (C) are **independent of i** → compute once, reuse for all queries. Vectorized: (φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV). O(NCM) time, O(N(C+M)) memory. No N×N matrix.
4. **Feature map choice.** Exponential kernel's φ is infinite-dimensional → can't linearize exact softmax. Polynomial degree-2 has exact finite φ (O(ND²M), good when N>D²). For their seq lengths they want something cheap and positive: φ(x)=elu(x)+1. Positive (elu≥-1 so elu+1≥0 → sim≥0, attention well-defined, "converges normally"). elu over relu so the gradient is not killed for x<0 (relu would zero gradients on the negative side, dead units). O(NDM).
5. **Causal masking.** V'_i sums only j≤i. Linearized: V'_i = φ(Q_i)ᵀ Σ_{j≤i} φ(K_j)V_jᵀ / (φ(Q_i)ᵀ Σ_{j≤i} φ(K_j)). Define prefix sums S_i=Σ_{j≤i}φ(K_j)V_jᵀ, Z_i=Σ_{j≤i}φ(K_j). Recurrence S_i=S_{i-1}+φ(K_i)V_iᵀ, Z_i=Z_{i-1}+φ(K_i). V'_i=φ(Q_i)ᵀS_i/(φ(Q_i)ᵀZ_i). O(N) over the sequence; O(1) per step.
6. **RNN equivalence.** Per-layer recurrence: s_0=0,z_0=0; s_i=s_{i-1}+φ(x_iW_K)(x_iW_V)ᵀ; z_i=z_{i-1}+φ(x_iW_K); y_i=f_l( φ(x_iW_Q)ᵀs_i / (φ(x_iW_Q)ᵀz_i) + x_i ). State (s,z) fixed size. This IS an RNN — recurrence over *time*, not depth (contrast Universal Transformers' depth recurrence). Generation: O(1)/step, constant memory.
7. **Constant-memory training (gradient as cumsum).** Naive autograd stores all S_i → memory ×max(D,M). Derive grads as cumulative sums so forward+backward are O(N) time, O(1) extra memory (a CUDA custom op).
   - Numerator V̄_ie = Σ_d Q_id Σ_{j≤i} K_jd V_je = Σ_d Σ_{j≤i} Q_id K_jd V_je.
   - ∂L/∂Q_lt: Q_lt only affects V̄_l → ∂L/∂Q_lt = Σ_e ∂L/∂V̄_le (Σ_{j≤l} K_jt V_je) ⇒ ∇_{Q_i}L = ∇_{V̄_i}L (Σ_{j≤i} K_j V_jᵀ)ᵀ. Forward cumsum (1→N), same S as forward.
   - ∂L/∂K_lt: K_l affects all V̄_i, i≥l → ∂L/∂K_lt = Σ_e Σ_{i≥l} ∂L/∂V̄_ie Q_it V_le ⇒ ∇_{K_i}L = (Σ_{j≥i} Q_j (∇_{V̄_j}L)ᵀ) V_i. Reverse cumsum (N→1), like BPTT.
   - ∂L/∂V_lt similarly ⇒ ∇_{V_i}L = (Σ_{j≥i} Q_j (∇_{V̄_j}L)ᵀ)ᵀ φ(K_i). Reverse cumsum (shares the reverse-cumsum matrix with ∇_K).
   - Denominator + the division: left to autograd (cheap, it's a dot with the cumsum of φ(K)).

## Design-decision → why (table)
- **sim must be ≥0:** otherwise the weighted average / normalizer is not a valid attention (can divide by ~0, negative weights). Non-negativity is the *only* constraint.
- **φ(q)ᵀφ(k) factorization (vs. keeping exp):** exp(qᵀk) does not factor across q and k, so you can't pull Q out of the sum; a dot-product of feature maps does → associativity → O(N). This is the whole trick.
- **elu(x)+1 (vs relu+1, vs exp's true φ, vs polynomial):** positivity (valid attention); cheap O(D); finite-dim (exp's φ is infinite, impossible); elu keeps gradient alive for x<0 (relu zeroes it → dead gradient). Polynomial deg-2 is exact-finite but O(D²), only worth it when N>D².
- **prefix-sum/recurrence for causal (vs masking the N×N matrix):** masking the explicit matrix throws away the O(N) win; the running state keeps it O(N) and yields O(1)/step inference.
- **state = (S,Z) two memories:** S is the kernelized KV outer-product accumulator (C×M), Z the normalizer accumulator (C). Both fixed size ⇒ constant memory.
- **custom cumsum gradient / CUDA op (vs naive autograd):** naive stores every S_i (×max(D,M) memory) — kills the long-sequence/deep-model use case the method exists for; the cumsum form keeps backward at O(1) extra memory.
- **eps in denominator:** numerical stability — φ(Q_i)ᵀZ_i can be ~0.

## Canonical code (idiap/fast-transformers), corresponds to scaffold
- `FeatureMap` / `elu_feature_map = ActivationFunctionFeatureMap.factory(lambda x: elu(x)+1)`.
- `LinearAttention.forward`: KV = einsum("nshd,nshm->nhmd", φK, V); Z = 1/(einsum(φQ, φK.sum(1))+eps); V = einsum(φQ, KV, Z). (unmasked, O(N))
- `CausalLinearAttention.forward`: Z from einsum with K.cumsum(1); unnormalized V from `causal_dot_product(φQ,φK,V)` (the CUDA/C++ prefix-sum op); multiply by Z.
- `CausalDotProduct` (torch.autograd.Function): forward = running S, V̄_i=φ(Q_i)S; backward = forward cumsum for ∇Q, reverse cumsum for ∇K,∇V — exactly the appendix equations. C++ kernel: vvt_dot accumulates φ(K_i)V_iᵀ into kv, vm_dot does φ(Q_i)·kv.
- `RecurrentLinearAttention.forward`: state (Si,Zi); Zi+=K, Si+=einsum("nhd,nhm->nhdm",K,V); V=einsum(Q,Si,Z) — the RNN inference form.
- `AttentionLayer`: shared QKV projections + output projection; attention impl is pluggable (full / linear / causal-linear).

## In-frame discipline
- Never name the target paper/method-as-paper. May name "linear attention"/"linear transformer" as the thing being built (mainly answer.md). Cite ancestors (Vaswani, Child, Kitaev, Tsai, Hochreiter, Blanc&Rippel) freely.
- No proposed-method eval numbers (the 4000× table, bits/dim, PER) in any deliverable. The motivating cost facts (O(N²), O(1) target) are context.
