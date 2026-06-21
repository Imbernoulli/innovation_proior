# S4 synthesis notes (Phase 1.5)

## The pain point (research question)
Long-range dependencies (LRD): real sequences need reasoning over 10k+ steps.
- RNNs/LSTMs: sequential, O(L) inference good, but vanishing/exploding gradients (Pascanu 2013) over long horizons; bounded effective memory.
- CNNs (WaveNet/TCN, dilated convs): parallel, but finite receptive field; need depth/dilation to reach far; not truly global.
- Transformers: O(L^2) attention compute+memory; efficient-transformer zoo (Performer, Linear Attention, Reformer) trades quality; still poor on LRA, Path-X = random.
Goal: a single layer with (i) principled unbounded memory, (ii) parallel training, (iii) O(1) per-step recurrent inference, (iv) ~linear cost in L and N.

## Background concepts
- Continuous linear SSM: x'(t)=Ax(t)+Bu(t), y(t)=Cx(t)+Du(t). D is a skip connection, drop it.
- HiPPO (gu2020hippo; voelker2019legendre LMU precursor): special A matrices making x(t) optimally memorize history of u via orthogonal-polynomial coefficients. HiPPO-LegS matrix:
  A_nk = -(2n+1)^.5(2k+1)^.5 if n>k; -(n+1) if n=k; 0 if n<k. Turns sMNIST 60%->98%.
- LSSL (gu2021lssl): first to use full SSM as a trainable deep layer, unify CTM/RNN/CNN views. But naive: O(N^2 L) compute, O(NL) memory. Their fast algorithm (char-poly based) numerically unstable (coeffs of (1-x)^N and its inverse grow ~2^N).

## Derivations to live out in reasoning.md

### 1. Bilinear discretization
Sample u_k = u(kΔ). Trapezoidal/bilinear rule on x'=Ax+Bu:
x_k - x_{k-1} = (Δ/2)(A x_k + A x_{k-1}) + (Δ/2)(B u_k + B u_{k-1}); simplest one-input version gives
(I - Δ/2 A) x_k = (I + Δ/2 A) x_{k-1} + Δ B u_k.
=> Ā = (I - Δ/2 A)^{-1}(I + Δ/2 A),  B̄ = (I - Δ/2 A)^{-1} Δ B,  C̄ = C.
Bilinear chosen over forward Euler (Ā=I+ΔA, unstable, no guarantee |eig|<1) and over ZOH; bilinear is the Cayley/Möbius transform, maps left-half plane to unit disk -> stability preserved. As Δ→0, Ā→I (relevant to instability proof).

### 2. Convolutional view
x_{-1}=0. Unroll: x_k = Σ_{j=0}^k Ā^{k-j} B̄ u_j. y_k = C̄ x_k = Σ_j C̄ Ā^{k-j} B̄ u_j.
=> y = K̄ * u (non-circular conv), with kernel
K̄ = (C̄B̄, C̄ĀB̄, C̄Ā²B̄, ..., C̄Ā^{L-1}B̄) ∈ R^L  (Krylov).
Given K̄, conv via FFT in O(L log L). Bottleneck: building K̄ needs L powers of Ā => O(N^2 L) compute, O(NL) mem.

### 3. Conjugation/diagonalization motivation
Lemma: (A,B,C) ~ (V^{-1}AV, V^{-1}B, CV) compute same map (change of state basis). Conjugation commutes with discretization.
If A diagonalizable A=VΛV^{-1}, kernel terms C̄Ā^kB̄ become Vandermonde (Λ^k) -> O((N+L)log²) via Vandermonde.
WALL: diagonalizing HiPPO. Its eigenvector matrix V_ij = C(i+j, i-j) has entries up to 2^{4N/3} (e.g. V_{3i,i}=C(4i,2i)≈2^{4i}). Ill-conditioned -> CV not computable. Diagonalization numerically infeasible.
Ideal: A normal (diagonalizable by unitary V, perfectly conditioned, spectral theorem). But HiPPO is NOT normal.

### 4. NPLR / DPLR
Observation: HiPPO = normal + low-rank. For LegS: A + ½(2n+1)^.5(2k+1)^.5 (rank-1, =P P^* with P_n=sqrt(n+½)) = -½I + S, S skew-symmetric (normal, pure-imaginary eigenvalues). So A = (normal) - PQ^*, i.e. A = VΛV^* - PQ^T with V unitary.
Conjugate by V: get DPLR over C: A = Λ - PQ^*, Λ diagonal, P,Q ∈ C^{N×r}, r=1 (LegS) or 2 (LegT).
But powering a DPLR matrix is still hard (low-rank term ruins it). Need a different route than powers.

### 5. Generating function trick (powers -> inverse)
Define truncated SSM generating function at node z:
K̂(z) = Σ_{i=0}^{L-1} C̄ Ā^i B̄ z^i.
Geometric-series collapse: Σ_{i=0}^{∞} C̄ Ā^i B̄ z^i = C̄ (I - Ā z)^{-1} B̄ (resolvent). Truncated:
Σ_{i=0}^{L-1} Ā^i z^i = (I - Ā^L z^L)(I - Ā z)^{-1}, so
K̂_L(z) = C̄ (I - Ā^L z^L)(I - Ā z)^{-1} B̄.
Evaluate at z = roots of unity ω_k = exp(-2πi k/L): at z^L = 1, factor (I - Ā^L) is constant -> fold into C̃ := (I - Ā^L)^* C (or learn C̃ directly).
=> K̂(ω_k) = C̃^* (I - Ā ω_k)^{-1} B̄. Then K̄ = iFFT(K̂) because K̂_j = Σ_k K̄_k exp(-2πi jk/L) is exactly the DFT. O(L log L) to recover.
KEY: powers Ā^i became a single inverse (I-Āz)^{-1}.

### 6. Back out original A from discretized (bilinear-resolvent lemma)
C̄^*(I - Ā z)^{-1} B̄ with Ā=(I-Δ/2 A)^{-1}(I+Δ/2 A), B̄=(I-Δ/2 A)^{-1}ΔB.
Factor out (I-Δ/2 A)^{-1} inside the bracket:
(I - Āz)^{-1} B̄ = [(I-Δ/2 A) - (I+Δ/2 A)z]^{-1} (I-Δ/2 A) · (I-Δ/2 A)^{-1} ΔB
= [(I-Δ/2 A) - (I+Δ/2 A)z]^{-1} ΔB
= [I(1-z) - (Δ/2)A(1+z)]^{-1} ΔB
= Δ/(1-z) [I - (Δ A)/(2 (1-z)/(1+z))]^{-1} B
= (2Δ)/(1+z) [2(1-z)/(1+z) I - ΔA]^{-1} B.
So: K̂(z) = C̃^* · (2Δ)/(1+z) · [2(1-z)/(1+z) I - ΔA]^{-1} B
Dividing through by Δ inside: = (2)/(1+z) C̃^* [ (2/Δ)(1-z)/(1+z) - A ]^{-1} B.
Define node g(z) = (2/Δ)(1-z)/(1+z). Then K̂(z) = 2/(1+z) · C̃^* (g(z) - A)^{-1} B.

### 7. Woodbury to remove low-rank
A = Λ - PQ^*. So (g - A) = (g - Λ) + PQ^*. Let R(z) = (g(z) - Λ)^{-1} (diagonal, trivial).
Woodbury: (Λ' + PQ^*)^{-1} = R - R P (1 + Q^* R P)^{-1} Q^* R, with Λ' = gI - Λ.
=> C̃^*(g-A)^{-1}B = C̃^* R B - C̃^* R P (1 + Q^* R P)^{-1} Q^* R B.
For r=1, (1 + Q^* R P) is a scalar. All four bilinear forms C̃^*RB, C̃^*RP, Q^*RB, Q^*RP.

### 8. Cauchy kernel
Each form C̃^* R B = Σ_j (c̃_j^* b_j)/(g(ω) - λ_j). Over all nodes ω this is a Cauchy matrix-vector product M_ij = 1/(ω_i - λ_j). Cauchy MVP: O(MN) naive, O((M+N)log²) exact arithmetic / FMM, O((M+N)log·log(1/ε)) numeric. So 4 Cauchy multiplies -> Õ(N+L) total, O(N+L) space. Recover K̄ by iFFT. DONE.

### 9. Recurrence in DPLR (O(N)/step)
Ā = A_1 A_0 where A_0 = (2/Δ)I + (Λ - PQ^*) (forward), D = ((2/Δ) - Λ)^{-1}, and A_1 = [ (2/Δ)I - Λ + PQ^* ]^{-1} = D - DP(1+Q^*DP)^{-1}Q^*D (Woodbury). Both DPLR -> O(N) MVP. B̄ = 2 A_1 B. Inverse of DPLR is DPLR. Equivalently, (I-Δ/2 A)^{-1}=(2/Δ)A_1.

## Architecture / design choices -> why
- O(N) DPLR params per SSM: paper derivation uses Λ, P, Q, B, C̃, Δ; the public S4 DPLR code stores half conjugate pairs and uses Q=P.conj() in the stabilized path.
- H independent copies of the 1-D SSM + position-wise linear mixing (O(H^2)) = like depthwise-separable conv with global kernels. Nonlinearity between layers (core SSM is linear).
- Δ (step size / timescale) learned per feature, init log-uniform in [dt_min,dt_max]=[1e-3,1e-1]; gives multi-timescale memory, also lets resolution change at test time.
- HiPPO init of A: principled memory; random A fails. Conjugate to DPLR (Λ,P) at init.
- Conjugate symmetry: A stored as N/2 conj pairs; kernel uses 2*Re(...). Halves cost.
- Lower lr (1e-3) + no weight decay on SSM params (A,B,Δ); standard lr+wd elsewhere.
- C̃ vs C: reparameterize to learn C̃ directly, drop the (I-Ā^L) recompute.
- rank-1 for LegS (the canonical), rank-2 for LegT.
- Why bilinear not Euler: stability (Cayley maps LHP->unit disk); Euler Ā=I+ΔA can be unstable.
- Why roots of unity nodes: makes truncation factor (I-Ā^L z^L) constant AND gives free iFFT recovery (the eval = DFT).
- Why generating function not direct kernel: turns L powers into 1 inverse -> Woodbury applies.
- Why normal+low-rank not plain diagonal: HiPPO eigenvectors are exp-ill-conditioned; normal part is unitarily diagonalizable (well-conditioned).

## Code grounding (state-spaces/s4)
- cauchy_naive(v,z,w): sum_n v_n/(z - w_n).
- SSMKernelDPLR.forward: _omega gives omega=exp(-2πi k/L) for k in [0,L/2], z = 2(1-omega)/(1+omega); A scaled by dt so node is (2/Δ)(1-ω)/(1+ω); stack [B,P],[C,Q]; v=B*C*dt; r=cauchy(v,z,A); rank-1 Woodbury k_f = r00 - r01 r10/(1+r11); k_f *= 2/(1+omega); k = irfft(k_f). Matches derivation exactly.
- nplr('legs'): P=sqrt(.5+arange(N)) rank-1; A+PP^* = -1/2 I + skew-symmetric part; diagonalize the shifted skew part -> Λ.
- S4D (diagonal) forward uses log_vandermonde — the simpler diagonal special case.
- S4Model backbone: encoder Linear -> stack of (S4 block + LayerNorm + residual + dropout) -> mean pool -> decoder Linear.
