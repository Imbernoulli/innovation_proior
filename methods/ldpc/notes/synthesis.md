# Synthesis — LDPC codes + iterative BP (sum-product) decoding

## Pain point / research question (pre-method, 1950s–60s coding)
Shannon 1948: for a memoryless channel of capacity C, random block codes of rate R<C and block length n→∞ have decoding error → 0 (exponentially in n). Existence proof via random coding + ML/typical decoding. BUT: a random code has no structure; ML decoding compares received word against all 2^{nR} codewords — exponential in n. So capacity is achievable in principle but the achieving codes are undecodable in practice. The whole game: find codes that (a) approach capacity yet (b) have a *cheap* decoder, ideally O(n) per iteration.

Algebraic codes (Hamming, BCH, Reed-Solomon, 1950s): structured, bounded-distance decoders correct up to ⌊(d−1)/2⌋ errors via the algebra of the code. But the decoder only uses *hard decisions* and corrects nothing beyond the guaranteed t; it discards the channel's soft a-posteriori probabilities. Gallager (thesis §1.4): "No way is known to make use of the a posteriori probabilities at the output of more general binary input channels" with algebraic codes — a characteristic limitation of algebraic vs probabilistic decoding. Computation ~ cube of block length for BCH on BSC. They sit far from capacity at long block lengths because the decoding restriction (correct ≤ t, nothing beyond) inflates P_e.

Sequential decoding of convolutional codes (Wozencraft, Fano): probabilistic, uses soft info, but computation per digit is a random variable (waiting-line problem), and R_comp < C bounds the practical rate (the "computational cutoff rate" conjecture). Threshold decoding (Massey): simplest, shift registers + threshold, but short constraint lengths, higher P_e.

## The method's chain (reasoning.md spine)
1. Want structure that makes decoding cheap. A *linear* code is defined by parity-check matrix H (Hx=0 over GF(2)). Decoding = inference: given received y, find most probable x with Hx=0.
2. What makes inference cheap? SPARSITY of H. Make H low-density: each column has a small fixed number j of 1s (each bit in j checks), each row a small fixed number k of 1s (each check on k bits). This is the (n,j,k) low-density code. Rate R ≥ 1 − j/k. (Gallager thesis §1.2, Fig 2.1, n=20,j=3,k=4.)
3. Tradeoff: low-density codes are NOT optimum for a given block length — max usable rate is bounded below capacity (thesis §1.2). But: minimum distance grows *linearly* with n for j≥3 (Gilbert-bound-like, thesis Ch.2, Thm 2.2 + δ_0 with H(δ_0)=(1−R)ln2). j=2 is bad (distance grows only logarithmically). So j≥3 gives good distance, and the simple decoder more than compensates for the slight rate loss.
4. Decoder #1 — bit-flipping (thesis §4.1): hard decisions, compute all checks, flip any bit in more than a fixed number of unsatisfied checks, recompute, iterate. Works because for a sparse H a single error makes *all j* of its checks fail, while any other bit shares at most one check with it → a bit in many failed checks is the suspect. Simple but only BSC, rate far below capacity.
5. Decoder #2 — probabilistic / sum-product (thesis §4.2, the heart). Uses a-posteriori probabilities (soft). Tree picture (Fig 4.1): digit d at root, its j checks on tier 1, the other k−1 bits per check on tier 1, those bits' other checks on tier 2, etc. Within the ensemble where tree digits are independent, derive the iteration.
   - Lemma 4.1 (even-parity generating function): m independent bits, bit ℓ is 1 with prob P_ℓ. Prob of an EVEN number of 1s = [1 + ∏(1−2P_ℓ)]/2. Proof: ∏(1−P_ℓ+P_ℓ t) has coeff of t^i = prob of i ones; add ∏(1−P_ℓ−P_ℓ t) (flips sign of odd powers), odd terms cancel, set t=1, /2. Odd = [1 − ∏(1−2P_ℓ)]/2.
   - Theorem 4.1 (one tier): Pr[x_d=0|{y},S]/Pr[x_d=1|{y},S] = (1−P_d)/P_d · ∏_{i=1}^{j} [1 + ∏_{ℓ=1}^{k−1}(1−2P_{iℓ})] / [1 − ∏_{ℓ=1}^{k−1}(1−2P_{iℓ})]. Proof: Bayes; for x_d=0 a check is satisfied iff even # of 1s among other k−1 bits → Lemma gives Pr[S|x_d=0,{y}] = ∏_i [1+∏(1−2P)]/2; for x_d=1 need ODD → ∏_i [1−∏(1−2P)]/2; ratio cancels the 1/2s.
   - Iterate over tiers: each check's incoming P's are themselves computed by Thm 4.1 omitting that check (extrinsic!). The "omit the check containing d" / "omit one parity set" is the extrinsic-information rule that keeps it a tree computation. Valid while tree branches independent (~ first m≈log_{(j−1)(k−1)} n tiers); dependencies appear when tree closes (cycles), assumed to roughly cancel.
6. LLR form (thesis Eq 4.5–4.6). Let α=sign, β=magnitude of the LLR ln((1−P)/P). Eq 4.6: variable node SUMS LLRs (α'_d β'_d = α_d β_d + Σ_i (∏ α_{iℓ}) f[Σ f(β_{iℓ})]); check node uses f(β)=ln((e^β+1)/(e^β−1)) which is its own inverse, applied to the sum of f's. This is exactly the boxplus. Variable = add evidence; check = f∘(Σf) on neighbors.
7. Modern tanh equivalence (VERIFIED numerically): with L=ln(P0/P1)=ln((1−P)/P), tanh(L/2)=1−2P. Check-node output LLR = 2·artanh( ∏ tanh(L_i/2) ) = ln([1+∏(1−2P)]/[1−∏(1−2P)]). Same as Gallager. Variable-node update = Lc + Σ (incoming check LLRs), extrinsic = omit the target. MacKay-Neal δq=q0−q1=tanh(L/2), δr=∏δq (Eq 1), exactly this.
8. Why sparsity ⇒ near-optimal BP: each bit in j checks, each check on k bits, both small & fixed → the local neighborhood out to depth m is a TREE (no repeated nodes) for m up to ~log n, because (j−1)(k−1) per tier means n must exceed (j−1)^m(k−1)^m. On a tree, BP is EXACT inference (Pearl 1988): the iteration computes the true posterior. Sparsity is what makes the neighborhood tree-like and the per-iteration cost O(edges)=O(n) (j 1s/col, jn/k checks). Cycles/short girth break independence → BP becomes approximate; girth must be large. Trapping sets / near-codewords (small bit sets whose induced subgraph has few odd-degree checks) are where BP gets stuck even at high SNR → the error floor. [CONTEXT/frontier — pre-method phenomenon: minimum-distance / Gilbert bound, and error floor / trapping sets as observed BP failure modes. Richardson 2003. Do NOT foreground.]
9. Cost / convergence: thesis §4.3 — BSC weak bound, p_{i+1}=p_0 − p_0[(1+(1−2p_i)^{k−1})/2]^2 + (1−p_0)[(1−(1−2p_i)^{k−1})/2]^2 for j=3; converges to 0 for p_0 below a threshold (Fig 4.4: j=3,k=6,R=0.5 → p_0=0.0395; j=3,k=5,R=0.4→0.0612; j=4,k=6,R=1/3→0.0748; j=3,k=5,R=0.25→0.1069). For j>3, threshold b chosen to minimize (Eq 4.16); P_e decreases exponentially with a root of n for j>3 (Eq 4.21). Iterations ~ log of block length. Computation per digit per iteration independent of n.

## Code grounding (pyldpc — code/pyldpc)
- code.parity_check_matrix: Gallager construction. n_equations=n·d_v/d_c. First block: rows have d_c consecutive 1s. Remaining d_v−1 blocks = column permutations of first block. (Exactly thesis Fig 2.1 construction.)
- coding_matrix: H → G via double Gauss-Jordan over GF(2), so Hx=0 ⇔ x=Gv.
- encoder.encode: d=Gv mod 2; x=(−1)^d (BPSK: bit 0→+1, bit 1→−1); y=x+σ·noise; σ=10^{−snr/20}.
- decoder.decode: var=10^{−snr/10}=σ²; Lc=2y/var (channel LLR). Lq (var→check msgs), Lr (check→var msgs).
  - HORIZONTAL (check→var): X=∏_{n'≠target} tanh(0.5·Lq); Lr=log((1+X)/(1−X))=2artanh(X). [tanh check rule]
  - VERTICAL (var→check): Lq[i,j]=Lc[j]+Σ_{i'≠i} Lr[i',j]. [extrinsic sum]
  - POSTERIOR: L_post[j]=Lc[j]+Σ_i Lr[i,j]; decision x=1 iff L_post≤0 (L=log P0/P1).
  - stop when Hx=0 (incode) or maxiter.
- Sign convention: L=log P(0)/P(1); positive LLR ⇒ bit 0.

## Design decisions → why
- Sparse H (not dense): dense H = generic linear code, ML-hard. Sparse → tree-like neighborhoods → cheap, near-exact BP. THE lever.
- Regular j,k fixed: analytic tractability + uniform hardware; column weight j≥3 for linear minimum distance (j=2 only logarithmic).
- Rate R≥1−j/k: jn/k checks (some dependent → slightly higher rate).
- Pass EXTRINSIC messages (omit the receiving edge): prevents a node's own current belief from being fed back to itself around length-2 paths → keeps the tree assumption valid one more tier; this is the "omit one parity set" in Thm 4.1.
- LLR/log domain: products of probabilities → sums; check rule f(β)=ln((e^β+1)/(e^β−1)) self-inverse; numerically stable, hardware = adders + one nonlinearity.
- tanh form: 1−2P=tanh(L/2) turns Lemma-4.1 product into ∏tanh(L/2); check LLR=2artanh(∏tanh).
- BPSK Lc=2y/σ²: AWGN, x=±1, log N(y;+1,σ²)/N(y;−1,σ²)=2y/σ².
- Iterate till Hx=0: a valid codeword is a fixed point; gives a built-in stopping/failure detector (MacKay: all GL errors were *detected*).

## Sources
- Gallager 1963 thesis (refs/gallager_thesis_1963.pdf): §1.2,1.3,1.4; Ch.2 Thm2.1/2.2; Ch.4 Lemma4.1, Thm4.1, Eq4.5/4.6, §4.3 Eq4.7–4.21, Fig4.4. (1962 IRE paper = condensed version, same content.)
- MacKay & Neal 1996 (refs/mackay_neal_1996.pdf): BP=Pearl belief-prop, exact iff graph cycle-free; q_mn/r_mn messages; δq/δr horizontal rule; cycle-removal construction; rediscovery.
- Tanner 1981 (web): bipartite graph view (variable/check nodes), recursive decode.
- Shannon 1948 (web): capacity, random-coding existence, ML intractable.
- tanh⇔f(β) equivalence: verified numerically (notes above).
- Frontier context only: Richardson 2003 error floors / trapping sets (web).
