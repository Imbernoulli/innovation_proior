# Synthesis — Layer Normalization

## Pain point at the time (2015-2016)
- Deep nets trained with SGD are slow; days of training. Distributed (Dean 2012) hits diminishing returns and communication cost.
- Orthogonal lever: change the forward computation to make optimization easier.
- Batch normalization (Ioffe & Szegedy 2015) did exactly this for feed-forward nets: normalize each summed input (pre-activation) across the mini-batch to zero mean / unit variance, then apply learned per-neuron gain g and bias b. Converges much faster with plain SGD; batch noise also regularizes.

## BN mechanics (the ancestor we build on)
- For neuron i in layer l: a_i = w_i^T h. BN computes per-neuron statistics across the data distribution:
  mu_i = E_x[a_i],  sigma_i = sqrt(E_x[(a_i - mu_i)^2]),  abar_i = (g_i/sigma_i)(a_i - mu_i).
- Expectation over full data is impractical -> estimate mu, sigma from the current mini-batch.
- Consequences / failure modes (the gaps to exploit):
  1. The statistic is per-neuron, computed ACROSS the batch -> quality depends on batch size. Small batches -> noisy/unreliable statistics. Online (batch size 1) impossible.
  2. RNNs: same weights reused every time-step, but summed-input distribution differs per time-step and sequences have varying length. Applying BN "the obvious way" needs separate stored statistics per time-step; a test sequence longer than any training sequence has no statistics. (Cooijmans 2016 confirmed best recurrent BN keeps independent per-time-step stats, and that gain must be init to 0.1 — fragile.)
  3. Train uses batch statistics; test uses stored running averages -> train != test computation, extra bookkeeping (running averages per layer).

## The key insight (transpose)
- BN normalizes each neuron over the batch dimension. The matrix of pre-activations is (batch x neurons). BN reduces over rows (batch); statistics are per-column (neuron).
- TRANSPOSE: reduce over columns (neurons) instead. Compute mean/variance over ALL hidden units in a layer, for a SINGLE training case. Statistics become per-example, independent of batch.
- Justification offered: changes in one layer's output cause highly correlated changes in next layer's summed inputs (esp. ReLU). So fixing the mean and variance of the summed inputs WITHIN a layer also damps "covariate shift" — without needing the batch.
- This removes all three failure modes at once: no batch dependence, works at batch size 1, identical at train/test, and for RNN it's per-time-step statistics computed from the current step's summed inputs only — one shared gain/bias across time.

## LN definition
- mu^l = (1/H) sum_i a_i^l ;  sigma^l = sqrt((1/H) sum_i (a_i^l - mu^l)^2).
- All H units in a layer share mu, sigma; different training cases get different mu, sigma. (vs BN: all cases share per-neuron mu,sigma.)
- Normalized: h = f( (g/sigma) ⊙ (a - mu) + b ), gain g and bias b per-neuron, applied after normalization before nonlinearity.

## LN for RNN
- a^t = W_hh h^{t-1} + W_xh x^t.
- h^t = f[ (g/sigma^t) ⊙ (a^t - mu^t) + b ], mu^t,sigma^t from the H summed inputs at step t.
- One set of g,b shared across all time-steps.
- Why it helps RNN specifically: in a plain RNN the average magnitude of summed inputs tends to grow or shrink every step -> exploding/vanishing. LN makes the layer invariant to re-scaling ALL summed inputs -> stable hidden-to-hidden dynamics, independent of sequence length.

## Related methods / baselines
- Weight normalization (Salimans & Kingma 2016): reparameterize w = g * v/||v||. Equivalent to normalizing summed input with mu=0, sigma=||w||_2. A pure reparameterization of the net. Decouples length from direction.
- Recurrent BN (Cooijmans 2016, Laurent 2015, Amodei 2015): BN extended to RNN; per-time-step stats; gain init 0.1.
- Path-normalized SGD (Neyshabur 2015): reparameterization invariance study in ReLU nets.
- KEY contrast: BN (using expected statistics) and weight norm are just re-parameterizations of the original net. LN is NOT a reparameterization -> different invariance properties.

## Invariance analysis (Section 5) — the theory core
Unified view: all three normalize a_i via two scalars mu, sigma and learn b, g:
  h_i = f( (g_i/sigma_i)(a_i - mu_i) + b_i ).
  BN/LN: mu,sigma per their eqs. Weight norm: mu=0, sigma=||w||_2.

Invariance table (Table 1):
                 W re-scale | W re-center | w (single vec) re-scale | Data re-scale | Data re-center | Single-case re-scale
  Batch norm     Inv        | No          | Inv                      | Inv           | Inv            | No
  Weight norm    Inv        | No          | Inv                      | No            | No             | No
  Layer norm     Inv        | Inv         | No                       | Inv           | No             | Inv

Derivations:
- Weight re-scaling (BN, WN): scale single neuron's w_i by delta -> a_i, mu_i, sigma_i all scale by delta -> normalized value unchanged. Invariant.
- LN is NOT invariant to single-vector scaling (because mu,sigma are computed over the whole layer, scaling one row changes that row's a_i but mu,sigma are layer-wide combos). Instead LN is invariant to scaling the ENTIRE weight matrix and to shifting all incoming weights: W' = delta*W + 1 gamma^T. Then a' = W' x = delta W x + 1 (gamma^T x); mu' = delta mu + gamma^T x (the added constant 1(gamma^T x) is the same scalar across all H rows, so it lands in the layer mean), sigma' = delta sigma. So (a'_i - mu')/sigma' = (delta(a_i) + gamma^T x - delta mu - gamma^T x)/(delta sigma) = (a_i - mu)/sigma. Output unchanged. Invariant to whole-matrix re-scale and re-center.
  Note: if normalization applied to input BEFORE weights, no such invariance.
- Data re-scaling: all methods invariant to re-scaling the whole dataset (summed inputs scale, mu/sigma scale, cancels). LN additionally invariant to re-scaling an INDIVIDUAL training case x' = delta x: h_i' = f( (g_i/sigma')(w_i^T delta x - mu') + b_i) = f( (g_i/(delta sigma))(delta w_i^T x - delta mu) + b_i) = h_i. BN/WN not (their stats are per-neuron over data, not per-case).
- BN invariant to data re-centering (subtracts mean); LN not (re-centering data shifts a_i by w_i^T c which differs per neuron, doesn't cancel in layer stats).

## Geometry / Riemannian metric (Section 6 + appendix) — full derivation
- Model output as distribution P(y|x;theta). KL between theta and theta+delta measures separation -> Riemannian manifold.
- Second-order Taylor (Amari 1998): ds^2 = KL[P(y|x;theta) || P(y|x;theta+delta)] ~ (1/2) delta^T F(theta) delta,
  F(theta) = E_{x,y}[ (d log P / d theta)(d log P / d theta)^T ] = Fisher information.
- GLM: log P(y|x;w,b) = ((a+b)y - eta(a+b))/phi + c(y,phi), E[y|x]=f(a+b), Var[y|x]=phi f'(a+b).
  H independent GLMs -> multi-dim GLM, theta = vec([W,b]^T).
  Fisher: F(theta) = E_x[ Cov[y|x]/phi^2 ⊗ [[x x^T, x],[x^T, 1]] ].
- Normalized GLM (add gain g): theta = vec([W,b,g]^T). Block F-bar_ij is given (Eq nfisher), with
  chi_i = x - dmu_i/dw_i - ((a_i-mu_i)/sigma_i) dsigma_i/dw_i.
  The 3x3 block per (i,j) scales the w-direction by g_i g_j/(sigma_i sigma_j).

Two takeaways:
1. Implicit learning-rate reduction via weight-vector growth: the w_i-direction block of F-bar is scaled by g_i/sigma_i. Output is invariant to scaling w_i (under BN/WN; for LN under whole-matrix), but if ||w_i|| doubles, sigma_i (or ||w_i||) doubles, so the curvature along w_i scales by ~ 1/2 (specifically by 1/delta if w scales by delta; halving when doubled). For the same parameter update, a larger weight norm means smaller effective step in output space -> implicit early stopping / self-stabilization. Harder to change orientation of a large-norm weight vector.
   Careful constant: paper says "change by a factor of 1/2" when norm grows "twice as large" — because sigma_i scales linearly with the norm and the block has 1/(sigma_i sigma_j) i.e. 1/sigma_i^2 on the diagonal -> factor 1/(delta^2)? Actually the block has g_i g_j/(sigma_i sigma_j) AND chi_i which contains x and derivative terms ~ O(1). Net diagonal scaling along w_i is 1/sigma_i^2 from the two 1/sigma but chi_i ~ x (O(1)) ... The paper's stated result: curvature along w_i changes by factor 1/2 when norm doubles. I'll present the mechanism (sigma grows with norm -> metric along w shrinks -> effective LR along w shrinks) without over-precising the exact exponent, matching the paper's qualitative "implicit early stopping" claim, but I will state it as: since sigma_i grows in proportion to ||w_i||, the Fisher curvature along w_i is suppressed by 1/sigma_i, so a fixed-size update moves the function less — and I'll keep the paper's "factor of 1/2 when twice as large" as the diagonal 1/sigma scaling. (BN/WN have output invariance to w_i scaling; LN to matrix scaling.)
2. Learning gain magnitude (appendix "Learning the magnitude of incoming weights"):
   Project a gain update delta_g into the metric. Under BN:
     ds^2 = (1/2) delta_g^T E_x[ Cov[y|x]/phi^2 ] delta_g.  -> depends ONLY on prediction-error covariance, NOT on input scale.
   Under LN:
     ds^2 = (1/2) delta_g^T (1/phi^2) E_x[ matrix with entries Cov(y_i,y_j|x) (a_i-mu)(a_j-mu)/sigma^2 ] delta_g.  -> depends on normalized activation, bounded.
   Under weight norm:
     entries Cov(y_i,y_j|x) a_i a_j/(||w_i|| ||w_j||).
   Standard GLM (project gain update to weight via delta_gi w_i/||w_i||):
     ds^2 = (delta_gi delta_gj/(2 phi^2)) E_x[ Cov(y_i,y_j|x) a_i a_j/(||w_i|| ||w_j||) ]  -> depends on a_i a_j i.e. on input data scale & weights.
   Conclusion: gain learning in normalized models is robust to input/parameter scaling; in standard GLM it is not. So normalization stabilizes the geometry of magnitude learning.

## Appendix: exact LN function and RNN equations
LN(z; alpha, beta) = (z - mu)/sigma ⊙ alpha + beta,  mu=(1/D)sum z_i, sigma=sqrt((1/D)sum(z_i-mu)^2).
(NOTE: appendix text says "alpha init to zeros, beta init to ones" but alpha is the multiplicative gain and beta the additive bias — this is a swap/typo in the paper relative to the main-text convention where g init 1, b init 0. The sensible/standard init is gain=1, bias=0. Main text Section 7 explicitly says default init gain=1, bias=0. I will use gain=1,bias=0.)

LN-LSTM (read/handwriting):
 (f,i,o,g_gate) = LN(W_h h_{t-1}; a1,b1) + LN(W_x x_t; a2,b2) + b
 c_t = sigmoid(f) ⊙ c_{t-1} + sigmoid(i) ⊙ tanh(g_gate)
 h_t = sigmoid(o) ⊙ tanh( LN(c_t; a3,b3) )
LN-GRU (order-emb/skip-thoughts):
 (z,r) = LN(W_h h_{t-1}; a1,b1) + LN(W_x x_t; a2,b2)
 hhat = tanh( LN(W x_t; a3,b3) + sigmoid(r) ⊙ LN(U h_{t-1}; a4,b4) )
 h_t = (1 - sigmoid(z)) h_{t-1} + sigmoid(z) hhat
DRAW: LN only on c_t before tanh.

## Code grounding
- Canonical: PyTorch nn.LayerNorm; labml annotated implementation (saved in code/layer_norm_labml.py).
  forward: dims = last len(normalized_shape); mean = x.mean(dims); var = E[x^2]-E[x]^2; (x-mean)/sqrt(var+eps); *gain + bias.
- LN-LSTM cell built on top, applying LN to W_h h and W_x x separately and to c_t.

## Design decisions -> why
- Statistics over neurons (not batch): removes batch-size dependence, enables batch size 1 / online, identical train/test. WHY transpose works: correlated layer-to-layer covariate shift can be damped by fixing layer-wise summed-input moments.
- Per-neuron gain g & bias b (not shared scalar): normalization removes representational degrees of freedom (forces zero-mean unit-var); g,b restore the ability to represent identity / any mean & scale per neuron. Same rationale as BN's gamma,beta.
- Applied to summed inputs (pre-activation), after normalization before nonlinearity: keeps the nonlinearity operating in a controlled regime; normalizing post-activation would interact badly with saturating units.
- LN applied to W_h h and W_x x SEPARATELY in RNN (two LN, then add bias): the two contributions have different scales/distributions; normalizing the sum jointly would let one dominate. Separate LN keeps each contribution standardized.
- Gain init 1, bias init 0: start as identity transform (after normalization) so LN doesn't disrupt initial dynamics. (Contrast: recurrent BN needed gain=0.1 to avoid vanishing through tanh; LN robust to init scale, found gain=1 best.)
- Not applied to final softmax/logit layer in feed-forward MNIST: prediction confidence depends on logit SCALE; LN's scale-invariance would erase that. So exclude output layer.
- eps inside sqrt: numerical stability when variance ~ 0 (e.g. all units equal).
- Why LN underperforms BN in ConvNets: in conv layers many units (near image boundary) are rarely active and have very different statistics from the rest; the "all units contribute similarly" assumption that makes layer-wide stats meaningful fails. FC layers satisfy it; conv layers don't.
