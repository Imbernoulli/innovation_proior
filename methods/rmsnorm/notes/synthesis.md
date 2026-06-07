# RMSNorm — synthesis notes

## Pain point / research question
- LayerNorm (Ba, Kiros, Hinton 2016) stabilizes training, decouples from batch, works for RNNs/variable length. But it adds per-step compute that becomes severe in deep/large nets, especially RNNs where a norm runs at every timestep. Net efficiency: faster convergence in *steps* but slower *per step* — the gain in steps is partly cancelled by wall-clock cost. Fig 1 of paper: same #steps LayerNorm beats baseline loss by a lot, but same wall-clock the advantage shrinks.
- Goal: keep the stabilization benefit of LayerNorm but cut the compute.

## Background concepts (load-bearing ancestors)
- **Internal covariate shift (ICS)** — Shimodaira 2000 / Ioffe-Szegedy 2015 motivation: a layer's input distribution drifts as earlier layers update, slowing training. (Footnote: Santurkar et al. 2018 argue it's really landscape smoothness, not ICS; we cite as alternative explanation.)
- **BatchNorm** (Ioffe & Szegedy 2015): standardize each activation using mean/var over the mini-batch. Re-scaling + re-centering invariance w.r.t. dataset; but couples across training cases → can't handle variable-length sequences / RNNs cleanly, needs running stats at test.
- **WeightNorm** (Salimans & Kingma 2016): reparameterize w = g·v/||v||, decouple length from direction of weight vectors. Weight-vector re-scaling invariant, but NOT input/dataset re-scaling invariant; on recognition tasks doesn't match BatchNorm accuracy.
- **LayerNorm** (Ba et al. 2016): for summed inputs a∈R^n to a layer, μ=(1/n)Σaᵢ, σ=√((1/n)Σ(aᵢ−μ)²), āᵢ=(aᵢ−μ)/σ·gᵢ. Statistics within one layer, no batch dependence. Two invariances credited for its success: **re-centering** (shift the inputs/weights → output unchanged because mean is subtracted) and **re-scaling** (scale inputs/weights → output unchanged because σ scales too).
- The central hypothesis to test: **which of the two invariances actually matters?** Paper hypothesizes re-scaling is the key one; re-centering (the mean subtraction) is dispensable.

## The method
- **RMSNorm**: drop μ. Normalize only by root mean square:
  āᵢ = aᵢ / RMS(a) · gᵢ, RMS(a) = √((1/n) Σ_{i=1}^n aᵢ²).
  No mean, no bias by default. When mean of a is 0, RMSNorm == LayerNorm. RMS forces a onto a √n-scaled unit sphere.
- Note vs L2/Euclidean norm: ||a|| = √n · RMS(a). Differ only by √n factor. WeightNorm uses L2 on weights; paper finds plain L2-norm of activations does NOT work for layer normalization — the √n scaling (i.e. dividing by √n inside) matters for robustness across vectors of different size. (Empirically L2-Norm baseline underperforms in MT.)

## Invariance analysis (derive)
- General form: y = f( (Wx)/RMS(a) ⊙ g + b ), a = Wx.
- Linearity: RMS(αx) = α·RMS(x) (α>0). Proof: RMS(αx)=√((1/n)Σα²xᵢ²)=|α|√((1/n)Σxᵢ²)=αRMS(x).
- **Weight matrix re-scaling invariant**: W'=δW → a'=δa, RMS(a')=δRMS(a), so (W'x)/RMS(a')=(δWx)/(δRMS(a))=(Wx)/RMS(a). y'=y. ✓
- **Input/dataset re-scaling invariant**: x'=δx → same cancellation. Extends to batch & whole dataset. ✓
- **Weight VECTOR re-scaling NOT invariant**: scaling individual rows w_i by different δ_i breaks linearity (RMS mixes all neurons' a_i, can't pull out a single δ_i). ✗ (LayerNorm same: ✗ for weight-vector re-scaling.)
- **Re-centering NOT invariant**: add shift → RMS does not subtract mean, so output changes. RMSNorm is NOT invariant to re-centering of weights or dataset. (Table: weight-matrix re-centering ✗, dataset re-centering ✗.) LayerNorm IS weight-matrix re-centering invariant.
- Single training case re-scaling: ✓ (like LayerNorm, unlike BatchNorm).

Invariance table (✓ invariant):
| | W re-scale | W re-center | w-vec re-scale | data re-scale | data re-center | single-case re-scale |
|BatchNorm| ✓|✗|✓|✓|✓|✗|
|WeightNorm|✓|✗|✓|✗|✗|✗|
|LayerNorm|✓|✓|✗|✓|✗|✓|
|RMSNorm|✓|✗|✗|✓|✗|✓|

## Gradient analysis (derive — VERIFIED)
- v = (Wx)/RMS(a) ⊙ g + b (the argument of f).
- ∂L/∂b = ∂L/∂v. ∂L/∂g = ∂L/∂v ⊙ (Wx)/RMS(a) = ∂L/∂v ⊙ ā/g... actually = ∂L/∂v ⊙ (normalized inputs). Both invariant to scaling of x and W (g-grad via RMS linearity). g-grad ∝ normalized inputs not raw → magnitude of g stable.
- Weight matrix grad: let a=Wx. ∂(a_j/RMS)/∂a_k = (1/RMS)[δ_jk − a_j a_k/(n RMS²)]. So R = (1/RMS)(I − (Wx)(Wx)ᵀ/(n RMS²)). VERIFIED by hand: ∂RMS/∂a_k = a_k/(n RMS); quotient rule gives exactly this.
- ∂L/∂W = Σ_i [ xᵀ ⊗ (diag(g ⊙ ∂L/∂v) × R) ]_i.
- Scaling x or W by δ: a→δa, RMS→δRMS, so R' = (1/(δRMS))(I − (δa)(δa)ᵀ/(n δ²RMS²)) = (1/δ)R. Negative correlation with δ.
  - Input scaling: x'=δx multiplies xᵀ by δ AND R by 1/δ → cancels → ∂L/∂W invariant to input scaling.
  - Weight scaling: W'=δW only scales R by 1/δ (x unchanged) → ∂L/∂W scaled by 1/δ → negatively correlated with weight scale = implicit learning-rate adaptation: big weights → small grads, keeps weight norm in check.

## pRMSNorm
- Neurons i.i.d. assumption → estimate RMS from first k = ⌈n·p⌉ elements: RMS_bar(a) = √((1/k)Σ_{i=1}^k aᵢ²). Normalize ALL n by this partial RMS.
- Linearity RMS_bar(αa)=α RMS_bar(a) still holds → same invariances as RMSNorm.
- Biased estimator, often inaccurate; gradient can explode with small k; but p=6.25% works in practice. Saves the reduction over the remaining (1−p) fraction.

## Why RMSNorm is cheaper
- LayerNorm needs TWO reductions: one for μ (sum), one for σ given μ (sum of squares of centered). Plus subtract μ from every element. RMSNorm needs ONE reduction (sum of squares) and no subtraction. ~7%–64% wall-clock speedup measured; biggest in RNNs (a norm per timestep) and Theano OE (64%), smallest in Transformer (7–9%, fewer sequential norms).

## Canonical code (code/rmsnorm_official.py — bzhangGo/rmsnorm)
- torch: scale=ones(d); norm_x = x.norm(2,-1); rms = norm_x/sqrt(d); x_normed = x/(rms+eps); return scale*x_normed (+offset if bias). Partial: split first int(d*p) dims, norm over those, d_x=partial_size.
- No bias/offset by default (no re-centering invariance to enforce). eps=1e-8.

## Code framework scaffold (pre-method)
- Generic feed-forward layer: a = Wx+b, y=f(a). A normalization module abstraction: nn.Module with learnable per-neuron gain g (and optional shift), a forward(x) that computes some within-layer statistic and rescales. The slot the method fills: which statistic, and how it rescales. LayerNorm baseline stub: subtract mean, divide by std.
