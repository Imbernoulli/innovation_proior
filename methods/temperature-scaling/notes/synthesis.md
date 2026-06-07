# Temperature Scaling — synthesis

Method: post-hoc calibration of a trained classifier. Divide logits by scalar T>0 before softmax; pick T by minimizing NLL on a held-out validation set. T fixed network, doesn't change argmax → accuracy unchanged.

## Problem (in-frame)
Modern deep nets (deep/wide ResNets w/ BN, little weight decay) are accurate but MISCALIBRATED — confidence > accuracy (overconfident). A decade ago shallow nets (LeNet) were well calibrated. Need confidence p̂ that equals P(correct | confidence=p̂).

## Diagnostic findings (pre-method, observed about EXISTING systems → context Background)
- Depth ↑, width ↑ → ECE ↑ (CIFAR-100 ResNet).
- BatchNorm → worse calibration despite slightly better accuracy.
- Less weight decay → worse calibration. Calibration keeps improving with more weight decay even past optimal-accuracy point.
- Disconnect: NLL overfits while 0/1 error keeps dropping. Network trades probabilistic fit for accuracy → overconfidence. (CIFAR-100: after LR drop at epoch 250, NLL overfits while test error drops 29%→27%.)

## Calibration metrics (pre-method)
- Reliability diagram: acc(B_m) vs conf(B_m), bins.
- ECE = Σ_m (|B_m|/n) |acc(B_m) − conf(B_m)|. M=15 bins.
- MCE = max_m |acc − conf|.
- NLL = −Σ log π̂(y_i|x_i). Minimized iff π̂ = true π.
- perfect calibration: P(ŷ=Y | p̂=p)=p ∀p.

## Baselines (prior calibration methods)
- Histogram binning (Zadrozny&Elkan 2001): bins, θ_m = avg positives. Non-param.
- Isotonic regression (Zadrozny&Elkan 2002): piecewise-constant non-decreasing f, min square loss. Generalizes hist binning (joint bins+values).
- BBQ (Naeini 2015): Bayesian model avg over binning schemes, Beta prior, closed-form marginal likelihood.
- Platt scaling (Platt 1999): q̂=σ(a z + b), fit a,b by NLL on val. Binary.
- Multiclass extensions: one-vs-all binning; Matrix scaling (Wz+b, params grow O(K²) → overfit); Vector scaling (W diagonal).
- Temperature scaling = single-param Platt: q̂=max_k softmax(z/T)^(k).

## Key derivation: entropy-max
Temperature scaling is the unique solution to:
 max_q  −Σ_i Σ_k q_i^k log q_i^k   (entropy)
 s.t. q prob dist; AND Σ_i z_i^{(y_i)} = Σ_i Σ_k z_i^k q_i^k  (avg true-class logit = avg weighted logit)
Lagrangian L = entropy + λ Σ_i [Σ_k z_i^k q_i^k − z_i^{y_i}] + Σ_i β_i Σ_k (q_i^k − 1).
∂L/∂q_i^k = −1 − log q_i^k + λ z_i^k + β_i  (paper writes −nK but the constant absorbs; the −1 from d/dq(q log q)=log q +1, the nK is from how they grouped; net: q ∝ exp(λ z_i^k)). Setting =0: q_i^k = exp(λ z_i^k + β_i − const). Normalize → q_i^k = exp(λ z_i^k)/Σ_j exp(λ z_i^j) = softmax(λ z_i). Set T=1/λ. → temperature scaling.
NOTE on the constant: d/dq[ -q log q ] = -(log q + 1). The paper's "-nK" is a grouping of constants; what matters is q^k = e^{λ z^k + β_i + c}; normalization kills c and β_i. I'll derive cleanly: stationarity gives log q_i^k = λ z_i^k + (β_i − 1), so q_i^k ∝ exp(λ z_i^k). Correct.

T behaviour: T→∞ → uniform 1/K (max uncertainty); T=1 → original; T→0 → point mass.

## Eval settings (pre-method datasets/metrics)
CIFAR-10/100, ImageNet, SVHN; NLP: 20News, Reuters, SST. Architectures ResNet, DenseNet, Wide-ResNet, LeNet, DAN, TreeLSTM. Hold-out validation set for fitting calibration param. Metrics ECE(15 bins), MCE, NLL, top-1 error. NO results.

## Code (faithful to guo et al temperature_scaling repo)
class ModelWithTemperature wraps model, has self.temperature (nn.Parameter init 1.5). forward: logits/T. set_temperature: collect val logits/labels, optimize T with LBFGS on NLL. T scalar.

## Design-why
- Single param T vs matrix/vector scaling: matrix scaling has K(K+1) params, overfits (poor on CIFAR-100/ImageNet large K). T can't overfit a held-out set with 1 param. T preserves ranking/accuracy (monotonic, doesn't change argmax). Empirically T alone ≈ best.
- Fit on NLL not ECE: NLL is smooth differentiable proper scoring rule; ECE non-differentiable (binned). NLL minimized at true probs.
- Why post-hoc on validation, not retrain: cheap, doesn't touch the net, accuracy untouched.
