**Problem (from step 2).** The gradient machine's **exact** split finding — pre-sort every feature,
scan every candidate threshold at every node — costs 3794.34 s/iteration on Higgs. The model is fine;
the exhaustive split search is the bottleneck. The plain least-squares-on-the-gradient tree also throws
away the second-order curvature that's cheaply available, and complexity is controlled only by depth and
shrinkage.

**Key idea.** **XGBoost**: (1) a **regularized second-order objective**. Taylor-expand the loss to
second order, gᵢ = ∂_ŷ L, hᵢ = ∂²_ŷ L, add Ω(f) = γT + ½λ‖w‖². Grouping by leaf gives a closed-form
optimal leaf weight wⱼ* = −Gⱼ/(Hⱼ+λ) and split gain
½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ, both pure sums of g and h, with γ pruning
weak splits automatically. (2) A **weighted quantile sketch**: the objective is a Hessian-weighted
squared loss Σᵢ ½hᵢ(f_t(xᵢ)−(−gᵢ/hᵢ))², so candidate split points are chosen to hold equal *Hessian
mass* (rank r_k(z) = (Σ_{x<z}h)/(Σh), candidates ε-apart → ~1/ε of them), not equal counts — a
histogram of per-bin (g,h) sums replaces the full scan. (3) **Sparsity-aware** split finding: enumerate
only non-missing entries and learn a per-node default direction for missing/zero values.

**Why it works.** Binning by Hessian-weighted quantiles turns each node's work from "scan all n rows"
into "sum g,h into a few hundred bins and scan those," collapsing per-iteration time by orders of
magnitude on large data, while the second-order regularized gain is a sharper split criterion than
least-squares-on-the-gradient, so accuracy holds or improves. The learned default direction makes sparse
features cheap (cost linear in *present* values) and handles missingness without imputation. Its
remaining cost: every histogram still sums over *all* data and *all* features each round — per-iteration
work stays proportional to (#data)×(#features).

**Change / code.** The regularized split-score primitives (real XGBoost source, `src/tree/param.h`),
plus the weighted-quantile approximate split-finding algorithm.

```cpp
// real XGBoost gain / leaf-weight (src/tree/param.h):
template <typename T1, typename T2>
XGBOOST_DEVICE inline static T1 ThresholdL1(T1 w, T2 alpha) {   // soft-threshold for L1
  if (w > +alpha) return w - alpha;
  if (w < -alpha) return w + alpha;
  return 0.0;
}
template <typename TrainingParams, typename T>
XGBOOST_DEVICE inline T CalcGainGivenWeight(
    const TrainingParams &p, T sum_grad, T sum_hess, T w) {
  return -(static_cast<T>(2.0) * sum_grad * w + (sum_hess + p.reg_lambda) * common::Sqr(w));
}
template <typename TrainingParams, typename T>
XGBOOST_DEVICE std::enable_if_t<std::is_floating_point_v<T>, T> CalcWeight(
    TrainingParams const &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight || sum_hess <= 0.0) return 0.0;
  T dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);   // w* = -G/(H+lambda)
  if (p.max_delta_step != 0.0f && ::fabs(dw) > p.max_delta_step)
    dw = ::copysign(p.max_delta_step, dw);
  return dw;
}
template <typename TrainingParams, typename T>
XGBOOST_DEVICE T CalcGain(TrainingParams const &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight || sum_hess <= 0.0) return static_cast<T>(0.0);
  if (p.max_delta_step == 0.0f) {
    if (p.reg_alpha == 0.0f)
      return common::Sqr(sum_grad) / (sum_hess + p.reg_lambda);            // G^2 / (H+lambda)
    return common::Sqr(ThresholdL1(sum_grad, p.reg_alpha)) / (sum_hess + p.reg_lambda);
  }
  T w = CalcWeight(p, sum_grad, sum_hess);
  T ret = CalcGainGivenWeight(p, sum_grad, sum_hess, w);
  return p.reg_alpha == 0.0f ? ret : ret + p.reg_alpha * std::abs(w);
}
```

```text
Approximate split finding with the weighted quantile proposal:
  for k = 1..m (features):
      propose S_k = {s_{k,1},...,s_{k,l}} by the WEIGHTED quantile sketch:
          rank r_k(z) = (1 / sum_h) * sum_{(x,h): x<z} h     # fraction of Hessian mass below z
          choose s so that |r_k(s_{k,j}) - r_k(s_{k,j+1})| < eps   # equal Hessian mass, ~1/eps points
  for k = 1..m:                                    # one data pass, accumulate per-bucket sums
      G_{kv} = sum_{ j: s_{k,v-1} < x_jk <= s_{k,v} } g_j
      H_{kv} = sum_{ j: s_{k,v-1} < x_jk <= s_{k,v} } h_j
  pick split maximizing  ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
# sparsity-aware: enumerate only non-missing entries; try missing->left and missing->right,
# keep the default direction with higher gain.
```

In the pure-L2 branch, `CalcGain` returns G²/(H+λ), so the split evaluator forms
Gain = ½[CalcGain(L)+CalcGain(R)−CalcGain(parent)] − γ; `reg_lambda` = λ, `reg_alpha` = an optional L1
penalty (an XGBoost extension beyond the pure-L2 Ω above) applied via `ThresholdL1`.
