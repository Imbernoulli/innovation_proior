The gradient machine's exact split finding sets a strong accuracy reference on Higgs — test AUC $0.839593$ — but at a punishing cost of $3794.34$ seconds *per iteration*. The model is fine; the exhaustive split search is the bottleneck. At every node, for every one of the $28$ features, it pre-sorts all (up to) $10.5$ million examples by that feature and scans every adjacent pair as a candidate threshold. That is the right thing if I need the *provably* optimal split, but I do not — I need a split that is good enough, found fast. Two things to fix: the split search itself, and, while I am rebuilding the learner, the way trees are scored, because the plain least-squares-on-the-gradient tree throws away curvature information I am already in a position to compute.

I propose **XGBoost**, and it rebuilds the learner on three fronts. The first is a *regularized second-order objective*. I write the objective for adding tree $f_t$ with an explicit regularizer on the tree,

$$\mathcal{L}^{(t)} = \sum_i L\big(y_i, \hat y_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t),\qquad \Omega(f)=\gamma T + \tfrac12\lambda\lVert w\rVert^2,$$

where $T$ is the number of leaves and $w$ the leaf weights, so $\gamma$ penalizes adding leaves and $\lambda$ shrinks their magnitudes. Writing the leaf-magnitude penalty *into the objective* — rather than controlling complexity only through depth and shrinkage as before — means the split score itself knows about overfitting. Now Taylor-expand the loss to second order around the current prediction with $g_i=\partial_{\hat y}L$ and $h_i=\partial^2_{\hat y}L$; dropping the constant,

$$\tilde{\mathcal{L}}^{(t)} = \sum_i\Big[g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\Big] + \gamma T + \tfrac12\lambda\sum_j w_j^2.$$

A tree assigns every example in leaf $j$ the same value $w_j$, so grouping by leaf with $G_j=\sum_{i\in I_j}g_i$ and $H_j=\sum_{i\in I_j}h_i$ gives $\tilde{\mathcal{L}}^{(t)}=\sum_j\big[G_j w_j + \tfrac12(H_j+\lambda)w_j^2\big]+\gamma T$, a sum of independent quadratics. Minimizing each leaf gives the closed-form optimal weight and structure score,

$$w_j^* = -\frac{G_j}{H_j+\lambda},\qquad \tilde{\mathcal{L}}^{(t)} = -\tfrac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T,$$

so a leaf's quality is $G_j^2/(H_j+\lambda)$ — a clean form in both gradient and Hessian sums — and the gain of splitting a node's set $I$ into $I_L,I_R$ drops out as the change in structure score,

$$\text{Gain} = \tfrac12\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma.$$

The $-\gamma$ is the cost of the extra leaf — a split happens only if its quality improvement exceeds $\gamma$, pruning weak splits automatically — and the whole criterion is sums of $g$ and $h$, so to evaluate any candidate threshold I only need running sums of gradients and Hessians on each side.

The second front is the split search. The exact scan evaluates every distinct feature value; that is where the seconds go. I do not need every threshold — I need a small set of candidate thresholds *representative* of the data's distribution, so the best-of-candidates split is nearly as good as the best-of-all. The classical tool is a quantile sketch: pick $\sim 1/\varepsilon$ candidates so each bucket between consecutive candidates holds an equal fraction of the data, bin every feature once into these buckets (a histogram), and at each node just sum gradients and Hessians into the bins and scan them — collapsing per-node cost from "scan all $n$" to "scan a few hundred bin boundaries." But "equal fraction of the *count*" is the wrong notion of representative here, and seeing why is the crux. Rewriting the objective by completing the square,

$$\tilde{\mathcal{L}}^{(t)} = \sum_i \tfrac12 h_i\Big(f_t(x_i) - \big(-g_i/h_i\big)\Big)^2 + \text{const},$$

is a *weighted* squared loss with per-example weight $h_i$. So an example's importance to the split objective is its Hessian, not $1$, and the candidate thresholds should make each bucket hold an equal share of the *Hessian mass*. That is the **weighted quantile sketch**: with the rank function

$$r_k(z) = \frac{1}{\sum h}\sum_{(x,h):\,x<z} h$$

— the fraction of total Hessian mass below $z$ on feature $k$ — I choose candidates $\{s_{k,1},\dots,s_{k,l}\}$ so that $|r_k(s_{k,j})-r_k(s_{k,j+1})|<\varepsilon$, giving roughly $1/\varepsilon$ candidates with equal Hessian mass between them. Ordinary quantile sketches handle equal-count buckets but not arbitrary per-point weights with a merge/prune accuracy guarantee, so this needs a new weighted sketch data structure that can merge summaries across data partitions and prune them back to size while bounding the rank error.

The third front is sparsity. Real tabular data has many missing values, zeros, and one-hot columns, and the exact scan wastes time visiting all those zeros at every node. I give each node a learned **default direction**: enumerate only the *non-missing* entries of a feature to find the split, and decide whether missing values go left or right by trying both and keeping the higher-gain direction. The split search then costs linear in the number of *present* values, not the full matrix, and missing values are routed by a learned rule rather than imputed. Putting it together, the per-iteration work goes from "pre-sort and scan all $10.5$M rows at every node" to "bin once into Hessian-weighted buckets, then at each node sum $g,h$ into a few hundred bins and scan those," and the second-order regularized gain is a sharper criterion than least-squares-on-the-gradient — so accuracy should hold or improve as the time per iteration collapses. What the histogram does *not* touch is that every node still sums $g$ and $h$ over *all* the data falling into the bins, for *all* features, every round: the per-iteration cost stays proportional to (number of data points) $\times$ (number of features), which is the next thing to question.

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
