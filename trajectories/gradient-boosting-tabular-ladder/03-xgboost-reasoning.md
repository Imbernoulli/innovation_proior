A single boosting iteration is costing 3794.34 seconds on Higgs. The model is fine — the gradient
framework is general and the accuracy is there — but the exact split search is doing an absurd amount of
work: at every node, for every one of the 28 features, it pre-sorts all the (up to) 10.5 million
examples by that feature and scans every adjacent pair as a candidate threshold. That is the right thing
to do if I need the *provably* optimal split, but I don't. I need a split that's good enough, found
fast. Two things to fix: the split search itself, and — while I'm rebuilding the learner — the way the
trees are scored, because the plain least-squares-on-the-gradient tree from the gradient machine throws
away information I'm already computing.

Start with the scoring, because it changes what the splits should optimize. The gradient machine fits
each tree to the *negative gradient* (first order) by squared error, then line-searches the leaves. But
for most losses I can cheaply get the *second* derivative too, and a second-order expansion gives a far
better per-node objective than squared error on the gradient. Write the objective for adding tree f_t to
the current model, with a regularization term Ω on the tree:

  L⁽ᵗ⁾ = Σᵢ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + f_t(xᵢ)) + Ω(f_t),   Ω(f) = γT + ½λ‖w‖²,

where T is the number of leaves and w the leaf weights — γ penalizes adding leaves, λ shrinks their
magnitudes. The regularization is new: the gradient machine controlled complexity only through tree
depth and shrinkage, but writing the leaf-magnitude penalty *into the objective* means the split score
itself knows about overfitting. Now Taylor-expand the loss to second order around the current
prediction. Let gᵢ = ∂_{ŷ}L(yᵢ, ŷ⁽ᵗ⁻¹⁾) and hᵢ = ∂²_{ŷ}L(yᵢ, ŷ⁽ᵗ⁻¹⁾) be the first and second
derivatives. Dropping the constant term:

  L̃⁽ᵗ⁾ = Σᵢ [ gᵢ f_t(xᵢ) + ½ hᵢ f_t(xᵢ)² ] + γT + ½λ Σⱼ wⱼ².

A tree assigns every example in leaf j the same value wⱼ, so group the sum by leaf with Gⱼ = Σ_{i∈Iⱼ}gᵢ
and Hⱼ = Σ_{i∈Iⱼ}hᵢ:

  L̃⁽ᵗ⁾ = Σⱼ [ Gⱼ wⱼ + ½(Hⱼ + λ) wⱼ² ] + γT.

This is a sum of independent quadratics in the wⱼ. Minimize each:

  wⱼ* = − Gⱼ / (Hⱼ + λ),

and plugging back gives the *structure score* of a fixed tree, L̃⁽ᵗ⁾ = −½ Σⱼ Gⱼ²/(Hⱼ+λ) + γT. So the
quality of a leaf is Gⱼ²/(Hⱼ+λ) — a clean, closed form using both gradient and Hessian sums. The gain of
splitting a node's example set I into I_L, I_R drops right out as the change in structure score:

  Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ.

That −γ is the cost of the extra leaf: a split only happens if its quality improvement exceeds γ, which
prunes weak splits automatically. This is the new split criterion, and it's all sums of g and h — so to
evaluate a candidate threshold I only need the running sums of gradients and Hessians on each side.

```cpp
template <typename TrainingParams, typename T>
XGBOOST_DEVICE T CalcGain(TrainingParams const &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight || sum_hess <= 0.0) return T(0.0);
  return common::Sqr(sum_grad) / (sum_hess + p.reg_lambda);   // G^2 / (H + lambda)
}
template <typename TrainingParams, typename T>
XGBOOST_DEVICE T CalcWeight(TrainingParams const &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight || sum_hess <= 0.0) return 0.0;
  return -sum_grad / (sum_hess + p.reg_lambda);               // w* = -G / (H + lambda)
}
```

Now the split search. The exact scan evaluates every distinct feature value as a threshold; that's where
the 3794.34 seconds go. I don't need every threshold — I need a small set of candidate thresholds that are
*representative* of the data's distribution, so that the best-of-candidates split is nearly as good as
the best-of-all. The classical tool is a quantile sketch: pick ~1/ε candidate split points so that each
bucket between consecutive candidates holds an equal *fraction* of the data. Then I only evaluate those
candidates, and the per-node cost collapses from "scan all n" to "scan the few hundred bucket
boundaries" once the per-bucket g and h sums are accumulated. Bin every feature once into these
candidate buckets — a histogram — and at each node just sum the gradients and Hessians falling into each
bin and scan the bins.

But "equal fraction of the data" is the wrong notion of representative here, and seeing why is the
crux. The objective I'm minimizing, rewritten, is

  L̃⁽ᵗ⁾ = Σᵢ ½ hᵢ ( f_t(xᵢ) − (−gᵢ/hᵢ) )² + const,

a *weighted* squared loss with per-example weight hᵢ. So an example's importance to the split objective
is its Hessian hᵢ, not 1. The candidate thresholds should therefore make each bucket hold an equal share
of the *Hessian mass*, not an equal count of rows. That is the **weighted quantile sketch**: define the
rank function

  r_k(z) = ( 1 / Σ h ) · Σ_{(x,h): x < z} h

— the fraction of total Hessian mass below z on feature k — and choose split candidates
{s_{k,1},…,s_{k,l}} so that |r_k(s_{k,j}) − r_k(s_{k,j+1})| < ε, giving roughly 1/ε candidates with
*equal Hessian* between them. Ordinary quantile sketches handle equal-count buckets but not arbitrary
per-point weights with a merge/prune accuracy guarantee, so this needs a new data structure — a
weighted sketch that supports merging summaries across data partitions and pruning them back to size
while bounding the rank error. With it, the proposal step costs almost nothing relative to the exact
scan, and the histogram of per-bin (g, h) sums replaces the full pre-sorted scan.

One more cost the exact scan hides: real tabular data is *sparse*. Higgs is dense, but most tabular
problems have many missing values, zero entries, and one-hot columns — and the exact scan wastes time
visiting all those zeros at every node. I can both speed this up and handle missingness principledly by
giving each node a learned **default direction**: enumerate only the *non-missing* entries of a feature
to find the split, and decide whether missing values go left or right by trying both and keeping the
direction with higher gain. The split search then iterates over non-missing entries only — its cost
becomes linear in the number of *present* values, not the full matrix — and missing values are routed by
a learned rule rather than imputed.

```
Algorithm (approximate, weighted-quantile split finding):
  for each feature k:
      propose candidates S_k = {s_{k,1},...,s_{k,l}} by the weighted quantile sketch
          (equal Hessian mass between consecutive candidates, ~1/eps of them)
  for each feature k:                       # one pass over data, accumulate per-bin sums
      G_{kv} = sum of g_j for j with s_{k,v-1} < x_jk <= s_{k,v}
      H_{kv} = sum of h_j for j with s_{k,v-1} < x_jk <= s_{k,v}
  scan the bins, pick the split maximizing  ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ
  (sparsity-aware: enumerate only non-missing entries; learn default direction for missing)
```

So the per-iteration work goes from "pre-sort and scan all 10.5M rows at every node" to "bin once into
Hessian-weighted buckets, then at each node sum gradients and Hessians into a few hundred bins and scan
those." The arithmetic per node drops by orders of magnitude on a dataset this size, and the second-order
regularized objective gives a sharper split score than least-squares-on-the-gradient, so accuracy should
hold or improve even as the time per iteration collapses.

This is **XGBoost**: a regularized second-order objective with the closed-form leaf weight −G/(H+λ) and
gain ½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]−γ, an approximate split finder driven by a *weighted*
quantile sketch that buckets by Hessian mass, and sparsity-aware split enumeration with learned default
directions. The expected payoff against the exact baseline's 3794.34 s/iter is a per-iteration time an
order of magnitude or more smaller, at matched-or-better Higgs AUC. The histogram is the lever that did
it — but I should notice what the histogram still does: at every node it sums gradients and Hessians
over *all* the data falling into the bins, every round, for every feature. The binning made each
threshold cheap, but the per-iteration cost is still proportional to (number of data points) × (number
of features). If I want to go faster still, the next thing to question is whether I really need *all* the
data and *all* the features in every histogram.
