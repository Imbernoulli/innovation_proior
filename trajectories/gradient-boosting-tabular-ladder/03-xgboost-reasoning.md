A single boosting iteration is costing 3794.34 seconds on Higgs. At 500 rounds the full run is
3794.34 × 500 ≈ 1.9×10⁶ seconds, about twenty-two days of wall-clock for one model at one hyperparameter
setting. That is not a tuning nuisance, it is a wall — it makes the exact splitter unusable at this scale. And I already counted the operations behind it:
roughly 10⁹–10¹⁰ threshold evaluations per iteration, each a gather through a pre-sorted index into
scattered memory. Dividing, 3794 s over ~2×10⁹ evaluations is on the order of a microsecond apiece,
which is exactly the signature of a memory-bound gather — a cache miss, not arithmetic. So the model is
fine; the gradient framework is general and the accuracy reference is strong (AUC 0.839593, and it is
*exact*, so nothing faster will beat that split, only approximate it). The exact split search is doing an
absurd amount of memory-bound work: at every node, for every one of the 28 features, it pre-sorts all
the (up to) 10.5 million examples by that feature and scans every adjacent pair as a candidate
threshold. That is the right thing to do if I need the *provably* optimal split, but I don't. I need a
split that's good enough, found fast. Two things to fix: the split search itself, and — while I'm
rebuilding the learner anyway — the way the trees are scored, because the plain
least-squares-on-the-gradient tree from the gradient machine throws away information I'm already
computing. That second point is the fork I deliberately set aside earlier: I noted then that carrying
the loss's *curvature* into split-finding was a genuine rebuild of the tree learner, worth deferring
until speed forced my hand. Speed has now forced my hand — I am rebuilding the split finder regardless —
so this is the moment to fold the second derivative in.

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

Two limits confirm the new objective *contains* the old one. Strip curvature and regularization — hᵢ = 1,
λ = 0, γ = 0 — and the leaf weight is wⱼ* = −Gⱼ/nⱼ, which for squared error is the mean residual, the
ordinary regression-tree leaf; the gain ½[G_L²/n_L + G_R²/n_R − G²/n] rewrites (with gᵢ = −rᵢ) as
½[n_L r̄_L² + n_R r̄_R² − n r̄²], exactly half the sum-of-squared-error reduction the exact GBM tree
splits on. So the second-order gain adds two things the variance-reduction criterion lacked: a Hessian
denominator that makes the score curvature-aware, and the λ, γ penalties. The second limit is the
logistic leaf: gᵢ = −(yᵢ − pᵢ), hᵢ = pᵢ(1−pᵢ) give wⱼ* = Σ(yᵢ − pᵢ)/(Σpᵢ(1−pᵢ) + λ), which at λ = 0 is
the per-leaf Newton step from before (the toy leaf's 0.4/0.62 = 0.645). The leaf value is the same Newton
step; the genuinely new thing is that the identical curvature Hⱼ now appears in the *gain* and steers
*which* splits get chosen, not just what value each leaf takes. That is the deferred fork, now paid for.

The three moving parts — the Hessian denominators, λ, and γ — act cleanly. On a candidate split with
(G_L, H_L) = (−1.2, 0.9) and (G_R, H_R) = (+0.8, 0.7), parent (−0.4, 1.6), at λ = 1 the leaf qualities
are 1.44/1.9 = 0.758, 0.64/1.7 = 0.376, and parent 0.16/2.6 = 0.062, so the gain is
½(0.758 + 0.376 − 0.062) − γ = 0.536 − γ: at γ = 0.5 it is +0.036 and the split is accepted, at γ = 0.6
it is −0.064 and rejected — γ is exactly the cost of an extra leaf. The leaf weight w_L* = 1.2/1.9 =
0.632 instead of the 1.2/0.9 = 1.333 it would be at λ = 0, and λ bites *hardest where H is smallest*: a
leaf built from confident, low-Hessian examples is one the model has least curvature information about, so
shrinking it most is the right regularization instinct. The same instinct explains the `min_child_weight`
guard — as H → 0 the gain estimate G²/(H+λ) becomes high-variance, computed from almost no Hessian mass,
so the guard refuses to split into a leaf the model cannot statistically support: a floor on curvature,
the second-order analogue of a minimum row count.

The L2 penalty ½λw² is the natural one because it keeps the leaf sub-problem a smooth quadratic with the
clean closed form, but nothing stops me from adding an L1 penalty α|w| to Ω as well, and it is worth
deriving what that does because it is not just "more shrinkage." Minimizing Gw + ½(H+λ)w² + α|w| over a
single leaf is a one-dimensional problem with a kink at w = 0. For w > 0 the stationary condition is
G + (H+λ)w + α = 0, giving w = −(G+α)/(H+λ), which is genuinely positive only when G < −α; for w < 0 it
is G + (H+λ)w − α = 0, giving w = −(G−α)/(H+λ), valid only when G > α; and for |G| ≤ α the subgradient of
α|w| straddles zero and the minimizer is exactly w* = 0. Collect the three cases and the optimal leaf is
w* = −S(G,α)/(H+λ) where S(G,α) is the *soft-threshold* — G − α if G > α, G + α if G < −α, and 0 in
between — which is precisely the `ThresholdL1` the real leaf-weight code applies before dividing by
(H+λ), and the gain then uses S(G,α)²/(H+λ) in place of G²/(H+λ). So the L1 term does something the L2
term cannot: it *zeroes out* whole leaves whose accumulated gradient magnitude |G| falls below α, pruning
them from the tree's contribution rather than merely shrinking them. L2 shrinks every leaf smoothly
toward zero; L1 sparsifies, killing the weakly-supported leaves outright. Both are now inside the split
score, so the criterion that chooses splits already accounts for the regularized value the resulting
leaves will actually take.

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
crux. Rewrite the objective by completing the square. Term by term, gᵢ f + ½ hᵢ f² = ½ hᵢ(f² +
2(gᵢ/hᵢ)f) = ½ hᵢ(f + gᵢ/hᵢ)² − ½ gᵢ²/hᵢ, and summing, the last piece is a constant independent of the
tree, so

  L̃⁽ᵗ⁾ = Σᵢ ½ hᵢ ( f_t(xᵢ) − (−gᵢ/hᵢ) )² + const,

a *weighted* squared loss with per-example weight hᵢ and target −gᵢ/hᵢ. So an example's importance to the
split objective is its Hessian hᵢ, not 1. The candidate thresholds should therefore make each bucket hold
an equal share of the *Hessian mass*, not an equal count of rows. Let me make the failure of equal-count
concrete so I am sure the distinction is real and not pedantic. Take ten examples on one feature, nine of
them with hᵢ = 0.01 (confident, already well-fit) and one with hᵢ = 1 (uncertain, right at the decision
boundary). The total Hessian mass is 9·0.01 + 1 = 1.09, and the single uncertain example carries
1/1.09 ≈ 92% of it. An equal-count sketch would split these into two buckets of five rows each, burying
the one example that owns 92% of the objective's weight in a bucket with four rows that own almost none —
the split boundary I most care about (the one next to the high-Hessian point) might not even be a
candidate. A Hessian-weighted sketch instead puts a candidate boundary essentially right at that point,
because that one example is nearly a whole bucket's worth of mass by itself. That is the **weighted
quantile sketch**: define the rank function

  r_k(z) = ( 1 / Σ h ) · Σ_{(x,h): x < z} h

— the fraction of total Hessian mass below z on feature k — and choose split candidates
{s_{k,1},…,s_{k,l}} so that |r_k(s_{k,j}) − r_k(s_{k,j+1})| < ε, giving roughly 1/ε candidates with
*equal Hessian* between them. Ordinary quantile sketches handle equal-count buckets but not arbitrary
per-point weights with a merge/prune accuracy guarantee, so this needs a new data structure — a
weighted sketch that supports merging summaries across data partitions and pruning them back to size
while bounding the rank error. The knob is ε: it fixes both the number of candidates, ~1/ε (choose
ε = 1/256 and I get about 256 split points per feature), and the guarantee — every proposed candidate's
Hessian-rank is within ε of a true weighted quantile, so the best split among the candidates is within
O(ε) of the exact best under the Hessian-weighted objective. The merge-and-prune property is not a
nicety here, it is a necessity of scale: I cannot globally sort 10.5M weighted rows per feature (that is
the exact scan I am fleeing), so I build small summaries on data shards, *merge* them with additive rank
error, and *prune* each merged summary back to 1/ε entries — the accuracy guarantee is what lets those
merges compose without the error blowing up across shards. With it, the proposal step costs almost
nothing relative to the exact scan, and the histogram of per-bin (g, h) sums replaces the full
pre-sorted scan.

One more cost the exact scan hides: real tabular data is *sparse*. Higgs is dense, but most tabular
problems have many missing values, zero entries, and one-hot columns — and the exact scan wastes time
visiting all those zeros at every node. I can both speed this up and handle missingness principledly by
giving each node a learned **default direction**: enumerate only the *non-missing* entries of a feature
to find the split, and decide whether missing values go left or right by trying both and keeping the
direction with higher gain. The split search then iterates over non-missing entries only — its cost
becomes linear in the number of *present* values, not the full matrix. Put a number on the win: a one-hot
column that is nonzero on 1% of rows costs the exact scan a pass over all n rows at every node, but costs
the sparsity-aware scan a pass over only the 1% present — a hundredfold reduction on that column — and
the missing 99% are routed by one learned bit — missing→left vs missing→right, chosen by comparing the
two gains — rather than imputed to a fake value that would distort the split. The missing rows carry their
own aggregate (G_0, H_0), computed once as the node total minus the present total, and the split evaluator
scores adding it to the left child versus the right and keeps the higher-gain direction (the one where the
missing rows' gradient sum reinforces the larger-magnitude leaf). That decision is two dot products
already sitting in registers from the present-value scan, so missingness costs essentially nothing and is
*learned* per node rather than imputed.

(The approximate weighted-quantile split-finding algorithm is written out in the answer.)

Let me size the speedup on paper so I know what to expect and can be shown wrong. The exact scan, per
feature per node, evaluates as many thresholds as there are distinct values in that node — up to n, so on
Higgs it is scanning on the order of millions of candidates per feature at the top nodes. The histogram
evaluates b − 1 ≈ 255 bins, a fixed small number regardless of n. So the *candidate-scan* work drops by a
factor of roughly (distinct values)/(255), which at the root is tens of thousands. But I must be honest
that this does not make the whole iteration tens-of-thousands-times faster, because accumulating the
per-bin sums still requires touching every row of the node once — that part stays O(n_node). What the
histogram actually removes is (a) the enormous candidate scan, replaced by a fixed 255-bin sweep, and (b)
the cache-hostile pre-sorted gather, replaced by *sequential* bin increments that stream through memory,
plus the binning is done once up front and reused across all nodes and rounds. Sequential streaming
versus scattered gathers is itself a large constant factor on a memory-bound workload. So my honest
estimate is a per-iteration speedup of *one to two orders of magnitude* — not the naive 10⁴, but far more
than a small constant — bringing 3794 s/iter down to somewhere in the low hundreds. The Higgs timing
table will pin the exact number; my falsifiable claim is that it lands one-to-two orders below 3794, and
that the test AUC does *not* fall — it should hold or edge *up*, because the second-order regularized
gain is a sharper criterion than least-squares-on-the-gradient and the λ, γ penalties fight the
overfitting the exact greedy splitter is prone to. Were the AUC to drop materially, that would say the
histogram approximation is too coarse at ε and I would need more bins; I don't expect that at 255.

So the per-iteration work goes from "pre-sort and scan all 10.5M rows at every node" to "bin once into
Hessian-weighted buckets, then at each node sum gradients and Hessians into a few hundred bins and scan
those." This is **XGBoost**: a regularized second-order objective with the closed-form leaf weight
−G/(H+λ) and gain ½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]−γ, an approximate split finder driven by a
*weighted* quantile sketch that buckets by Hessian mass, and sparsity-aware split enumeration with
learned default directions. The histogram is the lever that did it — but I should notice what the
histogram still does: at every node it sums gradients and Hessians over *all* the data falling into the
bins, every round, for every feature. The binning made each threshold cheap, but the per-iteration cost
is still proportional to (number of data points) × (number of features): it is exactly the O(n_node)
accumulation I admitted the histogram does not remove, multiplied across features. If I want to go faster
still, the next thing to question is whether I really need *all* the data and *all* the features in every
histogram — the two factors of that product are independent, and neither has been touched. Halving
either one, or better than halving it without paying accuracy, would drop the per-iteration cost again,
and there is no reason the histogram build must remain blind to the fact that most rows barely move the
gradient and many features are almost never on at the same time.
