Let me start from exactly what hurts. I have gradient boosting decision trees, and on the kind of data I now have to train on — millions of rows, and a feature space that runs from hundreds to literally tens of millions of columns — it is just too slow. I want to know precisely *where* the time goes before I touch anything, because if I optimize the wrong thing I waste the effort. A GBDT is an additive sequence of regression trees: at round t I have the current ensemble F_{t-1}, I compute for every instance a gradient of the loss against its current prediction, g_i = ∂L(y_i, F_{t-1}(x_i))/∂F_{t-1}(x_i), I fit a new tree to the negative gradients, I fold it in shrunk by a learning rate, repeat. The trees are tiny relative to the data; the gradient computation is one pass. The cost is in *growing each tree*, and inside that it is in *finding the best split at each node*. To split a node I need, for every feature, the gain of every candidate split, and to get those I have to look at the node's data. So per tree the work is on the order of #data × #feature. That product is the enemy. Everything else is noise next to it.

How do I find splits today? Two ways, and I should be honest about both. The pre-sorted way — sort each feature's values once, then enumerate every candidate split on the sorted order — gives me the exact optimum, but it carries the sorted index around, it is slow, and the memory is brutal: I am storing sorted permutations and maintaining them through every split. The histogram way is the one I actually want to build on: bucket each continuous feature into a small fixed number of bins, say 255 so a bin index fits in a single byte, and then for a node I make one pass over its rows accumulating, per bin, the sum of gradients and the count of instances. That gives me the node's histogram. To search splits I now walk the bin boundaries, of which there are at most #bin − 1, not all the sorted values. Building the histograms is O(#data × #feature); searching them is O(#bin × #feature). Since #bin (255) is far below #data (millions), the build dominates utterly — search is a rounding error. There is a lovely trick I get for free here: a parent node's histogram is exactly the sum of its two children's histograms, so I only build the histogram of the *smaller* child and recover the sibling by subtracting the small child from the parent, which costs O(#bin), not O(#data). That already halves a lot of the descent cost. But it does not change the headline: histogram building is O(#data × #feature), and that is what I am paying. So the lever is blunt and obvious — if I can reduce #data, or reduce #feature, *before* the build, I cut directly into the dominant term. The whole problem is that doing either without losing accuracy is not obvious at all. Let me take the two reductions one at a time.

Reducing #data first. The naive thing is to throw away rows. But which rows, and how, without changing what the trees learn? The boosting world already has the accuracy-preserving answer for *one kind* of booster: in AdaBoost each instance carries a sample weight that gets updated every round, and that weight is a genuine importance signal — there is a whole line of work that speeds up boosting by sampling instances according to that weight, filtering out the low-weight ones, adapting the ratio as training proceeds. That is exactly the shape of what I want: sample by importance, not uniformly, so I keep the instances that matter and drop the ones that do not. So let me just port that over to GBDT. And immediately I hit the wall: GBDT has *no native instance weight*. Each round hands me a gradient per instance and nothing else. There is no maintained per-instance importance to sample by. The AdaBoost machinery has no input here. Dead end as written.

The only row-subsampling that does survive into GBDT is stochastic gradient boosting — every round, take a uniform random subset of the data, fit that round's tree on it, done. It plainly reduces #data per round. But uniform is the problem: it treats every instance as equally informative for estimating the split gain, and the known failure mode is that a small uniform subset costs accuracy. So I am stuck between "the accurate samplers need a weight I don't have" and "the sampler I can use is uniform and hurts." I need an importance signal that GBDT actually produces.

So sit with what GBDT *does* hand me every round: a gradient g_i per instance. Is there importance hiding in it? Think about what a small gradient means. The gradient is ∂L/∂F at the current prediction; if |g_i| is small, the loss is nearly flat there, the model is already fitting that instance well — it is, in a real sense, already learned. If |g_i| is large, the instance is under-trained, the model is still wrong about it, there is error left to chase. So |g_i| *is* an importance signal, the very thing I said GBDT lacks — it lacks a *stored* weight, but the gradient magnitude is a perfectly good per-round importance, computed fresh every round, no bookkeeping. The straightforward move, then: discard the instances with small gradients, keep the ones with large gradients, fit the tree on those. That has to be cheaper and it focuses on the under-trained instances.

But wait — does dropping the small-gradient instances change *what I am estimating*? The split gain is

  V_j(d) = (1/n)[ (Σ_{x_i: x_ij≤d} g_i)² / n_l(d) + (Σ_{x_i: x_ij>d} g_i)² / n_r(d) ],

and the thing inside each square is a *sum of gradients over a side of the split*. If I just delete the small-gradient instances and compute this over what's left, I have changed the data distribution: every Σg over a side is now missing the contributions of all the dropped instances. The small gradients are individually small, yes, but there can be enormously many of them, and their sum is not negligible — it shifts both Σg terms and both counts n_l, n_r. So the gain I compute on the survivors is a *biased* estimate of the gain on the full data, and I will pick wrong splits. That is exactly the accuracy loss I was trying to avoid. Wall again, but a more informative one: the issue is not "drop the small gradients," it is "drop them and forget they ever existed." Their *mass* still has to be represented in the sum.

So keep all the large-gradient instances — there are few of them and each matters a lot — but for the small-gradient ones, instead of deleting them outright, sample a controlled number and scale the sample back up so it stands in for the whole tail. Concretely: sort by |g_i| descending; take the top a-fraction, call it A, and keep it whole. The remaining tail A^c has (1-a)n instances. Now sample b n instances from that tail, call the sample B, so inside A^c the inclusion fraction is b/(1-a). The expected raw gradient sum over B is therefore b/(1-a) times the full tail sum. To make B represent A^c in the split gain, the multiplier is forced: each sampled tail gradient gets weight (1-a)/b. Then the reweighted tail sum has expectation Σ_{A^c} g, while A still enters with weight 1 because it was never sampled. Now write the estimator:

  Ṽ_j(d) = (1/n)[ (Σ_{A_l} g_i + (1−a)/b · Σ_{B_l} g_i)² / n_l(d)
                + (Σ_{A_r} g_i + (1−a)/b · Σ_{B_r} g_i)² / n_r(d) ],

with A_l = {x_i∈A: x_ij≤d}, A_r = {x_i∈A: x_ij>d}, B_l, B_r the same for B. Sanity check the degenerate case a=0: then the tail is the whole dataset, I sample b n instances and weight them by 1/b, exactly the rescaled uniform sampler. So a=0 recovers stochastic gradient boosting. The novelty is purely in a>0: deterministically retain the large-gradient instances, sample only b n of the small-gradient tail, and use (1-a)/b to put the tail's mass back into the gradient sums.

Now I have to actually prove this doesn't lose much accuracy, because "roughly intact" is hand-waving and I burned myself once already by under-counting the tail. Let me bound the error E(d) = |Ṽ_j(d) − V_j(d)| between the GOSS gain and the true gain on a fixed split d. Both gains already include the common 1/n factor, so for the algebra I strip it off by writing U_j(d) = n·V_j(d) and Ũ_j(d) = n·Ṽ_j(d), bound |Ũ_j(d) − U_j(d)|, and divide by n at the end. I'll write D_{A^c} = max_{x_i∈A^c}|g_i|, the largest gradient magnitude in the tail, which is the residual scale that survives my keeping the top set.

Start with the difference of the unnormalized scores U. Each is a sum of two terms of the form (sum of gradients)²/(count). The counts n_l(d), n_r(d) are the same in both, so for each side the difference of the two squared-sum terms is a difference of squares, S̃² − S², which factors as (S̃ − S)(S̃ + S). For the left side, S̃_l − S_l = (1−a)/b · Σ_{B_l} g_i − Σ_{A^c_l} g_i, the gap between my rescaled tail sample and the true tail sum on the left, and S̃_l + S_l is a sum of the two gradient sums. So

  Ũ_j(d) − U_j(d)
    = C_l·( (1−a)/b · Σ_{B_l} g_i − Σ_{A^c_l} g_i )
    + C_r·( (1−a)/b · Σ_{B_r} g_i − Σ_{A^c_r} g_i ),

where C_l collects the (S̃_l + S_l)/n_l(d) factor — explicitly

  C_l = ( (1−a)/b · Σ_{B_l} g_i + Σ_{A^c_l} g_i + 2 Σ_{A_l} g_i ) / n_l(d),

and C_r the analogous right-side coefficient. (The 2Σ_{A_l} appears because the kept top set A contributes the same to both S̃_l and S_l, so it shows up doubled in the sum S̃_l + S_l.) Taking absolute values and pulling out the larger absolute coefficient,

  |Ũ_j(d) − U_j(d)| ≤ max{|C_l|, |C_r|} · ( | (1−a)/b · Σ_{B_l} g_i − Σ_{A^c_l} g_i |
                                                  + | (1−a)/b · Σ_{B_r} g_i − Σ_{A^c_r} g_i | ),

so the work is the same on each side: bound the coefficient max{|C_l|, |C_r|}, and bound the rescaled tail sampling error in a gradient sum. The two sides only cost harmless constant factors, which I can absorb into the Hoeffding logarithms.

Bound |C_l| first. Every gradient appearing in it is in absolute value at most D_{A^c} for the tail parts; the top-set part Σ_{A_l} g, divided by the count, is at most the average gradient on the left, which I'll call into the term D below. Pulling D_{A^c} out of the sample/tail indicator sums,

  |C_l| ≤ D_{A^c}/n_l(d) · | (1−a)/b · Σ 1[x_i∈B_l] − Σ 1[x_i∈A^c_l] | + 2D,

where the leftover 2D collects the contribution of the kept top set normalized by the count (the 2Σ_{A_l}g/n_l piece), and D = max(ḡ_l(d), ḡ_r(d)) with ḡ_l(d) = Σ_{x_i∈(A∪A^c)_l}|g_i| / n_l(d), the average absolute gradient on the left over the full node — the natural scale of a side's gradient. Rewrite the indicator difference in terms of empirical means: B_l comes from selecting b n instances out of the (1-a)n tail, so (1/(bn))Σ1[x_i∈B_l] and (1/((1-a)n))Σ1[x_i∈A^c_l] are both estimating the same left-side proportion of the tail, and their difference is a deviation of an empirical mean from its expectation. That is precisely a Hoeffding situation: B is sampled without replacement / uniformly from A^c, the indicator is bounded in [0,1], so with probability at least 1−δ,

  |C_l| ≤ D_{A^c}(1−a)n / n_l(d) · √( ln(2/δ) / (2bn) ) + 2D,

and identically |C_r| ≤ D_{A^c}(1−a)n / n_r(d) · √(ln(2/δ)/(2bn)) + 2D. The √(1/(bn)) is the sampling-noise scale: bigger sample b, or more data n, tighter.

Now the tail sampling error in the gradient sum, the second factor. For either side s∈{l,r}, (1−a)/b·Σ_{B_s} g rescales the selected tail instances on that side back to tail scale; its gap from the true Σ_{A^c_s} g is again an empirical-mean deviation, the summand bounded by D_{A^c}, so Hoeffding gives, with probability at least 1−δ,

  (1/n) | (1−a)/b · Σ_{B_s} g_i − Σ_{A^c_s} g_i | ≤ D_{A^c}(1−a) · √( ln(2/δ) / (2bn) ).

Since E(d) = |Ũ_j(d) − U_j(d)|/n, multiply the coefficient bound by the normalized side-gradient-error bounds and absorb the constant from the two sides into the logarithmic constants. The cross terms land in two pieces: the 2D part of C gives 2D · C_{a,b} · √(ln(1/δ)/n), and the sampling-deviation part of C gives C²_{a,b} · ln(1/δ) · max{1/n_l(d), 1/n_r(d)}, after the harmless Hoeffding constants are absorbed. Define the single constant that carries both,

  C_{a,b} = (1−a)/√b · max_{x_i∈A^c} |g_i| = (1−a)/√b · D_{A^c},

and the whole thing reads, with probability at least 1−δ,

  E(d) ≤ C²_{a,b} · ln(1/δ) · max{ 1/n_l(d), 1/n_r(d) } + 2D · C_{a,b} · √( ln(1/δ)/n ).

There it is, clean, and now I read off what it is telling me. Asymptotically the rate is O(1/n_l + 1/n_r + 1/√n). If the split is not pathologically unbalanced — both sides have at least O(√n) instances — then 1/n_l and 1/n_r are each O(1/√n) too, so the *whole* bound is O(1/√n) and the approximation error vanishes as n grows. That is the crucial qualification I have to respect: **GOSS is accurate precisely on large data, where 1/√n is small.** On a small dataset the bound is loose and I should not trust the sampling — which fits intuition, sampling needs volume. Good thing the entire motivation is big data.

And does GOSS actually beat plain random sampling, or did I just dress up SGB? Take a=0 as the random-sampling reference and ask when GOSS at general a does better at the same total sampling budget β. Random sampling uses C_{0,β}. GOSS keeps a·n large-gradient instances and samples (β−a)n from the tail, so it uses C_{a,β−a}. The condition is exactly C_{0,β} > C_{a,β−a}, i.e.

  α_a / √β > (1−a) / √(β−a),

where α_a = max_{A∪A^c}|g_i| / max_{A^c}|g_i|. Squaring and rearranging — α_a²(β−a) > (1−a)²β, so β(α_a² − (1−a)²) > α_a²a — gives β > a / (1 − ((1−a)/α_a)²). This is the precise win condition. It is easier to satisfy when the gradient magnitudes have a wide range, because pulling the top a-fraction into A sharply lowers max_{A^c}|g_i| and raises α_a; when the gradient magnitudes are nearly flat, α_a is close to 1 and the advantage over uniform sampling nearly disappears. That settles the row reduction: keep top-a by |g|, sample b·n of the rest, reweight the sample by (1−a)/b.

Let me also note what this does to generalization, just to be sure I'm not only fitting the training gain. The generalization error of GOSS, the gap between my sampled gain and the *true* gain under the data distribution, splits by triangle inequality into the sampling error I just bounded plus the ordinary full-data generalization gap; if my sampling approximation is accurate (which the theorem gives on large n), the GOSS generalization is close to the full-data generalization. Sampling also adds diversity across the base learners from round to round, which can potentially help the ensemble generalize. So nothing is broken on that front.

Now the second reduction: #feature. The lever is the same — cut the count that multiplies #data in the histogram build — but the row trick doesn't transfer, features aren't sampled the way rows are. The classical move to reduce features is to filter or project out the weak ones, PCA or projection pursuit. But that leans on the features being redundant, and in real engineered tabular data each feature is usually built to carry its own signal; drop any and you can lose accuracy. So I don't want to *remove* features. I want to *pack* them.

What do I actually know about these feature spaces? They are sparse. Massively. One-hot encodings, indicator features — most entries zero. And here is the structural fact that sparsity hands me: in a sparse space, many features are *mutually exclusive* — they essentially never take nonzero values on the same instance. The clearest case is a one-hot block: exactly one of the columns is nonzero per row, the rest are zero, so any two columns in the block are exclusive by construction. If two features are never simultaneously nonzero, then on every instance at most one of them carries information, so I could *store them in the same column* without collision — a nonzero in that column came from one or the other, unambiguously, as long as I can tell which. If I can pack a set of mutually exclusive features into one "bundle" column, the histogram build sees one column instead of many, and the build cost goes from O(#data × #feature) to O(#data × #bundle) with #bundle ≪ #feature. That's the whole prize, and crucially it loses *nothing*: I'm not discarding any feature's values, just relocating them so the zeros don't each cost a column.

Two questions to make this real. First, *which* features go in a bundle together. Second, *how* to merge them into one column so the original feature each value came from is recoverable.

Take "which" first. I want to partition the features into the fewest exclusive bundles. Is that easy? Let me look at the structure. Build a graph: one vertex per feature, and an edge between two features whenever they are not mutually exclusive, meaning they collide somewhere. An exclusive bundle is then a set of vertices with no edges among them, an independent set, and I want to cover all vertices with the fewest such sets. That is graph coloring: assigning each vertex a color so adjacent vertices differ, minimizing colors, where each color class is exactly an exclusive bundle. And graph coloring is NP-hard. So partitioning features into the smallest number of exclusive bundles is NP-hard too. The reduction is direct. Given any graph G = (V,E), create one feature for every vertex and one training instance for every edge. Set feature v nonzero on instance e exactly when vertex v is incident to edge e. Then two features are nonzero together on some instance exactly when their vertices share an edge in G, so non-exclusivity is exactly adjacency. A valid exclusive bundle is a color class, a valid coloring is a bundling, and an optimal bundling would give an optimal coloring. Since graph coloring is NP-hard, exact optimal bundling is NP-hard. No polynomial exact algorithm; I need an approximation.

A greedy graph-coloring-style algorithm is the natural choice and it has a good constant-factor approximation. Construct the graph with *weighted* edges, where each edge's weight is the number of conflicts between the two features (rows where both are nonzero) — because I'm going to allow a little conflict, weights matter. Sort features by their degree in the graph, descending — high-degree, conflict-prone features first, since they are the hardest to place and should be placed while bundles are empty. Then go down the ordered list: for each feature, try to drop it into an existing bundle if doing so keeps that bundle's total conflict under a budget; if none fits, open a new bundle. This is O(#feature²) to build the graph and runs once before any training, which is fine unless #feature is in the millions — and then even O(#feature²) hurts, so I'll keep a cheaper ordering in my pocket: order by the count of nonzero values instead of building the graph at all, since more nonzeros means more chances to conflict, so nonzero-count is a good proxy for degree. Same greedy loop, no graph.

Now the part I slipped in — "allow a little conflict." Demanding *perfect* exclusivity is wasteful: there are usually features that aren't 100% exclusive but conflict only rarely, and forcing them into separate bundles inflates #bundle for almost no reason. If I let a bundle absorb a feature that conflicts on a small fraction γ of instances, I get far fewer bundles. What does that cost in accuracy? When two features collide in a bundled column, I've effectively corrupted that column's value on the colliding instances — it's as if I randomly polluted a γ-fraction of the feature values. So I need to bound how much randomly polluting a γ-fraction of values can move the best split's gain. Let V be the maximum variance gain over all features on clean data, achieved at feature j_1; let V^γ be the maximum gain after γ-pollution, achieved at feature j_2. I want |V − V^γ| small. Since V = V_{j_1} ≥ V_{j_2} and V^γ = V^γ_{j_2} ≥ V^γ_{j_1}, I can sandwich:

  V − V^γ = V_{j_1} − V^γ_{j_2} ≤ V_{j_1} − V^γ_{j_1}   (replacing V^γ_{j_2} by the smaller V^γ_{j_1}),
  V^γ − V = V^γ_{j_2} − V_{j_1} ≤ V^γ_{j_2} − V_{j_2}   (replacing V_{j_1} by the smaller V_{j_2}),

so |V − V^γ| ≤ max over a single feature of |its clean gain minus its polluted gain|. And for a *single* feature, perturbing a γ-fraction of its values is a small perturbation of where its optimal split sits; using the split-point perturbation results for variance-gain estimation, a single feature's gain changes by at most [(1−γ)n]^{−2/3} under γ-pollution. So

  |V − V^γ| ≤ [(1−γ)n]^{−2/3}.

That decays in n but worsens as γ grows — since (1−γ)n shrinks as γ rises, [(1−γ)n]^{−2/3} climbs — so for a small γ on large n the accuracy hit is negligible while #bundle drops. Concretely: γ small, like a fraction of a percent, gives almost the baseline accuracy with markedly fewer bundles. If γ = 0 I demand pure exclusivity and bundle only the genuinely exclusive features, still a big win on one-hot-heavy data, just not the absolute minimum. So the budget K on a bundle's allowed conflict is K = γ·n, and the greedy loop adds a feature to a bundle only if the bundle's conflict count stays ≤ K.

Now "how to merge." This is where the histogram representation pays off again. Because the histogram stores *discrete bins* per feature, not continuous values, I can lay exclusive features into *disjoint bin ranges* inside one bundle and a single bin index tells me both which original feature fired and its value. Walk the bundle's features keeping a running offset: feature A originally takes bins [0,10); I assign offset 0, it stays [0,10). Feature B takes bins [0,20); I assign offset 10, so B's values become [10,30). Now in the merged column, a value in [0,10) means "A, this bin," a value in [10,30) means "B, this bin minus 10," and a zero means neither fired (exclusivity guarantees at most one is nonzero, so there's no collision to resolve, modulo the γ-fraction I deliberately allowed). Building the merged column: for each instance, scan the bundle's features, and if one is nonzero, write its bin plus that feature's offset; otherwise write 0. The bundle's histogram over this merged column is then exactly the concatenation of the individual features' histograms over their disjoint sub-ranges, so split search on the bundle recovers every split that split search on the individual features would have found — the gain is preserved, the build is on #bundle columns. That's EFB: bundle exclusive features by greedy coloring, merge by bin offsets.

There's a complementary sparse optimization I should keep, because it composes with this: I can make the basic histogram build skip zeros by keeping, per feature, a table of the instances with nonzero values and scanning only those — dropping a feature's build from O(#data) to O(#non-zero-data). It costs extra memory to maintain those tables through tree growth, but it doesn't conflict with EFB — I can still apply it inside a bundle when the bundle is itself sparse. So I keep it as a basic function and let EFB do the heavy lifting of cutting the column count.

One more thing the bundling buys me that I didn't ask for: cache locality. Features that were scattered far apart in memory are now adjacent inside one bundle column, so scanning a bundle touches contiguous memory, the cache hit rate goes up, and the histogram build gets faster beyond the bare #feature → #bundle count reduction. A free constant-factor on top.

Let me also settle the tree growth itself, since it's the third place where the cost concentrates and I've been assuming histograms. I'll grow leaf-wise — best-first — splitting at each step the single leaf whose split yields the largest loss reduction, rather than splitting all leaves of a depth level. Why: for a fixed budget of leaves, leaf-wise always spends the next split where it buys the most loss reduction, so it reaches a lower training loss than level-wise at the same number of leaves. Level-wise wastes splits on leaves that barely help just to keep the tree balanced. The cost of leaf-wise is that the trees get deep and asymmetric and can overfit on small data, so I cap them — a maximum number of leaves and a maximum depth — to bound that. With the histogram subtraction trick, leaf-wise is also natural to implement cheaply: split a leaf, build the histogram of the smaller child, subtract for the sibling, push both onto the priority queue keyed by their best gain.

While I'm at the split criterion, I should generalize the variance gain slightly to match what a regularized boosting objective actually wants, because the deployment will use it. The first-order variance gain sums gradients; but if I take a second-order view of the loss, each leaf's optimal value is a Newton step −G/(H+λ), where G is the sum of gradients and H the sum of second derivatives (Hessians) in the leaf, λ an L2 penalty on leaf weights, and the split gain becomes ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] minus a per-leaf complexity charge. For the squared loss H_i = 1 and this reduces back to the variance gain I analyzed for GOSS — so GOSS's accumulation of per-bin gradient sums just extends to accumulating per-bin (G, H) pairs, and the analysis carries over with g read as the first-order gradient. An L1 penalty λ1 enters as a soft-threshold on G in the leaf-value numerator. Categorical features I won't one-hot — that explodes into 2^{k−1}−1 partitions; instead I sort the category bins by their G/H (the leaf-value proxy) and search the ordered list for the best ordered partition, O(k log k). These are the standard knobs the histogram engine exposes.

So put the row reduction into the boosting loop as its own pass and let the feature reduction plus histogram-with-subtraction-and-leaf-wise be the split-finder it calls. The round looks like: predict on the full data, compute gradients and the importance |g|, keep the large-gradient set and the reweighted tail sample as the working set, then fit the tree on that set with the bundled, histogram split finder. Precompute the per-round constant fact = (1−a)/b once.

  Inputs: I (training data), d iterations, a (top rate), b (other rate), loss, weak learner L.
  models ← {};  fact ← (1−a)/b;  topN ← a·|I|;  randN ← b·|I|
  for i = 1 to d:
      preds ← models.predict(I)
      g ← loss.gradient(I, preds);  w ← {1,1,…,1}          # full-data gradients; unit weights
      sorted ← indices of I sorted by |g| descending
      topSet  ← sorted[1 : topN]                            # keep all large-gradient instances
      randSet ← random pick of randN from sorted[topN+1 : |I|]   # sample the small-gradient tail
      usedSet ← topSet ∪ randSet
      w[randSet] ← w[randSet] · fact                        # reweight the tail sample by (1−a)/b
      newModel ← L(I[usedSet], −g[usedSet], w[usedSet])     # fit a tree to −gradient on the working set
      models.append(newModel)

and the bundled, histogram-based, leaf-wise split finder it calls:

  # once, before training: bundle exclusive features (greedy graph coloring with conflict budget K = γ·n),
  #                        merge each bundle into one column by bin offsets (disjoint sub-ranges)
  # per node: build the histogram over the BUNDLE columns (sum of (g[,h]) per bin),
  #           use parent − smaller-child subtraction to get the sibling histogram in O(#bin),
  #           search bin boundaries for the max gain
  #           V(d) = ( (Σ g_left)² / n_left + (Σ g_right)² / n_right ) / n_node     # (second-order: ½ Σ G²/(H+λ) − γ·T)
  # grow leaf-wise: split the leaf with the largest gain; cap by num_leaves and max_depth

That is the whole method: GOSS cuts the rows the build sees, EFB cuts the columns it builds over, the histogram-with-subtraction and leaf-wise growth make each build and each split cheap, and the variance/second-order gain ties the leaf values back to the boosting objective.

Now, the form I actually ship sits inside a stock training harness: I hand the data and the parameters to the GBDT library that implements the histogram engine and let it train. The model fills the wrapper contract from the scaffold: it builds the library's `Dataset` objects from the train/valid feature matrices and 1-D labels, trains for up to num_boost_round rounds with early stopping on the validation set, and predicts by mapping each instance through the learned trees. For this deployment the objective is mean squared error, so the gradient is g_i = yhat_i - y_i and the tree fits the residual y_i - yhat_i, the classical residual-fitting form. The regularization knobs now have concrete settings: a learning-rate shrinkage on each tree, L1 and L2 penalties on the leaf weights, a cap on the number of leaves and the depth, and row- and column-sampling parameters. With the default GBDT booster, this wrapper does not switch on GOSS; it forwards `subsample` to the engine's bagging machinery and relies on the engine for the histogram, leaf-wise, and bundled-feature machinery. Filling the harness in the qlib shape:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.interpret.base import LightGBMFInt
from qlib.workflow import R


class LGBModel(ModelFT, LightGBMFInt):
    """qlib-style wrapper around the histogram-based, leaf-wise GBDT engine."""

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {
            "objective": loss,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1,
        }
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        ds_l = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                x, y = df["feature"], df["label"]
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w, free_raw_data=False), key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **kwargs,
        )
        for k in names:
            for key, val in evals_result[k].items():
                name = f"{key}.{k}"
                for epoch, metric in enumerate(val):
                    R.log_metrics(**{name.replace("@", "_"): metric}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)
```

Let me retrace the causal chain to be sure nothing is bolted on. The cost of GBDT is the histogram build, O(#data × #feature), so the only way to go materially faster is to cut #data or #feature before the build. Cutting #data needs an importance signal to sample by, and AdaBoost-style weight sampling can't port because GBDT has no native weight, but the gradient magnitude |g| is a per-round importance, large where the instance is under-trained, so keep the top-a by |g| whole and sample b·n instances from the small-gradient tail. Deleting the tail outright biases the gradient-sum-based gain, so rescale the sampled tail by (1−a)/b to put its mass back, which is GOSS, with a=0 recovering plain random sampling. The bound E(d) ≤ C²ln(1/δ)·max{1/n_l,1/n_r} + 2D·C·√(ln(1/δ)/n) with C = (1−a)/√b · max_{A^c}|g| shows the error is O(1/√n) on balanced splits, and the exact random-sampling comparison is C_{0,β} > C_{a,β−a}, equivalently α_a/√β > (1−a)/√(β−a). Cutting #feature cannot simply drop features, because there is no guaranteed redundancy, but sparse spaces have mutually exclusive features that can be packed into one column for free; choosing which to bundle is graph coloring, hence NP-hard, so a greedy degree-ordered coloring with a small conflict budget γ does it, and the pollution cost is at most [(1−γ)n]^{−2/3}. Merging by disjoint bin-offset ranges preserves every split. Histogram-with-subtraction and leaf-wise growth make each build and split cheap, and the variance / second-order gain ties leaf values to the boosting objective. The shipped form hands the data to the qlib-style wrapper around the GBDT engine: build `lgb.Dataset` objects, train with `lgb.train` under mse and the deployment parameters, and return predictions as a `pd.Series`.
