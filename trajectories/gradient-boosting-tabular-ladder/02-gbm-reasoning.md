The adaptive reweighting works, but it is wired to a single thing: a 0/1 "right or wrong" indicator on
a classification label. Each round I crush the weight of correctly-classified examples and lift the
misclassified ones, which — as the previous rung's own bookkeeping showed — is exactly stagewise
descent of the exponential loss exp(−yF) of the margin. That is one loss, and the margin is a *sign*: it
knows whether the committee lands on the right side of the label, never how far a continuous prediction
sits from a continuous target. If I want to predict a real-valued quantity, or fit a Huber loss robust
to outliers, or a Poisson loss for counts, the reweighting trick has no handle. The committee only ever
hears "this example is wrong," never "this example's prediction is off by +3.2 and the loss there is
steep." I want to keep the boosting structure — an additive model built one tree at a time, each tree
attacking what the committee gets wrong — but replace the classification-specific reweighting with
something that works for *any* differentiable loss.

Let me be honest about the routes before I pick one, because "generalize the reweighting" could mean
several things. The first temptation is to keep the reweighting machinery and just redefine "wrong" for
a regression target — say, weight example i by how large |yᵢ − F(xᵢ)| is. But that immediately runs
into the same wall that motivated the move: a weight is a scalar multiplier on an example, and it can
only tell the base learner *which* examples to care about, not *which direction* to move each one. For
classification the direction is implicit (flip toward the correct label), but for a continuous target
"attend to this example" is not enough — I need to tell the tree that example i's prediction should go
*up* by some amount and example j's should go *down*. A reweighting cannot carry a sign-and-magnitude
per example. So patching the weights is a dead end; I need a per-example *target*, not a per-example
importance.

There is a second route that is more than a straw man, because I know a whole line of work is pushing
exactly it: keep the reweighting frame but stop pretending one loss fits all, and instead *derive, for
each specific loss, its own bespoke reweighting* by treating the additive fit as a Newton step on that
loss. Take the logistic (binomial) loss as the worked case, because it is the one this route handles
most cleanly. Model the log-odds F(x) and write p = σ(F). A second-order expansion of the deviance at
the current F says the next increment should solve a *weighted least squares*: fit the base learner to
the working response zᵢ = (yᵢ − pᵢ)/(pᵢ(1−pᵢ)) with per-example weight wᵢ = pᵢ(1−pᵢ). This is iteratively
reweighted least squares dressed as boosting — the weight pᵢ(1−pᵢ) is largest for the uncertain
examples near p = ½ and vanishes for the confident ones, and the working response points each example
toward its label scaled by its local curvature. It is a real, coherent algorithm and it is genuinely
better than AdaBoost's exponential loss for noisy data, because pᵢ(1−pᵢ) is bounded while exp(−yF)
explodes on the examples the committee is most wrong about. So why not just adopt it? Two reasons, and
the first is decisive. It is *bespoke to one loss*. The working response and the weight I wrote are the
logistic loss's, hand-derived; for absolute error, or Huber, or Poisson, or a ranking loss, I would have
to sit down and derive a *new* working response and a *new* weight function each time, and some of those
losses do not even have the clean IRLS form the logistic one does. I would be maintaining a zoo of
loss-specific reweightings with no single mechanism that generates them. The second reason I will only be
able to state precisely once I have the general frame in hand, but I can name it now: this route pushes
the loss's *curvature* into the base learner's split-finding — the weights wᵢ change which splits look
good, not just what value each leaf takes — and that is a heavier rebuild of the tree learner than I want
to commit to before I even have the general recipe. I will come back and settle the curvature question
deliberately. For now I want the *one* mechanism that retargets to any differentiable loss by changing a
single formula, and neither patching-the-weights nor a per-loss IRLS gives me that.

Step back and write what I am actually building. The model is an additive expansion

  F(x) = Σₘ h_m(x),

built greedily: having F_{m−1}, I add one more term to reduce the total loss Σᵢ L(yᵢ, F(xᵢ)). The exact
greedy step would be h_m = argmin_h Σᵢ L(yᵢ, F_{m−1}(xᵢ) + h(xᵢ)) — a joint minimization over a whole
function, intractable for a general loss and a tree class. I need an approximation, and here is the
reframing that unlocks it. Forget for a moment that F is parameterized by trees and think of the
prediction *vector* — the n numbers F(x₁),…,F(xₙ) — as a single point in ℝⁿ. The total loss Φ = Σᵢ
L(yᵢ, F(xᵢ)) is then just a scalar function on ℝⁿ, and I want to decrease it. The direction of steepest
descent of Φ at the current point is the negative gradient, whose i-th component is

  gⁱ_m = −[ ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) ]_{F = F_{m−1}}.

If I could move each prediction independently by a small step along −∇Φ, I would reduce the loss — plain
gradient descent in ℝⁿ. But I cannot move the F(xᵢ) independently; the only thing I am allowed to add is
a single tree h_m(x), so I am constrained to move along directions a tree can produce. The negative
gradient lives in the unconstrained ℝⁿ, and I have to *project* it onto the subspace of functions my
base learner can represent.

The projection is the whole idea. The negative-gradient vector (g¹_m,…,gⁿ_m) is the ideal per-example
nudge — the **pseudo-residual**, the amount each prediction "wants" to move to lower the loss fastest —
and it carries the sign-and-magnitude a reweighting could not. I fit a regression tree to *them*, by
least squares:

  h_m = argmin_h Σᵢ ( gⁱ_m − h(xᵢ) )².

Why least squares specifically? Because the space of functions the tree can realize is a subspace (a
tree is a piecewise-constant function on axis-aligned regions), and the closest point of that subspace
to the target vector in ordinary Euclidean distance *is* the least-squares fit — the orthogonal
projection. So squared error is not an arbitrary choice of tree criterion here; it is the metric under
which "closest realizable approximation to the steepest-descent direction" means what I want it to mean.
And the loss has now vanished from the base learner's job entirely: the tree always solves a plain
squared-error regression, no matter what L is. *All* the loss-specific information has been pushed into a
single place — the formula for the pseudo-residual gⁱ_m. Swap the loss, swap one gradient expression,
and the entire machine retargets. That is exactly the generality AdaBoost lacked, and it is the single
mechanism the per-loss IRLS route could not give me: the logistic case, the absolute-error case, the
Poisson case are now *the same code* with one line changed.

Check that AdaBoost is the special case, to be sure I have not thrown anything away. Take L to be the
exponential loss L(y, F) = exp(−yF) with y ∈ {−1,+1}. Its negative gradient at F_{m−1} is
gⁱ_m = yᵢ·exp(−yᵢ F_{m−1}(xᵢ)) — the sign yᵢ points each example toward its label, and the magnitude
exp(−yᵢ F_{m−1}(xᵢ)) is large exactly on the examples the current margin gets wrong. That magnitude *is*
the AdaBoost weight the previous rung derived (it showed the sample weight entering a round is
proportional to exp(−yᵢ F(xᵢ))). So gradient boosting on the exponential loss reproduces AdaBoost's
reweighting as a byproduct of the gradient, and the new framework strictly contains it while reaching
past it to squared error, absolute error, Huber, logistic deviance, Poisson — anything with a gradient.

Two refinements the raw idea needs. First, fitting the tree to the pseudo-residuals fixes the tree's
*partition* of the input space (its leaves R_{jm}), but the least-squares fit would set each leaf's
output to the mean pseudo-residual, which is only optimal if L is squared error. For a general loss I
can do better: once the leaf regions are fixed, choose each leaf's output by an exact line search *in
the loss itself*,

  γ_{jm} = argmin_γ Σ_{xᵢ ∈ R_{jm}} L(yᵢ, F_{m−1}(xᵢ) + γ),

a separate one-dimensional optimization per leaf. Let me actually solve it for a few losses, with real
numbers, so I know what the line search does rather than trusting the shape of the formula. For squared
error L = ½(y − F)², the derivative in γ is Σ_{R}(yᵢ − F − γ) = 0, so γ_{jm} = mean of the residuals in
the leaf — and since the pseudo-residual for squared error is itself yᵢ − F, the whole thing collapses to
"fit residuals, leaf = mean residual." Trace it on three points y = (3, 5, 4), F_{m−1} = (2, 2, 2): the
pseudo-residuals are (1, 3, 2), a single leaf takes their mean γ = 2, and F becomes (4, 4, 4) — exactly
the leaf mean of y, ordinary regression-tree boosting, no surprises. For absolute error L = |y − F| the
pseudo-residual is sign(yᵢ − F) (so the tree structure is fit to ±1 signs), while the line search
Σ_{R}|yᵢ − F − γ| is minimized at the *median* residual. Here the two-stage split earns its keep: take
y = (1, 2, 3, 100), F = (0, 0, 0, 0). The pseudo-residuals are all +1 — the tree cannot even see that
100 is an outlier, every example just says "go up." The line search takes the median of the raw
residuals (1, 2, 3, 100), which is 2.5, not their mean 26.5. So the leaf steps up by 2.5 and the single
gross outlier does not drag the whole leaf by twenty-six; the structure comes from the signs, the
magnitude from the median, and that is precisely why absolute-error boosting is robust. For logistic
(binomial) deviance the pseudo-residual is yᵢ − pᵢ with pᵢ = σ(F_{m−1}(xᵢ)), and the leaf minimization
has no closed form, so I take a single Newton step,

  γ_{jm} = Σ_{R}(yᵢ − pᵢ) / Σ_{R} pᵢ(1 − pᵢ),

numerator the sum of gradients, denominator the sum of local curvatures pᵢ(1−pᵢ). Put numbers on it: a
leaf with three examples at current probabilities p = (0.5, 0.8, 0.3) and labels y = (1, 0, 1) has
gradients yᵢ − pᵢ = (0.5, −0.8, 0.7) summing to 0.4 and curvatures pᵢ(1−pᵢ) = (0.25, 0.16, 0.21)
summing to 0.62, so the Newton leaf is γ = 0.4/0.62 = 0.645. The naive least-squares leaf — the mean
pseudo-residual — would be 0.4/3 = 0.133, nearly a factor of five smaller: the plain fit badly
undershoots because it ignores that these examples sit on a shallow part of the logistic curve where a
larger step is warranted. So the structure is found by least squares on the first-order pseudo-residual,
but the leaf magnitude quietly uses the second derivative *locally, per leaf* — and I notice this is
exactly the logistic leaf value the IRLS route would have produced (Σ(y−p)/Σp(1−p) is what a weighted
least-squares fit with weight p(1−p) puts in a leaf). The two routes agree at the *leaf*; they disagree
only at *split-finding*, where IRLS wants the curvature weights and I am fitting plain squared error to
the gradient. That is the fork I promised to come back to. This two-step split — cheap squared-error fit
for the partition, exact (or Newton) line search for the magnitudes — is what makes gradient boosting
accurate across losses. And it lets me re-verify the AdaBoost connection at the leaf level: for the
exponential loss, minimizing Σ_{R} exp(−yᵢ(F + γ)) over a leaf splits into positives and negatives with
weighted masses W₊ = Σ_{R,+} exp(−yF) and W₋ = Σ_{R,−} exp(−yF), and setting the derivative
−W₊e^{−γ} + W₋e^{γ} = 0 gives γ = ½ log(W₊/W₋). For a stump covering the whole space with weighted
error ε, that is ½ log((1−ε)/ε) — exactly half the AdaBoost vote weight log((1−ε)/ε), the factor ½ being
the standard {−1,+1}-versus-0/1 convention. The line search recovers AdaBoost's coefficient on the nose.

Second, taking the full optimal step every round overfits — the model races to fit the training data
and generalizes poorly. So I shrink each step by a learning rate ν ∈ (0,1]:

  F_m(x) = F_{m−1}(x) + ν·γ_{jm}  for x ∈ R_{jm}.

Small ν means more rounds but better generalization; it is the regularization knob that trades trees
for accuracy. With ν in hand, the stagewise procedure is: compute pseudo-residuals from the current loss
gradient, fit a regression tree to them, line-search each leaf, take a shrunk step.

Now I can settle the fork I deferred, because I have the frame to state the cost precisely. The
per-loss IRLS route and a full *Newton step in function space* are the same move: use the second
derivative to precondition the whole descent direction, not just the per-leaf magnitude — replace the
plain squared-error split criterion with a Hessian-weighted one so the curvature shapes *which* splits
are chosen. It is tempting because for most losses the curvature hᵢ is as cheap to compute as the
gradient, and I am already computing it in the leaf line search. But putting hᵢ into split-finding
changes what the base learner optimizes at *every node*, which is a genuine rebuild of the tree learner
— the impurity criterion, the sums accumulated at each candidate threshold, the leaf-value formula all
change together. That is a larger commitment than I want to take on while I am still establishing that
the gradient view generalizes AdaBoost *at all*. The per-leaf Newton line search already captures the
curvature where it is cheapest and least disruptive to use — inside a fixed leaf, one scalar per leaf —
so I keep the split criterion as plain least squares on the first-order pseudo-residual and bank the
generality. The gradient machine is the right thing to lock in first; the split score, and whether to
carry the Hessian into it, can be revisited once *speed* forces my hand and I have to rebuild the split
finder anyway.

```python
def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, ...):
    loss = self._loss
    original_y = y
    raw_predictions_copy = raw_predictions.copy()
    for k in range(loss.K):                       # one tree per class (K=1 for regression/binary)
        if loss.is_multi_class:
            y = np.array(original_y == k, dtype=np.float64)
        # pseudo-residuals = negative gradient of the loss at the current model
        residual = loss.negative_gradient(y, raw_predictions_copy, k=k, sample_weight=sample_weight)
        # fit a regression tree to the pseudo-residuals by least squares (exact split finding)
        tree = DecisionTreeRegressor(criterion=self.criterion, splitter="best",
                                     max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes, ...)
        tree.fit(X, residual, sample_weight=sample_weight, check_input=False)
        # line-search each leaf in the real loss, then take a shrunk (learning_rate) step
        loss.update_terminal_regions(tree.tree_, X, y, residual, raw_predictions,
                                     sample_weight, sample_mask, learning_rate=self.learning_rate, k=k)
        self.estimators_[i, k] = tree
    return raw_predictions
```

Now the part that determines the cost, and I want to get its size right on paper before I run anything,
because it is going to be the thing this rung is measured on. The tree's structure is found by
`splitter="best"`, which for each candidate feature *pre-sorts* the examples by that feature value and
scans every adjacent pair as a candidate threshold, picking the split that most reduces squared-error
impurity on the pseudo-residuals. This is **exact** split finding: every possible threshold on every
feature is evaluated, so the split chosen is provably the best one for the current node. Count the work
on the spine dataset: n = 10.5M rows, d = 28 features. The initial per-tree sort costs O(n log n) per
feature, roughly 10.5M·log₂(10.5M) ≈ 10.5M·23 ≈ 2.4×10⁸ comparisons per feature, times 28 features
≈ 6.8×10⁹ just to sort — and this sort is rebuilt every round, because the pseudo-residuals change and,
under subsampling, the active row set changes. Then at each depth level the nodes partition the data, so
scanning every node's examples once per feature is O(n·d) ≈ 2.9×10⁸ threshold evaluations per level, and
a depth-8 tree pays that on the order of eight times, ≈ 2.3×10⁹ evaluations per tree — every one a gather
through a pre-sorted index into rows that are scattered all over memory, which is cache-hostile and
several times slower per operation than its raw op count suggests. So a single iteration is doing on the
order of 10⁹–10¹⁰ real operations, almost all of them memory-bound, and the run is 500 of those. There
is nothing in the loss or the model class that is expensive — the pseudo-residual is one subtraction per
example, the leaf line search is a handful of sums per leaf — the entire per-round wall-clock is this
exhaustive pre-sorted scan over all 10.5M rows at every node of every tree.

Let me also sanity-check the *shape* of the accuracy claim against a known limit, so I am not asserting
the AUC will be fine on faith. When L is squared error and I strip the regularization, the tree fits
yᵢ − F by least squares, which is ordinary variance-reduction splitting; and the exact best-first
splitter finds the globally optimal threshold at each node. There is no approximation anywhere in the
pipeline — the split is optimal, the leaf is exact, the only inexactness is the greedy stagewise
structure that all boosting shares. So on a dense numeric benchmark like Higgs the accuracy should be
essentially the best a depth-8, 500-round, ν = 0.1 gradient-boosted ensemble can do; there is no reason
to expect the exact splitter to *lose* AUC to anything, since anything faster will only *approximate*
the split it computes exactly. What I therefore expect when this meets the benchmark is a test AUC that
is genuinely competitive — high, and a reference the faster rungs will have to match — while the
training-seconds-per-iteration is the number that stands out, dominated entirely by the 10⁹–10¹⁰-per-tree
pre-sorted scan I just counted. I do not have that seconds figure yet; the Higgs timing table will tell
me exactly how bad, and my prediction is falsifiable in the specific sense that the per-iteration time
should be enormous *relative to any method that visits far fewer candidate thresholds* — orders of
magnitude, not a constant factor — because the bottleneck scales with (number of distinct feature values
scanned) × (number of nodes), and that is precisely the product a candidate-thinning scheme would cut.

So this is what I have: **gradient boosting** — boosting recast as steepest descent in function space,
where the loss enters only through the pseudo-residual gⁱ_m, the tree projects that gradient onto the
base-learner class by least squares, a per-leaf line search calibrates the step in the true loss, and
shrinkage ν regularizes. It dissolves AdaBoost's classification-only limitation: any differentiable loss
works by changing one gradient formula, and both the per-loss IRLS route and AdaBoost itself fall out as
special cases rather than competitors. The wall it leaves standing is speed. The exact splitter's
per-iteration cost is set entirely by re-sorting and re-scanning all 10.5M rows at every node, and that
cost is not incidental — it is the price of *provably* optimal splits I do not actually need. The next
thing to attack is not *what* the trees fit but *how* the splits are found: if I can replace "pre-sort
and re-scan all the data at every node" with something that visits a small, representative set of
candidate thresholds, the seconds-per-iteration should fall by orders of magnitude while the AUC holds —
and that is where I go next.
