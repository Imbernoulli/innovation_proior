The adaptive reweighting works, but it is wired to a single thing: a 0/1 "right or wrong" indicator on
a classification label. Each round I crush the weight of correctly-classified examples and lift the
misclassified ones, which — as the AdaBoost bookkeeping showed — is exactly stagewise
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
the AdaBoost weight I derived earlier (the sample weight entering a round is
proportional to exp(−yᵢ F(xᵢ))). So gradient boosting on the exponential loss reproduces AdaBoost's
reweighting as a byproduct of the gradient, and the new framework strictly contains it while reaching
past it to squared error, absolute error, Huber, logistic deviance, Poisson — anything with a gradient.

Two refinements the raw idea needs. First, fitting the tree to the pseudo-residuals fixes the tree's
*partition* of the input space (its leaves R_{jm}), but the least-squares fit would set each leaf's
output to the mean pseudo-residual, which is only optimal if L is squared error. For a general loss I
can do better: once the leaf regions are fixed, choose each leaf's output by an exact line search *in
the loss itself*,

  γ_{jm} = argmin_γ Σ_{xᵢ ∈ R_{jm}} L(yᵢ, F_{m−1}(xᵢ) + γ),

a separate one-dimensional optimization per leaf. What it does depends on the loss, and the differences
are instructive. For squared error L = ½(y − F)² the pseudo-residual is yᵢ − F and the line search
returns the leaf's mean residual, so the whole thing collapses to ordinary regression-tree boosting. For
absolute error L = |y − F| the pseudo-residual is only the *sign* sign(yᵢ − F), so the tree structure is
fit to ±1 signs while the line search takes the *median* residual in each leaf — and that split is
exactly why absolute-error boosting is robust: on a leaf holding y = (1, 2, 3, 100) with F = 0 the
pseudo-residuals are all +1 (the tree cannot even see the outlier), and the median step of 2.5 moves the
leaf up sensibly rather than being dragged toward the mean 26.5 by the single gross value. For logistic
(binomial) deviance the pseudo-residual is yᵢ − pᵢ with pᵢ = σ(F_{m−1}(xᵢ)), and the leaf minimization
has no closed form, so I take one Newton step,

  γ_{jm} = Σ_{R}(yᵢ − pᵢ) / Σ_{R} pᵢ(1 − pᵢ),

numerator the sum of gradients, denominator the sum of local curvatures pᵢ(1−pᵢ). On a leaf with
p = (0.5, 0.8, 0.3) and y = (1, 0, 1) the gradients sum to 0.4 and the curvatures to 0.62, so the Newton
leaf is γ = 0.4/0.62 = 0.645 — nearly five times the naive mean pseudo-residual 0.4/3 = 0.133, because
these examples sit on a shallow part of the logistic curve where a larger step is warranted. So the
structure is found by least squares on the first-order pseudo-residual, but the leaf magnitude uses the
second derivative *locally, per leaf* — and this is exactly the logistic leaf value the IRLS route would
have produced. The two routes agree at the *leaf*; they disagree only at *split-finding*, where IRLS
wants the curvature weights and I am fitting plain squared error to the gradient. That is the fork I
promised to come back to. This two-step split — cheap squared-error fit for the partition, exact (or
Newton) line search for the magnitudes — is what makes gradient boosting accurate across losses.

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
finder anyway. (The per-stage fit — pseudo-residual, tree, per-leaf line search, shrunk step — is in the
answer's `_fit_stage`.)

Now the part that determines the cost. The tree's structure is found by
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

The accuracy side of the claim I can pin down from a known limit. With squared error and no
regularization the tree fits yᵢ − F by ordinary variance-reduction splitting, and the exact best-first
splitter finds the globally optimal threshold at each node — there is no approximation anywhere except
the greedy stagewise structure all boosting shares. So on a dense numeric benchmark like Higgs the
accuracy should be essentially the best a depth-8, 500-round, ν = 0.1 ensemble can do: anything faster
can only *approximate* the split this computes exactly, so the exact splitter sets an accuracy reference
nothing should beat. What stands out instead is the training time, dominated entirely by the
10⁹–10¹⁰-per-tree pre-sorted scan. I do not have that figure yet, but the falsifiable prediction is not a
specific number — it is that the per-iteration time is enormous *relative to any method that visits far
fewer candidate thresholds*, orders of magnitude rather than a constant factor, because the bottleneck
scales with (distinct feature values scanned) × (nodes), exactly the product a candidate-thinning scheme
would cut.

So this is **gradient boosting** — boosting recast as steepest descent in function space, where the loss
enters only through the pseudo-residual gⁱ_m, the tree projects that gradient onto the base-learner class
by least squares, a per-leaf line search calibrates the step in the true loss, and shrinkage ν
regularizes. It dissolves AdaBoost's classification-only limitation: any differentiable loss works by
changing one gradient formula, with the per-loss IRLS route and AdaBoost itself falling out as special
cases. The wall it leaves standing is speed — the exact splitter's per-iteration cost is set entirely by
re-sorting and re-scanning all 10.5M rows at every node, the price of *provably* optimal splits I do not
actually need. So the next thing to attack is not *what* the trees fit but *how* the splits are found:
replace "pre-sort and re-scan all the data at every node" with something that visits a small,
representative set of candidate thresholds, and the seconds-per-iteration should fall by orders of
magnitude while the AUC holds.
