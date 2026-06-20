The adaptive reweighting works, but it is wired to a single thing: a 0/1 "right or wrong" indicator on
a classification label. Each round I crush the weight of correctly-classified examples and lift the
misclassified ones, which is a clever way to descend *one particular* loss — the exponential loss of
the margin — but it is the only loss it knows. If I want to predict a continuous target, or fit a
Huber/absolute loss that is robust to outliers, or a Poisson loss for counts, the reweighting trick has
no handle. The committee only ever hears "this example is wrong," never "this example's prediction is
off by +3.2 and the loss there is steep." I want to keep the boosting structure — an additive model
built one tree at a time, each tree attacking what the committee gets wrong — but replace the
classification-specific reweighting with something that works for *any* differentiable loss.

Step back and write what I'm actually building. The model is an additive expansion

  F(x) = Σₘ h_m(x),

and I build it greedily: having F_{m−1}, I add one more term to reduce the total loss
Σᵢ L(yᵢ, F(xᵢ)). The exact greedy step would be h_m = argmin_h Σᵢ L(yᵢ, F_{m−1}(xᵢ) + h(xᵢ)) — but
that joint minimization over a function h is intractable for a general loss. I need an approximation.

Here is the reframing that unlocks it. Forget that F is parameterized by trees for a moment and think
of the prediction *vector* — the n numbers F(x₁),…,F(xₙ) — as a point in ℝⁿ. The total loss
Φ = Σᵢ L(yᵢ, F(xᵢ)) is just a function on ℝⁿ, and I want to decrease it. The direction of steepest
descent of Φ at the current point is the negative gradient, whose i-th component is

  −[ ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) ]_{F = F_{m−1}}.

Call that gⁱ_m. If I could move the prediction vector by a small step in the direction −∇Φ, I'd reduce
the loss — that's just gradient descent. But I can't move each F(xᵢ) independently: I'm constrained to
move along directions that a *tree* can produce, because the only thing I'm allowed to add is a tree
h_m(x). So the negative gradient lives in the unconstrained space ℝⁿ, and I have to *project* it onto
the space of functions my base learner can represent.

The projection is the whole idea. The negative-gradient vector (g¹_m,…,gⁿ_m) is the ideal per-example
nudge — these are the **pseudo-residuals**, the amount each prediction "wants" to move to lower the
loss fastest. I fit a regression tree to *them*, by least squares:

  h_m = argmin_h Σᵢ ( gⁱ_m − h(xᵢ) )²,

i.e. I treat the negative gradient as a regression target and find the tree that best reproduces it.
That tree is the closest realizable approximation, in the base-learner's function class, to the
steepest-descent direction. Now the loss has vanished from the base-learner's job entirely: the tree
always solves a plain squared-error regression, no matter what L is. *All* the loss-specific
information has been pushed into a single place — the formula for the pseudo-residual gⁱ_m. Swap the
loss, swap one gradient expression, and the entire machine retargets. That is exactly the generality
AdaBoost lacked.

Check that AdaBoost is the special case, to be sure I haven't lost anything. Take L to be the
exponential loss L(y, F) = exp(−yF) with y ∈ {−1,+1}. Its negative gradient at F_{m−1} is
yᵢ·exp(−yᵢF_{m−1}(xᵢ)) — large exactly on the examples the current margin gets wrong, scaled by a
weight exp(−yᵢF_{m−1}(xᵢ)) that *is* the AdaBoost reweighting. So AdaBoost is gradient boosting on the
exponential loss; the new framework contains it and extends past it to squared error, absolute error,
Huber, logistic deviance, Poisson — anything with a gradient.

Two refinements the raw idea needs. First, fitting the tree to the pseudo-residuals fixes the tree's
*partition* of the input space (its leaves), but the least-squares fit gives each leaf the mean
pseudo-residual, which is only optimal if L is squared error. For a general loss I can do better:
once the leaf regions R_{jm} are fixed, choose each leaf's output value by an exact line search *in the
loss itself*:

  γ_{jm} = argmin_γ Σ_{xᵢ ∈ R_{jm}} L(yᵢ, F_{m−1}(xᵢ) + γ).

This is a separate one-dimensional optimization per leaf — for logistic deviance it's a single Newton
step, in closed form. So the tree's *structure* is chosen by least-squares on the gradient, but its
*leaf values* are chosen by minimizing the real loss along the directions that structure provides. This
two-step split is what makes gradient boosting work well across losses: the cheap squared-error fit
finds the partition, the exact line search calibrates the magnitudes.

Second, taking the full optimal step every round overfits — the model races to fit the training data
and generalizes poorly. So I shrink each step by a learning rate ν ∈ (0,1]:

  F_m(x) = F_{m−1}(x) + ν·γ_{jm}  for x ∈ R_{jm}.

Small ν means more rounds but better generalization; it is the regularization knob that trades trees
for accuracy. With ν in hand, the stagewise procedure is: compute pseudo-residuals from the current
loss gradient, fit a regression tree to them, line-search each leaf, take a shrunk step.

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

Now the part that determines the cost. The tree's structure is found by `splitter="best"`, which for
each candidate feature *pre-sorts* the examples by that feature value and scans every adjacent pair as
a candidate threshold, picking the split that most reduces squared-error impurity on the
pseudo-residuals. This is **exact** split finding: every possible threshold on every feature is
evaluated, so the split chosen is provably the best one for the current node. The price is that the
inner loop runs over all n examples for all d features at every node of every tree — and the pre-sort
plus the full scan is what makes a single iteration expensive when n is in the tens of millions. On a
dense, large dataset the cost per boosting round is dominated by this exhaustive split search.

So this is what I've got: **gradient boosting** — boosting recast as steepest descent in function
space, where the loss enters only through the pseudo-residual gⁱ_m, the tree projects that gradient
onto the base-learner class by least squares, a per-leaf line search calibrates the step in the true
loss, and shrinkage ν regularizes. It dissolves AdaBoost's classification-only limitation: any
differentiable loss works by changing one gradient formula. The wall I can already see is speed. The
exact, pre-sorted split scan is doing an enormous amount of work — sorting and re-scanning all of the
data at every node — and on a dataset with millions of rows the training time per iteration is going to
be brutal. The next thing to attack is not *what* the trees fit but *how* the splits are found.
