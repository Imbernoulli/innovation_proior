# Context: explaining a black-box prediction by its training data

## Research question

The best-performing models in vision, speech, and language are large, opaque function approximators. When such a model makes a particular prediction on a particular test input, a natural and increasingly consequential question is: *why?* The dominant line of interpretability work answers this in terms of the **fixed, already-trained model** — it asks how this prediction depends on the test input or on the model's internal structure. But a model's parameters are the output of an optimization over a training set, which raises a second, complementary "why": *which training examples are responsible for this prediction, and how would the prediction change if a particular training example were absent or slightly different?*

The goal is to attribute a test-time prediction back to individual training points — to trace the prediction through the learning algorithm and back to the data the parameters were estimated from. The setting of interest is models with millions of parameters, trained by stochastic gradient methods to a point that is not an exact loss minimizer, sometimes with non-convex or even non-differentiable losses.

## Background

**The counterfactual definition of responsibility.** The conceptually cleanest way to measure the effect of a training point `z` on a model is the *deletion diagnostic* / leave-one-out (LOO) experiment: refit the model on the training set with `z` removed, obtaining `θ̂_{-z}`, and compare `θ̂_{-z}` (or the test loss it induces) to the model `θ̂` fit on the full set. This is a counterfactual — "what would the model be if we had never seen `z`?" — and it is the gold-standard notion of how much `z` mattered. With `n` training points, computing all LOO answers means `n` separate retrainings.

**Robust statistics: studying an estimator through perturbations of its data.** A mature body of work in statistics studies how an estimator responds to its data. Hampel (1974), building on Jaeckel's (1972) infinitesimal jackknife, introduced the *influence curve*: an asymptotic, derivative-based description of how a functional of an estimator reacts to an infinitesimal contamination of the data distribution. Cook (1977), Cook & Weisberg (1980, 1982), Pregibon (1981) and others turned this into regression diagnostics — Cook's distance and its relatives — that flag *influential observations* and high-leverage points. This tradition operated on small linear or generalized-linear models (Cook & Weisberg (1980) work with `n=24`, `p=10`), where one can afford exact solutions or careful characterizations of the error. The estimator there is an M-estimator: `θ̂` minimizes an empirical risk `(1/n)Σ L(z_i,θ)`, and the diagnostics exploit the first-order optimality condition `Σ∇_θ L(z_i,θ̂)=0` to describe, in closed form, how `θ̂` moves when the weight on one observation is perturbed. The machinery is asymptotic and assumes a twice-differentiable, strictly convex risk (so that the Hessian `H = (1/n)Σ∇²_θ L(z_i,θ̂)` is positive definite and invertible) with `θ̂` at the true minimizer.

**The conditions in modern ML.** The diagnostics involve the Hessian of the empirical risk, a `p×p` matrix; forming and inverting it costs `O(np² + p³)`, at `p≈10⁶`. Modern models are routinely non-convex, so the risk has no unique minimizer and the Hessian can be indefinite. And modern training stops early and uses non-differentiable pieces (ReLUs, hinges), departing from both the "θ̂ is the exact minimizer" and the "twice-differentiable" assumptions. The classical influence-diagnostics literature has so far been applied to small, convex, differentiable, fully-converged models.

**Second-order information without the Hessian.** Two pieces of machinery from the optimization literature are relevant. Pearlmutter (1994) showed that the product of the Hessian with an arbitrary vector, `Hv`, can be computed *exactly* in the same time and memory as a single gradient evaluation — `O(p)` — without ever forming `H`, by applying a differential ("R-operator") technique to the gradient computation; equivalently `Hv = ∇_θ(v·∇_θ L)`. The naive alternative, a finite-difference `Hv ≈ [∇(θ+rv)-∇(θ)]/r`, is numerically unstable (cancellation as `r→0`). On top of HVPs, Martens (2010) built Hessian-free optimization, using conjugate gradients (CG) — which require only `Hv` — to solve linear systems `Hx=v` implicitly; and Agarwal, Bullins & Hazan (2017) gave LiSSA, a *stochastic* linear-time estimator of an inverse-Hessian-vector product based on a recursive (Neumann-series) reformulation of `H^{-1}`, where each iteration samples a single training point's Hessian rather than touching all `n`. (Agarwal et al. develop their estimator for generalized linear models, where the per-term HVP is cheap by construction.)

**The interpretability landscape this sits next to.** Contemporary explanation methods operate on the trained model: LIME (Ribeiro et al., 2016) fits a simple interpretable model locally around a test point; saliency / gradient methods (Simonyan et al., 2013) and input-perturbation / erasure methods (Li et al., 2016; Datta et al., 2016; Adler et al., 2016) probe how the prediction changes as the *test input* is perturbed. Separately, *adversarial test examples* (Goodfellow et al., 2015; Moosavi-Dezfooli et al., 2016) show that imperceptible perturbations of a **test** input can flip a prediction. And in security, training-set *poisoning* attacks (Biggio et al., 2012; Mei & Zhu, 2015) derive, from the KKT conditions of the learner, perturbations of the **training** data that degrade a model. A method that attributes predictions to training data would connect to all of these but is distinct from each: none of them traces a prediction back through the optimizer to the training set.

**Empirical facts known about the prior tools.** Ranking training points by raw Euclidean nearest-neighbor distance to the test point (a common heuristic, e.g. inside LIME-style explanations) reflects only the inputs and not the learning dynamics: if all inputs are non-negative (as with pixel intensities), every same-label training point has a non-negative relevant inner product. High-loss / outlier training points move an M-estimator's parameters more than well-fit points. And for a hinge loss, the second derivative is zero almost everywhere, so it carries no information about how close a support vector is to the margin.

## Baselines

- **Leave-one-out (deletion) retraining.** For each training point `z`, refit `θ̂_{-z} = argmin Σ_{z_i≠z} L(z_i,θ)` and report `L(z_test, θ̂_{-z}) − L(z_test, θ̂)`. This is the definitional ground truth for "how much did `z` matter for the prediction on `z_test`." It costs one full retraining per training point (per test point of interest if you want test-loss attribution), which is `O(n)` trainings.

- **Euclidean nearest neighbors / input-space similarity.** Rank training points by closeness to the test point in input (or feature) space; with equal norms this is ranking by the inner product `x·x_test`. Used as a cheap stand-in for "relevant training examples." It is a property of the inputs.

- **Classical regression influence diagnostics (Cook's distance and relatives; Hampel's influence curve).** Closed-form, derivative-based measures of how a single observation moves an estimator, exploiting the optimality condition of an M-estimator. Core math: the change in `θ̂` from perturbing one observation is expressed through the inverse of the (problem-specific) information / Hessian matrix and the observation's gradient/residual. Developed and validated on small, convex, twice-differentiable, fully-fit linear and generalized-linear models with exact matrix inverses.

- **Model-side interpretability (LIME; saliency / input perturbation).** Explain a fixed model's prediction by a local surrogate or by sensitivity to the test input. They explain the prediction in terms of the model and the test input.

- **KKT-based training-set poisoning attacks (Biggio et al. 2012; Mei & Zhu 2015).** Derive perturbations of the training data that hurt a learner, directly from its optimality conditions; demonstrated chiefly on SVMs and (generalized) linear models. Framed model-by-model and from the KKT conditions; for an SVM the demonstrated training perturbations were visibly distinguishable, and the framework is tied to continuous data and specific convex learners.

## Evaluation settings

The natural yardsticks are standard small-to-medium supervised classification problems where leave-one-out retraining is *possible* (so the approximation can be checked against ground truth) as well as deep models where it is not:

- **MNIST handwritten digits** (LeCun et al., 1998) — both binary tasks (1s vs. 7s) for logistic regression / linear SVM, and the full 10-class task; the canonical setting for checking a fast approximation against actual LOO retraining.
- **A frozen-features image task (dogs vs. fish extracted from ImageNet)** with an Inception-v3 network (Szegedy et al., 2016) restricted to its top layer (equivalent to logistic regression on bottleneck features) versus an RBF-kernel SVM — a setting to compare *how* two equally-accurate models rely on training data.
- **A tabular clinical task (hospital readmission, ~20K diabetic patients, 127 features)** with logistic regression — for diagnosing domain mismatch.
- **Email spam classification (Enron, bag-of-words)** with logistic regression — for prioritizing inspection of possibly-mislabeled training data.

Metrics / protocols that exist independently of any new method: correlation (e.g. Pearson's R) between a predicted change in test loss and the *actual* change measured by genuine LOO retraining; for the spam task, test accuracy and fraction-of-flipped-labels-recovered as a function of the fraction of training data inspected; for attacks, whether a visually-indistinguishable training perturbation (same 8-bit image) flips the test prediction. When comparing a fast inverse-Hessian solve to its slow exact counterpart, the yardstick is agreement with the exact CG solution.

## Code framework

Pre-existing primitives: an automatic-differentiation framework that gives gradients of a scalar loss with respect to parameters, and (via two passes of reverse-mode autodiff) Hessian-vector products at the cost of a gradient (Pearlmutter, 1994); a standard model + loss + optimizer training stack; conjugate-gradient and Newton-CG solvers (e.g. SciPy's `fmin_ncg`) that consume only matrix-vector products. The contribution will be filled into the empty slots below; everything outside the stubs is standard.

```python
import torch

# ---- standard, pre-existing training stack ----
def train_model(model, data, loss_fn, optimizer, num_steps):
    """Fit parameters by empirical risk minimization. Standard; not the contribution."""
    for step in range(num_steps):
        xb, yb = data.next_batch()
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)          # per-batch empirical risk (regularizer folded into loss_fn)
        loss.backward()
        optimizer.step()
    return model

def grad_params(loss, params):
    """Gradient of a scalar loss w.r.t. a parameter list (one reverse pass)."""
    return torch.autograd.grad(loss, params, create_graph=True)

def hvp(loss, params, v):
    """Hessian-vector product H v at the current params, using only autodiff.
    Costs about one gradient evaluation; never forms H. (Pearlmutter, 1994.)"""
    g = torch.autograd.grad(loss, params, create_graph=True)
    # TODO: combine g and v with a second reverse pass to obtain H v

# ---- the slots the method will occupy ----
def solve_with_hessian(loss_on_train, params, v):
    """H = (1/n) sum_i grad^2 L(z_i, theta) is the empirical-risk Hessian, too large to form
    or invert directly at p ~ 1e6. # TODO: the object we will define here."""
    pass

def explain_prediction(model, train_data, z_test, loss_fn, params):
    """Given a test point, score every training point by its effect on the test prediction,
    without retraining. # TODO: the quantity we will design and how to compute it at scale."""
    pass
```
