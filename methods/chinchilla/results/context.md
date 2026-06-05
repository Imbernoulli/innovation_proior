# Context

## Research question

Training a large language model is, in practice, a one-shot event. You know your compute budget in advance — how many accelerators you have and for how long — and you typically get exactly one attempt at the full-scale run, because a single run already costs an enormous amount of money and energy. So before you spend it, you must commit to two numbers that you cannot easily revise: how big to make the model (parameter count $N$), and how much data to train it on (token count $D$). These two are not free: a fixed compute budget $C$ couples them, because training cost grows with both the model size and the amount of data processed.

The precise problem is therefore an allocation problem. Given a fixed FLOPs budget $C$, choose $N$ and $D$ to minimize the final pretraining loss $L(N, D)$, subject to the constraint that the chosen $(N,D)$ actually costs $C$:
$$N_{\text{opt}}(C),\ D_{\text{opt}}(C) = \operatorname*{argmin}_{N,D\ \text{s.t.}\ \text{FLOPs}(N,D)=C} L(N,D).$$
A solution has to deliver the *functions* $N_{\text{opt}}(C)$ and $D_{\text{opt}}(C)$ — a recipe that, for any budget, tells you the optimal model size and the optimal number of training tokens — and it has to be estimable from a feasible set of smaller training runs, since fitting it by trial at full scale is exactly what we can't afford.

## Background

**The era of ever-larger models.** By the time this problem is posed, the largest dense Transformer language models exceed 500B parameters (e.g. a 530B model), with others at 175B and 280B. They show strong zero-shot, few-shot, and fine-tuned performance, and the field's working assumption is that the path forward is more parameters.

**Power laws relating size, data, compute, and loss.** Test loss falls predictably as a power law in model size, dataset size, and compute. The constraint that ties everything together is the cost model: a forward-and-backward pass over $D$ tokens through an $N$-parameter dense model costs approximately $\text{FLOPs}(N,D) \approx 6ND$ — roughly $2ND$ for the forward pass (about two FLOPs, a multiply and an add, per parameter per token) and about twice that for the backward. So to good approximation $C$ is proportional to the product $N\cdot D$, which is exactly what makes the size/data trade-off a one-dimensional constraint: spend more on $N$ and you must spend less on $D$, and vice versa.

**The prevailing allocation, and a clue that it may be off.** The influential prior analysis (Kaplan et al., 2020) concluded that when compute increases, most of it should go into a bigger model: a $10\times$ increase in budget calls for roughly a $5.5\times$ larger model but only a $1.8\times$ increase in tokens (exponents on the order of $N\propto C^{0.73}$, $D\propto C^{0.27}$). Following this, and GPT-3's setup, most large models were trained for roughly 300B tokens regardless of their size. But there is a methodological detail that should make one uneasy about that conclusion: the learning-rate schedule. With a cosine schedule, the learning rate is supposed to decay (by about $10\times$) over the course of training and reach its floor right as training ends. If, instead, you set one long cosine cycle and read off the loss at intermediate points — as you would if you estimate the loss-at-$D$ from a single longer run — those intermediate points have *not* had their learning rate decayed appropriately for stopping at $D$, and their loss is systematically too high. The diagnostic is sharp: when the cosine cycle length overshoots the actual number of training steps by more than about 25%, final performance is noticeably degraded. So the cosine cycle length should be matched to the intended number of training tokens, and any estimate that doesn't do this will mis-measure the loss at intermediate horizons — plausibly biasing the inferred frontier toward large models trained on few tokens. Many of the runs underlying the prevailing allocation were also quite small (a large fraction under 100M parameters).

**Risk decomposition as a functional form.** A classical way to decompose the expected risk of a trained predictor suggests a parametric form for $L(N,D)$. Predicting the next token is choosing $f:\mathcal X\to\mathcal D(\mathcal Y)$; the Bayes-optimal predictor $f^\star$ minimizes the cross-entropy over the true distribution. Restricting to Transformers of size $N$ gives the best-in-class $f_N$, and training for a single epoch over $D$ tokens with a finite number of gradient steps gives $\bar f_{N,D}$. Then
$$L(N,D) = L(f^\star) + \big(L(f_N)-L(f^\star)\big) + \big(L(\bar f_{N,D})-L(f_N)\big),$$
i.e. an irreducible Bayes risk (the entropy of natural text), plus a function-approximation gap that depends on $N$ (for two-layer networks expected to scale like $N^{-1/2}$; Siegel, 2020), plus a stochastic-approximation gap that depends on $D$ (early stopping of a stochastic first-order method, whose convergence rate is lower-bounded by $D^{-1/2}$ and is dimension-free; Robbins & Monro, 1951). This motivates the additive power-law shape $E + A/N^{\alpha} + B/D^{\beta}$.

**Numerical tools.** Robust regression with the Huber loss (Huber, 1964) downweights outliers; the log-sum-exp operator gives a numerically stable way to compute a log of a sum of exponentials; L-BFGS (Nocedal, 1980) is a standard quasi-Newton optimizer for smooth low-dimensional fits.

## Baselines

**Kaplan et al. (2020) — power-law scaling of language models.** Establishes that loss is a power law in $N$, $D$, and $C$, and gives an allocation rule that favours growing the model far faster than the data ($N\propto C^{0.73}$, $D\propto C^{0.27}$). Core idea: fit loss as a function of size and compute from many runs, then extrapolate. The gap it leaves: its allocation makes large models compute-optimal only if you trust the loss estimates at intermediate token counts, which depend on the learning-rate schedule being matched to the horizon — and much of its data is from small models and from runs whose schedule was not matched. That makes the allocation rule the key target to re-estimate.

**Single-method fits in general.** Any one estimation procedure — reading minima off training curves, or profiling loss across model sizes at fixed compute, or fitting a global parametric surface — carries its own biases (interpolation artifacts, the choice of curve to fit a parabola to, the functional form assumed). The gap: no single method is self-certifying. A trustworthy answer needs *multiple independent* estimators that agree.

## Evaluation settings

The quantity modeled is the smoothed final pretraining cross-entropy loss $L$ (a near-unbiased estimate of test loss in the effectively-infinite-data regime where the corpus exceeds the tokens consumed), as a function of $(N,D)$. The experimental material is a sweep of training runs: models ranging from under 70M to over 16B parameters, trained on 5B to over 400B tokens, with each configuration run for several token horizons and several FLOP budgets (nine fixed FLOP counts from $6\times10^{18}$ to $3\times10^{21}$ for the iso-compute profiles). The cosine cycle length is matched to each run's token count. The fitted laws are then extrapolated to a large target budget (on the order of $5.76\times10^{23}$ FLOPs) to predict the optimal $(N,D)$ there. Robustness is assessed by bootstrapping the runs and by repeating on additional corpora.

## Code framework

The primitives that already exist: a numerically stable log-sum-exp, a Huber penalty, and an L-BFGS optimizer (e.g. from SciPy or a deep-learning autodiff library). Given a table of training runs, each a triple $(N_i, D_i, L_i)$, the pieces to fill in are a parametric loss model, a routine that fits its parameters to the runs, and a routine that turns the fitted parameters into the optimal allocation for any budget — plus the two purely-empirical estimators that read the optimum straight off the runs.

```python
import numpy as np

def log_loss_pred(theta, logN, logD):
    # TODO: evaluate the fitted log-loss stably from log-parameters and
    #       logged model/data sizes.
    pass


def parametric_loss(N, D, params):
    # TODO: a closed-form model of final loss as a function of model size N
    #       and token count D, with a handful of fitted parameters.
    pass


def fit_parametric(runs):
    # runs: list of (N_i, D_i, L_i). TODO: fit params robustly (Huber) from a
    #       grid of initialisations using a smooth optimiser.
    pass


def optimal_allocation(C, params):
    # TODO: minimise the fitted loss subject to the compute constraint
    #       FLOPs(N, D) = C; return (N_opt, D_opt).
    pass


def envelope_optimum(C, run_curves):
    # TODO: empirical estimator -- the lowest-loss (N, D) on the FLOPs == C
    #       slice of the training-curve envelope.
    pass


def isoflop_optimum(runs_at_fixed_C):
    # TODO: empirical estimator -- at a fixed compute budget, find the model
    #       size that minimises final loss.
    pass


def fit_power_law(Cs, values):
    # TODO: fit value ∝ C**exponent in log-log space; return (coeff, exponent).
    pass
```
