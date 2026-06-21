# Context

## Research question

Training a large language model is, in practice, a one-shot event. You know your compute budget in advance — how many accelerators you have and for how long — and you typically get exactly one attempt at the full-scale run, because a single run already costs an enormous amount of money and energy. So before you spend it, you must commit to two numbers: how big to make the model (parameter count $N$), and how much data to train it on (token count $D$). These two are coupled: a fixed compute budget $C$ ties them together, because training cost grows with both the model size and the amount of data processed.

The problem is therefore an allocation problem. Given a fixed FLOPs budget $C$, choose $N$ and $D$ to minimize the final pretraining loss $L(N, D)$, subject to the constraint that the chosen $(N,D)$ actually costs $C$:
$$N_{\text{opt}}(C),\ D_{\text{opt}}(C) = \operatorname*{argmin}_{N,D\ \text{s.t.}\ \text{FLOPs}(N,D)=C} L(N,D).$$
A solution delivers the *functions* $N_{\text{opt}}(C)$ and $D_{\text{opt}}(C)$ — a recipe that, for any budget, gives the optimal model size and the optimal number of training tokens — and it is estimated from a feasible set of smaller training runs.

## Background

**The era of ever-larger models.** By the time this problem is posed, the largest dense Transformer language models exceed 500B parameters (e.g. a 530B model), with others at 175B and 280B. They show strong zero-shot, few-shot, and fine-tuned performance, and the field's working assumption is that the path forward is more parameters.

**Power laws relating size, data, compute, and loss.** Test loss falls predictably as a power law in model size, dataset size, and compute. The constraint that ties everything together is the cost model: a forward-and-backward pass over $D$ tokens through an $N$-parameter dense model costs approximately $\text{FLOPs}(N,D) \approx 6ND$ — roughly $2ND$ for the forward pass (about two FLOPs, a multiply and an add, per parameter per token) and about twice that for the backward. So to good approximation $C$ is proportional to the product $N\cdot D$, which makes the size/data trade-off a one-dimensional constraint: spend more on $N$ and you spend less on $D$, and vice versa.

**The prevailing allocation.** The influential prior analysis (Kaplan et al., 2020) concluded that when compute increases, most of it should go into a bigger model: a $10\times$ increase in budget calls for roughly a $5.5\times$ larger model but only a $1.8\times$ increase in tokens (exponents on the order of $N\propto C^{0.73}$, $D\propto C^{0.27}$). Following this, and GPT-3's setup, most large models were trained for roughly 300B tokens regardless of their size. One practice in this setting concerns the learning-rate schedule. With a cosine schedule, the learning rate decays (by about $10\times$) over the course of training and reaches its floor as training ends; when the cosine cycle length overshoots the actual number of training steps by more than about 25%, final performance is noticeably degraded. Much of the prior data also came from small models (a large fraction under 100M parameters).

**Risk decomposition of a trained predictor.** A classical way to analyze the expected risk of a trained predictor decomposes it into pieces. Predicting the next token is choosing $f:\mathcal X\to\mathcal D(\mathcal Y)$; the Bayes-optimal predictor $f^\star$ minimizes the cross-entropy over the true distribution, and its risk $L(f^\star)$ is the irreducible entropy of natural text. Restricting to Transformers of size $N$ gives the best-in-class $f_N$, and training for a single epoch over $D$ tokens with a finite number of gradient steps gives $\bar f_{N,D}$. Established results characterize the pieces of such a decomposition: function-approximation error for restricted classes tends to fall as a power of the dimension (for two-layer networks expected to scale like $N^{-1/2}$; Siegel, 2020), and the convergence of a stochastic first-order method under early stopping is lower-bounded by $D^{-1/2}$ and is dimension-free (Robbins & Monro, 1951).

**Numerical tools.** Robust regression with the Huber loss (Huber, 1964) downweights outliers; the log-sum-exp operator gives a numerically stable way to compute a log of a sum of exponentials; L-BFGS (Nocedal, 1980) is a standard quasi-Newton optimizer for smooth low-dimensional fits.

## Baselines

**Kaplan et al. (2020) — power-law scaling of language models.** Establishes that loss is a power law in $N$, $D$, and $C$, and gives an allocation rule that favours growing the model faster than the data ($N\propto C^{0.73}$, $D\propto C^{0.27}$). Core idea: fit loss as a function of size and compute from many runs, then extrapolate. Its allocation rests on loss estimates at intermediate token counts, with much of its data from small models and from runs read off long learning-rate schedules.

**Single-method fits in general.** Estimation procedures that can be applied to a table of runs include reading minima off training curves, profiling loss across model sizes at fixed compute, and fitting a global parametric loss surface. Each makes its own modeling choices (the interpolation, the curve fit to a profile, the assumed functional form).

## Evaluation settings

The quantity modeled is the smoothed final pretraining cross-entropy loss $L$ (a near-unbiased estimate of test loss in the effectively-infinite-data regime where the corpus exceeds the tokens consumed), as a function of $(N,D)$. The experimental material is a sweep of training runs: models ranging from under 70M to over 16B parameters, trained on 5B to over 400B tokens, with each configuration run for several token horizons and several FLOP budgets (nine fixed FLOP counts from $6\times10^{18}$ to $3\times10^{21}$ for the iso-compute profiles). The fitted laws are then extrapolated to a large target budget (on the order of $5.76\times10^{23}$ FLOPs) to predict the optimal $(N,D)$ there. Robustness is assessed by bootstrapping the runs and by repeating on additional corpora.

## Code framework

The primitives that already exist: a numerically stable log-sum-exp, a Huber penalty, and an L-BFGS optimizer (e.g. from SciPy or a deep-learning autodiff library). Given a table of training runs, each a triple $(N_i, D_i, L_i)$, together with iso-compute slices and per-run loss-vs-FLOPs curves, the task is to estimate the optimal allocation $N_{\text{opt}}(C),\ D_{\text{opt}}(C)$ from them.

```python
import numpy as np


def estimate_allocation(runs):
    # runs: table of (N_i, D_i, L_i), plus iso-compute slices and per-run
    #       loss-vs-FLOPs curves.
    # TODO: estimate N_opt(C), D_opt(C) from the runs.
    pass
```
