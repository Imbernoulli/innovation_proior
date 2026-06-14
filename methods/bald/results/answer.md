# BALD

BALD (Bayesian Active Learning by Disagreement) is an information-theoretic acquisition rule
for active learning. It scores a candidate input by the **mutual information between the
unknown label and the model parameters**, and queries the input that maximizes it. The key
move is to compute that mutual information in its **predictive (output-space) form** rather
than its parameter-space form, which makes it tractable, cheap, and well defined even for
nonparametric models. Intuitively, BALD prefers inputs on which the plausible models are each
*confident* yet *disagree* — i.e. points whose label resolves genuine model uncertainty rather
than irreducible label noise.

## Problem it solves

Pick the unlabeled input whose label most reduces uncertainty about the model parameters
`theta`. The principled myopic criterion is the expected drop in posterior entropy of `theta`,

```
argmax_x  H[theta | D]  -  E_{y ~ p(y|x,D)} [ H[theta | y, x, D] ].
```

In parameter space this is intractable (entropy of a high-dimensional or infinite-dimensional
posterior) and expensive (one re-inference per hypothetical label, `O(N_x N_y)` updates).

## Key idea

That objective is the conditional mutual information `I[theta, y | x, D]`. Mutual information
is symmetric, so it can be written with the entropies in **output** space:

```
I[theta, y | x, D]  =  H[y | x, D]  -  E_{theta ~ p(theta|D)} [ H[y | x, theta] ].
```

- **First term** `H[y|x,D]`: entropy of the marginal prediction (total uncertainty).
- **Second term** `E_theta[H[y|x,theta]]`: average uncertainty of each individual parameter
  setting (the aleatoric / observation-noise part).
- **Difference**: epistemic uncertainty = parameter disagreement. High when the marginal
  prediction is uncertain *but* each parameter setting is confident — the settings confidently
  disagree.

Why this form wins: entropies are in low-dimensional output space (finite even for
infinite-dimensional `theta`); `theta` is conditioned only on `D`, so only `O(1)` posterior
updates are needed; and the subtracted second term separates model uncertainty (a label can
fix it) from observation noise (it cannot). **Maximum Entropy Sampling** is the argmax-equivalent
case when the second term is zero or constant in `x`; **Query by Committee** is the degenerate
case that replaces the entropies with a hard vote (discarding confidence). Both fall out of
BALD.

## GP classifier: closed form

Model `f ~ GP(mu, k)`, probit likelihood `y|x,f ~ Bernoulli(Phi(f(x)))`. Any approximate
inference (EP / Laplace / ADF / sparse) gives a Gaussian posterior `f_x ~ N(mu_{x,D},
sigma^2_{x,D})`. Entropy in bits; `h(p) = -p log p - (1-p) log(1-p)`.

**First term**, via the exact probit–Gaussian convolution `∫ Phi(f) N(f|mu,s^2) df =
Phi(mu/sqrt(1+s^2))`:

```
H[y|x,D] ≈ h( Phi( mu_{x,D} / sqrt(sigma^2_{x,D} + 1) ) ).
```

**Second term**: the integral `∫ h(Phi(f)) N(f|mu,s^2) df` has no closed form. Taylor-expand
`ln h(Phi(x))` about 0 (even function, `h(Phi(0)) = 1` bit so the constant is 0, and
`f''(0) = -2/(pi ln 2)`):

```
ln h(Phi(x)) = -x^2/(pi ln 2) + O(x^4)   =>   h(Phi(x)) ≈ exp( -x^2 / (pi ln 2) ),
```

with the first omitted term of order `x^4` (peak abs error ~3e-3; <0.27% error in the integral).
Convolving this squared exponential against the Gaussian posterior, with `C = sqrt(pi ln 2 / 2)`:

```
E_f[H[y|x,f]] ≈ C / sqrt(sigma^2_{x,D} + C^2) · exp( -mu^2_{x,D} / (2 (sigma^2_{x,D} + C^2)) ).
```

**BALD-GPC objective** (maximize over `x`):

```
h( Phi( mu_{x,D} / sqrt(sigma^2_{x,D} + 1) ) )
   -  C / sqrt(sigma^2_{x,D} + C^2) · exp( -mu^2_{x,D} / (2 (sigma^2_{x,D} + C^2)) ),
   C = sqrt(pi ln 2 / 2).
```

Smooth and differentiable in `x` for usual kernels, so continuous queries can be found by
gradient ascent. Only `O(1)` posterior inference, so the accurate inference method (EP) is
affordable.

## Extensions

- **Nuisance parameters** `theta = {theta+, theta-}` (e.g. GP hyperparameters): maximize MI
  about `theta+` only, by integrating out `theta-`:

  ```
  H[ E_{p(theta+,theta-|D)}[y|x,theta+,theta-] ]
     - E_{p(theta+|D)}[ H[ E_{p(theta-|theta+,D)}[y|x,theta+,theta-] ] ].
  ```

- **Preference learning**: items `(u,v)`, `y=1` iff `u ≻ v`, model
  `P[y=1|u,v,f] = Phi((f(u)-f(v))/(sqrt(2) sigma_noise))`, wlog `sqrt(2) sigma_noise = 1`. Set
  `g(u,v) = f(u)-f(v)`; `g` is a GP (linear functional of a GP) with the anti-symmetric
  **preference kernel**

  ```
  k_pref((u_i,v_i),(u_j,v_j)) = k(u_i,u_j) - k(u_i,v_j) - k(v_i,u_j) + k(v_i,v_j).
  ```

  The likelihood is probit on `g`, so preference learning is GP classification and BALD applies
  unchanged.

## Deep nets: MC-dropout estimator

For an intractable weight posterior, draw `T` approximate posterior samples by `T` stochastic
forward passes with dropout left ON; `p^t = p(y|x,theta^t)` is the softmax of pass `t`. With
`p_bar = (1/T) Σ_t p^t`:

```
I(x) ≈ H[ p_bar ]  -  (1/T) Σ_t H[ p^t ]
     = - Σ_c p_bar_c log p_bar_c  +  (1/T) Σ_{t,c} p^t_c log p^t_c.
```

`I(x) ≥ 0` (Jensen). Query the `n` unlabeled points with the largest `I` by greedy top-`n`
ranking. `T ≈ 10` is a typical default.

## Working code

Fills the `query` slot of the pool-based active-learning harness, using the existing
stochastic-forward-pass primitive:

```python
import numpy as np
import torch
from .strategy import Strategy


class BALDDropout(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
        super(BALDDropout, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_drop = n_drop                       # T = number of posterior samples

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # T dropout forward passes -> probs[t, i, c] ~ p(y=c | x_i, theta^t)
        probs = self.predict_prob_dropout_split(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
        pb = probs.mean(0)                                       # p_bar = mean over T passes
        entropy1 = (-pb * torch.log(pb)).sum(1)                  # H[p_bar]  (total)
        entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)    # mean per-pass H (aleatoric)
        U = entropy2 - entropy1                                  # U = -(BALD mutual information)
        return idxs_unlabeled[U.sort()[1][:n]]                   # n smallest U = n largest I
```

The score is `entropy1 - entropy2 = I(x)`; the code ranks by `U = -I(x)` ascending and keeps
the first `n`, which are the `n` points of maximal mutual information (maximal model
disagreement).
