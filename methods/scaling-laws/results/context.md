## Research question

Transformer language models get better as they get bigger, are trained on more data, and are given more compute — but this knowledge is qualitative and anecdotal. Practitioners decide model size, dataset size, and training duration by intuition and by what fits on the available hardware. The question is whether a *predictive quantitative theory* exists: how does test loss $L$ depend on the controllable scales — number of (non-embedding) parameters $N$, dataset size $D$, training compute $C$, number of optimization steps $S$, and batch size $B$ — accurately enough to extrapolate? If such laws exist and are simple, they would answer the practical question that matters before any large run: given a fixed compute budget, how should it be divided between making the model bigger and training it longer?

## Background

**Performance is set by scale, not architecture details.** A striking empirical regularity underlies the whole enterprise: holding the non-embedding parameter count $N$ fixed, Transformer test loss varies only a few percent across very different shapes — depth, width, aspect ratio, number of attention heads can be moved over wide ranges (aspect ratio by a factor of 40) with little effect. So the loss is governed almost entirely by *scale*, and a one-number summary $N$ is meaningful. This is what makes a low-dimensional scaling theory possible at all.

**Power laws across orders of magnitude.** The second empirical regularity: when performance is bottlenecked by only one of $N$, $D$, or $C$ (the other two made abundant), test loss falls as a clean power law in that quantity, with no sign of bending over many orders of magnitude. Power laws of loss versus scale had been observed before across deep learning (Hestness et al., 2017; Rosenfeld et al., 2019); the task here is to nail them down quantitatively for Transformer language models and to combine them.

**Counting parameters and compute.** To state the laws one needs unambiguous definitions. For a Transformer with residual width $d_{\text{model}}$, $n_{\text{layer}}$ layers, feed-forward width $d_{\text{ff}}$, and attention width $d_{\text{attn}}$, summing the projection matrices (query/key/value, attention output, and the two feed-forward matrices) gives the non-embedding parameter count $N \approx 2 d_{\text{model}} n_{\text{layer}}(2d_{\text{attn}} + d_{\text{ff}})$, which with the conventional shape $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$ reduces to $N \approx 12\, n_{\text{layer}} d_{\text{model}}^2$. The embedding and positional-embedding parameters are deliberately excluded — including them muddies the laws. A forward pass costs about $C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ FLOPs per token (the leading 2 is the multiply-accumulate); when $d_{\text{model}} \gg n_{\text{ctx}}/12$ the context-dependent term is small, leaving $\approx 2N$, and including the backward pass (about twice the forward) gives the working estimate $C \approx 6N$ FLOPs per token, hence $C \approx 6ND$ over $D$ tokens.

**Batch-size / training-efficiency theory.** The relationship between batch size, number of steps, and compute is governed by a critical batch size $B_{\text{crit}}$ (McCandlish et al., 2018): below it, increasing the batch buys faster training at almost no extra compute; above it, returns diminish. The theory predicts $(S/S_{\text{min}} - 1)(E/E_{\text{min}} - 1) = 1$ for training to a fixed loss, with $S_{\text{min}}$ the minimum steps and $E_{\text{min}}$ the minimum examples, and $B_{\text{crit}} \equiv E_{\text{min}}/S_{\text{min}}$. $B_{\text{crit}}$ tracks the gradient noise scale, depends on the *loss attained* (not directly on model size), and grows as the loss falls. This is the tool needed to compare runs done at different (non-optimal) batch sizes on a common footing.

**Tokenization changes the loss only by a scale.** Changing the vocabulary or tokenization scheme leaves the model unchanged but rescales the measured cross-entropy by an overall factor (a different number of "tokens" carrying the same information). So any normalization constants tied to units of loss carry no fundamental meaning across tokenizations.

## Baselines

**Pre-existing scaling observations (Hestness et al., 2017; Rosenfeld et al., 2019).** Established that generalization error follows power laws in data and model size across several domains. Core idea: empirically fit error-vs-scale and observe power-law regions. The trends are reported per-factor and cover multiple domains, though not specifically Transformer language models.

**Hyperparameter-tuning / shape-search practice.** The prevailing way to improve a model was to tune architecture (depth/width/heads) and optimization at a fixed size. The empirical near-independence of loss from shape at fixed $N$ means that the dominant lever is scale, not shape.

**Single-factor extrapolation.** One can fit $L(N)$ alone, or $L(D)$ alone, or the empirical $L(C)$ at fixed batch size. $L(N)$ captures the parameter-count dependence; $L(D)$ captures the dataset-size dependence; and an $L(C)$ measured at a fixed batch size provides a compute-vs-loss trend at the given training configuration.

## Evaluation settings

The measured quantity is autoregressive cross-entropy test loss (nats/token) on a held-out split, with models trained on WebText2 and tested both there and on other text distributions to check universality. Models span over six orders of magnitude in non-embedding parameters; datasets are taken in fixed subsets down to small sizes to probe overfitting (with early stopping when test loss stops improving, and 10% dropout in the finite-data study); compute is measured in PetaFLOP-days. Context length is 1024 tokens. Training uses Adam (Adafactor for the largest models), a fixed step budget with warmup and cosine decay, and a range of batch sizes to measure $B_{\text{crit}}$. Fits are performed in log-log space; fit constants tied to vocabulary/tokenization carry no fundamental meaning.

## Code framework

The primitives that already exist: linear-regression / curve-fitting in log space (NumPy/SciPy), and the Transformer hyperparameter conventions. Given a set of training runs — each a tuple of scale quantities and the converged/early-stopped loss — the pieces to fill in start from the parameter- and compute-counting helpers and the single-variable power-law fit; the remaining slots are left open below.

```python
import numpy as np

def transformer_param_count(n_layer, d_model, d_ff=None, d_attn=None):
    # TODO: non-embedding parameter count N as a function of shape.
    pass


def forward_flops_per_token(N, n_layer, d_model, n_ctx):
    # TODO: forward-pass FLOPs per token; training compute per token is ~3x this.
    pass


def fit_power_law(X, L):
    # X: scale values (N or D or C); L: losses. TODO: fit L = (X_c / X)**alpha
    #    in log-log space; return (X_c, alpha).
    pass


def joint_loss(N, D, params):
    # TODO: a loss model L(N, D) over the two scales jointly.
    pass


def fit_joint_loss(runs):
    # runs: list of (N_i, D_i, L_i). TODO: fit the joint-law parameters.
    pass


def early_stopping_lower_bound(N, D, params, S_c, alpha_S):
    # TODO.
    pass


def compute_optimal_exponents(alpha_N, alpha_S, alpha_B):
    # TODO: how the controllable scales should grow with the compute budget.
    pass
```
