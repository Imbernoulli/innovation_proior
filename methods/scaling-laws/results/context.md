## Research question

Transformer language models clearly get better as they get bigger, are trained on more data, and are given more compute — but this knowledge is qualitative and anecdotal. Practitioners decide model size, dataset size, and training duration by intuition and by what fit on the available hardware. The precise problem is to replace that intuition with a *predictive quantitative theory*: write down how the test loss $L$ depends on the controllable scales — number of (non-embedding) parameters $N$, dataset size $D$, training compute $C$, number of optimization steps $S$, and batch size $B$ — accurately enough to extrapolate. If such laws exist and are simple, they answer the practical question that matters before any large run: given a fixed compute budget, how should it be divided between making the model bigger and training it longer, and how big a model and how much data should one choose? A solution must (1) define $N$ and $C$ cleanly enough that the laws come out clean, (2) establish the functional forms of $L$ versus each scale and fit their constants, (3) combine them into a joint law that captures finite-data overfitting and finite-time training, and (4) use that joint law to derive the compute-optimal allocation.

## Background

**Performance is set by scale, not architecture details.** A striking empirical regularity underlies the whole enterprise: holding the non-embedding parameter count $N$ fixed, Transformer test loss varies only a few percent across very different shapes — depth, width, aspect ratio, number of attention heads can be moved over wide ranges (aspect ratio by a factor of 40) with little effect. So the loss is governed almost entirely by *scale*, and a one-number summary $N$ is meaningful. This is what makes a low-dimensional scaling theory possible at all.

**Power laws across orders of magnitude.** The second empirical regularity: when performance is bottlenecked by only one of $N$, $D$, or $C$ (the other two made abundant), test loss falls as a clean power law in that quantity, with no sign of bending over many orders of magnitude. Power laws of loss versus scale had been observed before across deep learning (Hestness et al., 2017; Rosenfeld et al., 2019); the task here is to nail them down quantitatively for Transformer language models and to combine them.

**Counting parameters and compute.** To state the laws one needs unambiguous definitions. For a Transformer with residual width $d_{\text{model}}$, $n_{\text{layer}}$ layers, feed-forward width $d_{\text{ff}}$, and attention width $d_{\text{attn}}$, summing the projection matrices (query/key/value, attention output, and the two feed-forward matrices) gives the non-embedding parameter count $N \approx 2 d_{\text{model}} n_{\text{layer}}(2d_{\text{attn}} + d_{\text{ff}})$, which with the conventional shape $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$ reduces to $N \approx 12\, n_{\text{layer}} d_{\text{model}}^2$. The embedding and positional-embedding parameters are deliberately excluded — including them muddies the laws. A forward pass costs about $C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ FLOPs per token (the leading 2 is the multiply-accumulate); when $d_{\text{model}} \gg n_{\text{ctx}}/12$ the context-dependent term is small, leaving $\approx 2N$, and including the backward pass (about twice the forward) gives the working estimate $C \approx 6N$ FLOPs per token, hence $C \approx 6ND$ over $D$ tokens.

**Batch-size / training-efficiency theory.** The relationship between batch size, number of steps, and compute is governed by a critical batch size $B_{\text{crit}}$ (McCandlish et al., 2018): below it, increasing the batch buys faster training at almost no extra compute; above it, returns diminish. The theory predicts $(S/S_{\text{min}} - 1)(E/E_{\text{min}} - 1) = 1$ for training to a fixed loss, with $S_{\text{min}}$ the minimum steps and $E_{\text{min}}$ the minimum examples, and $B_{\text{crit}} \equiv E_{\text{min}}/S_{\text{min}}$. $B_{\text{crit}}$ tracks the gradient noise scale, depends on the *loss attained* (not directly on model size), and grows as the loss falls. This is the tool needed to compare runs done at different (non-optimal) batch sizes on a common footing.

**Risk intuition for the functional forms.** Two intuitions shape the ansatz. Overfitting at large dataset size should scale like the dataset variance, $\propto 1/D$, motivating an analytic $1/D$ expansion. And a loss model should be invariant in form under a change of vocabulary/tokenization, which only rescales the loss by an overall factor — so any normalization constants ($N_c$, $D_c$) must be free to absorb that rescaling and carry no fundamental meaning.

## Baselines

**Pre-existing scaling observations (Hestness et al., 2017; Rosenfeld et al., 2019).** Established that generalization error follows power laws in data and model size across several domains. Core idea: empirically fit error-vs-scale and observe power-law regions. The gap: not specific to Transformer language models, not unified into a joint $L(N,D)$ law, and not carried through to a compute-optimal allocation rule — they chart the phenomenon without delivering the predictive allocation recipe.

**Hyperparameter-tuning / shape-search practice.** The prevailing way to improve a model was to tune architecture (depth/width/heads) and optimization at a fixed size. The gap this leaves: the empirical near-independence of loss from shape at fixed $N$ means this effort yields only a few percent, while scaling $N$, $D$, $C$ yields orders of magnitude — so shape search is the wrong lever, and there was no quantitative law saying so.

**Single-factor extrapolation.** One can fit $L(N)$ alone, or $L(D)$ alone, or the empirical $L(C)$ at fixed batch size. Each is a real baseline trend but incomplete: $L(N)$ ignores finite data and overfitting; $L(D)$ ignores capacity; and an $L(C)$ measured at a fixed, non-critical batch size conflates compute-efficiency with the underlying trend and so extrapolates poorly. The gap: a trustworthy compute law needs runs standardized to the critical batch size and a joint law tying the factors together.

## Evaluation settings

The measured quantity is autoregressive cross-entropy test loss (nats/token) on a held-out split, with models trained on WebText2 and tested both there and on other text distributions to check universality. Models span over six orders of magnitude in non-embedding parameters; datasets are taken in fixed subsets down to small sizes to probe overfitting (with early stopping when test loss stops improving, and 10% dropout in the finite-data study); compute is measured in PetaFLOP-days. Context length is 1024 tokens. Training uses Adam (Adafactor for the largest models), a fixed step budget with warmup and cosine decay, and a range of batch sizes to measure $B_{\text{crit}}$. Fits are performed in log-log space; fit constants tied to vocabulary/tokenization carry no fundamental meaning.

## Code framework

The primitives that already exist: linear-regression / curve-fitting in log space (NumPy/SciPy), and the Transformer hyperparameter conventions. Given a set of training runs — each a tuple of scale quantities and the converged/early-stopped loss — the pieces to fill in are the parameter- and compute-counting helpers, the single-variable power-law fit, the joint loss model and its fit, the finite-data stopping estimate, and the routine that turns fitted exponents into the compute-optimal allocation.

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
    # TODO: a single law L(N, D) reducing to the N-only and D-only laws in the
    #       appropriate limits, with finite-D overfitting built in.
    pass


def fit_joint_loss(runs):
    # runs: list of (N_i, D_i, L_i). TODO: fit the joint-law parameters.
    pass


def early_stopping_lower_bound(N, D, params, S_c, alpha_S):
    # TODO: estimate the earliest finite-data stopping step from the gap between
    #       the finite-D loss and the infinite-D loss for the same model size.
    pass


def compute_optimal_exponents(alpha_N, alpha_S, alpha_B):
    # TODO: from the per-factor exponents, derive how the optimal model size,
    #       batch, steps, and data should scale with the compute budget.
    pass
```
