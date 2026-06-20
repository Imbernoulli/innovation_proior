The task is to denoise a single-cell RNA-seq UMI count matrix: the observed counts are a brutally
undersampled, dropout-riddled view of an underlying expression rate, and I want a function that maps
the noisy matrix to a smoother non-negative one closer to that rate. Everything is scored relatively,
through molecular cross-validation: I binomially thin the observed counts into two independent
Poisson halves, denoise one half, and measure how well the result predicts the other. Each metric —
log-normalized MSE and Poisson negative log-likelihood — is then normalized so that the raw counts
sit at exactly zero and the true rate sits at exactly one, and the combined score is the mean of the
two. Before I write anything that pools information across cells, I need to know what doing *nothing*
scores, because the entire scale is anchored on it. If I never run the do-nothing case through the
harness, I cannot tell whether the normalization brackets the methods correctly, whether the two
anchors really bound the achievable range, or whether a sign error has crept into the metric so that
"better" comes out negative.

What I propose for this first step is therefore not a denoiser at all but a calibration: the raw
baseline, the identity map $\hat X = X_{\text{train}}$. Its value is entirely diagnostic, and its
behavior is something I can predict exactly. Consider the log-normalized MSE term. It compares the
log-normalized training half against the log-normalized test half, and these are two independent
half-depth Poisson draws of the same latent rate, so they disagree on every entry by the sampling
noise alone — large, and a pure measurement of how noisy the data is. That large value is the correct
zero point: any real denoiser must shrink it by borrowing strength across cells. The Poisson term
tells the same story from the likelihood side. I rescale the training half to the test molecule
budget and treat it as a rate $\lambda$, then score $\text{mean}(\lambda - X_{\text{test}}\cdot
\log(\lambda+\varepsilon))$; predicting the noisy training half as the rate carries all of its
sampling noise into $\lambda$, so the NLL is poor, while the true-rate anchor, being the smooth thing
the counts were actually drawn from, scores far better. The decisive property is that the "method"
matrix here *is* the raw anchor the normalization is built from, so the numerator $\text{raw} -
\text{method}$ is identically zero and both normalized terms must come out as a clean $0.0000$ on
every dataset — not approximately, exactly. Anything else is a bug in the harness, not the method,
and I would much rather catch it now than three rungs up when I am reading a small improvement off a
miscalibrated scale.

There is nothing to tune; the denoiser is the identity. What it cannot do is the whole point and is
worth naming plainly: it pools no information across cells. The premise of scRNA-seq denoising is
that cells in the same biological state are independent noisy measurements of one rate, so averaging
similar cells beats down the Poisson noise — and the identity does none of that, leaving the entire
distance from zero to one for the methods that actually borrow strength between cells.

```python
import numpy as np

def denoise(X):
    """Raw baseline: no denoising. The evaluator's zero anchor."""
    return X.astype(np.float64).copy()
```
