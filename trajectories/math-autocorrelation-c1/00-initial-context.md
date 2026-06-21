## Research question

Among all non-negative functions `f` supported on `[-1/4, 1/4]`, how small can the ratio

```
R(f) = max_{|t|≤1/2} (f*f)(t) / (∫_{-1/4}^{1/4} f)^2
```

be made? The first autocorrelation inequality of Barnard and Steinerberger asks for the largest constant `C1` such that

```
max_{|t|≤1/2} (f*f)(t) ≥ C1 · (∫ f)^2
```

for every admissible `f`. The value `C1` is the infimum of `R(f)`.

The construction class is fixed to non-negative piecewise-constant step functions

```
f = Σ_{n=0}^{N−1} a_n · 1_[n, n+1),   a_n ≥ 0.
```

Translation and dilation invariance mean only the heights `a_n` and their count `N` matter. For such functions the autoconvolution is piecewise linear, and its maximum is attained at a discrete node. Discretizing gives the scale-normalized score

```
R(a) = 2 N · max_k (a*a)_k / (Σ_n a_n)^2,
```

where `(a*a)_k = Σ_n a_n a_{k−n}`. Every admissible height vector certifies `C1 ≤ R(a)`; lower `R` is better.

## Prior art / Background / Baselines

- **Flat indicator.** Uniform heights produce a triangular autoconvolution and score `R = 2.0`. This is far above the known lower bound, leaving a large constructive gap.

- **Best published construction.** A numerically optimized non-uniform step function has achieved `R = 1.5098`. It improves on the flat indicator, but the value is still well above the analytic lower bound, and the construction offers no clear structural principle for further reduction.

- **Analytic lower bound (Cloninger–Steinerberger, 2017).** Analytic arguments prove `C1 ≥ 1.28`. This is not an explicit function, so it provides a floor without showing what shape might approach it.

## Fixed substrate / Code framework

The harness is a deterministic evaluator. It clips the input vector to `[0, 1000]`, computes the self-convolution with an FFT, takes the peak, and returns `R`. The formula and clipping are frozen.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    """R = 2N * max(a*a) / (sum a)^2 for f = sum_n a_n 1_[n,n+1). Lower = tighter upper bound on C1."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)                  # length 2N-1: b_k = sum_n a_n a_{k-n}
    peak = float(np.max(b))                # max of the piecewise-linear autoconvolution = max node
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * peak / (s * s)
```

## Editable interface

The only editable component is the constructor that produces the height vector `a`. It may return any Python list of non-negative finite floats of length `N ≥ 1`. The harness clips negatives and large values internally, so the constructor is free to choose structured, optimized, or searched heights.

## Evaluation settings

A single deterministic functional `R(a)`. If a constructor uses randomness, the run is fixed to a stated seed for reproducibility. The reference points are the flat ceiling `2.0`, the best published construction `1.5098`, and the analytic lower bound `1.28`. The returned score is the whole result; lower is better.
