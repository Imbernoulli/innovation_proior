I am minimizing the normalized peak of the autoconvolution of a non-negative step function, $R = 2N \cdot \max_k (a*a)_k / (\sum_n a_n)^2$, where every admissible $f$ certifies an upper bound $C_1 \le R(f)$ on the first autocorrelation constant of Barnard–Steinerberger, so *lower is better*. Before chasing any clever shape I need to fix the ceiling: the value of $R$ on the simplest legal function, so I know what the functional even feels like and exactly how much every later rung has to claw back by going *below* it. This is the opposite reflex from the maximization sibling — here the baseline is a ceiling I push down, not a floor I climb toward.

The construction is the flat profile: all heights equal, $a_n = 1$ for every piece, which is just the discretized indicator of an interval. Its whole appeal is that the autoconvolution is the cleanest object in the class, so I can compute its score by hand and *understand* the functional rather than read a number off the evaluator. Take $f$ to be the indicator of an interval; by dilation invariance the length does not matter, so fix the support to be the unit interval. Its autoconvolution is the classic triangle — the "tent" — supported on twice the width, rising linearly to a single peak at the midpoint and falling back to zero. The peak is the key quantity, because the numerator of $R$ is exactly the peak of $f*f$. For the indicator of width $w$ and height $h$ the autoconvolution peaks at $h^2 w$ (full overlap at the center) while the mass is $\int f = hw$, so the continuous ratio is

$$\frac{\text{peak}}{(\int f)^2} = \frac{h^2 w}{h^2 w^2} = \frac{1}{w}.$$

The discrete evaluator carries a normalization factor that turns this dimensionless ratio into the constant on the fixed interval $[-1/4, 1/4]$, and I want to *trust* that factor rather than take it on faith. With $a_n = 1$ for $n = 0, \dots, N-1$, the discrete self-convolution $b = a*a$ is the triangular sequence $1, 2, 3, \dots$ rising to its peak and falling back symmetrically; its maximum is the center value $N$, where all $N$ overlaps line up. The mass is $\sum_n a_n = N$, so $(\sum_n a_n)^2 = N^2$, and the evaluator returns

$$R = \frac{2N \cdot \max_k b_k}{(\sum_n a_n)^2} = \frac{2N \cdot N}{N^2} = 2$$

exactly. The factor $2N$ is precisely the bookkeeping that makes this come out to the trivial continuous bound of $2$, so matching it by hand validates the harness for the harder rungs.

The load-bearing fact this rung exposes is that the score $2$ is *independent of $N$*. A flat vector of ten ones and a flat vector of a thousand ones both have a triangular autoconvolution and both score exactly $2$ — refining a flat profile leaves the triangle, and the value, untouched. So the piece count is, on its own, a red herring; only the *shape* of the heights moves $R$. That tells me where every later gain must come from: not from finer grids but from making the autoconvolution *less peaked* relative to its mass. The flat function is the unique maximally-symmetric member of the class — it maximizes the central self-overlap, since every piece aligns with every piece near zero shift, which is exactly the worst case for this minimization — and it has no internal degree of freedom to spend, no gradient to follow, because every piece is identical and $f*f$ is locked to a triangle. To get below $2$ I have to break that symmetry: introduce variation among the heights so that no single shift of $f$ against itself lines up too much overlap, which intuitively wants an asymmetric, structured profile — heavier near the ends, lighter in the middle, or genuinely irregular. The flat function buys none of that drop; it establishes the starting altitude at $2.0$, with the provable floor $C_1 \ge 1.28$ and the published record $1.5028628969$ marking how far below the bottom lies. As a final cross-check the same evaluator should score the published record sequence at $1.50286$, confirming the machine agrees with the literature on both ends.

```python
import numpy as np
from scipy.signal import fftconvolve

def autoconv_c1_ratio(a):
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1000.0)
    N = len(a)
    b = fftconvolve(a, a)
    peak = float(np.max(b))
    s = float(np.sum(a))
    if s < 0.01:
        return float("inf")
    return 2.0 * N * peak / (s * s)

def construct(N: int = 600):
    """Flat profile = discretized indicator; autoconvolution is a triangle, R = 2 exactly."""
    return [1.0] * N

if __name__ == "__main__":
    print(autoconv_c1_ratio(construct()))   # 2.0000000000000004
```
