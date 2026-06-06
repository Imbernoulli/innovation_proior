# The Parks–McClellan algorithm: optimal equiripple FIR design via the Remez exchange

## Problem

Design a length-`N` linear-phase FIR filter whose magnitude response best matches a desired
piecewise spec (passbands → gain 1, stopbands → gain 0, named transition gaps) in the **minimax
(Chebyshev) sense** — minimize the worst-case weighted error across the bands — while letting the
designer **fix the band edges exactly**. The window method is least-squares-optimal but its Gibbs
overshoot (~9%) never shrinks with `N` and it gives no edge control; frequency sampling and
maximal-ripple methods either don't minimize ripple directly or can't accept prescribed edges; an
LP minimizes the worst case but is slow. Parks–McClellan attains all three: minimax error, exact
edges, and speed by exploiting the equioscillation structure of the optimum.

## Key idea

1. **Linear phase ⇒ real trig amplitude, four cases.** With `h(n) = ±h(N-1-n)`, the response is
   `H = G(f)·e^{j·linear-phase}` with `G(f)` real. Crossing symmetry (±) with parity (`N` odd/even)
   gives four amplitude forms (cosine or sine sums).

2. **Unify to one cosine problem.** Every case factors as `G(f) = Q(f)·P(f)` with
   `P(f) = Σ_{k=0}^{r-1} α_k cos(2πkf)` a pure cosine polynomial and `Q ∈ {1, cos πf, sin πf,
   sin 2πf}`. Absorbing `Q` into the target and weight, `D̂ = D/Q`, `Ŵ = W·Q`, leaves a single
   problem: best weighted-Chebyshev approximation of `D̂` by the cosine polynomial `P`. With
   `x = cos(2πf)`, `cos(2πkf) = T_k(x)`, so this is ordinary degree-`(r-1)` polynomial
   approximation on `[-1,1]`.

3. **Alternation theorem.** `P` is the unique best weighted minimax approximation iff the weighted
   error `E(f) = Ŵ(f)[D̂(f) - P(f)]` attains its maximum magnitude `δ` with **alternating sign at
   at least `r+1` frequencies** `F_1 < ... < F_{r+1}`. This converts the min-max into a finite,
   checkable certificate.

4. **Closed-form level + cheap evaluation.** Given a reference set, the alternation equations
   `Ŵ(F_i)[D̂(F_i) - P(F_i)] = (-1)^i δ` give the deviation in closed form (`b_k` are the
   barycentric/Lagrange weights on the nodes `x_i = cos(2πF_i)`):

   ```
   δ_raw = ( Σ_k b_k D̂(F_k) ) / ( Σ_k (-1)^k b_k / Ŵ(F_k) ),
   δ = |δ_raw|, flipping the alternating signs if δ_raw < 0.
   ```

   and `P` is evaluated anywhere by barycentric Lagrange interpolation through all `r+1`
   ordinates `y_i = D̂(F_i) - (-1)^i δ/Ŵ(F_i)` after that `δ` makes the order-`r`
   divided difference vanish — no coefficient solve per iteration.

5. **Remez exchange.** Guess `r+1` references (equispaced); compute `δ` and `P`; evaluate the error
   on a dense grid (≈ `16·r` points); replace the reference with the `r+1` alternating local maxima
   of `|E|`; repeat. `δ` increases monotonically toward the optimum; stop when the dense-grid max
   of `|E|` no longer exceeds `δ` — that is the equioscillation certificate. Then recover the `α_k`
   by inverse DFT of `P` at `r` equispaced points, undo `Q`, and impose the symmetry to fill out
   `h(n)`.

## Algorithm

```
Inputs: numtaps N, bands, desired gains D, weights W, filter type.
1. From symmetry/parity pick the case; r = number of cosine terms; build dense grid on the bands.
2. Form D_hat = D/Q, W_hat = W*Q on the grid (Q from the case).
3. Initialize r+1 equispaced extremal indices.
4. Repeat (<= maxiter):
     x_i = cos(2*pi*F_i);  b_i = Lagrange weights;
     raw_delta = (sum b_i D_hat_i) / (sum (-1)^i b_i / W_hat_i);
     delta = abs(raw_delta), flipping the starting sign if raw_delta < 0;
     y_i = D_hat_i - (-1)^i delta / W_hat_i;
     signed_err(f) = W_hat(f)*(P(f) - D_hat(f)) on the dense grid;
     new extrema = r+1 alternating local maxima of |signed_err|;
     if max|signed_err| <= delta: converged.
5. alpha = inverse-DFT of P at r equispaced freqs; fold alpha (undo Q) into symmetric h(n).
```

## Code

The working engine is the Fortran-derived `remez`/`pre_remez` C routine exposed through
`scipy.signal.remez`. Typical use:

```python
import numpy as np
from scipy.signal import remez, freqz

# --- Lowpass: 8 kHz cutoff, 100 Hz transition, 325 taps, fs = 22.05 kHz ---
fs = 22050
taps = remez(325, [0, 8000, 8100, 0.5*fs], [1, 0], fs=fs)

# --- Bandpass: pass 2-5 kHz, 260 Hz transitions, 63 taps ---
edges = [0, 1740, 2000, 5000, 5260, 0.5*fs]
bp = remez(63, edges, [0, 1, 0], fs=fs)

# --- Stopband weighted 10x tighter than passband ---
lp = remez(101, [0, 0.1, 0.15, 0.5], [1, 0], weight=[1, 10])

# --- A wideband differentiator and a Hilbert transformer (antisymmetric h) ---
diff = remez(31, [0, 0.5], [1.0], type='differentiator')
hilb = remez(31, [0.05, 0.45], [1.0], type='hilbert')

w, H = freqz(taps, worN=2000, fs=fs)   # inspect the equiripple magnitude response
```

A faithful from-scratch implementation of the exchange core (mirroring the engine — barycentric
`freq_eval`, the closed-form `δ`, the alternating extremal search, inverse-DFT recovery):

```python
import numpy as np

def lagrange_weights(x):
    """b_k = 1 / prod_{j!=k} (x_k - x_j) on the reference nodes x_i = cos(2*pi*F_i)."""
    r = len(x); b = np.ones(r)
    for k in range(r):
        for j in range(r):
            if j != k:
                b[k] /= (x[k] - x[j])
    return b

def eval_P(xq, x, y, b):
    """Barycentric Lagrange value of P at cos-mapped points xq, interpolating y on nodes x."""
    xq = np.asarray(xq, dtype=float)
    out = np.empty_like(xq, dtype=float)
    for idx, xqi in np.ndenumerate(xq):
        hit = np.isclose(xqi, x, rtol=0.0, atol=1e-14)
        if np.any(hit):
            out[idx] = y[np.argmax(hit)]
        else:
            c = b / (xqi - x)
            out[idx] = (c @ y) / c.sum()
    return out

def pick_alternating_extrema(err, n_ext):
    """Return n_ext alternating local maxima of |err|, keeping the largest same-sign lobe."""
    peaks = []
    for i, value in enumerate(err):
        left = abs(err[i - 1]) if i else -np.inf
        right = abs(err[i + 1]) if i + 1 < len(err) else -np.inf
        if abs(value) >= left and abs(value) >= right:
            peaks.append(i)

    def compress(indices):
        out = []
        for i in indices:
            if err[i] == 0:
                continue
            if not out or np.sign(err[i]) != np.sign(err[out[-1]]):
                out.append(i)
            elif abs(err[i]) > abs(err[out[-1]]):
                out[-1] = i
        return out

    ext = compress(peaks)
    while len(ext) > n_ext:
        drop = min(range(len(ext)), key=lambda k: abs(err[ext[k]]))
        del ext[drop]
        ext = compress(ext)
    if len(ext) != n_ext:
        raise RuntimeError("could not find enough alternating extrema")
    return np.array(ext, dtype=int)

def remez_exchange(grid, des, wt, r, maxiter=25):
    """grid: dense freqs in [0,0.5]; des=D_hat, wt=W_hat prepared on the grid; r cosine terms."""
    step = (len(grid) - 1) / r
    ext = np.array([int(j * step) for j in range(r)] + [len(grid) - 1])
    dev = 0.0
    for _ in range(maxiter):
        x = np.cos(2*np.pi*grid[ext]); b = lagrange_weights(x)
        signs = (-1.0) ** np.arange(r + 1)
        raw_dev = (b @ des[ext]) / ((signs * b) @ (1.0 / wt[ext]))
        if raw_dev < 0:
            signs = -signs
            dev = -raw_dev
        else:
            dev = raw_dev
        y = des[ext] - signs * dev / wt[ext]                      # ordinates P interpolates
        P = eval_P(np.cos(2*np.pi*grid), x, y, b)
        err = (P - des) * wt                                      # weighted error on dense grid
        new_ext = pick_alternating_extrema(err, r + 1)             # r+1 alternating local maxima
        if np.max(np.abs(err)) <= dev * (1 + 1e-6):
            break
        ext = new_ext
    # cosine coefficients alpha_k via inverse DFT of P at r equispaced frequencies
    fpts = np.arange(r) / (2*r - 1)
    x = np.cos(2*np.pi*grid[ext]); b = lagrange_weights(x)
    signs = (-1.0) ** np.arange(r + 1)
    raw_dev = (b @ des[ext]) / ((signs * b) @ (1.0 / wt[ext]))
    if raw_dev < 0:
        signs = -signs
        dev = -raw_dev
    else:
        dev = raw_dev
    y = des[ext] - signs * dev / wt[ext]
    Pvals = eval_P(np.cos(2*np.pi*fpts), x, y, b)                 # P sampled at the r points
    cn = 2*r - 1
    alpha = np.empty(r)
    for j in range(r):                                            # inverse cosine transform of P
        s = sum(Pvals[k] * np.cos((2*np.pi/cn) * j * k) for k in range(1, r))
        alpha[j] = (2.0*s + Pvals[0]) * (2.0/cn if j else 1.0/cn)
    return alpha, dev
```

The `pick_alternating_extrema` step keeps, among same-sign error lobes, the grid point of largest
`|E|`, returning `r+1` of them sorted in frequency so the alternation pattern is preserved; the full
engine adds endpoint bookkeeping around this same exchange rule. Folding `alpha` back through `Q` and imposing
`h(n) = ±h(N-1-n)` produces the final impulse response.
