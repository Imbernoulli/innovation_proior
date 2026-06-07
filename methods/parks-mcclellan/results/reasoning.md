I want a linear-phase FIR filter of a fixed length `N` that meets a magnitude spec — say a lowpass: gain 1 up to some passband edge, gain 0 past some stopband edge, with a transition gap between them I don't care about. And I want it to be as good as a length-`N` filter can possibly be. So before anything else I should be honest about what "as good as possible" means, because that single choice is going to decide the whole method.

The obvious starting point is the window method. Take the ideal brick-wall lowpass, whose impulse response is the doubly-infinite sinc `h_d(n) = sin(ω_c n)/(π n)` away from the origin and `ω_c/π` at the origin, chop it down to `N` samples, and taper the ends with a window so the truncation isn't so violent. Let me think about what that actually buys me. Truncating to `N` samples and keeping the central `h_d(n)` is, term by term, the least-squares answer: with the support fixed to the central `N` taps, the integral-squared error is by Parseval the sum of squared coefficient errors, so within that fixed support the best I can do is keep each `h_d(n)` exactly — it's the orthogonal projection of the ideal response onto the fixed length-`N` subspace, and the leftover error is just the energy of the coefficients `|n|>M` that the support forces me to drop. Fine. So the rectangular window is *optimal in ISE*. But stare at what it does at the band edge. The truncation convolves the brick wall with the Dirichlet kernel (the transform of the rectangular window), and that produces the Gibbs phenomenon: a roughly 9% overshoot right at the discontinuity. And here's the thing that bugs me — that 9% doesn't go away as I make `N` larger. Longer filter, narrower "ears," same height. The worst-case ripple is *stuck* at about 9% no matter how many taps I spend.

That's the tell. Least-squares is the wrong objective for a filter spec. A spec is a worst-case statement — "passband ripple no more than δ_p, stopband down at least so many dB" — and the rectangular window, which is the *best* ISE answer, fails the worst-case criterion no matter how much I spend on it. Tapered windows (Hamming, Blackman, Kaiser) fix the overshoot — they push the sidelobes down 40, 80 dB — but they do it by *not minimizing* anything; they trade main-lobe width for sidelobe height by a fixed recipe, and crucially the band edge ends up wherever the windowed sum happens to put it. I get no direct grip on the cutoff frequency. For meeting a spec that names the edges, that's disqualifying on its own.

So the objective I actually want is the minimax one: minimize the *maximum* weighted deviation over the bands,

‖E‖ = max over the bands of  W(f) · |D(f) − G(f)|,

where `D(f)` is the desired amplitude (1 in passbands, 0 in stopbands), `G(f)` is what my filter realizes, and `W(f)` is a weight that lets me say "I care 10× more about stopband ripple than passband ripple." I want to drive that single number, the worst-case weighted error, to its smallest possible value for the given `N`, with the band edges fixed by me, not by the method.

Let me also rule out the other methods I know, to make sure minimax-with-fixed-edges is really the gap. Frequency sampling: set the response equal to the target at `N` equally spaced frequencies and inverse-DFT to get the taps. Exact at the samples, but it ripples between them, and the standard fix — free up a couple of transition-band samples and choose them to cancel ripple — works because `G` is linear in those samples, but it only optimizes ripple through a handful of knobs and it pins my edges to the sample grid. Not it. Linear programming: since `G(f)` will turn out to be linear in its coefficients, I could literally write "minimize δ subject to `−δ ≤ W(f_i)[D(f_i) − G(f_i)] ≤ δ` at every grid point `f_i`" and run a simplex. That genuinely minimizes the worst case *and* lets me fix the edges — so it's correct — but it's a constraint per grid point and a column per coefficient, which on the machines I have is only practical to ~100 parameters and is slow. There's also the maximal-ripple idea (Herrmann, Hofstetter): write equations forcing the error to `±δ` at a set of extremal frequencies and its derivative to zero there, get `N−1` nonlinear equations, Newton-solve. That makes a truly equiripple filter — but the extremal frequencies, and therefore the band edges, come *out* of the solve. I can't put my edges in. So nobody gives me minimax error *and* exact band-edge control *and* speed. That triple is the target.

Now let me set up the object I'm approximating, carefully, because linear phase imposes structure I'll need. The response is `H(e^{jω}) = Σ_{n=0}^{N-1} h(n) e^{-jωn}`. Linear phase means `H = ±|H| e^{-jαω}` with a constant group delay — the phase is a straight line in ω. For the phase to be exactly linear, the impulse response has to be symmetric, `h(n) = h(N−1−n)`, or antisymmetric, `h(n) = −h(N−1−n)`; either way the delay is `α = (N−1)/2`. Let me just verify the symmetric case gives me a real amplitude. Pair the terms `n` and `N−1−n`: `h(n)[e^{-jωn} + e^{-jω(N-1-n)}] = h(n) e^{-jω(N-1)/2} · [e^{jω((N-1)/2 - n)} + e^{-jω((N-1)/2 - n)}] = 2 h(n) e^{-jα ω} cos(ω((N-1)/2 − n))`. So the whole sum is `e^{-jαω}` times a **real** sum of cosines. Good. So `H(f) = G(f) · e^{j[(Lπ/2) − ((N-1)/2)·2πf]}` with `L=0` for the symmetric case, and `G(f)` real, and the linear-phase exponential has no effect on the magnitude. The thing I have to approximate `D(f)` with is this real `G(f)`.

Crossing symmetry (positive/negative) with parity (`N` odd/even) gives four cases, and `G(f)` is a finite trig sum in each: a cosine sum when `h` is symmetric, a sine sum when antisymmetric. Symmetric, odd `N`: `G(f) = Σ_{k=0}^{n} a(k) cos(2πkf)`, `n=(N−1)/2`. Symmetric, even `N`: `G(f) = Σ_{k=1}^{n} b(k) cos[2π(k−½)f]`. Antisymmetric, odd `N`: `Σ c(k) sin(2πkf)`. Antisymmetric, even `N`: `Σ d(k) sin[2π(k−½)f]`. Four different basis sets. I do *not* want to write four separate optimizers and four separate theories. Let me see if I can collapse them.

The half-integer cosines and the sines all look like a pure cosine sum times a fixed factor. Take the even-`N` symmetric case: `Σ_{k=1}^{n} b(k) cos[2π(k−½)f]`. Pull out `cos(πf)`: there's an identity `cos[2π(k−½)f] = cos(πf(2k−1))`, and a finite sum of those can be written as `cos(πf) · Σ_{k=0}^{n−1} b̃(k) cos(2πkf)` for some recombined coefficients `b̃`. Let me confirm the mechanism with the product-to-sum identity: `cos(πf)·cos(2πkf) = ½[cos(2π(k+½)f) + cos(2π(k−½)f)] = ½[cos(πf(2k+1)) + cos(πf(2k−1))]`. So multiplying a cosine sum by `cos(πf)` produces exactly the half-integer-frequency cosines I need, and I can match coefficients to go backwards — the even-`N` case is `cos(πf)` times a plain cosine polynomial. The same trick handles the two antisymmetric cases: `sin(2πkf) = sin(2πf)·[cosine polynomial]` and `sin[2π(k−½)f] = sin(πf)·[cosine polynomial]`, by the analogous product-to-sum identities (`sin·cos` splits into two sines, half-integer or integer). So in every one of the four cases,

G(f) = Q(f) · P(f),  with  P(f) = Σ_{k=0}^{r−1} α(k) cos(2πkf)  a pure cosine sum,

and `Q(f)` is one of `{ 1, cos πf, sin πf, sin 2πf }` — a fixed, known function of `f` that I can divide out. That's the unification. Now if I want to minimize `max W(f)|D(f) − Q(f)P(f)|`, I just absorb `Q` into the target and the weight:

D̂(f) = D(f) / Q(f),   Ŵ(f) = W(f) · Q(f),

and the problem becomes `min max Ŵ(f)|D̂(f) − P(f)|` over the coefficients of the *single* cosine polynomial `P`. (I have to be a little careful at any band endpoint where `Q(f)=0` — divide-by-zero — so I drop those isolated points from the set `F`: at such a point `G=Q·P` is pinned to zero no matter what `P` does, so it carries no free approximation to optimize, and it simply isn't part of the set I'm fitting `P` on. So I exclude those endpoints from the grid before forming `D̂=D/Q`.) One problem, one core, all four filter types and differentiators and Hilbert transformers fall out of it. That's worth a lot: I write the engine once.

So now the entire question is: best weighted-Chebyshev approximation of a continuous function `D̂` by a cosine polynomial `P` of `r` terms. Cosine polynomials feel special, but there's a substitution that makes them ordinary. Let `x = cos(2πf)`. Then `cos(2πkf) = T_k(cos 2πf) = T_k(x)`, the Chebyshev polynomial of degree `k`. So `P(f) = Σ_{k=0}^{r−1} α(k) T_k(x)` is just an **algebraic polynomial of degree `r−1` in `x`** on the interval `x ∈ [−1, 1]`. The trig problem *is* a polynomial Chebyshev-approximation problem in disguise. And polynomial Chebyshev approximation has a complete, classical theory. Let me reach for it.

What does the best approximation look like? Suppose `P*` is best — it minimizes `max Ŵ|D̂ − P|`, call that minimum value `δ`. The weighted error `E(f) = Ŵ(f)[D̂(f) − P*(f)]` has `|E| ≤ δ` everywhere and equals `δ` in magnitude at the points where the max is attained. Claim: those max-attaining points must be many, and the error must *alternate* in sign across them. Why? Suppose the error touched `±δ` with the right alternating pattern at only a few points — fewer than `r+1`. Then I could find a cosine polynomial `R` of degree `r−1` (i.e. with `r` free coefficients) that has the *same* sign as `E` at each of those extremal points: that's an interpolation condition at, say, `m ≤ r` points, and a degree-`r−1` polynomial has `r` coefficients, so I can always satisfy `m ≤ r` sign conditions. Nudge `P* → P* + εR` for small `ε`, so the new error is `E − εŴR`. At each old extremum, I've pushed the error *toward zero* (because `R` matches the sign of `E` there, so `E − εŴR` shrinks in magnitude), so the error magnitude drops below `δ` at every extremum, and by continuity stays below `δ` in a neighborhood; for small enough `ε` nothing new reaches `δ`. So I strictly reduced the max error — contradicting that `P*` was best.

Let me count exactly how many alternations I'm forced into. The escape `R` exists whenever the number of sign-alternations in `E` at its extrema is `≤ r`. To block every such `R`, I need the error to hit `±δ`, alternating, at **at least `r+1`** points. And conversely, if the error *does* equioscillate at `r+1` points with `|E|=δ`, then no competitor can beat it: if some `P'` had `max Ŵ|D̂−P'| < δ`, then `P* − P' = (D̂−P') − (D̂−P*)` would, at the `r+1` alternation points, take the sign of `−(D̂−P*)` (since `D̂−P'` is uniformly smaller in magnitude than the `±δ/Ŵ` that `D̂−P*` equals there) — so `P*−P'` alternates sign `r+1` times, hence has `r` sign changes, hence `≥ r` roots in `x`; but `P*−P'` is a polynomial of degree `≤ r−1`, so it must be identically zero, i.e. `P' = P*`. So equioscillation is both necessary and sufficient, and it pins down the optimum uniquely.

So I have it: **`P` is the best weighted minimax cosine approximation of degree `r−1` iff the weighted error `Ŵ(f)[D̂(f) − P(f)]` reaches its maximum magnitude with alternating sign at at least `r+1` frequencies.** This is the alternation theorem, and it converts an awful min-over-all-coefficients-of-a-max into a finite, checkable, *algebraic* condition.

Now, how do I use it to actually compute `P`? *If I knew the `r+1` extremal frequencies* `F_1 < F_2 < ... < F_{r+1}`, the alternation condition would be `r+1` equations,

Ŵ(F_i) [ D̂(F_i) − P(F_i) ] = (−1)^i δ,   i = 1, ..., r+1,

with `r+1` unknowns: the `r` coefficients of `P` plus the common level `δ`. That's a square linear system. I could just solve it. But solving an `(r+1)×(r+1)` system every iteration, for filters with `r` in the hundreds, is exactly the kind of cost I'm trying to avoid (it's why the LP approach was slow). Let me see if the structure gives me `δ` and `P` cheaper.

Rewrite each equation as `P(F_i) = D̂(F_i) − (−1)^i δ / Ŵ(F_i)`. The right-hand side is a known number *once I know `δ`*. I have `r+1` nodes but only `r` coefficients, so forcing a degree-`r−1` polynomial through all `r+1` ordinates `y_i := D̂(F_i) − (−1)^i δ/Ŵ(F_i)` is one condition too many; that extra consistency condition is precisely what determines `δ`. So `δ` is exactly the value that makes the `r+1` interpolation conditions collapse to a degree-`r−1` polynomial instead of a degree-`r` one.

How do I extract `δ` without forming `P`? Consider the `(r+1)`-node Lagrange picture in the `x`-variable, `x_i = cos(2πF_i)`. A degree-`r−1` polynomial through `r+1` prescribed values exists iff the order-`r` divided difference of those values is zero. That divided difference is `Σ_i b_i · y_i` where `b_i = 1 / Π_{j≠i}(x_i − x_j)` are the Lagrange/barycentric weights. So setting the divided difference of `y_i = D̂(F_i) − (−1)^i δ/Ŵ(F_i)` to zero:

Σ_i b_i [ D̂(F_i) − (−1)^i δ/Ŵ(F_i) ] = 0
⟹ Σ_i b_i D̂(F_i) = δ Σ_i (−1)^i b_i / Ŵ(F_i)
⟹ δ = [ Σ_i b_i D̂(F_i) ] / [ Σ_i (−1)^i b_i / Ŵ(F_i) ].

A closed form for `δ` — one dot product over another, no matrix solve. If the ratio is negative, that only means I guessed the starting alternation sign backwards; I flip all the signs to a sequence `s_i` (which is `(−1)^i` or its negation) and use positive `δ`. And once I have `δ`, I have the `r+1` ordinate values `y_i = D̂(F_i) − s_i δ/Ŵ(F_i)` that are consistent with a degree-`r−1` `P`, and I can evaluate `P` at *any* frequency by barycentric Lagrange interpolation through all `r+1` nodes; the degree-`r` term has been killed by the zero divided difference. (The barycentric form, `P(x) = [Σ_j b_j y_j/(x − x_j)] / [Σ_j b_j/(x − x_j)]`, is also the numerically stable way to do this for large `r`.) So per iteration I pay `O(r)` Lagrange weights plus an `O(r)` evaluation at each grid point — cheap.

But I *don't* know the extremal frequencies; that's the catch. That's what Remez's exchange idea is for, and now I can build it. Start with a guess of `r+1` reference frequencies — equally spaced across the bands is a fine cheap start. With that reference: compute `δ` by the formula above, and compute `P` by Lagrange interpolation. Now `P` equioscillates at `±δ` on the reference set *by construction*. But it is the best approximation only if those reference points are the *true* extrema — i.e. only if `|E(f)| = Ŵ(f)|D̂(f) − P(f)|` never exceeds `δ` anywhere off the reference. So check: sweep the actual error over a dense frequency grid and find where `|E|` is largest. If `max_grid |E| > δ`, my reference was wrong — there's a place the error overshoots `δ` that I didn't account for.

The exchange: replace the reference set with the new set of local maxima of `|E|` on the grid. I have to keep exactly `r+1` of them, and they have to alternate in sign so the alternation structure is maintained — so at each candidate extremum I keep the one with the largest `|E|` among same-sign neighbors, and I sort them in frequency. The band edges themselves are always candidate extrema even though `|E|` may not be a grid-local maximum there (the desired response jumps across the transition gap), so I force them into the candidate set; if that leaves a surplus I drop the weaker endpoint, which leaves the interior alternation untouched. Then iterate: recompute `δ` and `P` on the new reference, find the new error extrema, exchange again. Each pass, because the new reference points are places where `|E|` exceeded the old `δ`, the new positive `δ` (the level the alternation forces) can only go *up*: it is nondecreasing across iterations, bounded above by the true optimal `δ*`. So it climbs monotonically to `δ*`, and the reference points settle onto the true equioscillation frequencies. I stop when the dense-grid max of `|E|` no longer exceeds the current `δ` (within tolerance) — at that moment the error equioscillates at `r+1` points and is bounded by `δ` everywhere, which by the alternation theorem *is* the certificate of optimality. In practice this converges in something like six to a dozen exchanges. This whole-set exchange — moving all the references at once to the new extrema rather than one point at a time — is what makes it fast.

One detail on "dense grid": the bands are continua, but I can only search finitely many points. I lay down a grid with on the order of `16·r` points across the bands (grid density 16 times the number of cosine terms). The continuous maximum over a band is well-approximated by the maximum over a grid that fine, so the grid extrema track the true extrema closely; if I worried, I could refine near each found extremum, but density 16 is empirically plenty for the extrema to land in the right place.

Once the exchange has converged I have the optimal `δ` and the reference. Now I do need the actual cosine coefficients `α(k)` of `P` to build the filter. Since `P(f) = Σ_{k=0}^{r−1} α(k) cos(2πkf)` is a cosine series, I can get the `α` by sampling `P` at `r` equally spaced frequencies in `[0, ½]` and taking an inverse discrete cosine/Fourier transform — that's just reading off Fourier-cosine coefficients of a known band-limited cosine sum, exact because `P` has only `r` cosine terms. Then I undo the unification: the `α` are the coefficients of `P`, and `G = Q·P`, so I fold `Q` and the `α` back into the impulse response according to which of the four cases I'm in, and finally impose the symmetry `h(n) = h(N−1−n)` (or `= −h(N−1−n)`) to fill out the whole length-`N` filter. For the cases where `Q` is `cos πf` or `sin πf` or `sin 2πf`, "fold back" is a short recurrence that converts the `P`-coefficients into the `G`-coefficients (the product-to-sum identities, run in reverse), but it's all linear and finite.

Let me write the core in the same split I need for a working routine. The heart is a `remez` routine that does the exchange given the dense grid, the desired `D̂` and weight `Ŵ` already prepared on it, and an initial reference; and a `pre_remez` wrapper that sets up the grid, picks the case from symmetry/parity, builds `D̂` and `Ŵ` by dividing/multiplying by `Q`, runs the exchange, and folds the result into `h`.

```python
import numpy as np

def lagrange_weights(x):
    # b_k = 1 / prod_{j != k} (x_k - x_j) : the barycentric weights on the reference nodes.
    r = len(x)
    b = np.ones(r)
    for k in range(r):
        for j in range(r):
            if j != k:
                b[k] /= (x[k] - x[j])
    return b

def eval_P(xq, x, y, b):
    # Barycentric Lagrange evaluation of the cosine polynomial P at cos-mapped points xq,
    # interpolating values y on nodes x with weights b. (matches freq_eval: P = sum c*y / sum c)
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

def pick_alternating_extrema(err, n_ext, edges=()):
    # Keep local maxima of |err|, collapsing adjacent lobes with the same sign to the larger one.
    # Band edges (and the two grid endpoints) are always candidate extrema, since |err| there is
    # not a grid-local max across the transition gap yet a band-edge extremum the optimum lands on.
    forced = set(edges)
    peaks = []
    for i, value in enumerate(err):
        left = abs(err[i - 1]) if i else -np.inf
        right = abs(err[i + 1]) if i + 1 < len(err) else -np.inf
        if i in forced or (abs(value) >= left and abs(value) >= right):
            peaks.append(i)
    peaks.sort()

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
    # signs strictly alternate now; trim any surplus by dropping the smaller-|err| endpoint,
    # which keeps the interior alternation intact (an interior drop would merge two same-sign lobes)
    while len(ext) > n_ext:
        if abs(err[ext[0]]) <= abs(err[ext[-1]]):
            ext = ext[1:]
        else:
            ext = ext[:-1]
    if len(ext) != n_ext:
        raise RuntimeError("could not find enough alternating extrema")
    return np.array(ext, dtype=int)

def remez_exchange(grid, des, wt, n_cos, maxiter=25):
    # grid: dense frequencies in [0, 0.5]; des = D_hat, wt = W_hat already prepared on the grid.
    # n_cos = r = number of cosine terms; the reference set has r+1 points.
    r = n_cos
    # band edges: grid endpoints plus the two indices straddling each transition gap (where the
    # grid frequency jumps), which the exchange must always be allowed to keep as extrema
    df = np.diff(grid)
    gap = np.where(df > 1.5 * np.median(df))[0]
    edges = sorted({0, len(grid) - 1, *gap.tolist(), *(gap + 1).tolist()})
    # initial reference: r+1 equispaced grid indices, matching the original program's cheap start
    step = (len(grid) - 1) / r
    ext = np.array([int(j * step) for j in range(r)] + [len(grid) - 1])
    dev = 0.0
    for _ in range(maxiter):
        F = grid[ext]
        x = np.cos(2 * np.pi * F)                     # x = cos(2 pi f): the Chebyshev substitution
        b = lagrange_weights(x)
        # closed-form deviation: delta = sum b_k D_k / sum (-1)^k b_k / W_k     (the alternation level)
        signs = (-1.0) ** np.arange(r + 1)
        raw_dev = (b @ des[ext]) / ((signs * b) @ (1.0 / wt[ext]))
        if raw_dev < 0:
            signs = -signs
            dev = -raw_dev
        else:
            dev = raw_dev
        # ordinates P must interpolate so W*(D-P) is +-delta alternately on the reference
        y = des[ext] - signs * dev / wt[ext]
        # evaluate the weighted error on the whole dense grid
        xq = np.cos(2 * np.pi * grid)
        P = eval_P(xq, x, y, b)
        err = (P - des) * wt                         # the implementation uses the opposite sign of E
        # exchange: new reference = the r+1 alternating local maxima of |err| (band edges forced)
        new_ext = pick_alternating_extrema(err, r + 1, edges)
        if np.max(np.abs(err)) <= dev * (1 + 1e-6):   # grid max no longer beats delta -> optimal
            break
        ext = new_ext
    # recover the cosine coefficients alpha by inverse DFT of P sampled at r equispaced freqs
    fpts = np.arange(n_cos) / (2 * n_cos - 1)
    F = grid[ext]
    x = np.cos(2 * np.pi * F)
    b = lagrange_weights(x)
    signs = (-1.0) ** np.arange(r + 1)
    raw_dev = (b @ des[ext]) / ((signs * b) @ (1.0 / wt[ext]))
    if raw_dev < 0:
        signs = -signs
        dev = -raw_dev
    else:
        dev = raw_dev
    y = des[ext] - signs * dev / wt[ext]
    Pvals = eval_P(np.cos(2 * np.pi * fpts), x, y, b)
    alpha = idct_cosine_coeffs(Pvals)                  # alpha_k = Fourier-cosine coeffs of P
    return alpha, dev

def design_fir(numtaps, bands, desired, weight=None, type='bandpass', grid_density=16):
    neg = (type != 'bandpass')                         # symmetric vs antisymmetric impulse response
    nodd = numtaps % 2
    n_cos = numtaps // 2 + (nodd and not neg)          # r = number of cosine terms for this case
    grid, des, wt = build_grid(numtaps, bands, desired, weight, grid_density, n_cos)
    # absorb Q into the target/weight so the core only ever sees a pure-cosine problem:
    #   Q = 1 (sym odd) | cos(pi f) (sym even) | sin(pi f) / sin(2 pi f) (antisym):
    # build_grid drops the endpoints where the chosen Q vanishes (f=0.5 for cos(pi f);
    # f=0 for sin(pi f); f=0 and f=0.5 for sin(2 pi f)) so this division is safe.
    if not neg and not nodd:
        ch = np.cos(np.pi * grid); des = des / ch; wt = wt * ch
    elif neg and nodd:
        ch = np.sin(2 * np.pi * grid); des = des / ch; wt = wt * ch
    elif neg:
        ch = np.sin(np.pi * grid); des = des / ch; wt = wt * ch
    alpha, dev = remez_exchange(grid, des, wt, n_cos)
    h = fold_alpha_to_impulse_response(alpha, numtaps, neg, nodd)  # undo Q, impose h(n)=+-h(N-1-n)
    return h
```

The public call can stay as small as the specification itself:

```python
from scipy.signal import remez
# lowpass, 8 kHz cut, 100 Hz transition, 325 taps, 22.05 kHz sample rate; stopband weighted same as passband
taps = remez(325, [0, 8000, 8100, 11025], [1, 0], fs=22050)
```

The chain, end to end: a filter spec is a worst-case statement, so least-squares/window design is the wrong objective — its 9% Gibbs overshoot never shrinks and its band edges aren't mine to set. Minimize instead the maximum weighted error. Linear phase makes the amplitude a real trig sum in one of four cases; pulling out a fixed factor `Q` reduces all four to approximating a single cosine polynomial `P`, and `x = cos(2πf)` turns that into ordinary polynomial Chebyshev approximation. The alternation theorem says the optimum is the unique `P` whose weighted error equioscillates at `r+1` frequencies — a finite, checkable certificate. Given a reference set the alternation equations hand me `δ` in closed form and `P` by barycentric interpolation with no coefficient solve; the Remez exchange then iterates "compute `δ` and `P`, find where the error overshoots `δ`, move the references there," with `δ` climbing monotonically to the optimum, until the dense-grid error no longer beats `δ`. Inverse-DFT the converged `P` for the cosine coefficients, fold `Q` and the symmetry back in, and out comes the optimal equiripple linear-phase filter for the given tap count, with the band edges exactly where I put them.
