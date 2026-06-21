I want a linear-phase FIR filter of a fixed length $N$ that meets a magnitude spec — a lowpass with gain $1$ up to a passband edge, gain $0$ past a stopband edge, a transition gap in between I don't care about — and I want it to be as good as a length-$N$ filter can be. The first decision is what "as good as possible" means, because that choice fixes the whole method. The natural starting point is the window method: take the ideal brick-wall lowpass, whose impulse response is the doubly-infinite sinc $h_d(n) = \sin(\omega_c n)/(\pi n)$ with value $\omega_c/\pi$ at $n=0$, keep the central $N$ taps, and taper the ends. Keeping the central taps is exactly the least-squares answer — by Parseval the integral-squared error over a fixed length-$N$ support is the sum of squared coefficient errors, so the best one can do is keep each $h_d(n)$, the orthogonal projection onto the length-$N$ subspace. But look at the band edge: truncation convolves the brick wall with the Dirichlet kernel and produces the Gibbs phenomenon, a roughly $9\%$ overshoot that does **not** shrink as $N$ grows. A longer filter gives narrower ears at the same height; the worst-case ripple is stuck near $9\%$ however many taps I spend. That is the tell that least-squares is the wrong objective: a spec is a worst-case statement ("passband ripple $\le \delta_p$, stopband down so many dB"), and the *best* ISE answer fails it no matter the budget. Tapered windows (Hamming, Blackman, Kaiser) push the sidelobes down but minimize nothing — they trade main-lobe width for sidelobe height by a fixed recipe — and the band edge lands wherever the windowed sum puts it, giving no grip on the cutoff. For a spec that names its edges, that alone is disqualifying. Frequency sampling sets the response equal to the target at $N$ equispaced frequencies and inverse-DFTs; it is exact at the samples but ripples between them, optimizes ripple only through a few free transition samples, and pins edges to the sample grid. Linear programming genuinely minimizes the worst case and admits fixed edges, but with a constraint per grid point and a column per coefficient it is heavy — practical to only about $100$ parameters and slow. The maximal-ripple (Herrmann–Hofstetter) approach makes a truly equiripple filter, but the extremal frequencies, and hence the band edges, come *out* of the solve rather than in. Nobody delivers minimax error **and** exact edge control **and** speed; that triple is the target.

I propose the Parks–McClellan algorithm: minimize the worst-case weighted error directly, by exploiting the equioscillation structure of the optimum through a Remez exchange. The objective is the minimax (Chebyshev, $L_\infty$) one, $\min \max_f W(f)\,|D(f) - G(f)|$, where $D(f)$ is the desired amplitude ($1$ in passbands, $0$ in stopbands), $G(f)$ is what the filter realizes, and $W(f)$ is a weight that lets me say I care, say, $10\times$ more about stopband ripple than passband. To make this tractable I first use the structure linear phase imposes. Exactly linear phase forces $h(n) = h(N-1-n)$ (symmetric) or $h(n) = -h(N-1-n)$ (antisymmetric), with delay $\alpha = (N-1)/2$. Pairing the terms $n$ and $N-1-n$ in $H(e^{j\omega}) = \sum_n h(n) e^{-j\omega n}$ gives $2 h(n)\, e^{-j\alpha\omega}\cos(\omega(\alpha - n))$, so the whole sum is $e^{-j\alpha\omega}$ times a **real** amplitude $G(f)$, and the linear-phase exponential never touches the magnitude. Crossing symmetry with parity ($N$ odd/even) yields four cases, each a finite trig sum: a cosine sum when $h$ is symmetric, a sine sum when antisymmetric. I do not want four optimizers, so I collapse them. Using product-to-sum identities — for instance $\cos(\pi f)\cos(2\pi k f) = \tfrac12[\cos(\pi f(2k+1)) + \cos(\pi f(2k-1))]$ — every case factors as
$$G(f) = Q(f)\,P(f), \qquad P(f) = \sum_{k=0}^{r-1} \alpha_k \cos(2\pi k f),$$
with $P$ a pure cosine polynomial of $r$ terms and $Q \in \{1,\ \cos\pi f,\ \sin\pi f,\ \sin 2\pi f\}$ a fixed, known function. Absorbing $Q$ into the target and weight,
$$\hat D(f) = D(f)/Q(f), \qquad \hat W(f) = W(f)\,Q(f),$$
turns all four into one problem: best weighted-Chebyshev approximation of $\hat D$ by the single cosine polynomial $P$. (At a band endpoint where $Q(f)=0$ this division is singular, but there $G = Q P$ is pinned to zero independent of $P$, so it carries no free approximation; I drop those isolated endpoints from the grid before forming $\hat D$.) The final substitution $x = \cos(2\pi f)$ makes $\cos(2\pi k f) = T_k(x)$, the Chebyshev polynomial, so $P(f) = \sum_k \alpha_k T_k(x)$ is an ordinary algebraic polynomial of degree $r-1$ on $x \in [-1,1]$ — and ordinary polynomial Chebyshev approximation has a complete classical theory.

What makes the method work is the alternation (equioscillation) theorem, which converts the awful min-over-coefficients-of-a-max into a finite, checkable, algebraic condition: $P$ is the unique best weighted minimax approximation iff the weighted error $E(f) = \hat W(f)[\hat D(f) - P(f)]$ attains its maximum magnitude $\delta$ with **alternating sign at at least $r+1$ frequencies** $F_1 < \dots < F_{r+1}$. The necessity argument is constructive: if the error alternated at only $m \le r$ extrema, then because a degree-$(r-1)$ polynomial has $r$ free coefficients I could build a correction $R$ matching the sign of $E$ at all $m$ of them, and the nudge $P \to P + \varepsilon R$ would shrink the error magnitude at every extremum, beating $P$ — a contradiction. Sufficiency is the same idea run backwards: if $P$ equioscillates at $r+1$ points and some $P'$ beat it, then $P - P'$ would change sign $r$ times, forcing $r$ roots into a degree-$(r-1)$ polynomial, so $P' = P$. The theorem says the optimum is exactly the $P$ whose error equioscillates, which is a certificate I can test. The remaining work is computing $P$ given a candidate reference set. If I knew the $r+1$ extremal frequencies, the alternation conditions $\hat W(F_i)[\hat D(F_i) - P(F_i)] = (-1)^i\delta$ are $r+1$ equations in $r$ coefficients plus the level $\delta$ — a square solve, but an $(r+1)\times(r+1)$ solve every iteration is exactly the cost I want to avoid. Instead I exploit structure. Rewriting each equation as an ordinate $y_i = \hat D(F_i) - (-1)^i\delta/\hat W(F_i)$ that $P$ must interpolate, a degree-$(r-1)$ polynomial through $r+1$ values exists only if the order-$r$ divided difference of those values vanishes; that single consistency condition is what pins $\delta$. Writing the divided difference with the barycentric weights $b_i = 1/\prod_{j\ne i}(x_i - x_j)$ on the nodes $x_i = \cos(2\pi F_i)$ and setting it to zero gives $\delta$ in **closed form**:
$$\delta = \frac{\sum_i b_i\,\hat D(F_i)}{\sum_i (-1)^i\, b_i/\hat W(F_i)},$$
a dot product over a dot product, no matrix solve. If $\delta$ comes out negative I simply guessed the starting alternation sign backwards: flip the sign sequence $s_i$ and take $\delta > 0$. With $\delta$ in hand the ordinates $y_i$ are fixed and $P$ can be evaluated anywhere by barycentric Lagrange interpolation through all $r+1$ nodes — the degree-$r$ term has been killed by the zero divided difference, and the barycentric form $P(x) = \big[\sum_j b_j y_j/(x - x_j)\big]/\big[\sum_j b_j/(x - x_j)\big]$ is also the numerically stable way to do this for large $r$. So each iteration costs $O(r)$ for the weights and $O(r)$ per grid evaluation, not a solve.

The reference frequencies are unknown, and supplying them is the job of the Remez exchange. Start with $r+1$ equispaced reference indices — a cheap, adequate start. Each pass: compute $\delta$ and $P$ as above, so $P$ equioscillates at $\pm\delta$ on the current reference by construction; then sweep the actual weighted error over a dense grid of about $16r$ points across the bands and locate where $|E|$ is largest. If the grid max exceeds $\delta$, the reference was wrong, so I exchange — replace it with the $r+1$ alternating local maxima of $|E|$. The bookkeeping that keeps this honest is in `pick_alternating_extrema`: among adjacent same-sign error lobes I keep the one of largest $|E|$, I force the band edges and the grid endpoints to stay candidates (the desired response jumps across the transition gap, so $|E|$ at an edge need not be a grid-local max even though the optimum lands an extremum there), and if that leaves a surplus I drop the weaker *endpoint*, which preserves the interior alternation rather than merging two same-sign lobes. Because the new reference points are precisely where $|E|$ overshot the old $\delta$, the new level can only rise: $\delta$ is nondecreasing across iterations and bounded above by the true optimum $\delta^\*$, so it climbs monotonically to it while the references settle onto the true equioscillation frequencies, typically in six to a dozen exchanges. I stop when the dense-grid max of $|E|$ no longer beats $\delta$ — at that moment the error equioscillates at $r+1$ points and is bounded by $\delta$ everywhere, which by the alternation theorem *is* the optimality certificate. Finally I recover the cosine coefficients $\alpha_k$: since $P$ is a cosine sum of exactly $r$ terms, sampling it at $r$ equispaced frequencies and taking an inverse discrete cosine transform reads off the $\alpha_k$ exactly. Then I undo the unification — fold $Q$ and the $\alpha_k$ back into the impulse response by the short product-to-sum recurrence for the case at hand — and impose $h(n) = \pm h(N-1-n)$ to fill out the length-$N$ filter. The result is the optimal equiripple linear-phase filter for the given tap count, with the band edges exactly where I put them.

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

def pick_alternating_extrema(err, n_ext, edges=()):
    """Return n_ext alternating local maxima of |err|, keeping the largest same-sign lobe.
    Band edges and the two grid endpoints (edges) are always candidate extrema."""
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

def remez_exchange(grid, des, wt, r, maxiter=25):
    """grid: dense freqs in [0,0.5]; des=D_hat, wt=W_hat prepared on the grid; r cosine terms."""
    # band edges = grid endpoints + the indices straddling each transition gap (frequency jump)
    df = np.diff(grid)
    gap = np.where(df > 1.5 * np.median(df))[0]
    edges = sorted({0, len(grid) - 1, *gap.tolist(), *(gap + 1).tolist()})
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
        new_ext = pick_alternating_extrema(err, r + 1, edges)      # r+1 alternating extrema
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
