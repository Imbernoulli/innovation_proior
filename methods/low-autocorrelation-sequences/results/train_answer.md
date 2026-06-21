We want length-$n$ binary sequences $A = (a_0, \dots, a_{n-1})$ with $a_i \in \{+1,-1\}$ whose off-peak aperiodic autocorrelations $C_A(u) = \sum_{i=0}^{n-1-u} a_i a_{i+u}$ are collectively tiny, so that the merit factor

$$F(A) = \frac{n^2}{2 \sum_{u=1}^{n-1} C_A(u)^2}$$

is large — and not at one cute length but for an entire infinite family as $n \to \infty$. The peak $C_A(0) = n$ is fixed; everything happens in pushing the energy $\sum_{u \ge 1} C_A(u)^2$ down toward its floor. A random $\pm 1$ sequence has expected $1/F$ equal to $(n-1)/n \to 1$, so a generic sequence sits at $F \approx 1$ and any family worth presenting must beat that decisively. The honest obstacle is that the obvious route — hill-climb in the $2^n$ cube, flip a bit, recompute $F$, keep the improvement — is exactly what has been hammered for decades (it is the spin-glass "low-autocorrelation binary string" ground-state problem), and for large $n$ it stalls below $F \approx 6$, drowning in local optima. Search will not even tell us the asymptotic value. The known explicit families fare little better: the Barker sequences reach the ideal $|C(u)| \le 1$ and so $F = n$, but they provably exist only for $n \le 13$; the Rudin–Shapiro sequences are explicit but provably cap at $F \to 3$, as do every rotation of the maximal-length shift-register ($m$-) sequences. So what is needed is a sequence one can *write down* together with an $F$ one can *compute in closed form*: structure, then proof.

I propose the rotated Legendre (quadratic-residue / Paley) difference-set sequence, and its periodically extended refinement. The lever is that the aperiodic $C_A(u)$ is a windowed, interval-truncated sum with no group acting on it, but its cousin the periodic autocorrelation $R_A(u) = \sum_{i=0}^{n-1} a_i a_{(i+u) \bmod n}$ is a genuine cyclic-group object, and the two are tied by $R_A(u) = C_A(u) + C_A(n-u)$ for $0 < u < n$. A sequence with constant nonzero periodic autocorrelation is precisely a cyclic difference set, and the quadratic residues modulo a prime $p \equiv 3 \pmod 4$ form the Paley difference set with parameters $(p, (p-1)/2, (p-3)/4)$. Taking the Legendre symbol $x_i = (i \mid p)$ (with $x_0 := +1$, since $(0 \mid p) = 0$ gives no sign) yields the flattest possible nonzero periodic autocorrelation, the constant $R(u) = -1$ for every $u \neq 0$ — and the Legendre symbol is the quadratic *character*, which hands us Gauss sums and the Weil bound to estimate things. Crucially, flat periodic autocorrelation does **not** by itself force small aperiodic energy: $R(u) = -1$ only pins the *pair sums* $C(u) + C(n-u) = -1$, leaving the individual terms free to be large with cancelling signs. Indeed the bare Legendre sequence settles at $F \to 3/2$, barely above random. The resolution is that flatness is *rotation-invariant* — cyclically shifting by $\lfloor rn \rfloor$ leaves every $R(u)$ unchanged — while the windowed aperiodic energy is *not*, because rotation slides which residues sit at the window's two ends. So one flat-periodic certificate furnishes a one-parameter family $X_r$, all sharing $C_{X_r}(u) + C_{X_r}(n-u) = -1$ but splitting that pinned pair very differently; the rotation $r$ is the free knob, and we optimize it.

To compute $\sum_u C_{X_r}(u)^2$ as a function of $r$, write $C_{X_r}(u) = \sum_i ((i+s)(i+u+s) \mid p)$ with $s = \lfloor rn \rfloor$, using multiplicativity $(a \mid p)(b \mid p) = (ab \mid p)$: this is the quadratic character of a quadratic in $i$, summed over the *interval* the rotation selects. Passing to the Fourier domain, evaluate $X_p(z) = 1 + \sum_{j=1}^{p-1} (j \mid p) z^j$ at the $p$-th roots of unity $\zeta_k$; then $X_p(\zeta_k) - 1$ is a quadratic Gauss sum of modulus *exactly* $p^{1/2}$ for every $k \neq 0$, so the character spectrum is perfectly flat and the stray $1$ from $x_0 := +1$ is carried as an error term. Packaging the merit factor through the fourth-moment functional

$$L_A(a,b,c) = \frac{1}{n^3} \sum_k A(\zeta_k)\, A(\zeta_{k+a})\, \overline{A(\zeta_{k+b})}\, \overline{A(\zeta_{k+c})},$$

the quantity $1 + 1/F$ becomes an *exact* linear functional of $L_A$, and substituting the Gauss-sum values collapses the four-fold frequency product into a single character sum,

$$L_{X_p}(a,b,c) = \frac{1}{p} \sum_{x \in \mathbb{F}_p} \big( x(x+a)(x+b)(x+c) \mid p \big) + \Delta, \qquad |\Delta| \le 15\, p^{-1/2}.$$

Now the Weil bound does the work: the quadratic character of a degree-$4$ polynomial has sum at most $3 p^{1/2}$ in magnitude *unless* the quartic is a perfect square in $\mathbb{F}_p[x]$, which forces $L \to 0$ off those configurations. The quartic $x(x+a)(x+b)(x+c)$ is a perfect square exactly when its roots pair up — two distinct double roots or one quadruple root — and these are *precisely* the configurations where one of $a,b,c$ is $0$ and the other two coincide. That is the "ideal" pattern $I(a,b,c) = 1$ on exactly those triples and $0$ elsewhere, so $\max_{a,b,c} |L_{X_p}(a,b,c) - I(a,b,c)| \le 18\, p^{-1/2} \to 0$, fast enough to beat the $(\log p)^3$ blow-up the windowing introduces. Feeding $L \to I$ into the exact $L$-to-$F$ functional, the surviving interval-overlap terms (linear in $r$, hence quadratic when squared) give on the half-period

$$\frac{1}{\lim_{p \to \infty} F(X_r)} = \frac{1}{6} + 8\Big(r - \tfrac{1}{4}\Big)^2 \quad (0 \le r \le \tfrac12), \qquad = \frac{1}{6} + 8\Big(r - \tfrac{3}{4}\Big)^2 \quad (\tfrac12 \le r \le 1).$$

At the quarter rotation $r = 1/4$ the bracket vanishes and $F(X_{1/4}) \to 6$; at $r = 0$ it gives $1/F = 2/3$, i.e. $F \to 3/2$, which both explains and cross-checks the lousy bare-sequence number as the parabola at its worst offset. The quarter-turn is not a guess but the location where the Weil bound leaves its nonzero residue. This rests on two pillars: the difference-set property (flat Gauss-sum spectrum) and the Weil bound (everything off $I$ is negligible). The same machinery diagnoses why the neighbors stop at $3$: Rudin–Shapiro has its own aperiodic recurrence giving $F = 3/(1 - (-1/2)^m) \to 3$, and the $m$-sequences are an *additive*-character Singer difference set whose $L_A$ approaches a *different* ideal pattern $J$ that pins every rotation at $F \to 3$. It is the multiplicative/quadratic structure that lands on $6$. The construction also generalizes exactly as the proof permits — to Jacobi sequences $x_i = \prod_\ell (i \mid p_\ell)$ and to modified-Jacobi/twin-prime sequences — all hitting the same parabola and the same $F \to 6$ at the quarter rotation, provided no prime factor grows too slowly relative to $n$.

Is $6$ the ceiling? No. Take the optimal $X_{1/4}$ and append a fraction of its own front, extending to a total length fraction $T = 1 + \alpha$. Up to about $\sqrt{n}$ appended terms nothing moves at leading order, but appending a constant fraction lets every window crossing the old endpoint pick up structured rather than fresh-random terms, lowering the spread-out off-peak energy. The price is a single shift: at $u = n$ the appended $\alpha n$ block lands exactly on its original copy, every product becomes $+1$, and that one shift contributes about $((T-1)n)^2$ to the energy. Balancing the diffuse gains against this aligned quadratic obstruction, the windowed-character bookkeeping with both a rotation $R$ and a length fraction $T$ gives the two-variable limit

$$\frac{1}{g(R,T)} = 1 - \frac{4T}{3} + 4\sum_{m \in \mathbb{N}} \max\!\big(0,\, 1 - \tfrac{m}{T}\big)^2 + \sum_{m \in \mathbb{Z}} \max\!\Big(0,\, 1 - \big|1 + \tfrac{2R-m}{T}\big|\Big)^2,$$

which at $T = 1$ reduces (only $m = 1, 2$ survive the integer sum on $0 \le R \le 1/2$) to $1/6 + 8(R - 1/4)^2$, recovering the parabola. Optimizing jointly moves the maximum off $T = 1$ to

$$F_a = 6.342061\ldots, \quad \text{the largest root of } 29x^3 - 249x^2 + 417x - 27,$$

attained at $T = 1.057827\ldots$ (the middle root of $4x^3 - 30x + 27$) and $R = 3/4 - T/2$. The skew-symmetric Jacobi/product versions, with masks $(+,+,-,+)$ and $(+,+,-,-)$, are governed by the same $g$ after a fixed shift in $R$ (there written $R = 1/4 - T/2$, congruent modulo the half-period). The additive Galois analogue, with the rotation dependence gone and limiting function $1/h(T) = 1 - 2T/3 + 4\sum_{m \in \mathbb{N}} \max(0, 1 - m/T)^2$, tops out only at $F_b = 3.342065\ldots$, the largest root of $7x^3 - 33x^2 + 33x - 3$. So the multiplicative Legendre side is the one that genuinely clears $6$; the quarter-rotation $6$ was a local landmark, not the ceiling.

```python
import math
import numpy as np

def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

def valid_length(n):
    return n % 4 == 3 and is_prime(n)

def algebraic_sign(i, n):
    j = i % n
    if j == 0:
        return 1
    return 1 if pow(j, (n - 1) // 2, n) == 1 else -1

def build_sequence(n):
    if not valid_length(n):
        raise ValueError("n must be a prime with n % 4 == 3")
    return np.array([algebraic_sign(i, n) for i in range(n)], dtype=np.int64)

def rotate(A, r):
    n = len(A)
    return np.roll(A, -(int(np.floor(r * n)) % n))

def extend_or_truncate(A, t=1.0):
    n = len(A)
    length = int(np.floor(t * n))
    if length <= 0:
        raise ValueError("target length must be positive")
    return A[np.arange(length) % n]

def aperiodic_autocorr_sumsq(A):
    n = len(A)
    return sum(int(np.dot(A[:n-u], A[u:]))**2 for u in range(1, n))

def merit_factor(A):
    n = len(A)
    return n * n / (2.0 * aperiodic_autocorr_sumsq(A))

def periodic_autocorr(A, u):
    return int(np.dot(A, np.roll(A, -u)))

def asymptotic_merit_factor(r, t=1.0):
    R = float(r)
    T = float(t)
    if T <= 0:
        raise ValueError("t must be positive")

    positive_m = sum(
        max(0.0, 1.0 - m / T) ** 2
        for m in range(1, int(math.floor(T)) + 1)
    )
    lo = math.floor(2.0 * R - 2.0 * T) - 2
    hi = math.ceil(2.0 * R + 2.0 * T) + 2
    integer_m = sum(
        max(0.0, 1.0 - abs(1.0 + (2.0 * R - m) / T)) ** 2
        for m in range(lo, hi + 1)
    )
    inverse_g = 1.0 - 4.0 * T / 3.0 + 4.0 * positive_m + integer_m
    return 1.0 / inverse_g

if __name__ == "__main__":
    p = 10007
    X = build_sequence(p)
    assert {periodic_autocorr(X, u) for u in range(1, 40)} == {-1}
    for r in [0.0, 0.25, 0.5]:
        A = extend_or_truncate(rotate(X, r), 1.0)
        print(r, round(merit_factor(A), 3), round(asymptotic_merit_factor(r), 3))
    T = 1.057827
    R = 0.75 - T / 2.0
    print(round(asymptotic_merit_factor(R, T), 6))
```
