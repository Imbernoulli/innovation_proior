OK, let me set this up from the pain. I want length-`n` sequences of `±1` whose off-peak aperiodic autocorrelations `C(u) = sum_{i} a_i a_{i+u}` are collectively tiny — concretely, I want the merit factor `F = n^2 / (2 sum_{u≥1} C(u)^2)` to be large, and not just at one cute length but for a whole infinite family as `n → ∞`. The peak `C(0) = n` is fixed; all the action is in pushing the energy `sum_{u≥1} C(u)^2` down toward its floor.

How good is "nothing"? Take a random `±1` sequence. The expected `1/F` over all `2^n` of them works out to `(n-1)/n`, which goes to `1`. So a generic sequence sits at `F ≈ 1`, and any family I'd be proud of has to beat that decisively. The trouble is that just hill-climbing in the `2^n` cube — flip a bit, recompute `F`, accept if better — is exactly what people have hammered for decades (it's the same "low-autocorrelation binary string" ground-state problem the spin-glass people grind on), and for large `n` it stubbornly refuses to climb past about `F = 6`, drowning in local optima. So search isn't going to *tell me the answer*; if I want to know the asymptotic value I need a sequence I can *write down* and an `F` I can *compute in closed form*. Structure, then proof.

So where could structure come from? Stare at the energy I'm minimizing. The aperiodic `C(u)` is a sum over a *window* `i = 0 … n-1-u` — it's a truncated correlation, and truncated sums are analytically nasty; there's no group acting on them. But there's a cousin that *is* tame: the periodic autocorrelation `R(u) = sum_{i=0}^{n-1} a_i a_{(i+u) mod n}`, a genuine cyclic-group object. And these two are related by

  `R(u) = C(u) + C(n-u)`   for `0 < u < n`.

That's worth pausing on. The periodic correlation of a sequence is the *pair sum* of two aperiodic correlations, the one at shift `u` and the one at the complementary shift `n-u`. So if I had a sequence whose *periodic* autocorrelation were extremely flat — say constant and small at every nonzero shift — then for every `u`,

  `C(u) + C(n-u) = R(u) = small constant.`

Now, does flat periodic force small aperiodic? No — and I should be honest that it doesn't. The pair *sums* being pinned at a small constant is a real constraint, but it says nothing about the individual terms: I could have `C(u)` large and positive while `C(n-u)` is large and negative, and the sum stays small. So flat periodic does **not** imply small aperiodic energy. But it's a genuine lever. It pins down `n-1` linear combinations of the aperiodic correlations, and it gives me hope that, for the *right presentation* of such a sequence, the individual `C(u)` might also turn out collectively small. That hope is the whole game; I'll have to actually compute to see if it pays off.

Which sequences have flat periodic autocorrelation? Constant `R(u)` at all nonzero shifts is *exactly* the definition of a cyclic difference set: a `k`-subset `D` of `Z_n` where every nonzero element occurs the same number `λ` of times as a difference, with the dictionary `a_i = -1 ⟺ i ∈ D` giving `R(u) = n - 4(k - λ)` constant. So I want a difference set I can also analyze. And there's one sitting right there in number theory: for a prime `p`, the quadratic residues. Define

  `x_i = (i | p)`   (the Legendre symbol: `+1` if `i` is a nonzero square mod `p`, `-1` if a non-square),

with `x_0` set to `+1` since `(0 | p) = 0` doesn't give me a sign. For `p ≡ 3 (mod 4)` the residues form the Paley difference set with `(v,k,λ) = (p, (p-1)/2, (p-3)/4)`, so `k - λ = (p-1)/2 - (p-3)/4 = (p+1)/4` and the difference-set sign convention gives `R(u) = p - 4(k-λ) = p - (p+1) = -1` for every `u ≠ 0`. My Legendre sign convention reverses the nonzero signs, but because `(−1 | p) = -1`, the two exceptional terms in `sum_i x_i x_{i+u}` cancel and the same constant `R(u) = -1` survives. So this Legendre sequence has the flattest possible nonzero periodic autocorrelation: a constant `-1`. And the Legendre symbol is the quadratic *character*, which means I have Gauss sums and the Weil bound to actually estimate things.

Let me just compute the merit factor of the bare Legendre sequence and see. The value settles around `3/2`. That's it — `1.5`. Barely better than random. Disappointing, but instructive: the periodic flatness alone bought me almost nothing, exactly as I warned myself it might. The pair sums `C(u) + C(n-u) = -1` are pinned, but the individual `C(u)` are *not* small — they're large with cancelling signs. I need another knob.

The periodic flatness is invariant under **rotation**. Cyclically shifting the sequence by any amount doesn't change `R(u)` (it's a cyclic-group quantity), so every rotation `X_r` of the Legendre sequence *also* has `R(u) = -1` for all `u ≠ 0`, and *also* satisfies `C_{X_r}(u) + C_{X_r}(n-u) = -1`. But rotation absolutely *does* change the aperiodic correlations, because `C(u)` is a windowed sum — sliding which residues sit at the window's left and right ends changes the truncated character sum. So the same periodic certificate gives a one-parameter family `X_r`, `r ∈ [0,1)` (shift by `⌊rn⌋`), all sharing the flat-periodic property, but with *different* aperiodic energies. The constraint pins the pair sums; the rotation redistributes how the energy splits within each pinned pair, and some rotations may split it far more favorably than others.

Now I have to actually estimate `sum_u C_{X_r}(u)^2` as a function of `r`. Write the windowed correlation out:

  `C_{X_r}(u) = sum_{i in window} x_{i+⌊rn⌋} x_{i+u+⌊rn⌋} = sum_i (i+s | p)(i+u+s | p)`,  `s = ⌊rn⌋`,

and by multiplicativity `(i+s | p)(i+u+s | p) = ((i+s)(i+u+s) | p)`, a quadratic character of a quadratic polynomial in `i`, summed over an *interval* of `i`. Over the *full* period this would be a clean character sum I could nail with a Gauss sum, but I'm summing over a sub-interval — the window — and that's where the rotation `r` enters: `r` sets where the interval lives. The size of `C_{X_r}(u)` is governed by how the interval `[s, s+(n-u)]` of evaluation points interacts with the quadratic character structure, so the total energy has to be expressed through the overlap lengths of these rotated intervals. This is exactly the calculation that the bare-sequence disappointment told me I had to do — periodic flatness was never going to be enough; I had to descend into the windowed sums themselves.

As `r` slides the window, the favorable cancellations come and go smoothly, and `sum_u C(u)^2` (hence `1/F`) should be a smooth function of `r` over each half-period. If the only surviving terms after character-sum cancellation are the overlap lengths of intervals, those lengths are linear in `r`, and their squares will make `1/F` quadratic. The window of evaluation looks most balanced when it is centered a quarter-turn away — `r = 1/4` — so that is the offset I expect the calculation to select.

I need the character-sum calculation to tell me which overlap terms actually survive. I can organize the whole `sum_u C(u)^2` in the Fourier domain. Evaluate the polynomial `X_p(z) = 1 + sum_{j=1}^{p-1} (j|p) z^j` at the `p`-th roots of unity `ζ_k = e^{2πi k/p}`. Then `X_p(ζ_k) - 1 = sum_{j=1}^{p-1} (j|p) ζ_k^{jk}` is a **quadratic Gauss sum**, equal to `(k|p) i^{(p-1)^2/4} p^{1/2}` for `k ≠ 0`. So the nonconstant character part has exactly flat magnitude `p^{1/2}` at every nonzero frequency, and the extra `1` from the `x_0 := 1` convention is small enough to be carried as an error term. The merit factor of any rotation/truncation is built out of fourth moments of these frequency values, which I package as

  `L_A(a,b,c) = (1/n^3) sum_{k} A(ζ_k) A(ζ_{k+a}) \overline{A(ζ_{k+b})} \overline{A(ζ_{k+c})}`.

The point of `L_A` is that `1 + 1/F(A_r)` is an *exact* linear functional of `L_A` — when I expand `sum_u C(u)^2` and pass to frequencies, every term is one of these four-fold products. So if I can show `L_{X_p}(a,b,c)` is close to the "ideal" pattern

  `I(a,b,c) = 1` when one of `a,b,c` is `0` and the other two are equal, else `0`,

then the merit factor must converge to whatever the ideal pattern dictates. For the Legendre sequence, substitute the Gauss-sum values at the nonzero frequencies, and the multiplicativity of the symbol collapses the four-fold frequency product into a *single* character sum over `F_p`,

  `L_{X_p}(a,b,c) = (1/p) sum_{x in F_p} ( x(x+a)(x+b)(x+c) | p ) + Δ`,   with `|Δ| ≤ 15 p^{-1/2}`,

and now the Weil bound does the work. The inner sum is the quadratic character of a degree-`4` polynomial; unless that quartic is a perfect square in `F_p[x]`, the sum is at most `3 p^{1/2}` in magnitude, so divided by `p` it's `O(p^{-1/2}) → 0`. The quartic `x(x+a)(x+b)(x+c)` is a square exactly when its roots pair up: either two distinct double roots (sum `= p-2`) or one quadruple root (sum `= p-1`) — and those are *precisely* the configurations where one of `a,b,c` is `0` and the other two coincide, i.e. exactly where `I(a,b,c) = 1`. Everything else is killed by Weil. So

  `max_{a,b,c} | L_{X_p}(a,b,c) - I(a,b,c) | ≤ 18 p^{-1/2} → 0`,

and the convergence is fast enough (it beats the `(log p)^3` blow-up that the windowing introduces). Feed `L → I` into the exact `L`-to-`F` functional, and on the half-period `0 ≤ r ≤ 1/2` the surviving interval-overlap terms give

  `1 / lim F(X_r) = 1/6 + 8 (r - 1/4)^2`.

By the half-period symmetry, on `1/2 ≤ r ≤ 1` the same formula is

  `1 / lim F(X_r) = 1/6 + 8 (r - 3/4)^2`.

At `r = 1/4` the bracket vanishes and `1/F → 1/6`, i.e. **`F → 6`**. At the other extreme `r = 0` it gives `1/F = 1/6 + 8/16 = 2/3`, so `F → 3/2` — precisely the lousy `1.5` I measured on the bare sequence. The two cross-check, and the bare-sequence number wasn't a bug, it was the parabola evaluated at the worst offset. The rotation to `r = 1/4` quadruples the merit factor, from `3/2` to `6`. The quarter-turn minimum is now not a guess but a consequence of where the Weil bound leaves a nonzero residue: at the ideal pattern `I`. The whole thing rests on two pillars, the difference-set property (flat Gauss-sum spectrum) and the Weil bound (everything off the ideal pattern is negligible).

Good. So I have a *proven* family at `F → 6`. Before I ask whether I can do better, let me sanity-check the boundaries of the construction, because two adjacent constructions also have flat-ish spectra and I want to know why they *don't* reach `6`.

The Rudin–Shapiro pair, built by the append recursion `X^{(m)} = X^{(m-1)} ; Y^{(m-1)}`, `Y^{(m)} = X^{(m-1)} ; -Y^{(m-1)}` from `X^{(0)} = Y^{(0)} = [1]`: here the aperiodic correlations satisfy their *own* recurrence directly, no difference set needed, and unrolling it gives `F = 3 / (1 - (-1/2)^m) → 3` exactly. Clean, but stuck at `3` — and tweaking the recursion (different seed polynomials, the general `P_{X^{(m)}}(z) = P_{X^{(m-1)}}(z) ± z^{2^{m-1}} P^*_{X^{(m-1)}}(-z)` family) provably never exceeds `3`. So the recursive route caps at `3`. The m-sequences `x_i = (-1)^{Tr(β α^i)}` are a Singer difference set, so by the *same* periodic-flatness reasoning I'd hope a good rotation beats `3` — the mean of `1/F` over the `n` rotations is `(n-1)(n+4)/(3n^2) → 1/3`, which dangles the possibility. But running the rotation analysis (same machinery: their `L_A` approaches a *different* ideal pattern `J`, with `J(a,b,c) = 1` when `(c=a, b=0)` or `(a=b, c=0)`) pins *every* rotation at `F → 3`. The asymmetry is the lesson: it's the *quadratic* character (multiplicative, with the four-fold product collapsing to a quartic killed by Weil down to the `I`-pattern) that lands on `6`, while the *additive*-character m-sequences land their ideal pattern at a value that only gives `3`. The Legendre `6` is special to the multiplicative/quadratic-residue structure.

And the construction generalizes the way the proof says it should: the analysis only used multiplicativity of the symbol and the Gauss-sum/Weil estimates, so it carries to **Jacobi sequences** `x_i = prod_ℓ (i | p_ℓ)` — products of Legendre sequences over several primes — and to the **modified Jacobi / twin-prime** sequences, all of which hit the *same* parabola `g` and the same `F → 6` at the quarter rotation, provided no prime factor grows too slowly relative to `n`. The mechanism is robust; `6` is the natural ceiling of the rotated-difference-set idea.

Now — is `6` really the end? Let me look again at the appending direction, because there's a tell in it. Suppose I take the optimal `X_{1/4}` (`F → 6`) and start *appending its own initial elements* to the end, extending the length past `n`. Up to about `√n` appended elements, the merit factor is unchanged to leading order — a handful of extra terms can't move `sum_u C(u)^2 ~ n^2` by a relative amount. If I append a constant fraction `α n` of the sequence's own front, the total length fraction is `T = 1 + α`, and every aperiodic window that crosses the old endpoint picks up structured terms rather than fresh random signs. Those new terms enter the squared-correlation expansion through cross-products, so they can lower the normalized off-peak energy even though the final energy is still a sum of squares.

But there has to be a price, or appending would drive `F → ∞`, which can't be right. The obstruction is a single shift. At the exact shift `u = n`, the appended `α n` block lines up onto its original copy, so every product in that part of the sum is `(±1)^2 = +1`; that one shift contributes about `(α n)^2 = ((T - 1)n)^2` to `sum_u C(u)^2`. So appending is a tug-of-war: more appending improves many spread-out overlaps but inflates this one aligned overlap quadratically. The same windowed-character-sum bookkeeping now has both a rotation parameter `R` and a total length fraction `T`, and it gives a two-variable limit

  `1/g(R,T) = 1 - 4T/3 + 4 sum_{m∈N} max(0, 1 - m/T)^2 + sum_{m∈Z} max(0, 1 - |1 + (2R-m)/T|)^2`,

where `N` is the positive integers. At `T = 1`, the positive-`m` sum vanishes, and for `0 ≤ R ≤ 1/2` only the `m = 1` and `m = 2` terms in the integer sum survive, giving `(1 - 2R)^2 + (2R)^2`. Therefore

  `1/g(R,1) = 1 - 4/3 + (1 - 2R)^2 + (2R)^2 = 1/6 + 8(R - 1/4)^2`,

so the two-variable expression collapses to the old parabola when there is no appending. Optimizing `g(R,T)` jointly over `R` and `T` moves the optimum off `T = 1`: the maximum is

  `F_a = 6.342061…`,  the largest root of `29 x^3 - 249 x^2 + 417 x - 27`,

attained at `T = 1.057827…` (the middle root of `4 x^3 - 30 x + 27`) and `R = 3/4 - T/2` when I choose the representative `0 ≤ R < 1/2`. So the difference-set idea, pushed through periodic extension, genuinely clears `6` — the quarter-rotation `6` was a local landmark, not the ceiling. The related periodic and negaperiodic product constructions, using the short masks `(+,+,-,+)` and `(+,+,-,-)`, are governed by the same `g` after a fixed shift in `R`; when I force skew-symmetry through Jacobi sequences with all prime factors `1 mod 4`, the same point is naturally written as `R = 1/4 - T/2`, which is congruent modulo the half-period of `g`. If I repeat the append/periodic-mask analysis for the additive Galois family, the rotation dependence disappears and the limiting function is `h(T)` with `1/h(T) = 1 - 2T/3 + 4 sum_{m∈N} max(0, 1 - m/T)^2`; its maximum is `F_b = 3.342065…`, the largest root of `7 x^3 - 33 x^2 + 33 x - 3`. The multiplicative Legendre side is the one that reaches `F_a`.

Let me write the construction and the analysis down as code, building straight from the definitions and checking the formulas I derived. The construction fills the algebraic sequence slot; the closed-form function fills the `g(R,T)` slot.

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
    # Paley/Legendre periodic autocorrelation is exactly -1 in this prime case.
    return n % 4 == 3 and is_prime(n)

def algebraic_sign(i, n):
    # Legendre sequence: x_0 := +1 and x_i := (i | n) for 0 < i < n.
    j = i % n
    if j == 0:
        return 1
    return 1 if pow(j, (n - 1) // 2, n) == 1 else -1

def build_sequence(n):
    if not valid_length(n):
        raise ValueError("n must be a prime with n % 4 == 3")
    return np.array([algebraic_sign(i, n) for i in range(n)], dtype=np.int64)

def rotate(A, r):
    # Rotation preserves periodic autocorrelation but changes aperiodic windows.
    n = len(A)
    return np.roll(A, -(int(np.floor(r * n)) % n))

def extend_or_truncate(A, t=1.0):
    # Use the first floor(t*n) terms of the periodic extension.
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
    # Paley certificate: constant -1 for all nonzero u when n == 3 mod 4.
    return int(np.dot(A, np.roll(A, -u)))

def asymptotic_merit_factor(r, t=1.0):
    # g(R,T): rotated, then truncated or periodically extended to length floor(T*n).
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
    for r in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        A = extend_or_truncate(rotate(X, r), 1.0)
        print(r, round(merit_factor(A), 3), round(asymptotic_merit_factor(r), 3))
    T = 1.057827
    R = 0.75 - T / 2.0
    print(round(asymptotic_merit_factor(R, T), 6))  # 6.342061...
```

The causal chain, start to finish: search can't decide the asymptotic question, so I need an explicit family with a computable merit factor → the only analyzable handle is the *periodic* autocorrelation, which is flat (`= -1`) exactly for difference-set sequences, and the quadratic residues (Legendre symbol) give such a set *and* hand me Gauss sums → but flat periodic only pins the *pair sums* `C(u) + C(n-u) = -1`, so the bare sequence is stuck at `F = 3/2` → the periodic property is rotation-invariant while the aperiodic energy is not, so I rotate and optimize, and the windowed quadratic-character sums (Gauss sum for flat spectrum, Weil bound to kill everything off the ideal pattern `I`) yield `1/F = 1/6 + 8(r - 1/4)^2`, minimized at the quarter turn for `F → 6` → the same machinery shows the recursive (Rudin–Shapiro) and additive-character (m-sequence) families cap at `3`, marking `6` as special to the quadratic-residue structure, and it extends to Jacobi/twin-prime → appending introduces a total length fraction `T`, and balancing the spread-out gains against the aligned `u = n` obstruction gives `g(R,T)`, whose optimum is `F_a = 6.342061…` at `T = 1.057827…`, `R = 3/4 - T/2` modulo the half-period symmetry of `g`, while the additive Galois analogue tops out at `F_b = 3.342065…`.
