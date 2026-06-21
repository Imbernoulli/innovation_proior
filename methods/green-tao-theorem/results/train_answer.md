The problem is to show that the primes contain arbitrarily long arithmetic progressions. Szemerédi's theorem already guarantees long progressions in any set of positive upper density, but the primes have density zero, so that theorem does not apply directly. Trying to squeeze a proof out of existing quantitative bounds is hopeless, because the prime counting function only reaches a density of about 1 / log N, far below anything the dense recurrence machinery can handle. A naive alternative is to treat the primes as a random sparse set of about the same density, but the primes are not random in that sense: they are concentrated in reduced residue classes modulo every small prime, so they carry deterministic local arithmetic bias. Any counting argument that assumes uniformity across residues will fail unless that bias is removed.

The key insight is that the primes do not need to be dense in the integers; they only need to be dense inside a sparse ambient object that is sufficiently pseudorandom. The right tool is therefore a relative Szemerédi theorem. If a nonnegative majorant nu behaves randomly enough and the normalized prime weight f is bounded by nu while still having positive relative mean, then f inherits the Szemerédi conclusion. The task splits into proving such a transference principle and then constructing a majorant for the primes that satisfies the required finite pseudorandomness conditions.

The method is the Green–Tao theorem on arithmetic progressions in the primes. It proceeds in two linked steps. First, it establishes a relative Szemerédi theorem: whenever 0 <= f <= nu on a cyclic group, the average of f over k-term arithmetic progressions is bounded below by a positive constant depending only on k and the relative density of f, provided nu is k-pseudorandom. The proof uses the machinery of Gowers uniformity. The function f is decomposed into a bounded structured component and a Gowers-uniform error. A generalized von Neumann theorem shows that any progression average containing the uniform error is negligible, even when the functions are only majorized by nu rather than bounded by 1. The structured part is genuinely bounded after passing to a suitable sigma-algebra built from dual functions, and dense Szemerédi applies to it. The iteration that builds the sigma-algebra terminates because each step increases the L^2 energy of the conditional expectation of f by a definite increment.

Second, the method constructs a pseudorandom majorant for the primes. The local congruence bias is removed by the W-trick: fix a slowly growing product W of all primes below w(N), and restrict attention to primes of the form Wn + 1. This normalization makes the reduced residue classes modulo small primes equally likely. The majorant itself is built from the Goldston–Yıldırım truncated divisor sum Lambda_R(n) = sum_{d|n, d <= R} mu(d) log(R/d). Squaring this divisor sum gives a nonnegative sieve weight supported on almost-primes. On a safe interval, nu(n) is defined as a normalized constant multiple of Lambda_R(Wn + 1)^2 / log R, and it is set to 1 outside that interval. Goldston–Yıldırım estimates imply that nu has mean 1 + o(1), satisfies the linear forms condition and the correlation condition required for k-pseudorandomness, and majorizes a fixed constant multiple of the modified von Mangoldt prime weight. Applying the relative theorem to that modified prime weight yields a positive count of nondegenerate k-term progressions in the W-tricked primes, and the map n -> Wn + 1 turns them into genuine arithmetic progressions of primes.

```python
import sympy as sp
from sympy import symbols, log, product, primerange, mobius, sqrt


def green_tao_progressions(k=3, N=2000, w=5):
    """
    Demonstrate the two Green–Tao ingredients on a small scale:
    1. W-trick normalization of the primes.
    2. A truncated Goldston–Yildirim divisor-sum majorant.
    """
    # Small-prime product for the W-trick.
    W = product(p for p in primerange(2, w + 1))
    # Primes up to N that are congruent to 1 modulo W.
    tricked = [p for p in primerange(2, N + 1) if p % W == 1]
    # Map each W-tricked prime p to n with p = W*n + 1.
    n_values = [(p - 1) // W for p in tricked]

    # Truncated divisor-sum majorant Lambda_R^2 / log R.
    n_sym = symbols('n', integer=True, positive=True)
    R = int(N ** (1.0 / (k * 2 ** k)))  # tiny truncation radius for illustration

    def divisor_sum_majorant(x):
        x = int(x)
        if x == 0:
            return 0.0
        s = 0.0
        for d in range(1, min(R, x) + 1):
            if x % d == 0:
                s += mobius(d) * log(R / d)
        return max(s, 0.0) ** 2 / log(R)

    # Evaluate the majorant on the W-tricked index set.
    majorant = {n: divisor_sum_majorant(W * n + 1) for n in n_values}
    mean_nu = sum(majorant.values()) / max(len(n_values), 1)

    # Search for k-term arithmetic progressions among the W-tricked primes.
    n_set = set(n_values)
    progressions = []
    n_sorted = sorted(n_values)
    for i, a in enumerate(n_sorted):
        for j in range(i + 1, len(n_sorted)):
            b = n_sorted[j]
            d = b - a
            if all(a + t * d in n_set for t in range(k)):
                progressions.append(tuple(a + t * d for t in range(k)))

    return {
        "W": W,
        "tricked_primes": tricked[:10],
        "majorant_mean": float(mean_nu),
        "progressions": progressions[:5],
    }


if __name__ == "__main__":
    result = green_tao_progressions(k=3, N=5000, w=5)
    print(result)
```
