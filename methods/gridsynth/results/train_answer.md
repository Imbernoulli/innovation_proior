Fault-tolerant quantum computers can only implement a small discrete gate set cheaply: the single-qubit Clifford gates plus the T gate, which is far more expensive because it requires magic-state distillation. Since almost every quantum algorithm eventually needs a z-rotation Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}) for an angle θ that is not a multiple of π/4, the central compilation problem is to approximate Rz(θ) by a Clifford+T circuit U with ‖Rz(θ) − U‖ ≤ ε while using as few T gates as possible. Solovay–Kitaev solves this generically but pays a steep price: it treats Clifford+T as an arbitrary universal set and produces circuits of length O(log^{3.97}(1/ε)), well above the information-theoretic lower bound. Exhaustive search can find the true optimum, but its cost grows exponentially and it is impractical beyond modest precision. What is missing is an algorithm that is both efficient and tightly adapted to the algebraic structure of Clifford+T.

The method is called gridsynth. It rests on the Kliuchnikov–Maslov–Mosca characterization: a 2×2 unitary is representable exactly as a Clifford+T circuit if and only if all its entries lie in the ring D[ω] = Z[1/√2, i]. Moreover, exact synthesis converts any such unitary into a minimal-T-count circuit. For a unitary of the form U = [[u, −t†], [t, u†]] with u, t ∈ D[ω] and u†u + t†t = 1, the T-count can be taken to be 2k − 2, where k is the least denominator exponent of u. So the entire problem collapses to finding the admissible u ∈ D[ω] with the smallest denominator exponent.

To turn approximation into a discrete search, write z = e^{-iθ/2}. A short calculation shows ‖Rz(θ) − U‖² = 2 − 2 Re(z†u), so the closeness condition becomes ⟨z⃗, u⃗⟩ ≥ 1 − ε²/2. Geometrically this cuts a thin ε-region from the unit disk. Unitarity also requires the √2-conjugate u• to lie in the unit disk. Thus u must satisfy a two-dimensional grid problem: u ∈ A, u• ∈ B, where A is the ε-region and B is the unit disk, and we want solutions in order of increasing denominator exponent k. The key efficiency tool is a special grid operator G, a real-linear map preserving Z[ω] with determinant ±1. Such operators preserve solutions because u solves (A, B) exactly when Gu solves (G(A), G•(B)). By enclosing A and B in ellipses and iteratively reducing their skew, gridsynth finds an operator that makes both regions upright; once upright, the problem separates into independent one-dimensional grid problems over Z[√2] and can be enumerated cheaply.

For each candidate u produced in this order, gridsynth attempts to complete it to a unitary by solving t†t = ξ with ξ = 1 − u†u ∈ D[√2]. Writing ξ•ξ = n/2^ℓ, this relative-norm Diophantine equation is solvable exactly when ξ is doubly positive and every prime p ≡ 7 (mod 8) dividing n occurs to even multiplicity. Given the factorization of n, the solution is constructive; factoring is the only hard subproblem. With a factoring oracle, the first candidate whose equation succeeds yields an absolutely T-count-optimal circuit. Without one, the implementation caps factoring effort and conservatively accepts candidates whose n is prime, which is always solvable when n ≡ 1 (mod 8). Under the natural prime-distribution hypothesis this costs only an expected O(log(1/ε)) extra candidates and an additive O(log log(1/ε)) in T-count over the second-best solution, giving a typical T-count of 3 log₂(1/ε) + O(log log(1/ε)) for unstructured angles.

```python
import mpmath
from rings import omega, adj, adj2, real, denomexp
from gridproblems import ConvexSet, Ellipse, unitdisk, gridpoints2_increasing
from diophantine import diophantine_dyadic, run_bounded, Success
from exact_synthesis import synthesis_u2


def epsilon_region(epsilon, theta):
    zx, zy = mpmath.cos(-theta / 2), mpmath.sin(-theta / 2)
    d = 1 - epsilon**2 / 2
    rot = mpmath.matrix([[zx, -zy], [zy, zx]])
    shape = mpmath.matrix([[4 * (1 / epsilon)**4, 0],
                           [0, (1 / epsilon)**2]])
    ellipse = Ellipse(rot @ shape @ special_inverse(rot), (d * zx, d * zy))

    def contains(point):
        x, y = point.real, point.imag
        return x*x + y*y <= 1 and zx*x + zy*y >= d

    def intersect(p, v):
        disk = solve_quadratic(v.dot(v), 2 * v.dot(p), p.dot(p) - 1)
        if disk is None:
            return empty_interval()
        t0, t1 = disk
        vz = zx * v.x + zy * v.y
        rhs = d - (zx * p.x + zy * p.y)
        if vz == 0:
            return (t0, t1) if rhs <= 0 else empty_interval()
        cut = rhs / vz
        return (max(t0, cut), t1) if vz > 0 else (t0, min(t1, cut))

    return ConvexSet(ellipse, contains, intersect)


def gridsynth_stats(rng, prec_bits, theta, effort):
    digits = mpmath.ceil(15 + 2 * prec_bits * mpmath.log10(2))
    return with_fixed_precision(digits, gridsynth_internal, rng, prec_bits, theta, effort)


def gridsynth_internal(rng, prec_bits, theta, effort):
    epsilon = 2 ** (-prec_bits)
    region = epsilon_region(epsilon, theta)
    candidates = gridpoints2_increasing(region, unitdisk())

    for u in candidates:
        rng_try, rng = split_rng(rng)
        xi = real(1 - adj(u) * u)
        status, t = run_bounded(effort, diophantine_dyadic(rng_try, xi))
        if status == Success:
            return choose_completion(u, t), status
        record_candidate(u, status)


def choose_completion(u, t):
    if denomexp(u + t) < denomexp(u + omega * t):
        return matrix2x2((u, -adj(t)), (t, adj(u)))
    return matrix2x2((u, -adj(omega * t)), (omega * t, adj(u)))


def gridsynth_gates(rng, prec_bits, theta, effort):
    U, _ = gridsynth_stats(rng, prec_bits, theta, effort)
    return synthesis_u2(U)
```

Global phase is unobservable, so gridsynth also runs a twin branch for phase λ = e^{iπ/8} and returns the smaller T-count, covering both even and odd optimal T-counts. The result is a practical, number-theoretic synthesizer that is asymptotically near-optimal and, with a factoring oracle, exactly optimal.
