## Research question

Given an integer `n > 1`, decide whether `n` is prime in time polynomial in the bit length `log n`, with no random error and no unproved hypothesis. The basic trial-division definition of primality gives a correct algorithm, but it tests divisors up to `sqrt(n)`, which is exponential in `log n`. Fast randomized tests already make primality usable in cryptography, and deterministic tests conditional on ERH/GRH already suggest the problem should be easy. What is missing is an unconditional deterministic polynomial-time decision procedure for all inputs.

The target guarantee is deliberately stronger than "probably prime." A compositeness answer should be a proof by failed congruence or nontrivial gcd. A primality answer should follow from a theorem that rules out every composite, including Carmichael numbers, prime powers, and composites designed to pass many Fermat-style tests. The algorithm may use arithmetic in rings and polynomial rings, but every parameter it searches for must itself be bounded by a polynomial in `log n`.

## Background

Fermat's little theorem gives the old foothold: if `p` is prime and `gcd(a, p) = 1`, then `a^(p-1) equiv 1 mod p`. Repeated squaring makes this congruence fast to check. The problem is that the converse fails badly: some composite numbers satisfy the Fermat congruence for many bases, and Carmichael numbers satisfy it for every coprime base.

The binomial theorem gives a stronger, polynomial version of Fermat's test. For `gcd(a, n) = 1` and `n >= 2`,

`n` is prime if and only if `(X + a)^n equiv X^n + a mod n`

as an identity in `(Z/nZ)[X]`. If `n` is prime, the middle binomial coefficients are divisible by `n`. If `n` is composite, take a prime `q` dividing `n` and write `q^k || n`; then the coefficient of `X^q` in `(X+a)^n - (X^n+a)` is not zero modulo `n`. This identity is exact, but checking it directly requires inspecting up to `n` coefficients.

## Technical ingredients

Polynomial rings provide the next set of tools. In `R[X]/(h(X))`, a high-degree polynomial has a bounded-degree representative, so polynomial exponentiation can be done by repeated squaring with reduction after each multiplication. The danger is aliasing: two different full polynomial identities can become equal after quotienting. Any deterministic test that uses a quotient must compensate for information lost in that quotient.

Several standard facts are available. The multiplicative order `ord_r(n)` is the least positive `k` with `n^k equiv 1 mod r`, when `gcd(n, r) = 1`; it divides `phi(r)`. When `p` does not divide `r`, the `r`-th cyclotomic polynomial over `F_p` divides `X^r - 1`, and its irreducible factors have degree `ord_r(p)`. If an irreducible factor `h` is selected, `F_p[X]/(h)` is a finite field, and the image of `X` is a primitive `r`-th root of unity; the degree is greater than one exactly when `ord_r(p) > 1`. Also useful is a crude lcm bound: for `m >= 7`, the lcm of the first `m` positive integers is at least `2^m`.

## Baselines

**Trial division and the sieve viewpoint.** Divide by every `m <= sqrt(n)`. This is correct and deterministic, and a small-factor version is a useful prefilter. Its gap is runtime: `Omega(sqrt(n))` arithmetic trials is exponential in the input size.

**Fermat testing.** Check `a^(n-1) equiv 1 mod n` for selected bases `a`. This is fast by repeated squaring and gives immediate compositeness certificates when it fails. Its gap is structural: Carmichael numbers pass the congruence for every coprime base, so a fixed deterministic base list cannot be justified by Fermat alone.

**Solovay-Strassen and Miller-Rabin.** These strengthen Fermat using the Jacobi symbol or the square-root path to `a^(n-1)`. They give fast randomized algorithms with one-sided error bounds for every composite. Miller's deterministic version can be made polynomial under ERH/GRH by proving that a small exposing base exists. The gap is either randomness or a conjectural input.

**Pratt certificates and later certificate methods.** Pratt showed that primes have short certificates checkable in polynomial time, placing primality in NP as well as co-NP. Elliptic-curve and related certificate-producing methods can be very effective and can give verifiable proofs. Their gap, for this goal, is not the existence of certificates but a single unconditional deterministic polynomial-time decision algorithm with a simple worst-case bound.

**Quasi-polynomial deterministic tests.** Adleman, Pomerance, and Rumely gave a deterministic test running in `(log n)^(O(log log log n))` time, using deeper reciprocity laws. This was a major step away from exponential deterministic testing. Its gap is the remaining quasi-polynomial worst-case bound and the complexity of the machinery.

**Full polynomial identity testing.** The identity `(X+a)^n equiv X^n+a mod n` is a complete characterization of primes for coprime `a`, and randomized polynomial-identity methods can exploit compressed views of this identity. The gap is deterministic completeness: a compressed check can hide a bad coefficient unless the compression and the set of checked shifts come with a proof that no composite can keep passing.

## Evaluation settings

The main metric is worst-case bit complexity as a function of `log n`, with arithmetic operations counted on `O(log n)`-bit coefficients and degree-bounded polynomials. Correctness is one-sided at every intermediate rejection and exact at the final acceptance: every prime must pass, and every composite must be rejected.

Natural test families include primes, prime powers, composites with small prime factors, Carmichael numbers, and composites with no small factors. The correctness proof must explicitly cover prime powers and composites that pass Fermat congruences. Runtime analysis should track the cost of searching for auxiliary parameters, computing gcds, and performing repeated-squaring exponentiation in a quotient polynomial ring.
