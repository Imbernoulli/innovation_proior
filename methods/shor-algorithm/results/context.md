# Context

## Research question

Can a computer that obeys quantum mechanics factor large integers, and compute discrete logarithms, in time polynomial in the number of digits — when every known classical method needs superpolynomial time?

The stakes are sharp. The fastest classical factoring algorithm is the number field sieve, which factors an integer `N` in roughly `exp(c (log N)^{1/3} (log log N)^{2/3})` steps. Since the input is only `log N` bits long, this is superpolynomial in the input size, though still subexponential. Discrete logarithm modulo a prime `p` is in the same boat (Gordon's number-field-sieve adaptation, `exp(O((log p)^{1/3}(log log p)^{2/3}))`). These two problems are not interesting only as number theory; an entire generation of public-key cryptography — most prominently the RSA cryptosystem of Rivest, Shamir and Adleman (1978) — rests its security on the *presumed* difficulty of factoring. If factoring were polynomial-time on some physically realizable machine, RSA would be broken.

There is a deeper question underneath. The "quantitative" (or "strong") Church's thesis holds that any physical computing device can be simulated by a Turing machine with only polynomial slowdown — i.e. the class of efficiently solvable problems is machine-independent. Every classical counterexample proposed (analog machines, etc.) has been ruled out as unphysical or as needing exponentially precise parts or energy. But the universe is quantum, not classical. A solution to the factoring question that runs on a quantum machine would be a candidate counterexample to the strong Church's thesis, in a problem people care about — not a contrived oracle.

## Background

**The model of quantum computation.** A register of `n` two-state systems is described not by `n` bits but by a unit vector in a `2^n`-dimensional complex Hilbert space, `Σ_i a_i |S_i⟩` with `Σ|a_i|² = 1`. Two operations are available. (1) *Unitary evolution*: the state is multiplied by a unitary matrix; locality is imposed by allowing only one- and two-bit gates, and it is known that all one-bit gates together with the controlled-NOT form a universal set (Barenco et al. 1995; DiVincenzo 1995; Sleator–Weinfurter 1995; Deutsch et al. 1995). (2) *Measurement*: observing the register in the canonical basis returns `S_i` with probability `|a_i|²` and collapses the state onto `|S_i⟩`. The class of problems solvable in quantum polynomial time with bounded error is called BQP, by analogy with the classical BPP.

The single most important phenomenon is **interference**. A basis state can be reached along several computational paths; the amplitudes for those paths *add as complex numbers* before being squared. Paths can reinforce or cancel. A quantum algorithm is therefore not "try all inputs in parallel and read one off" (a measurement returns a single random outcome); it is "arrange the unitary so that the amplitudes of the *wanted* answers add up and the amplitudes of the *unwanted* answers cancel."

**Reversibility.** Unitary maps are invertible, so every step must be reversible. Bennett (1973) and Lecerf showed any polynomial-time classical computation can be made reversible at constant time cost, by computing `x → (x, F(x))`, copying out the answer, and uncomputing the scratch (the RECORD register) to erase it — crucial, because uncomputed garbage entangled with the answer would spoil the interference. Toffoli and Fredkin gates are universal for reversible classical logic.

**The quantum precursors.** Benioff (1980, 1982) showed reversible unitary evolution suffices to simulate a Turing machine — quantum mechanics is at least as powerful as classical computation. Feynman (1982, 1986) argued that simulating quantum systems on a classical computer seems to cost exponentially, and suggested a quantum-mechanical computer to avoid it — implicitly asking whether such a machine could compute *more* efficiently. Deutsch (1985, 1989) made the question precise, defining quantum Turing machines and quantum circuits. Deutsch–Jozsa (1992) and Berthiaume–Brassard (1992) exhibited problems quantum machines solve *exactly* in cases where classical machines need randomness — but nothing outside BPP, the accepted class of "efficiently solvable."

**The number theory that makes factoring reducible.** It has been known since Miller (1976) that, using randomization, factoring an odd integer reduces to *order-finding*: given a unit `x` modulo `N`, find the least `r` with `x^r ≡ 1 (mod N)`. The reduction is purely classical and polynomial-time. Supporting facts, all standard (Hardy–Wright; Knuth): the multiplicative group `(Z/p^α)*` modulo an odd prime power is cyclic; the Chinese remainder theorem decomposes a random choice mod `N` into independent random choices mod each prime power; Euler's totient `φ(r)` counts the integers below `r` coprime to it, and `φ(r)/r > δ/log log r` for a constant `δ` (Hardy–Wright, Thm 328). The Euclidean algorithm computes gcd in `O((log N)²)`.

**Continued fractions.** Every rational `c/q` has a finite continued-fraction expansion `[a_0, a_1, …]`, computed by a simple recurrence; its convergents `p_n/q_n` are exactly the best rational approximations, and there is at most one fraction with bounded denominator within a small enough window of a given real number (Hardy–Wright, Ch. X; Knuth).

**The discrete Fourier transform and the FFT.** The DFT of length `q` sends `|a⟩` to `q^{-1/2} Σ_c exp(2πi ac/q)|c⟩`. Classically the fast Fourier transform (Cooley–Tukey; Knuth) computes it in `O(q log q)` arithmetic operations by recursively exploiting the factorization of `q`. It is the standard transform associated with the additive structure of `Z_q`.

**Simon's problem (Simon 1994).** Bernstein–Vazirani (1993) gave the first superpolynomial oracle separation, but their problem looked contrived. Simon (1994) gave a natural one: a black-box `f` on `n`-bit strings promised to satisfy `f(x) = f(y)` iff `x ⊕ y ∈ {0, s}` for a hidden `s`. Classically, finding `s` needs `Θ(2^{n/2})` queries (you must collide). Simon's quantum algorithm needs `O(n)`. Its mechanism: put the input register in uniform superposition; compute `f` into a second register; *measure the second register*, collapsing the first onto a single coset `{x_0, x_0 ⊕ s}`; apply a Fourier transform over the binary vector space `(Z_2)^n` (a layer of Hadamards) to the first register; measure, obtaining a random `y` with `y · s = 0`; repeat to collect `n−1` independent linear constraints and solve for `s`. The transform turns a *hidden period* (here the period is the shift `s` of an XOR-translation) into *measurable* interference.

## Baselines

**Classical factoring — the number field sieve** (Lenstra–Lenstra–Manasse–Pollard; Lenstra–Lenstra 1993). The asymptotically best classical factorer: collect integers `x` with `x² ≡ y (mod N)` and `y` smooth, sieve to find a subset whose product is a perfect square, obtaining `u² ≡ v² (mod N)`, then `gcd(u−v, N)` is likely a nontrivial factor. Running time `exp(c (log N)^{1/3}(log log N)^{2/3})` — subexponential but still superpolynomial in `log N`.

**Miller's reduction (Miller 1976).** Factoring `N` reduces to computing multiplicative orders, assuming (in Miller's deterministic version) the extended Riemann hypothesis; the randomized version needs no unproven hypothesis. Choose random `x` with `gcd(x, N) = 1`, find the order `r` of `x`, and, if `r` is even and `x^{r/2} ≢ -1 (mod N)`, form `gcd(x^{r/2} − 1, N)`.

**Simon's algorithm (Simon 1994).** The quantum period-finder for XOR-periodicity over `(Z_2)^n`, described above.

**Deutsch–Jozsa / Bernstein–Vazirani.** Earlier oracle separations between quantum and classical computation.

## Evaluation settings

The natural yardstick is asymptotic gate/step count as a function of the input length `l = log N` (or `log p`), against the number field sieve's `exp(c l^{1/3}(log l)^{2/3})`. The relevant cost centers are the cost of reversible modular exponentiation (repeated squaring of `l`-bit numbers modulo `N`; `O(l)` multiplications, each `O(l²)` by longhand or `O(l log l log log l)` by Schönhage–Strassen), the gate cost of the Fourier transform to be built, and the number of times the whole procedure must be repeated to succeed with high probability. Correctness is judged by a success-probability bound and the count of classical post-processing steps (Euclidean gcd, continued-fraction expansion). A small instance such as `N = 15` or `N = 91 = 7·13` is the natural sanity check for a classical simulation of the procedure. No machine exists to run it on; the measure is the proof.

## Code framework

Available primitives: classical bignum arithmetic (random integers, gcd by Euclid, modular exponentiation by repeated squaring), and a small simulator of a qubit register supporting state preparation, applying a unitary described as a matrix or as a reversible classical function, and measurement.

```python
import math, random

# --- existing classical primitives ---
def gcd(a, b):                      # Euclid, O((log N)^2)
    while b: a, b = b, a % b
    return a

def modexp(base, exp, mod):         # repeated squaring
    r = 1; base %= mod
    while exp:
        if exp & 1: r = (r * base) % mod
        base = (base * base) % mod; exp >>= 1
    return r

# --- generic quantum-register simulator ---
class QuantumRegister:
    """A register of qubits over a 2^n-dim state vector."""
    def uniform_superposition(self, n_qubits): pass   # Hadamard each bit
    def apply_reversible(self, f):  pass              # |a>|0> -> |a>|f(a)>
    def apply_unitary(self, U):     pass              # multiply by a unitary
    def measure(self):              pass              # sample |amp|^2, collapse

# --- missing period-finding subroutine ---
def find_period(x, N):
    """Given x and N, return (a candidate for) the period r of a -> x^a mod N.
    A fast version has available:
      - a register in uniform superposition over a in [0, q)
      - x^a mod N computed into a second register (reversible modexp)
      - unitary operations and measurement on the register
    # TODO: build the period-finding subroutine
    """
    raise NotImplementedError

def factor(N):
    """Classical reduction shell (Miller); the order r comes from find_period."""
    # handle N even / prime power separately (classical)         # TODO
    while True:
        x = random.randrange(2, N)
        g = gcd(x, N)
        if g != 1:
            return g                                             # lucky hit
        r = find_period(x, N)                                    # <<< the slot
        # use r to extract a factor                              # TODO
        raise NotImplementedError
```
