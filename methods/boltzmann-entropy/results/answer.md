# Boltzmann's statistical entropy: S = k log W

## The problem

Clausius's entropy is defined only by dS = dQ_rev/T between equilibrium states, increases by fiat in
the second law, and has no molecular meaning and no value at all for a non-equilibrium gas. The task is
a mechanical definition of entropy that (i) reduces to Clausius's for an equilibrium gas, (ii) is
additive over independent bodies, (iii) is defined for every state, and (iv) explains why entropy
increases — given that the underlying mechanics is time-reversible.

## The key idea

Describe the gas at three levels: the macrostate (T, p, V); the **state distribution** = the
occupation numbers (w_0, w_1, …) counting how many molecules have each energy/velocity, ignoring which
molecules; and the **complexion** = a full labeled microscopic assignment. Grant every complexion equal
a-priori probability. Then the probability of a macrostate is the **number of complexions it
contains** — its multiplicity W. Equilibrium is the macrostate with the most complexions; the second
law is the gas drifting to that overwhelmingly dominant macrostate. Entropy measures the size of a
macrostate's complexion set, and because complexion-counts **multiply** for independent bodies while
entropy must **add**, entropy is the **logarithm** of the count:

    S = k · log W.

## Derivation 1 — the multiplicity and S = k log W

Discretize kinetic energy into rungs 0, ε, 2ε, …, pε. A state distribution (w_0,…,w_p) with
Σ w_i = n (molecule number) and Σ i·w_i = λ (energy, L = λε). The number of complexions realizing it is
the multinomial

    P = n! / (w_0! w_1! … w_p!).      [multiplicity W of the macrostate]

(Check, n = L/ε = 7, p = 7: the macrostate 3 at rung 0, 2 at rung 1, 1 at rung 2, 1 at rung 3 has
P = 7!/(3!2!1!1!) = 420, the maximum; the lopsided "all energy on one molecule" has P = 7!/(6!1!) = 7.)

For two independent bodies the joint multiplicity is the product, W(A+B) = W(A)·W(B). Entropy is
additive, S(A+B) = S(A)+S(B). Under the ordinary regularity expected of a physical state function, the
unique f with f(W_A W_B) = f(W_A)+f(W_B) is the logarithm, so

    S = k log W   (the permutability measure Ω = log W up to an additive constant, chosen so
                   Ω is additive: Ω(A+B) = Ω(A) + Ω(B)).

## Derivation 2 — the most probable distribution is Maxwell–Boltzmann

Maximize P ⇔ minimize Σ ln w_i!. Stirling (w ≫ 1): ln w! ≈ w ln w − w (the ½ln 2πw correction is
subextensive next to the terms that scale with n). Minimize Σ w_i ln w_i subject to the
two constraints with multipliers h, k:

    ∂/∂w_i [ Σ w ln w + h Σ w + k Σ i w ] = ln w_i + 1 + h + k i = 0
    ⇒ w_i = w_0 x^i,   x = e^{−k}     (geometric in rung, exponential in energy).

Imposing Σ w_i = n and Σ i w_i = λ and taking the tall-ladder limit p → ∞:

    x = λ/(n+λ).

With mean energy per molecule μ = L/n = λε/n, so n/λ = ε/μ, this gives x = 1/(1+ε/μ) → e^{−ε/μ}, hence

    w_s / n ∝ x^s = e^{−sε/μ} = e^{−E/μ}      (E = sε),

which is the Boltzmann factor e^{−E/kT} once the energy scale μ is identified with the temperature
scale kT in this one-coordinate count.

x depends only on the mean energy (temperature), not on n or λ separately. The Hessian quadratic form
δ²M = Σ (δw_i)²/w_i > 0 confirms a true maximum of P.

Continuum limit (w_i = ε f(iε)): maximize Ω = −∫ f ln f dx s.t. ∫f dx = n, ∫x f dx = L. The variation
∫[ln f + 1 + k + h x] δf dx = 0 forces

    f(x) = C e^{−h x}.

Counting in **velocity components** (the measure mechanics preserves, du dv dw — not energy, which
would give the wrong ω dω weight) yields the full Maxwell distribution

    f(u,v,w) = C e^{−h (m/2)(u²+v²+w²)}   ∝ e^{−mv²/2kT},

and the speed law f(v) ∝ v² e^{−mv²/2kT}. (Maximizing instead Σ ln w_i, or a with-replacement urn
multinomial, gives the wrong dispersion / no clean limit — only complexion-counting reproduces
equilibrium.)

## Derivation 3 — the H-theorem (dynamical second law)

Density f(x,t) over energy x evolves by the collision (transport) equation, with the molecular-chaos
collision-number ansatz (pre-collision pair density factorizes as f(x)f(x′)):

    ∂f(x,t)/∂t = ∫∫ [ f(ξ)f(x+x′−ξ) − f(x)f(x′) ] · K dx′ dξ.

Define H[f] = ∫ f ln f over the same one-particle cell measure. Then

    dH/dt = ∫ (ln f + 1) ∂f/∂t dx = ∫∫∫ ln f(x) · [σ − s] · K,

with s = f(x)f(x′), σ = f(ξ)f(x+x′−ξ). A collision admits four equivalent labelings. The two incoming
labelings contribute ln f(x) and ln f(x′) multiplying (σ − s); the reverse labelings contribute the two
outgoing logarithms with the opposite factor (s − σ), so they enter with minus signs after rewriting
against (σ − s). Averaging the four equal forms:

    dH/dt = (1/4) ∫∫∫ [ln f(x) + ln f(x′) − ln f(ξ) − ln f(x+x′−ξ)] (σ − s) K
          = (1/4) ∫∫∫ ln(s/σ) · (σ − s) · K dx dx′ dξ.

For any positive s, σ: if s > σ, then ln(s/σ) > 0 and (σ − s) < 0; if s < σ, then ln(s/σ) < 0 and
(σ − s) > 0; if s = σ, the product is zero. Therefore ln(s/σ)(σ − s) ≤ 0 pointwise. Hence

    dH/dt ≤ 0,

with equality iff f(x)f(x′) = f(ξ)f(x+x′−ξ) for every collision with nonzero kernel, i.e. ln f additive
in the conserved energy, i.e. f ∝ e^{−hx} in the rest frame — the Maxwell distribution. So H decreases
monotonically unless the distribution is Maxwellian; with S = −kH (up to a constant) this is the
second law: entropy increases to its maximum, equilibrium = Maxwell = maximum multiplicity. The
monotonicity is probabilistic, not
strictly mechanical: it rests on the pre-collision factorization, an overwhelmingly-good statistical
fact for astronomically many molecules — which is also why it does not conflict with the time-symmetry
of the underlying equations or with Liouville's theorem (entropy is the log of a coarse-grained
complexion count, not of the conserved fine-grained phase volume).

## The result, assembled

For an ideal monatomic gas the permutability measure evaluates to
Ω = (3N/2) + N ln[V (4πT/3m)^{3/2}] − N ln N, and with pV = (2/3)NT and dQ = N dT + p dV one finds
∫ dQ/T = (2/3) Ω, i.e. Clausius's thermodynamic entropy equals (a constant times) Boltzmann's log of
the multiplicity. The three landed pieces:

- **S = k log W** — entropy is the logarithm of the number of microstates of a macrostate; the log is
  forced by additivity of S against multiplicativity of W.
- **f ∝ e^{−E/kT}** — the Maxwell–Boltzmann distribution as the maximum-multiplicity state.
- **dH/dt ≤ 0, H = ∫ f ln f** — the dynamical second law; H minimal (entropy maximal) at Maxwell.

## Worked check (small enumerable case)

```python
from math import factorial, log

def arrangements(occ):                 # W = n! / prod(w_i!)
    n, p = sum(occ), factorial(sum(occ))
    for w in occ:
        p //= factorial(w)
    return p

def entropy(occ, k=1.0):               # S = k log W ; additive because W multiplies
    return k * log(arrangements(occ))

def macrostates(n, units, max_level):  # occupation tuples: sum=n, sum(i*w_i)=units
    def rec(level, m, e):
        if level == 0:
            if e == 0: yield (m,)
            return
        for w in range(m + 1):
            if w * level <= e:
                for tail in rec(level - 1, m - w, e - w * level):
                    yield tail + (w,)
    yield from rec(max_level, n, units)

def most_probable(n, units, max_level):
    return max(macrostates(n, units, max_level), key=arrangements)

def equilibrium_occupations(n, units, max_level):   # large-n: w_i ∝ x^i, x = mean/(mean+1)
    mean = units / n
    x = mean / (mean + 1.0)
    raw = [x**i for i in range(max_level + 1)]
    Z = sum(raw)
    return [n * r / Z for r in raw]

if __name__ == "__main__":
    eq = most_probable(7, 7, 7)
    print(eq, "W =", arrangements(eq), "S =", round(entropy(eq), 4))   # (3,2,1,1,0,0,0,0) W=420
    print("lopsided W =", arrangements((6,0,0,0,0,0,0,1)))             # 7
    print([round(w, 3) for w in equilibrium_occupations(7, 7, 7)])     # Boltzmann ladder
```
