Clausius gave entropy as a state function defined by reversible heat transfer, dS = dQ_rev/T, and summarized the second law as "the entropy of the universe tends to a maximum." That formulation is experimentally indispensable, but it is silent about what entropy is at the molecular level, why it should increase rather than decrease, and what its value is for a state that is not in equilibrium. A gas is a swarm of molecules governed by mechanics, and mechanics itself has no preferred direction of time: reverse every velocity and a spreading gas would re-collect just as lawfully. So a one-way increase cannot be a strict theorem of the microdynamics alone. Previous accounts either stayed macroscopic, like Clausius's integral, or stayed at equilibrium, like Maxwell's derivation of the velocity distribution from a functional equation. They did not close the gap between the reversible collisions and the irreversible rise of entropy, nor did they define entropy away from equilibrium.

The right move is to stop asking about a single detailed microstate and to ask, instead, how many detailed microstates correspond to a coarse description of the gas. Split the description into three levels. At the top is the macrostate — temperature, pressure, volume. At the bottom is the complexion, a complete labeled assignment of every molecule's position and velocity. In between is the state distribution: how many molecules have each energy or velocity, without tracking which molecule is which. If all complexions are equally likely a priori, then the probability of a macrostate is proportional to the number of complexions that realize it. Equilibrium is simply the macrostate that contains the overwhelming majority of complexions. This converts the second law from a postulate into a statement about counting.

The new method is Boltzmann's statistical entropy, S = k log W, where W is the multiplicity — the number of complexions compatible with a given macrostate. The logarithm is not a convenience for handling large numbers; it is forced by the requirement that entropy be additive. Multiplicities multiply for independent systems, W(A+B) = W(A) W(B), while entropy must add, S(A+B) = S(A) + S(B). Under the usual regularity assumptions, the only function that turns products into sums is the logarithm, so entropy must be proportional to log W. To compute W, discretize the kinetic-energy axis into rungs 0, ε, 2ε, ..., pε. A state distribution (w_0, ..., w_p) assigns w_i molecules to rung i, subject to Σ w_i = n and Σ i w_i = λ, where L = λε is the total energy. The number of complexions realizing this distribution is the multinomial n! / (w_0! w_1! ... w_p!). Maximizing this count, or equivalently minimizing Σ w_i ln w_i after Stirling's approximation, yields w_i ∝ x^i with x determined by the mean energy alone. In the continuum limit the equilibrium density is f ∝ e^{-E/kT}, the Maxwell-Boltzmann distribution. Counting must be done in velocity space, using the measure du dv dw that mechanics preserves; counting in energy space alone gives the wrong speed distribution.

This entropy is defined for any state distribution, equilibrium or not, so it solves the non-equilibrium problem. Its increase is explained by the gas wandering from a low-multiplicity macrostate toward the overwhelmingly dominant high-multiplicity macrostate. The dynamical counterpart is the H-theorem. Let H[f] = ∫ f ln f, where f(x,t) is the one-particle distribution. From the collision equation with the molecular-chaos assumption that pre-collision pairs are uncorrelated, the symmetry of a collision under relabeling and time reversal gives dH/dt = (1/4) ∫∫∫ ln(s/σ) (σ - s) K, where s and σ are the incoming and outgoing pair densities. The integrand is always non-positive, so dH/dt ≤ 0, with equality only when ln f is additive in the conserved energy, i.e. when f is Maxwellian. Since entropy is proportional to -H, entropy increases monotonically until the maximum-multiplicity equilibrium is reached. The irreversibility enters through the statistical assumption about pre-collision correlations, not through the mechanics itself, which remains time-reversible.

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
