# Grover's algorithm — quantum search in O(√N)

## Problem

Search an unstructured database of `N` items for the single one satisfying a Boolean condition `C`
(one unknown marked state `S_ν`; `C` evaluable in unit time). Classically this needs `Θ(N)` queries —
each query rules out at most one candidate. Grover's algorithm finds the marked item in `O(√N)`
oracle queries, a quadratic speedup, and `Ω(√N)` is a matching lower bound, so it is optimal up to a
constant.

## Key idea

Start in the uniform superposition and repeatedly apply two reflections:

1. **Oracle** `U_w = I − 2|w><w|` — a selective `π` phase rotation that flips the sign of the marked
   amplitude and leaves every magnitude unchanged (a reflection across the unmarked subspace).
2. **Diffusion / inversion about the mean** `D = 2|s><s| − I` (equivalently `D = −I + 2P`,
   `P_{ij}=1/N`) — reflects every amplitude about the average, lifting the lone sign-flipped one while
   barely moving the rest.

In the 2D plane spanned by the marked state `|w>` and the unmarked-uniform state `|s'>`, the initial
uniform state is `|s> = sin θ |w> + cos θ |s'>` with `sin θ = 1/√N`. Two reflections about lines at
angle `θ` compose to a **rotation by `2θ`** toward `|w>`. After `r` iterations the marked amplitude is
`sin((2r+1)θ)` and success probability `sin²((2r+1)θ)`. Choosing `r = ⌊π/(4θ)⌋ ≈ (π/4)√N` lands the
state within `θ` of `|w>`, giving failure probability `cos²((2r+1)θ) ≤ sin²θ = 1/N`, i.e. success
`≥ 1 − 1/N`. Running longer **over-rotates**: the probability is a sinusoid in `r`, so past the optimum
the state swings back toward `|s'>` and success drops — the iteration count must be precise.

## Algorithm

1. Prepare `|s> = H^{⊗n}|0…0>`: amplitude `1/√N` in all `N=2ⁿ` basis states (`O(log N)` gates).
2. Repeat `⌊π/(4θ)⌋ ≈ (π/4)√N` times, where `sin θ = 1/√N`:
   - apply the oracle (phase-flip the marked amplitude);
   - apply `D = 2|s><s| − I`, implemented locally as `W · R · W`, where
     `R = −I + 2|0…0><0…0|`, so `D = WRW` — each round costs `O(log N)`
     gates.
3. Measure; outcome `i` with probability `|amplitude_i|²`. The marked item appears with probability
   `≥ 1 − 1/N`.

Cost: `O(√N)` oracle queries.

## Geometry, made exact

In the `(|s'>, |w>)` basis, `U_w = diag(1, −1)` and `U_s = 2|s><s| − I =
[[cos 2θ, sin 2θ],[sin 2θ, −cos 2θ]]`, so

```
U_s U_w = [[cos 2θ, −sin 2θ],
           [sin 2θ,  cos 2θ]]   = rotation by 2θ.
```

The marked amplitude after `r` iterations is exactly `sin((2r+1)θ)`.

## Optimality (BBBV `Ω(√N)` lower bound)

Run any `T`-query algorithm on the empty oracle. For each candidate `y`, let `q_y(|φ_i>)` be the
squared amplitude querying `y` at step `i`. Since each `|φ_i>` is a unit vector, `Σ_y q_y(|φ_i>) ≤ 1`,
so `Σ_y Σ_{i<T} q_y ≤ T` and some `y` is queried with total magnitude `≤ T/N`. Planting the target at
such a `y` changes only the query slice of norm `√q_y` at each step. Summing the hybrid errors gives
distance `O(Σ_i √q_y) ≤ O(√(T Σ_i q_y)) ≤ O(T/√N)`. Distinguishing the planted oracle from the empty one
requires the final states to differ by `Ω(1)`, forcing `T/√N = Ω(1)`, i.e. `T = Ω(√N)`. Grover's
`(π/4)√N` matches this floor.

## Amplitude-vector implementation

```python
import numpy as np

def grover_search(N, marked, iterations=None):
    """Find `marked` among N items with O(sqrt N) oracle queries.

    Returns the measurement distribution and the iteration count used.
    """
    theta = np.arcsin(1.0 / np.sqrt(N))
    n_iter = iterations if iterations is not None else int(np.floor(np.pi / (4.0 * theta)))

    # |s> = H^{⊗ n} |0...0>: uniform amplitude 1/sqrt(N).
    state = np.full(N, 1.0 / np.sqrt(N))

    for _ in range(n_iter):
        # Oracle U_w = I - 2|w><w|: phase-flip the marked amplitude (sign), magnitudes unchanged.
        state[marked] *= -1.0

        # Diffusion D = 2|s><s| - I: inversion about the mean.
        mean = state.mean()
        state = 2.0 * mean - state

    # Each iteration = rotation by 2*theta, sin(theta) = 1/sqrt(N), in span{|w>, |s'>}.
    # Marked amplitude after r iters is sin((2r+1)theta); stop near (2r+1)theta = pi/2.
    probs = np.abs(state) ** 2          # measurement: outcome i with prob |amplitude_i|^2
    return probs, n_iter

```
