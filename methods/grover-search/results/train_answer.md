Unstructured search is the problem of finding one marked item among N items with no ordering, hash, or metric to exploit. Classically, each query inspects a single candidate and returns only whether that candidate is the target, so it eliminates at most one possibility. A deterministic algorithm needs up to N-1 queries, and even a randomized strategy needs Θ(N) queries to find the target with constant probability, because every probe is essentially an independent coin flip against a one-in-N chance. Quantum parallelism by itself does not help either: preparing a uniform superposition touches all N states, but measuring it still returns a uniformly random item and succeeds with probability only 1/N. The challenge is therefore not breadth but concentration — how to make the quantum amplitude pile up on the single marked state before measurement.

The method that solves this is Grover's algorithm. It prepares a uniform superposition over all N basis states and then repeatedly applies a pair of unitary operations. The first is the oracle, which flips the sign of the amplitude of the marked state and leaves every other amplitude unchanged. This is a selective π phase rotation; it is unitary and preserves all squared amplitudes, so by itself it is invisible to measurement, but it tags the target with a minus sign. The second operation is inversion about the mean, which reflects every amplitude about the average amplitude. Because the marked amplitude is the only one below the mean, the reflection pushes it upward while leaving the others nearly in place. Geometrically, the two reflections act in the two-dimensional plane spanned by the marked state and the uniform superposition over all unmarked states. The initial uniform state makes a small angle θ with the unmarked axis, where sin θ = 1/√N. A reflection across the unmarked axis followed by a reflection across the initial-state axis composes to a rotation by 2θ toward the marked state.

Iterating this pair r times rotates the state by 2rθ, so the amplitude on the marked state becomes sin((2r+1)θ) and the success probability is sin²((2r+1)θ). The optimal stopping point is near (2r+1)θ = π/2, which gives r ≈ (π/4)√N iterations. At that point the failure probability is at most 1/N, so the marked item is found with overwhelming probability. Running longer is harmful: the dynamics is periodic, and past the optimum the state over-rotates and the success probability falls back toward zero. The total cost is therefore O(√N) oracle queries. This matches the BBBV Ω(√N) lower bound for unstructured quantum search, which argues that any T-query algorithm can only perturb the final state by O(T/√N) when the target is moved to a rarely queried item, so constant success requires T = Ω(√N). Thus Grover's algorithm is optimal up to the constant factor.

```python
import numpy as np

def grover_search(N, marked, iterations=None):
    """Find `marked` among N items with O(sqrt N) oracle queries."""
    theta = np.arcsin(1.0 / np.sqrt(N))
    n_iter = iterations if iterations is not None else int(np.floor(np.pi / (4.0 * theta)))

    # Uniform superposition: amplitude 1/sqrt(N) on every basis state.
    state = np.full(N, 1.0 / np.sqrt(N))

    for _ in range(n_iter):
        # Oracle: phase-flip the marked amplitude (selective pi rotation).
        state[marked] *= -1.0

        # Diffusion: inversion about the mean, D = 2|s><s| - I.
        mean = state.mean()
        state = 2.0 * mean - state

    # After r iterations the marked amplitude is sin((2r+1)*theta).
    # Stop near (2r+1)*theta = pi/2 to maximize success probability.
    probs = np.abs(state) ** 2
    return probs, n_iter
```
