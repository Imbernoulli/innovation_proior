The old quantum theory of Bohr and Sommerfeld can predict the hydrogen spectrum, the fine structure, and the Stark effect, but it does so by building everything on the electron's orbit. That orbit is not merely hard to measure: a photon energetic enough to resolve it would Compton-scatter the electron out of the atom, so the orbit is operationally meaningless. Worse, the most reliable empirical relation in the theory, the Bohr frequency condition, already says that radiation comes from energy differences between stationary states, not from any orbital period. When the orbit is pressed into service for crossed electric and magnetic fields, helium, or many-electron atoms, it gives wrong answers or no answers at all, and it leaves the integer-versus-half-integer quantization ambiguity unresolved. The quantities that experiments actually hand us are the spectral lines themselves: a frequency and an intensity for each transition between a pair of states. A real mechanics of the atom should start from those two-index observables and never reconstruct an orbit.

The method that does this is matrix mechanics. It keeps the classical dynamical laws but reinterprets every physical quantity as a two-index array whose entries are attached to transitions between stationary states. A coordinate is no longer x(t) for a single orbit; it is a set of amplitudes X(n, n − α) oscillating at the observable transition frequency ω(n, n − α). The central question becomes how to multiply such arrays. Classically, multiplying two Fourier series convolves their harmonic coefficients symmetrically. In the new language, each factor must carry an ordered pair of states, and the frequency of a product term must be a Ritz-allowed frequency, meaning the second leg of the product must begin where the first leg ends. That forces the product to chain through the intermediate state: C(n, n − β) = Σ_α X(n, n − α) Y(n − α, n − β). This is exactly the row-by-column rule of matrix multiplication, and because the shared index sits in the middle of an ordered chain, swapping the factors gives a different result in general. The non-commutativity is not an extra assumption; it is forced by the same observability requirement that forced the chaining rule.

With this multiplication rule in hand, the equation of motion ẍ + f(x) = 0 becomes a relation among transition arrays, and products like x² or x³ are computed by chaining through intermediate states. The quantization condition is obtained by transcribing the phase integral in the same spirit. The classical condition ∮ p dq = n h fixes the action absolutely, but the action is only observable through its rate of change dJ/dn = h; the absolute level is arbitrary, and that arbitrariness is the source of the half-integer ambiguity. Differentiating the Fourier form of the phase integral and applying the dispersion-theory transcription turns the derivative into a difference between the line above a state and the line below it. The result is h = 4πm Σ_{α ≥ 1} [ |a(n + α, n)|² ω(n + α, n) − |a(n, n − α)|² ω(n, n − α) ], which is the Thomas–Reiche–Kuhn sum rule. The remaining additive constant is fixed by the normal-state condition a(n₀, n₀ − α) = 0 for all α > 0: the lowest state has no lower state to radiate to. This condition also forces integer quantum numbers and removes the half-integer problem cleanly.

On the anharmonic oscillator ẍ + ω₀²x + λx² = 0, the method produces recursion relations in which every product chains through an intermediate state. To lowest order one finds ω(n, n − 1) = ω₀ and a²(n, n − 1) = n h/(πmω₀), with the constant fixed by the normal-state boundary. The energy W = mẋ²/2 + mω₀²x²/2 then becomes W(n, n) = (n + ½) hω₀/(2π), because the diagonal energy receives contributions from both the upward and downward transitions adjacent to state n. The half-quantum of zero-point energy falls out of the kinematics rather than being put in by hand. The off-diagonal elements of the energy array vanish, so the energy is truly constant, and the Bohr frequency condition ω(n, n − 1)/2π = (1/h)[W(n) − W(n − 1)] follows as a consequence of the construction. For the quartic anharmonic oscillator the results agree with the independent Kramers–Born perturbation calculation. The same framework also reproduces known intensity sum rules for the rigid rotator, giving further confidence that the transition-array kinematics is correct.

```python
import numpy as np

def array_product(X, Y, states):
    """Chained product forced by the Ritz combination principle.

    C(i, k) = sum over intermediate m of X(i, m) * Y(m, k).
    X[(i, j)] and Y[(i, j)] give the amplitude on the transition i -> j.
    This is ordinary square-matrix multiplication and is non-commutative.
    """
    return {
        (i, k): sum(X.get((i, m), 0) * Y.get((m, k), 0) for m in states)
        for i in states for k in states
    }


def quantum_condition_lhs(a, omega, n, alpha_max, m):
    """Left-hand side of the observable quantum condition.

    Returns 4*pi*m * sum_{alpha>=1} [
        |a(n+alpha, n)|^2 * omega(n+alpha, n)
      - |a(n, n-alpha)|^2 * omega(n, n-alpha)
    ], which should equal Planck's constant h.
    """
    s = sum(
        abs(a.get((n + al, n), 0)) ** 2 * omega.get((n + al, n), 0)
        - abs(a.get((n, n - al), 0)) ** 2 * omega.get((n, n - al), 0)
        for al in range(1, alpha_max + 1)
    )
    return 4 * np.pi * m * s


def harmonic_oscillator(n, h, m, omega0):
    """Lowest-order harmonic-oscillator result from matrix mechanics.

    With the normal-state condition a(0, -1) = 0, the method gives
        |a(n, n-1)|^2 = n*h / (pi*m*omega0),
        W(n, n)       = (n + 1/2) * h*omega0 / (2*pi).
    """
    a2_down = n * h / (np.pi * m * omega0)          # |a(n, n-1)|^2
    a2_up = (n + 1) * h / (np.pi * m * omega0)      # |a(n+1, n)|^2
    energy = 0.25 * m * omega0 ** 2 * (a2_down + a2_up)
    return a2_down, energy


# Example: ground state and first excited state of a harmonic oscillator.
if __name__ == "__main__":
    h = 6.62607015e-34      # J*s
    m = 9.10938356e-31      # kg (electron mass, for illustration)
    omega0 = 2.0 * np.pi * 1e15  # rad/s

    for n in (0, 1):
        a2, W = harmonic_oscillator(n, h, m, omega0)
        print(f"n={n}: |a(n,n-1)|^2={a2:.3e}, W={(W / h):.3e} Hz")
```
