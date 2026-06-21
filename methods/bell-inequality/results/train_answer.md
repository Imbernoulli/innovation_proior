The EPR argument asks whether quantum mechanics can be completed with hidden variables that restore determinism and locality. The naive hope is that the two particles in a singlet state carry, in advance, instructions about how each will respond to any possible analyzer setting, with no influence traveling faster than light. Existing no-go theorems do not settle the question: von Neumann's proof assumes additivity of expectation values even for non-commuting observables, which is a property of quantum ensembles, not a reasonable constraint on individual hidden states, and Bohm's 1952 hidden-variable theory shows by construction that hidden variables are possible if one tolerates nonlocality. The real question, then, is whether any *local* hidden-variable theory can reproduce the quantum correlations.

The answer is Bell's inequality, in its experimentally robust CHSH form. It replaces philosophical debate with a bound that every local hidden-variable theory must satisfy and that quantum mechanics violates. A local theory is modeled in full generality by a shared hidden variable λ drawn from some distribution ρ(λ), and two outcome functions A(a, λ) and B(b, λ) with |A|, |B| ≤ 1, where A depends only on the local setting a and λ, and B depends only on the distant setting b and λ. The crucial locality assumption is the factorization: conditioned on λ, the two outcomes are statistically independent of the distant setting. From this alone one can derive a universal constraint on the correlations E(a, b) = ∫ dλ ρ(λ) A(a, λ) B(b, λ). With two settings per side, a, a′ on one analyzer and b, b′ on the other, define S = E(a, b) − E(a, b′) + E(a′, b) + E(a′, b′). For every local theory, pointwise inspection of the integrand shows |S| ≤ 2. Quantum mechanics predicts E(a, b) = −cos θ for the spin singlet, and with settings spaced 45° apart the same combination gives |S| = 2√2 ≈ 2.828, cleanly above the local ceiling.

The code below computes the quantum CHSH value for the singlet and contrasts it with a simple local hidden-variable model. The quantum part uses the cosine correlation and the canonical angle choices a = 0°, a′ = 90°, b = 45°, b′ = 135°. The local model assigns each particle an independent random hidden axis and deterministically flips its sign according to whether the axis aligns with the local field; averaging many pairs gives a local correlation whose CHSH combination never exceeds 2.

```python
import numpy as np

# Quantum-mechanical CHSH value for the spin singlet.
def quantum_chsh():
    # Settings in degrees: a=0, a'=90, b=45, b'=135
    angles = {
        ("a", "b"): 45.0,
        ("a", "b'"): 135.0,
        ("a'", "b"): 45.0,
        ("a'", "b'"): 45.0,
    }
    signs = {"a-b": +1, "a-b'": -1, "a'-b": +1, "a'-b'": +1}
    S = 0.0
    for key, theta in angles.items():
        term_key = "-".join(key)
        E = -np.cos(np.radians(theta))
        S += signs[term_key] * E
    return S

# Simple local hidden-variable model.
# Each particle carries a random unit axis; the response is sign(cos(angle to field)).
def local_model_chsh(n_samples=1_000_000):
    # Random hidden axes uniformly on the sphere, in the measurement plane.
    phi = np.random.uniform(0, 2 * np.pi, size=n_samples)
    # Particle 1 axis and opposite axis for particle 2.
    axis1 = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    axis2 = -axis1

    def response(axis, setting_deg):
        s = np.radians(setting_deg)
        field = np.array([np.cos(s), np.sin(s)])
        return np.sign(axis @ field)

    def E(a_deg, b_deg):
        A = response(axis1, a_deg)
        B = response(axis2, b_deg)
        return np.mean(A * B)

    E_ab = E(0, 45)
    E_abp = E(0, 135)
    E_apb = E(90, 45)
    E_apbp = E(90, 135)
    S_local = E_ab - E_abp + E_apb + E_apbp
    return S_local, (E_ab, E_abp, E_apb, E_apbp)

S_qm = quantum_chsh()
S_local, Es = local_model_chsh()

print(f"Quantum CHSH |S|  = {abs(S_qm):.4f}  (local bound = 2)")
print(f"Local model S     = {S_local:.4f}")
print("Local model correlations:", {k: f"{v:.4f}" for k, v in zip(
    ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"], Es)})
```

This is Bell's inequality: a quantitative, experimentally testable boundary between local hidden-variable theories and quantum mechanics. If an experiment measures the four correlations and finds |S| > 2, no local hidden-variable explanation can account for the data. The singlet state predicts the maximal quantum value 2√2, so the EPR question is no longer philosophical; it is settled by counting coincidences at the right analyzer angles.
