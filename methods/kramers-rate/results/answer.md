# Kramers' rate theory: barrier escape as Fokker–Planck diffusion

## Problem

Compute the rate at which a system escapes a metastable well over an energy
barrier E_b ≫ k_B T, and determine when the equilibrium-flux (transition-state)
prefactor is correct. The empirical Arrhenius law k = ν·exp(−E_b/k_B T) leaves ν
unexplained; transition-state theory computes ν from equilibrium statistics but
rests on two unverified dynamical assumptions — no recrossing of the saddle, and a
saddle population held at its equilibrium value — and so cannot say how the rate
depends on the system's coupling to its medium.

## Key idea

Model the medium by a single friction coefficient η (mass = 1, k_B = 1, so T is in
energy units) in a Langevin equation, and follow the **probability density in
phase space** ρ(p,q,t) by its Fokker–Planck (Klein–Kramers) equation. Treat escape
as a **quasi-stationary flux over the barrier divided by the well population** (rate
= flux/population). Solving the flow as a function of η reveals that the escape
rate **rises with η, plateaus at the transition-state value, then falls as 1/η** —
a turnover. The transition-state prefactor is the *ceiling*, attained only in an
intermediate band of friction; friction multiplies it by a computable
**transmission factor** κ ≤ 1.

## The phase-space diffusion equation (Klein–Kramers)

With ṗ = −U'(q) + X(t), q̇ = p, and the Einstein closure for the white noise
(drift −ηp, momentum diffusion 2ηT) — the strength 2ηT being forced by requiring
the Boltzmann distribution exp(−(½p²+U)/T) to be stationary, which is the
fluctuation–dissipation relation —

    ∂ρ/∂t = −K(q)∂ρ/∂p − p∂ρ/∂q + η·∂/∂p( pρ + T·∂ρ/∂p ),     K = −∂U/∂q.

This is a continuity equation; the escape rate is its stationary current over the
barrier divided by the well population (flux-over-population).

## The three closed-form rates

Let ω = ordinary frequency at the well bottom (angular ω_0 = 2πω), ω′ = ordinary
frequency of the unstable mode at the barrier top (angular ω_b = 2πω′), E_b =
barrier height, T = temperature (energy units), η = friction.

**Intermediate-to-high friction (parabolic-barrier, full solution).** Solving the
stationary equation near a parabolic barrier exactly (ansatz ρ = ζ·exp(−H/T) with
ζ a function of the unstable normal-mode combination u = p − a q′, which forces
(2πω′)² = a(a−η) and selects the root a − η > 0):

    r = (ω / 2πω′)·( √(η²/4 + (2πω′)²) − η/2 )·exp(−E_b/T)
      = κ · r_TST ,   r_TST = ω·exp(−E_b/T),

with the **transmission factor**

    κ = (1/2πω′)·( √(η²/4 + (2πω′)²) − η/2 ) = √(1 + (η/2ω_b)²) − η/2ω_b .

  - η/2 ≪ 2πω′:  κ → 1, so r → ω·exp(−E_b/T) = r_TST (the transition-state value).
  - η/2 ≫ 2πω′:  κ → 2πω′/η, so the **high-friction (Smoluchowski) rate**
        r → (2π·ω·ω′/η)·exp(−E_b/T)     [= (ω_0 ω_b)/(2πη)·exp(−E_b/T)],
    falling inversely with friction — the recrossing-limited regime.

**Very weak friction (energy-diffusion-limited).** When the friction is too feeble
to keep the barrier-energy states supplied, the rate is throttled by diffusion in
energy. With I(E) = ∮p dq the action and I_c = I(E_b) ≈ E_b/ω its barrier value,

    r = η·(I_c·ω/T)·exp(−E_b/T) ≈ η·(E_b/T)·exp(−E_b/T),

rising **linearly with η** — the energy-supply-limited regime. (It is the action at
the *barrier* energy, hence any anharmonicity of the well out near E_b, that
enters here.)

## The turnover and the validity of the transition-state prefactor

The rate-versus-friction curve rises ∝ η (energy diffusion), plateaus at r_TST =
ω·exp(−E_b/T), then falls ∝ 1/η (spatial diffusion). The transition-state
prefactor is the maximum, reached only for a band of friction — roughly
ω·T/E_b ≲ η ≲ 1.2 ω′ (the upper bound being the *ordinary* barrier frequency,
≈ ω_b/5, where κ ≈ 0.9) — within which it is accurate to ~10% for E_b/T ≈ 10.
Outside that band it overestimates the rate: at high friction by repeated
recrossing of the saddle, at low friction by failure of the energy supply
(Nachlieferung). The bridge directly across the turnover peak (η comparable to the
barrier frequency) is not given in closed form here; for large E_b/T the two valid
branches already agree on the plateau, so the unresolved region is narrow.

## Worked numerical illustration

Reproduces the escape-rate-vs-friction turnover and verifies the two analytic
limits of the transmission factor (illustrative case ω′ = ω, E_b/T = 10).

```python
import numpy as np

# units: mass = 1, k_B = 1.  omega0, omegab are ANGULAR frequencies.

def k_tst(omega0, Eb, T):
    """Transition-state value: equilibrium one-way flux over the saddle (the plateau)."""
    return (omega0 / (2.0 * np.pi)) * np.exp(-Eb / T)

def transmission_factor(omegab, eta):
    """kappa = lambda_+/omegab,  lambda_+ = -eta/2 + sqrt(omegab^2 + (eta/2)^2)
            = sqrt(1 + (eta/2/omegab)^2) - eta/(2*omegab).
    kappa -> 1 as eta -> 0 (TST);  kappa -> omegab/eta as eta -> inf (Smoluchowski)."""
    lam_plus = -eta / 2.0 + np.sqrt(omegab**2 + (eta / 2.0) ** 2)
    return lam_plus / omegab

def k_spatial(omega0, omegab, eta, Eb, T):
    """Intermediate-to-high friction: k = kappa * k_tst (plateau -> 1/eta)."""
    return transmission_factor(omegab, eta) * k_tst(omega0, Eb, T)

def k_energy(omega0, eta, Eb, T, Ib):
    """Weak-friction, energy-supply-limited: k = eta * Ib/T * (omega0/2pi) e^{-Eb/T}, ~ eta."""
    return eta * (Ib / T) * (omega0 / (2.0 * np.pi)) * np.exp(-Eb / T)

if __name__ == "__main__":
    omega0 = omegab = 2 * np.pi          # ordinary frequency omega = omega' = 1
    T, Eb = 1.0, 10.0                    # Eb/T = 10
    f = omega0 / (2.0 * np.pi)           # ORDINARY well frequency (= 1 here)
    Ib = Eb / f                          # near-harmonic barrier action I_c = 2*pi*Eb/omega0 = Eb/f
    ktst = k_tst(omega0, Eb, T)

    etas = np.logspace(-3, 3, 400) * omegab
    ks = k_spatial(omega0, omegab, etas, Eb, T)
    ke = k_energy(omega0, etas, Eb, T, Ib)
    true_rate = np.minimum(ks, ke)       # the worse bottleneck controls the rate
    i = int(np.argmax(true_rate))
    print("plateau k_TST            =", ktst)
    print("turnover peak eta/omegab =", etas[i] / omegab,
          " k_peak/k_TST =", true_rate[i] / ktst)

    # analytic limits of the transmission factor
    assert abs(transmission_factor(omegab, 1e-4 * omegab) - 1.0) < 1e-3            # -> TST
    g = 1e4 * omegab
    assert abs(transmission_factor(omegab, g) - omegab / g) / (omegab / g) < 1e-3  # -> 1/eta
    print("limits OK: kappa(eta->0)=1 (TST); kappa(eta->inf)=omegab/eta (1/eta).")
```

Output: the rate peaks just below the transition-state plateau near η/ω_b ≈ 0.016
(where η·E_b/(T·ω) ≈ 1, i.e. η ≈ ω·T/E_b with ω the ordinary frequency), with
k_peak/k_TST ≈ 0.99, and both analytic limits of κ check out — the
energy-diffusion branch (∝ η) and the spatial-diffusion branch (∝ 1/η) meeting
under the transition-state ceiling.
