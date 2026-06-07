# The Gaussian-Noise (GN) model of nonlinear propagation

## Problem

In an uncompensated coherent WDM fiber link, two impairments set the received signal-to-noise ratio: amplifier ASE noise and Kerr-nonlinear signal distortion. ASE is trivially additive Gaussian; the nonlinearity four-wave-mixes the whole densely packed comb and is hard to count. The GN model predicts the per-channel SNR analytically — fast enough to drive launch-power, modulation-and-coding, and spectrum-allocation decisions — instead of running a split-step NLSE/Manakov simulation per configuration.

## Key idea

Because dispersion is compensated digitally at the receiver and left uncompensated along the fiber, each channel's field "Gaussianizes" shortly after launch. The Kerr nonlinearity then generates an enormous number of uncorrelated in-band four-wave-mixing beats whose sum, by the central-limit theorem, is **additive Gaussian noise**, statistically independent of ASE. So the noise variances simply add:

SNR = P / (P_ASE + σ²_NLI),   BER = Ψ(SNR).

A first-order perturbation ("undepleted-pump") solution of the propagation equation, plus an average over the data symbols, reduces the nonlinear-interference (NLI) PSD to a closed-form double integral of the transmitted PSD — the **GN reference formula (GNRF)**:

G_NLI(f) = (16/27) γ² ∫∫ G_Tx(f₁) G_Tx(f₂) G_Tx(f₁+f₂−f)
           · |(1 − e^{−2α_fL_s} e^{jΔβ L_s}) / (2α_f − jΔβ)|²
           · sin²(N_s Δβ L_s / 2) / sin²(Δβ L_s / 2)  df₁ df₂,

with phase mismatch Δβ = 4π² (f₁−f)(f₂−f) [β₂ + πβ₃(f₁+f₂)].

Here α_f is the field loss (power ∝ e^{−2α_fz}), β₂, β₃ are dispersion, γ the nonlinearity, L_s the span length, N_s the number of identical spans. The middle factor is the per-span FWM-efficiency; the trailing factor is the coherent multi-span "phased-array" term. The dual-polarization Manakov averaging gives the 8/9 field-factor in the propagation equation and the 16/27 GN coefficient for SCI; XCI uses twice that coefficient because both symmetric beat orderings contribute.

## Consequences

- **Cubic law.** G_Tx enters the integral cubed, so NLI power is P_NLI = η P³, with η a per-link constant independent of launch power.
- **Optimum power / NLI wall.** SNR = P/(P_ASE + ηP³) has a maximum where dSNR/dP = 0 ⇒ P_ASE = 2ηP³ = 2 P_NLI. Thus P_opt = (P_ASE / 2η)^{1/3}, and at the optimum the denominator is 1.5 P_ASE, so MaxSNR = P_opt / (1.5 P_ASE).
- **Multi-span accumulation.** With large per-span dispersion the phased-array factor averages out and NLI accumulates incoherently: P_NLI ∝ N_s.
- **Closed-form approximation.** Modeling Nyquist channels as flat-top rectangles and keeping only self-channel (SCI) and cross-channel (XCI) interference (dropping the negligible three-distinct-channel MCI) collapses the 2D integral to inverse-hyperbolic-sine terms, giving NLI as a cheap sum over channel pairs.

## Algorithm (closed-form, per channel)

For the closed-form code, use the power-attenuation coefficient a = 2α_f, matching common fiber APIs. For each ordered pair (cut channel i, pump channel j), with offset Δf = f_j − f_i, baud rates R_i, R_j, span effective length L_eff = (1−e^{−aL_s})/a and asymptotic length L_a = 1/a:

ψ(Δf) = [asinh(π² L_a |β₂| R_i (Δf + R_j/2)) − asinh(π² L_a |β₂| R_i (Δf − R_j/2))] / 2 · L_eff² / (2π |β₂| L_a)

ρ_ij = γ² · w_ij · ψ(Δf) / (R_i R_j²),   η_ij = R_iρ_ij,   w_ii = 16/27 (SCI),   w_ij = 2·16/27 (XCI).

P_NLI(i) = N_s · Σ_j P_i P_j² η_ij,   SNR_i = P_i / (P_ASE + P_NLI(i)).

## Code

```python
import numpy as np

SPM_WEIGHT = 16.0 / 27.0          # self-channel Manakov weight
XPM_WEIGHT = 2.0 * 16.0 / 27.0    # cross-channel weight (both beat orderings)

class Channel:
    def __init__(self, f0, baud_rate, power, roll_off=0.0):
        self.f0 = f0                  # center frequency [Hz]
        self.baud_rate = baud_rate    # symbol rate [Hz] (~ bandwidth, Nyquist)
        self.power = power            # launch power per channel [W]
        self.roll_off = roll_off

class Span:
    def __init__(self, length, alpha, beta2, gamma, beta3=0.0):
        self.length = length          # [m]
        self.alpha = alpha            # power attenuation [1/m]
        self.beta2 = beta2            # [s^2/m]
        self.beta3 = beta3            # [s^3/m]
        self.gamma = gamma            # [1/(W*m)]

def effective_length(alpha, length):
    return (1.0 - np.exp(-alpha * length)) / alpha

def asymptotic_length(alpha):
    return 1.0 / alpha

def ase_power(gain, noise_figure, nu, bandwidth):
    h = 6.62607015e-34
    F = 10 ** (noise_figure / 10.0)
    return F * (gain - 1.0) * h * nu * bandwidth

def interaction_factor(df, baud_cut, baud_pump, beta2_cut, beta2_pump, L_eff, L_a):
    """Closed-form rectangular SCI/XCI FWM-efficiency island (asinh form)."""
    beta2 = (beta2_cut + beta2_pump) / 2.0
    abs_beta2 = abs(beta2)
    right = df + baud_pump / 2.0
    left  = df - baud_pump / 2.0
    val = (np.arcsinh(np.pi**2 * L_a * abs_beta2 * baud_cut * right) -
           np.arcsinh(np.pi**2 * L_a * abs_beta2 * baud_cut * left)) / 2.0
    val *= L_eff**2 / (2.0 * np.pi * abs_beta2 * L_a)
    return val

def coefficient_matrix(channels, span):
    """eta[i,j]: NLI on i per (P_i * P_j^2) from pump j (single span)."""
    n = len(channels)
    L_eff = effective_length(span.alpha, span.length)
    L_a   = asymptotic_length(span.alpha)
    eta = np.zeros((n, n))
    for i, cut in enumerate(channels):
        for j, pump in enumerate(channels):
            df = pump.f0 - cut.f0
            w  = SPM_WEIGHT if i == j else XPM_WEIGHT
            p  = interaction_factor(df, cut.baud_rate, pump.baud_rate, span.beta2, span.beta2, L_eff, L_a)
            eta_density = span.gamma**2 * w * p / (cut.baud_rate * pump.baud_rate**2)
            eta[i, j] = cut.baud_rate * eta_density
    return eta

def nli_power(channels, span, n_spans):
    """Cubic-in-power NLI; incoherent accumulation over identical spans."""
    eta = coefficient_matrix(channels, span) * n_spans
    P = np.array([c.power for c in channels])
    Pcut  = np.outer(P, np.ones(len(P)))
    Ppump = np.outer(np.ones(len(P)), P)
    return np.sum(Pcut * Ppump**2 * eta, axis=1)   # P^3 scaling

def channel_snr(channels, span, n_spans, p_ase):
    # p_ase is the total accumulated link ASE (n_spans amplifiers); nli_power
    # already accumulates over n_spans, so both terms are link totals.
    P = np.array([c.power for c in channels])
    return P / (p_ase + nli_power(channels, span, n_spans))

def optimum_launch_power(eta_self, p_ase):
    """For NLI = eta_self * P^3: argmax P/(p_ase + eta_self*P^3)."""
    return (p_ase / (2.0 * eta_self))**(1.0/3.0)   # P_ASE = 2*P_NLI at optimum
```

The closed-form NLI (`coefficient_matrix`/`interaction_factor`) is the asinh evaluation of the SCI+XCI islands of the GNRF; `nli_power` realizes NLI = P_cut·P_pump²·η summed over pumps (the cubic law); `channel_snr` adds ASE and NLI in variance; `optimum_launch_power` is the cube-root rule from setting dSNR/dP = 0. For the exact (non-rectangular, fully coherent multi-span) case the same η is instead obtained by numerically integrating the GNRF double integral.
