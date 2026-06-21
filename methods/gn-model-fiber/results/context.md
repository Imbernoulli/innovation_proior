## Research question

A long-haul wavelength-division-multiplexed (WDM) optical link carries dozens of densely packed channels down hundreds of kilometers of single-mode fiber, with an optical amplifier at the end of every span to make up the loss. Two things degrade the received signal: amplified-spontaneous-emission (ASE) noise from the amplifiers, and signal distortion from the fiber's Kerr nonlinearity. ASE is easy — it is additive white Gaussian noise whose power follows directly from amplifier gain and noise figure. The Kerr nonlinearity is the hard part: it couples every frequency to every other through four-wave mixing, and in a dense WDM comb the number of mixing products is astronomical.

The concrete question is: given a link (fiber type, span length, number of spans, amplifier noise) and a WDM loading (which channels are on, at what power, with what bandwidth and spacing), how does one predict the per-channel signal-to-noise ratio at the receiver fast enough to evaluate thousands of candidate launch-power, modulation-and-coding, and spectrum-packing configurations — rather than running a full split-step simulation of the propagation equation for each one? The split-step solver is the ground truth, and it costs minutes-to-hours per configuration. A network optimizer that searches over power per channel, modulation/coding choice, and guard-band placement works from the transmitted power spectral density (PSD) and the link parameters and needs the nonlinear-interference (NLI) power on each channel combined with ASE into an SNR.

## Background

**Coherent transmission and DSP-compensated dispersion.** Modern long-haul systems are coherent: the receiver mixes the incoming field with a local oscillator and digitizes both quadratures of both polarizations, so the full complex optical field is available in DSP. Chromatic dispersion — the frequency-dependent group delay that smears pulses — is a linear, deterministic, invertible distortion, so it is undone in the digital domain by a static all-pass equalizer. Crucially, this means the dispersion is *not* compensated optically along the fiber: there are no dispersion-compensating modules in the line. The link is "uncompensated." Each channel's pulses therefore spread out enormously as they propagate (thousands of symbols overlapping), and only get re-aligned at the very end in DSP.

**The Kerr effect and the nonlinear Schrödinger / Manakov equation.** Silica's refractive index depends weakly on optical intensity (the Kerr effect), so a propagating field phase-modulates itself by an amount proportional to its own instantaneous power. For a single polarization, propagation of the slowly varying field envelope E(z,t) obeys the nonlinear Schrödinger equation (NLSE):
∂E/∂z = −jβ(f)E − αE − jγ|E|²E (written here mixing time/frequency for the linear and nonlinear parts), where α is the field loss coefficient (power attenuates as e^{−2αz}), β(f) is the propagation constant whose curvature is dispersion (β(f) ≈ 2π²β₂f² + (4/3)π³β₃f³ keeping second- and third-order terms), and γ is the nonlinearity coefficient. For the two-polarization case relevant to polarization-multiplexed (PM) systems, after averaging over the fast random birefringence one uses the Manakov equation, which has the same structure but with the Kerr term scaled by 8/9 and coupling the two polarizations through total intensity.

**Four-wave mixing.** Written in frequency, the Kerr term −jγE·E*·E is a double convolution: three field components at frequencies f₁, f₂, f₃ beat together and deposit a new component at f₁ − f₂ + f₃. This is four-wave mixing (FWM). The efficiency of a given beat depends on phase matching — the dispersion-induced phase walk-off Δβ accumulated over the span — through a factor of the form (1 − e^{−2αL}e^{jΔβL})/(2α − jΔβ), familiar from classical FWM-among-CW-tones calculations. In a dense WDM comb every triple of spectral components contributes such a beat; the new components landing inside a given channel's band are that channel's nonlinear interference.

**ASE noise and OSNR.** Each amplifier adds ASE with one-sided PSD G_ASE = F(G−1)hν per amplifier (F = 2n_sp the noise figure, G the gain), independent white Gaussian noise added at the channel power level. The linear-channel figure of merit is OSNR = P / P_ASE over a reference bandwidth; for an ideal coherent receiver with matched filtering the constellation SNR equals the in-band OSNR, and BER = Ψ(SNR) for a known function Ψ set by the modulation format (e.g. for PM-QPSK, BER = ½erfc(√(SNR/2))).

**What uncompensated propagation does to the waveform.** Because dispersion is left uncompensated, each channel's waveform after a short distance is a sum of thousands of overlapping, dispersed symbols. The Kerr nonlinearity then acts on this heavily dispersed field, and from span to span the large accumulated dispersion scrambles the relative phases of the FWM beats. Split-step studies of such links observe that the in-band distortion is broadband, in contrast to the structured, phase-correlated character seen in dispersion-managed links, where dispersion is periodically zeroed optically and the beats stay coherent. Two further facts are known a priori: at the launch power where systems are operated, the nonlinear distortion is comparable to or smaller than the ASE (the perturbation is mild), and the distortion power grows much faster with launch power than ASE does, so there is a sweet spot in launch power — push too hard and nonlinearity dominates, too soft and ASE dominates.

## Baselines

**Full split-step NLSE/Manakov simulation.** The reference method: discretize the fiber into short steps, alternate a linear (dispersion+loss) operator in frequency with a nonlinear (Kerr) operator in time. Core idea: numerically integrate the exact propagation equation; with enough steps it is essentially exact and captures every nonlinear effect. The actual algorithm is the symmetric split-step Fourier method. Each configuration requires a fresh simulation over many random data realizations to get a stable SNR estimate, and yields a number rather than a closed form.

**Classical FWM-among-CW-tones analysis.** Long-standing closed-form theory for the power of FWM products generated by a small number of continuous-wave tones, built around the same phase-matching efficiency factor (1 − e^{−2αL}e^{jΔβL})/(2α − jΔβ) and, for multiple identical spans, a "phased-array" coherent-addition factor sin²(N_sΦ/2)/sin²(Φ/2). Core idea: enumerate the discrete beats f_i − f_j + f_k, weight each by its phase-matching efficiency, sum the powers. It is built for a handful of discrete unmodulated carriers.

**Perturbation analysis of the nonlinear channel.** First-order regular perturbation ("undepleted-pump" / Volterra-series first term): expand the field as linear solution plus a small correction driven by the Kerr term evaluated on the linear field, E ≈ E_LIN + E_NLI with E_NLI(z,f) = e^{Γ}∫e^{−Γ}Q_NLI(z')dz'. Core idea: when nonlinearity is weak, solve the NLSE to first order by treating the Kerr term as a known source. The perturbation correction for a modulated WDM signal is a triple frequency integral; the closed forms in use cover the special case of identical spans and identical, equally-spaced channels.

## Evaluation settings

The natural yardstick is the split-step NLSE/Manakov simulator as ground truth, with agreement measured as the error in predicted per-channel SNR (equivalently NLI power, or the implied maximum reach / optimum launch power). Representative link/format settings of the time: standard single-mode fiber with dispersion D ≈ 16-17 ps/(nm·km) (β₂ ≈ −21 ps²/km), power attenuation ≈ 0.2 dB/km, nonlinearity γ ≈ 1.3 (W·km)⁻¹, spans of ~80-100 km with lumped EDFA gain compensating span loss and noise figure ~5 dB; PM-QPSK / PM-16QAM / PM-64QAM at symbol rates of tens of GBaud; Nyquist-WDM or quasi-Nyquist channel spacing on a ~50 GHz or flexible grid; tens of channels across the C-band. The metrics are NLI PSD G_NLI(f) per channel, the resulting SNR/GSNR per channel, the optimum per-channel launch power, and the maximum number of spans (reach) at a target SNR.

## Code framework

The pieces that already exist: a way to describe each WDM channel (center frequency, baud rate / bandwidth, launch power, roll-off), the fiber/span parameters, and standard primitives for the linear pieces (ASE power from gain and noise figure; effective length from loss and span length). The missing piece is the map from the transmitted spectrum to nonlinear-interference power.

```python
import numpy as np

# ---- existing primitives ----------------------------------------------------
class Channel:
    def __init__(self, f0, baud_rate, power, roll_off=0.0):
        self.f0 = f0                 # center frequency [Hz]
        self.baud_rate = baud_rate   # symbol rate [Hz] (~ bandwidth for Nyquist)
        self.power = power           # launch power per channel [W]
        self.roll_off = roll_off

class Span:
    def __init__(self, length, alpha, beta2, gamma, beta3=0.0):
        self.length = length         # span length [m]
        self.alpha = alpha           # power attenuation [1/m]; power ~ exp(-alpha*z)
        self.beta2 = beta2           # 2nd-order dispersion [s^2/m]
        self.beta3 = beta3           # 3rd-order dispersion [s^3/m]
        self.gamma = gamma           # nonlinearity [1/(W*m)]

def effective_length(alpha, length):
    # region over which nonlinearity acts
    return (1.0 - np.exp(-alpha * length)) / alpha

def ase_power(gain, noise_figure, nu, bandwidth):
    # amplifier ASE power in a reference bandwidth; F = 10**(NF/10)
    h = 6.62607015e-34
    F = 10 ** (noise_figure / 10.0)
    return F * (gain - 1.0) * h * nu * bandwidth   # dual-pol handled by caller

# ---- nonlinear-noise slot ---------------------------------------------------
def nli_power(channels, span, n_spans):
    """Nonlinear-interference power on every WDM channel,
    from the transmitted spectrum and the link parameters."""
    # TODO
    pass
```
