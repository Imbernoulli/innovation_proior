OK, let me think this through from scratch. I have a long-haul WDM link: dozens of densely packed channels, hundreds of kilometers of fiber, an amplifier at the end of every span. I want one number per channel — the signal-to-noise ratio at the receiver — and I want it fast, as a function of the launch powers, so that an optimizer can search over power and spacing and modulation without me running a split-step simulation for every candidate. The split-step solver is exact and it is the ground truth, but it costs hours per configuration and gives me a number, not a formula I can differentiate. So I want to replace it with math.

Two things hurt the signal. The amplifiers add spontaneous-emission noise — that part is trivial, it's additive white Gaussian, with a power I can write down from the gain and the noise figure, G_ASE = F(G−1)hν, and over a reference bandwidth it just sits in the denominator of an OSNR. The painful part is the fiber's Kerr nonlinearity. The refractive index of silica depends on intensity, so the field phase-modulates itself by an amount proportional to its own power, and once I have many channels packed together that self-phase-modulation cross-couples everything: every triple of frequency components beats together and deposits energy somewhere else. In a dense comb that's an astronomical number of mixing products. I don't see how to count them by hand.

Let me write down what I actually need, because BER is a red herring — for any coherent QAM system with an ideal receiver, BER = Ψ(SNR), where Ψ is fixed by the modulation format (for PM-QPSK it's ½erfc(√(SNR/2)); other formats, other Ψ). So I only need the SNR on the constellation. And the SNR is the signal power divided by the total noise variance measured on the recovered constellation after DSP. If the only impairment were ASE I'd have SNR = Ā²/σ²_ASE with σ²_ASE = ∫G_ASE|H_Rx|²df. The whole game is: what does the Kerr nonlinearity add to that denominator?

Here's the thing I want to be true. If — and it's a big if — the nonlinear distortion that lands in-band behaves like *additional additive Gaussian noise*, statistically independent of the ASE, then I can just add variances: SNR = Ā²/(σ²_ASE + σ²_NLI), with σ²_NLI = ∫G_NLI(f)|H_Rx|²df. Then the entire problem collapses to finding one object: G_NLI(f), the power spectral density of the nonlinear interference. Everything downstream — SNR, BER, optimum power — falls out of that one function. So let me see whether that "if" can be earned, and if so, compute G_NLI(f).

Why would Kerr distortion ever look like white Gaussian noise? Because of the dispersion — specifically because this link is *uncompensated*. In a modern coherent system I don't undo dispersion optically along the line; I undo it digitally at the receiver, with a static equalizer, because dispersion is linear and invertible. So along the fiber the dispersion runs free. Each channel's pulses smear over thousands of symbol slots within a few tens of kilometers. The transmitted waveform of a channel, after that much dispersion, is a sum of an enormous number of overlapping, randomized contributions — and by the central-limit tendency its statistics drift toward those of a complex Gaussian process. The field "Gaussianizes" after launch. So the nonlinearity is acting on something that, statistically, looks Gaussian and whose only relevant descriptor is its average PSD. That's the lever. It also tells me where this will *fail*: if I periodically compensated dispersion optically (a dispersion-managed link), the field wouldn't Gaussianize the same way, the beats would stay phase-correlated span to span, and the distortion would be structured, not white Gaussian. So whatever I build is for uncompensated links only. Good — let me lean into the Gaussian picture and make it quantitative.

Let me set up the propagation. Single polarization first, I'll fix the dual-pol factor later. In frequency, the NLSE is ∂E/∂z = −jβ(f)E − αE + Q_NLI, where α is the *field* loss (power goes as e^{−2αz}, careful with that factor of two), β(f) is the propagation constant — its curvature is dispersion, and to the orders I care about β(f) = 2π²β₂f² + (4/3)π³β₃f³ — and the Kerr term is Q_NLI(z,f) = −jγ E(z,f) * E*(z,−f) * E(z,f), a double convolution. Two convolutions: three field components at f₁, f₂, f₃ combine and land at f₁ − f₂ + f₃. That's four-wave mixing, and it's exactly the structure I was worried about.

Now I need a signal model that makes this double convolution tractable *and* respects the Gaussianization I just argued for. The convolution over a continuous spectrum is hopeless to average by hand. But if the spectrum were a set of discrete spectral lines, the convolutions become sums over line indices and the deltas do the integrals for me. So let me model the WDM signal as a periodic process — period T₀, very long — which forces its spectrum onto a grid of lines spaced f₀ = 1/T₀. I'll make it a filtered periodic complex white Gaussian process: start from white Gaussian lines, PWGN(f) = √f₀ Σₙ ξₙ δ(f − nf₀) with ξₙ independent unit-variance complex Gaussians, then shape it by √G_Tx(f) so the average line powers follow the actual transmitted PSD. In time, E(t) = Σₙ √(f₀ G_Tx(nf₀)) ξₙ e^{j2πnf₀t}. This is a zero-mean complex Gaussian process (matches Gaussianization), it's periodic (so I get spectral lines), and its average PSD is shaped like a real WDM signal. The periodicity is just a device — I'll let T₀ → ∞, i.e. f₀ → 0, at the very end, and the line spectrum thickens back into a continuum. The average signal power is P_E = ∫G_E df = f₀ Σ G_Tx(nf₀) ≈ ∫G_Tx df = P_Tx, exact as f₀ → 0. Good.

Substitute this into the Kerr term at the fiber input, z = 0. The double convolution with three copies of E gives a triple sum:
Q_NLI(0,f) = −jγ f₀^{3/2} Σₘ Σₙ Σ_k ξₘ ξₙ* ξ_k √(G_Tx(mf₀)G_Tx(nf₀)G_Tx(kf₀)) δ(f − [m−n+k]f₀).
So a triple (m,n,k) of input lines produces a beat at line index i = m − n + k. Collect all triples that hit a given output line: A_i = {(m,n,k): m − n + k = i}. The number of these is finite because G_Tx is bandlimited.

Before I average anything, I should notice that not all of these triples are the same kind of beast. Look at the subset where m = n or k = n. Take m = n: then ξₘξₙ*ξ_k = |ξₘ|² ξ_k, and summing over all m of |ξₘ|² G_Tx(mf₀) just reconstructs the *total signal power* times ξ_k. So those terms don't behave like random beats at all — they collapse to a deterministic constant multiplying the field itself. Physically that's self- and cross-phase modulation acting as a power-dependent phase rotation. Call that subset X_i (the triples with m = n or k = n), and the genuine four-wave part Ã_i = A_i − X_i. The X_i piece contributes a term −j2γP_Tx · E to the equation — a pure phase shift proportional to total power. When I eventually take |·|² to get a power spectral density, a pure phase term washes out completely; it carries no NLI power and just shifts everybody's phase by the same amount. So I can split it off and drop it, and concentrate on Ã_i. (If I *didn't* split it off I'd be double-counting it as noise, which would be wrong — it's deterministic.)

Now, to actually solve for the field I need the perturbative move. The exact NLSE with Q_NLI on the right depends on the unknown E through Q_NLI itself — circular. But I argued earlier that at the operating point the nonlinearity is mild: systems run where the NLI is comparable to or below the ASE. So expand to first order: write E ≈ E_LIN + E_NLI, where E_LIN is the pure linear solution (dispersion + loss, no Kerr) and E_NLI is a small correction driven by the Kerr term *evaluated on the linear field*. This is the undepleted-pump idea borrowed straight from classical FWM: treat the signal as an undepleted pump generating a weak mixing product. Formally, with Γ(z,f) = ∫(−jβ − α)dζ the linear propagator, E_NLI(z,f) = e^{Γ}∫₀^z e^{−Γ(z')}Q_NLI,Ã(z',f)dz'. That's the key object: integrate the four-wave source over the span, propagated linearly. It's an approximation — Q_NLI really does depend on E — but the double convolution scrambles frequencies so thoroughly that the dependence of the source at a given f on the field at that same f washes out, and split-step checks will have to justify that the perturbation stayed small. Anyway, this is the only route to a closed form; the alternative is the split-step solver I'm trying to escape.

Do the span integral. The source carries e^{−2αz'} from the field-power decay and a phase e^{jΔβ z'} from the dispersive walk-off of the three beating components, where the phase mismatch is Δβ = β([m−n+k]f₀) − β(mf₀) + β(nf₀) − β(kf₀). So the integral is ∫₀^L e^{−2αz'}e^{jΔβ z'}dz' = (1 − e^{−2αL}e^{jΔβL})/(2α − jΔβ). That's exactly the FWM efficiency factor from the classical CW theory — reassuring, the discrete-line picture reproduces it. After the integral, E_NLI is again a set of lines, E_NLI(z,f) = Σ_i μ_i δ(f − if₀), so it's still a periodic process and its PSD is just the average line powers: G_E_NLI(f) = Σ_i E{|μ_i|²} δ(f − if₀). So I need E{|μ_i|²}.

This is where the Gaussian assumption gets earned or broken. |μ_i|² is a double sum over pairs of triples (m,n,k) and (m′,n′,k′), each from Ã_i, and inside the expectation sits E{ξₘ ξₙ* ξ_k ξₘ′* ξₙ′ ξ_k′*}. The ξ's are zero-mean, independent, unit-variance complex Gaussians. For such variables, a product of three ξ's and three ξ*'s averages to zero *unless every ξ pairs with a conjugate ξ* of the same index* — otherwise some single, unpaired ξ leaves a factor E{ξ} = 0, or a non-conjugate pair leaves E{ξ²} = 0. So the only surviving terms are the index pairings (m=m′, n=n′, k=k′) or (m=k′, n=n′, k=m′), and for those E{|ξₘ|²|ξₙ|²|ξ_k|²} = 1. Let me classify the triples by how many indices coincide, the way the classical FWM literature does, but tracking the statistics. The triples with all three indices different — the genuine non-degenerate FWM — pair up two ways and there are of order M² of them (M = number of lines in the band). The triples with m = k (degenerate FWM) only pair one way and there are of order M of them. The triples with m = n or n = k are the phase-modulation terms I already split off into X_i. And m = n = k is a single self-phase term that vanishes.

So as I let the comb get finer, M → ∞ (which is exactly f₀ → 0, the continuum limit), the non-degenerate FWM contributions grow as M² while the degenerate ones grow only as M and become negligible, and the phase-modulation terms are gone. What's left is a sum over M²-many *uncorrelated, zero-mean* beat contributions, each individually small. A sum of many independent zero-mean terms is, by the central-limit theorem, Gaussian. There it is — that's why the in-band distortion is Gaussian. Not by fiat: it's that the uncompensated dispersion makes the field Gaussian, the four-wave mixing then produces a huge number of uncorrelated beats, and their sum Gaussianizes. And since the ξ's driving NLI are the same data symbols but combined in a scrambled, decorrelated way, the NLI ends up effectively independent of the linear signal and of the ASE, so I really can add variances. The "if" is earned.

And there's a bonus that makes this computable: the messy double sum over triples collapses, after the averaging, into a double sum over just two free line indices. Writing it out for one span, single polarization,
G_E_NLI(f) = 2γ² f₀³ e^{−2αz} Σ_i δ(f − if₀) Σₘ Σ_k G_Tx(mf₀)G_Tx(kf₀)G_Tx([m+k−i]f₀) · |(1 − e^{−2αz}e^{jΔβz})/(2α − jΔβ)|².
Three factors of G_Tx multiplied together — that's the seed of everything. The pure-phase factor e^{−j2γP_Tx z_eff} from the X_i power term sits outside as a common phase and dies under |·|², confirming it was right to drop it.

Now the dual polarization, because real systems are polarization-multiplexed. Switch to the Manakov equation: same structure, but the nonlinear field source is scaled by the 8/9 Manakov factor after averaging over random birefringence, and the source contains same-polarization and cross-polarization products driven by the total two-polarization intensity. I also split the transmitted PSD across the two polarizations, G_Tx → G_Tx/2 on each, which puts a 2^{-1/2} into each field amplitude. Run the same Gaussian averaging, square the 8/9 field factor, and account for the surviving same- and cross-polarization contractions. The bookkeeping turns the single-pol leading constant 2γ²f₀³ into (16/27)γ²f₀³ for the self-channel term. Cross-channel terms get the same 16/27 Manakov constant for each ordering, and the two symmetric orderings make the XCI weight 2·16/27. I'll carry 16/27 from here on.

Multiple spans. Assume identical spans, each with an amplifier exactly making up the span loss. The NLI generated in each span propagates linearly to the end, so the contributions add — but they add *coherently*, with a phase that depends on which span generated them. Span h's contribution differs from span 1's by a phase e^{jΦ(h−1)} with Φ = Δβ·L_s. Summing h = 0..N_s−1 is a geometric series, (1 − e^{jN_sΦ})/(1 − e^{jΦ}), whose magnitude-squared is the phased-array factor sin²(N_sΦ/2)/sin²(Φ/2) — formally identical to the radiation pattern of a phased antenna array, and again exactly what classical multi-span FWM gives. So the multi-span PSD is the single-span one times that factor.

Now take the continuum limit f₀ → 0, replacing the line sums with integrals over continuous frequencies f₁, f₂ while the conjugated component sits at f₁ + f₂ − f for the channel under test. The phase mismatch needs to be written explicitly. With β(f) = 2π²β₂f² + (4/3)π³β₃f³, the combination β(f₁+f₂−f) − β(f₁) + β(f) − β(f₂) factors cleanly: the f² terms give 2π²β₂·2(f₁−f)(f₂−f), the f³ terms give (4/3)π³β₃·3(f₁−f)(f₂−f)(f₁+f₂), so altogether Δβ = 4π²(f₁−f)(f₂−f)[β₂ + πβ₃(f₁+f₂)]. Putting it together gives the reference formula I was after:

G_NLI(f) = (16/27)γ² ∫∫ G_Tx(f₁)G_Tx(f₂)G_Tx(f₁+f₂−f)
           · |(1 − e^{−2αL_s}e^{jΔβ L_s})/(2α − jΔβ)|²
           · sin²(N_s Δβ L_s/2)/sin²(Δβ L_s/2) df₁ df₂,    with Δβ = 4π²(f₁−f)(f₂−f)[β₂ + πβ₃(f₁+f₂)].

This is the whole model in one line. Given the transmitted spectrum and the link, a double integral gives the NLI PSD on any channel; ASE adds in variance; SNR = P/(P_ASE + σ²_NLI); BER = Ψ(SNR). One split-step replaced by one integral.

Now the payoff that makes this useful for allocating power and spectrum. Stare at the integrand: G_Tx appears *three times*, multiplied. So if I scale every channel's launch power by a factor — write G_Tx = P · g with g a normalized shape — the NLI PSD scales as P³. The NLI power on a channel is therefore P_NLI = η·P³, where η bundles up the whole double integral and depends only on the link and the spectral *shape*, not on the absolute power. That's a clean cubic law, and η is a per-link constant I can precompute once.

So the per-channel SNR is SNR = P/(P_ASE + ηP³). Two competing terms: ASE in the denominator is fixed, NLI grows as the cube of power while the signal grows only linearly. Push power up and the cubic eventually swamps you — an NLI wall. There's an optimum. Differentiate: d/dP [P/(P_ASE + ηP³)] = 0 ⇒ (P_ASE + ηP³) − P·3ηP² = 0 ⇒ P_ASE = 2ηP³. So at the optimum the ASE power is exactly twice the NLI power, P_ASE = 2·P_NLI, and the optimum launch power is P_opt = (P_ASE/2η)^{1/3} — the cube-root rule. Plug it back: at the optimum the denominator is P_ASE + P_ASE/2 = (3/2)P_ASE, so the maximum SNR is P_opt/(1.5·P_ASE). This is the lever for the whole engineering layer: η from the integral gives me each channel's optimum power and the SNR there, and per-channel SNR/GSNR is exactly the quantity that drives power allocation, modulation-and-coding choice, and how tightly I can pack the spectrum.

For multiple spans the phased-array factor matters. When per-span dispersion is large, Δβ L_s sweeps fast across the integration region and the phased-array factor averages, so the NLI just accumulates as the number of spans: P_NLI ∝ N_s (incoherent accumulation). That's also the regime where channels get added and dropped in a network, breaking span-to-span coherence anyway — so for network use I take the incoherent assumption and let NLI scale linearly with N_s.

The double integral is still a double integral, and I want something cheaper still for an optimizer that calls this per channel-pair millions of times. Approximate. Real Nyquist-WDM channels have nearly rectangular PSDs with tiny guard bands, so model each channel as a flat-top rectangle. Then in the f₁–f₂ plane the integrand's weight is concentrated where the three rectangles overlap, and that splits into a small set of "islands": the self-channel-interference (SCI) island where the channel beats with itself, the cross-channel-interference (XCI) islands where it beats with one neighbor, and the multi-channel-interference (MCI) where three distinct channels mix. The weight kernel decays fast away from the channel center, so MCI — three different neighbors — is negligible, and dropping it turns a triple sum into a double sum (cost ∝ N_ch² instead of N_ch³). Over a single rectangular island the 2D FWM-efficiency integral has an analytic value: it integrates to an inverse-hyperbolic-sine. To write the engineering formula the way fiber software usually stores loss, I switch from field loss α to power attenuation a = 2α. Then L_a = 1/a is the power-loss asymptotic length and L_eff = (1−e^{−aL_s})/a is the span effective length. The integral over the island where the channel at f_cut (baud rate R) interacts with a pump channel of baud rate B at offset Δf evaluates to
ψ ≈ {asinh(π² L_a |β₂| R (Δf + B/2)) − asinh(π² L_a |β₂| R (Δf − B/2))}/2 · L_eff²/(2π|β₂|L_a),
and the self term (Δf = 0, B = R) is ψ_SCI ≈ asinh((π²/2)|β₂|L_a R²)/(2π|β₂|L_a) · L_eff². The asinh is just the closed-form value of that rectangular 2D integral, with the L_eff² carrying the span-length dependence; it makes physical sense too — NLI grows only logarithmically with bandwidth/offset once dispersion is strong, because dispersion progressively de-phases distant beats. If I store a local PSD coefficient at the cut-channel center, it is γ²·w·ψ/(R·B²). The channel noise power multiplies that local PSD by the cut bandwidth R, so the per-pair power coefficient is η = R·γ²·w·ψ/(R·B²) = γ²·w·ψ/B². The weight is w = 16/27 for the self (SCI) contribution and w = 2·(16/27) for each cross (XCI) contribution, because the cross beat counts both orderings. Multiply by P_cut·P_pump², sum over pump channels, and I have the NLI on the channel — a handful of asinh evaluations.

Let me write it as code. The data structure I want is a coefficient matrix over channel pairs: compute each asinh interaction once from the span and channel geometry, keep the local-white-noise conversion explicit, then form NLI = P_cut·P_pump²·η over pumps, and the SNR and cube-root optimum power follow directly.

```python
import numpy as np

SPM_WEIGHT = 16.0 / 27.0          # same-polarization / self-channel weight (Manakov 8/9 bookkeeping)
XPM_WEIGHT = 2.0 * 16.0 / 27.0    # cross-channel weight: both beat orderings

class Channel:
    def __init__(self, f0, baud_rate, power, roll_off=0.0):
        self.f0 = f0
        self.baud_rate = baud_rate
        self.power = power
        self.roll_off = roll_off

class Span:
    def __init__(self, length, alpha, beta2, gamma, beta3=0.0):
        self.length = length
        self.alpha = alpha      # power attenuation [1/m]
        self.beta2 = beta2
        self.beta3 = beta3
        self.gamma = gamma

def effective_length(alpha, length):
    return (1.0 - np.exp(-alpha * length)) / alpha

def asymptotic_length(alpha):
    return 1.0 / alpha

def ase_power(gain, noise_figure, nu, bandwidth):
    h = 6.62607015e-34
    F = 10 ** (noise_figure / 10.0)
    return F * (gain - 1.0) * h * nu * bandwidth

def interaction_factor(df, baud_cut, baud_pump, beta2_cut, beta2_pump, L_eff, L_a):
    # closed-form value of the rectangular SCI/XCI FWM-efficiency island,
    # the asinh that the 2D integral collapses to.
    beta2 = (beta2_cut + beta2_pump) / 2.0
    abs_beta2 = abs(beta2)
    right = df + baud_pump / 2.0
    left  = df - baud_pump / 2.0
    val = (np.arcsinh(np.pi**2 * L_a * abs_beta2 * baud_cut * right) -
           np.arcsinh(np.pi**2 * L_a * abs_beta2 * baud_cut * left)) / 2.0
    val *= L_eff**2 / (2.0 * np.pi * abs_beta2 * L_a)
    return val

def coefficient_matrix(channels, span):
    # eta[i, j]: NLI on channel i contributed per (P_i * P_j^2) by pump channel j.
    n = len(channels)
    alpha, beta2, gamma = span.alpha, span.beta2, span.gamma
    L_eff = effective_length(alpha, span.length)
    L_a   = asymptotic_length(alpha)
    eta = np.zeros((n, n))
    for i, cut in enumerate(channels):
        for j, pump in enumerate(channels):
            df = pump.f0 - cut.f0
            weight = SPM_WEIGHT if i == j else XPM_WEIGHT   # SCI vs XCI
            p = interaction_factor(df, cut.baud_rate, pump.baud_rate, beta2, beta2, L_eff, L_a)
            eta_density = gamma**2 * weight * p / (cut.baud_rate * pump.baud_rate**2)
            eta[i, j] = cut.baud_rate * eta_density
    return eta

def nli_power(channels, span, n_spans):
    # cubic-in-power NLI; incoherent (linear) accumulation over identical spans.
    eta = coefficient_matrix(channels, span) * n_spans
    P = np.array([c.power for c in channels])
    Pcut  = np.outer(P, np.ones(len(P)))        # P_i
    Ppump = np.outer(np.ones(len(P)), P)         # P_j
    nli = np.sum(Pcut * Ppump**2 * eta, axis=1)  # NLI on each channel ~ P_i * sum_j P_j^2 * eta
    return nli                                   # the P^3 scaling lives here

def channel_snr(channels, span, n_spans, p_ase):
    p_nli = nli_power(channels, span, n_spans)   # accumulated over n_spans
    P = np.array([c.power for c in channels])
    return P / (p_ase + p_nli)                   # p_ase = total link ASE; ASE and NLI add in variance

def optimum_launch_power(eta_self, p_ase):
    # for a channel with NLI = eta_self * P^3: maximize P/(p_ase + eta_self*P^3)
    # => p_ase = 2*eta_self*P^3 => P_opt = (p_ase/(2*eta_self))**(1/3)
    return (p_ase / (2.0 * eta_self))**(1.0/3.0)
```

The causal chain, start to finish: dispersion is compensated in DSP, so the fiber runs uncompensated and each channel's field Gaussianizes after launch; the Kerr nonlinearity four-wave-mixes the densely packed Gaussian comb into an enormous number of uncorrelated in-band beats, whose sum is, by the central-limit theorem, additive Gaussian noise statistically independent of the ASE; a first-order perturbation of the NLSE/Manakov equation plus an averaging over the data symbols turns that noise into a closed-form double integral of the transmitted PSD — the GN reference formula — in which the spectrum enters cubed, so NLI = η·P³ with η a per-link constant; that cubic law gives SNR = P/(P_ASE + ηP³), an NLI wall, and a cube-root optimum launch power with ASE = 2·NLI at the peak; and approximating Nyquist channels as rectangles collapses the integral to a sum of asinh terms, giving a per-channel SNR cheap enough to drive power, modulation, and spectrum allocation.
