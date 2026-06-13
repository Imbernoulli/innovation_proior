## Research question

Many engineered systems are specified by a probability of failure that is required to be *extremely small*. A digital communication link is quoted a target bit-error rate (BER) or outage probability of 10⁻⁶, 10⁻⁹, or lower; an ultra-reliable link must deliver a packet correctly with probability 0.99999 or better. The quantity of interest is

    p_f = P(A) = E_p[ 1_A(X) ],

the probability, under the system's own noise/fading distribution p, that the random input X lands in a *failure set* A (a decoding error, a deep fade, an outage). The model is usually too complicated to integrate in closed form — the decoder, the demodulator, the channel are a black box — so p_f must be *estimated by simulation*: draw inputs, run the system, count failures.

The pain point is the rarity itself. When p_f is on the order of 10⁻⁹, a failure essentially never happens in any simulation of practical length, so a naive "draw and count" estimator either reports zero or is dominated by sampling noise. The problem to solve: estimate a probability that is too small to observe directly, to a controlled *relative* accuracy, within a feasible number of simulated trials.

## Background

**The crude Monte Carlo estimator and its variance.** To estimate p_f = E_p[1_A] one draws X₁,…,X_n i.i.d. from p, sets the Bernoulli variable S_i = 1_A(X_i), and averages: p̂ = (1/n) Σ S_i. Each S_i has mean μ = p_f and variance σ² = p_f(1 − p_f). The standard error is σ/√n and the confidence interval has half-width proportional to σ/√n. What matters for a tiny probability is *relative* accuracy: the ratio σ/μ = √((1 − p_f)/p_f) ≈ p_f^(−1/2), which grows without bound as p_f → 0. Pinning the relative error to a fixed level therefore needs a sample size that scales like 1/p_f; this binomial scaling is the wall that simulation of rare events runs into.

**Variance reduction by changing the sampling distribution.** A separate, older idea in Monte Carlo practice is that the same expectation can be written as an average over a *different* sampling distribution, as long as each sample is reweighted to compensate. If q is another density that is positive wherever the integrand is non-zero, then E_p[h(X)] = ∫ h(x)p(x)dx = ∫ h(x)[p(x)/q(x)]q(x)dx = E_q[h(X) p(X)/q(X)]. The multiplicative factor p/q is the *likelihood ratio*. Choosing q so that the integrand has small variance under q reduces the number of samples needed; which q is best, and whether a good one can be reached for the distributions at hand, is left open.

**Light tails and large deviations.** For the noise distributions that arise in communications — Gaussian thermal noise above all — the tails are light: the probability of a large excursion decays exponentially. Some standard machinery exists for such tails. For a distribution with a moment generating function K(θ) = E[e^{θX}], the family e^{θx}p(x)/K(θ) is an exponential reweighting ("tilting" or "twisting") of p, parameterized by θ. For a random walk with negative drift that must climb to a high level b before ruin, the value γ solving E[e^{γΔ}] = 1 (the Lundberg constant) governs the asymptotic decay of the ruin probability (Sigman, IEOR 4703 notes; Asmussen & Glynn). Large-deviations theory characterizes the asymptotic decay rate of such light-tailed rare events.

**The rare-event phenomena in communications.** These motivate why tiny p_f must be estimated at all, and where the failure mass concentrates:
- *Decoding errors at high SNR.* For coded BPSK over an additive-white-Gaussian-noise channel, the word/bit error probability falls off sharply with signal-to-noise ratio; in the operating regime the error event is dominated by the rare noise realizations that push the received vector across a decision boundary near the minimum-distance neighbour. The closed-form uncoded BPSK BER is Q(√(2 Eb/N0)), a convenient yardstick.
- *Error floors of sparse-graph codes.* Iterative (belief-propagation) decoders of low-density parity-check codes show, at high SNR, a flattening of the error curve — an "error floor" — caused by small structured configurations of the graph (trapping / absorbing sets) into which the decoder settles (Richardson, 2003). The residual error probability there is far below what any feasible crude simulation can measure.
- *Deep fades in Rayleigh fading.* When the channel gain is Rayleigh-distributed, the BER tail and the outage probability are dominated by the rare *deep fades* in which the instantaneous gain collapses, momentarily destroying the link.
- *Polarization-mode-dispersion outage in optical links.* The differential group delay of a fibre fluctuates; a rare large excursion produces a system outage whose probability is specified at the 10⁻⁶–10⁻⁹ level.

In every case the failure probability that must be certified is far below the reach of direct counting, and the failure mass is concentrated in an identifiable corner of the input space (a noise direction, a fade depth, a graph configuration).

## Baselines

**Crude (naive) Monte Carlo.** Draw inputs from the true model p, run the system, count failures, divide by n. Unbiased and trivially correct, but its relative standard error is ≈ 1/√(n p_f); reaching, say, 10% relative accuracy demands n on the order of 100/p_f trials. For p_f = 10⁻⁹ that is ~10¹¹ runs of the full decoder — the gap this leaves open is feasibility itself.

**More samples / longer runs.** Simply increasing n shortens the interval but cannot beat the 1/√(n p_f) scaling; the cost to certify a 10⁻⁹ event stays astronomical. The interval length is measured in units of σ, which dwarfs μ — "95% confident he weighs 140 ± 500 pounds." Any workable solution has to *attack σ*, not just grow n.

**Antithetic / control-variate variance reduction.** Generic variance-reduction tricks (negatively correlated sample pairs; subtracting a correlated, known-mean control variate) lower σ² by a constant factor. They help, but the factor is bounded and independent of p_f, so they do not change the 1/p_f scaling that makes rare events hard. They reduce variance generically, without reference to the specific structure of the failure set.

## Evaluation settings

The natural test bed is a link-level simulator with the pieces that already exist:
- **Channel models.** BPSK (and higher-order QAM) over an additive-white-Gaussian-noise channel; the Rayleigh (and Rician) flat-fading channel; block-fading variants. The noise/fade laws are standard and parameterized by signal-to-noise ratio (Eb/N0) or average SNR.
- **Codes / receivers.** Uncoded slicing (threshold at zero for BPSK); linear block and LDPC codes with belief-propagation decoding; the demodulator/decoder treated as a black box that maps a received vector to a decision.
- **Metrics.** Bit-error rate, word/packet-error rate, outage probability — all of the form P(failure set). The figure of merit for an *estimator* of these is its variance (equivalently, the number of trials to reach a target relative accuracy) and whether it stays unbiased.
- **Reference points.** Where a closed form exists (uncoded BPSK BER = Q(√(2 Eb/N0))), it serves as ground truth to check an estimator against; elsewhere the crude-MC estimate at a feasible (higher) p_f anchors the comparison.

## Code framework

The surrounding harness is a BPSK/AWGN simulator plus a generic Monte-Carlo averaging loop. The unresolved piece is the estimator for a tiny failure probability.

```python
import numpy as np
from scipy.special import erfc

def ebn0_to_sigma(ebn0_db):
    """Noise std for unit-energy BPSK at a given Eb/N0 (dB): sigma^2 = N0/2."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return np.sqrt(1.0 / (2.0 * ebn0))

def theoretical_ber(ebn0_db):
    """Closed-form BPSK BER used when the uncoded link has a known answer."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))

# --- crude baseline: draw from the true model, count failures ---------------
def crude_mc_ber(ebn0_db, n, rng):
    sigma = ebn0_to_sigma(ebn0_db)
    y = 1.0 + sigma * rng.standard_normal(n)
    errors = (y < 0.0).astype(float)
    return errors.mean(), errors.var(ddof=1) / n

# --- rare-event estimator slot ---------------------------------------------
def accelerated_ber_estimator(ebn0_db, n, rng, parameter=None):
    """Estimate P(A) when A is too rare to count directly.

    TODO: design an estimator that observes enough failures to estimate P(A)
    without losing unbiasedness or empirical variance reporting.
    """
    pass
```
