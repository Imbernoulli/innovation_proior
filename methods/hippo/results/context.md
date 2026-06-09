# Context

## Research question

A model that consumes a signal one step at a time must, at every instant, hold a usable summary of *everything it has seen so far*. The history grows without bound, but storage is fixed. So the real problem is: maintain, online and incrementally, a bounded-size representation of the cumulative history of an input — a representation rich enough that downstream predictions can be made from it, and cheap enough to update as each new value arrives.

Two things make this hard. First, "summarize the past" is underspecified until we say *how much each moment of the past matters* and *what a good summary even means* — there is no agreed-upon yardstick for whether one bounded summary is better than another. Second, whatever recurrence maintains the summary must do so without privileging a particular timescale: if the same signal is presented faster or slower, or sampled irregularly, the summary should degrade gracefully rather than break. A solution would need (i) a single principle that explains existing memory mechanisms rather than yet another heuristic, (ii) the ability to address dependencies of any length without a prior on the sequence length or sampling rate, and (iii) provable guarantees — in particular control over how gradients propagate backward through time, and over how fast the approximation error shrinks with the size of the representation.

## Background

**The stateful-recurrence approach and its failure mode.** The classical way to carry information forward is to keep a hidden state that is updated at each step: a recurrent network. In principle a recurrent state has unbounded context. In practice, training it by backpropagation through time runs into vanishing and exploding gradients (Bengio et al. 1994; Pascanu et al. 2013): the sensitivity of a late output to an early input is a long product of Jacobians, whose norm typically decays or blows up exponentially in the time gap. The diagnostic observation that anchors the whole field is exactly this — gradient magnitudes through a plain recurrence decay geometrically, so the effective memory horizon is short regardless of how much capacity the state has. Any principled memory mechanism has to be measured against this: what is the magnitude of ∂(late output)/∂(early input), and does it vanish?

**Orthogonal polynomials and function approximation.** A separate, mature body of theory concerns approximating functions by polynomials. Any (probability) measure μ on the real line equips square-integrable functions with an inner product ⟨f,g⟩_μ = ∫ f g dμ and hence a Hilbert-space geometry. Orthogonalizing the monomials 1, x, x², … under this inner product (Gram–Schmidt) yields a unique sequence of orthogonal polynomials P_0, P_1, … with deg P_n = n and ⟨P_i, P_j⟩_μ = 0 for i ≠ j. Their value is that the best degree-(<N) polynomial approximation of a function f is obtained simply by reading off coefficients c_i = ⟨f, P_i⟩_μ / ‖P_i‖²_μ — no optimization, a closed form. The classical families (Jacobi, of which Legendre and Chebyshev are special cases; Laguerre; Hermite) each correspond to a standard measure, and the Fourier basis can be seen as orthogonal polynomials z^n on the unit circle.

The Legendre polynomials in particular are orthogonal under the *uniform* measure on [−1,1], with normalization (2n+1)/2 ∫_{−1}^1 P_n P_m dx = δ_{nm}, endpoint values P_n(1)=1 and P_n(−1)=(−1)^n, and derivative recurrences
- (2n+1)P_n = P_{n+1}' − P_{n−1}', P_{n+1}' = (n+1)P_n + xP_n',

which combine to give P_n' = (2n−1)P_{n−1} + (2n−5)P_{n−3} + … and (x+1)P_n'(x) = nP_n + (2n−1)P_{n−1} + (2n−3)P_{n−2} + …. As with any orthogonal family, the derivative of a degree-n member is a polynomial of degree n−1. Generalized Laguerre polynomials play the analogous role for the weight x^α e^{−x} on [0,∞), with the recurrence d/dx L_n^{(α)} = −(L_0^{(α)} + … + L_{n−1}^{(α)}).

**Differentiating through moving integrals.** When a quantity is an integral over a domain whose limits move with a parameter t, its derivative picks up boundary terms (the Leibniz rule): ∂_t ∫_{α(t)}^{β(t)} h(x,t) dx = ∫ ∂_t h dx + β'(t) h(β(t),t) − α'(t) h(α(t),t). Equivalently, writing the limits as an indicator 𝟙_{[α,β]} and differentiating distributionally, ∂_t 𝟙_{[α(t),β(t)]} = β'(t)δ_{β(t)} − α'(t)δ_{α(t)} with Dirac deltas.

**ODE discretization.** A linear ODE dc/dt = Ac + Bf is turned into a step recurrence by approximating ∫_t^{t+Δt} of the right-hand side: forward Euler keeps the left endpoint, c(t+Δt) = (I+ΔtA)c + ΔtBf; backward Euler the right endpoint, c(t+Δt) = (I−ΔtA)^{−1}(c+ΔtBf); the bilinear/trapezoid rule averages them, c(t+Δt) = (I−Δt/2 A)^{−1}(I+Δt/2 A)c + Δt(I−Δt/2 A)^{−1}Bf; the generalized bilinear transform interpolates with a parameter α; and zero-order hold integrates exactly under a piecewise-constant input, c(t+Δt) = e^{ΔtA}c + A^{−1}(e^{ΔtA}−I)Bf. The step size Δt is a hyperparameter that all of these share.

## Baselines

**LSTM and GRU (Hochreiter & Schmidhuber 1997; Cho et al. 2014).** Augment the recurrent state with multiplicative gates that interpolate between keeping the old state and writing the new input — a convex update c ← (1−g)c + g·(input), with g a learned function of input and state. This smooths the optimization landscape and empirically lengthens memory, and careful parameterization makes gated RNNs surprisingly strong. But the gating is a heuristic: it is not derived from any statement of what the state is supposed to optimally represent, and it offers no guarantee on gradient decay.

**Time dilation view of gates (Tallec & Ollivier 2018).** Argues gates are fundamentally about letting a recurrence choose its own timescale (a learnable rate of forgetting), and that getting this timescale right is essential. This frames the central role of a timescale/step-size parameter — and the fragility that comes with mis-setting it — but stops at gates.

**Orthogonal / unitary RNNs (Arjovsky et al. 2016).** Constrain the recurrent matrix to be (near-)orthogonal so its repeated application neither shrinks nor amplifies, directly targeting the vanishing/exploding product of Jacobians. Effective on synthetic long-memory tasks but found less robust across realistic tasks (Henaff et al. 2016), and it constrains the dynamics rather than deriving what should be remembered.

**Fourier Recurrent Unit (Zhang et al. 2018).** Each unit maintains a running discrete Fourier coefficient of the input at one (randomly chosen) frequency. Has provably bounded gradients *provided* a timescale hyperparameter is set to roughly 1/T for known horizon T — the bound comes from (1−Δt)^T = Θ(1) when Δt = Θ(1/T). The guarantee is real but contingent on knowing T, and it is specific to the Fourier basis; it is unclear how to get an analogous running-coefficient recurrence for other polynomial bases.

**Legendre Memory Unit (Voelker et al. 2018, 2019).** A recurrent cell whose linear core maintains coefficients of a Legendre expansion of a fixed-length sliding window of the input. It was derived from the opposite direction — modeling spiking neurons as a time-lagged linear time-invariant system and approximating that delay with Padé approximants in the frequency domain; the Legendre-polynomial interpretation was observed afterward, and no fully self-contained proof of the update was available. Empirically strong on long-memory tasks, but it carries a window-length hyperparameter θ (a timescale prior) and is motivated rather than derived as the solution to a stated approximation problem.

**Neural ODEs (Chen et al. 2018).** Parameterize the state's continuous-time dynamics by a general nonlinear network and integrate with a black-box ODE solver. Flexible and natural for irregularly sampled series (Rubanova et al. 2019), and the gaps between timestamps directly supply the integration step. But general nonlinear ODEs are expensive to solve and can train slowly.

## Evaluation settings

The natural yardsticks for a long-range memory mechanism, all predating any new method: pixel-by-pixel sequential MNIST and its harder permuted variant (a fixed permutation destroys local structure, forcing genuine long-range integration over 784 steps); the synthetic copying and adding tasks that probe whether a fixed input can be recalled after a long, controllable delay; and online function-reconstruction on continuous signals (e.g. band-limited white-noise processes), where one measures how well the stored coefficients reconstruct the input over very long horizons and how fast the recurrence runs. To test *timescale robustness* specifically, the natural protocol is to train at one sampling rate and evaluate on dilated/contracted or sub-sampled (missing-data) versions of the same signals. Quality is measured by classification accuracy, by reconstruction mean-squared error, and by wall-clock update cost as a function of state size N.

## Code framework

What already exists: numpy/scipy for linear algebra and special functions (Legendre/Laguerre evaluation, matrix exponentials, triangular solves), and PyTorch for the trainable layer and its recurrence. The piece below is the known scaffold; the one empty slot is the memory operator itself — the rule that maps an incoming input value to an updated bounded-size state, plus a way to read an approximation of the past history back out of that state.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, linalg as la, special as ss


class MemoryOperator(nn.Module):
    """Maintains a length-N summary c of an online scalar signal f.
    forward: sequence of inputs -> sequence of length-N states.
    reconstruct: a stored state -> an approximation of the input history."""
    def __init__(self, N, **kwargs):
        super().__init__()
        self.N = N
        # TODO: define the online update rule that maps each incoming value
        #       to an updated length-N state, and whatever fixed data it needs.

    def forward(self, inputs):                      # inputs: (length, ...)
        # TODO: roll the update over the sequence -> (length, ..., N)
        pass

    def reconstruct(self, c):
        # TODO: c -> approximation of f over the represented history
        pass
```
