# Context

## Research question

A model that consumes a signal one step at a time must, at every instant, hold a usable summary of *everything it has seen so far*. The history grows without bound, but storage is fixed. The problem is: maintain, online and incrementally, a bounded-size representation of the cumulative history of an input — rich enough that downstream predictions can be made from it, and cheap enough to update as each new value arrives. The summary depends on *how much each moment of the past matters* and on how the same signal presented faster or slower, or sampled irregularly, is to be handled.

## Background

**The stateful-recurrence approach.** The classical way to carry information forward is to keep a hidden state that is updated at each step: a recurrent network. A recurrent state has unbounded context. Training it by backpropagation through time involves vanishing and exploding gradients (Bengio et al. 1994; Pascanu et al. 2013): the sensitivity of a late output to an early input is a long product of Jacobians, whose norm typically decays or blows up exponentially in the time gap. Gradient magnitudes through a plain recurrence decay geometrically. A relevant quantity here is the magnitude of ∂(late output)/∂(early input).

**Orthogonal polynomials and function approximation.** A separate, mature body of theory concerns approximating functions by polynomials. Any (probability) measure μ on the real line equips square-integrable functions with an inner product ⟨f,g⟩_μ = ∫ f g dμ and hence a Hilbert-space geometry. Orthogonalizing the monomials 1, x, x², … under this inner product (Gram–Schmidt) yields a unique sequence of orthogonal polynomials P_0, P_1, … with deg P_n = n and ⟨P_i, P_j⟩_μ = 0 for i ≠ j. Their value is that the best degree-(<N) polynomial approximation of a function f is obtained simply by reading off coefficients c_i = ⟨f, P_i⟩_μ / ‖P_i‖²_μ — no optimization, a closed form. The classical families (Jacobi, of which Legendre and Chebyshev are special cases; Laguerre; Hermite) each correspond to a standard measure, and the Fourier basis can be seen as orthogonal polynomials z^n on the unit circle.

The Legendre polynomials in particular are orthogonal under the *uniform* measure on [−1,1], with normalization (2n+1)/2 ∫_{−1}^1 P_n P_m dx = δ_{nm}, endpoint values P_n(1)=1 and P_n(−1)=(−1)^n, and derivative recurrences
- (2n+1)P_n = P_{n+1}' − P_{n−1}', P_{n+1}' = (n+1)P_n + xP_n',

which combine to give P_n' = (2n−1)P_{n−1} + (2n−5)P_{n−3} + … and (x+1)P_n'(x) = nP_n + (2n−1)P_{n−1} + (2n−3)P_{n−2} + …. As with any orthogonal family, the derivative of a degree-n member is a polynomial of degree n−1. Generalized Laguerre polynomials play the analogous role for the weight x^α e^{−x} on [0,∞), with the recurrence d/dx L_n^{(α)} = −(L_0^{(α)} + … + L_{n−1}^{(α)}).

**Differentiating through moving integrals.** When a quantity is an integral over a domain whose limits move with a parameter t, its derivative picks up boundary terms (the Leibniz rule): ∂_t ∫_{α(t)}^{β(t)} h(x,t) dx = ∫ ∂_t h dx + β'(t) h(β(t),t) − α'(t) h(α(t),t). Equivalently, writing the limits as an indicator 𝟙_{[α,β]} and differentiating distributionally, ∂_t 𝟙_{[α(t),β(t)]} = β'(t)δ_{β(t)} − α'(t)δ_{α(t)} with Dirac deltas.

**ODE discretization.** A linear ODE dc/dt = Ac + Bf is turned into a step recurrence by approximating ∫_t^{t+Δt} of the right-hand side: forward Euler keeps the left endpoint, c(t+Δt) = (I+ΔtA)c + ΔtBf; backward Euler the right endpoint, c(t+Δt) = (I−ΔtA)^{−1}(c+ΔtBf); the bilinear/trapezoid rule averages them, c(t+Δt) = (I−Δt/2 A)^{−1}(I+Δt/2 A)c + Δt(I−Δt/2 A)^{−1}Bf; the generalized bilinear transform interpolates with a parameter α; and zero-order hold integrates exactly under a piecewise-constant input, c(t+Δt) = e^{ΔtA}c + A^{−1}(e^{ΔtA}−I)Bf. The step size Δt is a hyperparameter that all of these share.

## Baselines

**LSTM and GRU (Hochreiter & Schmidhuber 1997; Cho et al. 2014).** Augment the recurrent state with multiplicative gates that interpolate between keeping the old state and writing the new input — a convex update c ← (1−g)c + g·(input), with g a learned function of input and state. Careful parameterization makes gated RNNs strong on long-memory tasks.

**Time dilation view of gates (Tallec & Ollivier 2018).** Argues gates are fundamentally about letting a recurrence choose its own timescale (a learnable rate of forgetting). This frames the central role of a timescale/step-size parameter.

**Orthogonal / unitary RNNs (Arjovsky et al. 2016).** Constrain the recurrent matrix to be (near-)orthogonal so its repeated application neither shrinks nor amplifies, addressing the product of Jacobians directly. Effective on synthetic long-memory tasks (Henaff et al. 2016).

**Fourier Recurrent Unit (Zhang et al. 2018).** Each unit maintains a running discrete Fourier coefficient of the input at one (randomly chosen) frequency. Has bounded gradients when a timescale hyperparameter is set to roughly 1/T for horizon T — the bound comes from (1−Δt)^T = Θ(1) when Δt = Θ(1/T). Specific to the Fourier basis.

**Legendre Memory Unit (Voelker et al. 2018, 2019).** A recurrent cell whose linear core maintains coefficients of a Legendre expansion of a fixed-length sliding window of the input. It was derived by modeling spiking neurons as a time-lagged linear time-invariant system and approximating that delay with Padé approximants in the frequency domain; the linear core admits a Legendre-polynomial interpretation. Strong on long-memory tasks, with a window-length hyperparameter θ.

**Neural ODEs (Chen et al. 2018).** Parameterize the state's continuous-time dynamics by a general nonlinear network and integrate with a black-box ODE solver. Natural for irregularly sampled series (Rubanova et al. 2019), where the gaps between timestamps directly supply the integration step.

## Evaluation settings

The standard yardsticks for a long-range memory mechanism: pixel-by-pixel sequential MNIST and its permuted variant (a fixed permutation destroys local structure, forcing long-range integration over 784 steps); the synthetic copying and adding tasks that probe whether a fixed input can be recalled after a long, controllable delay; and online function-reconstruction on continuous signals (e.g. band-limited white-noise processes), where one measures how well the stored representation reconstructs the input over long horizons and how fast the recurrence runs. To test timescale robustness, one protocol is to train at one sampling rate and evaluate on dilated/contracted or sub-sampled (missing-data) versions of the same signals. Quality is measured by classification accuracy, by reconstruction mean-squared error, and by wall-clock update cost as a function of state size N.

## Code framework

What already exists: numpy/scipy for linear algebra and special functions (Legendre/Laguerre evaluation, matrix exponentials, triangular solves), and PyTorch for the trainable layer and its recurrence. The scaffold below has one open slot: the memory operator itself — the rule that maps an incoming input value to an updated bounded-size state, plus a way to read an approximation of the past history back out of that state.

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
