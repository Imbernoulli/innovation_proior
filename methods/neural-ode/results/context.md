# Context

## Research question

Deep models build expressive maps by stacking many small transformations of a hidden state. The workhorse primitive is the residual block,

  h_{t+1} = h_t + f(h_t, θ_t),  t ∈ {0 … T},

which lets gradients flow through very deep stacks.

A separate but related question comes from density modeling. Likelihood-based models that transform a simple base distribution through an invertible map must, by the change-of-variables formula, evaluate the **log-determinant of the transformation's Jacobian** — a cubic-cost operation in the dimension. The whole design space of such models is shaped by keeping that determinant cheap.

The question is how to parameterize and train continuous-depth hidden-state transformations, and how to compute exact likelihoods in such a framework.

## Background

**Residual networks (He et al. 2016) and their differential-equation reading.** The identity-skip update h_{t+1} = h_t + f(h_t, θ_t) is what makes very deep networks trainable. Several lines of work (Lu et al. 2017, "Beyond Finite Layer Neural Networks"; Haber & Ruthotto 2017, "Stable Architectures for Deep Neural Networks"; Ruthotto & Haber 2018) observed that this update is exactly the forward Euler discretization of a continuous dynamical system: writing the update with a step size Δt gives h_{t+Δt} = h_t + Δt·f(h_t,θ_t), which approximates dh/dt = f(h,t,θ). As the network gets deeper and each step smaller, the chain of hidden states traces the solution of that differential equation. This reframing was used mainly to analyze stability and to motivate reversible architectures; in those works the network itself remained a discrete stack of hand-set depth.

**Numerical solvers for initial value problems.** Given dh/dt = f(h,t,θ) and an initial h(t₀), a solver produces h(t₁) = h(t₀) + ∫_{t₀}^{t₁} f dt. Euler's method is the simplest. Over a century of development (Runge 1895; Kutta 1901; Dormand–Prince; Hairer, Nørsett & Wanner 1987) produced high-order Runge–Kutta methods and implicit multistep methods (the Adams/BDF families in LSODE/VODE). Modern adaptive solvers estimate the local truncation error at each step (e.g. by comparing two embedded methods of different order), shrink or grow the step to keep the error under a user tolerance, and thereby decide on the fly how many times to evaluate f. Implicit methods solve a nonlinear system per step and are more stable, but their internal iterations make direct differentiation through the solver awkward.

**Existence and uniqueness.** Picard's theorem (Coddington & Levinson 1955) guarantees that an initial value problem has a unique solution when f is uniformly Lipschitz in h and continuous in t. Standard networks with finite weights and Lipschitz nonlinearities (tanh, relu) satisfy this.

**Sensitivity analysis: how to differentiate an ODE solution.** The classical literature on differentiating the solution of an initial value problem with respect to its parameters (Pontryagin 1962; and, for continuous-time networks specifically, LeCun 1988; Pearlmutter 1995) offers more than one route. **Forward sensitivity** propagates the derivative of the state with respect to parameters, ∂h/∂θ, alongside the state; its cost is quadratic in the number of variables.

**The memory problem and reversibility.** Reversible residual networks (Gomez et al. 2017, RevNet; Chang et al. 2017; Haber & Ruthotto 2017) attack O(L) memory cost by making each block analytically invertible — partition the hidden units into two groups and use a coupling structure so activations can be recomputed on the backward pass rather than stored, giving constant memory in depth.

**Adaptive computation.** A separate line (Graves 2016, Adaptive Computation Time; Jernite et al. 2016; Figurnov et al. 2017) trains auxiliary networks that decide how many times to apply a recurrent or residual block, so harder inputs get more computation. This adds parameters and overhead at both training and test time.

**Likelihood models and the change-of-variables formula.** A bijective map z₁ = f(z₀) transforms densities by

  log p(z₁) = log p(z₀) − log|det ∂f/∂z₀|.

Computing that log-determinant is O(D³) in general. The literature engineers transformations whose Jacobian is cheap. Coupling-layer flows (NICE, Dinh et al. 2014; RealNVP, Dinh et al. 2016) split the dimensions and transform one half conditioned on the other, making the Jacobian triangular so its determinant is the product of the diagonal (linear cost). Planar flows (Rezende & Mohamed 2015) use a single-hidden-unit perturbation z + u·h(wᵀz+b) whose determinant follows from the matrix-determinant lemma, |1 + uᵀ∂h/∂z|; stacking many such one-unit layers adds capacity.

## Baselines

**Residual network (He et al. 2016).** h_{t+1} = h_t + f(h_t, θ_t), a discrete stack of L blocks with independent parameters per block.

**Reversible residual network (Gomez et al. 2017; Chang et al. 2017).** Makes each block invertible — y₁ = x₁ + F(x₂), y₂ = x₂ + G(y₁) — so activations are recomputed during backprop, achieving O(1) memory in depth.

**Learned adaptive computation (Graves 2016; Figurnov et al. 2017).** Trains a secondary network to choose the number of block applications per input.

**Forward-sensitivity gradients through solvers (e.g. the Stan library, Carpenter et al. 2015; dolfin-adjoint, Farrell et al. 2013).** These differentiate ODE solutions either by forward sensitivity analysis or by backpropagating through the individual operations of the solver.

**Coupling-layer normalizing flows (NICE, Dinh et al. 2014; RealNVP, Dinh et al. 2016).** Exact-likelihood invertible models with triangular Jacobians, so the log-determinant is the sum of log-diagonal terms (linear cost).

**Planar normalizing flow (Rezende & Mohamed 2015).** z(t+1) = z(t) + u·h(wᵀz(t)+b), with log-density change −log|1 + uᵀ∂h/∂z| by the matrix-determinant lemma.

## Evaluation settings

- **Supervised image classification.** MNIST handwritten digits. The natural yardstick against a residual-network baseline of comparable size: test error, parameter count, and the memory cost as a function of depth and the amount of computation (number of function evaluations) used. A small architecture that downsamples twice and then applies a few residual blocks is the standard reference point.
- **Density estimation / generative modeling.** Two-dimensional toy densities (e.g. two-circles, two-moons) and known target distributions, used to compare an invertible flow's ability to match a density and to sample. Metrics: the training objective itself — either KL divergence to a known target density, or expected log-likelihood E_{p(x)}[log q(x)] for maximum-likelihood density estimation — and qualitative visualization of the learned transformation.
- **Irregularly-sampled time series.** Sequences observed at arbitrary, non-uniform time points (a setting where binned RNNs struggle), with a fraction of points subsampled. The natural baseline is a recurrent network, optionally given the time gap to the next observation as an extra input. Metric: predictive root-mean-squared error on held-out time points extending beyond the training interval, i.e. extrapolation quality.

## Code framework

The primitives below already exist: a base layer/module abstraction, an optimizer, an autodiff engine that provides reverse-mode gradients and **vector-Jacobian products** for any differentiable function, a numerical initial-value-problem solver (`integrate`), and a standard training loop. The contribution will be filled into the empty stubs.

```python
import torch
import torch.nn as nn


# A differentiable function of a hidden state, parameterized by a network.
class StateFunction(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, h):
        return self.net(h)


# A numerical solver primitive: given a rule for how a state changes over an
# independent variable, advance an initial state across a requested span.
def integrate(state_change_rule, h0, *spec):
    # TODO: advance h0 according to state_change_rule over the requested span.
    pass


# A transformation that maps an input hidden state to an output, intended to
# replace a stack of residual blocks.
class HiddenStateTransform(nn.Module):
    def __init__(self, state_fn):
        super().__init__()
        self.state_fn = state_fn

    def forward(self, h0):
        # TODO: produce the output hidden state from h0.
        pass


# How to get gradients through the transformation, w.r.t. its input and its
# parameters.
class TransformGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_fn, h0, *params):
        # TODO: run the forward transform.
        pass

    @staticmethod
    def backward(ctx, grad_out):
        # TODO: return gradients w.r.t. h0 and params.
        pass


# A likelihood model that carries a density through the transformation, using
# the change-of-variables formula to track how the log-density changes.
class DensityTransform(nn.Module):
    def __init__(self, state_fn):
        super().__init__()
        self.state_fn = state_fn

    def log_prob(self, x, base_log_prob):
        # TODO: transform x to the base space while accumulating the change in
        #       log-density, then add the base log-probability.
        pass


def train_step(model, batch, optimizer, loss_fn):
    optimizer.zero_grad()
    loss = loss_fn(model, batch)
    loss.backward()
    optimizer.step()
    return loss.item()
```
