# Context

## Research question

Given only *unpaired* empirical samples of two distributions $X_0\sim\pi_0$ and $X_1\sim\pi_1$ on $\mathbb{R}^d$, find a transport map $T$ — a coupling $(Z_0,Z_1)$ with $Z_0\sim\pi_0$, $Z_1=T(Z_0)\sim\pi_1$ — that turns one distribution into the other. With $\pi_0$ an elementary noise distribution and $\pi_1$ data this is generative modeling; with $\pi_0,\pi_1$ two data domains it is domain transfer / image-to-image translation. The question is how to learn such a transport from unpaired samples via a neural network in a way that is trainable from data and applicable at inference.

## Background

A unifying lens is to represent the transport *implicitly as a continuous-time process* and let a neural network carry the dynamics. Two families dominate.

**Continuous normalizing flows (CNFs).** Model the transport as an ODE $\mathrm{d}Z_t=v(Z_t,t)\,\mathrm{d}t$ with a neural drift $v$. Pushing $\pi_0$ through the ODE induces a density on $Z_1$ whose log-likelihood obeys the instantaneous change-of-variables formula $\frac{\mathrm d}{\mathrm dt}\log\rho_t(Z_t)=-\mathrm{tr}(\partial_z v)$, so the model can be trained by maximum likelihood without restricting the architecture to be analytically invertible. Two facts about ODE flows matter downstream. First, **trajectories of a well-posed ODE cannot cross**: if two solution curves met at a point $(z,t)$ they would have to continue identically (uniqueness), so a single-valued velocity field forces a deterministic, non-crossing flow. Second, the marginal law $\rho_t$ of any process driven by a velocity field $v_t$ satisfies the **continuity (transport) equation** $\partial_t\rho_t+\mathrm{div}(v_t\rho_t)=0$ in the weak sense — equivalently, $\frac{\mathrm d}{\mathrm dt}\mathbb E[h(X_t)]=\mathbb E[\nabla h(X_t)\!\cdot\! v_t(X_t)]$ for test functions $h$ — and under mild regularity this equation has a unique solution for a given initial law. These are classical facts about ODE flows.

**Diffusion / score-based SDEs.** Define a forward noising SDE $\mathrm{d}U_t=b(U_t,t)\,\mathrm{d}t+\sigma_t\,\mathrm{d}W_t$, typically an Ornstein–Uhlenbeck process, that collapses data into an approximate Gaussian; then learn the time-reversal as a generative SDE. Training reduces to a regression (denoising score matching): along a corruption $V_t=\alpha_t X_1+\beta_t\xi$ with $\xi\sim\mathcal N(0,I)$, a network regresses the noise/score, $\min_v\int_0^1 w_t\,\mathbb E\|v(V_t,t)-Y_t\|^2\mathrm{d}t$, with target $Y_t$ a known linear combination of $V_t$ and $\xi$. The schedules $\alpha_t,\beta_t$ follow from the chosen SDE: the OU drift gives an exponential $\alpha_t=\exp(-\tfrac14 a(1-t)^2-\tfrac12 b(1-t))$ and $\beta_t=\sqrt{1-\alpha_t^2}$ (variance-preserving) or $\beta_t=1-\alpha_t^2$ (sub-variance-preserving), or $\alpha_t=1$ with growing $\beta_t$ (variance-exploding).

A diagnostic observation tied the two families together: any such SDE admits a **probability-flow ODE** with the *same marginal laws* at every time, obtained by replacing the reverse-SDE score term with its half-strength deterministic counterpart. So a stochastic, many-step diffusion sampler can be replaced by a deterministic ODE that can be integrated with a numerical solver. This is the basis of deterministic few-step samplers derived from diffusion models.

**Optimal transport (OT).** OT frames the same goal as choosing the coupling that minimizes a transport cost $\mathbb E[c(Z_1-Z_0)]$ for a cost function $c$ (e.g. $c(\cdot)=\|\cdot\|^\alpha$, $\alpha\ge1$). It gives a principled notion of a "nice" coupling and a rich convex-analysis toolkit. In one dimension the unique *monotone* coupling is simultaneously optimal for all non-negative convex costs.

## Baselines

**GANs** (Goodfellow et al. 2014; Arjovsky et al. 2017). A generator pushes noise to data, trained against a discriminator via a minimax objective, achieving one-step sampling with strong image quality.

**VAEs / discrete normalizing flows** (Kingma & Welling 2013; Dinh et al. 2016; Rezende & Mohamed 2015). Likelihood-based one-/few-step models. VAEs use a conditional-Gaussian decoder plus a variational bound; normalizing flows use specially designed invertible layers with tractable Jacobians.

**Neural-ODE maximum likelihood** (Chen et al. 2018, FFJORD). Train an ODE drift $v$ by maximizing the likelihood of $Z_1$ under $\pi_1$ via the change-of-variables formula, with path-length / transport regularizers sometimes added to encourage particular paths.

**Score-based SDEs and their probability-flow ODEs / deterministic implicit samplers** (Song et al. 2019, 2020; Ho et al. 2020). Train by denoising score matching (stable regression, scalable), sample by simulating the reverse SDE or its equivalent probability-flow ODE. The initial distribution is set by the SDE schedule (typically a fixed Gaussian), and the design space includes noise schedule, weighting, and solver.

**Optimal-transport solvers** (Villani; Peyré & Cuturi 2019; Korotin et al. 2021). Compute couplings minimizing a chosen transport cost.

## Evaluation settings

Natural yardsticks that exist independently of any new method. **Image generation:** CIFAR-10 ($32\times32$) for unconditional generation; LSUN and CelebA-HQ at higher resolution; sample quality measured by Fréchet Inception Distance and recall, and crucially as a function of the **number of function evaluations / discretization steps** (1, 2, 5, … steps) so that few-step behavior is visible. **Image-to-image translation and domain transfer:** unpaired domain datasets such as AFHQ (cat/dog/wild, $512\times512$), MetFaces, CelebA-HQ; evaluated by translation quality and trajectory smoothness. **Low-dimensional diagnostics:** 2-D toy distributions (Gaussian mixtures, checkerboards) where full trajectories, path straightness, and transport cost can be plotted directly. Standard model and training stack: time-conditioned U-Net backbones (DDPM++/NCSN++), Adam/AdamW, exponential moving average of weights, and either fixed-step (Euler) or adaptive (RK45) ODE solvers at inference.

## Code framework

The primitives that already exist, with empty slots for the contribution.

```python
import torch

# --- known: an elementary source distribution and a data distribution ---
def sample_pi0(batch_size, shape):
    """Draw from the source pi_0 (e.g. standard Gaussian noise)."""
    return torch.randn(batch_size, *shape)

def sample_pi1(batch_size):
    """Draw from the data distribution pi_1 (a minibatch of data)."""
    raise NotImplementedError

# --- known: a generic time-conditioned vector-field network ---
class VelocityField(torch.nn.Module):
    """v_theta(x, t): R^d x [0,1] -> R^d, a standard time-conditioned U-Net."""
    def forward(self, x, t):
        raise NotImplementedError

# --- open slot: a training objective that produces v_theta from (x0, x1) pairs. ---
def transport_loss(model, x0, x1):
    raise NotImplementedError("transport objective")

# --- known: a black-box ODE/numerical integrator over a learned drift ---
@torch.no_grad()
def integrate(model, z0, N):
    """Integrate dZ_t = model(Z_t, t) dt from t=0 to t=1, returning Z_1."""
    raise NotImplementedError("standard Euler / RK45 solver")

# --- known: a standard supervised training loop ---
def train(model, data_iter, opt):
    for x1 in data_iter:
        x0 = sample_pi0(x1.shape[0], x1.shape[1:])
        loss = transport_loss(model, x0, x1)   # <- the slot above
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```
