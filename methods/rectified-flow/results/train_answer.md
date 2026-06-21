I would describe the method as Rectified Flow, a way to learn a deterministic transport between two distributions that trains like a denoiser but samples like a one-step or few-step map. The starting point is the oldest frustration in generative modeling: diffusion models give stable regression training, yet drawing a sample means integrating a reverse process over many sequential network evaluations, while GANs generate in one call but are adversarial and unstable. Rectified Flow resolves this tension by directly learning an ordinary differential equation whose trajectories are as straight as possible, so a coarse solver, even a single Euler step, lands close to the target.

The key observation is that the slowness of diffusion sampling is not an intrinsic price of deterministic generation but a consequence of the path shape inherited from a stochastic differential equation. A probability-flow ODE derived from a diffusion model has the same marginals as the reverse SDE, yet its trajectories are curved and move at non-uniform speed because the schedules, such as exponential alpha and beta terms, were chosen to make the SDE work, not to make the ODE easy to integrate. A curved, uneven path is exactly what forces a numerical solver to take tiny steps. Rectified Flow instead asks what path a solver would prefer, and the answer is a straight line traveled at constant speed. If a particle moved from a source point to a target point along a straight line, one Euler update would reach the endpoint exactly.

Of course a straight line is not a flow by itself. Given source samples from pi_0 and target samples from pi_1, one can form the interpolation X_t = (1-t)X_0 + tX_1 for t in [0,1]. This line has constant velocity X_1 - X_0, but it is non-causal because the velocity at time t depends on the endpoint X_1, and different pairs of points can have lines that cross, so a single-valued velocity field cannot follow all of them. Rectified Flow turns this multivalued direction field into a proper flow by regressing a neural velocity field v_theta onto the line direction. Concretely, one samples a pair (X_0, X_1), samples t uniformly, forms the interpolant X_t, and trains the network to predict the constant target X_1 - X_0 from the current state and time. The objective is simply the expected squared error between v_theta(X_t, t) and X_1 - X_0. This is plain supervised regression with no discriminator, no likelihood, and no SDE machinery.

At its minimum this regression produces the conditional mean velocity v^X(x,t) = E[X_1 - X_0 | X_t = x], which is single-valued by construction. It averages the directions of all lines passing through the point (x,t), resolving crossings by taking the mean outgoing direction. The resulting ODE dZ_t = v^X(Z_t,t)dt is therefore an honest, non-crossing flow. The crucial fact is that this averaging does not disturb the marginals. Both the interpolation X_t and the flow Z_t solve the same continuity equation with the same velocity field and start from the same initial distribution, so under the usual uniqueness conditions their laws coincide at every time, including t=1. Hence Z_1 is distributed as pi_1, and the flow is a valid transport.

The same linear geometry also gives a useful transport-cost guarantee. Let (Z_0,Z_1) be the coupling produced by integrating the flow from independent pairs (X_0,X_1). For every convex cost c, the expected cost E[c(Z_1 - Z_0)] is no larger than E[c(X_1 - X_0)]. This follows from two applications of Jensen's inequality, one over time and one over the conditional expectation at each point. It is a Pareto improvement over all convex costs simultaneously, not a minimization of any single cost. The improvement can be decomposed exactly: for the quadratic cost the decrease equals the straightness of the flow plus a term measuring how much the underlying lines cross. This identity is the engine behind reflow.

Reflow means recoupling on the flow's own output and fitting a fresh flow on the new pairs. Starting from an arbitrary coupling, define Z^{k+1} as the rectified flow of the coupling (Z_0^k, Z_1^k). Each round preserves marginals and does not increase any convex cost, and the decomposition shows that the sum of straightness and non-crossing gaps across rounds is bounded by the initial quadratic transport cost. Therefore the best round has a gap that shrinks as O(1/K), which means iterated reflow drives the trajectories toward straight lines. A straight flow can be integrated with one or a handful of Euler steps, giving the desired fast inference. In practice one or two reflow rounds are enough before estimation error dominates.

The method also reveals why diffusion probability-flow ODEs look the way they do. They are a special case of the same regression objective, but with a non-straight interpolation X_t = alpha_t X_1 + beta_t noise whose schedules come from an Ornstein-Uhlenbeck process. Rectified Flow removes that inherited curvature by choosing the constant-speed line alpha_t = t, beta_t = 1-t. Once the flow is nearly straight, it can be distilled into a literal one-step map T_hat(x_0) = x_0 + v_theta(x_0,0), trained at t close to zero. This final distillation differs from reflow: reflow builds a new, straighter coupling, while distillation approximates the current coupling as fast as possible.

The canonical name is Rectified Flow. It applies unchanged to generation, when pi_0 is Gaussian noise and pi_1 is data, and to unpaired image-to-image translation, when pi_0 and pi_1 are two domains. The only ingredients are a time-conditioned velocity network, the straight-line regression objective, an ODE solver at inference, and optionally reflow and distillation. I have included below a compact, self-contained Python illustration that demonstrates the core idea on a low-dimensional example: source samples from a Gaussian are pushed toward target samples from a mixture of Gaussians by a small neural velocity field trained with the rectified-flow loss, and the learned flow is integrated with a few Euler steps.

```python
import torch
import torch.nn as nn
import numpy as np

class VelocityField(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))

def sample_pi0(n, dim=2):
    return torch.randn(n, dim)

def sample_pi1(n, dim=2):
    mix = torch.randint(0, 3, (n,))
    centers = torch.tensor([[-1.5, -1.5], [1.5, 0.0], [0.0, 1.5]], dtype=torch.float32)
    out = centers[mix] + 0.25 * torch.randn(n, dim)
    return out

def rectified_flow_loss(model, x0, x1, eps=1e-3):
    b = x1.shape[0]
    t = torch.rand(b, device=x1.device) * (1.0 - eps) + eps
    t_ = t.view(-1, 1)
    xt = t_ * x1 + (1.0 - t_) * x0
    target = x1 - x0
    v = model(xt, t)
    return ((v - target) ** 2).mean()

@torch.no_grad()
def euler_sample(model, z0, n_steps=10):
    x = z0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device)
        x = x + model(x, t) * dt
    return x

def train():
    dim = 2
    model = VelocityField(dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5000):
        x1 = sample_pi1(256, dim)
        x0 = sample_pi0(256, dim)
        loss = rectified_flow_loss(model, x0, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    return model

if __name__ == "__main__":
    torch.manual_seed(0)
    model = train()
    z0 = sample_pi0(500)
    z1_est = euler_sample(model, z0, n_steps=5)
    print("source mean:", z0.mean(dim=0).tolist())
    print("pushed mean:", z1_est.mean(dim=0).tolist())
```
