I would describe the method as Rectified Flow, a way to learn a deterministic transport between two distributions that trains like a denoiser but samples like a one-step or few-step map. The starting point is the oldest frustration in generative modeling: diffusion models give stable regression training, yet drawing a sample means integrating a reverse process over many sequential network evaluations, while GANs generate in one call but are adversarial and unstable. Rectified Flow resolves this tension by directly learning an ordinary differential equation whose trajectories are as straight as possible, so a coarse solver, even a single Euler step, lands close to the target.

The key observation is that the slowness of diffusion sampling is not an intrinsic price of deterministic generation but a consequence of the path shape inherited from a stochastic differential equation. A probability-flow ODE derived from a diffusion model has the same marginals as the reverse SDE, yet its trajectories are curved and move at non-uniform speed because the schedules, such as exponential alpha and beta terms, were chosen to make the SDE work, not to make the ODE easy to integrate. A curved, uneven path is exactly what forces a numerical solver to take tiny steps. Rectified Flow instead asks what path a solver would prefer, and the answer is a straight line traveled at constant speed. If a particle moved from a source point to a target point along a straight line, one Euler update would reach the endpoint exactly.

Of course a straight line is not a flow by itself. Given source samples from pi_0 and target samples from pi_1, one can form the interpolation X_t = (1-t)X_0 + tX_1 for t in [0,1]. This line has constant velocity X_1 - X_0, but it is non-causal because the velocity at time t depends on the endpoint X_1, and different pairs of points can have lines that cross, so a single-valued velocity field cannot follow all of them. Rectified Flow turns this multivalued direction field into a proper flow by regressing a neural velocity field v_theta onto the line direction. Concretely, one samples a pair (X_0, X_1), samples t uniformly, forms the interpolant X_t, and trains the network to predict the constant target X_1 - X_0 from the current state and time. The objective is simply the expected squared error between v_theta(X_t, t) and X_1 - X_0. This is plain supervised regression with no discriminator, no likelihood, and no SDE machinery.

At its minimum this regression produces the conditional mean velocity v^X(x,t) = E[X_1 - X_0 | X_t = x], which is single-valued by construction. It averages the directions of all lines passing through the point (x,t), resolving crossings by taking the mean outgoing direction. The resulting ODE dZ_t = v^X(Z_t,t)dt is therefore an honest, non-crossing flow. The crucial fact is that this averaging does not disturb the marginals. Both the interpolation X_t and the flow Z_t solve the same continuity equation with the same velocity field and start from the same initial distribution, so under the usual uniqueness conditions their laws coincide at every time, including t=1. Hence Z_1 is distributed as pi_1, and the flow is a valid transport.

The same linear geometry also gives a useful transport-cost guarantee. Let (Z_0,Z_1) be the coupling produced by integrating the flow from independent pairs (X_0,X_1). For every convex cost c, the expected cost E[c(Z_1 - Z_0)] is no larger than E[c(X_1 - X_0)]. This follows from two applications of Jensen's inequality, one over time and one over the conditional expectation at each point. It is a Pareto improvement over all convex costs simultaneously, not a minimization of any single cost. The improvement can be decomposed exactly: for the quadratic cost the decrease equals the straightness of the flow plus a term measuring how much the underlying lines cross. This identity is the engine behind reflow.

Reflow means recoupling on the flow's own output and fitting a fresh flow on the new pairs. Starting from an arbitrary coupling, define Z^{k+1} as the rectified flow of the coupling (Z_0^k, Z_1^k). Each round preserves marginals and does not increase any convex cost, and the decomposition shows that the sum of straightness and non-crossing gaps across rounds is bounded by the initial quadratic transport cost. Therefore the best round has a gap that shrinks as O(1/K), which means iterated reflow drives the trajectories toward straight lines. A straight flow can be integrated with one or a handful of Euler steps, giving the desired fast inference. In practice one or two reflow rounds are enough before estimation error dominates.

The method also reveals why diffusion probability-flow ODEs look the way they do. They are a special case of the same regression objective, but with a non-straight interpolation X_t = alpha_t X_1 + beta_t noise whose schedules come from an Ornstein-Uhlenbeck process. Rectified Flow removes that inherited curvature by choosing the constant-speed line alpha_t = t, beta_t = 1-t. Once the flow is nearly straight, it can be distilled into a literal one-step map T_hat(x_0) = x_0 + v_theta(x_0,0), trained at t close to zero. This final distillation differs from reflow: reflow builds a new, straighter coupling, while distillation approximates the current coupling as fast as possible.

The canonical name is Rectified Flow. It applies unchanged to generation, when pi_0 is Gaussian noise and pi_1 is data, and to unpaired image-to-image translation, when pi_0 and pi_1 are two domains, and the velocity network itself needs nothing method-specific: a standard time-conditioned U-Net (DDPM++/NCSN++), trained with Adam and an exponential moving average of the weights, the same backbone diffusion models already use. The training step is exactly the regression objective derived above: draw a pair, draw t, form the linear interpolant, and regress the network onto the constant target x1 - x0.

```python
import torch

def rectified_flow_loss(model, x0, x1, eps=1e-3):
    """One step of the rectified-flow regression.
    x0 ~ pi_0 (e.g. Gaussian noise), x1 ~ pi_1 (data); for reflow, (x0,x1) are
    (z0, ODE(z0)) pairs produced by the previous flow."""
    b = x1.shape[0]
    t = torch.rand(b, device=x1.device) * (1.0 - eps) + eps      # t ~ Unif(0,1)
    t_ = t.view(-1, *([1] * (x1.dim() - 1)))                      # broadcast over data dims
    x_t   = t_ * x1 + (1.0 - t_) * x0                            # linear interpolation X_t
    target = x1 - x0                                            # constant line velocity
    v = model(x_t, t * 999)                                     # velocity field v_theta(x_t, t)
    return ((v - target) ** 2).mean()                          # || (X1-X0) - v(X_t,t) ||^2
```

Sampling is the black-box integrator run on the learned field: a fixed-step Euler solver, exact in one step if the flow is already straight and a good approximation with a handful of steps otherwise, and an adaptive RK45 solver for an accurate reference or for generating clean reflow targets.

```python
@torch.no_grad()
def euler_sample(model, z0, N=1, eps=1e-3):
    """Integrate dZ_t = v_theta(Z_t, t) dt from Z_0 ~ pi_0 to Z_1."""
    x, dt = z0.clone(), 1.0 / N
    for i in range(N):
        t = torch.ones(z0.shape[0], device=z0.device) * (i / N * (1.0 - eps) + eps)
        x = x + model(x, t * 999) * dt
    return x

@torch.no_grad()
def rk45_sample(model, z0, eps=1e-3):
    """Adaptive black-box ODE solve, for an accurate reference / reflow targets."""
    import numpy as np
    from scipy import integrate
    shape = z0.shape
    def ode_func(t, x_flat):
        x = torch.tensor(x_flat, device=z0.device, dtype=torch.float32).reshape(shape)
        vt = torch.ones(shape[0], device=z0.device) * t
        return model(x, vt * 999).reshape(-1).cpu().numpy()
    sol = integrate.solve_ivp(ode_func, (eps, 1.0), z0.reshape(-1).cpu().numpy(),
                              rtol=1e-5, atol=1e-5, method='RK45')
    return torch.tensor(sol.y[:, -1], dtype=torch.float32).reshape(shape).to(z0.device)
```

Reflow is the outer loop the straightening argument licenses: run the accurate solver from a fresh batch of pi_0 draws to get each one's deterministic image, recouple on those pairs in place of the original (x0, x1), and refit. Because rectification is marginal-preserving and lowers the transport cost, the best of these rounds gets straighter at an O(1/K) rate, though in practice estimation error from refitting on finite samples, not the theorem, limits how many rounds are worth running.

```python
def reflow(model_ctor, train_one_flow, pi0_sampler, pi1_data, K=1, n_pairs=4_000_000):
    """K rounds of recoupling; each round retrains a fresh flow on (z0, ODE(z0)) pairs."""
    model = train_one_flow(rectified_flow_loss, pi0_sampler, pi1_data)   # 1-rectified flow
    for k in range(K):
        z0 = pi0_sampler(n_pairs)
        z1 = rk45_sample(model, z0)                                      # deterministic pairing
        model = train_one_flow(rectified_flow_loss, given_pairs=(z0, z1))  # refit on coupling
    return model
```

And once a flow is nearly straight, it collapses to a literal one-step map by distilling only the t=0 slice of the same objective, T_hat(z0) = z0 + v(z0, 0); on images a perceptual loss such as LPIPS tends to do better than raw L2 on this near-deterministic map.

```python
def distill_one_step_loss(model, z0, z1):
    """One-step map T_hat(z0) = z0 + v(z0, 0); implemented at a small endpoint eps."""
    t = torch.full((z0.shape[0],), 1e-3, device=z0.device)
    v = model(z0, t * 999)
    return ((v - (z1 - z0)) ** 2).mean()        # or LPIPS(z0 + v, z1)
```
