We have a robot with a known but strongly nonlinear model and a cost it must minimize over a short horizon while acting in real time — re-deciding the controls every twenty milliseconds, executing one step, then re-optimizing. The system is control-affine with noise entering through the actuators, $dx = (f(x,t) + G(x,t)u)\,dt + B(x,t)\,dw$, and the objective over a finite horizon is $\min_u \mathbb{E}[\phi(x_T) + \int_t^T (q(x,t) + \tfrac12 u^\top R u)\,dt]$ — an arbitrary state cost $q$ plus a quadratic control cost. The difficulty is precisely the costs and dynamics I want to write down. On a quadrotor I want to charge an impulse the instant the body touches an obstacle — a discontinuous indicator, not a smooth barrier; on a race car I want a nonlinear tire model and a cost that rewards sliding through a corner. Solving the problem exactly means the stochastic Hamilton-Jacobi-Bellman PDE for the value function $V(x,t)$, which is hopeless above a handful of state dimensions because discretizing $V$ over the state space is the curse of dimensionality. The fast trajectory optimizers everyone actually runs, the DDP/iLQG family, Taylor-expand the dynamics to first or second order along a nominal trajectory and approximate the cost as a quadratic, then do a backward Riccati sweep to read off a local feedback law — elegant and fast near a good nominal, but built entirely on derivatives. A discontinuous crash cost has no derivative and no honest quadratic model; the best DDP can do is swap in a smooth surrogate, a sum of exponential barriers around every nearby obstacle, which distorts the objective and scales with the obstacle count. I want something derivative-free, that swallows arbitrary nonlinear dynamics and discontinuous costs whole, and that is principled rather than a made-up score.

I propose MPPI, Model Predictive Path Integral control. The starting point is to take the HJB seriously rather than discard it. For this control-affine system with quadratic control cost the optimal control is $u^* = -R^{-1}G^\top V_x$, and the HJB carries a term $V_x^\top G R^{-1} G^\top V_x$ that is quadratic in $V$ and therefore blocks any linear solution. But that gradient-squared shape is exactly what a logarithmic, Cole-Hopf-type change of variables is built to remove. Write $V = -\lambda\log\psi$ for a positive constant $\lambda$; then $V_x = -\lambda\,\psi_x/\psi$, and the nonlinear HJB term and the diffusion term $\tfrac12\operatorname{tr}(BB^\top V_{xx})$ both produce quadratics in $\psi_x/\psi$. They cancel precisely when I tie the noise to the control cost through a single scalar,
$$B_c(x,t)\,B_c(x,t)^\top = \lambda\,G_c(x,t)\,R(x,t)^{-1}\,G_c(x,t)^\top,$$
where the subscript $c$ denotes the actuated block — noise acts only where the controls act. This is a genuine assumption with a price (it forces the noise covariance to follow $R^{-1}$ and confines noise to actuated channels), but it is the one that pays: with it the HJB for $V$ collapses to a *linear* backward PDE for $\psi$, the Chapman-Kolmogorov equation. A linear backward PDE with terminal condition $\psi(x_T,T)=\exp(-\phi(x_T)/\lambda)$ is exactly what the Feynman-Kac lemma evaluates as an expectation over forward diffusion trajectories, without ever discretizing the state:
$$\psi(x_0,t_0) = \mathbb{E}_P\big[\exp(-\tfrac1\lambda S(\tau))\big], \qquad S(\tau) = \phi(x_T) + \int q\,dt,$$
where $P$ is the *uncontrolled* dynamics, the system with $u=0$. The value function has become a path integral — the free-energy / partition-function object of statistical mechanics with $\lambda$ as temperature — and differentiating with respect to the initial state turns it into a ratio of trajectory expectations for the control itself, $u^*\,dt = M\cdot \mathbb{E}_P[\exp(-S/\lambda)\,B\,dw]/\mathbb{E}_P[\exp(-S/\lambda)]$ with $M = R^{-1}G_c^\top(G_c R^{-1}G_c^\top)^{-1}$, reducing to $G_c^{-1}$ when the actuated block is square. There is no backward sweep: the optimal control is an expectation I approximate by forward simulation.

The trouble is that this expectation is under $P$, the uncontrolled dynamics. Sampling from $P$ is switching the machine on and waiting for the natural noise to do something useful; for well-engineered hardware that noise is tiny, so almost every sampled trajectory is high-cost, every weight $\exp(-S/\lambda)$ is numerically zero, and the Monte-Carlo estimate is dominated by a freak lucky sample or is the indeterminate $0/0$, with enormous variance. The textbook cure is importance sampling: draw from a better distribution $q$ and reweight by the likelihood ratio $p/q$. Prior path-integral work — the PI$^2$ reward-weighted averaging of control variations — reshaped only the *mean* of $q$ via Girsanov's theorem, shifting the controls one samples around so the rollouts cluster near a promising trajectory. That clears the derivative-free and principled requirements, but it leaves a stubborn failure. On a cart-pole swing-up, mean-shifting draws every rollout with the natural system variance, which is small, so the samples are tiny wiggles bunched tightly around the current trajectory; none dares the big committed swing that gets the pole over the top, the weighted average of mediocre near-identical trajectories is another mediocre trajectory, and the controller never swings up. What is locked is the *variance* of the sampling distribution. So the defining contribution of MPPI is a generalized importance-sampling scheme that changes the variance too — and keeps the reweighting exact.

The load-bearing step is computing $p(\tau)/q(\tau)$ when $q$ has both a shifted mean and a scaled covariance. Discretize a step as $x_{t+1} = x_t + (f + Gu)\Delta t + B\epsilon\sqrt{\Delta t}$; by the Markov property each trajectory density is a product of Gaussians $Z^{-1}\exp(-\tfrac{\Delta t}{2}\sum_i (z_i - m_i)^\top C_i^{-1}(z_i - m_i))$, with $z_i$ the realized rate after subtracting passive drift, $\mu_i = G_c u_i$ the controlled rate shift. Let $q$ scale the actuated noise block by $A_{t_i}$, so its covariance in the Gaussian quadratic is $\Lambda_i = A_{t_i}^\top \Sigma_i A_{t_i}$ against $p$'s $\Sigma_i = B_c B_c^\top$. Dividing the two densities, the determinant prefactors give $\prod_i |A_{t_i}|$ and the exponent is a difference of two quadratics in $z_i$, which is itself a single quadratic. Define the inverse-covariance difference $\Gamma_i^{-1} = \Sigma_i^{-1} - \Lambda_i^{-1}$ and complete the square. Expanding $(z_i-\mu_i)^\top\Lambda_i^{-1}(z_i-\mu_i)$ gives $\zeta_i = z_i^\top\Gamma_i^{-1}z_i + 2\mu_i^\top\Lambda_i^{-1}z_i - \mu_i^\top\Lambda_i^{-1}\mu_i$; rewriting in the deviation from the controlled mean $\tilde z_i = z_i - \mu_i$ and then splitting every $\Gamma_i^{-1}$ back into $\Sigma_i^{-1} - \Lambda_i^{-1}$, the $\Lambda_i^{-1}$ linear and constant terms cancel exactly. What survives is
$$Q_i = (z_i - \mu_i)^\top \Gamma_i^{-1} (z_i - \mu_i) + 2\mu_i^\top \Sigma_i^{-1}(z_i - \mu_i) + \mu_i^\top \Sigma_i^{-1}\mu_i,$$
so $p/q = (\prod_i|A_{t_i}|)\exp(-\tfrac{\Delta t}{2}\sum_i Q_i)$. The last two $Q_i$ terms are precisely Girsanov's mean-change terms; the *first* term is entirely new and carries the variance change, since $\Gamma_i^{-1} = \Sigma_i^{-1} - \Lambda_i^{-1}$ vanishes exactly when $A_t = I$ — it is a penalty on exploring with larger variance than the natural noise. A pure mean-change is the special case $A_t = I$ of this; now I can change the variance with the right likelihood ratio.

This folds cleanly into the algorithm. The state-independent determinant $\prod_i|A_{t_i}|$ cancels between numerator and denominator, and the remaining $Q_i$ is a function of the trajectory, so it joins the running cost. Specializing to the case I run — $G_c$ square so $M = G_c^{-1}$, scalar covariance scaling $A = \sqrt{\nu}\,I$ giving one exploration knob $\nu$, and writing the noise as a random control perturbation $\delta u$ so $B\epsilon/\sqrt{\Delta t} = G\,\delta u$ — the $G_c^{-1}$ and $G$ cancel and the update collapses to a reward-weighted average of random control variations,
$$u_{t_i}^* \approx u_{t_i} + \frac{\sum_k \exp(-\tilde S(\tau_{i,k})/\lambda)\,\delta u_{i,k}}{\sum_k \exp(-\tilde S(\tau_{i,k})/\lambda)},$$
with the augmented running cost reducing to $\tilde q = q + \tfrac{1-\nu^{-1}}{2}\delta u^\top R\,\delta u + u^\top R\,\delta u + \tfrac12 u^\top R u$ — the importance ratio simply re-injects the original quadratic control cost the path-integral expectation had absorbed. The free-energy view confirms the same weight independently: with $F = \log\mathbb{E}_P[\exp(-S/\lambda)]$, Jensen gives $-\lambda F \le \mathbb{E}_Q[S] + \lambda\,D_{KL}(Q\Vert P)$, tight at the Gibbs distribution $q^*(V) \propto \exp(-S(V)/\lambda)\,p(V)$, so importance-sampling the optimal trajectory distribution from the current Gaussian yields weight $w(V)\propto\exp(-S(V)/\lambda)$. Here $\lambda$ is the temperature: $\lambda\to0$ puts all mass on the single best trajectory (greedy), $\lambda\to\infty$ is a uniform average that ignores cost. The free-energy view also exposes a numerical landmine — $\exp(-S/\lambda)$ over raw costs underflows or overflows — fixed by shifting costs by the minimum sampled cost $\rho$, which multiplies weights by $\exp(\rho/\lambda)/\exp(\rho/\lambda)=1$ so the best sample gets weight $\exp(0)=1$ and every other exponent is non-positive: $w\propto\exp(-(S-\min S)/\lambda)$.

Carried into the setting I actually plan in — a fixed pre-trained latent dynamics model that rolls a batch of action sequences forward and returns a scalar cost per sequence, with no analytic $B$ or $R$ — the update survives the abstraction completely, because it only ever uses rollout costs and action perturbations. I maintain a per-timestep Gaussian over the action sequence, a mean and a standard deviation per timestep, and each iteration sample a population, roll them all out, score them, and refit. The mean refit is exactly the reward-weighted average, $\text{mean} = \sum_k w_k\,\text{action}_k$ with $w_k = \exp((\min\text{cost}-\text{cost}_k)/\lambda)$ normalized; the Gaussian mean plays the role of the nominal control $u$ and the samples play $u + \delta u$. The piece mean-only methods could not do lands naturally: I also refit the *standard deviation* as the weighted spread of the good samples, $\text{std} = \sqrt{\sum_k w_k(\text{action}_k - \text{mean})^2}$. When the good samples cluster the std shrinks and I exploit; when they spread it stays wide and I keep exploring — the exploration variance is adapted online from the rollouts rather than frozen at a fixed natural noise, exactly the cart-pole failure I diagnosed. I initialize the std deliberately large ($\text{max\_std}\approx 2$) so the first iteration escapes local basins. One practical truncation: rather than soft-weight all $N$ samples I first keep the top-$k$ lowest-cost elites and apply the exponential weight *within* that set, because far-out high-cost samples still carry tiny nonzero weight that contaminates the fit — especially the outlier-sensitive variance estimate — while the soft weighting preserves the graded "how much better" information among the survivors. This is the precise difference from the cross-entropy method, which looks superficially identical (sample a Gaussian, score, refit) but refits to the *unweighted* mean and std of the elites, every elite counted equally and all magnitude information discarded at the hard threshold; MPPI keeps the same elite truncation but applies the soft Gibbs weight $\exp((\min\text{cost}-\text{cost})/\lambda)$, so a markedly better elite pulls the mean and tightens the variance more than a marginal one. In code the inverse temperature is stored as a cost-gap multiplier, $\text{score} = \exp(\text{temperature}\cdot(\min\text{cost}-\text{cost}))$ so $\text{temperature}=1/\lambda$; a small temperature ($0.005$ here) is a large $\lambda$, a soft broad weighting that averages over the elites rather than collapsing onto one — the robust choice when the model rollouts are themselves noisy. The whole thing sits in the standard receding-horizon wrapper: each control step run these sampling iterations, sample one final elite according to its path-integral weight, execute it, re-observe and re-plan.

```python
import numpy as np
import torch
from einops import rearrange


class MPPIPlanner(Planner):
    """Model Predictive Path Integral planner. Maintains a per-timestep Gaussian
    over action sequences; each iteration samples a population, scores it by the
    model rollout cost, and refits the Gaussian to the cost-exponentiated (Gibbs)
    weighted elite samples -- refining BOTH mean and std."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=500, n_iters=15,
                 max_std=2.0, num_elites=64, temperature=0.005, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.max_std = max_std                 # large initial exploration width
        self.num_elites = num_elites           # top-k kept before soft weighting
        self.temperature = temperature         # = 1 / lambda (small => soft, broad weighting)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # action-sequence Gaussian: per-timestep mean and std
        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)

        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            # sample a population of action sequences from the current Gaussian
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)            # T B A

            # score every sequence by rolling it through the model (lower is better)
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init).unsqueeze(1)        # B 1
            losses.append(cost.min().item())

            # keep the top-k elites, then soft-weight within that set
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Gibbs / path-integral weights over the cost gap from the best sample
            # (cost-shift = numerical conditioning; it multiplies the weights by 1)
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))     # B'
            score /= score.sum(0)

            # refit mean (reward-weighted average) and std (weighted spread = variance change)
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9)                                                  # T A
            std = torch.sqrt(torch.sum(
                score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                dim=1) / (score.sum(0) + 1e-9))

        # commit a plan: draw one final elite according to its path-integral weight
        score_np = score.cpu().numpy()
        selected = elite_actions[:, np.random.choice(np.arange(score_np.shape[0]), p=score_np)]

        return PlanningResult(
            actions=selected,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
