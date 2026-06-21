In distributional reinforcement learning the object of interest is the full law of the return $Z^\pi=\sum_{t\ge0}\gamma^tR_t$, not merely its mean $Q^\pi(x,a)=\mathbb{E}[Z^\pi(x,a)]$. The theory that motivates this rests on one attractive fact and one inconvenient obstruction. The fact is that the distributional Bellman operator $\mathcal{T}^\pi Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(x',a')$ is a $\gamma$-contraction in the maximal Wasserstein metric $\bar d_p(Z_1,Z_2)=\sup_{x,a}W_p(Z_1(x,a),Z_2(x,a))$, with fixed point $Z^\pi$. Wasserstein is exactly the right metric here: it measures how far probability mass has to be transported, so it stays finite and well-behaved even when two distributions have disjoint supports — which a Bellman update creates constantly, since scaling by $\gamma$ and shifting by $r$ moves atoms off any fixed grid. The obstruction is that this very metric cannot in general be minimized from samples by stochastic gradient descent. If one forms an empirical distribution $\hat Y_m=\frac1m\sum\delta_{Y_i}$ from samples of a target and minimizes the sample loss $\mathbb{E}[W_p(\hat Y_m,\cdot)]$, the minimizer of that expected sample loss is *not* the minimizer of the true $W_p$; the gradient is biased. So the metric the theory loves is precisely the one a sample-based learner cannot descend.

The existing distributional agent, C51, dodges this rather than resolving it. It pins a comb of fixed support locations $z_1\le\cdots\le z_N$ on a predetermined interval $[V_{\min},V_{\max}]$, makes the *probabilities* on those atoms the learnable quantities, and when the Bellman update shifts the target atoms off the grid it *projects* the target back onto the comb and minimizes a KL divergence after projection. It performs well, but look at what it actually optimizes: KL after a heuristic projection — not Wasserstein. The contraction theorem therefore does not explain its success; it demands $[V_{\min},V_{\max}]$ as domain knowledge plus a fixed resolution; and the projection exists only because fixed atoms force disjoint-support collisions. Parametric Gaussian/Laplace return models fare worse still — a unimodal family with only asymptotic KL guarantees. What we need is an algorithm that is genuinely Wasserstein end-to-end, trainable online from single transitions, with no projection and no support bounds.

I propose QR-DQN, quantile regression DQN. The decisive move is to transpose the categorical parametrization. C51 fixes the locations and learns the probabilities — it learns the *vertical* axis of the distribution, and that is where the biased transport gradient lives. Instead, fix the probabilities to be uniform, $q_i=1/N$, and make the *locations* $\theta_i$ the learnable thing, so the predicted distribution is $$Z_\theta(x,a)=\frac1N\sum_{i=1}^N\delta_{\theta_i(x,a)}.$$ Now I am not learning how much mass sits at fixed heights; I am learning *where* $N$ equal lumps of mass should sit, and "where the $i$-th of $N$ equal-mass lumps sits" is exactly a *quantile* of the return. Three things improve at once. The support is no longer pinned to $[V_{\min},V_{\max}]$ — the locations slide out to wherever the returns actually live, at a resolution that adapts per state. There is no projection: when the Bellman target's atoms move they are just numbers, compared directly to my locations, and disjoint supports are a non-issue because nothing is on a fixed grid. And, crucially, estimating quantiles is something I *can* do from samples without a biased gradient.

Which quantiles? I find them by minimizing $W_1$ between an arbitrary target $Y$ and the uniform-$N$-Dirac distribution supported on ordered locations $\theta_1\le\cdots\le\theta_N$. Writing cumulative levels $\tau_i=i/N$, $\tau_0=0$, the inverse CDF of $Z_\theta$ is a staircase equal to $\theta_i$ on each level-cell $(\tau_{i-1},\tau_i]$, so $$W_1(Y,Z_\theta)=\sum_{i=1}^N\int_{\tau_{i-1}}^{\tau_i}\big|F_Y^{-1}(\omega)-\theta_i\big|\,d\omega.$$ The cells decouple — each $\theta_i$ appears only in its own integral — so I minimize each one separately. The pointwise subgradient of $|F^{-1}(\omega)-\theta|$ in $\theta$ is $+1$ when $\theta>F^{-1}(\omega)$, i.e. $\omega<F(\theta)$, and $-1$ otherwise, giving cell subgradient $$\frac{\partial}{\partial\theta}\int_{\tau_{i-1}}^{\tau_i}\big|F^{-1}(\omega)-\theta\big|\,d\omega=\big(F(\theta)-\tau_{i-1}\big)-\big(\tau_i-F(\theta)\big)=2F(\theta)-(\tau_{i-1}+\tau_i).$$ Setting it to zero gives $F(\theta)=\frac{\tau_{i-1}+\tau_i}{2}$, so the $W_1$-optimal location is the quantile at the cell *midpoint*, not the cell edge: $$\hat\tau_i=\frac{\tau_{i-1}+\tau_i}{2}=\frac{2i-1}{2N},\qquad \theta_i=F_Y^{-1}(\hat\tau_i).$$ This tells me exactly which quantiles my $N$ locations should chase — the midpoint levels $(2i-1)/(2N)$.

The unbiasedness, I should stress, does not come from the quantile parametrization by itself: minimizing sample-$W_p$ is still biased even here. It comes from the *loss* I use to hit each quantile. That is precisely what quantile regression provides. To estimate the $\tau$-quantile of $Z$ from samples, use the asymmetric loss $$\rho_\tau(u)=u\,(\tau-\mathbb{1}_{u<0}),\qquad u=\hat Z-\theta,$$ which charges $\tau\,|u|$ on underestimation ($u>0$) and $(1-\tau)\,|u|$ on overestimation ($u<0$). Its subgradient in $\theta$ is $\mathbb{E}[\mathbb{1}_{\hat Z<\theta}]-\tau=\Pr(\hat Z<\theta)-\tau$, zero exactly when $\theta=F_Z^{-1}(\tau)$. The error-gradient $\tau-\mathbb{1}_{u<0}$ depends *only on the sign* of $u$, so a single sample $\hat Z$ furnishes an unbiased stochastic gradient. That is the whole escape: I cannot descend $W_p$, but I can descend the quantile-regression loss whose minimizers are the very locations that minimize $W_1$. Combining the cell decomposition with quantile regression, the objective $\sum_i\mathbb{E}_{\hat Z}[\rho_{\hat\tau_i}(\hat Z-\theta_i)]$ is minimized at the $W_1$ projection $\theta_i=F_Z^{-1}(\hat\tau_i)$ and has unbiased sample gradients — end-to-end Wasserstein by way of quantile regression on the midpoint quantiles. The tabular policy-evaluation update makes the mechanism vivid: $\theta_i(x)\leftarrow\theta_i(x)+\alpha(\hat\tau_i-\mathbb{1}\{r+\gamma z'<\theta_i(x)\})$ with $z'\sim Z_\theta(x')$, which pushes a location *up* by $\alpha\hat\tau_i$ when it sits below the sampled target and *down* by $\alpha(1-\hat\tau_i)$ when above, so high-$\hat\tau_i$ locations settle high in the distribution.

One wrinkle must be fixed before nonlinear function approximation. $\rho_\tau$ is non-smooth at $u=0$ and its gradient magnitude stays constant ($\tau$ or $1-\tau$) as $u\to0$; with a deep network this kinked, non-shrinking gradient makes the locations jitter. I round the kink with a Huber loss, quadratic near zero and linear in the tails, $$\mathcal{L}_\kappa(u)=\begin{cases}\tfrac12u^2,&|u|\le\kappa\\[2pt]\kappa\big(|u|-\tfrac12\kappa\big),&|u|>\kappa,\end{cases}$$ and define the *quantile Huber loss* by multiplying it by the asymmetric quantile weight using the magnitude $|\tau-\mathbb{1}_{u<0}|$ so the loss stays nonnegative: $$\rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}\{u<0\}\big|\,\mathcal{L}_\kappa(u).$$ At $\kappa=1$ the underlying Huber piece is exactly the gradient-clipped squared error DQN already used, so I am simply swapping its symmetric Huber for an asymmetric one. The hard-loss case $\kappa=0$ is implemented as a separate branch $\rho^0_\tau(u)=|\tau-\mathbb{1}_{u<0}|\,|u|$, because plugging $\kappa=0$ into $\mathcal{L}_\kappa$ would zero out the tail.

For control, the greedy action must respect the unchanged objective — I am enriching the representation, not redefining optimality — so the next action is greedy on the *mean* of the next-state value distribution, which for a uniform-Dirac distribution is just the average of its locations: $a^\star=\arg\max_{a'}\frac1N\sum_j\theta_j(x',a')$, a drop-in for DQN's $\arg\max_a Q$. The concrete agent is the DQN torso with three minimal changes: the output layer becomes size $|\mathcal{A}|\times N$, emitting $N$ quantile locations per action; the loss is the quantile Huber loss rather than the scalar Huber; and the optimizer is Adam rather than RMSProp, since Adam's per-parameter moments suit gradients whose scale differs across quantile levels in an all-pairs asymmetric regression. Per transition, a target network supplies $a^\star$ and the bootstrapped target locations $\mathcal{T}\theta_j=r+\gamma\theta_j(x',a^\star)$ (with $\gamma$ zeroed at terminals, and no projection onto any grid), and every predicted location is regressed at its own level against *all* $N$ target locations: $$\mathcal{L}=\sum_{i=1}^N\mathbb{E}_j\big[\rho^\kappa_{\hat\tau_i}\big(\mathcal{T}\theta_j-\theta_i(x,a)\big)\big]=\frac1N\sum_{i=1}^N\sum_{j=1}^N\rho^\kappa_{\hat\tau_i}\big(\mathcal{T}\theta_j-\theta_i(x,a)\big).$$ The only extra knob over DQN is $N$, the number of quantiles — the resolution of the distribution. On Atari I take $N=200$, $\kappa=1$ (QR-DQN-1) or $\kappa=0$ (QR-DQN-0), Adam with $\alpha=5\times10^{-5}$ and $\epsilon_{\text{Adam}}=0.01/32$.

The guarantee I worried about at the outset survives, though not in the obvious metric. Define the quantile projection $\Pi_{W_1}Z$ that places locations at $F_Z^{-1}(\hat\tau_i)$. Because two projections place mass at the *same* cumulative levels, their optimal coupling pairs $F_{\nu_1}^{-1}(\hat\tau_i)$ with $F_{\nu_2}^{-1}(\hat\tau_i)$, so $d_\infty(\Pi_{W_1}\nu_1,\Pi_{W_1}\nu_2)=\max_i|F_{\nu_1}^{-1}(\hat\tau_i)-F_{\nu_2}^{-1}(\hat\tau_i)|$. Reducing to single Diracs by branch-splitting and using a quantile-ordering contradiction argument, one shows $|\theta_u-\psi_v|\le\max_i|\theta_i-\psi_i|$ at each level, hence $$\bar d_\infty\big(\Pi_{W_1}\mathcal{T}^\pi Z,\Pi_{W_1}\mathcal{T}^\pi Y\big)\le\gamma\,\bar d_\infty(Z,Y).$$ So the combined operator $\Pi_{W_1}\mathcal{T}^\pi$ is a $\gamma$-contraction in $\infty$-Wasserstein, with a unique fixed point $\hat Z^\pi$ that the iteration and its stochastic approximation converge to; and since $\bar d_p\le\bar d_\infty$, convergence in $d_\infty$ gives convergence for every $p$. The contraction is genuinely specific to $p=\infty$: a two-Dirac, $\gamma=1$ counterexample (one state branching to $x_1,x_2$ with probabilities $2/3,1/3$, all rewards zero, with the distributions differing only by moving one atom a unit distance) makes the projected operator *grow* the distance from $2^{-1/p}$ to $1$ for every finite $p$, so $\Pi_{W_1}\mathcal{T}^\pi$ is not even a non-expansion in $\bar d_p$ for $p<\infty$. The convergence I rely on lives precisely in $d_\infty$.

```python
import torch
import torch.nn as nn

N = 200
KAPPA = 1.0

class DistributionalQNetwork(nn.Module):
    def __init__(self, n_actions, n=N):
        super().__init__()
        self.n_actions, self.n = n_actions, n
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n))
        self.register_buffer("levels", (torch.arange(n, dtype=torch.float32) + 0.5) / n)  # midpoints

    def prediction(self, x):
        return self.net(x.float() / 255.0).view(-1, self.n_actions, self.n)   # (B, A, N)

    def greedy_action(self, x):
        return self.prediction(x).mean(dim=2).argmax(dim=1)                   # greedy on the mean

def bootstrap_target(target_net, rewards, next_obs, dones, gamma):
    with torch.no_grad():
        nq = target_net.prediction(next_obs)                                  # (B, A, N)
        a_star = nq.mean(dim=2).argmax(dim=1)
        batch = torch.arange(next_obs.size(0), device=next_obs.device)
        next_theta = nq[batch, a_star]                                        # (B, N)
        not_done = 1.0 - dones.float().view(-1, 1)
        return rewards.float().view(-1, 1) + gamma * not_done * next_theta    # (B, N); no projection

def pairwise_distribution_loss(theta, target, levels, kappa=KAPPA):
    u = target.unsqueeze(1) - theta.unsqueeze(2)                              # u_ij = Ttheta_j - theta_i
    tau = levels.view(1, -1, 1)
    weight = (tau - (u.detach() < 0).float()).abs()
    if kappa == 0:
        rho = weight * u.abs()
    else:
        abs_u = u.abs()
        huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
        rho = weight * huber                                                  # |tau_i - 1{u<0}| L_kappa
    return rho.mean(dim=2).sum(dim=1).mean()                                  # sum_i E_j[rho], batch-mean

def loss_fn(online_net, obs, actions, target, kappa=KAPPA):
    pred = online_net.prediction(obs)
    batch = torch.arange(obs.size(0), device=obs.device)
    theta = pred[batch, actions]                                              # (B, N)
    return pairwise_distribution_loss(theta, target, online_net.levels, kappa)
```
