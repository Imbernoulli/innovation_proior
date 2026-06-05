I keep circling back to one uncomfortable fact about the distributional setup. The distributional Bellman operator $\mathcal{T}^\pi Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(x',a')$ is a $\gamma$-contraction in the maximal Wasserstein metric $\bar d_p$ — that's the clean theory, and Wasserstein is the *right* metric because it measures how far mass has to move, so it's finite and sensible even when two distributions have disjoint supports, which a Bellman update creates constantly (scale by $\gamma$, shift by $r$, and your atoms land off any fixed grid). So the metric I'd want to descend toward the fixed point is Wasserstein. But there's a theorem that says I *cannot* minimize Wasserstein from samples by SGD: if I form an empirical distribution $\hat Y_m=\frac1m\sum\delta_{Y_i}$ from samples of a target and minimize the sample loss $\mathbb{E}[W_p(\hat Y_m,\cdot)]$, the minimizer of that expected sample loss is not the minimizer of the true $W_p$. The gradient is biased. So the metric the theory loves is the metric the learner can't follow.

The fixed-atom categorical agent dodges this. It puts a comb of fixed locations $z_1\le\cdots\le z_N$ on a predetermined interval $[V_{\min},V_{\max}]$, makes the *probabilities* $q_i$ on those atoms the learnable thing, and when the Bellman update shifts the atoms off the grid it *projects* the target back onto the comb and minimizes a KL. It works, but stare at what it actually optimizes: KL after a heuristic projection — not Wasserstein. So the contraction theorem doesn't explain its success, it needs me to hand it $[V_{\min},V_{\max}]$ as prior knowledge, and the projection step exists only because fixed atoms force disjoint-support collisions. I want an algorithm that is genuinely Wasserstein end-to-end, trainable online from single transitions, with no projection and no support bounds. Let me try to find where the bias actually comes from, because that's where the fix has to live.

Why is the sample-Wasserstein gradient biased? Because $W_p$ is built from the *quantile function* $F^{-1}$, and a single sample is a draw from the distribution, not an observation of a quantile. When I push the locations to match an empirical sample, the optimal transport plan reshuffles which sample pairs with which prediction atom, and the gradient of that matching, averaged over sample sets, doesn't equal the gradient of the population transport. The trouble is that the categorical agent's free variables are the *probabilities* on fixed locations — it's trying to learn the *vertical* axis of the distribution. What if I turn the parametrization on its side?

So turn it on its side: fix the *probabilities* to be uniform, $q_i=1/N$, and make the *locations* $\theta_i$ the learnable thing. So the predicted distribution is
$$Z_\theta(x,a)=\frac1N\sum_{i=1}^N\delta_{\theta_i(x,a)}.$$
Now I'm not learning how much mass sits at fixed heights; I'm learning *where* $N$ equal lumps of mass should sit. And "where the $i$-th of $N$ equal lumps of mass sits" is exactly a *quantile* of the distribution. So this transposed parametrization is estimating quantiles of the return. Three things immediately look better. The support isn't pinned to $[V_{\min},V_{\max}]$ — the locations can slide out to wherever the returns actually live, with resolution that adapts per state. There's no projection: when the Bellman target's atoms move, they're just numbers, and I compare them to my locations directly; disjoint supports are a non-issue because nothing is on a fixed grid. And — the part I have to verify — estimating quantiles is something I might be able to do from samples *without* a biased gradient.

Let me pin down which quantiles to estimate, by actually minimizing $W_1$ between an arbitrary target $Y$ and a uniform-$N$-Dirac distribution $U$ supported on the ordered locations $\theta_1\le\cdots\le\theta_N$. Write the cumulative levels $\tau_i=i/N$, $\tau_0=0$. The CDF of $U$ is a staircase: on the level-interval $(\tau_{i-1},\tau_i]$ its inverse-CDF equals $\theta_i$. Since $W_1=\int_0^1|F_Y^{-1}(\omega)-F_U^{-1}(\omega)|\,d\omega$, and $F_U^{-1}$ is piecewise-constant equal to $\theta_i$ on each level-cell,
$$W_1(Y,U)=\sum_{i=1}^N\int_{\tau_{i-1}}^{\tau_i}\big|F_Y^{-1}(\omega)-\theta_i\big|\,d\omega.$$
The cells decouple — each $\theta_i$ only appears in its own integral — so I minimize each $\int_{\tau}^{\tau'}|F^{-1}(\omega)-\theta|\,d\omega$ separately. Take the subgradient in $\theta$. Pointwise, $\frac{\partial}{\partial\theta}|F^{-1}(\omega)-\theta|$ is $+1$ when $\theta>F^{-1}(\omega)$ and $-1$ when $\theta<F^{-1}(\omega)$. The condition $\theta>F^{-1}(\omega)$ is $\omega<F(\theta)$ (the quantile function is increasing). So
$$\frac{\partial}{\partial\theta}\int_\tau^{\tau'}|F^{-1}(\omega)-\theta|\,d\omega
=\int_\tau^{F(\theta)}(+1)\,d\omega+\int_{F(\theta)}^{\tau'}(-1)\,d\omega
=\big(F(\theta)-\tau\big)-\big(\tau'-F(\theta)\big)=2F(\theta)-(\tau+\tau').$$
Set it to zero: $F(\theta)=\frac{\tau+\tau'}{2}$, i.e. the minimizer is the quantile at the *midpoint* of the cell, $\theta=F^{-1}\!\big(\frac{\tau+\tau'}{2}\big)$. So the $W_1$-optimal locations are the *midpoint quantiles*
$$\hat\tau_i=\frac{\tau_{i-1}+\tau_i}{2}=\frac{2i-1}{2N},\qquad \theta_i=F_Y^{-1}(\hat\tau_i).$$
That's clean and it tells me precisely which quantiles my $N$ locations should chase: not the cell edges $i/N$, but the cell centers $(2i-1)/(2N)$. (If $F^{-1}$ jumps at the midpoint there's a flat set of minimizers, but the midpoint quantile is always one of them.)

Now, can I hit those midpoint quantiles from samples without bias? First a caution: just having a quantile parametrization does *not* by itself unbias the Wasserstein gradient — I can construct a $Z$ (take $Z=\frac1N\sum\delta_i$ and look at the leftmost location: with nonzero probability the empirical sample has no atom at the true value, and the transport pairing then pulls the gradient the wrong way) where $\arg\min\mathbb{E}[W_p(\hat Z_m,Z_\theta)]\ne\arg\min W_p(Z,Z_\theta)$. So minimizing sample-$W_p$ directly is still biased even here. The unbiasedness has to come from the *loss I use to hit each quantile*, not from $W_p$ itself.

This is exactly the problem quantile regression solves. To estimate the $\tau$-quantile of a distribution $Z$ from samples, use the asymmetric loss
$$\rho_\tau(u)=u\,(\tau-\mathbb{1}_{u<0}),\qquad u=\hat Z-\theta,$$
which charges $\tau\,|u|$ when I underestimate ($\hat Z>\theta$, $u>0$) and $(1-\tau)\,|u|$ when I overestimate ($u<0$, since $u(\tau-1)=|u|(1-\tau)$). Minimize $\mathbb{E}_{\hat Z\sim Z}[\rho_\tau(\hat Z-\theta)]$: the subgradient in $\theta$ is $\mathbb{E}[-(\tau-\mathbb{1}_{\hat Z<\theta})]=\mathbb{E}[\mathbb{1}_{\hat Z<\theta}]-\tau=\Pr(\hat Z<\theta)-\tau$, which is zero exactly when $\Pr(\hat Z<\theta)=\tau$, i.e. $\theta=F_Z^{-1}(\tau)$. So quantile regression's minimizer is the quantile I want, and crucially its gradient depends only on the *sign* of $u=\hat Z-\theta$ — $\tau-\mathbb{1}_{u<0}$ — so a single sample $\hat Z$ gives an *unbiased* stochastic gradient. That's the whole escape: I can't descend $W_p$, but I can descend the quantile-regression loss whose minimizers are the very locations that minimize $W_1$. Combining the cell decomposition with quantile regression, the location objective
$$\sum_{i=1}^N\mathbb{E}_{\hat Z\sim Z}\big[\rho_{\hat\tau_i}(\hat Z-\theta_i)\big]$$
is minimized at $\theta_i=F_Z^{-1}(\hat\tau_i)$ — the $W_1$ projection — and it has unbiased sample gradients. End-to-end Wasserstein, by way of quantile regression on the midpoint quantiles.

Let me write the tabular policy-evaluation update to make the gradient concrete. The target for quantile $\hat\tau_i$ at state $x$ is a Bellman sample $r+\gamma z'$ with $z'\sim Z_\theta(x')$; the per-sample loss is $\rho_{\hat\tau_i}\big((r+\gamma z')-\theta_i(x)\big)$, with $u=(r+\gamma z')-\theta_i(x)$. Its negative subgradient in $\theta_i(x)$ is $+(\hat\tau_i-\mathbb{1}_{u<0})$, so SGD ascends... no, descends the loss by stepping along the negative gradient:
$$\theta_i(x)\leftarrow\theta_i(x)+\alpha\big(\hat\tau_i-\mathbb{1}\{\,r+\gamma z'<\theta_i(x)\,\}\big).$$
Read it: if my $i$-th location sits *below* the sampled target ($\theta_i<r+\gamma z'$, so $u>0$, indicator $0$), I push it *up* by $\alpha\hat\tau_i$; if it sits above, I push it *down* by $\alpha(1-\hat\tau_i)$. For a high quantile $\hat\tau_i$ near 1 the up-pushes are big and the down-pushes tiny, so the location settles high in the distribution; for a low $\hat\tau_i$ the reverse. That's quantile regression doing TD, and a single bootstrapped sample suffices. In practice it's better to use all the next-state locations: draw the full set $\{\theta_j(x')\}$ and average the update over all pairs $(\theta_i(x),\theta_j(x'))$ rather than one sampled $z'$.

There's a wrinkle I should fix before going to nonlinear function approximation. $\rho_\tau$ is not smooth at $u=0$: its derivative jumps from $-(1-\tau)$ to $\tau$ across zero, and its magnitude stays constant ($\tau$ or $1-\tau$) as $u\to0$. With a deep network this constant-magnitude, kinked gradient near zero is the kind of thing that hurts optimization — there's no shrinking of the step as the error gets small, so the locations jitter. Round off the kink with a Huber loss: quadratic near zero, linear in the tails. The Huber loss is
$$\mathcal{L}_\kappa(u)=\begin{cases}\tfrac12u^2,&|u|\le\kappa\\[2pt]\kappa\big(|u|-\tfrac12\kappa\big),&|u|>\kappa,\end{cases}$$
and the *quantile* Huber loss just multiplies it by the same asymmetric quantile weight, using the magnitude $|\tau-\mathbb{1}_{u<0}|$ so the loss stays nonnegative:
$$\rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}\{u<0\}\big|\,\mathcal{L}_\kappa(u),\qquad \rho^0_\tau=\rho_\tau.$$
For $\kappa=1$ the underlying Huber piece is $\frac12u^2$ inside $[-1,1]$ and $|u|-\frac12$ outside — exactly the gradient-clipped squared error the scalar agent already used — so I am swapping its Huber for an *asymmetric* Huber. The $\kappa=0$ case is a separate branch back to the hard quantile loss $|\tau-\mathbb{1}_{u<0}|\,|u|$; I cannot get it by plugging $0$ into $\mathcal{L}_\kappa$, because that would zero out the tail.

Now control. The distributional Bellman optimality operator picks the next action greedily, but greedily with respect to *what*? The whole objective is still to maximize expected return — I'm enriching the representation, not changing what "optimal" means — so the greedy action is the one maximizing the *mean* of the next-state value distribution:
$$\mathcal{T}Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(x',a^\star),\qquad a^\star=\arg\max_{a'}\mathbb{E}_{z\sim Z(x',a')}[z]=\arg\max_{a'}\tfrac1N\sum_j\theta_j(x',a').$$
The mean of a uniform-Dirac distribution is just the average of its locations, so $a^\star$ is the argmax of the per-action location-average — a drop-in for DQN's $\arg\max_a Q$.

For a concrete agent I take the DQN torso and make three minimal changes. First, the output layer becomes size $|\mathcal{A}|\times N$: $N$ quantile locations per action instead of one $Q$ per action. Second, the loss is the quantile Huber loss instead of the scalar Huber. Third, the old RMSProp recipe is no longer the natural default once the loss becomes an all-pairs asymmetric regression objective; Adam's per-parameter moments are a better optimization match for gradients whose frequency and scale differ across quantile levels. The per-transition target is built from a target network: compute $a^\star$ from the target net's next-state location-averages, form the bootstrapped target locations $\mathcal{T}\theta_j=r+\gamma\,\theta_j(x',a^\star)$ for all $j$ (with $\gamma$ zeroed at terminals), and then pull each prediction location $\theta_i(x,a)$ toward the *set* of target locations using the quantile Huber loss summed over the predicted quantiles and averaged over the target samples:
$$\mathcal{L}=\sum_{i=1}^N\mathbb{E}_j\big[\rho^\kappa_{\hat\tau_i}\big(\mathcal{T}\theta_j-\theta_i(x,a)\big)\big]=\frac1N\sum_{i=1}^N\sum_{j=1}^N\rho^\kappa_{\hat\tau_i}\big(\mathcal{T}\theta_j-\theta_i(x,a)\big).$$
Every predicted location $\theta_i$ is regressed, at its own quantile level $\hat\tau_i$, against *all* $N$ bootstrapped target locations — the all-pairs version of the tabular update. There's no projection and no $[V_{\min},V_{\max}]$; the only extra knob over DQN is $N$, the number of quantiles, which is just the resolution at which I approximate the distribution. With one output per action the network shape collapses back to the DQN shape, though under hard quantile regression that single output is a median target rather than a mean target; as $N$ grows it resolves finer quantiles, including the low-probability tails.

I should check the thing that worried me at the outset: does composing the Bellman operator with this quantile projection still contract? Function approximation plus a projection can destroy a contraction. Let me work out the metric in which it survives, because the answer is not the obvious one. Define the quantile projection $\Pi_{W_1}Z=\arg\min_{Z_\theta}W_1(Z,Z_\theta)$, which by the midpoint lemma places locations at $F_Z^{-1}(\hat\tau_i)$. First a useful fact about $d_\infty$ between two such projections: since both project to locations at the *same* cumulative levels $\hat\tau_i$, the optimal coupling between the two uniform-Dirac distributions pairs $F_{\nu_1}^{-1}(\hat\tau_i)$ with $F_{\nu_2}^{-1}(\hat\tau_i)$ (matching equal masses in order), so
$$d_\infty(\Pi_{W_1}\nu_1,\Pi_{W_1}\nu_2)=\max_{i}\big|F_{\nu_1}^{-1}(\hat\tau_i)-F_{\nu_2}^{-1}(\hat\tau_i)\big|.$$
Now the contraction. Reductions first: rewards can be taken deterministic and, since $W_p$ is translation-invariant, shifted to zero; and since $\mathcal{T}^\pi$ is already a $\gamma$-contraction, it suffices to prove the projected step is a non-expansion at $\gamma=1$. I can also reduce $N$-Dirac value distributions to *single* Diracs by splitting each transition into $N$ branches (transition to $(x_i,a_i)$ with prob $p_i$ becomes $N$ branches each prob $p_i/N$, one carrying $\delta_{\theta_j(x_i,a_i)}$), which has the same law after the operator. So it's enough to handle the single-Dirac case.

Single Diracs: $Z(x,a)=\delta_{\theta(x,a)}$, $Y(x,a)=\delta_{\psi(x,a)}$. From $(x',a')$ the successors are $(x_i,a_i)$ with probabilities $p_i$, so $\mathcal{T}^\pi Z(x',a')=\sum_i p_i\delta_{\theta_i}$ and $\mathcal{T}^\pi Y(x',a')=\sum_i p_i\delta_{\psi_i}$ (writing $\theta_i=\theta(x_i,a_i)$). Project each onto a single Dirac at its $\tau$-quantile. Let $\theta_u$ be the $\tau$-quantile of $\mathcal{T}^\pi Z(x',a')$ and $\psi_v$ that of $\mathcal{T}^\pi Y(x',a')$; then $d_\infty$ of the projections is $|\theta_u-\psi_v|$. I claim this can't exceed $\max_i|\theta_i-\psi_i|$. Suppose for contradiction $|\theta_u-\psi_v|>|\theta_i-\psi_i|$ for *every* $i$, and WLOG $\theta_u\le\psi_v$. Partition the successor index set: $I_{\le\theta_u}=\{i:\theta_i\le\theta_u\}$ and $I_{\ge\psi_v}=\{i:\psi_i\ge\psi_v\}$. If the assumption held, no index could be in both (an $i$ with $\theta_i\le\theta_u$ and $\psi_i\ge\psi_v$ would have $|\theta_i-\psi_i|\ge\psi_v-\theta_u=|\theta_u-\psi_v|$, contradicting strict inequality), so $I_{\le\theta_u}\subseteq I_{<\psi_v}$. But $\theta_u$ is the $\tau$-quantile of $\sum_ip_i\delta_{\theta_i}$, so $\sum_{i\in I_{\le\theta_u}}p_i\ge\tau$; hence $\sum_{i\in I_{<\psi_v}}p_i\ge\tau$, which forces the $\tau$-quantile of $\sum_ip_i\delta_{\psi_i}$ to be *strictly below* $\psi_v$ — contradicting that $\psi_v$ *is* that quantile. So the assumption is impossible, $|\theta_u-\psi_v|\le\max_i|\theta_i-\psi_i|$, and taking the max over levels $\hat\tau_i$ and the sup over $(x',a')$:
$$\bar d_\infty(\Pi_{W_1}\mathcal{T}^\pi Z,\Pi_{W_1}\mathcal{T}^\pi Y)\le\gamma\,\bar d_\infty(Z,Y).$$
So the *combined* operator $\Pi_{W_1}\mathcal{T}^\pi$ is a $\gamma$-contraction in the $\infty$-Wasserstein metric, the supremum gap between inverse CDFs, with a unique fixed point $\hat Z^\pi$ that the iteration and its stochastic approximation converge to. And because $\bar d_p\le\bar d_\infty$, convergence in $d_\infty$ gives convergence in every $p$.

But I should be careful not to overclaim: does the same hold in $\bar d_p$ for $p<\infty$? Let me try to break it, because if it breaks I want the example. Two Diracs ($N=2$), $\gamma=1$, one start state $x$ going to $x_1$ with prob $2/3$ and $x_2$ with prob $1/3$, all rewards zero. Take
$$Z(x_1)=\tfrac12\delta_0+\tfrac12\delta_2,\ Y(x_1)=\tfrac12\delta_1+\tfrac12\delta_2,\quad Z(x_2)=\tfrac12\delta_3+\tfrac12\delta_5,\ Y(x_2)=\tfrac12\delta_4+\tfrac12\delta_5.$$
At $x_1$ the distributions differ only by moving one atom from $0$ to $1$, half the mass over distance $1$, so $d_p(Z(x_1),Y(x_1))=(\tfrac12\cdot1^p)^{1/p}=2^{-1/p}$; same at $x_2$ (atom $3\to4$); so $\bar d_p(Z,Y)=2^{-1/p}$. Now back up to $x$:
$$\mathcal{T}^\pi Z(x)=\tfrac13\delta_0+\tfrac13\delta_2+\tfrac16\delta_3+\tfrac16\delta_5,\qquad \mathcal{T}^\pi Y(x)=\tfrac13\delta_1+\tfrac13\delta_2+\tfrac16\delta_4+\tfrac16\delta_5.$$
Project onto two equal Diracs: the locations are the $25\%$ and $75\%$ quantiles. For $\mathcal{T}^\pi Z(x)$ the CDF reaches $1/3$ at $0$ and $2/3$ at $2$, so the $25\%$ quantile is $0$ and the $75\%$ quantile is $3$ (mass $1/3+1/3=2/3<3/4$ by $2$, crossing $3/4$ at $3$); $\Pi Z(x)=\tfrac12\delta_0+\tfrac12\delta_3$. Likewise $\Pi Y(x)=\tfrac12\delta_1+\tfrac12\delta_4$. So
$$\bar d_p(\Pi\mathcal{T}^\pi Z,\Pi\mathcal{T}^\pi Y)=\big(\tfrac12(|1-0|^p+|4-3|^p)\big)^{1/p}=1\;>\;2^{-1/p}=\bar d_p(Z,Y).$$
The distance *grew* — for every finite $p$ — so $\Pi_{W_1}\mathcal{T}^\pi$ is not even a non-expansion in $\bar d_p$, $p<\infty$. Only the $\infty$-Wasserstein contraction is real; the others fail. Good to know, and it means the convergence I rely on lives specifically in $d_\infty$.

So the line is clear: I can't descend Wasserstein, but I can transpose the categorical parametrization to uniform-mass / variable-location, recognize the locations as midpoint quantiles, hit them with the unbiased quantile-regression loss (Huberized for smoothness), act greedily on the mean of the next-state distribution, and the projected Bellman operator provably contracts in $\infty$-Wasserstein. Now the code, on the DQN torso with output $|\mathcal{A}|\times N$.

```python
import torch
import torch.nn as nn

N = 200          # number of quantile locations per action
KAPPA = 1.0      # Huber threshold; kappa=0 uses the hard quantile loss branch

class DistributionalQNetwork(nn.Module):
    # DQN torso; head emits N quantile LOCATIONS per action (uniform 1/N mass each).
    def __init__(self, n_actions, n=N):
        super().__init__()
        self.n_actions, self.n = n_actions, n
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n),
        )
        # midpoint quantile levels  tau_hat_i = (2i-1)/(2N) = (i+0.5)/N
        self.register_buffer("levels", (torch.arange(n, dtype=torch.float32) + 0.5) / n)  # (N,)

    def prediction(self, x):
        return self.net(x.float() / 255.0).view(-1, self.n_actions, self.n)   # (B, A, N): theta_i(x,a)

    def greedy_action(self, x):
        q = self.prediction(x).mean(dim=2)       # Q(x,a) = (1/N) sum_i theta_i(x,a) -- act on the MEAN
        return q.argmax(dim=1)

def bootstrap_target(target_net, rewards, next_obs, dones, gamma):
    with torch.no_grad():
        next_q = target_net.prediction(next_obs)                   # (B, A, N)
        a_star = next_q.mean(dim=2).argmax(dim=1)                  # greedy on next-state mean
        batch = torch.arange(next_obs.size(0), device=next_obs.device)
        next_theta = next_q[batch, a_star]                         # (B, N): theta_j(x', a*)
        not_done = 1.0 - dones.float().view(-1, 1)
        # T theta_j = r + gamma theta_j(x', a*)  (gamma=0 at terminal); no projection onto any grid
        Ttheta = rewards.float().view(-1, 1) + gamma * not_done * next_theta
    return Ttheta                                                  # (B, N)

def pairwise_distribution_loss(theta, target, levels, kappa=KAPPA):
    # pairwise TD errors u_ij = T theta_j - theta_i      (B, N_i, N_j)
    u = target.unsqueeze(1) - theta.unsqueeze(2)
    tau = levels.view(1, -1, 1)                                    # tau_hat_i along the i-axis
    weight = (tau - (u.detach() < 0).float()).abs()                # 1{u<0} is a weight, not a path
    if kappa == 0:
        rho = weight * u.abs()                                     # hard quantile loss
    else:
        abs_u = u.abs()
        huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
        rho = weight * huber                                       # quantile Huber loss
    # E_j over target samples (mean over j), sum over predicted quantiles i, mean over batch
    return rho.mean(dim=2).sum(dim=1).mean()                       # sum_i E_j[ rho ] then batch-mean

def loss_fn(online_net, obs, actions, target, kappa=KAPPA):
    pred = online_net.prediction(obs)
    batch = torch.arange(obs.size(0), device=obs.device)
    theta = pred[batch, actions]                                   # (B, N): theta_i(x,a)
    return pairwise_distribution_loss(theta, target, online_net.levels, kappa)
```

Everything now fits together: Wasserstein is the metric the distributional Bellman operator contracts in, but its sample gradient is biased, so a categorical agent retreats to KL-after-projection and loses the guarantee; transposing the parametrization to uniform mass on $N$ variable *locations* turns the locations into quantiles; the $W_1$-optimal locations are the midpoint quantiles $\hat\tau_i=(2i-1)/(2N)$ because the cell-wise subgradient is $2F(\theta)-(\tau+\tau')$; the quantile-regression loss $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$ hits those locations with an unbiased sign-only stochastic gradient; the quantile Huber branch smooths that loss for a deep net while the $\kappa=0$ branch keeps the hard loss available; greedy control still uses the next-state *mean*; the DQN torso only needs an $|\mathcal{A}|\times N$ head, the all-pairs loss, and Adam; and the projected operator $\Pi_{W_1}\mathcal{T}^\pi$ provably $\gamma$-contracts in $\infty$-Wasserstein, while the finite-$p$ metrics fail by the counterexample above.
