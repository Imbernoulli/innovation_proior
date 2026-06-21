The decoupled-target rung did what the bias analysis predicted, and the numbers mark its ceiling. Double DQN landed Breakout $170.7$ (seeds $141.6/125.6/244.8$), Seaquest $6789$ ($7178/5386/7804$), Pong $20.7$ ($20.4/20.6/21.0$). Pong is essentially solved — three seeds within $0.6$ — and tells me only that the critic functions. The information is in the other two, and it is *spread*: Seaquest swings from $5386$ to $7804$, the widest of any game, and Breakout from $125.6$ to $244.8$. That pattern is the residual disease, not a new one. Decoupling removed the non-uniform overestimation, but $\theta^-$ is still a stale copy of $\theta$, so right after each sync the target briefly reverts to a plain max, and on Seaquest's eighteen actions — exactly where the inflation $\epsilon\frac{m-1}{m+1}$ is largest — that residual moves the greedy policy from seed to seed. The deeper tell is that the critic is still a single scalar per action: a point estimate with no notion of its own spread, which absorbs all the stochasticity of returns and bootstraps it into one number that wobbles. Tightening the sync gap would shave a little more bias but cannot change *what* the critic represents. The ceiling is representational, and to break it I have to stop collapsing the return to its mean.

I train an agent to maximize expected return, so I learn $Q(x,a)=\mathbb{E}[Z(x,a)]$, one number per state-action. But $Z(x,a)$ — the actual return — is a random variable: in Seaquest, "surface, refill oxygen, score thousands" with some probability and "suffocate now, score little" with the rest, genuinely bimodal. The mean collapses that to a value the agent rarely receives, and throws away the *spread* — exactly what separates a $5386$ seed from a $7804$ one. If I could learn the whole law of $Z$ I would carry strictly more information. The catch in RL is that there are no given targets; I bootstrap, learning a guess from a guess. So the question is whether the machinery survives in distribution-space, and I let the representation and loss fall out of what the math allows.

The recursion exists: peeling the first reward off the return gives $Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A')$, a distributional Bellman operator $\mathcal{T}^\pi$ that scales the successor distribution by $\gamma$, convolves with transition noise, and shifts by the reward. The contraction lives or dies on the metric. KL is the deep-learning instinct, but the $\gamma$-scaling kills it — scaling *locations* toward zero leaves the *probabilities* untouched, so KL is unchanged under the very operation that should bring two distributions closer, and stays infinite for disjoint supports. KL, total variation, Kolmogorov are all vertical distances, blind to horizontal movement. The metric that sees the $\gamma$-shrink is Wasserstein, $d_p(F,G)=(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du)^{1/p}$, under which $\mathcal{T}^\pi$ is a $\gamma$-contraction with unique fixed point $Z^\pi$. Wasserstein is the *right* metric — and that is the problem, because I cannot minimize it from samples by SGD: the minimizer of the expected sample-Wasserstein loss is not the minimizer of the true $W_p$, the gradient is biased. (Target $\frac12\delta_0+\frac12\delta_1$ sampled as one Dirac, fit by $p\delta_0+(1-p)\delta_1$: true distance $|p-\frac12|$ is minimized at $p=\frac12$, but the expected sampled distance is constant $\frac12$, so its gradient points nowhere.) The fixed-atom route dodged this — learn probabilities on a prescribed grid, KL after a projection — but at a cost: it must be handed $[V_{\min},V_{\max}]$, it optimizes KL-after-projection rather than Wasserstein, and the projection exists only because fixed atoms force support collisions. I want something genuinely Wasserstein-aware, trainable online, with no projection and no support bounds — the scaffold gives me no way to know Seaquest's range (thousands) versus Pong's ($\pm21$) a priori, and guessing it wrong is its own failure mode.

I propose **QR-DQN**, a quantile-regression distributional critic, and the move that makes it work is to **transpose the parametrization**. The fixed-atom agent learns probabilities on fixed locations — the vertical axis. Turn it on its side: fix the probabilities to uniform $1/N$ and make the *locations* $\theta_i$ learnable, $Z_\theta(x,a)=\frac1N\sum_i\delta_{\theta_i(x,a)}$. Now I am learning *where* $N$ equal lumps of mass sit, and "where the $i$-th of $N$ equal lumps sits" is precisely a *quantile* of the return. Three wins fall out at once: the support is no longer pinned to any $[V_{\min},V_{\max}]$ (the locations slide to wherever Seaquest's thousands and Pong's tens actually live, adapting per state); there is no projection (the shifted Bellman-target locations are just numbers I compare to mine, disjoint supports a non-issue); and estimating quantiles is something I *can* do from samples without a biased gradient.

Which quantiles, and what loss? Minimizing $W_1$ between a target $Y$ and a uniform-$N$-Dirac on ordered $\theta_1\le\cdots\le\theta_N$ decouples cell-by-cell, $W_1=\sum_i\int_{\tau_{i-1}}^{\tau_i}|F_Y^{-1}(\omega)-\theta_i|\,d\omega$ with $\tau_i=i/N$, and the subgradient of one cell is $2F(\theta)-(\tau_{i-1}+\tau_i)$, zero at $F(\theta)=\frac{\tau_{i-1}+\tau_i}{2}$. So the $W_1$-optimal location is the quantile at each cell's *midpoint*, $\hat\tau_i=\frac{2i-1}{2N}$ — the centers, not the edges $i/N$. Now hit those midpoint quantiles from samples without bias. The parametrization alone does not unbias Wasserstein; the unbiasedness must come from the loss. Quantile regression supplies it: the $\tau$-quantile minimizes $\mathbb{E}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose subgradient in $\theta$ is $\Pr(\hat Z<\theta)-\tau$, zero exactly at $\theta=F^{-1}(\tau)$, and crucially it depends only on the *sign* of $u=\hat Z-\theta$, so a single sample gives an unbiased stochastic gradient. That is the escape: I cannot descend $W_p$, but I can descend the quantile-regression loss whose minimizers are the very locations that minimize $W_1$ — end-to-end Wasserstein by way of quantile regression on the midpoint quantiles.

One wrinkle before a deep net: $\rho_\tau$ has a kink at $u=0$, its gradient magnitude staying constant ($\tau$ or $1-\tau$) right down to zero error, so the locations jitter and never settle. I round it off with a Huber — quadratic inside $[-\kappa,\kappa]$, linear outside — weighted by the asymmetric factor $|\tau-\mathbb{1}_{u<0}|$. At $\kappa=1$ this is the gradient-clipped squared error, but made asymmetric per quantile level. This is also the right answer for the variance I saw on Seaquest: the squared loss put all its weight on the single mean and let a few large-return transitions dominate the gradient, whereas the quantile Huber spreads supervision across $N$ levels and clips tail influence, so a high-return Seaquest outlier shapes the high quantiles instead of yanking one scalar around. Control stays unchanged because the objective is still expected return — I act on the per-action mean of the locations, $a^\star=\arg\max_{a'}\frac1N\sum_j\theta_j(x',a')$, a drop-in for the previous rung's $\arg\max_a Q$. The per-transition target uses the target net: compute $a^\star$ from its next-state location means, form $\mathcal{T}\theta_j=r+\gamma\,\theta_j(x',a^\star)$ for all $j$ ($\gamma$ zeroed at terminals), and regress each predicted location $\theta_i(x,a)$ against the whole *set* of target locations with the quantile Huber summed over predicted quantiles $i$ and averaged over target samples $j$ — the all-pairs version of the tabular update. The only new knob over the scalar critic is $N$.

Fitting the edit surface: the fixed `NatureDQNEncoder` gives 512 features, the head becomes `Linear(512, n_actions * N)` reshaped to $(B, n_{\text{actions}}, N)$, and `forward` returns the per-action mean over the $N$ locations so the loop's eval-argmax works unchanged. I take $N=200$, the standard Atari resolution and exactly the width the scaffold's budget check is sized around ($1.05\times$ the $|\mathcal A|\times200$ head), so I sit at the budget, not over it; the midpoint levels are the fixed buffer $\hat\tau_i=(2i-1)/(2N)$; and $\kappa=1$ is hard-coded since the harness exposes no $\kappa=0$ hard-loss branch. I keep one optimizer detail from the distributional recipe — Adam with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$, a larger-than-default $\epsilon$ that stabilizes the all-pairs asymmetric regression whose per-quantile gradient scales differ — while the learning rate stays the scaffold's fixed $1\times10^{-4}$, and the target sync stays the hard copy. Against the measured numbers I expect Pong to hold $\approx21$, Breakout to clear $170$ (into the mid-$200$s) and Seaquest to clear $6789$ (into the $9000$s), and — the falsifiable part — the seed-to-seed variance *not* to blow up as the mean rises, because the Huber's tail-clipping and distributional averaging are exactly the stabilizers the bare squared loss lacked. If instead the mean climbs but the spread widens, that says $N=200$ on a frozen encoder is under-resolving the tails, and the next move is to stop fixing the quantile levels at all.

```python
class QNetwork(nn.Module):
    """QR-DQN quantile Q-network: NatureDQNEncoder (fixed) + quantile head."""

    def __init__(self, envs, n_quantiles=200):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_quantiles)

    def forward(self, x):
        """Return Q-values as mean of quantile values per action."""
        features = self.encoder(x)
        quantiles = self.head(features).view(len(x), self.n, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, x):
        """Return raw quantile values: [batch, n_actions, n_quantiles]."""
        features = self.encoder(x)
        return self.head(features).view(len(x), self.n, self.n_quantiles)


class ValueAlgorithm:
    """QR-DQN -- Quantile Regression DQN with distributional value learning."""

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.target_network_frequency = args.target_network_frequency
        self.n_quantiles = 200
        self.kappa = 1.0  # Huber loss threshold

        self.q_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

    def select_action(self, obs, epsilon):
        """Greedy action selection using mean of quantile values."""
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """QR-DQN update: quantile Huber loss."""
        with torch.no_grad():
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(batch.next_observations)  # [batch, n_actions, N]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(batch.next_observations)), next_actions]  # [batch, N]
            # Compute target quantile values
            target_quantiles = batch.rewards + self.gamma * next_quantiles_best * (1 - batch.dones)

        # Get current quantile values for taken actions
        current_quantiles_all = self.q_network.get_quantiles(batch.observations)  # [batch, n_actions, N]
        current_quantiles = current_quantiles_all[torch.arange(len(batch.observations)), batch.actions.flatten()]  # [batch, N]

        # Quantile Huber loss
        # Pairwise TD errors: [batch, N (pred), N (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Huber loss element-wise
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= self.kappa,
                            0.5 * td_errors ** 2,
                            self.kappa * (abs_td - 0.5 * self.kappa))

        # Asymmetric weighting by quantile level
        tau = self.tau.view(1, -1, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
