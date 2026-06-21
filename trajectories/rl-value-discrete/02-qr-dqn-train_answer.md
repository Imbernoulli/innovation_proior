The dueling head did its job and told me exactly where its limit is, in the LunarLander column. CartPole held at 500 on two of three seeds and 461.6 on the third; Acrobot came in at a mean of $-82.6$, on par with a plain value head, because its dense negative-time-to-goal reward makes the action matter at nearly every step and leaves the value/advantage split little redundancy to exploit. The signal is LunarLander: mean $89.06$, with per-seed returns $\{127.35, 229.07, \mathbf{-89.25}\}$. One seed found a genuinely good policy, one a mediocre one, and one fell into the crash basin and came out negative. That negative seed is the tell: the greedy policy systematically lands in the deceptive failure region, and the mean is dragged down by it. The dueling architecture improved how the *mean* state value is estimated and shared across actions — but the failure that survives is not about how I estimate the mean. It is that I estimate *only* a mean, and on LunarLander the return is sharply bimodal: a safe landing scores a few hundred, a crash scores a large negative, and a single scalar $Q = \mathbb{E}[Z]$ averages those two worlds into a number corresponding to neither, giving the argmax no way to tell "high mean because it sometimes lands and sometimes crashes" from "high mean because it reliably lands." So the next move is not a better head for the mean — it is to stop collapsing the return to its mean at all.

I propose **QR-DQN**: learn the full return distribution by quantile regression. The object is now the random return $Z(s,a)$ itself, and three things have to survive in distribution space for the machinery to work — a Bellman recursion the object obeys, a contraction so iterating it converges, and a loss I can train from sampled transitions. The recursion is the easy part: peel off the first reward and $Z(s,a) \stackrel{D}{=} R(s,a) + \gamma Z(s',a')$, an equality *in distribution*. The distributional Bellman operator scales the next-state distribution by $\gamma$ (shrinking it toward 0), shifts by the reward, and mixes over transitions. The contraction is where the design is decided, and it surprised me: this operator does *not* contract in KL. Scale two distributions on $\{1,2\}$ by $\gamma = 0.5$ to $\{0.5,1\}$ — the probabilities are untouched, so KL is exactly unchanged, no contraction, and disjoint supports stay KL-infinite no matter how far I shrink them toward 0. KL, total variation, Kolmogorov sup-CDF distance are all *vertical* metrics, comparing mass at matched locations, blind to how far apart the locations are. The Bellman update is a *horizontal* operation. The metric that sees it is **Wasserstein**, $d_p(F,G) = \big(\int_0^1 |F^{-1}(u) - G^{-1}(u)|^p\,du\big)^{1/p}$, a transport distance that is finite for disjoint supports and scales with $\gamma$; in the maximal metric $\bar d_p = \sup_{s,a} d_p$ the operator is a $\gamma$-contraction with the true return distribution as its unique fixed point. So the metric the theory loves is exactly the one a softmax cannot minimize.

And there is a hard wall behind it: the sample gradient of Wasserstein is **biased**. Form an empirical target from samples and minimize the sample $W_p$, and the minimizer of the expected sample loss is not the minimizer of the true $W_p$ — because $W_p$ is built from the quantile function $F^{-1}$, a single sample is a draw and not a quantile observation, and the optimal transport reshuffles which sample pairs with which prediction so its averaged gradient misses the population transport gradient. The metric the theory wants is the one I cannot descend from single transitions. The escape is to change *what* I parametrize. The categorical route fixes atom *locations* and learns *probabilities*; I turn that on its side. Fix the probabilities uniform, $q_i = 1/N$, and make the *locations* $\theta_i$ learnable: $Z_\theta(s,a) = \frac1N \sum_i \delta_{\theta_i(s,a)}$. "Where the $i$-th of $N$ equal lumps of mass sits" is exactly a **quantile**, so this transposed parametrization estimates quantiles of the return — and it buys three things at once: the support is unpinned, so locations slide to wherever the returns actually live with per-state adaptive resolution (CartPole's $0$ to $500$, LunarLander's $-400$ to $+300$, Acrobot's $-500$ to $-60$ would all have to share one fixed grid otherwise); there is no projection, because the Bellman target's atoms are just numbers I compare to my locations; and the quantiles may be reachable from samples without a biased gradient.

Which quantiles, and with what loss, is the load-bearing derivation. Minimize $W_1$ between an arbitrary target $Y$ and a uniform-$N$-Dirac distribution on ordered locations $\theta_1 \le \dots \le \theta_N$. With cumulative levels $\tau_i = i/N$ the inverse-CDF of the uniform comb is the staircase equal to $\theta_i$ on $(\tau_{i-1}, \tau_i]$, so $W_1 = \sum_i \int_{\tau_{i-1}}^{\tau_i} |F_Y^{-1}(\omega) - \theta_i|\,d\omega$; the cells decouple, each $\theta_i$ appearing only in its own integral. The subgradient of $\int_\tau^{\tau'} |F^{-1}(\omega) - \theta|\,d\omega$ in $\theta$ is $2F(\theta) - (\tau + \tau')$, zero when $F(\theta) = (\tau + \tau')/2$, so the $W_1$-optimal location is the quantile at the *midpoint* of each cell:

$$\theta_i = F_Y^{-1}(\hat\tau_i), \qquad \hat\tau_i = \frac{2i-1}{2N}.$$

Cell centers, not cell edges. But knowing the target quantiles is not enough — minimizing sample $W_p$ is biased even with this parametrization, so the unbiasedness must come from the *loss that hits each quantile*, and that loss is **quantile regression**. To estimate the $\tau$-quantile from samples, use the asymmetric loss $\rho_\tau(u) = u(\tau - \mathbf{1}\{u<0\})$ with $u = \hat Z - \theta$: it charges $\tau|u|$ on underestimates and $(1-\tau)|u|$ on overestimates, and its subgradient in $\theta$ is $\Pr(\hat Z < \theta) - \tau$, zero exactly at $\theta = F^{-1}(\tau)$. The gradient depends only on the *sign* of $u$ — $\tau - \mathbf{1}\{u<0\}$ — so a single sample gives an **unbiased** stochastic gradient. That is the whole trick: I cannot descend $W_p$, but I can descend the quantile-regression loss whose minimizers are the very locations that minimize $W_1$.

One wrinkle for a deep net: $\rho_\tau$ is kinked at $u=0$ and its gradient magnitude stays constant ($\tau$ or $1-\tau$) as $u \to 0$, so the step never shrinks and the locations jitter. I round the kink with a **Huber** loss — quadratic inside $|u| \le \kappa$, linear outside — weighted by the asymmetric factor, giving the *quantile Huber* loss $\rho_\tau^\kappa(u) = |\tau - \mathbf{1}\{u<0\}| \cdot L_\kappa(u)$. With $\kappa = 1$ the inner piece is $\frac12 u^2$ and the outer is $|u| - \frac12$ — exactly the gradient-clipped squared error a scalar agent already uses, now made asymmetric. Control stays mean-greedy, because the objective is still expected return: the greedy action is $\arg\max_a \frac1N \sum_j \theta_j(s,a)$, a drop-in for DQN's $\arg\max_a Q$. The bootstrapped target locations are $T\theta_j = r + \gamma\,\theta_j(s',a^\*)$ (with $\gamma$ zeroed at terminals), and each predicted location $\theta_i(s,a)$, at its own level $\hat\tau_i$, is regressed against *all* $N$ target locations via the all-pairs loss $\frac1N \sum_i \sum_j \rho_{\hat\tau_i}^\kappa(T\theta_j - \theta_i)$. No projection, no $[v_{\min}, v_{\max}]$; the only new knob over DQN is $N$.

On this edit surface the torso is the **fixed MLP encoder** ($\text{obs\_dim} \to 120 \to 84$), not a conv stack, so `QNetwork` changes only at the head — a linear map $84 \to |A|\cdot N$ reshaped to $(\text{batch}, |A|, N)$, with `forward` returning the per-action mean over the $N$ quantiles so the harness argmaxes a clean $(\text{batch}, |A|)$. I set $N = 50$, not the larger generic value: on single-environment classic control within a 500k-step budget, 50 locations already resolve the bimodality I care about — safe-landing mass versus crash mass — without inflating the head, and a coarser comb trains faster and more stably here. The midpoint levels $\hat\tau_i = (2i-1)/(2N)$ are fixed buffers, $\kappa = 1$. For the bootstrap I select $a^\*$ greedily on the target network's next-state quantile means and take *that* network's quantiles for $T\theta_j$ — the scaffold's plain-DQN style target (select and evaluate both on the target net), *not* the double-DQN split, since the frozen loop only lets me change the head and the loss. Terminals zero $\gamma$ via $(1 - \text{dones})$. I drop the dueling head deliberately: distributional learning is a change to the output object and loss, and stacking it on the two-stream head would conflate two rungs — I want the distributional effect isolated. The bet is entirely on LunarLander, and it is sharp: if modeling the return distribution is the right fix, the quantile head should let the greedy policy distinguish a reliably-landing action from a high-variance one, so I expect the worst seed to climb out of the negative basin and the mean to rise well above 89 with a tighter spread; CartPole should merely hold 500 (the risk being that a 50-way head destabilizes a solved task) and Acrobot stay in the low $-80$s. My one reservation is the tail: 50 uniform-mass quantiles are a coarse picture of a heavy-tailed return, and if the low-probability crash tail is where resolution is thinnest, that is precisely the opening for fixing the support and learning the mass instead.

```python
# EDITABLE region of custom_value_discrete.py — step 2: QR-DQN
class QNetwork(nn.Module):
    """Quantile Q-network for QR-DQN: MLPEncoder (fixed) + n_actions x n_quantiles head."""

    def __init__(self, obs_dim, n_actions, n_quantiles=50):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_quantiles)

    def forward(self, obs):
        """Return Q-values as mean of quantile values per action."""
        features = self.encoder(obs)
        quantiles = self.head(features).view(len(obs), self.n_actions, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, obs):
        """Return raw quantile values: [batch, n_actions, n_quantiles]."""
        features = self.encoder(obs)
        return self.head(features).view(len(obs), self.n_actions, self.n_quantiles)


class ValueAlgorithm:
    """QR-DQN -- Quantile Regression DQN with distributional value learning."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_quantiles = 50
        self.kappa = 1.0  # Huber loss threshold
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        q_values = self.q_network(obs_t)
        return torch.argmax(q_values, dim=1).item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(next_obs)  # [batch, n_actions, n_quantiles]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(next_obs)), next_actions]  # [batch, n_quantiles]
            # Compute target quantile values
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles_best * (1 - dones.unsqueeze(1))

        # Get current quantile values for taken actions
        current_quantiles = self.q_network.get_quantiles(obs)  # [batch, n_actions, n_quantiles]
        current_quantiles = current_quantiles[torch.arange(len(obs)), actions]  # [batch, n_quantiles]

        # Quantile Huber loss
        # Pairwise TD errors: [batch, n_quantiles (pred), n_quantiles (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # [batch, N, N]

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

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
