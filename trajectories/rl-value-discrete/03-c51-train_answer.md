The quantile head did the thing I bet on, and the LunarLander column proves it: where the dueling head had $\{127.35, 229.07, \mathbf{-89.25}\}$ for a mean of $89$, QR-DQN posted $\{152.64, 243.81, 194.99\}$ for a mean of $197.15$ — the negative seed gone, the spread collapsed from a 318-point range to 91, every seed solidly positive. CartPole pinned at a clean 500 on all three seeds (the 50-way head did not destabilize a solved task), and Acrobot landed at $-80.07$, a hair better than dueling and right in the low $-80$s, since its return is nearly unimodal. The rung worked. But $197$ is good, not great, and the seeds are still ragged — seed 42 at $152$ is well below seed 123 at $244$, and a consistently-landing agent should be in the 200s on every seed. The quantile picture is still coarse where it matters most, and I can name why precisely: QR-DQN learns $N=50$ equal-mass locations, so resolution is spread evenly across the *probability* axis, and the low-probability crash tail of LunarLander — the rare large-negative outcome that decides the worst seeds — gets the same one-in-50 mass budget as the dense middle. On a single 500k-step run those tail locations are the noisiest part of the head, and a noisy tail location feeds straight into the bootstrap mean and the greedy argmax. The classic-control return range is *known* and bounded, so I can fix the locations and learn the mass instead.

I propose **C51**, a categorical distributional value function on a fixed grid. The starting point is unchanged — I refuse to collapse the return to its mean, and I keep the distributional Bellman recursion $Z(s,a) \stackrel{D}{=} R(s,a) + \gamma Z(s',a')$, whose operator $\gamma$-scales the next-state distribution (a *horizontal* shrink toward 0), shifts by the reward, and mixes over transitions, so it contracts in **Wasserstein** and emphatically not in KL, total variation, or Kolmogorov distance, which compare mass at matched locations and are blind to the shrink. QR-DQN's whole trick was to be Wasserstein-aware by learning locations; here I deliberately give that up, so I need to know exactly what I am buying. The distribution class is a **categorical** law on a fixed grid: pick $N$ canonical returns evenly spaced, $z_i = v_{\min} + i\,\Delta z$ with $\Delta z = (v_{\max} - v_{\min})/(N-1)$, and let the network emit a probability per atom via softmax, $Z_\theta(s,a) = \sum_i p_i(s,a)\,\delta_{z_i}$. This is expressive — any shape on the grid, multimodal included — and it is a softmax classifier per action. Bounding the support is not just a concession but a feature: it bakes in the prior that returns beyond the range are equally extreme, and, the point that answers my doubt, it lets me *spend* the $N$ atoms across the *value* axis at uniform spacing, so the crash tail gets honest grid resolution at its actual return magnitude rather than one slippery quantile location. The cost is that I must supply $[v_{\min}, v_{\max}]$, which on classic control is acceptable because the ranges are known.

The loss is where the categorical representation forces a different path. Wasserstein is the metric the theory wants, but its sampled gradient is **biased** — exactly what drove me to quantiles last rung. With a mixture target seen one sample at a time, the partition inequality $d_p^p(P,Q) \le \mathbb{E}_I\,d_p^p(P_i, Q)$ is strict, so SGD on the sampled $W_p$ descends an upper bound whose minimizer is wrong. The clean counterexample is target $\frac12\delta_0 + \frac12\delta_1$ and prediction $p\delta_0 + (1-p)\delta_1$: the true $d_1 = |p - \frac12|$ is minimized at $p = \frac12$, but the *expected sampled* distance is $\frac12(1-p) + \frac12 p = \frac12$, constant in $p$, gradient zero everywhere. QR-DQN dodged this by switching the loss to quantile regression, whose minimizers are the $W_1$-optimal *locations*; with fixed locations I cannot, because the locations are not free. What I *can* minimize unbiasedly from samples is **cross-entropy**, the softmax's native loss. The only obstruction is geometric: the Bellman update sends each atom $z_j$ to $r + \gamma z_j$, which almost never lands on the grid $\{z_i\}$, so the target $TZ_\theta$ and my parametrization $Z_\theta$ live on disjoint supports — and a KL between disjoint supports is exactly the useless vertical quantity I have been avoiding. So I project the shifted target back onto my grid before taking cross-entropy.

The projection respects the geometry by distributing each shifted atom's mass onto its two nearest grid neighbors by linear interpolation, clamping anything outside $[v_{\min}, v_{\max}]$ to the endpoints. The shifted atom $\hat T z_j = \mathrm{clamp}(r + \gamma z_j, v_{\min}, v_{\max})$ sits at fractional grid position $b_j = (\hat T z_j - v_{\min})/\Delta z \in [0, N-1]$, between $l = \lfloor b_j \rfloor$ and $u = \lceil b_j \rceil$; the lower atom receives weight $(u - b_j)$ and the upper $(b_j - l)$, summing to 1 so mass is preserved, each times the source probability $p_j$. Accumulating over all source atoms gives the projected target $m$. One subtlety in code: when $b_j$ is exactly an integer, $l = u$ and the lower weight $(u - b_j) = 0$ would drop the mass, so I send the full $p_j$ to that single atom — the `(l == u)` correction. Terminal transitions zero $\gamma$ via $(1 - \text{dones})$, collapsing every shifted atom to $r$. The target is built from the **target network** (frozen bootstrap, as in DQN), and the greedy action is chosen on the *mean* of the next-state distribution, $a^\* = \arg\max_a \sum_i z_i\,p_i(s',a)$, keeping action selection a drop-in for epsilon-greedy DQN. The sample loss is the cross-entropy between the projected target $m$ and the prediction,

$$-\sum_i m_i \log p_i(s,a),$$

the cross-entropy term of $\mathrm{KL}\big(\Phi\,\hat T Z_{\theta^-}(s,a)\,\|\,Z_\theta(s,a)\big)$. The distributional Bellman update has become **multiclass classification** over the $N$ atoms — exactly the unbiased-gradient regime I retreated to.

What I am trading versus QR-DQN is the crux. KL is insensitive to the atom *values* — it only matches mass — so this loss is *not* Wasserstein-aware the way the quantile loss was, badly chosen $v_{\min}, v_{\max}$ make the fixed grid a real handicap, and the projection introduces a small bias the quantile approach avoided. What I get back is that the crash-tail mass is now a softmax probability on a grid point at the tail's actual return magnitude, learned by stable cross-entropy, rather than a single sliding location that has to *be* the tail quantile — and for a heavy-tailed bimodal return on a bounded-range task, that is the right trade, aimed precisely at the under-resolved tail I diagnosed. On this edit surface the torso is the **fixed MLP encoder** ($\text{obs\_dim} \to 120 \to 84$), so `QNetwork` changes only at the head: a linear map $84 \to |A|\cdot N$ reshaped to $(\text{batch}, |A|, N)$ and softmaxed over the atom axis, with `forward` returning the per-action mean $\sum_i z_i p_i$ so the harness argmaxes a clean $(\text{batch}, |A|)$. I set $N = 51$ atoms. The support is the load-bearing scaffold choice: $v_{\min} = -500$, $v_{\max} = +500$, *not* the narrow $[-10, 10]$ the generic recipe uses, because that range was tuned for clipped-reward Atari while here the returns are unclipped and genuinely span this range (CartPole reaches $+500$, LunarLander runs roughly $-400$ to $+300$, Acrobot roughly $-500$ to $-60$); a grid that did not cover $-500$ to $+500$ would clamp the crash tail and the CartPole cap to the endpoints and defeat the whole point of switching to a fixed grid. That gives $\Delta z = 1000/50 = 20$ — coarse in absolute terms, but the bimodal structure I care about (a few-hundred-point landing peak versus a few-hundred-negative crash peak) is comfortably resolved at that spacing. I drop both the dueling head and the quantile head to isolate the categorical effect, with cross-entropy in place of quantile Huber, Adam at the scaffold learning rate. Falsifiably against QR-DQN's $\{500, 197.15, -80.07\}$: CartPole should stay near 500 (the only risk being the coarse grid blurring the cap), Acrobot on par in the low $-80$s and perhaps a touch noisier since its return sits in the dense lower-middle of a wide grid, and LunarLander is the test — if the under-resolved tail was really what kept QR-DQN at $197$ with ragged seeds, the fixed-grid mass model should lift the mean above $197$ and tighten the worst seed, because the crash tail now has honest resolution at its true magnitude.

```python
# EDITABLE region of custom_value_discrete.py — step 3: C51
class QNetwork(nn.Module):
    """Distributional Q-network for C51: MLPEncoder (fixed) + n_actions x n_atoms head."""

    def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=-500, v_max=500):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_atoms)

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        return q_values

    def get_action(self, obs, action=None):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(obs)), action]


class ValueAlgorithm:
    """C51 -- Categorical DQN with distributional value learning."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_atoms = 51
        self.v_min = -500.0
        self.v_max = 500.0
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        action, _ = self.q_network.get_action(obs_t)
        return action.item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            _, next_pmfs = self.target_network.get_action(next_obs)
            next_atoms = rewards.unsqueeze(1) + self.gamma * self.target_network.atoms * (1 - dones.unsqueeze(1))
            # Projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(obs, actions)
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_values = (old_pmfs * self.q_network.atoms).sum(1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
