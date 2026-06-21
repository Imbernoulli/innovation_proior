The substrate I am handed is plain DQN: a single linear head on a fixed 84-dimensional feature, emitting one number per action, regressed toward $r + \gamma \max_{a'} Q(s',a';\theta^-)$ under a squared error. The encoder, the uniform replay, the epsilon schedule, the hard target sync are all frozen, so the only lever I have is the head on top of those 84 features and how it is trained. Before reaching for distributional machinery, I want the cheapest question first: is the *architecture* of that head wasting data I already have? Because if it is, I can fix it without touching the algorithm — same target, same loss, same replay — and a free win at the bottom of the ladder is worth taking.

Staring at what the head does exposes the waste. It maps the shared feature to $|A|$ numbers, estimating each more or less independently as $|A|$ separate linear functions, and in most control states the action barely matters — a lander hovering upright and centered gets nearly the same future whether it fires left, fires right, or does nothing. Across the bulk of the state space the $|A|$ outputs are nearly equal, and the only thing that varies and matters is their common level: *how good is it to be in this state at all*. Worse is what one TD step touches. The loss is $(y - Q(s,a))^2$ with $Q(s,a)$ read by `gather` on the single sampled action $a$, so the gradient flows back only into that one action's column. The "how good is this state" quantity — shared across all $|A|$ actions and leaned on at every bootstrap, since the target is a $\max$ over next-state values — is only ever nudged through whichever action happened to be sampled. The most important quantity in the network gets the most diluted update.

I propose **Dueling DQN**: reshape the head, not the algorithm. The value of a state and the relative merit of the actions within it are two different objects, and the single head conflates them. There is a name for the split — the **advantage** $A(s,a) = Q(s,a) - V(s)$, due to Baird (1993): $V(s)$ is how good the state is, $A(s,a)$ how much better or worse action $a$ is than the state's baseline. So I keep the fixed encoder and replace the one linear head with two — a **value stream**, a linear map $84 \to 1$ producing $V(s)$, and an **advantage stream**, a linear map $84 \to |A|$ producing $A(s,\cdot)$ — then recombine them into a single $Q$ output *inside* the forward pass. The recombination must live there, not in the loss, because the whole appeal is that I keep the exact input/output interface of an ordinary Q-network: `forward(obs)` still returns $(\text{batch}, |A|)$ values, the frozen harness still argmaxes them, the same max-target TD loss still drives it, and the algorithm is unchanged. Two separate heads with two separate losses would forfeit that drop-in property.

The defining design choice is *how* to recombine, and the obvious answer is wrong. The naive sum $Q(s,a) = V(s) + A(s,a)$ reads straight off $A = Q - V$, but the loss only ever sees $Q$: take any constant $c$ and set $V'(s) = V(s) + c$, $A'(s,a) = A(s,a) - c$, and $V' + A' = Q$ identically, so the loss is blind to the shift. An entire one-parameter family of $(V,A)$ pairs produces the same $Q$, and gradient descent has no reason to land on the one where $V$ is the true value — $V$ could drift to $V + 100$ with $A$ absorbing the $-100$. The decomposition is **unidentifiable**, and two streams trading a constant back and forth is exactly the slack that makes optimization wander. So I subtract a per-state reference before forming $Q$. The **max** reference, $Q(s,a) = V(s) + (A(s,a) - \max_{a'} A(s,a'))$, forces $V$ to equal $\max_a Q$ exactly — clean textbook semantics — and it kills the bad trade, since under $A' = A - c$ the centered advantage is unchanged while $V' = V + c$ moves $Q$. But the subtracted $\max_{a'} A$ tracks whichever action currently leads, and during learning that argmax flips as estimates wobble; every flip makes the reference jump and the whole advantage vector re-center against a new maximum — a moving, jumpy target, exactly the instability I am trying to remove.

So I use the smoother anchor, the other identity $\mathbb{E}_a[A] = 0$, in its simplest policy-independent form — subtract the **uniform mean**:

$$Q(s,a) = V(s) + \Big(A(s,a) - \tfrac{1}{|A|}\sum_{a'} A(s,a')\Big).$$

This pins $V$ just as well: under $A' = A - c$ the mean shifts by $-c$ too, so the centered advantage is unchanged while $V' = V + c$ moves $Q$. The cost is the exact semantics — at the greedy action $Q(s,a^\*) = V + (A(s,a^\*) - \overline{A}) \neq V$ in general, so $V$ becomes the *mean* of the Q-values over actions rather than $\max_a Q$. What I buy is stability: the mean of $|A|$ numbers is smooth and slowly moving and does not lurch when the argmax flips, so the advantages track the mean rather than chase the optimal action's advantage. For a rung whose whole purpose is a free architectural win that must not destabilize the algorithm, that is the right trade. And crucially, subtracting any per-state constant never changes the rank order of actions, so $\arg\max_a Q$ is identical to the naive sum and to $\arg\max_a A$ — the greedy and epsilon-greedy policies are preserved exactly; the aggregation is a training-time offset-control device that touches what is learned, not what is acted.

The payoff is in the gradient flow, the original complaint. Writing the sampled action as $j$ and $Q_j = V + A_j - \frac1n\sum_k A_k$ with $n = |A|$, for any TD loss $\ell(Q_j, y)$ the chain rule gives $\partial Q_j/\partial V = 1$, $\partial Q_j/\partial A_j = 1 - 1/n$, and $\partial Q_j/\partial A_k = -1/n$ for $k \neq j$. So $V(s)$ receives the **full** TD signal on *every* sampled transition regardless of which action was taken — the shared state value is now learned from all the data instead of being smuggled through one action's column — while the advantage stream gets a contrastive signal whose components sum to zero, matching the fact that its absolute offset must not matter. In states where actions are equivalent, the centered advantages can settle to $\approx 0$ and the value is fixed through $V$ rather than through $|A|$ separate near-equal numbers. That is the redundancy I started from, removed by construction.

Grounding this on the actual edit surface settles where it differs from the generic Atari version. There is no conv torso to share and no trunk-gradient pathology: the shared trunk is the **fixed MLP encoder** ($\text{obs\_dim} \to 120 \to 84$), which I cannot touch, so I cannot push the split earlier and there is no $1/\sqrt2$ feature-gradient rescale to apply — both streams are single linear maps off the same frozen 84-dim feature, costing a trivial $+84$ parameters over the linear head. The bootstrap target stays the scaffold's **plain DQN max target** $r + \gamma \max_{a'} Q(s',a';\theta^-)$, *not* the double-DQN selection/evaluation split the generic recipe is usually paired with, because decoupling overestimation is a different rung's concern and the loop is frozen. The loss stays the squared TD error. The one stabilizer I keep, faithful to the generic recipe, is a **global gradient-norm clip at 10** after `backward()` — the two-stream head with a bootstrapped max target can occasionally produce a large update, and clipping tames it cheaply without interacting with the frozen loop. I expect CartPole pinned near 500 (the cap is already reached, so the test is whether it *holds*), Acrobot in the low $-80$s and roughly on par (its dense negative-time-to-goal reward means the action matters at nearly every step, so the split has little redundancy to exploit), and LunarLander — long hovering stretches interleaved with a few decisive corrections — as the place the value stream should most help, though a single linear value off a frozen 84-dim feature is not much capacity, so at least one seed is at real risk of the deceptive crash basin.

```python
# EDITABLE region of custom_value_discrete.py — step 1: Dueling DQN
class QNetwork(nn.Module):
    """Dueling Q-network: MLPEncoder (fixed) + separate value and advantage heads."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim)
        # Value stream
        self.value_head = nn.Linear(ENCODER_FEATURE_DIM, 1)
        # Advantage stream
        self.advantage_head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, obs):
        features = self.encoder(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueAlgorithm:
    """DuelingDQN -- Dueling Deep Q-Network."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

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
            target_max, _ = self.target_network(next_obs).max(dim=1)
            td_target = rewards + (1 - dones) * self.gamma * target_max

        old_val = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
```
