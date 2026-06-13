**Problem.** Deep Q-learning regresses $Q(S_t,A_t;\theta)$ onto
$R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$. The same estimates *select* the greedy next action and
*evaluate* it, and since $\max$ is convex, any estimation error biases the target *upward* (Jensen:
$\mathbb{E}[\max_a Q(s',a)]\ge\max_a\mathbb{E}[Q(s',a)]$). The bias is non-uniform across states, so it
corrupts the relative value ordering the greedy policy reads off, and bootstrapping propagates it — the
predicted value runs above the realized return.

**Key idea.** Decouple selection from evaluation. Reuse the target network already present in the
scaffold as the second estimator: the **online** net $\theta$ selects the greedy next action, the
**target** net $\theta^-$ scores it,
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. This is the minimal
change to DQN — one line, no new parameters, everything else (encoder, replay, $\epsilon$-greedy,
periodic copy) untouched.

**Why it works.** In an all-equal-true-value state with balanced errors of mean-squared spread $C$ over
$m$ actions, the single-estimator max overshoots by at least $\sqrt{C/(m-1)}$ (tight). An independent
evaluator drops that floor to $0$. The reuse is imperfect because $\theta^-$ is a stale copy of
$\theta$ (correlated errors; right after a sync it reverts to plain Q-learning), so this removes much,
not all, of the bias — most where the action count is largest (Seaquest's 18 actions).

**Scaffold edit / hyperparameters.** Head unchanged (fixed `NatureDQNEncoder` → 512 →
`Linear(512, n_actions)`); no capacity added. Loss = `mse_loss` on the TD error (the harness's protocol,
not Huber). Adam at `learning_rate = 1e-4`; hard target copy every `target_network_frequency = 1000`
steps via the soft-update formula at `tau = 1.0`. The only difference from DQN is the
`self.q_network(...).argmax(...)` selection inside `update`.

**What to watch.** Predicted values should fall toward realized returns; a steadier, slightly better
greedy policy than DQN across all three games, with Pong near its ceiling, Breakout in the low hundreds,
Seaquest the most variable. The ceiling is structural: $Q$ is still a scalar mean of a return
distribution the agent never sees — which is what the next rung breaks.

```python
class ValueAlgorithm:
    """DoubleDQN -- Double Deep Q-Network with hard target updates."""

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_network_frequency = args.target_network_frequency

        self.q_network = QNetwork(envs).to(device)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        """Epsilon-greedy action selection."""
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """DoubleDQN update: online net selects action, target net evaluates."""
        with torch.no_grad():
            # Double Q-learning: online net selects best action, target net evaluates it
            best_actions = self.q_network(batch.next_observations).argmax(dim=1, keepdim=True)
            target_q = self.target_network(batch.next_observations).gather(1, best_actions).squeeze()
            td_target = batch.rewards.flatten() + self.gamma * target_q * (1 - batch.dones.flatten())

        old_val = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_param.data
                )

        return {"td_loss": loss.item(), "q_values": old_val.mean().item()}
```
