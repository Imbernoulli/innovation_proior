The floor the whole ladder reacts to is ordinary deep Q-learning on the fixed scaffold: a frozen Nature-DQN encoder feeding a single linear head, a uniform replay buffer, $\epsilon$-greedy acting, and a periodic target copy. When I instrument plain DQN honestly — freeze the greedy policy, roll it out, and compare the value the network *predicts* for that policy against the *actual* discounted return it collects — the two should coincide, because the predicted value of a policy is by definition the return you expect from running it. Instead the prediction climbs well above the realized return, and on the harder games it runs away while the score stagnates. That is the one structural disease I have to fix first, and diagnosing where it lives tells me the minimal first edit.

The bias cannot come from the optimizer — least-squares regression toward a target is unbiased for that target — so it lives in the target itself. DQN regresses $Q(S_t,A_t;\theta)$ onto $Y_t = R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$, and the only term that is a function of my noisy estimates is $\max_a Q(S_{t+1},a;\theta^-)$. The max is convex, so by Jensen, even with *per-action unbiased* estimates $Q(s',a)=Q_*(s',a)+\epsilon_a$ with $\mathbb{E}[\epsilon_a]=0$, we get $\mathbb{E}[\max_a(Q_*(s',a)+\epsilon_a)]\ge\max_a Q_*(s',a)$: the expected max of noisy values exceeds the max of true values. The max goes hunting across the $m$ actions and selects whichever estimate landed highest, preferentially picking the action whose noise was most positive — positive noise gets selected, negative noise discarded — and then I bootstrap this inflated number backward through the chain of states. I can make the worst case exact. In an all-equal-value state ($Q_*(s,a)=V_*(s)$ for every $a$) with balanced errors $\sum_a\epsilon_a=0$ and mean-squared spread $\frac1m\sum_a\epsilon_a^2=C>0$, the single-estimator max overshoots by at least $\sqrt{C/(m-1)}$, and the bound is tight (put $\epsilon_a=\sqrt{C/(m-1)}$ on $m-1$ actions and $-\sqrt{(m-1)C}$ on the last). The companion i.i.d.-uniform case gives $\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *grows* with the number of actions — so Seaquest's eighteen actions are exactly where the inflation should be worst. And the bias is not the harmless uniform kind that preserves the $\arg\max$: it depends on action count and on how well each region was fit, so it is non-uniform, it scrambles the *relative* ordering the greedy policy reads off, and bootstrapping propagates that corruption.

I propose **Double DQN with a decoupled target**: split the two jobs the single max conflates. Rewrite the target as $\max_a Q(S_{t+1},a;\theta^-)=Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta^-);\theta^-\big)$, and the disease is plain — the same numbers $\theta^-$ both *select* the greedy next action (inner $\arg\max$) and *evaluate* it (outer $Q$). The action picked is by construction the one whose estimate is largest, i.e. whose noise is most positive, and then that same inflated estimate is read off as its value; selection and evaluation are perfectly correlated in their error, and that correlation is what turns "noisy" into "biased high." The cure is to evaluate with a *different* set of estimates. If the selecting net's chosen action is $a^\star$, an evaluator whose error on $a^\star$ is independent of *why* $a^\star$ was selected is not conditioned-on-being-large, so it is an unbiased estimate of $Q_*(S_{t+1},a^\star)$; in the all-equal stress state its expected value is just $V_*(S_{t+1})$, and the tight-example floor of $\sqrt{C/(m-1)}$ drops to zero. This is the idea behind the tabular two-estimator scheme of van Hasselt (2010) — two tables, one picks and the other scores.

What makes it the right *first* rung is that I install the decoupling without any new machinery. The literal two-estimator recipe would train and store a whole second network and arrange symmetric random-assignment updates — a lot to bolt onto a delicate system, and it would confound "does fixing the max help" with "does a second network help." But the scaffold already hands me a second set of weights: the target network $\theta^-$, a frozen stale copy of $\theta$. So I let the **online** net $\theta$ select the greedy next action and the **target** net $\theta^-$ score it:
$$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big).$$
Compared to plain DQN — where both select and evaluate run on $\theta^-$ — the *only* change is whose $\arg\max$ supplies the action, $\theta$ instead of $\theta^-$. No new parameters, no extra network, no meaningful extra compute, and the target-sync rule is untouched. The decoupling is imperfect, and I am honest about why: $\theta^-$ is a stale copy of $\theta$, so their errors are correlated, and right after each `target_network_frequency`-step sync the two coincide exactly and the target momentarily reverts to a plain max. So this removes much of the bias — specifically the non-uniform part that corrupts relative ordering — but not all of it, with the residual largest where the action count is largest. The lever to push further (widening the sync gap so $\theta$ and $\theta^-$ drift apart) is visible from the analysis, but I am installing the decoupling here, not tuning that knob, so I keep the scaffold's hard copy every $1000$ steps.

The concrete fill matches the edit surface exactly rather than a generic harness. The head stays the scaffold default — fixed `NatureDQNEncoder` → 512 features → a single `nn.Linear(512, n_actions)`, no capacity added — and `select_action` is greedy on those Q-values ($\epsilon$ is handled by the loop). The whole edit is in `update`: under `no_grad`, the online net's $\arg\max$ over $Q(S_{t+1},\cdot)$ selects the next action, the target net gathers that action's value, $y=R_{t+1}+\gamma\,Q(\cdot;\theta^-)(1-\text{done})$; then I regress $Q(S_t,A_t;\theta)$ onto $y$ and step Adam at the scaffold's `learning_rate = 1e-4`, with the hard target copy on every `target_network_frequency`-th step written through the soft-update formula at $\tau=1$. One detail I keep from the harness rather than the textbook: the loss is plain `mse_loss` on the TD error, not a Huber — the scaffold's protocol and budget check are built around the squared loss, and the only thing I change relative to DQN is which network supplies the selection inside the target. The expected outcome is a *correct but modest* rung: predicted values pulled back toward realized returns, a steadier and slightly better greedy policy across all three games, Pong essentially solved at its ceiling, Breakout in the low hundreds, and Seaquest the most variable because its long horizon and eighteen actions leave the most residual correlated-error bias. That ceiling is structural — $Q$ is still a single scalar, the mean of a return distribution the agent never sees — which is exactly what the next rung must break.

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
