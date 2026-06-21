The substrate hands me a fixed off-policy loop — one MuJoCo environment, a FIFO replay buffer of $(s, s', a, r, d)$ transitions, a long uniform-random warm-up, and one minibatch update per environment step — and asks me to fill in the learning algorithm. The body emits a real-valued action vector each step, and I want a controller learned from reward alone, model-free, with action selection cheap enough to run online. The natural starting point is the DQN recipe from value-based control: learn $Q(s,a;\theta)$, store transitions in replay, use a target network so the Bellman target does not move on the same step as the regressor, and act greedily by $\arg\max_a Q(s,a)$ with target $y = r + \gamma \max_{a'} Q(s',a')$. On a small discrete action set the network emits one value per action and the max is a reduction over a short vector. Put continuous torques in there and it breaks: $a \in \mathbb{R}^m$, and a neural $Q(s,a)$ hands me no $\arg\max$. I could run gradient ascent or random shooting over $a$, but then I am solving a non-convex inner problem for every action and for every bootstrap target in a minibatch, which defeats a real-time controller. Discretization is worse — bin each actuator and the action count is exponential in the joints ($3^7 = 2187$ for seven joints at three levels, before any fine torque resolution), and the grid throws away the metric structure that makes 0.30 and 0.31 nearby torques. The $\max$ over actions is the thing I must remove.

I propose DDPG, the deep deterministic policy gradient. The $\max$ exists because the value function is being asked to do two jobs at once — produce both an evaluation and the greedy action — so I split them: keep a critic $Q(s,a)$ for evaluation, and introduce a separate differentiable actor $\mu(s)$ whose only job is to emit the action I will actually use. Acting becomes one forward pass. The training question is then how to move $\mu$'s parameters so that $Q(s,\mu(s))$ goes up, and since the critic is differentiable in the action and the actor in its parameters, the chain rule answers it:

$$\nabla_\theta\, Q(s, \mu_\theta(s)) = \nabla_\theta\, \mu_\theta(s)\cdot \nabla_a Q(s,a)\big|_{a=\mu_\theta(s)}.$$

The action-gradient $\nabla_a Q$ says which direction in action space increases value at this state; the policy Jacobian $\nabla_\theta \mu_\theta$ pulls that direction back into parameter space. The inner action optimizer is amortized — instead of solving $\arg\max_a Q$ every step, I train a network whose output drifts toward actions the critic rates highly.

This is more than a heuristic for a fixed batch of states. Starting from the deterministic value functions $V^\mu(s) = Q^\mu(s,\mu_\theta(s))$ and $Q^\mu(s,a) = \mathbb{E}_{r,s'}[r + \gamma V^\mu(s')]$ and differentiating in $\theta$, the next-state distribution's dependence on the current action is not missing — it lives inside $\nabla_a Q^\mu$, because $Q^\mu$ is the whole action-value function. Unrolling the recursion and integrating over the start-state distribution gives

$$\nabla_\theta J = \int \rho^\mu(s)\, \nabla_\theta \mu_\theta(s)\cdot \nabla_a Q^\mu(s,a)\big|_{a=\mu(s)}\, ds,$$

an average over *states only* under the discounted visitation measure $\rho^\mu$. There is no action integral, and that absence is what makes off-policy learning cheap here: a stochastic policy-gradient estimator carries a sampled action and a score term, so off-policy data drags in action likelihood ratios, but here the target action is just $\mu(s')$, so I collect with a noisy behavior policy, learn $Q^\mu$ from the transitions using $\mu(s')$ in the backup, and pay no action-space importance ratio. Sampling states from replay does re-weight the actor update by the buffer's state distribution — a behavior-weighted approximation, not exact recovery of the start-state gradient — but I need coverage, not a high-variance continuous-action correction, which is exactly why the off-policy buffer is the right home.

So I keep an actor $\mu(s)$ and a critic $Q(s,a)$. The deterministic Bellman equation has no action expectation at the next state, $Q^\mu(s,a) = \mathbb{E}_{r,s'}[r + \gamma\, Q^\mu(s', \mu(s'))]$, and for a sampled transition with done flag $d$ the regression target must stop bootstrapping at true terminals: $y = r + \gamma(1-d)\, Q(s', \mu(s'))$. The critic minimizes squared error to that target. The actor takes the chain-rule step that raises $Q(s,\mu(s))$; because optimizers minimize, the actor loss is the *negative* value, $\mathcal{L}_{\text{actor}} = -Q(s,\mu(s)).\text{mean}()$, and minimizing it is ascent on the critic.

What makes the deep version stable is taming the same self-reference that destabilized deep Q-learning. If I compute $y$ with the live critic, every critic step moves both the prediction and the target; replay fixes the data correlation, but the target can still chase itself, and here it also depends on the *actor*, which is moving too. So both learned functions get slow target copies, and the stable target is $y = r + \gamma(1-d)\, Q_{\text{targ}}(s', \mu_{\text{targ}}(s'))$. Hard periodic copies would work, but the actor and critic are coupled — the critic target uses the target actor, the actor update uses the live critic — so smooth Polyak tracking, where the targets drift instead of jump, keeps the regression target from lurching: $\theta_{\text{targ}} \leftarrow (1-\tau)\theta_{\text{targ}} + \tau\theta$ with $\tau = 0.005$, which the substrate already provides as `soft_update`. Exploration is separate from the deterministic policy precisely because learning is off-policy: the deterministic actor outputs one action per state and explores nothing on its own, so the behavior policy adds independent Gaussian noise from outside, $a \leftarrow \text{clip}(\mu(s) + \sigma\cdot\varepsilon,\, -a_{\max}, a_{\max})$ with $\sigma = 0.1\,a_{\max}$. This is faithful to the off-policy argument — the noise only changes which transitions enter replay; the critic target and actor loss stay about the clean deterministic policy — and the long uniform-random warm-up does the heavy lifting of initial coverage.

In the implementation I keep the default deterministic-tanh `Actor` and the single `QNetwork` (the dimensions are fixed by the parameter-count check anyway), and fill `OffPolicyAlgorithm`: build `target_actor` and `qf1_target` copies, give `select_action` the Gaussian exploration noise, and in `update` compute the target under `no_grad`, regress the critic every call, and — respecting the loop's `policy_frequency=2` — only every second call take the actor step and the two soft-updates. This is the floor every later rung is measured against, and I expect it to learn but to be brittle: with a single critic there is nothing to cap overestimation. The actor is *defined* by ascending the critic, so it is drawn toward wherever the critic bulges upward, which is exactly where the approximator's error happens to be positive; it seeks those overestimates, generates more data there, and the loop can feed on itself, while the single deterministic target action lets it sit on a narrow spurious peak. I expect this to bite hardest on the higher-dimensional hidden Ant, where the critic has the most room to overestimate and the actor the most directions to chase a spurious peak — pointing the next rung squarely at overestimation control.

```python
class OffPolicyAlgorithm:
    """DDPG — Deep Deterministic Policy Gradient."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            target_q = self.qf1_target(next_obs, next_actions).view(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.qf1(obs, actions).view(-1)
        critic_loss = F.mse_loss(current_q, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            actor_loss = -self.qf1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.qf1_target, self.qf1, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
```
