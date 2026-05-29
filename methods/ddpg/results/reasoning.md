OK, let me think this through from scratch. I have a robot arm in simulation with several joints, and at every timestep I have to emit a vector of torques, real numbers, one per joint. Same story for a car with steering and throttle, or for a walking body with many actuators. I want to learn a controller from reward alone, model-free, and I want the action choice to be cheap enough to run online. So the question is: what learning algorithm gives me a good continuous-action policy without solving a control problem inside every environment step?

The obvious thing to reach for is the deep Q-network recipe. It learns an action-value function `Q(s,a|θ)` with a neural net, stores transitions in replay, and uses a target network so the Bellman target does not move on the same step as the regressor. But its control rule is greedy: `a = argmax_a Q(s,a)`. Its target is greedy too: `y = r + γ max_{a'} Q(s',a')`. On a small discrete action set, the network emits one value per action and the max is just a reduction over a short vector.

Now put continuous torques in there. `a ∈ R^m`. A neural `Q(s,a)` does not hand me an `argmax`. I could run gradient ascent over `a`, or random shooting, or another optimizer, but then I am solving a non-convex inner problem for every action and for every bootstrap target in a minibatch. That defeats the point of a real-time controller.

Discretization looks tempting until I count. Seven joints with only `{−k, 0, k}` per joint already gives `3^7 = 2187` actions, before any fine torque resolution. The count grows exponentially with the number of actuators, and the grid also throws away metric structure: `0.30` and `0.31` are nearby torques, but two action labels do not know that unless the value function relearns it. Discretization is a dead end.

The max over actions is the thing I need to remove. It appears because the value function is being asked to produce both an evaluation and the greedy action. I can split those jobs. Keep a critic `Q(s,a)`, but introduce another differentiable function `μ(s)` whose job is to emit the action I will actually use. Then acting is one forward pass. The training question becomes: how do I move the parameters of `μ` so that `Q(s, μ(s))` increases?

The critic is differentiable in the action, and the actor is differentiable in its parameters. Using the policy-Jacobian-then-action-gradient convention, the local chain rule gives

`∇_θ Q(s, μ_θ(s)) = [∇_θ μ_θ(s)] [∇_a Q(s,a)]_{a=μ_θ(s)}`.

The action-gradient says which direction in action space would increase value at this state; the policy Jacobian pulls that direction back into parameter space. So the inner action optimizer is being amortized: instead of solving `argmax_a Q(s,a)` at every timestep, I train a network whose output moves toward actions the critic rates highly.

I still need to check that this is not just a heuristic for a fixed batch of states. Start from the deterministic value functions:

`V^μ(s) = Q^μ(s, μ_θ(s))`

and

`Q^μ(s,a) = E_{r,s'}[r + γ V^μ(s') | s,a]`.

Differentiate `V^μ(s)` with respect to `θ`. There are two pieces. The current action changes because `μ_θ(s)` changes, and future values change because all future actions are also produced by `μ_θ`. Holding the current action slot of `Q^μ(s,a)` open for the first piece gives

`∇_θ V^μ(s) = [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} + [∇_θ Q^μ(s,a)]_{a=μ_θ(s)}`.

The second term follows only the downstream dependence of `Q^μ` on the policy parameters while holding the current action fixed:

`[∇_θ Q^μ(s,a)]_{a=μ_θ(s)} = γ E_{s'~p(·|s,μ_θ(s))}[∇_θ V^μ(s')]`.

That is the useful recursion:

`∇_θ V^μ(s) = [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} + γ E_{s'}[∇_θ V^μ(s')]`.

The derivative of how the next-state distribution changes with the current action is not missing; it is inside `∇_a Q^μ(s,a)`, because `Q^μ` is the whole action-value function. The remaining recursive term is only the later dependence of future actions on `θ`. Unroll it:

`∇_θ V^μ(s_0) = E[Σ_{t≥0} γ^t [∇_θ μ_θ(s_t)] [∇_a Q^μ(s_t,a)]_{a=μ_θ(s_t)} | s_0]`.

Integrating over the start-state distribution gives

`∇_θ J = ∫ ρ^μ(s) [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} ds`,

where `ρ^μ(s)=Σ_{t≥0} γ^t P(s_t=s | μ)` is the unnormalized discounted visitation measure under the start-state distribution. If I instead sample from the normalized distribution `d^μ=(1-γ)ρ^μ`, the same expression has a constant `1/(1-γ)` in front, and that constant is absorbed by the step size. The key structural fact survives either convention: the policy gradient is an average over states only. There is no action integral.

That absence of an action integral matters for off-policy learning. A stochastic policy-gradient estimator has a sampled action and a score term, so off-policy data normally brings in action likelihood ratios. Here the action in the target policy is just `μ(s)`. If I collect data with a noisy behavior policy `β`, the critic can still learn `Q^μ` from transitions by using the target action `μ(s')` in the Bellman backup; the environment transition `(s,a,r,s')` is valid regardless of why action `a` was chosen. For the actor update, sampling states from replay changes the state weighting, and the formal off-policy actor-critic step drops a downstream term involving `∇_θ Q^μ`. So this is a behavior-weighted approximation, not a magic recovery of the original start-state gradient. But there is still no action-space importance ratio to pay. I need coverage of the states the learned policy will care about, not a high-variance correction over continuous actions.

I keep an actor `μ(s|θ^μ)` and a critic `Q(s,a|θ^Q)`. The deterministic Bellman equation has no action expectation at the next state:

`Q^μ(s,a) = E_{r,s'}[r + γ Q^μ(s', μ(s'))]`.

For a sampled transition with terminal flag `d`, the regression target has to stop bootstrapping at true terminals:

`y = r + γ(1-d) Q(s', μ(s'))`.

The critic minimizes mean-squared error to that target. The actor takes the chain-rule step that increases `Q(s, μ(s))`. In code, because optimizers minimize losses, the actor loss must be the negative value estimate: `loss_pi = -Q(s, μ(s)).mean()`. Minimizing that is gradient ascent on the critic.

A naive neural actor-critic still worries me. The critic target contains the same kind of self-reference that made deep Q-learning unstable: if I compute `y` with the live critic, every critic step moves both the prediction and the target. Replay fixes the data correlation, but the target can still chase itself. The discrete-action value recipe already has the right repair: compute the bootstrap with a target network. Here the target depends on two learned functions, the critic and the actor, so both need target copies. The stable target becomes

`y_i = r_i + γ(1-d_i) Q_targ(s'_i, μ_targ(s'_i))`.

Hard-copying target parameters every so often would work in principle, but the actor and critic are coupled: the critic target uses the target actor, and the actor update uses the live critic. Smooth tracking makes the regression target drift instead of jump. I use Polyak averaging with `ρ` close to one:

`θ_targ ← ρ θ_targ + (1-ρ) θ`, with `ρ=0.995`.

Equivalently, the target moves by `τ=1-ρ=0.005` of the live-target gap each update. The exact coefficient matters in code: multiplying the target by `polyak` and adding `(1-polyak)` times the live parameter implements the formula above.

Exploration is separate from the deterministic target policy because learning is off-policy. During data collection I can use `β(s)=μ(s)+noise`. In heavy physical systems I might choose a temporally correlated Ornstein-Uhlenbeck perturbation; with zero mean and unit time step its discretization is `x ← x + θ_ou(-x) + σξ`, which is the sign I want because the drift pulls the noise state back toward zero. For the PyTorch implementation, I keep the behavior simpler: a long initial phase of uniform random actions, then independent Gaussian action noise, `a ← μ(s) + noise_scale * randn(act_dim)`, clipped to the symmetric action bounds. That is faithful to the off-policy argument: the noise only changes which transitions enter replay, while the critic target and actor loss remain about the clean deterministic policy.

The actor must output actions inside the environment bounds, so its final nonlinearity is `tanh`, scaled by `act_limit` under the usual symmetric-bound assumption; hidden layers use ReLU MLP blocks. The critic estimates a scalar for a state-action pair, so it concatenates observation and action and sends the combined vector through an MLP ending in one output, squeezed to shape `(batch,)`. The target networks are deep copies of the live actor-critic and have `requires_grad=False` because they move only by Polyak averaging. Adam updates both live networks. The code defaults are `hidden_sizes=(256,256)`, `pi_lr=q_lr=1e-3`, replay size `10^6`, minibatch size `100`, `start_steps=10000`, `update_after=1000`, `update_every=50`, `act_noise=0.1`, `gamma=0.99`, and `polyak=0.995`.

I seed the RNGs, build the environment and the actor-critic, copy it into targets, and store every transition in a FIFO replay buffer. While `t <= start_steps`, I act uniformly at random so the buffer is not a narrow slice of state space. After that, I act with `μ(s)` plus Gaussian noise. Once at least `update_after` interactions exist, every `update_every` environment steps, I run `update_every` gradient steps so the environment-step to gradient-step ratio stays one. The critic step minimizes `((Q(s,a)-backup)**2).mean()` with `backup = r + γ(1-d)Q_targ(s', μ_targ(s'))` computed under `torch.no_grad()`. The actor step freezes the critic parameters for efficiency and minimizes `-Q(s, μ(s)).mean()`. Then every target parameter gets the Polyak update.

The causal chain is tight: continuous actions break the cheap greedy max in value-based control; discretization explodes; a deterministic actor amortizes the argmax; the deterministic policy-gradient theorem justifies the actor step as policy-Jacobian times critic action-gradient over discounted state visitation; off-policy replay is usable because there is no action integral or action likelihood ratio, though the replay state distribution changes the weighting; Bellman regression needs the terminal mask; deep bootstrapping needs target actor and target critic copies; Polyak averaging keeps those targets slow; and the behavior policy gets explicit exploration noise while the learned policy stays deterministic. The implementation is the small PyTorch version of exactly that.

```python
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)
        self.act_limit = act_limit
    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
    def forward(self, obs, act):
        return torch.squeeze(self.q(torch.cat([obs, act], dim=-1)), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def ddpg(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,
         start_steps=10000, update_after=1000, update_every=50,
         act_noise=0.1, max_ep_len=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = ac.q(o, a)
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ
        return ((q - backup) ** 2).mean()

    def compute_loss_pi(data):
        o = data['obs']
        return -ac.q(o, ac.pi(o)).mean()

    def update(data):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = False
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        for p in ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    total_steps = steps_per_epoch * epochs
    o, ep_len = env.reset(), 0
    for t in range(total_steps):
        a = get_action(o, act_noise) if t > start_steps else env.action_space.sample()
        o2, r, d, _ = env.step(a)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            o, ep_len = env.reset(), 0
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                update(replay_buffer.sample_batch(batch_size))
```
