We want a policy learned entirely from a fixed dataset $D$ of transitions $(s,a,r,s')$ gathered by some behavior policy $\pi_\beta$, with no further environment interaction, and we want it to be genuinely better than $\pi_\beta$. The difficulty is one that has no analogue online: to improve we must estimate the value of actions other than those we have seen, but a value function is only trustworthy on the state-action distribution it was trained on. Standard approximate dynamic programming makes this catastrophic, because its bootstrap target $r(s,a) + \gamma \max_{a'} Q_{\hat\theta}(s',a')$ takes a $\max$ over *all* actions $a'$, including ones $\pi_\beta$ never produced at $s'$. On those out-of-distribution actions the function approximator extrapolates, and the $\max$ operator actively hunts for whichever action the network happens to over-value; the inflated target backs up through the Bellman recursion, and the greedy policy $\pi(s)=\arg\max_a Q_\theta(s,a)$ steers straight toward the over-valued actions. The error feeds itself into a divergence, not a small bias.

Every prior remedy buys safety with a knob. Policy-constraint methods (BCQ, BEAR, BRAC, TD3+BC, AWAC) keep the learned policy near $\pi_\beta$; value-regularization methods (CQL, Fisher-BRC) push $Q$ down on OOD actions and up on dataset actions. Both families work, but both are fundamentally a dial between "improve a lot" and "don't get burned by extrapolation," and every one of them still, somewhere in the loop, evaluates a *learned* $Q$ at an action that was not in the data — BCQ's generative model can emit OOD actions, CQL must query OOD actions precisely to push them down. There is a cleaner-looking family that sidesteps the query entirely: the one-step methods use a SARSA objective, bootstrapping with the dataset's own next action,
$$L(\theta) = \mathbb{E}_{(s,a,s',a')\sim D}\big[\big(r(s,a) + \gamma\, Q_{\hat\theta}(s',a') - Q_\theta(s,a)\big)^2\big],$$
so no OOD action is ever touched. But this is a mean-squared error, so its optimum fits $Q_\theta(s,a)$ to the *mean* of the TD targets over $a'\sim\pi_\beta$; the fixed point is the Bellman equation for $\pi_\beta$, i.e. it learns $Q^{\pi_\beta}$ — pure policy evaluation. Following it with a single improvement step (greedy or advantage-weighted) is safe and simple, but it never iterates the Bellman backup, so it cannot propagate value along a path that no single dataset trajectory walks end to end. On a maze whose dataset is one optimal trajectory among ninety-nine random ones, reaching a good value at the start requires *stitching* fragments from many suboptimal trajectories — value flowing backward across transitions from different trajectories — which is exactly what iterated dynamic programming does and what a single step cannot. The one-step methods collapse there.

So I am caught between a safe-but-insufficient SARSA mean and a powerful-but-dangerous $\max$. The resolution I propose is Implicit Q-Learning (IQL). The crux is this: what the backup actually wants is not the mean over $a'$ and not the unrestricted $\max$, but the maximum over *in-support* actions, $\max_{a':\pi_\beta(a'|s')>0} Q_{\hat\theta}(s',a')$ — a max (so it improves, and iterating it does real dynamic programming) that never reaches an OOD action (so it stays safe). The obstacle is that I cannot compute that restricted max directly: enumerating or sampling actions and querying $Q$ at each puts me right back to evaluating $Q$ at actions outside the data. The reframing that breaks the deadlock is to fix a state $s$ and read $Q_{\hat\theta}(s,a')$, as $a'$ ranges over $\pi_\beta(\cdot|s)$, as a *random variable* whose randomness comes from the action. SARSA's MSE gives its mean; what I want is the maximum over its support. So I need a statistic that sits high in this random variable's distribution and that I can estimate from the sampled in-data actions alone, querying $Q$ only at those actions.

Expectile regression supplies exactly that statistic. The $\tau$-expectile of a random variable $X$ is the minimizer of an asymmetric squared loss,
$$m_\tau = \arg\min_m \mathbb{E}\big[L_2^\tau(x - m)\big], \qquad L_2^\tau(u) = \big|\tau - \mathbf{1}(u<0)\big|\,u^2.$$
For a residual $u = x - m$: if $u>0$ (a sample above the estimate) the weight is $\tau$; if $u<0$ it is $1-\tau$. At $\tau=0.5$ both weights are $\tfrac12$ and this is ordinary MSE, so $m_{0.5}$ is the mean; for $\tau>0.5$ the samples *above* the estimate are weighted more heavily, so to balance the gradient the estimate moves up, and the larger $\tau$ is, the more the upper samples dominate and the higher $m_\tau$ climbs. The load-bearing fact is the limit: for $X$ with bounded support and supremum $x^*$, $\lim_{\tau\to1} m_\tau = x^*$. The minimizer satisfies the expectile balance equation $\tau\,\mathbb{E}[(X-m_\tau)_+] = (1-\tau)\,\mathbb{E}[(m_\tau-X)_+]$; it lies in the convex hull of the support (so never exceeds $x^*$) and is monotone in $\tau$. If the limit stayed below $x^*$ by some $\varepsilon$, there is positive mass above the limit plus $\varepsilon/2$, so $\mathbb{E}[(X-m_\tau)_+]$ stays bounded away from zero while $(1-\tau)\mathbb{E}[(m_\tau-X)_+]\to0$ — contradicting the balance equation. So the upper expectile, in the limit, *is* the in-support maximum, and it is reachable by SGD on in-sample data. I prefer the expectile loss (asymmetric L2) over the analogous quantile loss (asymmetric L1) for a concrete reason: I need only one statistic, the upper tail, not the whole distribution as in distributional RL, and the expectile loss is a one-line reweighting of the MSE the code already runs, which also optimizes a little better than the L1 quantile loss.

The naive move would be to apply the expectile directly to the raw TD residual, $L(\theta)=\mathbb{E}[L_2^\tau(r + \gamma Q_{\hat\theta}(s',a') - Q_\theta(s,a))]$, but this breaks, and the failure dictates the architecture. That target carries *two* sources of randomness: the action $a'\sim\pi_\beta(\cdot|s')$, over which I *want* to be optimistic (the best $a'$ is the improvement signal), and the stochastic transition $s'\sim p(\cdot|s,a)$, over which I emphatically do *not*. An upper expectile rewards high targets indiscriminately, so it would reward a target that is high merely because the environment happened to transition into a lucky good state — conflating "there exists a better action here" with "I got lucky with the dice," and that dynamics-optimism compounds over the horizon into a wildly overoptimistic value. The fix is to separate the two with a dedicated value network $V_\psi(s)$ that takes the expectile over actions only, with the transition held fixed:
$$L_V(\psi) = \mathbb{E}_{(s,a)\sim D}\big[L_2^\tau\big(Q_{\hat\theta}(s,a) - V_\psi(s)\big)\big].$$
Here both $s$ and $a$ come from $D$, so for a given $s$ the only randomness in the regression target is the action, and $V_\psi(s)$ becomes the $\tau$-expectile of $Q$ over dataset actions — optimistic over actions, with no dynamics in it. Then back this up into $Q$ with an *ordinary* MSE that averages the transition honestly:
$$L_Q(\theta) = \mathbb{E}_{(s,a,s')\sim D}\big[\big(r(s,a) + \gamma\, V_\psi(s') - Q_\theta(s,a)\big)^2\big].$$
The MSE is correct precisely because $V_\psi(s')$ has already done the optimistic action-selection; what remains is to average $\gamma V_\psi(s')$ over $s'\sim p(\cdot|s,a)$, and a mean is the right way to average dynamics. The division of labor is clean: $V$ takes the upper expectile over actions, $Q$ takes the mean over transitions, both losses touch only dataset $(s,a,s')$, and no policy and no OOD action appear anywhere in value training.

This is provably multi-step dynamic programming, not merely a safe per-step heuristic. Define the fixed-point objects with $\mu=\pi_\beta$ and $E^\tau$ the $\tau$-expectile: $V_\tau(s) = E^\tau_{a\sim\mu(\cdot|s)}[Q_\tau(s,a)]$ and $Q_\tau(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}[V_\tau(s')]$. First, $\tau_1<\tau_2 \Rightarrow V_{\tau_1}\le V_{\tau_2}$ pointwise: unrolling the recursion, each substitution replaces a $V_{\tau_1}$ by its definition and bumps one expectile from $\tau_1$ up to $\tau_2$, and the backup $r+\gamma\mathbb{E}_{s'}[\cdot]$ is monotone, so the inequality carries all the way down — higher $\tau$ gives uniformly higher value, the signature of a policy-improvement step. Second, $V_\tau(s)\le \max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$, where $Q^*$ is the optimal value constrained to in-support actions, because an action expectile is no larger than the in-support max and both backups are monotone $\gamma$-contractions, so iterating the smaller operator stays below the fixed point. Combining the monotone increase with the expectile-limit lemma, the operators $T_\tau$ converge pointwise to the in-support max operator and
$$\lim_{\tau\to1} V_\tau(s) = \max_{a:\pi_\beta(a|s)>0} Q^*(s,a).$$
So the method spans a spectrum: $\tau=0.5$ recovers SARSA (policy evaluation of $\pi_\beta$) and $\tau\to1$ recovers Q-learning restricted to in-support actions (true multi-step dynamic programming up to the constrained optimum), with $\tau$ the dial. Because larger $\tau$ leans on extreme upper residuals it is a harder, higher-variance optimization, so I treat $\tau$ as a hyperparameter rather than slamming it to 1. Two stabilizers carry over from value-based RL: the bootstrapped $Q$ in $L_V$ is a slowly-moving Polyak target $\hat\theta\leftarrow(1-\alpha)\hat\theta+\alpha\theta$, and since bootstrapping over-estimates I use clipped double-Q, $Q=\min(Q_1,Q_2)$, wherever a single $Q$ value is needed.

Value training is deliberately policy-free, so I still need to extract a policy, and the extraction must obey the same commandment: never query $Q$ at an unseen action. That rules out $\arg\max_a Q$ (it searches OOD actions) and DDPG-style $\nabla_a Q$ ascent (it evaluates $Q$ at the policy's possibly-OOD actions). What I *can* do is reweight the dataset's own actions, via advantage-weighted regression: the KL-constrained improvement problem $\max_\pi \mathbb{E}_{a\sim\pi}[A(s,a)]$ subject to $\mathrm{KL}(\pi\|\pi_\beta)\le\varepsilon$ has closed-form solution $\pi^*(a|s)\propto\pi_\beta(a|s)\exp(A(s,a)/\lambda)$, and projecting it onto a parametric policy by weighted maximum likelihood gives
$$L_\pi(\phi) = \mathbb{E}_{(s,a)\sim D}\big[\exp\big(\beta(Q_{\hat\theta}(s,a) - V_\psi(s))\big)\,\log\pi_\phi(a|s)\big],$$
with advantage $A=Q-V$ and inverse temperature $\beta$. This only ever evaluates dataset actions — it reweights observed $(s,a)$ pairs by how advantaged they are — so it queries nothing unseen, inherits an implicit stay-near-$\pi_\beta$ constraint, and makes a good initialization for online finetuning. $\beta\to0$ is behavioral cloning, $\beta\to\infty$ greedily concentrates on the highest-advantage actions; one numerical guard clips the weight $\exp(\beta A)$ to at most $100$ so a few huge-advantage transitions cannot dominate. Per gradient step I update $V$ by expectile regression from the target critic, update the policy by advantage-weighted regression, update $Q$ by MSE onto $r+\gamma V(s')$, and Polyak-update the target critic; the policy never feeds back into value training, so extraction decouples cleanly from value learning. Canonically: 2-layer 256-unit ReLU MLPs, Adam at $3\times10^{-4}$ for all three nets with cosine decay on the actor learning rate, $\gamma=0.99$, Polyak $\alpha=0.005$, batch 256, 1M steps; $\tau=0.7,\beta=3.0$ for MuJoCo locomotion, $\tau=0.9,\beta=10.0$ for Ant Maze, $\tau=0.7,\beta=0.5$ for Kitchen/Adroit, with a Gaussian policy of state-independent std and tanh-bounded mean.

```python
import copy
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__(); self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__(); self.net = mlp([obs_dim, *hidden, 1])
    def forward(self, s):
        return self.net(s).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.mean = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))      # state-independent std
    def dist(self, s):
        mean = torch.tanh(self.mean(s))                        # official code bounds the Gaussian mean
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
    def act(self, s):
        return torch.tanh(self.mean(s))


def expectile_loss(diff, tau):                                 # asymmetric L2
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff ** 2).mean()


def update_v(value, target_critic, batch, tau):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.minimum(q1, q2)                              # clipped double-Q
    return expectile_loss(q - value(s), tau)                   # V <- tau-expectile of Q over data actions


def update_actor(policy, target_critic, value, batch, beta):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.minimum(q1, q2)
        adv = q - value(s)
        weight = torch.clamp(torch.exp(beta * adv), max=100.0)
    return -(weight * policy.log_prob(s, a)).mean()            # advantage-weighted regression


def update_q(critic, value, batch, discount):
    s, a, r, s2, mask = (batch["obs"], batch["act"], batch["rew"],
                         batch["obs2"], batch["mask"])          # mask = 1 - done
    with torch.no_grad():
        target_q = r + discount * mask * value(s2)             # honest MSE; V already picked the action
    q1, q2 = critic(s, a)
    return ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()


def polyak(critic, target_critic, alpha):
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        tp.data.mul_(1 - alpha).add_(alpha * p.data)


def update(batch, critic, target_critic, value, policy, opts, hp):
    v_loss = update_v(value, target_critic, batch, hp["tau"])
    opts["v"].zero_grad(); v_loss.backward(); opts["v"].step()

    a_loss = update_actor(policy, target_critic, value, batch, hp["beta"])
    opts["pi"].zero_grad(); a_loss.backward(); opts["pi"].step()

    q_loss = update_q(critic, value, batch, hp["discount"])
    opts["q"].zero_grad(); q_loss.backward(); opts["q"].step()

    polyak(critic, target_critic, hp["alpha"])


def train_offline(dataset, critic, value, policy, hp, steps=int(1e6), batch_size=256):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q":  torch.optim.Adam(critic.parameters(), lr=3e-4),
        "v":  torch.optim.Adam(value.parameters(),  lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4),
    }
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opts["pi"], steps)
    for _ in range(steps):
        update(dataset.sample(batch_size), critic, target_critic, value, policy, opts, hp)
        sched.step()

# hp = dict(tau=0.7, beta=3.0, discount=0.99, alpha=0.005)   # locomotion
# hp = dict(tau=0.9, beta=10.0, discount=0.99, alpha=0.005)  # antmaze
```
