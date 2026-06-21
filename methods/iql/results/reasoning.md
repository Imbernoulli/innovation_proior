Let me start from the exact thing that goes wrong in offline RL, because the whole method has to be built around avoiding it. I have a fixed dataset D of transitions (s,a,r,s') collected by some behavior policy π_β, and no way to interact with the environment. I want a policy better than π_β. The standard tool is approximate dynamic programming: minimize the TD error

  L_TD(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ max_{a'} Q_θ̂(s',a') − Q_θ(s,a) )² ],

with a target network θ̂ and a greedy policy π(s) = argmax_a Q_θ(s,a). Online this is fine. Offline it's poison, and I want to name exactly why. The bootstrap target contains max_{a'} Q_θ̂(s',a'). That max ranges over *all* actions a', including actions that never appear in D for state s'. On those out-of-distribution actions the function approximator has no data — it extrapolates, and the extrapolation is almost always *upward*, because the max operator actively hunts for whichever action the network happens to over-value. So the target is inflated, the inflation backs up through the Bellman recursion, and the policy — being the argmax — steers straight toward the over-valued actions. The error feeds itself. This isn't a small bias; it's a divergence.

So everyone adds a knob. Constrain the policy to stay near π_β (BCQ fits a generative model of dataset actions and only maxes over sampled candidates; TD3+BC adds a behavioral-cloning term; AWAC uses an implicit KL constraint). Or regularize the value: CQL adds a term that pushes Q down on OOD actions and up on dataset actions. Both families work, but both are fundamentally a dial between "improve a lot" and "don't get burned by extrapolation," and you have to tune where you sit on it per dataset. And every one of them still, at some point, evaluates a *learned* Q at an action that wasn't in the data — BCQ's generative model can still emit OOD actions, CQL has to query OOD actions precisely to push them down. The OOD query is still in the loop.

There's a cleaner-looking family that sidesteps the query entirely: the one-step methods. Take the SARSA objective — bootstrap with the *dataset's* next action a', not a max:

  L(θ) = E_{(s,a,s',a')~D}[ ( r(s,a) + γ Q_θ̂(s',a') − Q_θ(s,a) )² ].

Now no OOD action is ever touched: every a and a' came from D. This is mean-squared error, so its optimum fits Q_θ(s,a) to the *mean* of the TD targets — meaning the fixed point satisfies Q_{θ*}(s,a) ≈ r(s,a) + γ E_{s'~p, a'~π_β}[Q_θ̂(s',a')]. That's the Bellman equation for π_β: this objective learns Q^{π_β}, the value of the behavior policy. Then extract a policy greedily, or by advantage weighting, in one shot (Onestep RL, AWR). Safe and simple. But it's only *policy evaluation* of π_β followed by a single improvement step. It never iterates the Bellman backup, so it can't propagate value along a path that no single dataset trajectory walks end-to-end. Picture a maze whose dataset is one optimal trajectory buried in ninety-nine random ones: to get a good value at the start you need to *stitch* fragments from many suboptimal trajectories, which requires the value of a good downstream state to flow backward across transitions from *different* trajectories — exactly what iterated dynamic programming does and what a single step cannot. On those stitching tasks the one-step methods collapse.

So I'm caught between two safe-but-insufficient SARSA-based ideas and a powerful-but-dangerous max. Let me look hard at what the SARSA objective is actually missing, because I'd love to keep its in-sample safety and somehow inject the improvement. SARSA's MSE learns the *mean* of r + γ Q_θ̂(s',a') over the dataset's a' (and over the stochastic s'). The mean over dataset actions is the value of the average dataset action — that's why I only get Q^{π_β}. But what I *want* in the backup is not the mean over a'; I want the value of the *best in-support action*: something like

  r(s,a) + γ max_{a' : π_β(a'|s')>0} Q_θ̂(s',a') − Q_θ(s,a),

a max restricted to actions the behavior policy could actually produce at s'. That restriction is the whole game: it's a max (so it improves, and iterating it does real dynamic programming) but it never reaches an OOD action (so it stays safe). The problem is that I can't *compute* that restricted max directly — to take a max over in-support a' I'd have to enumerate or sample actions and query Q at each, and the moment I sample an action and query Q I'm back to evaluating Q at actions outside the data. I need the max over in-support actions *without ever querying Q at any specific a'*.

Reframe it. Fix a state s. As a' ranges over the behavior distribution π_β(·|s), the quantity Q_θ̂(s,a') is a *random variable* — randomness coming from the action. The mean of that random variable is what SARSA's MSE gives me. The *maximum* over the support of that random variable is what I want. So I need a statistic of this action-induced random variable that sits high in its distribution — ideally at the top of its support — and I need to estimate it from samples (the dataset actions at s) without evaluating Q anywhere except at those sampled, in-data actions.

Mean regression gives the mean. What gives the upper tail? Expectile regression. The τ-expectile of a random variable X is the minimizer of an *asymmetric* squared loss,

  m_τ = argmin_m E[ L_2^τ(x − m) ],   L_2^τ(u) = |τ − 1(u<0)|·u².

Let me make sure this does what I think. For a residual u = x − m: if u > 0 (a sample above my estimate) the weight is |τ − 0| = τ; if u < 0 (below) the weight is |τ − 1| = 1 − τ. At τ = 0.5 both weights are ½ and this is just MSE, so m_{0.5} is the mean. For τ > 0.5 the positive residuals — the samples *above* the estimate — are weighted more heavily, so to balance the gradient the estimate has to move *up*. The bigger the τ, the more the upper samples dominate, and m_τ climbs toward the top of the distribution. And as τ → 1 it should approach the supremum of the support. Let me actually check the signs through the first-order condition, because that is the load-bearing claim.

Lemma: for a random variable X with bounded support and supremum x*, lim_{τ→1} m_τ = x*. The minimizer satisfies the expectile balance equation

  τ E[(X − m_τ)_+] = (1 − τ) E[(m_τ − X)_+].

The minimizer lies in the closed convex hull of the support, so it never exceeds x*. It is also monotone in τ: increasing τ raises the penalty on underestimates and lowers the penalty on overestimates, so τ₁ < τ₂ gives m_{τ₁} ≤ m_{τ₂}. Now suppose the limit stayed below x* by some ε. Then there is positive probability mass above the limit plus ε/2, so E[(X − m_τ)_+] stays bounded away from zero along large τ, while (1 − τ)E[(m_τ − X)_+] goes to zero because the support is bounded. That contradicts the balance equation. So the monotone limit must be x*. The upper expectile, in the limit, is the in-support maximum. That's the operator I wanted, expressed as a regression I can do with SGD on in-sample data.

The conditional version is what I need — I want a *state-conditional* expectile, a network m_τ(s):

  argmin_{m(·)} E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − m(s) ) ].

This trains a value network V(s) to be the τ-expectile of Q over the dataset actions at s, using only (s,a) pairs from D, querying Q only at dataset actions. As τ → 1, V(s) → max over in-support actions of Q. The improvement is in, the OOD query is out.

But wait — before I separate things into a V network, let me try the most direct thing and watch it break, because the failure tells me why the V network has to exist. The direct move is to just swap SARSA's MSE for the expectile loss on the *same* TD residual:

  L(θ) = E_{(s,a,s',a')~D}[ L_2^τ( r(s,a) + γ Q_θ̂(s',a') − Q_θ(s,a) ) ].

This takes an upper expectile of r + γ Q_θ̂(s',a'). The trouble: that target carries *two* sources of randomness at once. One is the action a' ~ π_β(·|s') — the randomness I *want* to take the upper expectile over, because the best a' is the improvement signal. The other is the stochastic transition s' ~ p(·|s,a) — the randomness I emphatically do *not* want to be optimistic about. An upper expectile rewards high targets indiscriminately, so it will reward a target that's high merely because the environment *happened* to transition into a lucky good state s', even when there was no better action — it conflates "there exists a better action here" with "I got lucky with the dice." Being optimistic over the dynamics is exactly the kind of optimism that, compounded over a long horizon, produces a wildly overoptimistic value. So I must take the expectile over actions *only*, and average honestly over transitions.

Separate the two. Introduce a value network V_ψ(s) whose job is purely the action-expectile, with the transition fixed:

  L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − V_ψ(s) ) ].

Here both s and a come from D, the only randomness in the regression target for a given s is the action a, so V_ψ(s) becomes the τ-expectile of Q over dataset actions — the optimistic-over-actions quantity, with no dynamics in it at all. Then back this up into Q with an *ordinary* MSE that averages over the next-state transition honestly:

  L_Q(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ V_ψ(s') − Q_θ(s,a) )² ].

The MSE here is correct precisely because V_ψ(s') has already done the optimistic action-selection; what remains is to average γ V_ψ(s') over s' ~ p(·|s,a), and a mean (MSE) is the right way to average the dynamics — no lucky-sample optimism. Notice the lovely division of labor: V takes the upper expectile over actions; Q takes the mean over transitions; and both losses touch only dataset (s,a,s'). No policy appears anywhere in value training, and no OOD action is ever queried.

Let me confirm this actually does multi-step dynamic programming and converges to the right thing, not just that each step is safe. Define the fixed-point objects recursively with μ = π_β as the action distribution and E^τ denoting the τ-expectile:

  V_τ(s) = E^τ_{a~μ(·|s)}[ Q_τ(s,a) ],   Q_τ(s,a) = r(s,a) + γ E_{s'~p(·|s,a)}[ V_τ(s') ].

First, monotonicity in τ: I claim τ₁ < τ₂ ⟹ V_{τ₁}(s) ≤ V_{τ₂}(s) for every s. Expand V_{τ₁} and use, at each layer, that the τ₁-expectile is ≤ the τ₂-expectile (the lemma's monotonicity) and that the backup operator r + γ E_{s'}[·] is monotone (it preserves inequalities):

  V_{τ₁}(s) = E^{τ₁}_{a~μ}[ r(s,a) + γ E_{s'}[V_{τ₁}(s')] ]
           ≤ E^{τ₂}_{a~μ}[ r(s,a) + γ E_{s'}[V_{τ₁}(s')] ]
           = E^{τ₂}_{a~μ}[ r(s,a) + γ E_{s'} E^{τ₁}_{a'~μ}[ r(s',a') + γ E_{s''}[V_{τ₁}(s'')] ] ]
           ≤ E^{τ₂}_{a~μ}[ r(s,a) + γ E_{s'} E^{τ₂}_{a'~μ}[ r(s',a') + γ E_{s''}[V_{τ₁}(s'')] ] ]
           ≤ … ≤ V_{τ₂}(s),

unrolling the recursion: each substitution replaces a V_{τ₁} by its definition and bumps one more expectile from τ₁ up to τ₂, and the chain of monotone backups carries the inequality all the way down. So higher τ gives a uniformly higher value — exactly the behavior of a *policy improvement* step, which is the sign that raising τ is doing real improvement, not just curve-fitting.

Next, an upper bound. For any τ, V_τ(s) ≤ max_{a : π_β(a|s)>0} Q*(s,a), where Q* is the optimal value *constrained to in-support actions*,

  Q*(s,a) = r(s,a) + γ E_{s'}[ max_{a' : π_β(a'|s')>0} Q*(s',a') ].

This holds by comparing Bellman operators. For any fixed bounded Q, the action expectile is no larger than the maximum over in-support actions, so the expectile backup is pointwise no larger than the in-support optimality backup; both backups are monotone γ-contractions under the idealized exact-fit assumptions. Iterating the smaller operator from the same bounded initialization therefore stays below the fixed point Q*. Finally combine the two directions: the operators T_τ increase with τ and, by the expectile-limit lemma, converge pointwise to the in-support max operator T*. A monotone bounded sequence of γ-contraction fixed points converges to the fixed point of the limiting operator, giving

  lim_{τ→1} V_τ(s) = max_{a : π_β(a|s)>0} Q*(s,a).

So the method spans a whole spectrum: at τ = 0.5 the expectile is the mean and I recover SARSA — pure policy evaluation of π_β — and as τ → 1 I recover Q-learning restricted to in-support actions, i.e. true multi-step dynamic programming up to the constrained optimum. τ is the dial between them. Larger τ approximates the max better but, because it leans on the extreme upper residuals, is a harder, higher-variance optimization, so I'll treat τ as a hyperparameter rather than just slamming it to 1.

A couple of stabilizers I'll carry over from value-based RL. The bootstrapped Q in L_V should be a slowly-moving target network Q_θ̂ (Polyak: θ̂ ← (1−α)θ̂ + αθ), so V chases a stable target. And bootstrapping over-estimates, so I'll use clipped double-Q: maintain two critics and take Q = min(Q1, Q2) wherever I need a single Q value (in the V-loss and in the policy step). I'll choose the expectile loss (asymmetric L2) over the analogous quantile loss (asymmetric L1) for a concrete reason: I only need *one* statistic, the upper expectile — I'm not estimating the whole distribution the way distributional RL does — and the expectile loss is a one-line reweighting of the MSE that RL code already runs, which also turned out to optimize a little better than the L1 quantile loss.

Now I have a near-optimal in-support Q (and V), but no policy — the value training was deliberately policy-free. I need to extract one, and the extraction must obey the same commandment: never query Q at an unseen action. So I cannot do argmax_a Q (that searches OOD actions) and I cannot do a DDPG-style ∇_a Q ascent (that evaluates Q at the policy's possibly-OOD actions). What I *can* do is reweight the dataset's own actions. This is the advantage-weighted regression idea: the KL-constrained improvement problem max_π E_{a~π}[A(s,a)] subject to KL(π‖π_β) ≤ ε has the closed-form solution π*(a|s) ∝ π_β(a|s)·exp(A(s,a)/λ), and projecting that onto a parametric policy by weighted maximum likelihood gives

  L_π(φ) = E_{(s,a)~D}[ exp( β·(Q_θ̂(s,a) − V_ψ(s)) )·log π_φ(a|s) ],

with advantage A = Q − V and inverse temperature β. This only ever evaluates dataset actions — it just *reweights* observed (s,a) pairs by how advantaged they are — so it queries no OOD action, and it inherits an implicit "stay near π_β" constraint, which is exactly what you want offline and also makes the policy a good starting point for later online finetuning. β → 0 makes it behavioral cloning (safe, copies the data); β → ∞ makes it greedily concentrate on the highest-advantage actions. One numerical guard: a few transitions can have huge advantages and their exp-weights would dominate the loss, so clip the weight exp(βA) to at most 100.

The pieces compose into two stages, and the key structural fact is that the policy never feeds back into value training — V and Q are learned with no reference to π_φ — so extraction can run concurrently with, or entirely after, the value learning. Per gradient step I update V from the target critic by expectile regression, update the policy by advantage-weighted regression, update Q by MSE onto r + γ V(s'), and Polyak-update the target critic.

Let me write it as real code. The networks first — a double critic, a value net, and a Gaussian policy with a state-independent standard deviation:

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.log_std = nn.Parameter(torch.zeros(act_dim))   # state-independent std
    def dist(self, s):
        mean = torch.tanh(self.mean(s))                # official JAX code bounds the Gaussian mean
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
```

The expectile loss — the asymmetric square that turns mean regression into upper-tail regression:

```python
def expectile_loss(diff, tau):
    # diff = Q_target - V ; positive residual (Q above V) weighted by tau,
    # negative by (1 - tau). tau>0.5 pulls V up toward the in-support max of Q.
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff ** 2).mean()
```

The three updates and the target sync, in the order V → policy → Q → Polyak:

```python
def update_v(value, target_critic, batch, tau):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.min(q1, q2)                       # clipped double-Q
    v = value(s)
    return expectile_loss(q - v, tau)               # V <- tau-expectile of Q over dataset actions


def update_actor(policy, target_critic, value, batch, beta):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.min(q1, q2)
        v = value(s)
        weight = torch.clamp(torch.exp(beta * (q - v)), max=100.0)   # exp(beta * advantage), capped
    log_prob = policy.log_prob(s, a)
    return -(weight * log_prob).mean()              # advantage-weighted behavioral cloning


def update_q(critic, value, batch, discount):
    s, a, r, s2, mask = (batch["obs"], batch["act"], batch["rew"],
                         batch["obs2"], batch["mask"])   # mask = 1 - done
    with torch.no_grad():
        target_q = r + discount * mask * value(s2)  # honest MSE target: V already picked the action
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
```

and the offline training loop that just samples the static dataset and applies this update:

```python
def train_offline(dataset, critic, value, policy, hp, steps=int(1e6), batch_size=256):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q":  torch.optim.Adam(critic.parameters(), lr=3e-4),
        "v":  torch.optim.Adam(value.parameters(),  lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4),
    }
    # cosine-decay the actor lr over training; clipped double-Q throughout
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opts["pi"], steps)
    for _ in range(steps):
        batch = dataset.sample(batch_size)          # static dataset, no env interaction
        update(batch, critic, target_critic, value, policy, opts, hp)
        sched.step()
hp_locomotion = dict(tau=0.7, beta=3.0, discount=0.99, alpha=0.005)
hp_antmaze = dict(tau=0.9, beta=10.0, discount=0.99, alpha=0.005)
hp_kitchen_adroit = dict(tau=0.7, beta=0.5, discount=0.99, alpha=0.005)
```

The chain, end to end: offline, the Q-learning max over a' queries out-of-distribution actions and overestimates, so the policy chases the error; SARSA's in-sample MSE is safe but only learns Q^{π_β} and does no improvement; one-step methods inherit that safety but can't iterate dynamic programming and so can't stitch. What I actually want is a max over *in-support* actions — improvement without OOD queries — but I can't compute it by sampling actions and querying Q. Read Q over the behavior actions as a per-state random variable: its mean is what SARSA gives, its in-support max is what I want, and the τ-expectile (asymmetric-L2 regression) estimates the upper tail of that random variable from in-sample actions alone, reaching the support's sup as τ → 1. Applying the expectile to the raw TD residual would wrongly be optimistic about lucky stochastic transitions, so split it: a value net V takes the upper expectile over actions (Q_θ̂(s,a) − V regressed asymmetrically), and Q is backed up onto r + γ V(s') by honest MSE that averages the dynamics. This is provably multi-step dynamic programming — V_τ is monotone in τ, bounded by the in-support optimum, and converges to it as τ → 1, spanning SARSA (τ=0.5) to constrained Q-learning (τ→1). With clipped double-Q and a Polyak target for stability, the value training is entirely policy-free and never touches an unseen action; the policy is then extracted by advantage-weighted regression exp(β(Q−V))·log π_φ over dataset actions only — improvement-with-an-implicit-constraint, still no OOD query — which decouples cleanly from value learning and is exactly what makes the result a good initialization for online finetuning.
