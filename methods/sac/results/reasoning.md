Let me start from what's actually hurting, on continuous-control problems with neural-net policies. Two things, and they pull in opposite directions. On one side, the algorithms I trust to be stable are on-policy — policy gradients, TRPO, PPO, A3C. They work, but every batch of experience I collect gets used for exactly one gradient step and then thrown away, because the gradient is only valid for the policy that generated the data. So I pay millions of environment steps for tasks that aren't even that hard, and it gets worse as the task scales. On the other side, off-policy methods can reuse a replay buffer — that's where the sample efficiency lives. The cleanest off-policy story for continuous actions is DDPG. But every time I touch DDPG I get burned: nudge the learning rate or the exploration noise and it goes from "solves the task" to "flatlines," and on the genuinely hard high-dimensional tasks — Ant, the 21-DoF Humanoid — it often just makes no progress at all, while a slower on-policy method still crawls forward. So the thing I want is a single algorithm that is off-policy (so I can reuse data) and stable across seeds and basically untuned. Those two requirements are exactly the ones that, historically, you have to trade off against each other.

So let me look hard at why DDPG is brittle, because if I understand the mechanism I'll know what to change. DDPG is really Q-learning wearing an actor. In continuous actions you can't compute max_a Q(s,a) — there's no enumerating the actions — so DDPG trains a deterministic policy μ_θ(s) whose entire job is to approximate argmax_a Q(s,a). The Q-function is learned by the usual Bellman regression toward r + γ Q_targ(s', μ_targ(s')), and the actor is pushed uphill on Q by the chain rule, ∇_θ Q(s, μ_θ(s)) = ∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ_θ(s). Clean. But notice what the determinism costs. A deterministic policy explores nothing on its own — for any state it outputs one action — so I have to bolt exploration on from outside, Ornstein-Uhlenbeck or Gaussian noise, and now I have a noise process to tune too. And the actor is chasing a moving, noisy Q-function that it is simultaneously shaping; the actor collapses onto whatever ridge of Q looks tallest right now, the Q-function overestimates because it's being maximized through this actor, and the whole coupled system has a tiny basin of stable hyperparameters. The brittleness isn't bad luck; it's structural. A point-mass actor sitting on top of a value function it's also distorting.

That makes me suspicious of determinism itself. What if the actor were stochastic, and not just stochastic-with-noise-added, but stochastic because being stochastic is *valued*? If the policy spread its mass over comparably-good actions instead of committing to one, it would explore on its own, it wouldn't collapse the instant one action looks marginally better, and — intuitively — a value function defined over a broad policy is a smoother target than one defined over a knife-edge argmax.

There's a framework that bakes exactly this in: maximum-entropy RL. Instead of maximizing Σ_t E[r(s_t,a_t)], maximize the reward plus the entropy of the policy at each step,

  J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[ r(s_t,a_t) + α·H(π(·|s_t)) ],

where H(π(·|s)) = E_{a~π}[−log π(a|s)] and α is a temperature trading reward against randomness. As α→0 this collapses back to ordinary RL, so I'm not abandoning the objective, I'm generalizing it. And the reasons this is the *right* generalization, not just a regularizer, are concrete. A policy rewarded for entropy explores widely but still gives up on clearly bad actions (their low reward dominates). It can hold multiple modes — when two actions are equally good it keeps both, instead of arbitrarily picking one and forgetting the other exists. And Ziebart's analysis says maximum-entropy policies are robust to model and estimation error, which is exactly the fragility I'm fighting. So entropy isn't a hack for exploration; it's a different, more forgiving optimization landscape.

One bookkeeping note before I forget it: α can be folded away. The objective is α times the same problem with reward r/α and entropy coefficient 1, so if I scale rewards by c = 1/α the temperature disappears from the equations and gets absorbed into the reward magnitude. From here I'll mostly drop α and remember that the reward scale is the inverse temperature — large reward scale means low temperature and the entropy term matters less, small reward scale means high temperature and entropy dominates. I'll come back to this; it turns out to be the one knob I can't avoid tuning.

Now, maximum-entropy RL is not new, and I should be honest about why the existing way of solving it doesn't already give me what I want. The standard treatment solves directly for the optimal soft Q-function. When you redo the Bellman optimality argument with the entropy term, the hard max over actions turns into a "soft max" — a log-sum-exp. The optimal soft value is V(s) = α log ∫ exp(Q(s,a)/α) da, and the optimal policy is the energy-based (Boltzmann) distribution π(a|s) ∝ exp(Q(s,a)/α). Beautiful in the tabular/discrete case. In continuous actions it's a nightmare to *use*: you cannot sample from exp(Q(s,·)) in closed form, because Q is an arbitrary neural net and that's an arbitrary unnormalized density over a continuous space. Soft Q-learning handles this by training a *separate sampling network* with amortized Stein variational gradient descent to approximate draws from exp(Q). So now I have an inference procedure inside my RL loop, and its quality bounds everything — the convergence of the method hinges on how well that sampler matches the true posterior. And conceptually it's not even an actor-critic: the Q-function is estimating the *optimal* Q*, and the "actor" is just an approximate sampler that only influences Q through the data it collects. That's a lot of machinery and a lot of places to be unstable. I want the entropy benefits without the energy-based sampling.

So here's the tension I'm sitting in. DDPG gives me a clean actor-critic loop and off-policy reuse but a brittle deterministic actor. Soft Q-learning gives me the entropy framework but drags in approximate energy-based sampling and isn't a real actor-critic. What I want is the maximum-entropy objective, solved by an actual policy-iteration-style actor-critic — evaluate the *current* policy's soft value, then improve the policy — with a stochastic actor I can actually sample from cheaply.

Let me try to build that from policy iteration and see where it breaks. Policy iteration alternates two steps: evaluate the current policy's value, then improve the policy using that value. Let me write down what each step has to be in the maximum-entropy world.

Evaluation first. I want the soft value of a *fixed* policy π — not the optimal value, the value of *this* π — so I can criticize it. Define the soft state value as the expected action value minus the log-policy (which is the reward-plus-entropy bookkeeping, since E_a[−log π] is exactly the entropy):

  V(s) = E_{a~π}[ Q(s,a) − log π(a|s) ].

And the soft Q is reward now plus discounted soft value next:

  T^π Q(s,a) ≜ r(s,a) + γ E_{s'~p}[ V(s') ].

Does iterating Q ← T^π Q actually converge? Let me check, because if it doesn't the whole plan is dead. Substitute V into the backup and pull the entropy into the reward. Define an entropy-augmented reward

  r_π(s,a) ≜ r(s,a) + γ E_{s'~p}[ H(π(·|s')) ]  = r(s,a) + E_{s'~p}[ E_{a'~π}[ −log π(a'|s') ] ]·γ,

wait — let me be careful and just expand. V(s') = E_{a'~π}[ Q(s',a') − log π(a'|s') ]. So

  T^π Q(s,a) = r(s,a) + γ E_{s'~p}[ E_{a'~π}[ Q(s',a') − log π(a'|s') ] ]
             = ( r(s,a) + γ E_{s'~p, a'~π}[ −log π(a'|s') ] ) + γ E_{s'~p, a'~π}[ Q(s',a') ]
             = r_π(s,a) + γ E_{s'~p, a'~π}[ Q(s',a') ],

with r_π(s,a) = r(s,a) + γ E_{s'~p, a'~π}[ −log π(a'|s') ] absorbing the entropy term into the reward. Now this is *exactly* an ordinary policy-evaluation Bellman backup, just with reward r_π in place of r. The ordinary backup with a fixed policy is a γ-contraction in sup norm — subtract two iterates and the entropy/reward terms cancel, leaving |T^π Q_1 − T^π Q_2| = γ |E_{s',a'}[Q_1 − Q_2]| ≤ γ ‖Q_1 − Q_2‖_∞ — so it has a unique fixed point and Q^k → Q^π. The one thing I need is that r_π is bounded, i.e. that the entropy is bounded; with finite action sets (the tabular setting I'm using for the proof) it is. Good — soft policy evaluation converges. That's the critic.

Now improvement, and this is where I have to be cleverer than soft Q-learning. I just evaluated π_old and got Q^{π_old}. I'd *like* my new policy to be the energy-based π ∝ exp(Q^{π_old}) — that's the direction of improvement the soft theory points at. But that's exactly the thing I refuse to sample from. So instead of *becoming* the energy-based distribution, let me *project onto it*: keep my policy inside a tractable family Π (say Gaussians, things I can sample and evaluate), and within Π move it as close as possible to exp(Q^{π_old})/Z. Closeness in what sense? KL divergence is the natural choice for fitting a distribution to an unnormalized target, and it's convenient: put the new policy as the first argument,

  π_new(·|s) = argmin_{π'∈Π} D_KL( π'(·|s) ‖ exp(Q^{π_old}(s,·)) / Z^{π_old}(s) ).

Z is the (intractable) normalizer of the energy-based target, but watch — it depends only on s, not on π', so when I expand the KL it's an additive constant in π' and won't affect the argmin or any gradient with respect to the policy. So I never have to compute it. The intractable partition function, the thing that forced soft Q-learning into SVGD, just drops out because I'm fitting a tractable π instead of sampling from exp(Q).

But is this projected step actually an *improvement*? I replaced "become exp(Q)" with "get as close to exp(Q) as Π allows," and that's an approximation — I need to know whether the projected policy is no worse, or the whole policy-iteration construction is unjustified. Write the KL objective out; dropping the s-only constant log Z,

  J_{π_old}(π'(·|s)) = E_{a~π'}[ log π'(a|s) − Q^{π_old}(s,a) + log Z^{π_old}(s) ].

π_new minimizes this over Π. π_old itself is in Π, so it is a feasible point of the minimization, and the minimizer can do no worse than it:

  J_{π_old}(π_new(·|s)) ≤ J_{π_old}(π_old(·|s)).

Now expand both sides and see what that inequality buys. The right-hand side, with π'=π_old:

  E_{a~π_old}[ log π_old(a|s) − Q^{π_old}(s,a) + log Z^{π_old}(s) ]
    = log Z^{π_old}(s) − E_{a~π_old}[ Q^{π_old}(s,a) − log π_old(a|s) ]
    = log Z^{π_old}(s) − V^{π_old}(s),

since V^{π_old}(s) = E_{a~π_old}[ Q^{π_old} − log π_old ] by definition. The left-hand side, with π'=π_new:

  E_{a~π_new}[ log π_new(a|s) − Q^{π_old}(s,a) + log Z^{π_old}(s) ]
    = log Z^{π_old}(s) − E_{a~π_new}[ Q^{π_old}(s,a) − log π_new(a|s) ].

Put the inequality J_new ≤ J_old together (the log Z(s) cancels off both sides since it's the same s):

  − E_{a~π_new}[ Q^{π_old}(s,a) − log π_new(a|s) ] ≤ − V^{π_old}(s),

i.e.

  E_{a~π_new}[ Q^{π_old}(s,a) − log π_new(a|s) ] ≥ V^{π_old}(s).      (★)

So under the *new* policy, the old Q minus the new log-policy is at least the old soft value. That's the lever. Now run it through the soft Bellman equation for the old policy and bootstrap. Start from

  Q^{π_old}(s,a) = r(s,a) + γ E_{s'~p}[ V^{π_old}(s') ]
                ≤ r(s,a) + γ E_{s'~p}[ E_{a'~π_new}[ Q^{π_old}(s',a') − log π_new(a'|s') ] ],

where I used (★) to replace V^{π_old}(s') by the larger quantity. The right side is the soft backup of Q^{π_old} under π_new applied once. Now the bracket again contains a Q^{π_old}(s',a'), and I can apply (★) to *its* V again, and again — each expansion pushes the inequality one more step into the future and never reverses it because (★) holds at every state. In the limit, repeatedly applying the soft Bellman operator of π_new (which I showed converges, soft policy evaluation) takes the right-hand side to Q^{π_new}. So

  Q^{π_old}(s,a) ≤ Q^{π_new}(s,a)  for all (s,a).

The projected step really is an improvement — the new policy's soft Q dominates the old one's everywhere. The approximation of "fit instead of become" cost me nothing in monotonicity.

And then the full algorithm — alternate soft evaluation and soft (projected) improvement — converges to the best policy *in Π*. The sequence Q^{π_i} is monotonically non-decreasing in i by the improvement lemma, and it's bounded above because reward and entropy are both bounded, so it converges to some π*. At the fixed point, π* is the KL-minimizer against its own exp(Q^{π*}), so for any other π in Π, J_{π*}(π*(·|s)) ≤ J_{π*}(π(·|s)); running the same bootstrap argument as the improvement lemma with that inequality gives Q^{π*}(s,a) ≥ Q^{π}(s,a) for every π∈Π and every (s,a). So π* is optimal within the tractable class — and crucially the guarantee holds *regardless of how I parameterize Π*, which is exactly what soft Q-learning couldn't promise, since its guarantee was tied to how well a sampler approximated exp(Q). I have a real actor-critic with a convergence proof. Now make it run at scale.

The proof lives in the tabular world: exact backups to convergence, exact KL minimization at every state. With continuous states and a neural-net Q I can't run either step to convergence — too expensive — so I'll approximate policy iteration the way actor-critic always does: parameterize everything and take stochastic gradient steps on each objective instead of solving it. I'll need a Q-network Q_θ and a policy network π_φ. Do I also need a value network V_ψ? In principle no: V(s) = E_{a~π}[Q(s,a) − log π(a|s)] is determined by Q and π, and I can estimate it with one action sample from the current policy without bias. So a separate V is redundant — yet I'll keep one anyway, because the soft Q-target needs V at the *next* state, and bootstrapping a Q-network off a quantity computed from that same Q-network is exactly the unstable self-reference that target networks exist to tame. A separate, slowly-tracked V_ψ gives me a stable, low-variance target to regress Q onto, and it's convenient to train alongside. So three networks: Q_θ, V_ψ, π_φ, plus a slow target copy of the value net for bootstrapping.

The value loss is just the definition of V turned into a regression. V_ψ should match E_{a~π}[Q − log π]:

  J_V(ψ) = E_{s~D}[ ½ ( V_ψ(s) − E_{a~π_φ}[ Q_θ(s,a) − log π_φ(a|s) ] )² ],

with s drawn from the replay buffer D (off-policy is fine — V is a regression target, and the buffer just supplies states) but the *action* drawn fresh from the current π_φ, since I want the value of the current policy, not of whatever stale policy filled the buffer. Differentiating with a single action sample gives the unbiased estimator

  ∇_ψ J_V(ψ) = ∇_ψ V_ψ(s) ( V_ψ(s) − Q_θ(s,a) + log π_φ(a|s) ),   a ~ π_φ(·|s).

The Q loss is the soft Bellman residual. In the scaled-reward units I am using, Q_θ should match the soft backup c r + γ E_{s'}[V(s')], and to keep the target stable I evaluate V at the next state through the *target* value network V_{ψ̄}:

  J_Q(θ) = E_{(s,a)~D}[ ½ ( Q_θ(s,a) − Q̂(s,a) )² ],   Q̂(s,a) = c r(s,a) + γ E_{s'~p}[ V_{ψ̄}(s') ],

  ∇_θ J_Q(θ) = ∇_θ Q_θ(s,a) ( Q_θ(s,a) − c r(s,a) − γ V_{ψ̄}(s') ).

Here both s and a come straight from the replay buffer — this term is genuinely off-policy, which is the whole point: the Bellman residual for Q only needs transitions (s,a,r,s'), not on-policy actions. The target V_{ψ̄} is an exponentially-moving average of ψ, ψ̄ ← τ ψ + (1−τ) ψ̄, the standard slow-tracking trick. That's the critic side, all off-policy, all from the buffer.

Now the actor, the interesting part. The improvement step says: minimize, over φ, the KL of π_φ against exp(Q_θ)/Z_θ. As a loss over buffer states,

  J_π(φ) = E_{s~D}[ D_KL( π_φ(·|s) ‖ exp(Q_θ(s,·)) / Z_θ(s) ) ]
         = E_{s~D}[ E_{a~π_φ}[ log π_φ(a|s) − Q_θ(s,a) + log Z_θ(s) ] ],

and log Z_θ(s) is independent of φ, so it drops. The objective is E_{s,a~π_φ}[ log π_φ(a|s) − Q_θ(s,a) ]. Now how do I differentiate this through φ? The trouble is the expectation is *over* π_φ, so φ appears both inside the integrand (the log π) and in the sampling distribution. The textbook off-policy answer is the likelihood-ratio / score-function estimator — write ∇_φ E_{a~π_φ}[f] = E[ f ∇_φ log π_φ ] — which doesn't need to differentiate through the action at all. But that throws away something I have for free: the target inside the expectation is Q_θ, a neural net I can backpropagate through. The score-function estimator treats Q_θ as a black-box scalar and pays for it in variance. I'd rather use the gradient of Q with respect to the action.

To do that I have to get φ out of the sampling distribution, which is exactly what the reparameterization trick is for. Write the action as a deterministic, differentiable transform of the state and an external noise variable,

  a = f_φ(ε; s),   ε ~ N(0, I) fixed,

so sampling a is sampling ε (φ-free) and pushing it through f_φ. Now the expectation is over the fixed ε, and φ lives only in the integrand:

  J_π(φ) = E_{s~D, ε~N}[ log π_φ( f_φ(ε;s) | s ) − Q_θ( s, f_φ(ε;s) ) ].

Differentiate straight through. The gradient has two routes for φ — the explicit log π_φ, and the action f_φ that feeds both log π_φ and Q_θ:

  ∇_φ J_π(φ) = ∇_φ log π_φ(a|s) + ( ∇_a log π_φ(a|s) − ∇_a Q_θ(s,a) ) ∇_φ f_φ(ε;s),   a = f_φ(ε;s).

The −∇_a Q_θ · ∇_φ f term is precisely the DDPG-style deterministic policy gradient — ∇_a Q backpropagated through the action into the policy params — except now it flows through a *stochastic* reparameterized action instead of a deterministic μ, and it's accompanied by the entropy term ∇(log π) that pulls the policy toward spreading out. So the reparameterized actor update is the deterministic policy gradient generalized to any tractable stochastic policy, with the entropy bonus folded in. Off-policy critic, low-variance reparameterized actor gradient, intrinsic stochasticity — the tension I started with dissolves.

Now I have to pick the tractable family Π and a concrete f_φ. Gaussian is the obvious reparameterizable choice: a = μ_φ(s) + σ_φ(s) ⊙ ε. But Gaussians have unbounded support, and physical actions live in a box, [−1,1] per dimension. If I just clip, the gradient at the boundary dies and the log-prob is wrong. So squash the Gaussian sample through a tanh: sample u from the Gaussian, output a = tanh(u). That keeps it reparameterizable (tanh is differentiable) and bounds the action. But squashing changes the density, so I have to correct the log-prob with the change-of-variables Jacobian. With u ∈ R^D having density μ(u|s) and a = tanh(u) elementwise,

  π(a|s) = μ(u|s) |det(da/du)|^{-1}.

The map is elementwise, so the Jacobian da/du = diag(1 − tanh²(u_i)) is diagonal and its determinant is the product of the diagonal, giving a clean sum in log space:

  log π(a|s) = log μ(u|s) − Σ_{i=1}^D log( 1 − tanh²(u_i) ).

So every action's log-prob is the raw Gaussian log-prob minus Σ log(1 − tanh²(u_i)). One numerical caution I'll want in code: 1 − tanh²(u) underflows for large |u|, so rather than log(1 − tanh²(u)) directly I use the equivalent, stable form 2(log 2 − u − softplus(−2u)), which is the same quantity written so it never logs a near-zero. With this, my reparameterized f_φ is "MLP → (μ, log σ); u = μ + σ·ε; a = tanh(u)," and the log-prob carries the squash correction.

A couple of components I should pull in from hard experience with value-based methods. Anything that maximizes through a noisy Q-estimate biases it upward, and here both the value update and the policy update lean on Q. The cheap, well-tested fix is two Q-functions, Q_{θ1} and Q_{θ2}, trained independently on the same Bellman residual, and wherever I *use* a Q-value to push the value net or the policy, I take the *minimum* of the two. That deliberately accepts some pessimism because positive bias is the more dangerous error when the policy is allowed to chase Q peaks. The target network has the same kind of practical motivation: ψ̄ tracks ψ by an exponential moving average with coefficient τ. Small τ means a slow, stable target but slow learning; τ=1 with a periodic hard copy is the other extreme. I want a small τ (around 0.005) so the bootstrapped target moves on a slower timescale than the regressor chasing it.

That leaves the one knob I flagged earlier and can't dodge: the reward scale c = 1/α, the inverse temperature. Because I folded α into the reward, c controls how much the entropy term matters relative to reward, and therefore how stochastic the optimal policy is. Make c too small and the entropy term dominates — the policy goes nearly uniform and ignores the reward signal, so it never exploits. Make c too large and entropy becomes negligible — the policy collapses toward deterministic early, loses exploration, and gets stuck in poor local optima. Somewhere in between the policy keeps enough spread to explore while still chasing reward, and that is the balance I expect to tune per environment.

Putting the loop together: it's an off-policy actor-critic. At each environment step, sample a ~ π_φ(·|s), step the env, store (s,a,r,s') in the replay buffer D. Then take one (or a few) gradient steps on a minibatch sampled from D: descend J_Q on each θ_i (Bellman residual against c r + γ V_{ψ̄}(s')), descend J_V on ψ (using min of the two Qs and a fresh action from π_φ), descend J_π on φ (reparameterized, using min of the two Qs), and slide the target value weights ψ̄ ← τψ + (1−τ)ψ̄. Everything the critic and actor train on is off-policy data from the buffer; the only on-policy samples are the fresh actions drawn from π_φ when forming the value and policy objectives, which is exactly what makes those terms estimate the *current* policy's soft value rather than a stale one. At evaluation time, since the policy is deliberately stochastic for training, I take the mean action (the deterministic μ_φ(s)) as the low-noise readout of the learned policy.

I need the code to preserve the dependency structure I just derived. The actor is a squashed Gaussian that returns both the sampled action and its corrected log-prob; two Q-nets and a V-net plus a target V; the Q loss is computed and stepped before the value loss; the policy loss is recomputed after those critic updates, with Q parameters frozen so the policy still receives ∂Q/∂a but Q itself is not updated by the actor objective; then the target value weights move by Polyak averaging.

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20, 2

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class SquashedGaussianActor(nn.Module):
    # tractable, reparameterizable policy family Pi: Gaussian -> tanh squash
    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden), nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))
    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()      # reparameterized: a = f_phi(eps; s)
        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            # change-of-variables tanh correction, numerically stable form
            logp = logp - (2 * (math.log(2.0) - u - F.softplus(-2 * u))).sum(-1)
        else:
            logp = None
        a = torch.tanh(u) * self.act_limit               # squash into the action box
        return a, logp

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden) + [1])
    def forward(self, o, a): return self.q(torch.cat([o, a], -1)).squeeze(-1)

class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden) + [1])
    def forward(self, o): return self.v(o).squeeze(-1)

def compute_q_loss(data, q1, q2, vf_targ, gamma, reward_scale):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    # critic Q: soft Bellman residual toward reward_scale * r + gamma * V_targ(s'), off-policy
    with torch.no_grad():
        q_target = reward_scale * r + gamma * (1 - d) * vf_targ(o2)
    loss_q = 0.5 * ((q1(o, a) - q_target) ** 2).mean() + \
             0.5 * ((q2(o, a) - q_target) ** 2).mean()

    return loss_q

def compute_v_loss(data, actor, q1, q2, vf):
    o = data['obs']
    # value V: regress onto E_{a~pi}[min Q - log pi]  = the soft value of the current policy
    with torch.no_grad():
        a_pi, logp = actor(o)
        q_pi = torch.min(q1(o, a_pi), q2(o, a_pi))       # min of two Qs: fight overestimation
        v_target = q_pi - logp
    return 0.5 * ((vf(o) - v_target) ** 2).mean()

def compute_policy_loss(data, actor, q1, q2):
    o = data['obs']
    # actor: minimize KL  ==  E[ log pi - Q ], through the reparameterized action (Z dropped)
    a_pi, logp = actor(o)
    q_pi = torch.min(q1(o, a_pi), q2(o, a_pi))
    return (logp - q_pi).mean()

def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad

def update(data, actor, q1, q2, vf, vf_targ, opt_q, opt_v, opt_pi, gamma, reward_scale, tau):
    loss_q = compute_q_loss(data, q1, q2, vf_targ, gamma, reward_scale)
    opt_q.zero_grad();  loss_q.backward();  opt_q.step()

    loss_v = compute_v_loss(data, actor, q1, q2, vf)
    opt_v.zero_grad();  loss_v.backward();  opt_v.step()

    set_requires_grad(q1, False); set_requires_grad(q2, False)
    try:
        loss_pi = compute_policy_loss(data, actor, q1, q2)
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
    finally:
        set_requires_grad(q1, True); set_requires_grad(q2, True)

    with torch.no_grad():                                # slow-tracking target value net
        for p, pt in zip(vf.parameters(), vf_targ.parameters()):
            pt.data.mul_(1 - tau); pt.data.add_(tau * p.data)
```

The causal chain, start to finish: I wanted off-policy efficiency and stability at once, but DDPG's deterministic actor was the source of its brittleness and gave no exploration, while soft Q-learning's energy-based policy gave exploration but forced intractable sampling and wasn't a real actor-critic. Maximizing reward-plus-entropy gives a forgiving, multimodal objective; phrasing improvement as a *KL projection* of a tractable policy onto exp(Q) — rather than sampling from exp(Q) — kills the intractable partition function, makes it a genuine actor-critic, and stays monotonically improving because π_old is feasible in Π and the soft Bellman inequality bootstraps forward. Approximating the tabular steps with networks and SGD turns the exact operations into critic, value, and actor updates; reparameterizing the actor turns the policy objective into a low-variance DDPG-style gradient that flows ∇_a Q through a stochastic, tanh-squashed Gaussian; and twin Q-functions with a min, a slow target value net, and reward scale as inverse temperature are the practical pieces that keep the loop aligned with the derivation.
