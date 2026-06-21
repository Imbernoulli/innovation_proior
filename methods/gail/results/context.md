## Research question

We want to learn how to *act* in a sequential decision problem purely from a
fixed batch of expert demonstrations. The setting is deliberately austere: the
learner is handed a set of sampled trajectories from an expert policy
$\pi_E$ — sequences of state-action pairs — and nothing else. There is no
reward (cost) signal of any kind, the learner may not query the expert for the
correct action in new states, and it may not interact with the expert during
training. The learner *may* interact with the environment (roll out its own
policy and observe where it goes), but it never sees what the expert would have
done in the states the learner itself visits.

The object we actually care about is a **policy**: a mapping from states to
a distribution over actions that reproduces the expert's behavior, and that
generalizes to states the learner reaches on its own. The question is how to
extract such a policy from the demonstrations, in high-dimensional
continuous-control environments with neural-network function approximation.

## Background

**The MDP and discounted occupancy.** We work in a $\gamma$-discounted
infinite-horizon Markov decision process with states $s\in\mathcal S$, actions
$a\in\mathcal A$, transition dynamics $P(s'\mid s,a)$, and initial-state
distribution $p_0$. A stationary stochastic policy $\pi(a\mid s)$ induces a
distribution over trajectories; for a per-step cost $c(s,a)$ we write
$\mathbb E_\pi[c] = \mathbb E\big[\sum_{t=0}^\infty \gamma^t c(s_t,a_t)\big]$
with $s_0\sim p_0$, $a_t\sim\pi(\cdot\mid s_t)$, $s_{t+1}\sim P(\cdot\mid s_t,a_t)$.
A central object is the **occupancy measure** of a policy,
$\rho_\pi(s,a) = \pi(a\mid s)\sum_{t=0}^\infty \gamma^t P(s_t=s\mid\pi)$, the
(discounted) distribution of state-action pairs an agent encounters under $\pi$.
It makes expected costs linear: $\mathbb E_\pi[c] = \sum_{s,a}\rho_\pi(s,a)\,c(s,a)$.
A classical result (Puterman 2014) says the set of valid occupancy measures
$\mathcal D = \{\rho_\pi : \pi\in\Pi\}$ is exactly the set of nonnegative
functions satisfying the Bellman flow constraints
$\sum_a \rho(s,a) = p_0(s) + \gamma\sum_{s',a}P(s\mid s',a)\rho(s',a)$ — an affine
(hence convex) set. Syed et al. (2008, Theorem 2) sharpen this to a one-to-one
correspondence between policies and occupancy measures: any $\rho\in\mathcal D$
is the occupancy measure of the unique policy
$\pi_\rho(a\mid s)=\rho(s,a)/\sum_{a'}\rho(s,a')$.

**Causal entropy.** The $\gamma$-discounted causal entropy of a policy is
$H(\pi)=\mathbb E_\pi[-\log\pi(a\mid s)]$ (Bloem & Bambos 2014). It measures how
"undecided" the policy is across the states it visits, and it will play the role
of a tie-breaking regularizer.

**Convex duality.** For a function $f$ on $\mathbb R^{\mathcal S\times\mathcal A}$,
its convex conjugate is $f^*(x)=\sup_y x^\top y - f(y)$. For a convex-concave
saddle function on convex sets, minimax duality lets the inner $\min$ and outer
$\max$ be exchanged (Millar 1983; Boyd & Vandenberghe 2004).

**Single-step fitting under covariate shift.** Ross & Bagnell (2010) and Ross,
Gordon & Bagnell (2011) analyze fitting a policy to expert decisions one timestep
at a time: a classifier trained on the expert's state distribution that makes a
per-step error at rate $\varepsilon$ incurs a regret that grows quadratically in
the horizon, since errors move the agent into states off the expert's
distribution. Cost-function-based imitation, which scores *whole trajectories*,
has been applied to problems from taxi-route prediction (Ziebart et al. 2008) to
quadruped footstep planning (Ratliff et al. 2009).

**Generative adversarial networks.** Goodfellow et al. (2014) train a generator
$G$ against a discriminator $D$ that tries to tell generated samples from real
data; at the optimum of the inner classification problem the objective equals (up
to a constant) the Jensen-Shannon divergence between the generated and true data
distributions, so driving the generator to fool $D$ matches the two
distributions. The tool was originally applied to images.

## Baselines

**Behavioral cloning (Pomerleau 1991).** Treat imitation as supervised learning:
collect the expert's $(s,a)$ pairs and fit $\pi(a\mid s)$ by maximum likelihood
(or regression for continuous actions). It optimizes a single-timestep loss under
the *expert's* state distribution and requires no environment interaction.

**Inverse reinforcement learning, maximum causal entropy form (Russell 1998;
Ng & Russell 2000; Ziebart et al. 2008, 2010).** Rather than copy actions,
recover a cost function under which the expert is optimal, from a family
$\mathcal C$:
$$\max_{c\in\mathcal C}\Big(\min_{\pi\in\Pi} -H(\pi)+\mathbb E_\pi[c]\Big)-\mathbb E_{\pi_E}[c].$$
The inner $\min_\pi -H(\pi)+\mathbb E_\pi[c]$ is itself an (entropy-regularized)
reinforcement-learning problem, $\mathrm{RL}(c)=\arg\min_\pi -H(\pi)+\mathbb E_\pi[c]$,
so each update to the cost requires (approximately) solving RL in an inner loop
(Levine & Koltun 2012; Finn et al. 2016). The cost ranks entire trajectories, and
the procedure returns a cost function from which a policy is obtained by a
separate planning/RL pass.

**Apprenticeship learning over linear cost classes (Abbeel & Ng 2004; Syed &
Schapire 2007; Syed et al. 2008).** Find a policy that does at least as well as
the expert across a class $\mathcal C$ of cost functions:
$\min_\pi \max_{c\in\mathcal C}\mathbb E_\pi[c]-\mathbb E_{\pi_E}[c]$. With
$\mathcal C$ a set of linear combinations of basis functions $f_1,\dots,f_d$, the
inner $\max$ has a closed form: the $\ell_2$-ball class gives feature-expectation
matching $\|\mathbb E_\pi[f]-\mathbb E_{\pi_E}[f]\|_2$ (Abbeel & Ng 2004), and the
simplex class gives the worst-case-feature objective of MWAL/LPAL (Syed &
Schapire 2007; Syed et al. 2008). Ho et al. (2016) made this scale to large
continuous-control problems by writing the gradient of the inner-maximized
objective as a policy gradient with the maximizing cost $c^\*$ and taking a
trust-region natural-gradient policy step. The cost class $\mathcal C$ is a
low-dimensional linear subspace spanned by hand-designed features.

**Trust region policy optimization (Schulman et al. 2015a).** Not an imitation
method but the policy optimizer the scalable approaches rely on: a
KL-divergence-constrained natural-gradient step that improves an advantage-based
surrogate while keeping the new policy close to the old one, so a noisy gradient
estimate cannot blow the policy up. Generalized advantage estimation (Schulman
et al. 2015b) with a fitted value baseline reduces the variance of the gradient.

## Evaluation settings

The natural testbeds are physics-based continuous control tasks, from
low-dimensional classics — cartpole (Barto et al. 1983), acrobot, mountain car
(Moore & Hall 1990) — to high-dimensional locomotion (HalfCheetah, Hopper,
Walker, Ant, and a 3D Humanoid), simulated in MuJoCo (Todorov et al. 2012) and
exposed through OpenAI Gym (Brockman et al. 2016), each carrying a built-in true
cost function. The protocol that suits this problem: run a policy-gradient RL
method (TRPO) on each true cost to produce an expert policy; sample expert
datasets of varying sizes (a handful to tens of trajectories, ~50 state-action
pairs each); train each imitation method on each dataset with a fixed amount of
environment interaction; and compare with negative-cost performance normalized so
the expert scores 1 and a random policy scores 0. Policies and any learned
cost model share a fixed architecture (two hidden layers of 100 units,
$\tanh$), randomly initialized. The natural baselines to measure against are
behavioral cloning, feature-expectation matching, and the game-theoretic
(simplex) apprenticeship learner.

## Code framework

The primitives that already exist: an environment with a `step`/`reset`
interface and a trajectory sampler; a parameterized stochastic policy (Gaussian
for continuous actions, categorical for discrete) with log-probability, entropy,
and KL operations; a trust-region natural-gradient policy step; a value-function
baseline with generalized advantage estimation; and a generic neural-net builder
plus Adam. The open slot is a learned per-transition signal: something that,
given the current policy's sampled $(s,a)$ pairs and the expert's $(s,a)$ pairs,
produces a per-step value for the policy optimizer and is itself fit from data.

```python
# --- existing primitives ---------------------------------------------------
class StochasticPolicy(nn.Module):
    def action_dist(self, obs): ...          # params of a(.|s)
    def log_prob(self, obs, act): ...
    def entropy(self, obs): ...
    def kl(self, obs, old_dist): ...

def trpo_step(policy, obs, act, advantages, max_kl): ...   # KL-constrained NG step
def gae(rewards, values, ep_lens, gamma, lam): ...         # advantage estimates
class ValueFunction(nn.Module): ...                        # baseline, fit to returns
def sample_trajectories(env, policy, n): ...               # obs, act, episode lengths

# --- open slot --------------------------------------------------------------
class LearningSignal(nn.Module):
    """Maps (s,a) pairs to a per-step signal for the policy, and is fit from
    the current-policy samples and the expert samples."""
    def __init__(self, obs_dim, act_dim, hidden=100):
        pass  # TODO: parameters of the signal model

    def score(self, obs, act):
        pass  # TODO: scalar score for each transition

    def reward(self, obs, act):
        pass  # TODO: per-step learning signal handed to the policy optimizer

    def fit(self, optimizer, policy_obs, policy_act, expert_obs, expert_act):
        pass  # TODO: update the signal from policy vs expert (s,a) pairs

def imitation_loop(env, expert_obs, expert_act, policy, value_fn,
                   obs_dim, act_dim, iters, gamma, gae_lam, max_kl):
    signal = LearningSignal(obs_dim, act_dim)
    signal_opt = Adam(signal.parameters(), lr=...)
    for _ in range(iters):
        obs, act, ep_lens = sample_trajectories(env, policy, n)
        rew = signal.reward(obs, act)
        adv = gae(rew, value_fn(obs), ep_lens, gamma, gae_lam)
        trpo_step(policy, obs, act, adv, max_kl)
        idx = subsample(len(expert_obs), size=len(obs))
        signal.fit(signal_opt, obs, act, expert_obs[idx], expert_act[idx])
        rew_for_value = signal.reward(obs, act)
        value_fn.fit(obs, returns_from(rew_for_value, ep_lens, gamma))
```
