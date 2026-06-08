## Research question

A single indivisible resource — one machine, one experimenter, one clinician's next
trial slot — has to be committed, one step at a time, to exactly one of several
competing activities that evolve stochastically as you work on them. Each activity is
a reward process: while you work on it, it pays a random reward and its internal
state advances in a Markovian way; while you ignore it, it pays nothing and stays
frozen, waiting. You observe the outcome of each step before choosing the next, and
future reward is geometrically discounted by a factor `0 < β < 1`. The goal is a rule
for deciding, at every step, which activity to advance, so as to maximize the total
expected discounted reward over an infinite horizon.

Three concrete instances make the problem vivid and sharpen what a solution must
achieve:

- **Sequential clinical trials / the Bernoulli multi-armed bandit.** Each of `n`
  treatments has an unknown success probability `θ_i`; a convenient state records
  posterior exponents `(a_i, b_i)`, with density proportional to
  `θ_i^{a_i}(1-θ_i)^{b_i}`. Patients arrive one at a time; you must assign each to a
  treatment, see success or failure, and update. You want to maximize discounted
  successes (equivalently, learn the best treatment while not wasting too many patients
  on bad ones). The exploration/exploitation tension is the point: a treatment that has
  looked good might be lucky; one that looked bad might be unlucky.

- **Stochastic scheduling on one machine.** There are `n` jobs with random,
  independently distributed processing times and rewards on completion. At each time
  unit you may switch freely (no switching cost) to any job, finished-or-not. Which
  job to push on next so as to maximize discounted completion reward?

- **Industrial research-project selection / search.** Several lines of inquiry; effort
  is allocated to one at a time; each may "pay off" at a random future moment. Which to
  fund this quarter?

What unites them: **effort is allocated sequentially among competing candidates**, each
candidate is a self-contained stochastic process whose state only moves when you touch
it, and the candidates do not interact except through the constraint that you can only
touch one per step. A good solution would have to handle an effectively enormous joint
state space without enumerating it — and ideally would say something interpretable
about *each candidate on its own*, because that is how a practitioner thinks ("how
promising is treatment 3 right now?").

## Background

**Dynamic programming and the optimality principle (Bellman, 1957).** Any of these
problems is, formally, a Markov decision process: a state space, a finite set of
controls per state, transition kernels, discounted rewards. Bellman's optimality
principle — "an optimal policy has the property that whatever the initial state and
initial decision, the remaining decisions must constitute an optimal policy with regard
to the state resulting from the first decision" — yields a backward-induction recursion
that in principle solves it exactly. For the whole family `F` of activities the
functional equation is

    R(F, x) = max_{u} { R(F, x, u) + β · E_{x,u}[ R(F, y) ] },

the max running over which activity to advance. This is a complete answer in principle.
The catastrophe is dimensionality: the state of `F` is the **product** of the per-activity
state spaces. With `n` activities of `k` states each, the joint space has `k^n` states;
for the Bayesian bandit each activity's posterior state already lives in an infinite
space, and the product is hopeless. Backward induction over
the product MDP is exponential in `n`. This is the wall every direct approach hits.

**Existence of a clean optimal policy (Blackwell, 1965).** For a discounted MDP with
finite control sets and uniformly bounded rewards, Blackwell proved that the supremum of
total expected discounted reward is attained, and attained by a *deterministic,
stationary, Markov* policy that satisfies the DP functional equation. So a well-behaved
optimal policy is guaranteed to exist and to be expressible as a fixed function of the
current (joint) state. This is the ground on which everything stands: it tells us there
*is* an optimal stationary rule to find — it says nothing yet about whether that rule
factors across activities.

**The Bayesian two-armed bandit (Bellman, 1956).** The earliest scalable-looking attack
formulated the two-armed bandit with a Bayesian prior and obtained properties of the
optimal policy and value function for the special case of two arms, one of them a
"standard" arm of *known* reward. This already contains, in embryo, the idea of measuring
an uncertain arm against a known yardstick — but it was confined to `n = 2` and produced
no structure that scales.

**Optimal stopping (Chow, Robbins & Siegmund, 1971).** The theory of when to stop a
single evolving process to maximize an expected payoff. Two features matter here. First,
its central object is a *stopping time* `τ` — a rule, depending only on the observed
history, for when to quit. Second, its "monotone case": when the one-step-lookahead rule
keeps recommending the same action, that lookahead rule is actually optimal — a situation
where the otherwise hard stopping problem collapses to a myopic comparison. The language
of stopping times, and this collapse-to-myopic phenomenon, are exactly the kind of
single-process structure a per-activity summary would need.

**Priorities and the `cμ` heuristic.** In scheduling and queueing it was folklore (and in
restricted settings provable, e.g. Sevcik 1972 for computer job scheduling, and the
classical `cμ`-rule for a single server) that one should give priority by a per-job
index — roughly, holding cost times completion rate — and serve the highest-index job.
These were known to be optimal only in special, often myopic, regimes. They are
suggestive: they say "maybe a *scalar per candidate* is enough", but they did not say
why, or when, or what the scalar really is in the general stochastic case.

**The diagnostic fact that frames the whole subject.** A simple but load-bearing
observation about these allocation problems: an activity you decline to advance *now* is
not lost. Because its state is frozen while ignored, you can come back to it later and
get *exactly the same* sequence of rewards from it — only shifted in time, hence only
re-discounted. There is no "the road not taken closes behind you" effect, unlike, say,
choosing a fork on a journey where the unchosen exits vanish. This non-irrevocability is
a structural property of the problem class, and it is the crack through which the
dimensionality wall can be broken.

## Baselines

- **Full backward induction on the product MDP (Bellman 1957; Blackwell 1965).**
  Core idea: solve `R(F,x) = max_u { R(F,x,u) + β E_{x,u}[R(F,y)] }` by value/policy
  iteration over the joint state `x = (x_1, …, x_n)`. Exact and general. **Gap:** the
  joint state space is the product of the per-arm spaces — `k^n`, or infinite-dimensional
  for Bayesian arms — so it is computationally infeasible beyond a couple of arms. It also
  yields no interpretable per-arm quantity.

- **Bayesian two-armed bandit, special-case analysis (Bellman 1956).** Core idea:
  with one known "standard" arm and one unknown arm, characterize the optimal switch via
  the DP equation in the unknown arm's posterior. **Gap:** restricted to `n = 2` and to
  one-unknown-one-known; gives no construction that extends to many genuinely competing
  unknown arms.

- **Myopic / one-step-lookahead and `cμ`-type priority rules (Sevcik 1972; classical
  `cμ`).** Core idea: rank candidates by an easily computed scalar — immediate expected
  reward, or holding-cost × completion-rate — and serve the top one. Cheap and
  per-candidate. **Gap:** optimal only in special "deteriorating"/monotone regimes; in
  general a candidate whose prospects *improve* with work (e.g. an arm you are still
  learning about) is undervalued by a myopic index, so these rules are provably
  suboptimal when the future matters (β near 1).

- **Optimal-stopping solutions for a single process (Chow–Robbins–Siegmund 1971).**
  Core idea: for one process, choose a stopping time to maximize expected discounted
  payoff; in the monotone case the myopic rule is optimal. **Gap:** addresses one process
  in isolation, with no notion of *competing* processes and no rule for interleaving
  several. It is a tool, not yet an allocation policy.

## Evaluation settings

The natural yardsticks are canonical problem families with this allocation structure:

- **Bernoulli bandit / sequential clinical trials.** Arms with unknown success
  probabilities can be represented by posterior exponents `(a_i, b_i)`, with density
  proportional to `θ_i^{a_i}(1-θ_i)^{b_i}`; state updates by Bayes' rule on each
  success/failure to
  `(a_i + 1, b_i)` or `(a_i, b_i + 1)`; per-step reward is the posterior mean
  `(a + 1)/(a + b + 2)`; discount `β ∈ (0,1)`. Metric: total expected discounted reward.

- **Normally-distributed rewards.** Arms whose unknown mean `θ` has a Normal prior
  `N(ξ, m^{-1})` updated by observations of known variance `b^2`; a continuous-reward
  analogue of the Bernoulli bandit, natural for clinical trials with a continuous outcome.

- **Single-machine stochastic scheduling.** `n` jobs with given completion-hazard
  sequences `p_i(t)` and completion rewards `V_i`; metric: total expected discounted
  reward, or (as `β → 1`) total expected delay cost with per-unit delay cost `c_i`.

- **Search problems.** Several populations searched for a rare individual; reward accrues
  on first success; metric: expected discounted time to first success.

The protocol throughout is **discrete time, stationary** (no explicit time-dependence in
transitions or rewards) and **infinite-horizon discounted** — the stationarity is essential,
since with a finite horizon a per-candidate ranking is known *not* to be optimal in
general.

## Code framework

The available machinery can be represented as a single-process object, a generic
discounted-DP solver, and a simulator that steps a family of processes under a supplied
policy. The unresolved pieces are the per-process scalar summary and the rule that uses
those summaries to choose the next process.

```python
BETA = 0.9  # discount factor in (0,1)

class BanditProcess:
    """A single Markov reward process: states, transitions, per-step reward.
    Its state advances only when this process is the one advanced."""
    def __init__(self, states, transition, reward, initial_state):
        self.states = states          # iterable of states
        self.transition = transition  # transition(s) -> {s': prob}
        self.reward = reward          # reward(s) -> expected immediate reward R(s,1)
        self.initial_state = initial_state

    def expected_reward(self, s):     # R(s, 1)
        return self.reward(s)

    def step(self, s):                # sample next state given we advance from s
        # TODO: sample s' ~ transition(s)
        pass


def discounted_value_iteration(states, transition, reward, beta, tol=1e-9):
    """Generic 1-D discounted DP solver over a single process's own state space.
    Returns a value function V(s)."""
    # TODO: standard value iteration on a single process's states
    pass


def summarize_process(proc, beta):
    """Collapse one process to a scalar computed from its own states and transitions."""
    # TODO: compute the per-process summary
    pass


def allocation_policy(processes, states, summaries):
    """Choose which process to advance from current per-process states."""
    # TODO: turn per-process summaries into the next action
    pass


def simulate(processes, policy, beta, horizon):
    """Step the family forward under a policy; only the chosen process advances."""
    states = [p.initial_state for p in processes]
    total = 0.0
    discount = 1.0
    for t in range(horizon):
        j = policy(processes, states)         # which process to advance
        total += discount * processes[j].expected_reward(states[j])
        states[j] = processes[j].step(states[j])
        discount *= beta
    return total
```
