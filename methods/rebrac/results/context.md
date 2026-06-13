# Context: offline continuous-control RL and the minimalist-baseline question (circa 2021-2023)

## Research question

We are handed a fixed dataset `D` of transitions `(s, a, r, s')` collected by some unknown
behavior policy (or mixture of policies), and asked to learn the best possible continuous
control policy `pi(s)` from it **without ever interacting with the environment**. The MDP is
the usual `{S, A, P, R, gamma}` with `S subset R^n`, `A subset R^m`, and the objective is the
discounted return `E_pi[sum_t gamma^t R(s_t, a_t)]`. The catch that makes offline different
from ordinary off-policy RL is precise and well documented: any bootstrapped value method
must evaluate `Q(s', a')` at the next action `a'` chosen by the *current* policy, but offline
`a'` is frequently an action the dataset never contains. A function approximator asked for
`Q` at such an out-of-distribution (OOD) action returns an essentially unconstrained
extrapolation, and because the policy is trained to *maximize* `Q`, it is actively pulled
toward whichever OOD actions the critic happens to overrate. The overrated target then feeds
the next Bellman backup, the error compounds, values diverge, and the policy collapses. So
the goal is concrete: suppress this OOD value overestimation strongly enough that learning is
stable, while still letting the policy exploit the genuinely good in-distribution actions —
and do it on D4RL MuJoCo locomotion (HalfCheetah, Hopper, Walker2d) such that one
configuration generalizes across datasets rather than overfitting a single one.

A second, methodological pressure sits on top of the algorithmic one. By this time the
offline-RL literature has produced many algorithms of escalating complexity — conservative
critic penalties, large critic ensembles, learned generative behavior models, expectile value
functions — and each tends to ship not only its core idea but a bundle of "minor"
design/implementation choices (network depth, batch size, inter-layer normalization, learning
rates, actor pre-training). These choices are rarely ablated against a clean baseline, so it
is impossible to attribute reported gains to the algorithm versus the bundle. A solution that
mattered would therefore have to be *minimal and honestly attributable*: few moving parts,
each one individually justified and ablatable, no secondary networks, negligible compute
overhead, and a parameter count no larger than the prior baselines (the contribution must be
algorithmic, not capacity).

## Background

**The overestimation feedback loop.** In Q-learning the greedy target `y = r + gamma
max_{a'} Q(s', a')` is biased upward: if the value estimate carries even zero-mean noise
`eps`, then `E_eps[max_{a'}(Q(s',a') + eps)] >= max_{a'} Q(s',a')` (Thrun & Schwartz 1993),
so a consistent overestimation is injected and then propagated through the Bellman equation.
In continuous control the maximization is implicit (the actor follows the deterministic
policy gradient `nabla_phi pi · nabla_a Q`), and the same upward bias was shown to occur for
deterministic actor-critic (Fujimoto et al. 2018): the value estimate of the actor-updated
policy exceeds that of the policy that would be obtained against the true value. Offline, this
loop is far more dangerous than online, because online the agent can at least visit the
overrated state-action and get a corrective low reward, whereas offline the dataset is frozen
and the correction never arrives.

**Behavior regularization as the dominant remedy.** The prevailing wisdom is that the
learned policy must be kept close to the data-generating behavior policy `pi_b`. The general
statement of this idea (Wu, Tucker & Nachum 2019) is to add a divergence penalty `D(pi(·|s),
pi_b(·|s))` and recognize there are exactly two places to put it: inside the **critic target**
(a *value penalty* — pessimize the bootstrap toward behavior) or inside the **actor
objective** (a *policy regularization* — pull the acting policy toward behavior). Various
divergences were tried — KL, kernel MMD, Wasserstein — with no consistent winner, and the
two penalty strengths were treated as a single shared coefficient.

**Normalization as an extrapolation regularizer (diagnostic finding).** A separate empirical
thread observed that applying Layer Normalization (Ba et al. 2016) inside the critic
markedly stabilizes value learning on offline data, and gave the mechanism explicitly (Ball
et al. 2023). If the last hidden representation `psi(s,a)` is layer-normalized before the
final linear head `w`, then for *any* input — including OOD `(s,a)` —
`|Q(s,a)| = |w^T relu(psi(s,a))| <= ||w|| · ||relu(psi(s,a))|| <= ||w|| · ||psi(s,a)|| <= ||w||`,
because LayerNorm forces `||psi(s,a)||` to a bounded constant. The Q-value off-distribution is
therefore bounded by the head weight norm and cannot run away far above the values seen in
data. A toy fit (inputs on a radius-0.5 circle, target `y = ||x||`) shows a plain ReLU MLP
extrapolating without bound outside the support while the LayerNorm variant stays bounded.

**Depth, batch size, discount as understudied knobs.** Most off-policy bases (TD3, SAC) ship
with two hidden layers by default; several later offline works quietly switched to three and
reported it as "important", with the scaling intuition that depth helps given enough data
(Kaplan et al. 2020). Batch sizes above 256 are uncommon but were used to accelerate
convergence (Nikulin et al. 2022), following large-batch optimization (You et al. 2017,
2019) with the learning rate scaled accordingly. The discount factor is usually fixed at
0.99, but for long, sparse-reward tasks this is arithmetically fragile: an episode of length
`L` with a single terminal reward sees that reward discounted by `gamma^L`, and `0.99^1000 ~=
4·10^-5` essentially erases it at the start state. These were each observed in isolation;
their combined effect on a minimal baseline is what is unexamined.

**Diagnostic findings about prior offline algorithms.** When the heavy implementation
adjustments of complex offline methods (custom architectures, actor pre-training, reward
bonuses, sampled-action approximations) are stripped back to the underlying base algorithm,
their performance drops substantially (Fujimoto & Gu 2021) — evidence that the bundled
choices, not only the headline algorithm, carry much of the gain. Independently, offline-
trained policies exhibit high episodic variance and instability relative to online-trained
ones, even for very simple offline methods, suggesting the instability is intrinsic to the
offline setting rather than to any one algorithm.

## Baselines

**TD3 (Fujimoto, van Hoof & Meger 2018).** The online off-policy base for continuous
control. Three mechanisms, all aimed at the overestimation/variance problem. (1) *Clipped
double-Q*: keep twin critics and form the target from the minimum,
`y = r + gamma · min_{i=1,2} Q_{theta'_i}(s', a')`; the min acts as an approximate
upper-bound suppressor, biasing toward underestimation, which (unlike overestimation) is not
propagated because low-valued actions are simply avoided by the policy. (2) *Target policy
smoothing*: add clipped noise to the target action,
`a' = pi_{phi'}(s') + clip(N(0, sigma), -c, c)`, `sigma = 0.2`, `c = 0.5`, so nearby actions
are forced to share value and the target stops overfitting narrow Q peaks. (3) *Delayed
policy and target updates*: update the actor and the slow targets `theta' <- tau·theta +
(1-tau)·theta'` (`tau = 5e-3`) only every `d = 2` critic steps, letting critic error settle
before each policy step. **Gap:** TD3 has nothing that keeps the policy near a fixed dataset;
run offline, the actor walks straight into the OOD region and the clipped-min cannot rescue a
critic whose entire off-distribution surface is unconstrained.

**TD3+BC (Fujimoto & Gu 2021).** The minimalist offline baseline that has become the
de-facto comparison point. It changes the online base in essentially one line: add a
behavior-cloning term to the deterministic-policy-gradient actor update,
`pi = argmax_pi E_{(s,a)~D}[ lambda·Q(s, pi(s)) - (pi(s) - a)^2 ]`,
using squared error as the divergence and a single coefficient. To keep one coefficient
transferable across reward scales it normalizes the value term: since the BC term is
`O(action^2) <= 4` for actions in `[-1,1]` but `Q` scales with the reward magnitude, it sets
`lambda = alpha / ( (1/N) sum_{(s_i,a_i)} |Q(s_i,a_i)| )`, with `alpha = 2.5` and `lambda`
treated as a stop-gradient scalar (it rescales the loss, it is not differentiated). It also
normalizes state features `s_i <- (s_i - mu_i)/(sigma_i + eps)`, `eps = 1e-3`. The critic is
left exactly as in TD3. Architecturally it keeps TD3's two hidden layers, no inter-layer
normalization, batch 256. **Gaps:** (1) only the *actor* is regularized — the critic target
still bootstraps `min Q(s', a')` at an unconstrained next action `a'`, which at the next
state can itself be OOD and overrated; (2) a single coupled penalty strength means actor-side
and critic-side caution cannot be set independently; (3) the architecture and optimization
defaults (depth, normalization, batch, learning rate) are inherited from the online base and
never revisited, so whatever those choices leave on the table is left there.

**BRAC (Wu, Tucker & Nachum 2019).** The general behavior-regularized actor-critic framework
that names the two penalty locations. Value penalty puts the divergence inside the critic
target,
`min_Q E[ (r + gamma·(Q̄(s', a') - alpha·D̂(pi(·|s'), pi_b(·|s'))) - Q(s,a))^2 ]`,
and policy regularization puts it inside the actor objective,
`max_pi E[ E_{a''~pi}[Q(s, a'')] - alpha·D̂(pi(·|s), pi_b(·|s)) ]`.
It surveyed KL, kernel MMD and Wasserstein for `D̂` and found no consistent advantage.
**Gap:** the two penalties were given the *same* coefficient `alpha` (or only one location
was enabled at a time), and the framework was instantiated with stochastic policies and
sample-based density divergences requiring a separately learned behavior model — never
collapsed to the cheap deterministic-policy case nor allowed to weigh its two sides
separately.

**IQL (Kostrikov, Nair & Levine 2022).** A contrasting offline approach that sidesteps
querying OOD actions entirely: it fits an expectile value function `V` by asymmetric
regression (default expectile `tau = 0.7`) so the critic never has to evaluate actions
outside the data, then extracts a policy by advantage-weighted regression with temperature
`beta = 3.0`. **Gap:** the policy is only ever pushed toward dataset actions reweighted by
advantage; it cannot improve beyond the best in-support action by the same maximize-`Q`
mechanism that an actor-critic uses, and it adds its own pair of knobs (`tau`, `beta`).

## Evaluation settings

The natural yardstick is the D4RL benchmark (Fu et al. 2020) on MuJoCo continuous-control
locomotion — HalfCheetah, Hopper, Walker2d — using the `medium-v2` datasets (and, more
broadly, the random / medium / expert / medium-replay / medium-expert / full-replay variants).
Agents train for one million gradient steps on the static dataset and are evaluated by
rolling out the deterministic policy in the environment; the metric is the D4RL normalized
score (0 = random policy, 100 = expert policy), averaged over evaluation episodes and over
several unseen training seeds. Sparse-reward navigation (AntMaze) and dexterous manipulation
(Adroit) are the harder relatives in the same benchmark. A method is judged on its *average
across datasets* under a fixed configuration, so dataset-specific tuning that does not
generalize is penalized. All MLP hidden widths are held at 256 and total trainable parameters
are capped relative to the strongest baseline, so capacity cannot be the source of any gain.

## Code framework

The method drops into a standard offline actor-critic harness that already exists: a
deterministic policy network, a Q-function network, an Adam optimizer per network, soft
target copies, and a replay buffer that hands back minibatches sampled from the static
dataset. What is *not* yet decided is how the actor and critic losses should be shaped to
suppress OOD overestimation, and which architectural/normalization choices the two networks
should carry — those are exactly the slots to fill.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, layernorm=False):
    """Generic MLP factory. Hidden width fixed at 256 elsewhere. The reasoning
    will decide whether/where inter-layer normalization belongs."""
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        # TODO: inter-layer normalization may or may not belong here
        if layernorm:
            pass
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """pi(s) = max_action * tanh(net(s)). Depth/width and whether to normalize
    between layers are choices to be settled."""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = None  # TODO: the actor network we will design

    def forward(self, state):
        pass


class Critic(nn.Module):
    """Q(s, a). Twin critics for the clipped-min target are standard; depth and
    whether to normalize between layers are choices to be settled."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = None  # TODO: the critic network we will design

    def forward(self, state, action):
        pass


class OfflineActorCritic:
    """Off-policy deterministic actor-critic over a static dataset. The standard
    TD3 machinery (twin critics, clipped-min target, target policy smoothing,
    delayed actor/target updates, soft target updates) is available; what is not
    settled is how the actor and critic losses should be regularized so that
    out-of-distribution bootstraps stop being overestimated."""

    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=5e-3, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        self.actor = DeterministicActor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.total_it = 0

    def train(self, batch):
        self.total_it += 1
        state, action, reward, next_state, done, next_action_data = batch

        # ---- critic update ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            target_q = torch.min(self.critic_1_target(next_state, next_action),
                                 self.critic_2_target(next_state, next_action))
            # TODO: how should the bootstrap target be shaped so that an
            #       out-of-distribution next action is not trusted?
            target_q = reward + (1 - done) * self.discount * target_q

        critic_loss = (F.mse_loss(self.critic_1(state, action), target_q)
                       + F.mse_loss(self.critic_2(state, action), target_q))
        # ... optimize critics ...

        # ---- delayed actor update ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            # TODO: the actor objective we will design — trade value
            #       maximization against staying with the dataset
            actor_loss = None
            # ... optimize actor; soft-update targets ...
```

The two `# TODO` slots in the losses, and the two `None` network bodies with their
normalization choice, are what the method fills in.
