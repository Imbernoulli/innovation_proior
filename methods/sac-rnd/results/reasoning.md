Let me start from the one disease that organizes all of offline RL, because the method has to be built directly on top of it. I have a fixed dataset `D` of transitions collected by some behavior policy, no environment to query, and I want a policy better than the one that collected the data. The standard machinery is a bootstrapped critic: regress `Q(s,a)` toward `r + γ Q(s', a')` where `a'` is whatever the policy proposes at the next state. Online this self-corrects, because if the critic over-values an action the agent will eventually try it and see the truth. Offline there is no "eventually." The actor proposes actions that were never in `D` at `s'`, the critic has no data there and extrapolates — and the extrapolation is almost always *upward*, because the actor is optimized to climb wherever the critic bulges, and the bulges are disproportionately the positive-error regions. The inflated value backs up through the Bellman recursion, the policy chases it, and nothing ever pulls it back. So every offline method is, at heart, a way of keeping the value (and the policy) honest on actions the data does not support.

The cleanest way to *say* what I want is in the language of uncertainty. If I had, for every `(s, a)`, a reliable number `b(s,a)` measuring how unfamiliar that action is at that state — small on the data, large off it — then I could simply subtract a multiple of it from the value and from the policy objective: be pessimistic exactly where I am ignorant. This is the anti-exploration framing, and it is worth dwelling on because it tells me where to look for `b`. Online exploration *adds* a novelty bonus `+b` to chase the unknown; offline I want to *subtract* the same novelty signal `−b` to flee the unknown. The two are the same estimator with opposite sign. So the question becomes: what is a good, cheap novelty estimator over `(s, a)` pairs?

The ensemble answer is known and strong. Keep `N` critics, initialize them differently, and use their disagreement at `(s, a)` as the uncertainty — where they agree, the data pinned them down; where they scatter, it is OOD. SAC-N and EDAC do exactly this and are near the top of D4RL. But the uncertainty signal *is* the ensemble, so the cost scales with `N` — ten, fifty critics — and in a setting where I am told the parameter budget is capped and the contribution must be algorithmic rather than capacity, an `N`-critic ensemble is precisely the move I cannot make. I want ensemble-quality pessimism from a *single* small extra network.

The canonical ensemble-free novelty signal in online RL is Random Network Distillation, and it is cheap in exactly the way I need. A fixed, randomly-initialized *target* network `g(x)`, and a *predictor* `ĝ(x)` trained by regression to match `g` on the inputs I actually observe. On inputs that appear often in training the predictor learns to mimic the target and the error `‖ĝ(x) − g(x)‖²` goes to zero; on inputs never seen the predictor was never trained there and the error stays large. One frozen network, one trained network, a squared error. Online this drives exploration; the natural question is whether the *same* squared error, computed over `(s, a)` inputs and trained only on dataset `(s, a)`, behaves as a clean OOD-action detector — small on in-data actions, large on the OOD actions the offline policy must be kept away from.

So the first concrete design is direct. Take SAC as the continuous base — a Tanh-Gaussian actor, twin critics with a `min` target, auto-tuned entropy — because its stochastic actor already samples the actions I need to score. Add an RND module whose predictor and target both take `(s, a)` as input and emit an embedding; train the predictor on dataset `(s, a)` by MSE to the frozen target; define `b(s,a) = ‖ĝ(s,a) − g(s,a)‖²`; and subtract `β·b` wherever the policy could escape to OOD actions. Then run it.

There is a prior report that plain offline RND is "not discriminative enough" as an OOD detector, and before I either trust this design or change it I want to understand *why* a distillation error trained on `(s,a)` would fail to discipline the actor — the diagnosis is the whole method. The worry is about how the action enters the predictor. If I concatenate the action onto the state at the input, `[s, a] → ĝ`, then the predictor is an MLP whose first layer mixes a high-dimensional state with a low-dimensional action. My suspicion is that the action's influence on the predictor's error is *diluted* by the state, so the error becomes a weak, flat function of the action — and a flat penalty is one the actor can slide along to an OOD action without paying. But "diluted" and "flat" are claims I am asserting, not computing. I should not build a method on a mechanism I have only imagined. Let me put numbers on it.

First, the dilution itself. Take halfcheetah-like dimensions, `obs_dim = 17`, `act_dim = 6`, standardized states (unit scale per dim) and a tight dataset-action blob `a ∼ 0.3·N(0, I)`. Then `mean ‖s‖ = 4.06` and `mean ‖a‖ = 0.71`, so the action is `0.71 / √(4.06² + 0.71²) ≈ 17%` of the concatenated input vector's norm. And the first linear layer's weight mass splits `‖W_s‖ = 7.98` over the state columns versus `‖W_a‖ = 4.67` over the action columns — the action gets about 37% of the first-layer weight but is fed an input block that is only ~17% of the input energy. So the action genuinely is the minority partner at the input of a concat predictor. That much is real.

Now the claim that matters: does that dilution actually make the *bonus* a flat function of the action, and does it let the actor escape? Here I have to be careful, because this is exactly the kind of statement I would otherwise just assert. So I trained both a concat RND and a FiLM RND (FiLM = the action modulates the stream over the state; described below) to convergence on the dataset blob, and measured two things. The mean gradient magnitude of the bonus with respect to the action, evaluated *on* in-data actions, came out `‖∂b/∂a‖ ≈ 0.0062` for concat versus `≈ 0.0103` for FiLM — concat's error surface is about 1.7× flatter in the action direction. That is consistent with the dilution story, and it is a real measured difference rather than a hoped-for one.

But the second measurement is where I hit a wall, and it is worth reporting honestly because it changed how confidently I can state the mechanism. The decisive test is escapability: start the actor at a clearly OOD action and let it descend the bonus by gradient — does it find a low-error OOD pocket and stay there (penalty escapable), or is it dragged back toward the data (penalty effective)? I ran this from random OOD actions of norm ≈ 7, far outside the `‖a‖ ≈ 0.71` data blob. Under *concat*, the optimizer pulled the action from `‖a‖ = 7.03` down to `1.62` while the bonus fell from `0.266` to `0.010`; under *FiLM*, from `6.92` to `0.86` with the bonus `0.915 → 0.008`. In both cases the only way the optimizer found to make the bonus small was to march the action back toward the data norm. My clean static toy does **not** reproduce a binary "concat is escapable, FiLM is not." It reproduces only the *quantitative* part — concat's bonus is measurably flatter in the action and lets the action settle a bit further out (`1.62` vs `0.86`) — not the dramatic failure the prior report describes. (I also checked a 1-D-action / single-state version for clarity, and there too both conditionings descended back into the in-data band.) So the honest reading is: the conditioning controls the *sharpness* of `b` in the action, and FiLM is sharper, but I cannot claim from this toy alone that concat is catastrophically escapable. The full failure must involve the closed loop a static check can't see — high-dimensional states dominating the concat input even more than in my toy, the actor and predictor co-adapting over the whole training run, and the actor exploiting whatever shallow low-error directions exist. I'll proceed treating "sharper conditioning is better" as supported and "concat is fatally escapable" as a plausible-but-unverified explanation of the prior report, to be settled by the actual D4RL ablation rather than by my toy.

This still reframes the problem usefully. Whatever the exact severity, I want the predictor's error to be a *sharp* function of the action — genuinely small on dataset actions and rising fast the moment the action leaves the data — and the lever for that is the architecture by which the action conditions the network, not the loss. Concatenation mixes the action into the first linear layer and then lets it diffuse, which is precisely the flat-in-the-action behavior I measured. I want the action to *control the computation* over the state more forcefully. Feature-wise linear modulation does this: let one input (the state) be the feature stream flowing through the MLP, and let the other input (the action) produce, through its own small linear map, per-unit scale and shift parameters `(γ, β)` that multiply and offset a hidden layer — `h ← γ ⊙ h + β`. Now the action does not merely add a few input dimensions; it *reshapes* the function the network computes over the state. A different action yields a different `(γ, β)` and hence a different target embedding to match, which is the mechanism behind the larger `‖∂b/∂a‖` I measured for FiLM. (Equivalently one can swap the roles — action as feature, state as context — which is the form the reference uses; the point is the same multiplicative conditioning.) Whether this multiplicative gating is *enough* to close the escape route entirely is the thing my toy could not settle and the thing the ablation must — concatenation against FiLM (and against gated/bilinear variants, at the first vs. last hidden layer), watching the OOD-detection quality move. I'm committing to FiLM conditioning of the RND prior as the change most likely to matter, while being clear it is an *architecture* fix, not a new objective.

With a `b` that is at least sharper in the action, I have to wire it in, and the placement matters as much as the signal. There are two distinct routes by which OOD actions corrupt the value, and I want to close both. The first is the actor: at a state `s` the policy proposes `π(s)`, which may be OOD, and the actor objective rewards climbing `Q(s, π(s))`. So in the actor loss I subtract `β·b(s, π(s))` — the policy is paid to climb `Q` but charged for unfamiliarity, so it is pulled toward actions that are both high-value and in-distribution. The second route is the bootstrap itself: the critic target evaluates `Q(s', a')` at the *next* action `a'` the policy samples, which can already be OOD regardless of what the actor is doing this step, and that inflated next-value backs up through every Bellman step. So I subtract `β·b(s', a')` *inside the critic target* as well: `target = r + γ(1−done)·[min_i Q_i(s', a') − α·logπ(a'|s') − β·b(s', a')]`. Penalizing at both the actor and the target is the anti-exploration mirror of how an online bonus would be added to both the policy reward and the value — it suppresses the over-valuation at its source (the backup) and steers the policy away from it (the actor) at once.

A couple of details that are easy to get wrong and matter. The bonus scale drifts during training — early on the predictor matches nothing and `b` is huge; later it shrinks. I can see this in the toy: at initialization the OOD bonus was `O(0.3–0.9)` and after training the in-data error was `O(10⁻⁵)`, four orders of magnitude apart, so a fixed `β` would mean wildly different effective conservatism over a run. I normalize: divide `b` by a running standard deviation of the raw distillation error, so `β` controls a stable, unit-free penalty. The target network must stay *frozen* (no gradient) — it is the fixed random reference; only the predictor trains, and only on dataset `(s, a)`, never on the actor's proposed actions, or the system could game itself by making the reference easy. The SAC base keeps its automatic entropy temperature `α` tuned to a target entropy of `−dim(A)`; the twin critics keep LayerNorm and the `min`; targets are Polyak-updated. And `β` is the one knob that genuinely needs per-dataset setting, because how aggressively to flee OOD actions depends on how much the dataset covers — a dataset of near-expert trajectories wants a light penalty (stay close, exploit), a broad mediocre dataset can tolerate or even needs a heavier one. So `β` is swept per dataset rather than fixed.

Let me write the pieces. The RND module first — a FiLM-conditioned predictor and a frozen FiLM target, both emitting a small embedding, with a running-std buffer for the bonus:

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FiLMMLP(nn.Module):
    """MLP over `feature`, FiLM-modulated by `context`: the context's affine
    (gamma, beta) scale/shift the first hidden layer. Sharp conditioning."""
    def __init__(self, feature_dim, context_dim, out_dim, hidden=256):
        super().__init__()
        self.film = nn.Linear(context_dim, 2 * hidden)
        self.l1 = nn.Linear(feature_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, feature, context):
        gamma, beta = torch.chunk(self.film(context), 2, dim=-1)
        h = F.relu(gamma * self.l1(feature) + beta)
        h = F.relu(self.l2(h))
        return self.out(h)


class RND(nn.Module):
    """Anti-exploration novelty over (s, a). Action is the FiLM feature, state the
    context: the prior is conditioned on the action so the bonus is escapable only
    by returning to in-distribution actions. Target frozen; predictor trained."""
    def __init__(self, obs_dim, act_dim, embedding_dim=32):
        super().__init__()
        self.predictor = FiLMMLP(act_dim, obs_dim, embedding_dim)
        self.target = FiLMMLP(act_dim, obs_dim, embedding_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.register_buffer("rms_var", torch.ones(()))

    def _embed(self, s, a):
        pred = self.predictor(a, s)
        with torch.no_grad():
            targ = self.target(a, s)
        return pred, targ

    def bonus(self, s, a):
        pred, targ = self._embed(s, a)
        return ((pred - targ) ** 2).sum(-1) / (torch.sqrt(self.rms_var) + 1e-8)

    def distill_loss(self, s, a):
        pred, targ = self._embed(s, a)
        raw = ((pred - targ) ** 2).sum(-1)
        with torch.no_grad():
            self.rms_var.mul_(0.99).add_(0.01 * raw.var(unbiased=False))
        return raw.mean()
```

The SAC actor (Tanh-Gaussian with the squashing log-prob correction) and the LayerNorm twin critic:

```python
class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_sigma = nn.Linear(hidden, act_dim)

    def sample(self, s):
        h = self.trunk(s)
        dist = Normal(self.mu(h), self.log_sigma(h).clamp(-5.0, 2.0).exp())
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1) - torch.log(1 - torch.tanh(raw) ** 2 + 1e-6).sum(-1)
        return torch.tanh(raw), logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)
```

And the update, with the penalty subtracted in both places:

```python
def update(batch, actor, critics, critic_targs, rnd, log_alpha, target_entropy,
           opts, beta, gamma, tau):
    s, a, r, s2, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
    alpha = log_alpha.exp()

    rnd_loss = rnd.distill_loss(s, a)                              # predictor on dataset (s, a)
    opts["rnd"].zero_grad(); rnd_loss.backward(); opts["rnd"].step()

    with torch.no_grad():                                          # critic target: minus beta * bonus
        a2, logp2 = actor.sample(s2)
        q2 = torch.min(critic_targs[0](s2, a2), critic_targs[1](s2, a2))
        q2 = q2 - alpha * logp2 - beta * rnd.bonus(s2, a2)
        target = r + (1 - done) * gamma * q2
    c_loss = sum(F.mse_loss(c(s, a), target) for c in critics)
    opts["critic"].zero_grad(); c_loss.backward(); opts["critic"].step()

    pi, logp = actor.sample(s)                                     # actor: minus beta * bonus
    q_pi = torch.min(critics[0](s, pi), critics[1](s, pi))
    a_loss = (alpha.detach() * logp + beta * rnd.bonus(s, pi) - q_pi).mean()
    opts["actor"].zero_grad(); a_loss.backward(); opts["actor"].step()

    alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()  # auto-tune temperature
    opts["alpha"].zero_grad(); alpha_loss.backward(); opts["alpha"].step()

    for c, ct in zip(critics, critic_targs):                      # Polyak
        for p, tp in zip(c.parameters(), ct.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
```

The chain, end to end: offline, bootstrapping over the policy's OOD next actions over-values them and the policy chases the inflation, so the cure is an *uncertainty* penalty — the anti-exploration mirror of an online novelty bonus. RND gives a cheap, ensemble-free novelty signal (frozen target, trained predictor, squared error), but how the action enters the prior controls how sharp `b` is in the action: with `[s,a]` concatenation the action is the input minority (~17% of the input norm in my check) and the bonus is measurably flatter in the action than under multiplicative conditioning — the quantitative core of the "not discriminative enough" report, even if my static toy could not reproduce its full severity. The fix is therefore the *conditioning*: FiLM-modulate the prior by the action so the predictor's error is a sharper function of the action — the choice most likely to make `b` minimizable only by staying in-distribution, to be confirmed by the concat-vs-FiLM ablation. Compute `b(s,a) = ‖ĝ−g‖²` normalized by a running std, and subtract `β·b` from *both* the actor objective (steer the policy in-distribution) and the critic bootstrap target (suppress the over-valuation at its source), on a SAC base with auto-tuned entropy, twin LayerNorm critics, and a per-dataset `β`. One small extra network buys the pessimism an ensemble would otherwise charge `N` critics for.
