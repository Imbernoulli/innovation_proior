Let me start from the one disease that organizes all of offline RL, because the method has to be built directly on top of it. I have a fixed dataset `D` of transitions collected by some behavior policy, no environment to query, and I want a policy better than the one that collected the data. The standard machinery is a bootstrapped critic: regress `Q(s,a)` toward `r + γ Q(s', a')` where `a'` is whatever the policy proposes at the next state. Online this self-corrects, because if the critic over-values an action the agent will eventually try it and see the truth. Offline there is no "eventually." The actor proposes actions that were never in `D` at `s'`, the critic has no data there and extrapolates — and the extrapolation is almost always *upward*, because the actor is optimized to climb wherever the critic bulges, and the bulges are disproportionately the positive-error regions. The inflated value backs up through the Bellman recursion, the policy chases it, and nothing ever pulls it back. So every offline method is, at heart, a way of keeping the value (and the policy) honest on actions the data does not support.

The cleanest way to *say* what I want is in the language of uncertainty. If I had, for every `(s, a)`, a reliable number `b(s,a)` measuring how unfamiliar that action is at that state — small on the data, large off it — then I could simply subtract a multiple of it from the value and from the policy objective: be pessimistic exactly where I am ignorant. This is the anti-exploration framing, and it is worth dwelling on because it tells me where to look for `b`. Online exploration *adds* a novelty bonus `+b` to chase the unknown; offline I want to *subtract* the same novelty signal `−b` to flee the unknown. The two are the same estimator with opposite sign. So the question becomes: what is a good, cheap novelty estimator over `(s, a)` pairs?

The ensemble answer is known and strong. Keep `N` critics, initialize them differently, and use their disagreement at `(s, a)` as the uncertainty — where they agree, the data pinned them down; where they scatter, it is OOD. SAC-N and EDAC do exactly this and are near the top of D4RL. But the uncertainty signal *is* the ensemble, so the cost scales with `N` — ten, fifty critics — and in a setting where I am told the parameter budget is capped and the contribution must be algorithmic rather than capacity, an `N`-critic ensemble is precisely the move I cannot make. I want ensemble-quality pessimism from a *single* small extra network.

The canonical ensemble-free novelty signal in online RL is Random Network Distillation, and it is cheap in exactly the way I need. A fixed, randomly-initialized *target* network `g(x)`, and a *predictor* `ĝ(x)` trained by regression to match `g` on the inputs I actually observe. On inputs that appear often in training the predictor learns to mimic the target and the error `‖ĝ(x) − g(x)‖²` goes to zero; on inputs never seen the predictor was never trained there and the error stays large. One frozen network, one trained network, a squared error. Online this drives exploration; the natural question is whether the *same* squared error, computed over `(s, a)` inputs and trained only on dataset `(s, a)`, behaves as a clean OOD-action detector — small on in-data actions, large on the OOD actions the offline policy must be kept away from.

So the first concrete design is direct. Take SAC as the continuous base — a Tanh-Gaussian actor, twin critics with a `min` target, auto-tuned entropy — because its stochastic actor already samples the actions I need to score. Add an RND module whose predictor and target both take `(s, a)` as input and emit an embedding; train the predictor on dataset `(s, a)` by MSE to the frozen target; define `b(s,a) = ‖ĝ(s,a) − g(s,a)‖²`; and subtract `β·b` wherever the policy could escape to OOD actions. Then run it.

There is a prior report that plain offline RND is "not discriminative enough" as an OOD detector, and before I trust or discard this design I want to pin down exactly which failure that phrase could name, because it hides two different diseases with two different cures. One is that the bonus itself is a bad detector — score a dataset action and a wildly OOD action and the two numbers come back too close to separate. The other is that the detector is fine on its own but the actor, once the bonus is subtracted into its training objective and pushed through gradient descent, cannot actually use it — a landscape can be accurate everywhere and still hand an optimizer nothing it can walk downhill on toward the data. These call for different fixes, so I want a way to tell them apart before touching the architecture.

The detection question has a clean, contained test: freeze the concat predictor and target right after pretraining on dataset `(s,a)`, with no actor and no bootstrapping anywhere nearby, and score the bonus along a dial of distributional shift — the dataset actions themselves, actions with Gaussian noise added at a few increasing scales, and actions drawn from a wide uniform spread. A bad detector gives a bonus that sits roughly flat across that whole dial, unable to separate a barely-perturbed action from a wildly OOD one. A good detector gives a bonus that climbs with the shift, and the bar I hold it to is whether that climb comes close to what a trained critic-ensemble's disagreement gives on the identical probe — an ensemble is expensive to train but its disagreement is a signal I already trust as a ruler for "good separation." If the bonus tracks the shift at anything like that level, detection is cleared as a suspect and the real defect has to be downstream, in what the actor's gradient descent does with an already-good signal rather than in the signal itself.

That downstream question needs its own isolated test, because inside full SAC the actor's behavior is explained by two things moving at once — the critic it is climbing, and whatever the bonus's own landscape does to a gradient step through it — and I cannot tell which is responsible while both are live. Strip the confound the same way: keep the entropy term, so the actor still has a reason to stay stochastic, but drop the critic entirely, leaving only "minimize `β·b(s, π(s))`" against an RND pair already pretrained and frozen on the dataset. Now the actor has exactly one thing to optimize. An actor facing a landscape it can actually walk down should converge toward the same low bonus a dataset action already gets — since that bonus is near-zero exactly on the data by construction — and its distance to real dataset actions should shrink as training proceeds. An actor facing a landscape gradient descent cannot reliably navigate can settle into any nearby direction that locally lowers the bonus without that direction pointing back at the data at all; the bonus can plateau above the achievable floor while the actor's distance to the dataset actions has no reason to shrink, and can just as easily grow.

What decides which of those two outcomes a given conditioning produces is architectural: the actor's gradient step through `b(s,a)` is, at every point, following `-∂b/∂a`, the anti-gradient field over the action at that state, and whether that field composes into a path back to the data depends on how the action enters the network, not on how accurate the frozen bonus is at any single point. Concatenation folds the action into the same first linear layer as the state and lets it diffuse through the rest of the MLP with no structural reason for nearby actions to produce nearby gradients — the field can be locally inconsistent even where the bonus's raw values are a good detector, exactly the gap the detection-only probe cannot see. So the decision rule for the critic-free test is concrete: whichever conditioning lets that actor actually reach the dataset-level bonus floor and keep closing its distance to real dataset actions, instead of plateauing above it or drifting away, is the one that belongs in the prior.

Feature-wise linear modulation is built for exactly the property that test rewards: let the state be the feature stream flowing through the MLP, and let the action produce, through its own small linear map, per-unit scale and shift `(γ, β)` that multiply and offset a hidden layer — `h ← γ ⊙ h + β`. The action no longer adds a few input dimensions that later layers may or may not preserve; it reshapes the function the network computes over the state, so a small change in the action produces a correspondingly small, smooth change in every downstream activation instead of one more entry mixed into a big first-layer dot product — a field the actor's gradient descent can actually integrate into a path toward the data, which is exactly what concatenation's diffuse, unstructured mixing does not guarantee. (One can swap the roles — action as feature, state as context — which is the form the code below uses; the multiplicative-conditioning principle is the same.) I'm committing to FiLM conditioning of the RND prior on this basis: it targets the optimization landscape the actor has to descend, not the objective, and it is not a claim that concatenation was ever blind to OOD actions — the detection probe already ruled that out.

With a prior conditioned so that an actor's gradient descent through `b` can actually reach the data, I still have to wire it in, and the placement matters as much as the signal. There are two distinct routes by which OOD actions corrupt the value, and I want to close both. The first is the actor: at a state `s` the policy proposes `π(s)`, which may be OOD, and the actor objective rewards climbing `Q(s, π(s))`. So in the actor loss I subtract `β·b(s, π(s))` — the policy is paid to climb `Q` but charged for unfamiliarity, so it is pulled toward actions that are both high-value and in-distribution. The second route is the bootstrap itself: the critic target evaluates `Q(s', a')` at the *next* action `a'` the policy samples, which can already be OOD regardless of what the actor is doing this step, and that inflated next-value backs up through every Bellman step. So I subtract `β·b(s', a')` *inside the critic target* as well: `target = r + γ(1−done)·[min_i Q_i(s', a') − α·logπ(a'|s') − β·b(s', a')]`. Penalizing at both the actor and the target is the anti-exploration mirror of how an online bonus would be added to both the policy reward and the value — it suppresses the over-valuation at its source (the backup) and steers the policy away from it (the actor) at once.

A couple of details that are easy to get wrong and matter. The bonus scale drifts across training for a structural reason, not an incidental one: the predictor is trained by regression toward the target on dataset `(s,a)` only, so as that regression converges its error on the data it was trained on falls toward the objective's own minimum, while nothing pulls the error down on actions it never saw. Early in training, before the predictor has matched anything, the bonus is large everywhere, dataset actions included; well into training, the in-data bonus can be orders of magnitude smaller than it started while the OOD bonus has no such pull dragging it down with it. A fixed `β` multiplying a quantity that shrinks by orders of magnitude over the run would mean wildly different effective conservatism at the start versus the end of training. I normalize: divide `b` by a running standard deviation of the raw distillation error, so `β` controls a stable, unit-free penalty. The target network must stay *frozen* (no gradient) — it is the fixed random reference; only the predictor trains, and only on dataset `(s, a)`, never on the actor's proposed actions, or the system could game itself by making the reference easy. The SAC base keeps its automatic entropy temperature `α` tuned to a target entropy of `−dim(A)`; the twin critics keep LayerNorm and the `min`; targets are Polyak-updated. And `β` is the one knob that genuinely needs per-dataset setting, because how aggressively to flee OOD actions depends on how much the dataset covers — a dataset of near-expert trajectories wants a light penalty (stay close, exploit), a broad mediocre dataset can tolerate or even needs a heavier one. So `β` is swept per dataset rather than fixed.

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

The chain, end to end: offline, bootstrapping over the policy's OOD next actions over-values them and the policy chases the inflation, so the cure is an *uncertainty* penalty — the anti-exploration mirror of an online novelty bonus. RND gives a cheap, ensemble-free novelty signal (frozen target, trained predictor, squared error) that, checked in isolation, already discriminates in-data from OOD actions about as well as an ensemble does — so the naive design's failure is not detection but minimization: how the action enters the prior decides whether an actor's gradient descent through `b` composes into a path back to the data, and `[s,a]` concatenation gives that descent no structural reason to behave smoothly as the action varies. The fix is therefore the *conditioning*: FiLM lets the action reshape the network's computation over the state multiplicatively, so nearby actions produce nearby gradients and an actor minimizing `b` can actually walk itself back toward the data instead of stalling or drifting on an inconsistent field. Compute `b(s,a) = ‖ĝ−g‖²` normalized by a running std, and subtract `β·b` from *both* the actor objective (steer the policy in-distribution) and the critic bootstrap target (suppress the over-valuation at its source), on a SAC base with auto-tuned entropy, twin LayerNorm critics, and a per-dataset `β`. One small extra network buys the pessimism an ensemble would otherwise charge `N` critics for.
