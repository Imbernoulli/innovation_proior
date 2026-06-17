# Context: learning robot policies from demonstration (circa 2021-2022)

## Research question

We have a set of expert demonstrations — pairs of observation and action — and we want a policy
that reproduces the demonstrated behavior. In its simplest form this is supervised regression: learn
a map from the current observation `O` to an action `a`. The trouble is that robot actions are not a
well-behaved regression target. Three properties make them hard.

First, the demonstrations are **multimodal**: for the same observation there are often several
genuinely correct actions. To push a block toward a goal a person may go around it from the left or
from the right; both are valid, and the data contains both. A model that has to commit to one number
per observation is structurally wrong here — fit it with a squared-error loss and it will place the
prediction at the *average* of the demonstrated actions, which may be an action no expert ever took
and which can be exactly the wrong thing to do (drive straight into the block).

Second, actions are **sequentially correlated**. Good behavior is smooth and temporally consistent
over a horizon; the right action now depends on a commitment to a plan, not just on the instantaneous
observation. A policy that decides each step in isolation can flip between two valid plans on
consecutive steps and produce jittery, self-defeating motion.

Third, many tasks demand **high precision** — millimeter-scale insertion, careful pouring — so the
policy class must be able to represent sharp, confident action distributions, not just blurry means.

The precise goal: a policy class that (1) can represent an *arbitrary* conditional action
distribution `p(a | O)`, including sharp multimodal ones, so it never has to average incompatible
modes; (2) can produce temporally consistent action *sequences* rather than one myopic step; (3)
trains stably and reliably enough that we can pick a checkpoint without re-evaluating dozens of them
on hardware; and (4) is cheap enough at inference to run a closed loop in real time, including from
raw images. Each prior approach below buys a subset of these and pays for it somewhere else.

## Background

The dominant framing is behavior cloning: treat policy learning as supervised learning on
`<observation, action>` pairs and predict actions at test time. It has carried a surprising amount of
real-world manipulation and driving. The open question is the **form of the policy** — how the action
(or distribution over actions) is represented and supervised — because that form is what decides
whether multimodality, sequence structure, and precision can be captured at all.

Two field-wide observations about *existing* systems frame the problem. (a) A direct
observation→action regressor trained with an L2 loss behaves as if the action given the observation
were a single Gaussian; on demonstrations that branch into several modes it collapses to the mean of
the branches, and this shows up as the policy stalling at decision points or producing invalid
in-between actions. (b) Policies that model each timestep's action independently — even when each
step's distribution is itself multimodal — produce temporally inconsistent rollouts: consecutive
steps can be drawn from different modes, so the executed trajectory alternates between two otherwise
valid plans and never commits to either.

A separate body of work, developed for image and audio synthesis, is relevant as *machinery*. A
generative model can be defined by a **forward noising process** that gradually corrupts a clean data
point `x_0` into Gaussian noise over `K` steps, paired with a learned **reverse process** that
removes noise step by step to generate a sample. With Gaussian noising
`q(x^k | x^{k-1}) = N(√(1-β_k) x^{k-1}, β_k I)`, the corruption has a closed form at any level:
writing `α_k = 1-β_k` and `ᾱ_k = ∏_{s≤k} α_s`,

```
q(x^k | x^0) = N(√ᾱ_k · x^0, (1 - ᾱ_k) I),   i.e.   x^k = √ᾱ_k · x^0 + √(1-ᾱ_k) · ε,  ε ~ N(0,I).
```

The reverse step is modeled as a Gaussian `N(μ_θ(x^k,k), σ_k² I)`, and training minimizes a
variational bound on the data likelihood, which (because every term is a KL between Gaussians) reduces
to matching `μ_θ` to the tractable forward-process posterior mean. A separate line on **score
matching** (Song & Ermon 2019) showed that a network can instead be trained to estimate
`∇_x log q(x^k)`, the gradient of the log-density at each noise level, and that sampling can be done by
Langevin-style steps that follow this estimated gradient field plus injected noise. These two views —
denoising and score estimation — are known to be closely related. A practical detail: a **cosine**
noise schedule (Nichol & Dhariwal 2021), `ᾱ_k ∝ cos²((k/K + s)/(1+s) · π/2)`, distributes the noise
levels more usefully than a linear one and controls which frequencies of the signal the model attends
to. Conditioning machinery exists too: **FiLM** (Perez et al. 2018) modulates a convolutional feature
map channel-wise from a conditioning vector, and sinusoidal/positional embeddings inject the scalar
step index `k` into the network.

These generative tools were built to draw images. Whether and how they bear on representing a robot
*policy* — what plays the role of `x`, whether the observation enters the noising process or only
conditions it, what is even being generated — is open at this point.

## Baselines

These are the prior policy classes a new method would be measured against and would react to.

**Explicit regression policy (ALVINN, Pomerleau 1988; and most behavior cloning).** Map observation
directly to action with a feedforward network, supervised by L2 regression on the demonstrated
action. One forward pass per step, trivial to train. *Limitation:* the L2 objective corresponds to a
unimodal Gaussian likelihood, so it cannot represent multimodal demonstrations — it averages the
modes — and it struggles on high-precision tasks where the conditional action distribution is sharp.

**Discretized-action / classification policies (e.g. Transporter-style, Zeng et al. 2021).** Quantize
the action space into bins and predict a categorical distribution; a categorical *can* be multimodal.
*Limitation:* the number of bins needed to cover a continuous action space grows exponentially with
the action dimension, so this does not scale beyond low-dimensional or hand-designed action
primitives.

**Mixture-density / LSTM-GMM policies (robomimic, Mandlekar et al. 2021; MDN, Bishop 1994).** Predict
the parameters of a Gaussian mixture over the next action (optionally with a recurrent backbone for
history). A mixture is multimodal by construction. *Limitation:* the number of mixture components must
be chosen in advance; training is prone to mode collapse and is sensitive to hyperparameters; and
because each step's mixture is predicted independently, rollouts are temporally inconsistent — they
jitter between modes across consecutive steps.

**Clustering-plus-offset transformer policies (BET, Shafiullah et al. 2022).** k-means-cluster the
demonstrated actions, then predict a cluster and a continuous offset with a transformer. Handles
multimodality via the discrete cluster choice and scales to longer histories. *Limitation:* the
number of clusters must be specified, and the per-step modeling again leaves the executed sequence
without temporal action consistency, so it produces jittery actions that fail to commit to one plan.

**Implicit (energy-based) policy (IBC, Florence et al. 2021).** Represent the action distribution
implicitly with an energy `E_θ(o,a)`:

```
p_θ(a | o) = e^{-E_θ(o,a)} / Z(o,θ),     Z(o,θ) = ∫ e^{-E_θ(o,a)} da   (intractable in a).
```

An action is produced by *minimizing* the energy over `a` (by sampling or gradient descent at
inference). Because several actions can share low energy, an implicit policy naturally represents
multimodal and even set-valued maps, and it can express sharp, discontinuous decision boundaries that
an explicit regressor cannot. It is trained by an InfoNCE-style loss that equals the negative
log-likelihood of the EBM,

```
L_InfoNCE = -log [ e^{-E_θ(o,a)} / ( e^{-E_θ(o,a)} + Σ_{j=1}^{N_neg} e^{-E_θ(o, ã_j)} ) ],
```

where the negative samples `{ã_j}` are used to approximate the intractable normalizer `Z(o,θ)`.
*Limitation:* the quality of that approximation hinges on drawing good negatives, and inaccurate
negative sampling is a known cause of unstable EBM training — training error spikes and the
evaluation success rate oscillates throughout training. The practical cost of this instability is
that checkpoint selection becomes unreliable: one ends up evaluating many checkpoints on hardware to
choose a final policy.

**Trajectory-diffusion planner (Diffuser, Janner et al. 2022).** Applies the denoising-diffusion
machinery to *planning* by learning a diffusion model over the **joint** trajectory of states and
actions `p(A, O)`, and conditioning on goals via image-style inpainting of the trajectory. It
demonstrates that diffusion can model trajectory-level structure. *Limitation:* because the
observation/state is part of the modeled variable, the full (encoder and decoder) model must be run
at *every* denoising iteration, which is prohibitively expensive for real-time visuomotor control;
and inpainting-based goal conditioning is incompatible with a receding-horizon execution scheme.

## Evaluation settings

The natural yardsticks already in use at this time:

- **Simulated manipulation suites** under the behavior-cloning protocol — e.g. the robomimic tasks
  (Mandlekar et al. 2021), a planar pushing task (Push-T, from the IBC line), a block-pushing task,
  and a multi-stage kitchen task (Gupta et al. 2019) — spanning low-dimensional state observations and
  image observations, single- and multi-task, with demonstrations from one or several operators.
  Metric: task success rate (or, for Push-T, target-area coverage / IoU) over a fixed set of initial
  conditions.
- For **offline continuous control**, the D4RL benchmark (Fu et al. 2020): fixed datasets of
  transitions for MuJoCo locomotion environments (hopper, walker2d, halfcheetah), with the normalized
  score as the standard metric — train on a frozen buffer, then evaluate the policy by rolling it out
  in the environment.
- **Real-robot tasks** on standard arms (UR5, Franka) with teleoperated demonstrations and
  camera observations; success rate / coverage over repeated trials from matched initial conditions.
- Protocol points that matter for a fair comparison: action representation (position vs. velocity
  control; for rotations, axis-angle or a continuous 6D parameterization), action normalization,
  the observation horizon, and how a checkpoint is selected (a method that needs per-checkpoint
  hardware evaluation is at a real disadvantage). Inference latency is itself a constraint for any
  closed-loop real-time deployment.

## Code framework

The policy plugs into an existing imitation/offline-RL harness: a data pipeline that yields batches of
`<observation, action>` (already normalized into a bounded box), an optimizer and a cosine learning-rate
schedule, an exponential-moving-average copy of the weights for evaluation, and an evaluation loop that
resets a batch of environments, asks the policy for an action from the current observation, and steps.
What is *not* settled is the policy itself — the representation of the action distribution, how it is
trained, and how an action is produced at inference. That is the single empty slot.

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class PolicyCore(torch.nn.Module):
    """Backbone slot for the action model. The harness fixes only the tensor shapes:
    observations come in, action-shaped tensors come out."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs, work=None, step=None):
        # TODO: the model we will design.
        raise NotImplementedError


class Policy:
    """The policy owns the action-model core, optimizer state, an evaluation copy of
    the weights, and a bounded action box [-1, 1]."""

    def __init__(self, core, act_dim, lr, ema_rate=0.995, device="cpu"):
        self.core = core.to(device)
        self.core_ema = copy_and_freeze(core).to(device)
        self.opt = torch.optim.Adam(self.core.parameters(), lr=lr)
        self.act_low, self.act_high = -1.0, 1.0
        self.ema_rate = ema_rate

    def loss(self, act, obs):
        # TODO: the training objective we will design.
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, obs, n_samples):
        # TODO: the action-generation rule we will design.
        raise NotImplementedError

    @torch.no_grad()
    def ema_update(self):
        for p, pe in zip(self.core.parameters(), self.core_ema.parameters()):
            pe.mul_(self.ema_rate).add_(p, alpha=1 - self.ema_rate)


def train(policy, dataloader, gradient_steps, device, ema_update_interval=1):
    sched = CosineAnnealingLR(policy.opt, T_max=gradient_steps)
    for step, batch in zip(range(gradient_steps), loop(dataloader)):
        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)
        loss = policy.loss(act, obs)
        policy.opt.zero_grad(); loss.backward(); policy.opt.step()
        sched.step()
        if step % ema_update_interval == 0 and step >= 1000:
            policy.ema_update()


def evaluate(policy, env, normalizer, num_envs, device):
    obs = env.reset()
    while not done(env):
        obs_t = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)
        act = policy.sample(obs_t, n_samples=num_envs)
        obs, rew, done, info = env.step(act.clip(-1., 1.).cpu().numpy())
```

The harness supplies normalized `<obs, act>` batches and a place to read out an action per
environment; `loss`, `sample`, and the action representation they imply are what the method fills in.
