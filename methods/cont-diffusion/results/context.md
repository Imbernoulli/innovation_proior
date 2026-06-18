# Context: learning objectives for a continuous action head on a parallel-decoding VLA

## Research question

We have a 7B vision-language-action model (a pretrained vision-language model fine-tuned for
low-level robot control) whose decoder has already been converted from token-by-token
autoregression to **parallel decoding with action chunking**: instead of emitting one discrete
action token at a time under a causal mask, the trunk receives a block of placeholder action
slots and, under bidirectional attention, maps them in a single forward pass to a chunk of `K`
future control vectors. Each control vector is a 7-D delta end-effector pose (3 position, 3
orientation, 1 gripper), normalized to `[-1, +1]`. The trunk's final hidden states at the
action slots are the only handle we have on the actions.

The open problem is the **local action method**: the algorithm that turns those action-slot
hidden states into a chunk of real-valued actions, together with the objective that trains it.
A first answer is to read each hidden state straight through a small head to a continuous vector
and regress it to the demonstration. That works, but robot demonstration data is frequently
**multimodal** — at a given observation an expert may have several equally valid ways to proceed
(go left or right around an obstacle; regrasp now or later) — and a head trained to output one
vector per observation is structurally a point estimator (mean under L2, median under L1). When
the conditional action distribution `p(actions | observation)` has multiple modes, a single
compromise action can be invalid (the midpoint between "left" and "right" drives into the
obstacle). The goal is a local action method, sitting on the same parallel-decoding trunk and
trained from the same fixed offline demonstrations, that can represent a **multimodal**
conditional action distribution rather than collapsing it to a point — without changing the base
model, the chunk size, or the single-pass decoding that makes the policy fast.

## Background

**The base policy and what is already fixed.** The trunk is a fused dual vision encoder
(SigLIP + DINOv2) feeding a Llama-2 7B decoder through an MLP projector; robot proprioceptive
state is projected into the same embedding space as one extra token; multiple camera views each
contribute 256 patch embeddings. Parallel decoding (empty action embeddings that differ only in
their positional encoding, fed as input, with the causal mask replaced by bidirectional
attention) is borrowed from action-chunking imitation learning (Zhao et al. 2023, ACT) and lets
the decoder predict all `K·7` action scalars at once. Fine-tuning is parameter-efficient (LoRA),
because the downstream demonstration sets are tiny (hundreds of trajectories) relative to
pretraining. None of this is in question here; the question is purely the head and objective
hanging off the action-slot hidden states.

**Behavioral cloning and the multimodality problem.** Imitation by regression minimizes a
per-sample distance between a single predicted action and the logged action. Under a squared
(L2) loss the optimum is the conditional mean `E[a | o]`; under an absolute (L1) loss it is the
conditional median. Both are *point* estimators. It is a documented failure mode of behavioral
cloning that when the demonstrations are multimodal a point estimate can collapse the alternatives
into a low-density compromise action; this is exactly what motivated energy-based and generative
policy classes (Florence et al. 2021, Implicit BC) over plain regression heads.
So the limitation we must get past is intrinsic to fitting one deterministic vector per
observation, not to any particular trunk.

**Generative modeling of continuous data by gradual denoising.** Outside robotics, the
prevailing way to fit a complicated, possibly multimodal continuous distribution had converged
on a particular construction (Sohl-Dickstein et al. 2015; Ho, Jain & Abbeel 2020; "denoising
diffusion probabilistic models"). Fix a **forward process** that takes a clean datum `x_0` and
adds Gaussian noise over `T` steps with a variance schedule `beta_1..beta_T`, `q(x_t | x_{t-1})
= N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)`. Writing `alpha_t = 1 - beta_t` and `bar_alpha_t =
prod_{s<=t} alpha_s`, the forward process has a closed-form marginal at any step,

  x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) eps,    eps ~ N(0, I),

so noising to an arbitrary level is one operation. A model is trained to **reverse** this
process. Ho et al. (2020) showed the variational bound, after reparameterizing the reverse mean,
reduces to a simple denoising-score-matching-style objective: predict the noise that was added.
With a network `eps_theta`, the loss they actually train is

  L_simple = E_{x_0, eps, t}  || eps - eps_theta( sqrt(bar_alpha_t) x_0 + sqrt(1-bar_alpha_t) eps, t ) ||^2.

Sampling starts at `x_T ~ N(0, I)` and walks back down the chain; with the noise
parameterization each reverse step is

  x_{t-1} = (1/sqrt(alpha_t)) ( x_t - ((1-alpha_t)/sqrt(1-bar_alpha_t)) eps_theta(x_t, t) ) + sigma_t z.

The model never has to represent the density explicitly and is free to produce different samples
from different starting noise, so it can cover multiple modes.

**Two refinements to that construction that are pre-existing facts here.** First, the variance
schedule matters. A linear `beta_t` schedule (Ho et al.) destroys almost all signal in the last
fraction of the forward process, wasting steps; Nichol & Dhariwal (2021) proposed a **cosine**
schedule defined directly on the cumulative product,

  bar_alpha_t = f(t) / f(0),    f(t) = cos^2( ((t/T + s) / (1 + s)) * (pi/2) ),

which keeps `bar_alpha_t` changing slowly near `t = 0` and `t = T` and roughly linear in the
middle; later action-diffusion work adopted this schedule for control data. Second, the forward
chain used for training need not be the chain used for sampling: Song, Meng & Ermon (2020) showed
a family of **non-Markovian** processes share the same training objective, giving a generative
update that first forms a prediction of the clean datum,

  x0_hat = ( x_t - sqrt(1 - bar_alpha_t) eps_theta(x_t, t) ) / sqrt(bar_alpha_t),

and then steps to the previous selected noise level `s < t` as
`x_s = sqrt(bar_alpha_s) x0_hat + sqrt(1 - bar_alpha_s - sigma_t^2) eps_theta + sigma_t z`.
With `sigma_t = 0` the update is **deterministic**, and the same trained `eps_theta` can be sampled
on a sparse subsequence of timesteps, so far fewer reverse steps are needed than the number used in
the forward chain.

**Conditioning a denoiser on an observation, for control.** In robot imitation, this generative
construction had been applied to action sequences (Chi et al. 2023, "Diffusion Policy"). The key
adaptation is to model the *conditional* distribution `p(A_t | O_t)` of an action chunk `A_t`
given an observation `O_t`, rather than a joint over states and actions, by feeding the
observation into the noise predictor: `L = MSE(eps^k, eps_theta(O_t, A_t^k, k))`, where `A_t^k`
is the clean action chunk corrupted to noise level `k`. They
also used the cosine schedule and receding-horizon execution (predict a chunk, execute part of
it, replan), and reported that this generative policy class handled multimodal demonstrations
where regression collapsed. In those systems the observation is injected into a *dedicated*
conditioning network (FiLM on a 1-D conv U-Net, or cross-attention in a small transformer) that
is separate from the perception backbone.

## Baselines

- **Discrete autoregressive tokens (the base VLA's own recipe; Brohan et al. 2023, Kim et al.
  2024).** Each action dimension is normalized to `[-1, +1]` and uniformly binned into 256 bins;
  the decoder predicts the bins as language tokens with cross-entropy. Generation is sequential
  (`K·7` decoder passes for a chunk), so it is slow (3–5 Hz) and incompatible with cheap action
  chunking; the 256-bin discretization also throws away fine-grained action detail, and finer
  bins make each token rarer in the data and hurt generalization. It is a point predictor over a
  discretized grid: it can in principle place mass on several bins, but tying precision to bin
  count and paying `K·7` sequential passes makes it the thing the parallel-decoding redesign was
  meant to replace.

- **L1 regression head (the simplest continuous head; Zhao et al. 2023).** Replace the decoder's
  output layer with a small MLP that maps the action-slot hidden states directly to continuous
  actions, and train to minimize the mean absolute error to the demonstration. It is fast (one
  trunk pass, negligible head cost), keeps full continuous precision, and converges quickly. Its
  limitation is the one above: the absolute-error optimum is the conditional median, a single
  vector per observation, so on multimodal demonstrations it cannot represent the spread of valid
  actions and will sit between modes. It is the strong, fast point-estimate baseline that any
  distribution-representing method must justify its extra inference cost against.

- **External generative policies trained from scratch (Chi et al. 2023; Zhao et al. 2023).**
  Diffusion Policy and ACT are strong imitation policies but are *separate* small models with
  their own perception stacks, not heads on a large pretrained VLA. They establish that a
  multimodal action distribution helps, but they do not answer how to obtain that capability
  *inside* an already-pretrained, parallel-decoding VLA trunk whose only output handle is the
  action-slot hidden states, and whose conditioning (vision, language, proprio) is already fused
  in the trunk rather than supplied to a side network.

## Evaluation settings

The natural yardstick is the **LIBERO** simulation benchmark (Liu et al. 2024): a Franka Emika
Panda arm with demonstrations of camera images, robot state, language annotations, and delta
end-effector-pose actions, organized into task suites (spatial layouts, objects, goals,
long-horizon). Each suite provides a few hundred expert demonstrations across ten tasks. Policies
are fine-tuned per suite from the same base model; the metric is task **success rate** over many
rollout trials, and the protocol uses chunk size `K = 8` and executes a full chunk before
replanning. Inference efficiency (action-generation throughput and latency) is the second axis,
since a method that needs many forward passes per chunk pays a latency cost that a single-pass
head does not. The reduced-budget variant used here fixes the base model (a 7B VLA), LoRA rank,
batch size and gradient accumulation, learning rate (`5e-4`), warmup and decay schedule, and a
capped number of training steps, and evaluates success rate on disjoint task subsets.

## Code framework

The trunk, data pipeline, optimizer, and benchmark plumbing already exist. A read-only runtime
owns prompt/image preparation, the multimodal forward pass, action un-normalization, and metric
helpers; the local action method is a single module to fill in. The runtime's `forward` accepts
two optional method-supplied inputs — a tensor of per-action-token feature embeddings and a
tensor of auxiliary embeddings — and returns the trunk's final-layer `text_hidden_states`; the
action method decides how to extract the action-slot hidden states from those, what (if anything)
to inject as action-token features, and how to turn the result into a chunk of normalized actions.

```python
import torch
import torch.nn as nn

from mlsbench.action_method_runtime import (
    EvalActionMethodRuntime,
    ForwardPassResult,
    TrainActionMethodRuntime,
)

# Already-known constants for this scaffold.
ACTION_DIM = 7              # 3 position + 3 orientation + 1 gripper, per timestep
NUM_ACTIONS_CHUNK = 8       # K: timesteps predicted per forward pass

# Narrow knobs the method may set; everything else is fixed by the harness.
CONFIG_OVERRIDES = {
    # "lr_warmup_steps": ...,
    # "num_steps_before_decay": ...,
}


class CustomActionMethod(nn.Module):
    """Local action method: action-slot hidden states -> a chunk of normalized actions."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        # TODO: the modules the local action method needs.
        pass

    def extract_action_hidden_states(
        self,
        runtime: "TrainActionMethodRuntime | EvalActionMethodRuntime",
        forward: ForwardPassResult,
    ) -> torch.Tensor:
        # Select the action-slot hidden states out of forward.text_hidden_states.
        # TODO
        pass

    def training_step(self, runtime: "TrainActionMethodRuntime"):
        # Run runtime.forward(...), map the action-slot hidden states to a prediction,
        # and return (loss, metrics) against runtime.ground_truth_actions.
        # TODO
        pass

    def predict_actions(self, runtime: "EvalActionMethodRuntime"):
        # Produce a chunk of normalized actions for evaluation.
        # TODO
        pass
```
