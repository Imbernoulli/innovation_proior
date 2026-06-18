# Cont-Diffusion: a conditional denoising-diffusion action head on a parallel-decoding VLA

## Problem

On a parallel-decoding, action-chunking vision-language-action trunk, the simplest continuous
action head regresses the action-slot hidden states to a single vector per timestep (L1 → median,
L2 → mean). Robot demonstrations can be multimodal, so a point estimator can collapse incompatible
modes to one invalid compromise action. **Cont-Diffusion** replaces the point head with a
generative one that represents the full conditional distribution `p(action chunk | observation)`
and samples from it, while reusing the existing trunk as the conditioning network.

## Key idea

Train the trunk-plus-head to **reverse a fixed Gaussian corruption** of the action chunk
(denoising diffusion). The clean chunk `x_0` is noised in one shot via the closed-form forward
marginal; a network predicts the added noise; at inference the policy starts from pure noise and
denoises back to a clean chunk. The conditioning is supplied by feeding the current noisy chunk
back through the trunk (each noisy scalar injected as one action-token embedding) plus a
timestep token, so the bidirectional trunk re-reads the noisy actions against the fused
vision-language-proprio context at every diffusion step.

## Final form

Notation: `alpha_t = 1 - beta_t`, `bar_alpha_t = prod_{s<=t} alpha_s`, chunk size `K`, action
dim `D = 7`.

- **Schedule.** Squared-cosine (`squaredcos_cap_v2`), `bar_alpha_t = f(t)/f(0)`,
  `f(t) = cos^2(((t/T + s)/(1+s)) * pi/2)` — keeps `bar_alpha_t` smooth at both ends for
  the control-data denoising schedule used by the code path.
- **Forward (one shot).** `x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) eps`,
  `eps ~ N(0, I)`.
- **Training objective (epsilon-prediction MSE).**
  `L = E_{x_0, eps, t} || eps - eps_theta(x_t, t, observation) ||^2`,
  where `eps_theta` is the trunk forward pass (noisy chunk + timestep injected) followed by the
  MLP noise predictor on the action-slot hidden states.
- **Sampler (DDIM, deterministic by default, step-skippable).** From `x_T ~ N(0, I)`, for each
  selected timestep `t`, form `x0_hat = (x_t - sqrt(1 - bar_alpha_t) eps_theta)/sqrt(bar_alpha_t)`.
  If `s` is the next lower selected timestep, take
  `x_s = sqrt(bar_alpha_s) x0_hat + sqrt(1 - bar_alpha_s - sigma_t^2) eps_theta + sigma_t z`.
  The scheduler call leaves `eta` at its default `0`, so `sigma_t = 0` and the update is
  deterministic. Inference latency = one trunk pass per selected step.

Components: a per-scalar **noisy-action projector** (`R^1 -> R^d`, one embedding per action
scalar, matching the trunk's one-slot-per-action-dimension layout), a **sinusoidal timestep
encoder** (one extra token), and an **MLP-residual noise predictor** (pre-LN residual blocks)
that maps the 7 per-dimension hidden states of each timestep to that timestep's 7-D noise.

## Code

```python
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from mlsbench.action_method_runtime import (
    EvalActionMethodRuntime,
    FIXED_NUM_DIFFUSION_STEPS_INFERENCE,
    FIXED_NUM_DIFFUSION_STEPS_TRAIN,
    ForwardPassResult,
    TrainActionMethodRuntime,
)
from prismatic.models.action_heads import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    NoisePredictionModel,
    SinusoidalPositionalEncoding,
)
from prismatic.models.projectors import NoisyActionProjector


CONFIG_OVERRIDES = {
    "lr_warmup_steps": 800,        # diffusion co-trains trunk + denoiser; needs more warmup
    "num_steps_before_decay": 6000,
}


class CustomActionMethod(nn.Module):
    """Conditional denoising-diffusion action head (Cont-Diffusion)."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        self.noisy_action_projector = NoisyActionProjector(llm_dim=input_dim)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=FIXED_NUM_DIFFUSION_STEPS_TRAIN,
            beta_schedule="squaredcos_cap_v2",
        )
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def extract_action_hidden_states(self, runtime, forward: ForwardPassResult) -> torch.Tensor:
        if runtime.phase == "train":
            mask = runtime.current_action_mask | runtime.next_actions_mask
            return (
                forward.text_hidden_states[mask]
                .reshape(runtime.batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(runtime.hidden_dtype)
            )
        start = runtime.num_prompt_tokens
        end = start + NUM_ACTIONS_CHUNK * ACTION_DIM
        return forward.text_hidden_states[:, start:end, :].to(runtime.hidden_dtype)

    def build_action_token_features(self, runtime, noisy_actions: torch.Tensor) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        flattened = noisy_actions.reshape(batch_size, -1).unsqueeze(-1)  # (B, K*7, 1)
        return self.noisy_action_projector(flattened).to(runtime.hidden_dtype)

    def sample_noisy_actions(self, ground_truth_actions):
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        noise = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=device, dtype=ground_truth_actions.dtype,
        )
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,), device=device,
        )
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)
        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(device).unsqueeze(1)
        return {
            "noise": noise,
            "noisy_actions": noisy_actions,
            "diffusion_timestep_embeddings": timestep_embeddings,
        }

    def predict_noise(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.noise_predictor(rearranged)

    def _rollout_diffusion(self, runtime):
        curr = torch.randn(
            size=(runtime.batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=runtime.device, dtype=runtime.hidden_dtype,
        )
        self.noise_scheduler.set_timesteps(FIXED_NUM_DIFFUSION_STEPS_INFERENCE)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((runtime.batch_size,), int(t), device=runtime.device, dtype=torch.long)
            timestep_embeddings = self.time_encoder(timesteps).to(runtime.hidden_dtype).unsqueeze(1)
            action_token_features = self.build_action_token_features(runtime, curr)
            forward = runtime.forward(
                action_token_features=action_token_features,
                diffusion_timestep_embeddings=timestep_embeddings,
            )
            actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
            noise_pred = self.predict_noise(actions_hidden_states).reshape(curr.shape)
            curr = self.noise_scheduler.step(noise_pred, t, curr).prev_sample
        return curr

    def training_step(self, runtime: TrainActionMethodRuntime):
        noisy = self.sample_noisy_actions(runtime.ground_truth_actions)
        action_token_features = self.build_action_token_features(runtime, noisy["noisy_actions"])
        forward = runtime.forward(
            action_token_features=action_token_features,
            diffusion_timestep_embeddings=noisy["diffusion_timestep_embeddings"],
        )
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        noise_pred = self.predict_noise(actions_hidden_states).reshape(noisy["noise"].shape)
        loss = nn.functional.mse_loss(noise_pred, noisy["noise"], reduction="mean")
        metrics = {"loss_value": float(loss.item())}
        if runtime.should_compute_aux_metrics:
            with torch.no_grad():
                predicted_actions = self._rollout_diffusion(runtime)
            metrics.update(runtime.compute_action_l1_metrics(predicted_actions))
        return loss, metrics

    def predict_actions(self, runtime: EvalActionMethodRuntime):
        normalized_actions = self._rollout_diffusion(runtime)
        return normalized_actions, None
```
