**Problem.** A point-estimate continuous head (L1/L2) collapses a multimodal action distribution to a
single vector — the average of "swerve left" and "swerve right" is "into the obstacle." The action
method should represent `p(action chunk | observation)` and sample from it. The most expressive head
the PD&AC scaffold can host is a conditional denoising-diffusion decoder, so the ladder opens here.

**Key idea (Cont-Diffusion on the PD&AC trunk).** Learn to reverse a fixed Gaussian corruption of the
action chunk. Forward marginal `xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε` (one-shot), predict the noise `ε_θ(xₜ, t)`
with a trunk-conditioned denoiser, train with `MSE(ε_θ, ε)`, and sample by walking the reverse chain
from `N(0,I)` down to a clean chunk. Conditioning is the trunk itself: the noisy chunk is injected as
per-scalar action-token features through a `NoisyActionProjector` (Linear(1→d)→GELU→Linear(d→d)), the
timestep as one sinusoidal token, and the noise is read off the action-slot hidden states by a pre-LN
MLP-residual `NoisePredictionModel`. The scheduler is a DDIM sampler on a squared-cosine schedule
(slow at both ends, useful for low-range action data), `FIXED_NUM_DIFFUSION_STEPS_TRAIN = 50`,
`FIXED_NUM_DIFFUSION_STEPS_INFERENCE = 50`.

**Why it is the weakest rung here.** Inference is one trunk forward *per* diffusion step (~50 passes
through 7B per chunk), and — decisively — the denoiser is a far harder fitting problem than direct
regression, needing 100K-250K steps in the published regime to become competitive. This task fixes the
budget at 6000 steps for 5 H200h validity, so the reverse process is under-trained: sampled actions
stay close to noise and success sits near the floor, with eval times several times larger than a
single-pass head. Multimodality is a real advantage, but it is the advantage that most needs training
time the protocol does not grant.

**Hyperparameters.** `CONFIG_OVERRIDES`: `lr_warmup_steps = 800` (more warmup than a point head, since
trunk + denoiser co-train), `num_steps_before_decay = 6000`. `NoisePredictionModel` width `hidden_dim *
ACTION_DIM` in, `hidden_dim` working width, `action_dim = 7` out; `NoisyActionProjector(llm_dim = 4096)`;
`SinusoidalPositionalEncoding(dim = 4096)`; `DDIMScheduler(beta_schedule="squaredcos_cap_v2")`.

```python
"""MLS-Bench baseline: official OpenVLA-OFT PD&AC + Cont-Diffusion path."""

from __future__ import annotations

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
from prismatic.models.action_heads import ACTION_DIM, NUM_ACTIONS_CHUNK, NoisePredictionModel, SinusoidalPositionalEncoding
from prismatic.models.projectors import NoisyActionProjector


CONFIG_OVERRIDES = {
    "lr_warmup_steps": 800,
    "num_steps_before_decay": 6000,
}


class OfficialDiffusionActionMethod(nn.Module):
    """Official Cont-Diffusion baseline under the unified action-method API."""

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

    def extract_action_hidden_states(
        self,
        runtime: TrainActionMethodRuntime | EvalActionMethodRuntime,
        forward: ForwardPassResult,
    ) -> torch.Tensor:
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

    def build_action_token_features(
        self,
        runtime: TrainActionMethodRuntime | EvalActionMethodRuntime,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        flattened = noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
        return self.noisy_action_projector(flattened).to(runtime.hidden_dtype)

    def sample_noisy_actions(self, ground_truth_actions):
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        noise = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=device,
            dtype=ground_truth_actions.dtype,
        )
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=device,
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
        rearranged_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.noise_predictor(rearranged_hidden_states)

    def _rollout_diffusion(self, runtime: TrainActionMethodRuntime | EvalActionMethodRuntime):
        curr_noisy_actions = torch.randn(
            size=(runtime.batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=runtime.device,
            dtype=runtime.hidden_dtype,
        )
        self.noise_scheduler.set_timesteps(FIXED_NUM_DIFFUSION_STEPS_INFERENCE)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full(
                (runtime.batch_size,),
                int(t),
                device=runtime.device,
                dtype=torch.long,
            )
            timestep_embeddings = self.time_encoder(timesteps).to(runtime.hidden_dtype).unsqueeze(1)
            action_token_features = self.build_action_token_features(runtime, curr_noisy_actions)
            forward = runtime.forward(
                action_token_features=action_token_features,
                diffusion_timestep_embeddings=timestep_embeddings,
            )
            actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
            noise_pred = self.predict_noise(actions_hidden_states).reshape(curr_noisy_actions.shape)
            curr_noisy_actions = self.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample
        return curr_noisy_actions

    def training_step(self, runtime: TrainActionMethodRuntime):
        noisy_dict = self.sample_noisy_actions(runtime.ground_truth_actions)
        action_token_features = self.build_action_token_features(runtime, noisy_dict["noisy_actions"])
        forward = runtime.forward(
            action_token_features=action_token_features,
            diffusion_timestep_embeddings=noisy_dict["diffusion_timestep_embeddings"],
        )
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        noise_pred = self.predict_noise(actions_hidden_states).reshape(noisy_dict["noise"].shape)
        loss = nn.functional.mse_loss(noise_pred, noisy_dict["noise"], reduction="mean")
        metrics = {"loss_value": float(loss.item())}
        if runtime.should_compute_aux_metrics:
            with torch.no_grad():
                predicted_actions = self._rollout_diffusion(runtime)
            metrics.update(runtime.compute_action_l1_metrics(predicted_actions))
        return loss, metrics

    def predict_actions(self, runtime: EvalActionMethodRuntime):
        normalized_actions = self._rollout_diffusion(runtime)
        return normalized_actions, None


CustomActionMethod = OfficialDiffusionActionMethod
```
