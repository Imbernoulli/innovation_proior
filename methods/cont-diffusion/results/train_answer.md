On a parallel-decoding, action-chunking vision-language-action trunk, the most natural way to produce continuous actions is to read each action-slot hidden state through a small head and regress it to the demonstration action. That design is fast and preserves full continuous precision, but it is intrinsically a point estimator. Minimizing mean absolute error learns the conditional median; minimizing mean squared error learns the conditional mean. When robot demonstrations are multimodal—when the same observation admits several equally good next actions, such as swerving left or right around an obstacle—a single vector has to compromise between the modes, and the compromise can be invalid. This is not a tuning issue; it is a representational limitation of outputting one vector per observation.

The alternatives each have their own shortcomings for this scaffold. Discretizing each action dimension into 256 bins and predicting them as language tokens keeps the model's native cross-entropy objective, but generation becomes sequential across the chunk, which defeats the purpose of parallel decoding and action chunking. Training a separate diffusion policy from scratch can model multimodal action distributions, but it does not reuse the pretrained VLA trunk and its already-fused vision-language-proprioception conditioning; the question here is how to obtain that generative capability inside the existing trunk, whose only handle is the action-slot hidden states. What is needed is a head that turns the trunk into a conditional distribution over action chunks and then samples from it, while training with a simple regression-style loss and no adversary or partition function.

The method is Cont-Diffusion. It treats the action chunk as a continuous signal and learns to reverse a fixed Gaussian corruption process, exactly as denoising diffusion models do for images, but conditioned on the VLA trunk instead of on a separate side network. The forward process corrupts a clean action chunk x_0 in one shot: x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) epsilon, where epsilon is standard Gaussian noise and bar_alpha_t comes from a cosine noise schedule. A network predicts the noise epsilon_theta(x_t, t, observation), and training minimizes the mean squared error between the predicted noise and the true noise. Because the model learns to denoise rather than to output one action, it can represent multiple valid action modes and sample different ones from different initial noise.

The crucial adaptation to the VLA scaffold is the conditioning. The noisy action chunk is flattened into K times 7 scalars, each projected to the trunk hidden dimension, and fed back into the trunk as action-token features. A single sinusoidal timestep embedding token is added so the trunk knows the noise level. Bidirectional attention in the trunk then mixes the noisy actions with vision, language, and proprioception. The noise prediction is read off the resulting action-slot hidden states by a small MLP-residual head that groups the seven hidden states belonging to each timestep and maps them to that timestep's seven-dimensional noise. During inference, a normalized action chunk is initialized from standard Gaussian noise and denoised through the reverse process. DDIM sampling with the squared-cosine schedule is used so the same trained model can be evaluated on a shorter subsequence of timesteps, keeping the latency-quality tradeoff explicit. Each reverse step costs one trunk pass, so the method trades extra latency for the ability to model a full conditional action distribution.

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
    "lr_warmup_steps": 800,
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
