The parallel-decoding trunk hands me, at each of the $K\cdot D$ action slots, a final-layer hidden state already fused over vision, language, and proprioception. The cheapest decoder reads those hidden states through a small MLP and regresses a 7-D action against the logged demonstration. But a regression objective collapses a distribution to a point: minimizing absolute error picks the conditional *median*, minimizing squared error the conditional *mean*, and either way the head emits a single vector per observation. Manipulation demonstrations are not single-valued — at a fork the expert sometimes reaches left, sometimes right; the logged actions at nearly identical observations form a genuinely multimodal distribution, and the average of "swerve left" and "swerve right" is "drive into the obstacle." This is structural, not a tuning artifact: the head emits one vector while the world wants a distribution. So if multimodality is the deciding factor on this scaffold, the strongest head should be the most expressive one, and I open the ladder there to find out.

I propose **Cont-Diffusion** on the PD&AC trunk: a conditional denoising-diffusion decoder that models the whole distribution $p(\text{action chunk}\mid\text{observation})$ and samples from it. The construction trains with a plain regression-flavored loss — no adversary, no partition function, no negative samples — yet still draws from a multimodal distribution, which is exactly what rules out the alternatives I considered (a mixture density network fixes the component count up front and is prone to mode collapse; an energy-based head needs MCMC and negative samples at inference, bolting a second optimization loop onto a 7B trunk).

The mechanism is to wreck a clean action chunk with Gaussian noise a little at a time over a long chain, $q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I)$ for small $\beta_t$, until the chunk is pure noise — then learn to undo one step of that corruption so I can start from $x_T\sim\mathcal{N}(0,I)$ and walk the chain backward to a clean sample. The forward corruption is fixed and trivial; all the learning lives in the reverse step. Because each generation begins from a fresh noise draw, different seeds land in different basins: the reverse chain is a stochastic map from noise to data, so it covers multiple modes without my ever naming them. That multimodality falls out of the construction for free.

To avoid unrolling the chain during training, write $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_s\alpha_s$. Because independent Gaussians add in variance, the chain telescopes to a one-shot marginal
$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad\varepsilon\sim\mathcal{N}(0,I),$$
so I can jump straight to any noise level in one step, with $\bar\alpha_t$ running from $\approx 1$ (barely noised) down to $\approx 0$ (pure noise). For the reverse step I model $p_\theta(x_{t-1}\mid x_t)$ as Gaussian — justified because for small $\beta_t$ the true reverse conditional is approximately Gaussian — so I need its mean. The clean route is the posterior $q(x_{t-1}\mid x_t,x_0)$, whose mean is a known combination of $x_t$ and $x_0$, but $x_0$ is exactly what I lack at sampling time. The reparameterization that unlocks it: the marginal gives $x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon)/\sqrt{\bar\alpha_t}$, so the only unknown is the noise $\varepsilon$. I therefore predict $\varepsilon$ with a network $\varepsilon_\theta(x_t,t)$. Substituting back, each timestep's term of the variational bound collapses to a weighted $\lVert\varepsilon-\varepsilon_\theta\rVert^2$, and dropping the per-timestep weight (the practical choice that trains best) leaves one clean loss,
$$L=\mathbb{E}\,\bigl\lVert\varepsilon-\varepsilon_\theta\bigl(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\;t\bigr)\bigr\rVert^2.$$
Sample a clean chunk, a noise tensor, a timestep; form the noised chunk in one shot; ask the network for the noise; take the MSE.

For sampling I do not run the literal ancestral chain, which costs all $T$ forward passes through a 7B trunk. The objective only ever constrained the marginals, never the Markov forward chain, so a family of non-Markovian forward processes shares those marginals and the same trained $\varepsilon_\theta$. For that family the generative step factors into "estimate the clean chunk, then re-noise to the next level": $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$, then $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma^2}\,\varepsilon_\theta+\sigma z$. With $\sigma=0$ the step is deterministic and runnable on a sparse subsequence of timesteps — the DDIM sampler, with the protocol fixing the step count. The noise schedule needs care: a linear $\beta_t$ drives $\bar\alpha_t$ to nearly zero well before $t=T$, wasting the last fraction of the chain on what is already pure noise. I instead use a squared-cosine schedule that defines $\bar\alpha_t$ directly on the cumulative product, $\bar\alpha_t=f(t)/f(0)$ with $f(t)=\cos^2\!\bigl(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\bigr)$, keeping $\bar\alpha$ near 1 at the start and descending smoothly toward 0 so every reverse step does useful work — which matters more for low-range action data than for images.

The part specific to *this* scaffold is the conditioning. I have no dedicated observation path; my observation is already fused inside the 7B trunk, and the only handle is the trunk's action-slot hidden states — the trunk *is* the conditioning network. So I feed the current noisy chunk and the diffusion timestep into the trunk and read the noise prediction off the returned hidden states. The trunk lays out each action timestep as $\text{ACTION\_DIM}=7$ input slots, one per dimension, so a $K$-timestep chunk is $K\cdot 7$ scalar slots and my noisy chunk is exactly $K\cdot 7$ reals. I treat each noisy *scalar* as a one-dimensional token and project it into the trunk's width with a tiny `NoisyActionProjector` (Linear$(1\to d)\to$GELU$\to$Linear$(d\to d)$) broadcast over all $K\cdot 7$ scalars — per-scalar rather than per-action, so the noisy chunk aligns slot-for-slot with how the trunk already attends to actions. The integer timestep is sinusoidally encoded into the embedding space and handed in as one extra token through `diffusion_timestep_embeddings`. One training forward then samples $t$ and $\varepsilon$, forms $x_t$ in one shot via the scheduler's `add_noise`, projects the $K\cdot 7$ scalars, encodes $t$, and calls `runtime.forward(...)`; I select the action-slot hidden states (by the `current | next` mask at train time, by the contiguous post-prompt span at eval time), rearrange $(B,K\cdot 7,d)\to(B,K,7\cdot d)$ to gather each timestep's seven per-dimension hidden states into one feature, and run a pre-LayerNorm MLP-residual `NoisePredictionModel` that emits 7 noise values per timestep; the loss is the MSE against the $\varepsilon$ I added. At inference I roll out the reverse chain from $\mathcal{N}(0,I)$, denoising one scheduled step at a time.

What I am accepting, and the protocol fact that decides this rung's fate: inference is one trunk forward *per* diffusion step, and the runtime fixes 50 steps, so a chunk costs $\sim 50$ passes through a 7B model versus one for a point head. More decisive than latency is convergence — the objective only ever shows the network a randomly-noised chunk at a random timestep and asks for the noise, a far harder fitting problem than regressing the action directly, one the published recipe trains for 100K–250K steps to make competitive. This protocol fixes the budget at 6000 steps to stay inside 5 H200h. So I expect the denoiser to be *under-trained*: the reverse process will not yet walk noise back to coherent chunks, sampled actions will stay close to noise, success will sit near the floor on every subset, and eval times will be several times larger than any single-pass head because of the 50-step rollout. Diffusion is the weakest rung here not because it is a worse idea — its multimodality is a real advantage the point heads lack — but because it is the idea that most needs training time this protocol does not grant, and that diagnosis is what should push the next rung toward a head that fits inside 6000 steps even at the cost of multimodality.

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
