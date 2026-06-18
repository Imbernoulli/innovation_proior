Let me start from what actually goes wrong, because the fast version of this head already works and I need to be honest about why it isn't enough. I have a parallel-decoding trunk: it ingests vision, language, and proprioception, and at the action slots it hands me a block of final-layer hidden states, one bundle per future timestep in the chunk. The cheapest thing I can do — and the thing I'd reach for first — is run each bundle through a small MLP, read off a 7-D continuous action, and train it to match the logged demonstration with a mean absolute error. One trunk pass, a negligible head, full continuous precision, fast convergence. So what's wrong with it? Stare at the objective. Minimizing E|a − a_pred| over the data picks, for each observation, the conditional median of the logged actions; minimizing E(a − a_pred)² would pick the conditional mean. Both are a single number per observation. That's fine if, given what the robot sees, there is essentially one right action. But demonstrations aren't like that. At a fork an expert sometimes goes left and sometimes right; sometimes regrasps now, sometimes later; the logged actions at nearly identical observations form a *multimodal* distribution. A point estimator handed a bimodal target sits at the average of the modes, and the average of "swerve left" and "swerve right" is "drive straight into the obstacle." This isn't a tuning problem I can anneal away — it's structural. The head outputs one vector; the world wants a distribution. So the real task isn't "predict the action," it's "represent p(actions | observation) richly enough to be multimodal," and then sample one action from it.

So I need a head that models a whole conditional distribution over the action chunk, not a point. What are my options for fitting a complicated, possibly multimodal continuous distribution? I could put a mixture density network on top — predict the means, variances, and weights of a Gaussian mixture. But I have to pick the number of components up front, mixtures are notoriously fiddly to train (mode collapse, components dying), and a fixed small mixture is a clumsy way to express the geometry of valid action chunks across a whole task suite. I could go energy-based: learn an energy E(o, a) and define p(a|o) ∝ exp(−E), which is genuinely expressive and handles multimodality, and people have done exactly this for imitation. But sampling from it means running an inner optimization or MCMC at inference, the training needs negative sampling and is finicky, and I'd be bolting a second inference loop onto a 7B trunk. I want something that trains with a plain regression-flavored loss — no adversary, no partition function, no negative samples — but still produces samples from a multimodal distribution.

There's a construction that fits that description exactly, and it's worth rebuilding it from the ground up rather than invoking it. Take a clean action chunk x₀ and deliberately wreck it with Gaussian noise, a little at a time, over a long chain: q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ) xₜ₋₁, βₜ I), for a schedule of small βₜ. After enough steps the data is indistinguishable from pure noise. Now — and this is the move — if I can learn to *undo one step* of this corruption, to go from xₜ back toward xₜ₋₁, then I can start from pure noise x_T ~ N(0,I) and walk the chain backwards all the way to a clean sample. The forward corruption is fixed and trivial; all the learning is in the reverse step. And because I start each generation from a fresh draw of noise, different seeds land in different basins — the reverse chain is a stochastic map from noise to data, so it can cover multiple modes without me ever naming them. That's the multimodality I needed, and it falls out of the construction for free.

Let me make the forward process usable, because I do not want to actually iterate the corruption chain during training. Write αₜ = 1 − βₜ and ᾱₜ = ∏_{s=1}^t αₛ. Each step scales by √αₜ and injects variance βₜ. Composing two steps: xₜ = √αₜ xₜ₋₁ + √(1−αₜ) εₜ₋₁ and xₜ₋₁ = √αₜ₋₁ xₜ₋₂ + √(1−αₜ₋₁) εₜ₋₂. Substituting, the deterministic factor accumulates as √(αₜ αₜ₋₁), and the two independent Gaussian noise terms with variances (1 − αₜ) and αₜ(1 − αₜ₋₁) sum — independent Gaussians add in variance — to total variance αₜ − αₜαₜ₋₁ = 1 − αₜαₜ₋₁. Induct and the whole chain telescopes:

  xₜ = √(ᾱₜ) x₀ + √(1 − ᾱₜ) ε,    ε ~ N(0, I).

Beautiful — I can jump straight to any noise level t in one shot, no chain to unroll. ᾱₜ runs from ≈1 (barely noised) down to ≈0 (pure noise) as t goes from 1 to T.

Now, what should the reverse step predict? The reverse transition p_θ(xₜ₋₁ | xₜ) I'll model as Gaussian — justified because for small βₜ the true reverse conditional is approximately Gaussian — so I need its mean. The clean way in is the *posterior* q(xₜ₋₁ | xₜ, x₀), which is tractably Gaussian because everything is linear-Gaussian. Its mean is a known combination of xₜ and x₀. So the most literal parameterization is to have the network predict that posterior mean. But x₀ is exactly what I don't have at sampling time. Here's the reparameterization that unlocks it: the forward marginal says xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε, so x₀ = (xₜ − √(1−ᾱₜ) ε)/√ᾱₜ. The only unknown in x₀, given xₜ, is the noise ε that was added. So instead of predicting x₀ or the posterior mean, predict **ε** — the noise — with a network ε_θ(xₜ, t). Substituting x₀(xₜ, ε) into the posterior mean and simplifying gives the reverse mean as

  μ_θ(xₜ, t) = (1/√αₜ)( xₜ − ((1−αₜ)/√(1−ᾱₜ)) ε_θ(xₜ, t) ),

i.e. "take the noisy action, subtract off a scaled version of the predicted noise, rescale." Why is ε the right target and not x₀ or the posterior mean directly? Because when I plug the ε-parameterization back into the variational bound, each timestep's term collapses to a weighted ‖ε − ε_θ‖² — a denoising objective — and dropping the per-timestep weight (which is the practical choice that trains best) leaves a single clean loss:

  L = E_{x₀, ε, t} ‖ ε − ε_θ( √ᾱₜ x₀ + √(1−ᾱₜ) ε, t ) ‖².

That's it: sample a clean action chunk, sample a noise tensor, sample a timestep, form the noised chunk in one shot, ask the network for the noise, take the MSE. No adversary, no partition function, no negative samples — a regression loss, which is exactly the simplicity I wanted, but it fits a full distribution. Predicting x₀ instead was tried by others and gave worse samples early; predicting the posterior mean is what ε-prediction is equivalent to but in worse-conditioned coordinates. ε-prediction it is.

Now sampling. The literal reverse chain (the ancestral sampler) does, for t = T down to 1,

  xₜ₋₁ = (1/√αₜ)( xₜ − ((1−αₜ)/√(1−ᾱₜ)) ε_θ(xₜ, t) ) + σₜ z,   z ~ N(0,I),

with a small injected σₜ z except at the last step. That works but it forces me to take all T reverse steps, because it's the exact reversal of a Markov chain of length T. For a robot controller that's a problem: T forward passes through a 7B trunk per action chunk is a lot of latency. So I want to keep the *same trained ε_θ* but sample in fewer steps. The handle is that the training objective above only ever constrains the marginals √ᾱₜ x₀ + √(1−ᾱₜ) ε; it never actually demanded the *Markov* forward chain. A whole family of forward processes — non-Markovian ones — share those same marginals and hence the same trained network. For that family the generative step factors into two interpretable pieces: first predict the clean chunk from the current noisy chunk,

  x̂₀ = ( xₜ − √(1−ᾱₜ) ε_θ(xₜ, t) ) / √ᾱₜ,

then re-noise it to the next (lower) level along a chosen subsequence of timesteps,

  xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1 − ᾱₜ₋₁ − σ²) ε_θ(xₜ, t) + σ z.

Set σ = 0 and the step is **deterministic** — "estimate x₀, then jump to the next noise level" — and I can run it on a sparse subsequence of timesteps, so far fewer reverse passes than the forward chain length. I'll keep this sampler available so inference latency is tunable; for the protocol here I'll run the full step count, but the option to skip steps is exactly why I prefer this generative form over the ancestral one.

The schedule βₜ needs care, and the obvious linear ramp is wrong for my data. The action chunk is low-dimensional and lives in [−1,+1]; a linear βₜ pushes ᾱₜ to nearly zero well before t = T, so the last big fraction of the chain is operating on what is already pure noise — wasted steps where there's no signal left to denoise, and abrupt noise-level jumps in the middle. I want ᾱₜ to change *slowly* near both ends and roughly linearly in the middle, so every reverse step is doing useful work. The construction that does this defines ᾱₜ directly on the cumulative product rather than on βₜ:

  ᾱₜ = f(t)/f(0),   f(t) = cos²( ((t/T + s)/(1 + s)) · (π/2) ),

a squared-cosine with a small offset s. That keeps ᾱ near 1 at the start (don't destroy structure immediately) and descends smoothly to near 0, with a gentle approach so the high-noise end isn't wasted. For small-range continuous action data this matters more than for images, and it's what the action-sequence diffusion work settled on, so I'll use the squared-cosine schedule.

Now the part that's specific to *this* setup and isn't in any of the generic recipes: where does the observation conditioning go, and what is ε_θ actually attached to? In a from-scratch action-diffusion policy, the noise predictor is its own network and the observation is piped in through a dedicated conditioning path — FiLM on a conv backbone, or cross-attention in a small transformer. I don't have that luxury and I don't want it: my observation is *already* fused — vision, language, proprioception — inside the 7B trunk, and the only handle I have on it is the trunk's hidden states at the action slots. The trunk is the conditioning network. So the question becomes: how do I feed the *current noisy action chunk* and the *diffusion timestep* into the trunk so that the hidden states it returns are conditioned on (observation, noisy action, t), and then read the noise prediction off those hidden states?

The noisy chunk first. The trunk was wired so that each action timestep occupies 7 input slots — one per action dimension, because the base model tokenized each 7-D action into 7 separate tokens and I'm keeping that layout so I don't have to touch the trunk's sequence handling. So a chunk of K timesteps is K·7 scalar slots. My noisy action chunk is exactly K·7 real numbers. The clean way to inject it is to treat each noisy *scalar* as a one-dimensional token and project it up into the trunk's embedding dimension: a tiny MLP from R¹ → R^{d_llm}, applied to each of the K·7 scalars. That gives K·7 embeddings I can drop into the action slots. Why per-scalar (input dim 1) and not per-7-D-action (input dim 7)? Because the slot layout is one slot per action dimension; one embedding per scalar keeps the noisy action aligned slot-for-slot with how the trunk already lays out and attends to actions. So I build a `NoisyActionProjector`: Linear(1 → d) → GELU → Linear(d → d), broadcast over all K·7 noisy scalars.

The timestep next. The network must know *how noisy* the input is — ε_θ depends on t — so I encode the integer timestep into the embedding space and hand it to the trunk as one extra token. The standard sinusoidal encoding does this: for timestep value x, emb = [sin(x·ω_k), cos(x·ω_k)] over a geometric grid of frequencies ω_k spanning a wide range, so different timesteps get near-orthogonal, smoothly-varying codes. One timestep token, appended alongside the noisy-action tokens.

So one training forward looks like: sample t, sample ε, form the noised chunk xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε in one shot, project xₜ's K·7 scalars to action-token embeddings, encode t to a timestep embedding, and call the trunk forward with both. The trunk fuses these with the (fixed) vision/language/proprio context under bidirectional attention and returns final hidden states. From those I select the action-slot hidden states. At train time the slots are identified by a mask over the demonstration's action tokens (the current action plus the next actions of the chunk); at eval time, where I'm feeding placeholders, the slots are a contiguous span right after the prompt of length K·7. Either way I reshape to (batch, K·7, d).

Now map those hidden states to a noise prediction. Each action timestep has 7 hidden states, one per action dimension; I want to predict that timestep's 7-D noise from them. So I rearrange (batch, K·7, d) into (batch, K, 7·d) — gathering the 7 per-dimension hidden states for a timestep into one feature vector — and run an MLP that takes 7·d in and emits 7 out, per timestep. The MLP itself: I'll use residual blocks with pre-layer-norm, because this is a fairly deep small net being trained jointly with a giant trunk and I want stable gradients; pre-LN feedforward residual blocks (LayerNorm → Linear → ReLU, added to the input) are the well-behaved default. A couple of such blocks between an input projection and an output projection is plenty — the trunk does the heavy lifting; this head only has to read out the noise. The loss is MSE between this prediction (reshaped back to the chunk shape) and the ε I actually added. Plain mean-squared error, the simplified objective.

At inference I run the reverse process. Start from curr = N(0,I) of shape (1, K, 7). Set the sampler to its timestep subsequence. For each t in that schedule: encode t, project the current noisy chunk to action-token embeddings, forward the trunk, extract the action-slot hidden states, predict the noise, and ask the sampler to take one denoising step curr ← step(noise_pred, t, curr) — which internally forms x̂₀ and jumps to the next level. After the last step, curr is a clean normalized action chunk. Note the cost I'm accepting: this is one trunk forward *per diffusion step*, so latency scales with the number of reverse steps — the price of representing a distribution instead of a point, and exactly why I kept the deterministic, step-skippable sampler.

One more design check before I write it. Why inject the noisy action back through the trunk at all, instead of conditioning a standalone denoiser on a single pooled observation embedding? Because the trunk's bidirectional attention lets every action slot attend to every other action slot *and* to the full multimodal context jointly, at every diffusion step — the denoiser is the 7B model itself, re-reading the noisy chunk against the observation each step, rather than a small head reading a frozen summary. That's strictly more expressive conditioning, and it reuses the parallel-decoding machinery I already built rather than adding a parallel conditioning network. It costs forward passes, which I've already accounted for. The pieces are forced now: a per-scalar noisy-action projector, a sinusoidal timestep encoder, a cosine-schedule noise scheduler with a deterministic few-step sampler, an MLP-residual noise predictor reading the action-slot hidden states, an MSE-on-ε training step, and a reverse-diffusion rollout for prediction. Let me write it into the action-method slot.

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
    NoisePredictionModel,           # MLP-residual net: (B,K,7*d) -> (B,K,7) noise
    SinusoidalPositionalEncoding,   # integer timestep -> (B, d) embedding
)
from prismatic.models.projectors import NoisyActionProjector  # per-scalar R^1 -> R^d


# Diffusion needs more warmup than the point-estimate head; trunk + denoiser co-train.
CONFIG_OVERRIDES = {
    "lr_warmup_steps": 800,
    "num_steps_before_decay": 6000,
}


class CustomActionMethod(nn.Module):
    """Conditional denoising-diffusion action head on the parallel-decoding trunk."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        # Noise predictor: 7 per-dimension hidden states per timestep -> that timestep's 7-D noise.
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim * ACTION_DIM,  # 7 hidden states concatenated
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        # Inject each noisy action SCALAR (dim 1) as one action-token embedding.
        self.noisy_action_projector = NoisyActionProjector(llm_dim=input_dim)
        # Forward marginals + deterministic step-skippable reverse sampler, cosine schedule.
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=FIXED_NUM_DIFFUSION_STEPS_TRAIN,
            beta_schedule="squaredcos_cap_v2",   # squared-cosine: slow at both ends
        )
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def extract_action_hidden_states(self, runtime, forward: ForwardPassResult) -> torch.Tensor:
        # Pull the action-slot hidden states out of the trunk's final layer.
        if runtime.phase == "train":
            mask = runtime.current_action_mask | runtime.next_actions_mask
            return (
                forward.text_hidden_states[mask]
                .reshape(runtime.batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(runtime.hidden_dtype)
            )
        # Eval: placeholders occupy a contiguous K*7 span right after the prompt.
        start = runtime.num_prompt_tokens
        end = start + NUM_ACTIONS_CHUNK * ACTION_DIM
        return forward.text_hidden_states[:, start:end, :].to(runtime.hidden_dtype)

    def build_action_token_features(self, runtime, noisy_actions: torch.Tensor) -> torch.Tensor:
        # (B,K,7) noisy chunk -> (B, K*7, 1) scalars -> (B, K*7, d) action-token embeddings.
        batch_size = noisy_actions.shape[0]
        flattened = noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
        return self.noisy_action_projector(flattened).to(runtime.hidden_dtype)

    def sample_noisy_actions(self, ground_truth_actions):
        # Forward diffusion in one shot: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps.
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
        # (B, K*7, d) -> (B, K, 7*d) -> predict (B, K, 7) noise.
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.noise_predictor(rearranged)

    def _rollout_diffusion(self, runtime):
        # Reverse process: start from pure noise, denoise one trunk-pass per step.
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
            curr = self.noise_scheduler.step(noise_pred, t, curr).prev_sample  # x_t -> x_{t-1}
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
        loss = nn.functional.mse_loss(noise_pred, noisy["noise"], reduction="mean")  # L_simple
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

The causal chain, start to finish: the fast continuous head is a point estimator — L1 gives the conditional median, L2 the conditional mean — and on multimodal demonstrations a point estimate averages incompatible modes into invalid actions, so I needed a head that represents the whole conditional action distribution. Among distribution learners, mixtures need a fixed component count and are fragile, and energy-based models need inner-loop sampling and negative samples; the denoising-diffusion construction trains with a plain MSE-on-noise loss yet samples a multimodal distribution by learning to reverse a fixed Gaussian corruption. The forward chain telescopes to a one-shot marginal xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε, and reparameterizing x₀ in terms of the added noise makes ε the natural prediction target, collapsing the variational bound to L = E‖ε − ε_θ(xₜ, t)‖². A squared-cosine schedule keeps ᾱₜ useful at both ends for low-range action data, and the non-Markovian generative step (predict x̂₀, jump to the next level, σ=0) gives a deterministic sampler that can skip steps, making inference latency tunable. The observation conditioning is the trunk itself: I inject the noisy chunk as per-scalar action-token embeddings and the timestep as one sinusoidal token, let the bidirectional trunk re-read the noisy chunk against the fused vision-language-proprio context every diffusion step, read the action-slot hidden states, and predict the noise with a small pre-LN MLP-residual net. Training is one forward per step with an MSE loss; prediction is a reverse rollout, one trunk pass per diffusion step, from noise to a clean action chunk — paying forward passes to buy a distribution where the point head only had a vector.
