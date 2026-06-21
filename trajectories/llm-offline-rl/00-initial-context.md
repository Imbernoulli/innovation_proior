## Research question

We have a math-specialized SFT model (`Qwen2.5-Math-1.5B-Instruct`) and step-level preference pairs over math solutions (`xinlai/Math-Step-DPO-10K`, used here as response-level chosen/rejected pairs). The design target is the **offline preference loss**: the map from a `(prompt, chosen, rejected)` triple to a policy gradient that makes the model prefer correct solutions over incorrect ones without a reward model and without on-policy sampling. The base model, data, schedule, and judge-free sympy evaluation are fixed. The loss must beat at least one standard variant on the average of GSM8K, MATH-500, and AIME 2024.

The math setting is the hard part. A chosen and a rejected solution often share almost every token and diverge at one wrong step, so a contrastive loss that widens the chosen-minus-rejected margin by pushing the rejected down also drags the near-identical chosen down. Greedy accuracy depends on the **absolute** likelihood of a correct chain, not just the relative margin. A loss can show growing margins while the probability of the correct answer falls. The unresolved tension is between margin and absolute likelihood: a pairwise preference loss can improve the former while hurting the latter.

## Prior art / Background / Baselines

These methods descend from the KL-constrained reward-maximization objective, whose closed-form optimum is an exponential tilt of the reference policy.

- **Three-stage RLHF with PPO (Ziegler et al., 2019; Ouyang et al., 2022).** Fit a reward model by Bradley-Terry MLE and optimize the policy with PPO under a KL penalty. Gap: three coupled models, online sampling, a critic, and a sensitive KL coefficient — heavy and unstable.
- **Bradley-Terry preference model (Bradley & Terry, 1952).** Model preference probability as the logistic of a reward difference. Gap: only reward differences are identified, so the model gives no direct offline policy update.
- **Reward-weighted / advantage-weighted regression (Peters & Schaal, 2007; Peng et al., 2019).** Approximate the exponential-tilt optimum by weighting sampled actions with `exp(r/β)`. Gap: still needs reward values and on-policy samples, and the partition normalizer is only approximated.
- **DPO / `sigmoid` (Rafailov et al., 2023).** Reparameterize the Bradley-Terry reward as `β log(π_θ/π_ref)` and optimize the sigmoid preference loss directly. Gap: the loss only widens the chosen-minus-rejected margin on summed log-probs; when the sequences share most tokens, pushing the rejected down can push the chosen's absolute likelihood down.
- **IPO (Azar et al., 2023).** Replace the sigmoid with a squared loss on length-normalized log-prob margins. Gap: the fixed target margin does not adapt to pair difficulty and can over-regularize, sharing DPO's vulnerability to collapsing the likelihood of sequences that share tokens with rejected completions.
- **ORPO (Hong et al., 2024).** Combine supervised and preference objectives into a single odds-ratio loss without a reference model. Gap: the odds ratio still suppresses rejected sequences, and without a reference the objective lacks a KL anchor against over-optimization.
- **SimPO (Meng et al., 2024).** Drop the reference model and enforce a fixed reward margin on length-normalized log-probs. Gap: the margin ignores problem difficulty and can shrink absolute likelihoods while satisfying the pairwise constraint.

## Fixed substrate / Code framework

The full-parameter DPO-stage training loop (LLaMA-Factory) is frozen. Base model: `Qwen2.5-Math-1.5B-Instruct` with the `qwen` chat template. Training: full-parameter on 4×GPU, ZeRO-2, `lr=5e-7`, cosine schedule with 10% warmup, 4 epochs, `cutoff_len=2048`, `pref_beta=0.1` (variant-specific). The loop tokenizes each `(prompt, chosen, rejected)` triple, runs the policy forward once over the concatenated chosen+rejected batch, and computes per-response summed label log-probabilities `log π_θ(y|x)` via `get_batch_logps` (prompt/pad masking is handled). Reference-based losses also run a frozen reference model under `no_grad` to get `log π_ref(y|x)`.

Reference loading is decided once in `finetuning_args.py`:

```python
self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
```

So the reference is present for `sigmoid`, `ipo`, and `custom`, and absent for `orpo`, `simpo`. The loop decides in `concatenated_forward` whether the per-response log-prob is summed or length-averaged:

```python
if self.loss_type in ["ipo", "orpo", "simpo"]:
    all_logps = all_logps / valid_length
```

`ipo`/`orpo`/`simpo` see average per-token log-prob; `sigmoid`/`custom` see summed log-prob. `get_batch_loss_metrics` adds an optional SFT term (`ftx_gamma * (-chosen_logps_avg)`) and logs reward accuracies/margins. The selected `pref_loss` sets `self.loss_type`, `pref_beta` sets `self.beta`, and `simpo_gamma` sets `self.simpo_gamma`.

## Editable interface

Exactly one region is editable — the body of `compute_preference_loss` in `trainer.py` (the helper methods `dpo_loss`, `odds_ratio_loss`, `simpo_loss` it dispatches to already exist), plus the one line in `finetuning_args.py` that decides `use_ref_model`. Every method fills the same contract: given the four sequence log-probs — `policy_chosen_logps`, `policy_rejected_logps`, `reference_chosen_logps`, `reference_rejected_logps` (the last two `None` when reference-free) — return `(losses, chosen_rewards, rejected_rewards)`, a per-example loss vector and the two implicit-reward tensors used only for logging. The default fill dispatches each named variant to its helper:

```python
def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""Compute loss for preference learning."""
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    return losses, chosen_rewards, rejected_rewards
```

Each method is selected by a `pref_loss` flag (`sigmoid`, `simpo`, `ipo`, `orpo`, and finally `custom`), which sets `self.loss_type` and routes through this method. Reference-free methods land in the top branch; reference-based ones land in `self.dpo_loss`. A `custom` loss adds an `elif self.loss_type == "custom"` branch on the appropriate side and, if it needs the reference model, leaves the `finetuning_args.py` line so `"custom"` is not excluded.

## Evaluation settings

Three judge-free math benchmarks, all graded by MathRuler's sympy + mathd checker (no LLM judge), with greedy decoding (temperature 0) from a single vLLM engine loaded once per pass, seed {42}:

- **GSM8K** (1.32K grade-school problems, `gsm8k_accuracy`)
- **MATH-500** (500-problem MATH subset, `math500_accuracy`)
- **AIME 2024** (30 competition problems, `aime2024_accuracy`)

Higher is better on all three; the target metric is the average. AIME moves in ~3.3-point quanta and is high-variance; GSM8K is near-saturated for this base model and barely moves; MATH-500 is the middle ground where differences are most legible.
