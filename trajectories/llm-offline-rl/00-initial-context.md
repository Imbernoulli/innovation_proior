## Research question

I have a math-specialized SFT model (`Qwen2.5-Math-1.5B-Instruct`) and a pile of step-level
preference pairs over math solutions (`xinlai/Math-Step-DPO-10K`, used here as response-level
chosen/rejected pairs). The single thing being designed is the **offline preference loss** — the map
from a `(prompt, chosen, rejected)` triple to a scalar gradient on the policy — that teaches the model
to prefer correct solutions over incorrect ones *without* a reward model and *without* an on-policy
sampling loop. Everything else (the base model, the data, the training schedule, the judge-free sympy
evaluation) is fixed. The loss must beat at least one standard variant on the average of three math
benchmarks (GSM8K, MATH-500, AIME 2024).

The math-reasoning setting is what makes this sharp. A chosen and a rejected solution to the same
problem often share almost every token — they branch at one wrong step — so any contrastive objective
that widens the chosen-minus-rejected gap by *pushing the rejected down* drags the near-identical
chosen down with it. The quantity that actually decides the benchmark, greedy accuracy, depends on the
**absolute** likelihood the model assigns to a correct chain, not just the relative margin. A loss can
look like it is "winning" (margin growing, reward accuracy high) while the probability of the correct
answer quietly falls. That tension — relative margin vs. absolute correct-answer likelihood — is the
axis the whole ladder is fighting over.

## Prior art before the first rung (preference-optimization lineage)

The ladder reacts to one settled idea: the KL-constrained reward-maximization objective
`max_π E_{y∼π}[r(x,y)] − β·KL(π‖π_ref)`, whose closed-form optimum is the exponential tilt
`π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)`. The lineage below is the set of ways to *reach* that optimum
offline, each with the gap that the next rung exploits.

- **Three-stage RLHF with PPO (Ziegler 2019; Stiennon 2020; Ouyang 2022).** Fit a reward model by
  Bradley-Terry MLE, then push the policy toward high reward with on-policy PPO under a KL leash. Gap:
  three coupled models, sampling from the multi-billion-parameter policy in the training loop, a
  value/critic to estimate, and a KL coefficient to babysit — heavy and unstable. The offline ladder
  exists to avoid this stage entirely.
- **Bradley-Terry preference model (Bradley & Terry 1952).** `p*(y_w ≻ y_l|x) = σ(r(x,y_w) − r(x,y_l))`
  — the logistic of the reward *difference*. Everything downstream is a fit of some implicit reward
  through this sigmoid; because only the difference appears, the reward is identified only up to a
  function of x, and the partition function of the tilt cancels in every pair. This cancellation is the
  hinge the first rung swings on.
- **Reward-weighted / advantage-weighted regression (Peters & Schaal 2007; Peng 2019).** Reach the
  exp-tilt optimum by weighting sampled actions by `exp(r/β)`. Gap: still needs reward values and
  sampling, and handles the intractable partition normalizer only approximately.

## The fixed substrate

A full-parameter DPO-stage training loop (LLaMA-Factory) is frozen and must not be touched. It is the
standard paired-preference harness: the base model is `Qwen2.5-Math-1.5B-Instruct` with the `qwen` chat
template; training is full-parameter on 4×GPU with ZeRO-2, `lr=5e-7`, cosine schedule with 10% warmup,
4 epochs, `cutoff_len=2048`, `pref_beta=0.1` (variant-specific). The loop tokenizes each
`(prompt, chosen, rejected)` triple, runs the policy forward once over the concatenated chosen+rejected
batch, and computes per-response **summed** label log-probabilities `log π_θ(y|x)` via
`get_batch_logps` (the autoregressive shift and the prompt/pad masking are handled for me). For the
reference-using loss types it also runs a frozen reference model under `no_grad` to get
`log π_ref(y|x)`. Whether a reference model is loaded at all is decided once, in `finetuning_args.py`:

```
self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
```

so the reference is present for every reference-based loss (`sigmoid`, `ipo`, `custom`) and absent for
the reference-free ones (`orpo`, `simpo`). The loop also decides, in `concatenated_forward`, whether
the per-response log-prob is **summed** or **length-averaged**:

```
if self.loss_type in ["ipo", "orpo", "simpo"]:
    all_logps = all_logps / valid_length      # average per-token log-prob
```

— `ipo`/`orpo`/`simpo` see the *average* log-prob; `sigmoid`/`custom` see the *summed* log-prob. Around
the loss, `get_batch_loss_metrics` adds an optional SFT term (`ftx_gamma·(−chosen_logps_avg)`) and logs
reward accuracies/margins. The selected `pref_loss` is read into `self.loss_type`, `pref_beta` into
`self.beta`, `simpo_gamma` into `self.simpo_gamma`.

## The editable interface

Exactly one region is editable — the body of `compute_preference_loss` in `trainer.py` (the helper
methods `dpo_loss`, `odds_ratio_loss`, `simpo_loss` it dispatches to already exist), plus the one line
in `finetuning_args.py` that decides `use_ref_model`. Every method on the ladder is a fill of this same
contract: given the four sequence log-probs — `policy_chosen_logps`, `policy_rejected_logps`,
`reference_chosen_logps`, `reference_rejected_logps` (the last two `None` when reference-free) — return
`(losses, chosen_rewards, rejected_rewards)`, a per-example loss vector and the two implicit-reward
tensors used only for logging. The default fill dispatches each named variant to its helper:

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

Each method on the ladder is selected by a `pref_loss` flag (`sigmoid`, `simpo`, `ipo`, `orpo`, and
finally `custom`), which sets `self.loss_type` and routes through this method. The reference-free
methods land in the top branch; the reference-based ones land in `self.dpo_loss`. A `custom` loss adds
an `elif self.loss_type == "custom"` branch in the appropriate side and, if it needs the reference
model, leaves the `finetuning_args.py` line so `"custom"` is *not* excluded.

## Evaluation settings

Three judge-free math benchmarks, all graded by MathRuler's sympy + mathd checker (no LLM judge),
greedy decoding (temperature 0) from a single vLLM engine loaded once per pass, one seed {42}:
**GSM8K** (1.32K grade-school problems, `gsm8k_accuracy`), **MATH-500** (500-problem MATH subset,
`math500_accuracy`), and **AIME 2024** (30 competition problems, `aime2024_accuracy`). Higher is better
on all three; the target is the average. AIME, at 30 problems, moves in ~3.3-point quanta and is the
high-variance benchmark; GSM8K is near-saturated for this base model and barely moves; MATH-500 is the
middle ground where differences are most legible.
