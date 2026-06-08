## Research question

GPT-style language-model pretraining at a fixed model and data budget: the entire training pipeline —
architecture, tokenizer, dataset, batch construction, and evaluation — is frozen, and the *only* thing
being designed is the **optimizer** (and, optionally, the learning-rate schedule). The single question
is whether a different update rule reaches a lower validation cross-entropy than AdamW with cosine
annealing, under the same compute. Everything else about the run is fixed; the optimizer is the one
free variable.

## Prior art before the first rung (the SGD → Adam → AdamW lineage)

The first rung reacts to the default optimizer, which is itself the endpoint of a short lineage. These
are the methods that precede the ladder.

- **SGD (template).** θₜ = θₜ₋₁ − α gₜ. General, but a single α must serve coordinates of wildly
  different gradient scale, and progress along low-curvature directions is slow. Gap: no per-coordinate
  adaptivity, no smoothing.
- **Classical momentum (Polyak 1964).** mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. Accelerates low-curvature
  directions and damps oscillation, but commits to the old velocity direction *before* consulting the
  current gradient. Gap: the momentum step ignores fresh information; still no per-coordinate scale.
- **Adam (Kingma & Ba 2014).** mₜ = β₁mₜ₋₁ + (1−β₁)gₜ; vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²; bias-correct;
  θₜ = θₜ₋₁ − α m̂ₜ/(√v̂ₜ + ε). The √vₜ gives every coordinate its own learning rate — the robustness
  everyone relies on. Gap: the momentum it carries is the *classical* kind, and the per-coordinate
  preconditioning is purely *diagonal* — blind to correlations between the entries of a weight matrix.
- **AdamW (Loshchilov & Hutter 2019).** Decouples weight decay from the gradient: shrink the weights
  directly, θ ← (1−αλ)θ, then take the adaptive step, instead of folding λθ into gₜ (where it would be
  divided by √vₜ and distorted). This is the substrate's default optimizer. Gap inherited from Adam:
  classical momentum and a diagonal preconditioner.

## The fixed substrate

A nanoGPT pretraining loop is frozen and must not be touched. **Model**: GPT-2 Medium — 24 layers, 16
heads, d = 1024, ≈355M parameters, weight-tied token embedding and LM head, no biases, no dropout.
**Data**: FineWeb 10B (`sample-10BT`), GPT-2 tokenizer, block size 1024, ≈7.1B training tokens.
**Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP, bfloat16 autocast,
`torch.compile`. The loop fixes the LR application (`get_lr` → cosine decay with linear warmup over 4%
of steps, base LR 6e-4, min_lr = base/10), global gradient clipping at 1.0, and the base
hyperparameters β₁ = 0.9, β₂ = 0.95, weight_decay = 0.1. Per step the loop reads `get_lr(it,...)` and
sets `param_group['lr'] = lr * param_group.get('lr_scale', 1.0)` on every group — so an optimizer can
expose an `lr_scale` per group to run at a different effective rate while still riding the shared
schedule shape.

## The editable interface

Exactly one method and (optionally) one constant are editable in `nanoGPT/custom_pretrain.py`:

1. **`configure_optimizers(self, weight_decay, learning_rate, betas, device_type)`** — builds and
   returns the optimizer. It must return something with `.zero_grad()`, `.step()`, and `.param_groups`.
   The default groups parameters by dimension: weight decay on 2D parameters (`p.dim() >= 2`), none on
   1D (`p.dim() < 2`).
2. **`get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)`** — the schedule; signature must
   be kept. (Default: cosine with linear warmup.)
3. **`CONFIG_OVERRIDES`** — an optional dict allowing a method to override `learning_rate`,
   `weight_decay`, `warmup_iters`, `min_lr`, or `grad_clip`.

Every method on the ladder is a fill of this same `configure_optimizers` contract. The starting point
is the scaffold default — **AdamW (fused)**: split params into decay/no-decay by dimension, build
`torch.optim.AdamW`. Each rung replaces this construction (and, where it helps, sets `CONFIG_OVERRIDES`).

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — default fill (AdamW, fused)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
```

## Evaluation settings

A single seed (42). The primary metric is **validation cross-entropy on FineWeb** (`val_loss`,
**lower is better**) under the gpt-345m training run. Secondary metrics from the same run are perplexity
on WikiText-2 and LAMBADA (lower is better). A separate downstream evaluation (`lm-eval-345m`) reports
zero-shot accuracy on ARC-Easy, HellaSwag, PIQA, and WinoGrande (**higher is better**). Ranking here is
by val_loss first (lower better), corroborated by the downstream accuracies. These are settings only;
the numbers come after each run.
