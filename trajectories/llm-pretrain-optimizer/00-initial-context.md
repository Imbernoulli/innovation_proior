## Research question

GPT-style language-model pretraining at a fixed model and data budget. The entire training pipeline — architecture, tokenizer, dataset, batch construction, and evaluation — is frozen; the only design variable is the **optimizer** (and optionally the learning-rate schedule). The question is whether a different update rule reaches lower validation cross-entropy than AdamW with cosine annealing under the same compute.

## Prior art / Background / Baselines

- **SGD.** Updates parameters by subtracting a single learning-rate-scaled gradient. Limitation: one step size must serve coordinates with very different gradient magnitudes, slowing progress in low-curvature directions.
- **Classical momentum (Polyak 1964).** Maintains a running velocity and steps in that averaged direction. Limitation: the velocity is updated before the current gradient is incorporated, so the step can overshoot or lag sudden gradient changes; it also keeps a single global step size.
- **Adam (Kingma & Ba 2014).** Keeps per-coordinate moving averages of gradient and squared gradient and normalizes each update by estimated standard deviations. Limitation: the normalized update can be noisy early in training and can shrink updates in directions where squared-gradient estimates are large.
- **AdamW (Loshchilov & Hutter 2019).** Applies weight decay directly to parameters before taking an Adam-style adaptive step. Limitation: it inherits Adam's sensitivity to noisy second-moment estimates and its per-coordinate normalization behavior.

## Fixed substrate / Code framework

The substrate is a frozen nanoGPT pretraining run. **Model:** GPT-2 Medium (24 layers, 16 heads, d=1024, ≈355M params), weight-tied embeddings, no biases, no dropout. **Data:** FineWeb 10B (`sample-10BT`), GPT-2 tokenizer, block size 1024, ≈7.1B training tokens. **Training:** 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP, bfloat16 autocast, `torch.compile`. The LR schedule is cosine with linear warmup over 4% of steps, base LR 6e-4, min_lr = base/10. Global gradient clipping is fixed at 1.0, and base hyperparameters are β₁=0.9, β₂=0.95, weight_decay=0.1. Each step sets `param_group['lr'] = lr * param_group.get('lr_scale', 1.0)`, so an optimizer can optionally expose an `lr_scale` per group.

## Editable interface

The editable region is in `nanoGPT/custom_pretrain.py`:

1. **`configure_optimizers(self, weight_decay, learning_rate, betas, device_type)`** — builds and returns the optimizer. It must provide `.zero_grad()`, `.step()`, and `.param_groups`. The default groups parameters by dimension: weight decay on 2D parameters (`p.dim() >= 2`), none on 1D (`p.dim() < 2`).
2. **`get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)`** — the schedule; signature must be kept. (Default: cosine with linear warmup.)
3. **`CONFIG_OVERRIDES`** — an optional dict allowing a method to override `learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, or `grad_clip`.

The default fill is **AdamW (fused)** below. An optimizer implementation replaces the body of `configure_optimizers` and may set `CONFIG_OVERRIDES`.

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

All runs use seed 42. Primary metric: validation cross-entropy on FineWeb (`val_loss`, lower better) from the `gpt-345m` training run. Secondary metrics: WikiText-2 and LAMBADA perplexity (lower better). A separate `lm-eval-345m` run reports zero-shot accuracy on ARC-Easy, HellaSwag, PIQA, and WinoGrande (higher better). Ranking is by `val_loss`, corroborated by downstream accuracies.
