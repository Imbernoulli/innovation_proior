**Problem.** The optimizer is the only free variable in a frozen GPT-2 Medium pretraining run. The
weakest principled rung is the smallest change to the default AdamW: upgrade its momentum from classical
to accelerated, leaving the adaptive per-coordinate rescaling and the decoupled weight decay untouched.

**Key idea.** NAdam = Adam with a simplified-Nesterov look-ahead folded into the update (Sutskever et al.
2013): advance the momentum step by one so the step uses the freshly updated first moment, with the
look-ahead absorbed into the bias-corrected blend of the advanced moment and the current gradient, while
differentiating only at the current parameters. The √v diagonal preconditioner is unchanged — this
refines the *direction*, not the *geometry* of the update.

**Step-1 edit.** Keep the substrate's default dimension-based parameter grouping (decay on 2D, none on
1D), the substrate's betas (0.9, 0.95), base LR (6e-4), and weight decay (0.1). Swap only the
constructor: `torch.optim.AdamW(...)` → `torch.optim.NAdam(..., decoupled_weight_decay=True)`, the
AdamW-style decoupled decay. No change to `get_lr`, no `CONFIG_OVERRIDES`. (NAdam has no fused path, so
this runs the plain optimizer.)

**What to watch.** A diagonal refinement, not a geometry change, so expect it to land close to vanilla
AdamW — small movement in val_loss. The real lever is the *shape* of the update (per-coordinate diagonal
→ something that respects the weight matrix's structure), which is what the next rung must attack.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 1: AdamW with Nesterov momentum (NAdam)
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
        optimizer = torch.optim.NAdam(optim_groups, lr=learning_rate, betas=betas,
                                      decoupled_weight_decay=True)
        print("using NAdam optimizer")
        return optimizer
```
