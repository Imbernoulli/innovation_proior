# Supervised Fine-Tuning (SFT), Distilled

SFT adapts a pretrained autoregressive language model by continuing next-token training on `(prompt -> expert demonstration)` pairs. The prompt is conditioning context; the loss is computed only on the demonstration tokens.

## Objective

For prompt `q` and expert demonstration `y = (y_1, ..., y_T)`, the autoregressive policy assigns

```text
π_θ(y | q) = Π_t π_θ(y_t | q, y_<t)
```

The SFT loss is the token-averaged negative log-likelihood of the demonstration:

```text
L_SFT(θ) = -(1 / |y|) Σ_{t=1}^{|y|} log π_θ(y_t | q, y_<t)
```

Equivalently: concatenate `[q, y]`, use shifted next-token cross-entropy, mask prompt and padding targets, sum the remaining losses, and divide by the number of valid demonstration tokens.

## Key Details

- **Shifted autoregressive alignment.** Logits at position `j` score token `j+1`, so implementation uses `shift_logits = logits[..., :-1, :]` and `shift_labels = input_ids[..., 1:]`.
- **Prompt masking.** The model should learn `p(response | prompt)`, not `p(prompt)`, so only prediction positions whose next-token target is in the demonstration contribute loss.
- **Token average.** Dividing by valid demonstration-token count avoids making long demonstrations dominate solely by length.
- **Behavior-cloning ceiling.** SFT sees positive demonstrations only. It cannot rank alternative responses, improve beyond the demonstrator, or train recovery from the model's own mistakes.

## Unified-Gradient View

Start from

```text
J(θ) = E_{τ~π_θ}[r(τ | q)] - μ KL(π_β(. | q) || π_θ(. | q)).
```

Since

```text
KL(π_β || π_θ) = E_{τ~π_β}[log π_β(τ | q) - log π_θ(τ | q)],
```

the data-adherence term contributes

```text
∇_θ[-μ KL(π_β || π_θ)] = μ E_{τ~π_β}[∇_θ log π_θ(τ | q)].
```

That is the SFT ascent direction for demonstration log-likelihood, equivalently the negative gradient of `L_SFT`. Written as `∇ log π_θ = (1 / π_θ) ∇π_θ`, it has the unified estimator form with `π_ref = π_θ`, fixed positive advantage `Â = +1` for every demonstration token or sequence, global scale `μ`, and no PPO clipping mask.

## Working Code

This mirrors the FSDP SFT trainer loss: shift logits and labels, keep unreduced cross-entropy, multiply by `loss_mask`, then divide by valid tokens.

```python
import torch.nn as nn


def sft_loss(model, batch):
    """Masked, shifted, token-averaged next-token cross-entropy."""
    loss_mask = batch["loss_mask"][:, :-1].reshape(-1)
    labels = batch["input_ids"][:, 1:]

    output = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
                   position_ids=batch["position_ids"],
                   use_cache=False)

    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels.contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1).to(shift_logits.device)
    loss_mask = loss_mask.to(shift_logits.device)

    per_token_loss = loss_fct(shift_logits, shift_labels)
    per_token_loss = per_token_loss * loss_mask

    valid_tokens = loss_mask.sum()
    return per_token_loss.sum() / valid_tokens
```

With this trainer convention, `loss_mask[:, :-1]` is aligned to prediction positions: a 1 at logit position `j` means the next-token target `x_{j+1}` belongs to the expert demonstration, while prompt and padding targets are 0.

## Practical Defaults

Continue from the pretrained checkpoint with the ordinary causal-LM optimizer and scheduler. Use demonstrations drawn from the deployment prompt distribution, and monitor downstream response quality or task accuracy alongside token validation loss.
