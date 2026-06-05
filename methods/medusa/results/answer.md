# Medusa

## Problem

LLM decoding is memory-bandwidth-bound: each forward pass streams the whole model from HBM to emit a single token, leaving FLOPs idle. We want to generate/confirm several tokens per pass — raising arithmetic intensity and cutting sequential steps — without quality loss and without the cost and serving complexity of a separate draft model (the burden of speculative decoding).

## Key idea

Drop the draft model; read multiple future-position predictions directly off the backbone's last hidden state. Add **K extra decoding heads** (one small residual layer each) that predict tokens t+2, t+3, … in parallel from h_t. Take each head's top candidates, assemble them into a **tree**, and verify all candidates in a single forward pass via a **tree attention mask** (no batch blow-up). Accept the longest candidate prefix that is **typical** under the original model.

## Final method

- **Medusa heads.** Head k predicts position t+k+1: p_t^{(k)} = softmax(W₂^{(k)}·(SiLU(W₁^{(k)}·h_t) + h_t)), W₁∈ℝ^{d×d}, W₂∈ℝ^{d×V}. Init **W₁=0** and **W₂=lm_head** so each head starts identical to the original LM head, then learns to look ahead.
- **Training.**
  - *Medusa-1 (frozen backbone):* L = Σ_{k=1}^K λ_k·(−log p_t^{(k)}(y_{t+k+1})), with λ_k = 0.8^k (far heads are more uncertain, so down-weighted). Backbone can be quantized; trains in hours on one GPU.
  - *Medusa-2 (joint):* L = L_LM + λ₀·L_{Medusa-1} with L_LM = −log p_t^{(0)}(y_{t+1}); plus smaller backbone learning rate and head warmup to protect the backbone.
  - *Self-distillation (no data):* generate replies from the model itself; train backbone with L_LM-distill = KL(p_original,t^{(0)} ‖ p_t^{(0)}); use a LoRA adapter so "the original model" is the adapter switched off (no second copy in memory).
- **Tree attention.** Take head k's top-s_k tokens; candidates = Cartesian product. Pack distinct tokens into a tree; each token attends only to its ancestors (custom mask); positional indices follow tree depth. One pass verifies all; tree size Σ_{k=1}^K ∏_{i=1}^k s_i. (Optionally build an unbalanced tree by greedily adding the node of highest calibrated accuracy a_k^{(i)}, maximizing expected acceptance Σ ∏ a.)
- **Typical acceptance.** Accept token x_{n+k} when p_original(x_{n+k}|…) > min(ε, δ·exp(−H(p_original(·|…)))), H = entropy. First token greedily and unconditionally accepted; commit the longest accepted prefix across branches. Reduces to greedy at temperature 0; accepts *more* as temperature rises (unlike rejection sampling, whose acceptance falls with temperature).

## Code

```python
import torch, torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)              # W1 = 0 -> head starts = backbone
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))

class MedusaHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads, num_layers=1, lm_head=None):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(*([ResBlock(hidden_size)] * num_layers),
                          nn.Linear(hidden_size, vocab_size, bias=False))
            for _ in range(num_heads))
        if lm_head is not None:
            for h in self.heads:
                h[-1].weight.data.copy_(lm_head.weight.data)   # W2 = lm_head
    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]    # K logit tensors

def medusa_loss(base_logits, medusa_logits, targets, lambdas, lambda0=0.0):
    loss = 0.0
    for k, logits in enumerate(medusa_logits, start=1):        # lambda_k = 0.8**k
        loss = loss + lambdas[k - 1] * nn.functional.cross_entropy(
            logits[:, :-k].reshape(-1, logits.size(-1)), targets[:, k:].reshape(-1))
    if lambda0 > 0:                                            # Medusa-2 adds backbone CE
        lm = nn.functional.cross_entropy(
            base_logits[:, :-1].reshape(-1, base_logits.size(-1)), targets[:, 1:].reshape(-1))
        loss = lm + lambda0 * loss
    return loss

def typical_accept(cand_tokens, base_probs, epsilon, delta):
    H = -(base_probs * base_probs.clamp_min(1e-9).log()).sum(-1)
    thresh = torch.minimum(torch.tensor(epsilon), delta * torch.exp(-H))
    p = base_probs.gather(-1, cand_tokens.unsqueeze(-1)).squeeze(-1)
    ok = p > thresh
    accepted = 0
    for k in range(cand_tokens.size(0)):
        if k == 0 or ok[k]:                                   # first token always accepted
            accepted += 1
        else:
            break
    return cand_tokens[:accepted]
```

This is the Medusa design: residual SiLU heads (ResBlock + Linear-to-vocab, W₁=0/W₂=lm_head), tree-attention verification, and typical acceptance — added to an existing LLM with no draft model and trainable on a single GPU.
