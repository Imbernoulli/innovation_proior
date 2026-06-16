Let me start from why LLM decoding is slow, because the fix has to attack the actual bottleneck, not an imagined one. Each autoregressive step does one forward pass and emits one token, and that pass is dominated by streaming the entire model's weights out of HBM into the accelerator. So I pay gigabytes of memory traffic to produce a single token, and the arithmetic units sit nearly idle the whole time. Decoding is memory-bandwidth-bound, not compute-bound. That reframes everything: I have spare FLOPs and spare arithmetic intensity. If I could predict and confirm *several* tokens in the same pass that already pays the weight-load, I'd get those extra tokens almost for free. So the goal is more accepted tokens per forward pass — equivalently, fewer sequential steps.

The existing answer is speculative decoding. A small draft model autoregressively guesses the next several tokens; the big target model then runs *once* over all the guesses in parallel — and crucially that single pass with a longer input is still one weight-load, so verifying k positions costs about the same memory traffic as generating one. Accepted guesses commit; the first rejection truncates. With rejection sampling you even keep the target's exact output distribution. Lovely in principle. But the draft model is the problem. It has to be small enough to be fast yet aligned enough that the target accepts its tokens, and the standard way to get one is to separately *pretrain* a smaller model — hundreds of GPU-hours — and a separately trained draft drifts from the target's distribution, lowering acceptance. Plus now I have to serve two models, which in a distributed setting is a genuine operational headache. So speculative decoding's whole burden is "where do I get and how do I run a second model."

So the real question becomes: can I get the speculative-decoding benefit — many tokens confirmed per pass — *without* a separate draft model? The draft model exists only to propose candidate future tokens. What if the proposals came from the target model itself? The target's last hidden state h_t already contains everything the model knows at position t; the original LM head turns h_t into a distribution over the next token. What if I bolt on a few *extra* heads that read the same h_t and predict the token two, three, four positions ahead? Then a single forward pass that produces h_t gives me, in parallel, predictions for several future positions — no second model, no separate pretraining, no two-model serving. This is the old idea of attaching extra decoding heads to one model (Stern et al. 2018, blockwise parallel decoding), and it's exactly the lever I want here.

So: given the backbone's last hidden state h_t at position t, add K extra heads. Head k predicts the token at position t+k+1 (the original LM head covers t+1). Now what should a head actually be? The simplest thing that could work is a linear map from h_t to the vocabulary. But h_t was trained to predict *next* token, and "the token two ahead" is a different, harder target — a bare linear probe off h_t may not have the capacity to re-aim. So give each head a tiny bit of nonlinearity: one feed-forward layer with a residual connection. The residual matters: it lets the head start as essentially "pass h_t through" and learn only the correction needed to look further ahead, rather than relearning the representation from scratch. So head k is W₂ applied to (SiLU(W₁·h_t) + h_t), softmaxed — W₁ a d×d square map, W₂ a d×V projection to the vocabulary. SiLU because that's the activation the backbone (Llama-style) already uses, so the head lives in the same functional family.

Initialization deserves thought, because a fresh random head would, at step zero, spew garbage that the verifier rejects every time and, if I'm training the backbone too, send a huge destabilizing gradient into it. So initialize W₁ to zero and W₂ identically to the original LM head. With W₁ = 0, SiLU(0)=0, so the head's input collapses to just the residual h_t, and W₂ = lm_head means the head's initial prediction *equals the original model's next-token prediction*. The head starts as a copy of the LM head and learns to shift its target forward. Clean start, no shock to the backbone.

Now, how do I train these heads, and at what cost? The whole pitch is "cheap to add to an existing model." So the basic recipe: freeze the backbone entirely, train only the heads. Head k is trained with cross-entropy against the true token k+1 positions ahead: L_k = −log p_t^{(k)}(y_{t+k+1}). One detail I notice: the further ahead a head predicts, the more uncertain it inherently is, so L_k grows with k. If I sum the head losses naively, the far heads dominate the gradient. So weight them down — λ_k decaying in k, e.g. λ_k = 0.8^k — so the near heads (which matter most for acceptance) get their due. Total loss for the frozen-backbone variant, call it Medusa-1: Σ_{k=1}^K λ_k·(−log p_t^{(k)}(y_{t+k+1})). Because the backbone is frozen and only provides hidden states, I can even run it quantized to save memory — heads for a 7B model train in a few hours on a single GPU. That's the "democratized" version.

If I have more compute, I can do better by training the heads *together with* the backbone — Medusa-2 — since a frozen backbone's hidden states weren't optimized for multi-position prediction. But now I risk wrecking the backbone's own next-token ability. So I add the backbone's LM cross-entropy L_LM = −log p_t^{(0)}(y_{t+1}) back into the objective, with a weight: L = L_LM + λ₀·L_Medusa-1. And I protect the backbone two more ways: a smaller learning rate for the backbone than for the heads (the backbone is already good; the heads need to move fast), and a warmup — train heads-only first (as in Medusa-1), then bring the backbone in, optionally ramping λ₀ up gradually — so the heads' initially-large gradients don't distort the backbone before they've stabilized.

Now one pass gives me predictions for K+1 future positions: the original LM head proposes the first token, and the K extra heads propose the tokens after it. The naive use, following speculative decoding, is to take the single top continuation — the LM head's top token, then head 1's top token, then head 2's top token, etc. — and verify that one chain. But I'm leaving the spare compute on the table. Each head doesn't just have a top guess; it has a *distribution*, and its second- and third-best guesses are often right. Since I'm bandwidth-bound, verifying more candidate tokens in the same pass is nearly free. So take the top-s_k tokens from head k and consider *combinations*: head 1's top s₁ tokens, each followed by head 2's top s₂ tokens, and so on — the Cartesian product under the LM-head first token. With s₁=2 and s₂=3 that's 2×3=6 length-2 suffixes after the first token. More candidates means a higher chance some long prefix is accepted, so longer expected acceptance per step.

But I can't just throw 6 separate sequences at the model — that's a 6× bigger batch, which multiplies the memory traffic and kills the whole point. The candidates share prefixes (all six in my example share the same single root token from the original LM head; the three children of head-1's first token share that token). So pack them into a *tree* and process it in one sequence, using attention to enforce the right history. Lay all the distinct candidate tokens out as nodes; each node attends only to its ancestors along its branch back to the root — not to tokens in sibling branches, which belong to different continuations. That's a custom attention mask: a token sees its predecessors in its own continuation and nothing else. Set the positional indices to match the tree depth, not the flat layout, so each token gets the position it would have in its own continuation. Now one forward pass over the tree verifies all candidates simultaneously, no batch blow-up. The number of tokens in the tree is Σ_{k=1}^K ∏_{i=1}^k s_i — for the 2,3 example that's 2 + 6 = 8 nodes.

(The flat Cartesian tree is the simple version. Different heads and different top-i ranks have different accuracies — head 1's top-1 is far more reliable than head 4's top-3 — so under a fixed node budget I shouldn't spend nodes uniformly. If I estimate a_k^{(i)}, the calibration accuracy that head k's i-th-ranked token is correct, then assuming independence the expected acceptance contributed by a candidate path is the product of its nodes' accuracies. A node's contribution is the product along the path to that node, so greedily grow the tree by repeatedly adding the connectable node with the highest path accuracy until the budget is hit — an unbalanced tree that maximizes expected acceptance. But the Cartesian tree already captures the idea.)

Last piece: which candidate prefix do I actually accept? I could reuse speculative decoding's rejection sampling, which would preserve the target's exact distribution. But it has a known flaw: its acceptance rate falls as sampling temperature rises, so in the very regime where people sample for diversity, it stops helping speed — and can hurt. The reason is structural: even if the head's distribution perfectly matched the target's, rejection sampling draws the two independently, so a proposed token can still be rejected; only under greedy decoding (draft = target, deterministically) is everything accepted. So exact distribution matching is costing me speed for a guarantee I may not need.

Do I actually need to reproduce the original model's distribution exactly? In practice, no — temperature is just a creativity knob, and higher temperature should mean the model is willing to accept *more* varied continuations, not fewer. So instead of "match the distribution," I'll aim for "accept candidates that are *typical* — not too improbable under the original model." Borrow the truncation-sampling idea: use the *original model's* own probability for each candidate token as the gauge, and accept a token when

p_original(x_{n+k} | x_1,…,x_{n+k−1}) > min( ε, δ·exp(−H(p_original(·|x_1,…,x_{n+k−1}))) ).

Two thresholds, both grounded in the truncation-sampling principles: ε is a hard floor — tokens above some absolute probability are meaningful and acceptable; the second term, δ·exp(−H), is entropy-adaptive — when the model's distribution at that position has high entropy H, exp(−H) is small, so the bar drops and many continuations count as reasonable; when the distribution is sharp (low entropy), exp(−H) is near 1 and the bar is high, so only confident tokens pass. Take the min so the *easier* of the two conditions governs. Accept the longest prefix of a candidate that satisfies this at every position; to guarantee progress, greedily decode and unconditionally accept the first token each step, then apply typical acceptance to the rest. Across all candidate branches, commit the longest accepted prefix.

Check the limits. At temperature 0 the distribution is a spike — only the argmax has nonzero probability — so this reduces to greedy decoding: exactly one token clears the bar, correct. As temperature rises above 0, with suitable ε, δ the greedy token always clears (it has the max probability), so the greedy result is always accepted, giving at least the greedy speedup; and higher temperature flattens the distribution, lowering the entropy-adaptive bar, so *more* tokens are accepted and acceptance length grows with temperature — the opposite of rejection sampling's degradation. That's the behavior I wanted.

(One more wrinkle for Medusa-2 when the target's own training data isn't available, e.g. after RLHF the output distribution no longer matches any public dataset. Self-distillation fixes it: prompt the model itself to generate replies on a seed dataset like ShareGPT, train on that. But training the *backbone* on hard self-generated labels degrades it, so distill instead — match the original model's distribution with a KL loss, L_LM-distill = KL(p_original,t^{(0)} ‖ p_t^{(0)}). To avoid keeping two copies of the model in memory, fine-tune the backbone with a LoRA adapter — then "the original model" is simply the model with the adapter switched off, so the teacher is free.)

Let me write the core: the heads and the parallel decode loop.

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """One head layer: SiLU MLP with a residual; W1 (the linear) init to zero so the
    head starts as identity on h_t."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)        # W1 = 0  -> head starts = backbone
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))       # h_t + SiLU(W1 h_t)

class MedusaHeads(nn.Module):
    """K extra heads on the backbone's last hidden state; head k predicts token t+k+1.
    Final Linear (W2) is initialized from the original lm_head."""
    def __init__(self, hidden_size, vocab_size, num_heads, num_layers=1, lm_head=None):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(*(ResBlock(hidden_size) for _ in range(num_layers)),
                          nn.Linear(hidden_size, vocab_size, bias=False))
            for _ in range(num_heads))
        if lm_head is not None:                   # W2 = lm_head  -> match original
            for h in self.heads:
                h[-1].weight.data.copy_(lm_head.weight.data)
    def forward(self, hidden_states):             # one weight-load gives all heads
        return [head(hidden_states) for head in self.heads]   # K logit tensors

def medusa_loss(base_logits, medusa_logits, targets, lambdas, lambda0=0.0):
    # Medusa-1: sum_k lambda_k * CE(head_k, y_{t+k+1});  lambda_k = 0.8**k
    loss = 0.0
    for k, logits in enumerate(medusa_logits, start=1):
        loss = loss + lambdas[k - 1] * nn.functional.cross_entropy(
            logits[:, :-(k + 1)].reshape(-1, logits.size(-1)),
            targets[:, k + 1:].reshape(-1))
    if lambda0 > 0:                               # Medusa-2: + backbone CE
        lm = nn.functional.cross_entropy(
            base_logits[:, :-1].reshape(-1, base_logits.size(-1)),
            targets[:, 1:].reshape(-1))
        loss = lm + lambda0 * loss
    return loss

def typical_accept(greedy_token, medusa_suffix, base_probs_suffix, epsilon, delta):
    # accept the LM-head greedy token first, then the longest typical Medusa suffix
    H = -(base_probs_suffix * base_probs_suffix.clamp_min(1e-9).log()).sum(-1)
    thresh = torch.minimum(
        torch.full_like(H, epsilon), delta * torch.exp(-H))
    p = base_probs_suffix.gather(-1, medusa_suffix.unsqueeze(-1)).squeeze(-1)
    ok = p > thresh
    accepted_suffix = 0
    for k in range(medusa_suffix.size(0)):
        if ok[k]:
            accepted_suffix += 1
        else:
            break
    return torch.cat([greedy_token.view(1), medusa_suffix[:accepted_suffix]])
```

The causal chain: LLM decoding is memory-bandwidth-bound, so one weight-load should yield many tokens; speculative decoding does that with a draft model but the draft is expensive to train, drift-prone, and a serving burden; so drop the draft and read several future-position predictions straight off the backbone's last hidden state via K extra single-layer residual heads (W₁=0, W₂=lm_head so each head starts as the original head and only learns to look further), trained cheaply on a frozen backbone (Medusa-1, λ_k=0.8^k) or jointly with backbone protection (Medusa-2: + L_LM, smaller backbone LR, head warmup, self-distillation via a toggleable LoRA when data is missing); take each head's top-s_k tokens, assemble candidates as a tree, and verify them all in one pass with a tree attention mask + tree-aware positions (Σ∏s_i tokens, no batch blow-up); and accept the longest prefix that is *typical* under the original model — p > min(ε, δ·exp(−H)) — which degenerates to greedy at temperature 0 and, unlike rejection sampling, accepts *more* as temperature rises.
