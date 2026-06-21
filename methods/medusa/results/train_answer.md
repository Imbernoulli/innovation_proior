LLM decoding is slow for a structural reason that has nothing to do with the arithmetic the accelerator can do. Each autoregressive step runs one forward pass and emits exactly one token, and that pass is dominated by streaming the entire model's weights out of high-bandwidth memory into the accelerator. So I pay gigabytes of memory traffic to produce a single token while the arithmetic units sit nearly idle: decoding is memory-bandwidth-bound, not compute-bound. That reframes the whole problem. I already have spare FLOPs and spare arithmetic intensity, so if I could predict and confirm several tokens within the same pass that already pays the weight-load, those extra tokens would come almost for free. The goal is therefore to raise the number of accepted tokens per forward pass, equivalently to cut the number of sequential decode steps, without changing the model's output quality and without the operational burden of training and serving a second network.

The standard answer is speculative decoding. A small, fast draft model autoregressively guesses the next several tokens, then the large target model runs once over all the guesses in parallel. Because a single pass over a longer input is still one weight-load, verifying k positions costs about the same memory traffic as generating one; accepted guesses commit, the first rejection truncates, and with rejection sampling the accepted tokens are distributed exactly as the target would have sampled them. The trouble is the draft model. It has to be small enough to be fast yet aligned enough that the target accepts its continuations, and the usual route is to separately pretrain a smaller model, reportedly hundreds of GPU-hours, which then drifts from the target and lowers acceptance. On top of that I now have to serve two models, which in a distributed setting is a genuine operational headache. So speculative decoding's entire burden reduces to where do I get, and how do I run, a second model. Rejection sampling also has a wrinkle: its acceptance rate falls as sampling temperature rises, so in exactly the regime where users sample for diversity it stops helping and can even hurt, because even when the draft and target distributions match perfectly, drawing the two independently can still reject a token.

I propose Medusa, which keeps the speculative-decoding benefit of confirming many tokens per pass while throwing the draft model away entirely. The draft model exists only to propose candidate future tokens, so let the proposals come from the target itself. The backbone's last hidden state h_t already contains everything the model knows at position t, and the original LM head turns h_t into a distribution over the next token. I bolt on K extra decoding heads that read the same h_t and predict tokens further ahead: head k predicts the token at position t+k+1, while the original LM head covers t+1. A single forward pass that produces h_t then yields, in parallel, predictions for K+1 future positions, with no second model, no separate pretraining, and no two-model serving. Each head is deliberately small. A bare linear probe off h_t would not have the capacity to re-aim, because h_t was trained to predict the next token and the token two ahead is a harder target, so I give each head a single feed-forward layer with a residual connection: p_t^{(k)} = softmax(W_2^{(k)} (SiLU(W_1^{(k)} h_t) + h_t)), where W_1 is d-by-d and W_2 is d-by-V.

The residual lets the head begin as essentially "pass h_t through" and learn only the correction needed to look further ahead. SiLU is chosen because it is the activation the Llama-style backbone already uses. Initialization matters too. A fresh random head would emit garbage at step zero and send a destabilizing gradient into the backbone, so I set W_1 to zero and copy W_2 from the original LM head. With W_1 zero, SiLU(0)=0, the head's input collapses to the residual h_t, and with W_2 equal to lm_head the head's initial prediction equals the original model's next-token prediction. Each head therefore starts as an exact copy of the LM head and only learns to shift its target forward, a clean start with no shock to the backbone.

Training is built around the promise that this is cheap to add to an existing model. The basic recipe, Medusa-1, freezes the backbone entirely and trains only the heads with cross-entropy against the true token k+1 positions ahead. Because the further ahead a head predicts the more uncertain it is, I down-weight the far heads with lambda_k = 0.8^k so the near heads, which matter most for acceptance, get their due. Since the frozen backbone only supplies hidden states, it can even run quantized, so the heads for a 7B model train in a few hours on a single GPU. With more compute I train the heads jointly with the backbone in Medusa-2, because a frozen backbone's hidden states were never optimized for multi-position prediction; but now I risk wrecking the backbone's own next-token ability, so I fold the backbone LM cross-entropy back into the objective as L_LM plus lambda_0 times the Medusa-1 loss. I protect the backbone with a smaller learning rate than the heads and a warmup that trains heads-only first before bringing the backbone in. When the target's own training data is unavailable, for instance after RLHF, I use self-distillation: prompt the model itself for replies on a seed dataset and train with a KL loss against the original distribution, fine-tuning the backbone through a LoRA adapter so the teacher is simply the adapter switched off.

One pass now gives predictions for K+1 future positions, and taking only the single top chain leaves spare compute on the table. Each head has a full distribution, and its second- and third-best guesses are often right; since I am bandwidth-bound, verifying more candidate tokens in the same pass is nearly free. So I take the top-s_k tokens from head k and consider combinations, the Cartesian product under the LM-head first token. With s_1=2 and s_2=3 that is six length-2 suffixes after the first token, and more candidates raise the chance that some long prefix is accepted. But I cannot throw six separate sequences at the model, because that multiplies memory traffic and destroys the point. The candidates share prefixes, so I pack them into a tree and process it as one sequence. Every distinct candidate token becomes a node; each node attends only to its ancestors along its branch back to the root, never to tokens in sibling branches. That is a custom attention mask with positional indices set to tree depth rather than flat layout. One forward pass over the tree verifies all candidates simultaneously with no batch blow-up; the tree holds the sum over k of the product from i=1 to k of s_i nodes. I can also build an unbalanced tree by greedily adding the connectable node with the highest calibrated path accuracy under a fixed node budget, maximizing expected acceptance.

The last question is which candidate prefix to accept. Rejection sampling preserves the target's exact distribution, but its acceptance falls as temperature rises, so exact matching costs me speed for a guarantee I may not need. In practice I do not need to reproduce the original distribution exactly: higher temperature should mean the model accepts more varied continuations, not fewer. So instead of matching the distribution I accept candidates that are typical, not too improbable under the original model, using the original model's own probability as the gauge. I accept token x_{n+k} when p_original(x_{n+k} given the prefix) is greater than the minimum of epsilon and delta times exp(-H), where H is the entropy of the original model's distribution at that position. The hard floor epsilon accepts tokens above an absolute probability. The entropy-adaptive term delta exp(-H) drops the bar when the distribution is flat and raises it when the distribution is sharp. Taking the minimum lets the easier condition govern. To guarantee progress I greedily decode and unconditionally accept the first token each step, then apply typical acceptance to the rest, committing the longest accepted prefix across all branches. At temperature 0 this reduces to greedy decoding; as temperature rises, the greedy token always clears and a flatter distribution lowers the bar, so more tokens are accepted and acceptance length grows with temperature, the opposite of rejection sampling's degradation.

That is the canonical Medusa method: residual SiLU heads added to an existing LLM with no draft model, trained cheaply on a frozen backbone or jointly with backbone protection, verified through tree attention, and accepted by a typical-sampling criterion. The following Python snippet shows the core building blocks, a small tree-attention mask generator, and a numeric sanity check that the typical acceptance rule accepts more tokens as entropy grows.

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """One Medusa head layer: SiLU MLP with a residual.
    W1 is initialized to zero so the head starts as identity on h_t."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHeads(nn.Module):
    """K extra heads on the backbone's last hidden state.
    Head k predicts token t+k+1. The final Linear (W2) is copied from lm_head."""
    def __init__(self, hidden_size, vocab_size, num_heads, num_layers=1, lm_head=None):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(*(ResBlock(hidden_size) for _ in range(num_layers)),
                          nn.Linear(hidden_size, vocab_size, bias=False))
            for _ in range(num_heads)
        )
        if lm_head is not None:
            for h in self.heads:
                h[-1].weight.data.copy_(lm_head.weight.data)

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]


def medusa_loss(base_logits, medusa_logits, targets, lambdas, lambda0=0.0):
    """Medusa-1: weighted sum of per-head cross-entropies.
    Medusa-2 adds the backbone LM loss when lambda0 > 0."""
    loss = 0.0
    for k, logits in enumerate(medusa_logits, start=1):
        loss = loss + lambdas[k - 1] * nn.functional.cross_entropy(
            logits[:, :-(k + 1)].reshape(-1, logits.size(-1)),
            targets[:, k + 1:].reshape(-1))
    if lambda0 > 0:
        lm = nn.functional.cross_entropy(
            base_logits[:, :-1].reshape(-1, base_logits.size(-1)),
            targets[:, 1:].reshape(-1))
        loss = lm + lambda0 * loss
    return loss


def typical_accept(greedy_token, medusa_suffix, base_probs_suffix, epsilon, delta):
    """Accept the greedy token, then the longest typical Medusa suffix."""
    H = -(base_probs_suffix * base_probs_suffix.clamp_min(1e-9).log()).sum(-1)
    thresh = torch.minimum(torch.full_like(H, epsilon), delta * torch.exp(-H))
    p = base_probs_suffix.gather(-1, medusa_suffix.unsqueeze(-1)).squeeze(-1)
    ok = p > thresh
    accepted_suffix = 0
    for k in range(medusa_suffix.size(0)):
        if ok[k]:
            accepted_suffix += 1
        else:
            break
    return torch.cat([greedy_token.view(1), medusa_suffix[:accepted_suffix]])


def tree_attention_mask(parents):
    """Build a causal tree mask: node j attends to node i only if i is an ancestor of j.
    parents[k] is the index of the parent of node k; -1 means root."""
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for j in range(n):
        cur = j
        while cur != -1:
            mask[j, cur] = True
            cur = parents[cur]
    return mask


def demo():
    torch.manual_seed(0)
    hidden_size, vocab_size, num_heads = 64, 1000, 3
    fake_hidden = torch.randn(2, 12, hidden_size)
    fake_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    medusa = MedusaHeads(hidden_size, vocab_size, num_heads, lm_head=fake_lm_head)
    logits = medusa(fake_hidden)
    print("Head logit shapes:", [t.shape for t in logits])

    # Simulate a tree with root plus two branches: root -> (a, b), a -> (c, d).
    parents = [-1, 0, 0, 1, 1]
    mask = tree_attention_mask(parents)
    print("Tree mask:\n", mask.to(torch.int32))

    # Typical acceptance: when the distribution is flat, even non-greedy
    # suffix tokens are probable enough to pass the entropy-adaptive bar.
    for scale in [10.0, 1.0, 0.1]:
        logits = torch.randn(4, vocab_size) * scale
        probs = torch.softmax(logits, dim=-1)
        greedy = probs[0].argmax()
        suffix = torch.cat([torch.tensor([greedy]),
                            torch.randint(0, vocab_size, (3,))])
        accepted = typical_accept(greedy, suffix[1:], probs[1:], epsilon=0.05, delta=0.5)
        H = -(probs[1] * probs[1].clamp_min(1e-9).log()).sum()
        print(f"scale={scale:.1f} entropy~{H:.2f}: accepted {accepted.numel()} token(s)")


if __name__ == "__main__":
    demo()
```
