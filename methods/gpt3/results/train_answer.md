The problem with the dominant NLP recipe is that the adaptation step never went away. We can pre-train one task-agnostic Transformer architecture, but to make it useful on a new task we still collect thousands to hundreds of thousands of labeled examples and run gradient descent on them. That does not scale to the open-ended space of language tasks, it invites a high-capacity model to overfit narrow task distributions by exploiting spurious correlations, and it is nothing like how humans adapt from a short instruction or a couple of examples. Prompting a fixed language model had already been tried, but the results were weak enough that most people assumed the mechanism itself was a dead end rather than the model being too small.

The real question is whether prompting fails because the idea is wrong or because the model is too weak. The scaling laws for autoregressive Transformer language models give a clear reason to suspect the latter: test cross-entropy loss falls as a smooth power law in model size, data size, and compute, and architectural shape barely matters within a broad range. Since next-token prediction over a diverse corpus is implicitly a meta-learning objective — web text is full of implicit demonstrations like translation pairs, question-answer pairs, definitions, and repeated patterns — a model that becomes good at predicting the next token must also become good at inferring the rule governing a prefix and applying it inside a single forward pass. That makes in-context learning a capability that should scale with loss, not a separate mechanism that needs to be invented. The experiment, then, is to keep the autoregressive recipe unchanged and scale it hard, training a ladder of model sizes to see whether few-shot conditioning crosses from toy into useful.

The method I propose is GPT-3. It is an autoregressive decoder-only Transformer language model scaled to 175 billion parameters, built on the same foundation as GPT-2 with only modest engineering changes needed for depth and efficiency. At inference time it is conditioned on a text prompt that specifies the task: zero-shot with just an instruction, one-shot with a single demonstration, or few-shot with K demonstrations packed into the context window. No weight updates are performed for the new task; the adaptation happens entirely through the model's activations as it reads the context.

The architecture stays as close to GPT-2 as possible so that any improvement can be attributed to scale rather than a new gadget. It uses pre-normalization, with LayerNorm before each attention and MLP sublayer and a final LayerNorm before the output projection, so that gradients can flow cleanly down the residual path through 96 layers. The projections that write back into the residual stream are initialized with standard deviation scaled by 1 divided by the square root of twice the number of layers, which keeps the residual stream variance roughly constant with depth. The vocabulary is reversible byte-level BPE with about 50,000 merges, so the model can represent arbitrary strings without out-of-vocabulary tokens. The feed-forward width is four times the model width, multi-head attention uses a fixed head dimension, and the context window is doubled to 2048 tokens to make room for demonstrations. The largest models also alternate dense attention layers with locally-banded sparse attention layers to reduce the quadratic attention cost, though attention is a small enough fraction of total compute that this is treated as an efficiency improvement rather than a capability change.

I train a ladder of eight models from 125 million to 175 billion parameters, all on the same objective and the same data mix. The optimizer is Adam with beta2 set to 0.95 instead of the usual 0.999 for a shorter second-moment memory at this scale, a global gradient-norm clip of 1.0, and decoupled weight decay of 0.1 applied only to matrix weights. The learning rate is warmed up linearly over the first few hundred million tokens, then cosine-decayed to 10 percent of its peak over the bulk of training, and held flat afterward. Bigger models use larger batch sizes, set by measuring the gradient noise scale, and smaller peak learning rates. The training data is a quality-filtered, fuzzily-deduplicated Common Crawl plus WebText2, Books1 and Books2, and Wikipedia; high-quality sources are up-weighted rather than sampled in proportion to their size, and sequences are packed to 2048 tokens separated by an end-of-text token. Following the compute-optimal scaling prescription, the largest model is trained on comparatively few tokens relative to its size rather than to convergence, with total compute around 6 times N times D flops.

Evaluation is performed by building a prompt from an optional instruction, K demonstration pairs concatenated as context followed by completion, and a final query whose completion is left blank. For multiple-choice tasks, candidates are ranked by their per-token conditional likelihood given the prompt, sometimes divided by their unconditional likelihood under a generic lead-in to remove length and frequency bias. Binary classification is reframed as multiple choice with word labels. Free-form generation uses beam search with width 4 and a length penalty of 0.6. Because training on web data risks contamination of benchmark test sets, overlap between training data and evaluation sets is measured and flagged.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention. Large models alternate this with a
    locally-banded sparse variant for O(n*band) cost; dense form shown here."""
    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.register_buffer("mask",
            torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    """Pre-LN Transformer block; FFN width = 4 * d_model."""
    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_ctx)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(d_model, 4 * d_model),
            c_proj=nn.Linear(4 * d_model, d_model),
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m = self.mlp
        x = x + m.c_proj(F.gelu(m.c_fc(self.ln_2(x))))
        return x

class SequenceModel(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_layer, d_model, n_head):
        super().__init__()
        self.n_ctx = n_ctx
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(n_ctx, d_model)
        self.h = nn.ModuleList([Block(d_model, n_head, n_ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.n_ctx
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.n_ctx else idx[:, -self.n_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                nxt = torch.multinomial(probs, 1)
            else:
                _, nxt = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.endswith("weight") and not any(k in name for k in ("wte", "wpe", "ln")):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)

def build_prompt(tokenizer, instruction, demonstrations, query_context):
    parts = ([instruction] if instruction else [])
    parts += [ctx + " " + completion for ctx, completion in demonstrations]
    parts.append(query_context)
    return tokenizer.encode("\n".join(parts))

@torch.no_grad()
def _completion_logprob(model, prefix_ids, completion_ids):
    if not prefix_ids:
        raise ValueError("completion likelihood needs at least one prefix token")
    if not completion_ids:
        device = next(model.parameters()).device
        return torch.tensor(float("-inf"), device=device)
    device = next(model.parameters()).device
    full = prefix_ids + completion_ids
    logits, _ = model(torch.tensor([full[:-1]], dtype=torch.long, device=device))
    logp = F.log_softmax(logits[0], dim=-1)
    start = len(prefix_ids) - 1
    token_logps = [logp[start + i, tok] for i, tok in enumerate(completion_ids)]
    return torch.stack(token_logps).mean()

@torch.no_grad()
def score_choice(model, tokenizer, prompt_ids, completion, answer_context=None):
    comp = tokenizer.encode(completion)
    cond = _completion_logprob(model, prompt_ids, comp)
    if answer_context is None:
        return cond
    base = tokenizer.encode(answer_context)
    uncond = _completion_logprob(model, base, comp)
    return cond - uncond

@torch.no_grad()
def beam_search(model, idx, max_new_tokens, beam_width=4, length_penalty=0.6, eos_token_id=None):
    assert idx.size(0) == 1
    beams = [(idx, torch.tensor(0.0, device=idx.device), False)]
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        candidates = []
        for tokens, score, done in beams:
            if done:
                candidates.append((tokens, score, done))
                continue
            idx_cond = tokens if tokens.size(1) <= model.n_ctx else tokens[:, -model.n_ctx:]
            logits, _ = model(idx_cond)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            k = min(beam_width, logp.size(-1))
            vals, ids = torch.topk(logp, k, dim=-1)
            for val, tok in zip(vals[0], ids[0]):
                nxt = tok.view(1, 1)
                ended = eos_token_id is not None and int(tok.item()) == eos_token_id
                candidates.append((torch.cat([tokens, nxt], dim=1), score + val, ended))
        def normalized(item):
            tokens, score, _ = item
            gen_len = max(1, tokens.size(1) - prompt_len)
            return (score / (gen_len ** length_penalty)).item()
        beams = sorted(candidates, key=normalized, reverse=True)[:beam_width]
        if all(done for _, _, done in beams):
            break
    return max(beams, key=normalized)[0]

@torch.no_grad()
def few_shot_classify(model, tokenizer, instruction, demonstrations, query_context, choices):
    prompt = build_prompt(tokenizer, instruction, demonstrations, query_context)
    scores = [score_choice(model, tokenizer, prompt, c) for c in choices]
    return choices[int(torch.stack(scores).argmax().item())]

@torch.no_grad()
def few_shot_generate(model, tokenizer, instruction, demonstrations, query_context, max_new_tokens,
                      beam_width=4, length_penalty=0.6, eos_token_id=None):
    prompt = build_prompt(tokenizer, instruction, demonstrations, query_context)
    device = next(model.parameters()).device
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = beam_search(model, idx, max_new_tokens, beam_width, length_penalty, eos_token_id)
    return tokenizer.decode(out[0, len(prompt):].tolist())
```
