# GPT-3

## Problem

The pre-train/fine-tune paradigm gives one task-agnostic architecture but still needs a per-task labeled dataset (thousands to hundreds of thousands of examples) and a gradient-descent run for every new task. That doesn't scale to the open-ended space of language tasks, it overfits narrow task distributions (exploiting spurious correlations), and it's unlike how humans adapt — from an instruction or a couple of examples. The goal: a single fixed model that performs new tasks at inference time from a text-only specification, with **no gradient updates**.

## Key idea

Don't invent a new adaptation mechanism — **scale the autoregressive language-modeling recipe** and reinterpret task adaptation as conditioning. Next-token prediction over a diverse web corpus is implicitly a meta-learning objective: the corpus is full of implicit task demonstrations (translation pairs, Q→A, "word: definition", worked arithmetic, repeated sub-patterns), so minimizing loss forces the model to infer "what rule governs this prefix" and apply it — *in the forward pass, with no weight update*. This is an inner-loop (one forward pass over the context) / outer-loop (SGD over the corpus) meta-learning structure. Because LM loss scales as a smooth power law in model size, data, and compute, and capability tracks loss, in-context learning should be one of the capabilities that improves with scale; the model ladder tests whether examples in the context become more useful as size grows.

So the recipe is: train a very large autoregressive Transformer (175B parameters, the same architecture as GPT-2 with one efficiency tweak), and at test time provide the task as **zero-shot** (instruction only), **one-shot** (one demonstration), or **few-shot** (K demonstrations, K bounded by the context window) prompts — no fine-tuning.

## The model and recipe

- **Architecture**: decoder-only Transformer LM, identical to GPT-2 — pre-normalization (LN before each sublayer + a final LN), residual-path projections initialized with std = 0.02/√(2·n_layer) to keep the residual stream variance ~constant across depth, reversible byte-level BPE (~50k vocab), FFN width = 4·d_model. Context window **n_ctx = 2048** (doubled vs GPT-2, to fit demonstrations). One change vs GPT-2: **alternating dense and locally-banded sparse attention** layers (Sparse-Transformer style) to cut the O(n²) attention cost.
- **Eight sizes**, 125M → 175B params (12→96 layers, d_model 768→12288), to trace the scaling curve. All trained for 300B tokens.
- **Optimizer**: Adam (β₁=0.9, **β₂=0.95**, ε=1e-8), global grad-norm clip 1.0, decoupled weight decay 0.1 (matrix weights only). Linear LR warmup (~375M tokens) then cosine decay to 10% over 260B tokens. Batch size ramped up over the first few B tokens; **bigger model → larger batch (set via the gradient noise scale), smaller peak LR**.
- **Data**: filtered + fuzzily-deduplicated Common Crawl (kept iff `np.random.pareto(9) > 1 - doc_score`) plus WebText2, Books1/2, Wikipedia; high-quality sources up-weighted (not sampled proportional to size); sequences packed to 2048 with an end-of-text delimiter.
- **Compute-optimal under-training**: per the power laws, spend a fixed budget on a much larger model trained on comparatively few tokens (L(C_min)≈(3.1×10⁸ PF-days/C_min)^0.050 and N_opt ∝ C^0.73), stopping before convergence. Training compute ≈ 6·N·D flops (2 for the forward pass, times 3 for forward+backward) ≈ 3.14×10²³ for 174.6B parameters over 300B tokens.
- **In-context evaluation**: build a prompt = optional instruction + K (context, completion) demos + query; for multiple choice, rank candidates by per-token-normalized likelihood (optionally divided by the unconditional likelihood given a generic "Answer:" lead-in); binary tasks become multiple choice with word-named labels; free-form uses beam search (width 4, length penalty 0.6).

## Working code

The code follows the minGPT/nanoGPT module structure in dense-attention form. The few-shot interface is a harness around the fixed model, not a model change.

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
        self.c_proj = nn.Linear(d_model, d_model)          # residual-stream write-back
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
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))   # scale by 1/sqrt(d_head)
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
            c_proj=nn.Linear(4 * d_model, d_model),        # residual-stream write-back
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
        self.wte = nn.Embedding(vocab_size, d_model)       # byte-level BPE tokens
        self.wpe = nn.Embedding(n_ctx, d_model)            # learned positions
        self.h = nn.ModuleList([Block(d_model, n_head, n_ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight              # tied token/output embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():            # scaled residual init
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
    # decay matrix weights only; not biases / LayerNorm / embeddings
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.endswith("weight") and not any(k in name for k in ("wte", "wpe", "ln")):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)

# ---- in-context learning harness: fixed-model prompt + scorer ----

def build_prompt(tokenizer, instruction, demonstrations, query_context):
    parts = ([instruction] if instruction else [])
    parts += [ctx + " " + completion for ctx, completion in demonstrations]  # K-shot
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
        return cond                                   # per-token conditional log-likelihood
    base = tokenizer.encode(answer_context)           # generic lead-in, e.g. "Answer:"
    uncond = _completion_logprob(model, base, comp)
    return cond - uncond                              # divide out generic completion likelihood

@torch.no_grad()
def beam_search(model, idx, max_new_tokens, beam_width=4, length_penalty=0.6, eos_token_id=None):
    # batch size 1; rank partial continuations by sum log-prob / length**length_penalty
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

## Summary

GPT-3 is the GPT-2 architecture scaled to 175B parameters (with alternating dense/sparse attention and a 2048 context), trained with the standard autoregressive objective on quality-filtered web data, under the compute-optimal "large model / fewer tokens" regime from the scaling laws. Its contribution is not a new mechanism: it makes in-context learning a first-class evaluation interface by specifying tasks purely as text and adapting through the forward pass with no gradient updates.
