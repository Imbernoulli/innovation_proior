Let me think about the cost that's actually killing me. I have one enormous frozen T5 — up to 11 billion parameters — and I want it to do many downstream tasks. Full fine-tuning ("model tuning") gives top accuracy but produces a separate 11B copy *per task*. That's absurd at scale: the per-task storage and serving cost is the entire model, every time. I want one frozen model, shared across all tasks, plus a per-task signal so small it's a rounding error. So: what is the *smallest* thing I can learn that re-specializes a frozen model for a task?

There's a strong existence proof sitting right there in prompting. A frozen model can be conditioned just by prepending tokens P to the input — it maximizes Pr_θ(Y | [P; X]) with θ untouched — and GPT-3 showed that a large enough frozen model, given the right prompt, is a capable few-shot learner. So the prompt *alone* can carry the task. The catch with prompt design is that P is a sequence of real vocabulary tokens, so the prompt's representation is pulled from the frozen embedding table; choosing a good prompt means selecting discrete token IDs, by hand or by non-differentiable search. Discrete search is awkward, and worse, every prompt slot is constrained to be the embedding of some actual word. That's a hard ceiling on what the prompt can express.

So relax the constraint. The model doesn't consume token *IDs* — the first thing T5 does with input tokens is embed them into a matrix X_e ∈ R^{n×e}. The prompt only matters through its embeddings. So why force the prompt to be real words at all? Let the prompt be its *own* set of free parameters in embedding space: a soft prompt P_e ∈ R^{p×e}, p the prompt length, e the embedding dimension. Concatenate it to the embedded input, [P_e; X_e] ∈ R^{(p+n)×e}, and let that flow through the encoder-decoder exactly as normal. Train by maximizing Pr_{θ;θ_P}(Y | [P; X]) — but freeze θ and update *only* P_e by backpropagation. Prompt *design* selects from a fixed vocabulary of frozen embeddings; prompt *tuning* is the same idea with the discrete constraint removed — a fixed block of special "tokens" whose embeddings are free to move. The parameter cost is exactly E·P: embedding dimension times prompt length. With p around 100 and E a few thousand, that's a few hundred thousand parameters against an 11B model — under 0.01% per task.

Now, the crucial restraint, and it's a deliberate one. I could intervene deeper — make trainable activations at every layer, overwrite the per-layer key/value activations through the whole stack (that's the prefix-tuning route), which is more expressive and needs a reparametrization MLP to optimize stably. But I want to test a sharper hypothesis: maybe at sufficient *scale*, I don't need to reach into the network at all. Maybe a signal at the input layer alone — the cheapest possible intervention, just E·P parameters, no per-layer activations, no reparametrization, no extra modules in the backbone — is enough, and gets *more* enough as the model grows. So I'm going to tune only the input: just P_e. If that holds, it's not only the most parameter-efficient option, it's the cleanest — the frozen model is genuinely untouched, and the entire task lives in a small matrix prepended at the input.

Two design questions then decide whether this works: how to initialize P_e, and how long to make it. And one prerequisite question about the *base model* that turns out to dominate everything.

Take the base model first, because I hit a wall here. T5.1.1 is pre-trained *only* on span corruption: spans of the input are masked with unique sentinel tokens, and the target is the masked content delimited by sentinels — every pre-training target literally *begins* with a sentinel. So this model has never read fully natural text and has never been asked to produce fully natural text; its decoder carries a strong prior toward emitting sentinels. With full fine-tuning that prior is trivially overridden — I can move the decoder weights. But I'm *freezing* the model and steering it with a prompt, and a prompt at the input cannot adjust the decoder's output priors. So I should expect a span-corruption model to be a bad frozen base for prompting. And that's exactly what shows up: prompting mid-sized span-corruption T5 off the shelf is unreliable — on many tasks the model never emits a legal class label and scores 0%, with failure modes of copying sub-spans of the input or predicting the empty string, and this is stable across runs, not noise. Only some sizes work at all. Meanwhile GPT-3, a left-to-right model that always outputs natural text, prompts beautifully. The lesson: prompting wants a frozen model that was pre-trained to read and write *natural* text.

How do I get T5 into that regime without re-pretraining from scratch? Continue its self-supervised training for a short while under the *LM objective* instead — given a natural text prefix, produce the natural continuation. Do this once, for up to ~100K extra steps (about 10% of original pre-training), producing a single LM-adapted frozen model I reuse for every downstream task. The bet is that I can "quickly" convert a span-corruption model into something GPT-3-like that always emits realistic text. It's not obvious a late-stage objective switch takes — it might be a deep change — but empirically LM adaptation helps across all sizes, longer adaptation helps up to 100K steps, and the workaround of merely prepending a sentinel to downstream targets barely helps (it papers over the symptom, not the prior). So: LM-adapt once, then prompt-tune.

Now initialization of P_e. The simplest is random — sample each coordinate uniformly, say from [−0.5, 0.5]. But conceptually, my soft prompt modulates the frozen network the same way *text preceding the input* would, so a *word-like* starting point should land it in a region of embedding space the model already understands. So a better option: initialize each prompt token from an actual vocabulary embedding (drawn from the most common tokens). And for classification specifically, a third option that primes the output directly: initialize prompt tokens with the embeddings of the *class-label strings* themselves (averaging sub-token embeddings for multi-token labels) — like a verbalizer. Since the model must *produce* those label tokens in its output, seeding the prompt with them nudges it to restrict its output to the legal classes from the start. When the prompt is longer than the number of class labels, fall back to sampled-vocabulary for the remaining slots. I expect: random < sampled-vocab < class-label, and that's what the ablation shows — class-label init is best at small sizes, with large gaps between strategies. But — and this is the thread that ties the whole thing together — those gaps *shrink as the model scales*, and at XXL they essentially vanish. The bigger the model, the less the initialization matters.

Prompt length next. The parameter cost is E·P, so I want the smallest P that still performs well. Going from a single token to ~20 generally gives a large boost; past 20 the gains are marginal, and past ~100 it's even mildly detrimental for large models (a longer prompt starts to overfit, the same diminishing-then-declining pattern that shows up for continuous prefixes in general). So a default around 100 with the option to go much shorter. And again the scale story: at XXL the model performs well even with a *single-token* prompt — the larger the model, the less conditioning signal it needs to evoke a target behavior. So I'll default to length 100 but expect short prompts to be viable as size grows.

Notice the pattern across all three knobs — base-model objective, initialization, prompt length: each matters a lot at small and mid sizes, and each stops mattering at XXL. The largest model is robust to every non-ideal choice. That's the real finding here: it's not just that a single soft prompt at the input *can* re-specialize a frozen model, it's that the bigger the frozen model, the more forgiving and the more competitive this minimal intervention becomes — to the point where, at the 11B scale, tuning only a small input prompt closes the gap to full model tuning, despite having tens of thousands of times fewer task-specific parameters. The power is in the scale.

Optimization is unremarkable and I'll just pin it down: train only P_e with Adafactor, a constant learning rate (0.3 works), batch size 32, around 30K steps, weight decay 1e-5, β₂ decay 0.8, and parameter scaling off (the prompt is a single small tensor, so Adafactor's parameter-scaling heuristic isn't what I want here). Standard cross-entropy on the text-to-text target, early-stopping on dev.

A couple of free consequences fall out of "the task is a small input matrix, the model is frozen." Different examples in one batch can carry different prompts against the same backbone, so I can serve many tasks from one frozen model and even batch them together. And I can build a cheap ensemble for a single task by training several prompts and combining their predictions — many "models" at the cost of many small prompts plus one shared backbone. Both come for free from putting the task at the input and leaving θ alone.

Here is the code these decisions land on.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPrompt(nn.Module):
    """A trainable block of `length` virtual-token embeddings prepended to the input.
    Tunes ONLY the input layer: parameter cost = length * E."""
    def __init__(self, length, embed_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(length, embed_dim))

    @torch.no_grad()
    def init_from(self, strategy, frozen_embed, class_label_ids=None):
        if strategy == "random":                      # uniform [-0.5, 0.5]
            self.prompt.uniform_(-0.5, 0.5)
        elif strategy == "sampled_vocab":             # common vocabulary embeddings
            idx = torch.randint(0, 5000, (self.prompt.size(0),))
            self.prompt.copy_(frozen_embed(idx))
        elif strategy == "class_label":               # verbalizer-style priming
            rows = []
            for ids in class_label_ids:               # average multi-token labels
                rows.append(frozen_embed(ids).mean(0))
            for j in range(len(rows), self.prompt.size(0)):   # fall back to sampled vocab
                rows.append(frozen_embed(torch.randint(0, 5000, (1,))).squeeze(0))
            self.prompt.copy_(torch.stack(rows[: self.prompt.size(0)]))

    def forward(self, x_e):                           # prepend to embedded input
        p = self.prompt.unsqueeze(0).expand(x_e.size(0), -1, -1)
        return torch.cat([p, x_e], dim=1)             # [P_e ; X_e]


def loss_fn(frozen_t5, soft_prompt, input_ids, target_ids):
    x_e = frozen_t5.embed(input_ids)                  # frozen embedding lookup
    inputs_embeds = soft_prompt(x_e)                  # [(p+n) x E], rest of model unchanged
    logits = frozen_t5(inputs_embeds=inputs_embeds, decoder_target=target_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))


def train(frozen_t5, soft_prompt, loader, opt):
    for p in frozen_t5.parameters():
        p.requires_grad = False                       # one shared frozen backbone (LM-adapted)
    optim = Adafactor(soft_prompt.parameters(), lr=0.3, weight_decay=1e-5,
                      beta2_decay=0.8, parameter_scaling=False)
    for batch in loader:                              # batch size 32, ~30k steps
        loss = loss_fn(frozen_t5, soft_prompt, batch["input_ids"], batch["target_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```

The causal chain: serving an 11B model across many tasks forces a tiny per-task signal on a frozen backbone; prompting shows a context alone can carry a task, but discrete prompts are search-bound and capped to real-word embeddings, so make the prompt a free matrix in embedding space; bet that, with scale, intervening at the *input layer only* (just E·P parameters, no per-layer activations, no MLP) suffices; a frozen base must already read/write natural text, so LM-adapt the span-corruption T5 once before prompting; initialize the prompt from class-label (then vocabulary) embeddings to prime legal outputs; pick the shortest prompt that performs (≈100, fewer at larger scale); and the headline is that every one of these choices stops mattering as the model grows — at XXL a small input prompt matches full model tuning with tens of thousands of times fewer task-specific parameters.
