The cost that drives this whole problem is storage and serving at scale. I have one enormous frozen text-to-text Transformer — a T5 of up to 11 billion parameters — and I want it to handle many downstream tasks. Full fine-tuning, what I'll call model tuning, gives the best accuracy but produces a separate 11B copy of the model *per task*: the per-task footprint is the entire model, every time, which is untenable when serving many tasks. So the real question is the smallest thing I can learn that re-specializes a frozen model for a new task, leaving the shared weights untouched. There is a strong existence proof for "a tiny input signal can carry a task" in prompting: a frozen model can be conditioned simply by prepending tokens $P$ to the input so it maximizes $\Pr_\theta(Y \mid [P; X])$ with $\theta$ fixed, and a large enough frozen model given the right prompt is a capable few-shot learner. But the existing routes each fall short. Discrete prompt design forces $P$ to be a sequence of real vocabulary tokens, so the prompt's representation is pulled from the frozen embedding table; finding a good prompt means selecting discrete token IDs by hand or by non-differentiable search, and every slot is hard-capped at the embedding of some actual word — a ceiling on what the prompt can express. Prefix-tuning relaxes the discreteness but goes much further, learning trainable activation prefixes at *every* transformer layer with a reparametrization MLP for stable optimization, so it carries per-layer task-specific parameters and modifies the model's internal activations throughout the stack. Adapters insert small trainable bottleneck modules into the backbone. None of these is the minimal, weights-frozen, input-only intervention I am after.

I propose prompt tuning. The model never consumes token *IDs* directly: the first thing T5 does with input tokens is embed them into a matrix $X_e \in \mathbb{R}^{n \times e}$, so the prompt only ever matters through its embeddings. That observation removes the reason to keep the prompt made of real words at all. Let the prompt be its own block of free parameters living directly in embedding space — a soft prompt $P_e \in \mathbb{R}^{p \times e}$ with $p$ the prompt length and $e$ the embedding dimension. Concatenate it to the embedded input to form $[P_e; X_e] \in \mathbb{R}^{(p+n) \times e}$, push that through the encoder-decoder exactly as normal, and train by maximizing $$\Pr_{\theta;\,\theta_P}(Y \mid [P; X]),$$ but freezing $\theta$ and updating *only* $P_e$ by backpropagation. Prompt design selects from a fixed vocabulary of frozen embeddings; prompt tuning is that same idea with the discrete constraint deleted — a fixed block of special "virtual tokens" whose embeddings are free to move anywhere. The parameter cost is exactly $E \cdot P$, embedding dimension times prompt length; with $p \approx 100$ and $E$ a few thousand that is a few hundred thousand parameters against an 11B backbone, under 0.01% per task.

The crucial restraint is deliberate: I intervene *only at the input layer*, $P_e$ and nothing else. I could reach deeper and learn activation prefixes at every layer — more expressive, but needing the reparametrization MLP for stability — yet I want to test the sharper hypothesis that at sufficient scale I do not need to reach into the network at all. A signal at the input alone, the cheapest possible intervention with just $E \cdot P$ parameters, no per-layer activations, no reparametrization, no inserted modules, should be enough, and become *more* than enough as the model grows. This is the central finding, the power of scale: the larger the frozen model, the more competitive and the more forgiving this minimal intervention becomes, until at XXL (11B) it closes the gap to full model tuning — even matching the strong multi-task model-tuning baseline — despite carrying more than twenty thousand times fewer task-specific parameters.

Two design knobs decide whether the soft prompt actually works, and one prerequisite about the base model dominates everything. Take the base model first, because it is a wall. T5.1.1 is pre-trained *only* on span corruption: spans of the input are masked with unique sentinel tokens and the target is the masked content delimited by sentinels, so every pre-training target literally *begins* with a sentinel. Such a model has never read fully natural text nor produced fully natural text, and its decoder carries a strong prior toward emitting sentinels. Full fine-tuning overrides that prior trivially by moving the decoder weights — but a frozen model steered by an input prompt cannot adjust the decoder's output priors. So a span-corruption model should be a poor frozen base for prompting, and that is exactly what happens: prompting a mid-sized span-corruption T5 off the shelf is unreliable, on many tasks it never emits a legal class label and scores 0%, with failure modes of copying sub-spans of the input or predicting the empty string, stable across runs rather than noise. Meanwhile GPT-3, a left-to-right model that always outputs natural text, prompts beautifully. The lesson is that prompting wants a frozen base that was pre-trained to read and write *natural* text. To get T5 there without re-pretraining from scratch, I continue its self-supervised training briefly under the LM objective instead — given a natural prefix, produce the natural continuation — for up to about 100K extra steps, roughly 10% of original pre-training, *once*, yielding a single LM-adapted frozen model reused for every downstream task. Merely prepending a sentinel to downstream targets papers over the symptom and barely helps; the LM adaptation is what fixes the cause.

The second knob is how to initialize $P_e$. The simplest is random — sample each coordinate uniformly from $[-0.5, 0.5]$ — but conceptually the soft prompt modulates the frozen network the way text *preceding* the input would, so a word-like starting point lands it in a region of embedding space the model already understands. A better option is to initialize each prompt token from an actual vocabulary embedding, drawn from the 5,000 most common SentencePiece tokens. Best of all for classification is a verbalizer-style choice that primes the output directly: initialize prompt tokens with the embeddings of the *class-label strings* themselves, averaging sub-token embeddings for multi-token labels, and when the prompt is longer than the number of labels, fall back to sampled-vocabulary embeddings for the remaining slots. Because the model must *produce* those label tokens in its output, seeding the prompt with them nudges it to restrict its output to the legal classes from the start. So random lags behind word-like starts, and class-label init is best where the model is still fragile — but, again, the gaps shrink with scale and essentially vanish at XXL.

The third knob is prompt length. Since the cost is $E \cdot P$, I want the smallest $P$ that still performs. Going from a single token to about 20 matters a great deal for most sizes — one vector is a narrow channel for a whole task — while past 20 the extra slots are merely more continuous conditioning capacity, so the curve flattens, and past about 100 it can even dip for larger models, matching the diminishing-then-declining pattern seen for continuous prefixes. So a default near 100 is a conservative ablation point, but at XXL the model needs so little conditioning signal that even a single-token prompt becomes viable — the cleanest expression of the scale story. The same pattern runs through all three knobs: base-model objective, initialization, and prompt length each matter a lot at small and mid sizes and become much less decisive at XXL, where the largest model is robust to non-ideal choices.

Optimization is unremarkable and I pin it down for completeness: train only $P_e$ with Adafactor at a constant learning rate of 0.3, batch size 32, around 30K steps, weight decay $10^{-5}$, $\beta_2$ decay 0.8, and parameter scaling *off* — the prompt is a single small tensor, so Adafactor's parameter-scaling heuristic is not what I want here — with standard cross-entropy on the text-to-text target and early stopping on dev. Two consequences fall out for free from "the task is a small input matrix and the model is frozen": different examples in one batch can carry different prompts against the same backbone, so many tasks can be served from one frozen model and even batched together; and several prompts trained for a single task form a cheap ensemble, many "models" at the cost of many small prompts plus one shared backbone. The implementation is just the constraint made literal — create one trainable matrix, prepend it to the embedded input, and make the optimizer blind to everything else.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPrompt(nn.Module):
    """Trainable virtual-token embeddings prepended to the input. Cost = length * E."""
    def __init__(self, length, embed_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(length, embed_dim))

    @torch.no_grad()
    def init_from(self, strategy, frozen_embed, class_label_ids=None):
        if strategy == "random":
            self.prompt.uniform_(-0.5, 0.5)
        elif strategy == "sampled_vocab":
            idx = torch.randint(0, 5000, (self.prompt.size(0),))      # common tokens
            self.prompt.copy_(frozen_embed(idx))
        elif strategy == "class_label":
            rows = [frozen_embed(ids).mean(0) for ids in class_label_ids]
            while len(rows) < self.prompt.size(0):                    # fall back to vocab
                rows.append(frozen_embed(torch.randint(0, 5000, (1,))).squeeze(0))
            self.prompt.copy_(torch.stack(rows[: self.prompt.size(0)]))

    def forward(self, x_e):
        p = self.prompt.unsqueeze(0).expand(x_e.size(0), -1, -1)
        return torch.cat([p, x_e], dim=1)                            # [P_e ; X_e]


def loss_fn(frozen_t5, soft_prompt, input_ids, target_ids):
    x_e = frozen_t5.embed(input_ids)
    inputs_embeds = soft_prompt(x_e)
    logits = frozen_t5(inputs_embeds=inputs_embeds, decoder_target=target_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))


def train(frozen_t5, soft_prompt, loader):
    for p in frozen_t5.parameters():
        p.requires_grad = False                                       # frozen, LM-adapted base
    optim = Adafactor(soft_prompt.parameters(), lr=0.3, weight_decay=1e-5,
                      beta2_decay=0.8, parameter_scaling=False)
    for batch in loader:                                              # batch 32, ~30k steps
        loss = loss_fn(frozen_t5, soft_prompt, batch["input_ids"], batch["target_ids"])
        loss.backward(); optim.step(); optim.zero_grad()
```
