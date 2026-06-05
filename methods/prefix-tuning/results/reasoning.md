Let me start from the thing that's bugging me about fine-tuning. To adapt one pre-trained generation model — GPT-2 for table-to-text, BART for summarization — I copy the whole model and tune every weight, and now I own a full several-hundred-million-parameter checkpoint *per task*. If I'm serving many tasks, or many users each wanting their own behavior, that's the whole cost. I want one frozen backbone shared across everything, plus a tiny per-task object. So the question is: what's the smallest thing I can learn that re-purposes a frozen LM for a task?

There's a clue from prompting that I keep coming back to. A frozen LM can be steered by its context, with no weight changes at all. If I want it to say "Obama," I prepend "Barack" and the next-token probability of "Obama" shoots up. The context reaches into the model and reshapes the output distribution without touching a single parameter. So maybe, for a whole task, there exists *some* context that steers the LM to solve it — a context that tells it "you are now summarizing" or "you are now describing this table." If such a context exists, I'd pay almost nothing per task: just store the context.

So let me try the obvious version: write a natural-language instruction, like "summarize the following table in one sentence," prepend it, freeze everything. It fails. GPT-2 and BART mostly don't follow instructions — that capability only really shows up in the very largest models. So a hand-written instruction isn't the steering context I need. Fine, then *search* for a better instruction. But the instruction is a sequence of discrete tokens, and searching over discrete token sequences is combinatorial and brutal to optimize — gradient-guided discrete search (like AutoPrompt) exists, but it's expensive and, more fundamentally, every slot in the prompt is forced to be the embedding of an actual vocabulary word. That's a hard constraint that caps how expressive the steering context can be. The best discrete prompt is bounded by "what can be said with real words in these positions."

Here's the move that dissolves the discrete problem. The only reason the prompt is hard to optimize is that I'm insisting each position be a real token. But the LM doesn't consume tokens — it consumes their *embeddings*, continuous vectors. So why restrict to real-word embeddings? Let the prompt positions be *free continuous vectors*, optimized directly by gradient descent against the task loss. Now there's no discrete search — it's smooth optimization — and it's strictly more expressive than a discrete prompt, because a discrete prompt is just the special case where each vector happens to equal some real word's embedding. So: prepend a few "virtual tokens" whose embeddings are trainable free parameters, freeze the LM, and optimize those embeddings on the task. Call this the embedding-only approach.

Let me think about whether embedding-only is *enough* before I commit. The virtual-token embeddings sit at the very bottom of the network. Their influence has to propagate up through every layer and rightward to the real tokens, entirely through the frozen Transformer's computation. That's a long, indirect path: the only handle I have on what happens at layer 12 is whatever the frozen layers 1–11 decide to do with my bottom-layer vectors. I'm asking a small set of input vectors to bend the behavior of the whole deep stack through a fixed function. That's a lot of long-range dependency riding on very few, very distant parameters. I suspect this will be underpowered — and indeed, when I try it, embedding-only steering works but underperforms; tuning only the embedding layer just isn't expressive enough to match fine-tuning. So the chain of increasing expressiveness is: discrete prompting < embedding-only tuning < (something that intervenes deeper). I need to intervene deeper.

What does "deeper" mean concretely? Recall how the Transformer computes a position's activation: at each layer, position i attends to the previous layer's activations in its left context. So the activation of a virtual-token position isn't just its input embedding — it's a *whole stack* of per-layer activations, h_i = [h_i^(1); …; h_i^(n)], one vector per layer. In embedding-only, I only set h_i^(1) (the bottom) and let the frozen LM compute the rest. But I could instead make the virtual-token activations free at *every* layer. That is: don't just feed trainable embeddings at the input — directly supply the entire per-layer activation stack of the prefix positions as trainable parameters. Then the prefix's influence enters the real tokens' attention at every layer directly, not filtered through eleven frozen layers first. Much shorter dependency paths, far more tunable parameters, easier to optimize.

So here's the method. Prepend |P| prefix positions, getting z = [P; x; y] for an autoregressive LM (or a prefix on each side, [P; x; P′; y], for encoder-decoder). Allocate a trainable matrix P_θ of shape |P| × dim(h_i), where dim(h_i) is the full per-layer activation size (the concatenation over all layers). The recurrence is unchanged for real tokens, but the prefix positions read their activations straight from P_θ:

  h_i = P_θ[i, :]            if i is a prefix position,
  h_i = LM_φ(z_i, h_{<i})    otherwise.

Freeze φ entirely; the *only* trainable parameters are P_θ. The objective is identical to fine-tuning — maximize Σ_{i∈y} log p_φ(z_i | h_{<i}) — just over a tiny parameter set. Every real activation h_i (i not in the prefix) still depends on P_θ, because the prefix sits in the left context of every real token and is attended to at every layer. So the prefix steers both the *encoding of x* (guiding what the model extracts from the input) and the *generation of y* (steering the next-token distribution), through ordinary attention.

Where exactly to put the trainable positions matters, and I should reason it through rather than default to "the front." If I place them at the front — prefixing, [P; x; y] — the prefix is in the left context of both x and y, so it can influence how x is encoded *and* how y is generated. If instead I place them between x and y — infixing, [x; P; y] — then x is encoded *before* the trainable positions appear, so the trainable activations can only affect y, not the encoding of x. Prefixing strictly dominates infixing in reach, so I expect prefixing to do at least as well; empirically infixing is slightly worse, which matches this reasoning. Front it is.

Now a subtlety about implementation that's actually a gift. "Supply the prefix's per-layer key/value activations as extra left context" is exactly what a Transformer's cached-attention mechanism already does — the `past_key_values` interface lets me inject precomputed per-layer key/value tensors that every real position attends to, without recomputing anything. So the prefix is naturally represented as, per layer, a set of |P| key vectors and |P| value vectors. dim(h_i) is therefore 2 · n_layers · d_model worth of activation per prefix position (a key and a value of size d_model at each of the n_layers layers). I learn that whole stack.

Let me check the optimization behaves. If I make P_θ a raw free matrix and optimize it directly, training turns out to be unstable and very sensitive to learning rate and initialization, with a slight performance hit. The values in P_θ live in the LM's activation space, which has its own scale and correlations across the dim(h_i) coordinates; a raw matrix has no structure tying those coordinates together, so gradient descent thrashes. The fix is to reparametrize: instead of optimizing the big matrix directly, optimize a *smaller* matrix P′_θ of shape |P| × k (same number of rows = prefix length, but a much smaller column dimension k), and pass each row through an MLP to expand it to dim(h_i):

  P_θ[i, :] = MLP_θ(P′_θ[i, :]).

The MLP shares parameters across all prefix positions and across the expansion, which couples the coordinates and gives a smoother, better-conditioned optimization landscape — training is now stable. k is small (around 512 for table-to-text, 800 for summarization); the MLP maps k → dim(h_i). And here's the nice part: the MLP and P′_θ are only needed *during training*. Once trained, I just evaluate P_θ = MLP_θ(P′_θ) once and store the resulting |P| × dim(h_i) prefix matrix; the reparametrization machinery is thrown away. So the stored per-task object is exactly the prefix activations — nothing else.

How long should the prefix be? Longer prefix = more trainable parameters = more expressive, but also more to overfit. I expect performance to climb with |P| up to a point and then sag slightly as the longer prefix starts memorizing the training set rather than generalizing — and that's what happens: a threshold around 10 for table-to-text and around 200 for summarization (summarization is harder and needs more steering capacity), beyond which test performance dips slightly even as training loss keeps dropping. Importantly, a longer prefix barely costs anything at inference, because attention over the whole prefix is parallelized on the GPU — it's not like adding sequential depth.

One more thing, initialization, which matters a lot in low-data settings. If I initialize P′_θ (and hence the prefix) randomly, I get low performance with high variance — random vectors in activation space are a bad starting point. Far better: initialize the prefix from the *activations of real words* computed by the frozen LM itself. Run some real tokens through the LM, grab their per-layer activations, and use those to seed the prefix. This starts the prefix in a region of activation space the LM already "speaks," and it's concordant with the whole philosophy of disturbing the pre-trained model as little as possible. Task-relevant words ("summarization", "table-to-text") seed slightly better than irrelevant ones ("elephant", "divide"), but any real-word seeding beats random.

Step back and notice what I've gained beyond parameter count. Because the entire task-specific object is a prefix that lives in the *input/activation* side and the backbone is untouched, different examples in one batch can carry *different* prefixes against the same frozen model — so I can batch requests for different tasks or different users together, which a per-task fine-tuned checkpoint can't do. That falls out for free from "the task is encoded as context, not as weights."

Here is the code these decisions land on.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixTuning(nn.Module):
    """Trainable per-layer key/value prefix that steers a frozen LM.
    Reparametrized through an MLP for stable optimization; collapse to raw
    prefix activations after training."""
    def __init__(self, lm_config, prefix_len=10, mid_dim=512):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layer = lm_config.n_layer
        self.n_head = lm_config.n_head
        self.d_model = lm_config.n_embd
        self.head_dim = self.d_model // self.n_head
        # small matrix P'_theta: one low-dim vector per prefix position
        self.input_tokens = torch.arange(prefix_len)
        self.embed = nn.Embedding(prefix_len, self.d_model)
        # MLP expands d_model -> all layers' key+value activations (dim(h_i))
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, self.n_layer * 2 * self.d_model),  # 2 = key + value
        )

    def materialize(self, batch_size, device):
        """Produce past_key_values: per layer, (key, value) of shape
        [batch, n_head, prefix_len, head_dim]. This is dim(h_i) for the prefix."""
        idx = self.input_tokens.to(device)
        h = self.mlp(self.embed(idx))                              # [prefix_len, n_layer*2*d_model]
        h = h.unsqueeze(0).expand(batch_size, -1, -1)              # batch can carry distinct prefixes
        h = h.view(batch_size, self.prefix_len, self.n_layer * 2, self.n_head, self.head_dim)
        h = h.permute(2, 0, 3, 1, 4)                               # [n_layer*2, batch, n_head, len, hd]
        past = []
        for l in range(self.n_layer):
            past.append((h[2 * l], h[2 * l + 1]))                  # (key, value) for layer l
        return past   # prefix occupies the left context at EVERY layer


def loss_fn(frozen_lm, prefix, x_ids, y_ids):
    z = torch.cat([x_ids, y_ids], dim=1)                          # [P ; x ; y] via past_key_values
    past = prefix.materialize(z.size(0), z.device)
    out = frozen_lm(input_ids=z, past_key_values=past)            # frozen LM attends to the prefix
    logits = out.logits[:, x_ids.size(1) - 1:-1, :]               # predict the y positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))


def train(frozen_lm, prefix, loader, opt):
    for p in frozen_lm.parameters():
        p.requires_grad = False                                   # backbone frozen
    optim = torch.optim.AdamW(prefix.parameters(), lr=opt.lr)     # only the prefix module trains
    for batch in loader:                                          # default: 10 epochs, bs 5, lr 5e-5
        loss = loss_fn(frozen_lm, prefix, batch["x_ids"], batch["y_ids"])
        loss.backward(); optim.step(); optim.zero_grad()


# After training: store only prefix.materialize(1, dev); drop the MLP/embedding.
```

The causal chain: serving many tasks cheaply forces a tiny per-task object on a frozen backbone; "steer-by-context" suggests a learned context, but discrete prompts are hard to optimize and capped to real-word embeddings, so relax to continuous virtual-token vectors; those at the input layer alone are too weak (too long a dependency path through the frozen stack), so make the prefix's activations trainable at *every* layer — i.e. supply per-layer key/value activations via the existing cached-attention path; prefixing dominates infixing because it can shape both the encoding of x and the generation of y; raw optimization of the activation matrix is unstable, so reparametrize through a shared MLP from a small matrix and discard it after training; prefix length trades expressiveness against overfitting (≈10 vs ≈200 by task); and seeding from real-word activations stabilizes low-data training — yielding a method that stores only a small prefix per task, leaves the LM untouched, and even lets one batch mix prefixes across tasks.
