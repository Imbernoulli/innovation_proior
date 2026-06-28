Let me start from the thing that's bugging me about fine-tuning. To adapt one pre-trained generation model — GPT-2 for table-to-text, BART for summarization — I copy the whole model and tune every weight, and now I own a full several-hundred-million-parameter checkpoint *per task*. If I'm serving many tasks, or many users each wanting their own behavior, that's the whole cost. I want one frozen backbone shared across everything, plus a tiny per-task object. So the question is: what's the smallest thing I can learn that re-purposes a frozen LM for a task?

There's a clue from prompting that I keep coming back to. A frozen LM can be steered by its context, with no weight changes at all. If I want it to say "Obama," I prepend "Barack" and the next-token probability of "Obama" shoots up. The context reaches into the model and reshapes the output distribution without touching a single parameter. So maybe, for a whole task, there exists *some* context that steers the LM to solve it — a context that tells it "you are now summarizing" or "you are now describing this table." If such a context exists, I'd pay almost nothing per task: just store the context.

So let me try the obvious version: write a natural-language instruction, like "summarize the following table in one sentence," prepend it, freeze everything. It fails. GPT-2 and BART mostly don't follow instructions — that capability only really shows up in the very largest models. So a hand-written instruction isn't the steering context I need. Fine, then *search* for a better instruction. But the instruction is a sequence of discrete tokens, and searching over discrete token sequences is combinatorial and brutal to optimize — gradient-guided discrete search (like AutoPrompt) exists, but it's expensive and, more fundamentally, every slot in the prompt is forced to be the embedding of an actual vocabulary word. That's a hard constraint that caps how expressive the steering context can be. The best discrete prompt is bounded by "what can be said with real words in these positions."

The only reason the prompt is hard to optimize is that I'm insisting each position be a real token. But the LM doesn't consume tokens — it consumes their *embeddings*, continuous vectors. So why restrict to real-word embeddings? Let the prompt positions be *free continuous vectors*, optimized directly by gradient descent against the task loss. Now there's no discrete search — it's smooth optimization. And I should be precise about the relationship rather than just calling it "more expressive": a discrete prompt feeds, into each slot, one row of the frozen embedding matrix; the continuous version optimizes a free vector in that same space. So the set of discrete prompts is exactly the subset of continuous prompts where each vector happens to land on a real word's embedding row — a finite subset of a continuous space. The continuous parameterization therefore contains the discrete one as a special case and can only do at least as well at the training objective. That's a containment I can state confidently because it's a fact about the parameter sets, not about a benchmark. So: prepend a few "virtual tokens" whose embeddings are trainable free parameters, freeze the LM, and optimize those embeddings on the task. Call this the embedding-only approach.

Let me think about whether embedding-only is *enough* before I commit. The virtual-token embeddings sit at the very bottom of the network. Their influence has to propagate up through every layer and rightward to the real tokens, entirely through the frozen Transformer's computation. That's a long, indirect path: the only handle I have on what happens at layer 12 is whatever the frozen layers 1–11 decide to do with my bottom-layer vectors. I'm asking a small set of input vectors to bend the behavior of the whole deep stack through a fixed function. The containment argument from a moment ago does *not* extend here: I can say discrete ⊆ embedding-only as parameter sets, but "embedding-only is weaker than intervening deeper" is not something I can settle by a containment argument — it's an optimization-difficulty claim about a lot of long-range dependency riding on very few, very distant parameters. I genuinely don't know how badly it bites without running it, so I'll mark it as a worry rather than a proof and design for the case where it does bite: I want a knob that intervenes deeper, with shorter dependency paths.

What does "deeper" mean concretely? Recall how the Transformer computes a position's activation: at each layer, position i attends to keys and values from the previous positions in that same layer's left context. So the virtual positions should not merely have bottom embeddings that the frozen network turns into whatever it turns them into; they should supply the attention objects the later real tokens will actually read. For every layer, give the virtual positions their own trainable key vectors and value vectors. Then the prefix's influence enters the real tokens' attention at every layer directly, not filtered through eleven frozen layers first. Much shorter dependency paths, far more tunable parameters, easier to optimize.

The construction is now concrete. Prepend |P| prefix positions, getting z = [P; x; y] for an autoregressive LM (or a prefix on each side, [P; x; P'; y], for encoder-decoder). Let me actually count the width of the trainable object so I know what I'm allocating. One prefix position contributes, at each layer, exactly one key vector and one value vector, each of width d_model (across all heads). So per prefix position the flattened cache has width 2 (key, value) × n_layers × d_model. For a 12-layer model with d_model = 768 that's 2 × 12 × 768 = 18,432 numbers per prefix position; with |P| = 10 positions the whole per-task object is 184,320 numbers — under 0.2M parameters against a backbone of hundreds of millions. Good, that's the order of magnitude I was hoping for. Before flattening, the natural shape is [layer, key/value, head, prefix_position, head_dim]. Allocate P_θ as exactly that trainable per-layer key/value prefix. In the activation notation, the recurrence is unchanged for real tokens, but the prefix positions read their stored activations from P_θ:

  h_i = P_θ[i, :]            if i is a prefix position,
  h_i = LM_φ(z_i, h_{<i})    otherwise.

Freeze φ entirely; the only trainable parameters are the prefix-side parameters θ. The objective is identical to fine-tuning — maximize Σ_{i∈y} log p_φ(z_i | h_{<i}) — just over a tiny parameter set. Every real activation h_i (i not in the prefix) still depends on P_θ, because the prefix keys and values sit in the left context of every real token at every layer. So the prefix steers both the *encoding of x* (guiding what the model extracts from the input) and the *generation of y* (steering the next-token distribution), through ordinary attention.

Where exactly to put the trainable positions matters, and I should reason it through rather than default to "the front." Causal attention only lets a position attend leftward. If I place the trainable positions at the front — prefixing, [P; x; y] — every x position and every y position has the prefix in its left context, so the prefix can influence how x is encoded *and* how y is generated. If instead I place them between x and y — infixing, [x; P; y] — then by the time the trainable positions appear, the x activations are already fixed: each x position attends only to earlier x positions, never to the later P block, so P cannot touch the encoding of x at all; it reaches only y. The reach of infixing is thus a strict subset of the reach of prefixing — prefixing can do everything infixing can (and also shape x's encoding), so it can't be worse on the training objective. Front it is.

Now a subtlety about implementation is actually a gift. "Supply the prefix's per-layer key/value activations as extra left context" is exactly what a Transformer's cached-attention mechanism already does. The `past_key_values` interface lets me inject precomputed per-layer key/value tensors that every real position attends to, without recomputing anything. So the implementation object is a tuple with one `(key, value)` pair per layer, each shaped `[batch, n_head, prefix_len, head_dim]`. The learned prefix is not a new hidden layer inside the LM; it is a learned cache prepended at every layer.

The parameterization itself can become the bottleneck. P_θ is a raw cache spanning layers, heads, keys, and values, and those coordinates live in the LM's activation space with its own scales and correlations. If I push directly on every coordinate, optimization can become learning-rate and initialization sensitive. I want the served object to remain just the cache, but I can use a smoother training parameterization to get there: optimize a smaller matrix P'_θ of shape |P| × k (same number of rows = prefix length, but a much smaller column dimension k), and pass each row through an MLP to expand it to the flattened key/value cache:

  P_θ[i, :] = MLP_θ(P'_θ[i, :]).

The MLP shares parameters across prefix positions and maps the low-dimensional rows into a cache-shaped space, which is the stability reason for the reparameterization. k is 512 for table-to-text and 800 for summarization; the MLP maps k to the flattened key/value prefix. The MLP and P'_θ are only needed during training. Once trained, I evaluate P_θ = MLP_θ(P'_θ) once and store the resulting per-layer key/value tensors; the reparameterization machinery is thrown away. So the stored per-task object is exactly the prefix cache — nothing else.

Before I trust the `materialize` step, let me trace the shape algebra on a toy config, because this is exactly where an index transposition would silently produce keys where I meant values, or mix heads with layers. Take prefix_len = 3, n_layer = 2, n_head = 4, d_model = 8 (so head_dim = 2), batch = 2. The MLP outputs n_layer·2·d_model = 2·2·8 = 32 per row, so after running the 3 prefix rows through it I have [3, 32] — and 32 is precisely 2·n_layer·d_model per position, matching the width I counted earlier. Unsqueeze and expand to the batch: [2, 3, 32]. Reshape that 32 into [n_layer·2, n_head, head_dim] = [4, 4, 2], giving [2, 3, 4, 4, 2] = [batch, prefix_len, layer·kv, head, head_dim]. Now permute (2,0,3,1,4) → [4, 2, 4, 3, 2] = [layer·kv, batch, n_head, prefix_len, head_dim]. Indexing the leading dimension at 2l and 2l+1 then hands back two tensors each of shape [batch, n_head, prefix_len, head_dim] = [2, 4, 3, 2] — exactly the `past_key_values` shape the attention expects, and I get n_layer = 2 such (key, value) pairs. I ran this through and the printed shapes come out [2,4,3,2] for every key and value across both layers, with 2 pairs total. So the flatten→reshape→permute chain lands the right numbers in the right slots; no transposition bug.

The other place an off-by-one can hide is scoring. In `loss_fn` I form z = [x; y] as `input_ids` and feed the prefix through `past_key_values`, so the prefix occupies the cache but adds *no* positions to `input_ids` — z has length x_len + y_len, and logits has one row per z position. An autoregressive row at position t predicts token t+1, so to score the y tokens (which sit at positions x_len … x_len+y_len−1) I need the rows that *predict* those positions, i.e. rows x_len−1 … x_len+y_len−2. The slice `out.logits[:, x_len-1:-1, :]` selects rows x_len−1 up to the second-to-last. Let me check the endpoints concretely with x_len = 4, y_len = 3 (so z has length 7): the slice picks rows [3, 4, 5], which predict positions [4, 5, 6] — and y occupies exactly positions [4, 5, 6]. Three predicted rows for three y targets, perfectly aligned. If I'd written `[x_len:]` instead I'd have been off by one and scored y shifted left by a token, so the `−1` on each end is doing real work. Good.

How long should the prefix be? Longer prefix = more trainable parameters = more expressive, but also more room to overfit. The right length has to be a task hyperparameter: table-to-text can use a short prefix, while summarization plausibly needs much more steering capacity because the input is longer and the operation is less templatic. Importantly, a longer prefix barely costs anything at inference compared with adding new sequential layers, because attention over the whole prefix is parallelized on the GPU.

One more thing, initialization, which matters most when data are scarce. Random cache vectors are unlikely to sit in a region of activation space the LM already uses. Better: initialize the prefix from the activations of real words computed by the frozen LM itself. Run real tokens through the LM, grab their per-layer key/value activations, and use those to seed the prefix. This starts the prefix in a region of activation space the LM already "speaks," and it is concordant with the whole philosophy of disturbing the pre-trained model as little as possible. I'd expect this to help most in the low-data settings and to matter little once there's enough data to optimize away from any reasonable start; that's a guess I'd want to confirm on the subsampled splits.

Step back and notice what I've gained beyond parameter count. Because the entire task-specific object is a prefix that lives in the *input/activation* side and the backbone is untouched, different examples in one batch can carry *different* prefixes against the same frozen model — so I can batch requests for different tasks or different users together, which a per-task fine-tuned checkpoint can't do. That falls out for free from "the task is encoded as context, not as weights."

The implementation becomes the same frozen loss with only the prefix module exposed to the optimizer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixTuning(nn.Module):
    """Trainable per-layer key/value prefix that steers a frozen LM.
    Reparametrized through an MLP for stable optimization; collapse to the
    raw prefix cache after training."""
    def __init__(self, lm_config, prefix_len=10, reparam_dim=512, mlp_hidden_dim=512):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layer = lm_config.n_layer
        self.n_head = lm_config.n_head
        self.d_model = lm_config.n_embd
        self.head_dim = self.d_model // self.n_head
        # small matrix P'_theta: one k-dimensional vector per prefix position
        self.register_buffer("prefix_positions", torch.arange(prefix_len), persistent=False)
        self.prefix_basis = nn.Embedding(prefix_len, reparam_dim)
        # MLP expands k -> the flattened per-layer key/value cache
        self.mlp = nn.Sequential(
            nn.Linear(reparam_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, self.n_layer * 2 * self.d_model),  # 2 = key + value
        )

    def materialize(self, batch_size, device):
        """Produce past_key_values: per layer, (key, value) of shape
        [batch, n_head, prefix_len, head_dim]."""
        idx = self.prefix_positions.to(device)
        h = self.mlp(self.prefix_basis(idx))                       # [prefix_len, n_layer*2*d_model]
        h = h.unsqueeze(0).expand(batch_size, -1, -1)              # same task prefix for this batch
        h = h.reshape(batch_size, self.prefix_len, self.n_layer * 2, self.n_head, self.head_dim)
        h = h.permute(2, 0, 3, 1, 4)                               # [n_layer*2, batch, n_head, len, hd]
        past = []
        for l in range(self.n_layer):
            past.append((h[2 * l], h[2 * l + 1]))                  # (key, value) for layer l
        return tuple(past)   # prefix occupies the left context at EVERY layer


def loss_fn(frozen_lm, prefix, x_ids, y_ids):
    z = torch.cat([x_ids, y_ids], dim=1)                          # [P ; x ; y] via past_key_values
    past = prefix.materialize(z.size(0), z.device)
    out = frozen_lm(input_ids=z, past_key_values=past)            # frozen LM attends to the prefix
    logits = out.logits[:, x_ids.size(1) - 1:-1, :]               # predict the y positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))


def train(frozen_lm, prefix, loader, opt):
    for p in frozen_lm.parameters():
        p.requires_grad = False                                   # backbone frozen
    optim = torch.optim.AdamW(prefix.parameters(), lr=opt.lr)     # only prefix-side params train
    for batch in loader:                                          # default: 10 epochs, bs 5, lr 5e-5
        loss = loss_fn(frozen_lm, prefix, batch["x_ids"], batch["y_ids"])
        loss.backward(); optim.step(); optim.zero_grad()


# After training: store the materialized per-layer key/value tensors; drop the MLP/P'_theta.
```

The causal chain: serving many tasks cheaply forces a tiny per-task object on a frozen backbone; "steer-by-context" suggests a learned context, but discrete prompts are hard to optimize and capped to real-word embeddings, so relax to continuous virtual-token vectors (which contain the discrete prompts as a special case); those at the input layer alone are a worry because the dependency path through the frozen stack is long, so make the prefix cache trainable at every layer by supplying per-layer key/value tensors through the existing cached-attention path; prefixing dominates infixing because under causal attention its reach is a superset — it can shape both the encoding of x and the generation of y; raw optimization of that cache is unstable, so reparameterize through a shared MLP from a smaller matrix and discard it after training; prefix length trades expressiveness against overfitting; and real-word activation initialization keeps the learned cache near the LM's own activation space — yielding a method that stores only a small prefix per task, leaves the LM untouched, and lets one batch mix prefixes across tasks.
