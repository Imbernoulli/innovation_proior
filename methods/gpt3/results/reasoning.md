Let me start from what actually hurts. The recipe everyone uses now is: pre-train a big Transformer language model on a pile of text, then fine-tune it on a labeled dataset for whatever task I care about. The architecture is shared across tasks — that part is solved, one body, swap the head. But the adaptation isn't. To get a model to do sentiment, or entailment, or question answering, I still have to collect thousands to hundreds of thousands of labeled examples for that one task and run gradient descent on them. And I have to do that again for the next task, and the next.

Three things gnaw at me about this. First, the number of useful language tasks is basically unbounded — correct this person's grammar, give me an example of "bravery," critique this short story, summarize this in the voice of a pirate. I cannot build a labeled training set for each of those. Second, and this is subtler, when I take a model with enormous capacity and fine-tune it on a *narrow* task distribution, it has every incentive to grab whatever spurious correlation happens to separate the labels in that narrow set. It can score at "human level" on the benchmark and still be brittle the moment the input drifts off-distribution — the very expressiveness that made pre-training work is now overfitting to artifacts of one small dataset. Third, this is just not how people work. A person reads "tell me if this sentence is happy or sad" and does it. Or sees two examples of someone being brave and produces a third. No gradient descent on ten thousand labeled pairs. They can even switch tasks mid-sentence — do a little arithmetic in the middle of a conversation. I'd like a system with that fluidity.

So the goal sharpens: I want one fixed model that, at inference time, takes a task described in plain text — an instruction, or a few demonstrations, or both — and just does the task. No weight updates per task. The question is whether that's even possible, and if so, how.

There's a thread to pull on here. A plain autoregressive language model, the kind I train just to predict the next token, can already be *prompted* to do things. If I write "Passage. TL;DR:" and let it continue, it summarizes. If I write a few translation pairs and then an English sentence, it tends to produce the French. The input text is the task specification — I'm not changing the model at all, I'm just conditioning it. This has been tried; the trouble is the numbers. Prompting a language model zero-shot to answer open-domain questions gets you a few percent; on a conversational QA benchmark it lands tens of points behind a fine-tuned model. As a serious method this looks dead. The reflex is to conclude prompting is a toy and go back to fine-tuning.

But before I throw it out — *why* is it so bad? Is the mechanism wrong, or is the model just too weak? Those are very different diagnoses. If the mechanism is wrong, no amount of anything fixes it. If the model is just too small to carry the competence that prompting tries to draw out, then the fix isn't a new method, it's a bigger model. I need to figure out which.

The scaling regularity points toward the latter. The test cross-entropy loss of an autoregressive Transformer doesn't improve raggedly with size — it follows a clean power law. Across many orders of magnitude, the loss as a function of non-embedding parameter count N behaves like L(N) ≈ (N_c/N)^α_N with a small exponent, around α_N ≈ 0.076 and N_c ≈ 8.8×10¹³; as a function of dataset size D like (D_c/D)^α_D with α_D ≈ 0.095 and D_c ≈ 5.4×10¹³ tokens; and as a function of compute in the compute-optimal allocation like L(C_min) ≈ (C_c^min/C_min)^α_C^min with α_C^min ≈ 0.050 and C_c^min ≈ 3.1×10⁸ PF-days. Tiny exponents — you need to scale a lot to move the loss a little — but the point is the *smoothness*. It's a straight line on a log-log plot over seven orders of magnitude. And loss is barely sensitive to the *shape* of the network — depth versus width, how many heads, the feed-forward multiplier — it's the total scale N that drives it. So loss is predictable, and it's almost entirely a function of how big the model is.

Loss is the average negative log-probability the model assigns to the actual next token over held-out text. What does it take to make that number small? The model has to correctly anticipate continuations of arbitrary web text. And arbitrary web text is not a homogeneous blob — it's shot through with little tasks. A page lists English terms and their French equivalents. A forum thread has questions followed by answers. A glossary has "word: definition." A worksheet has "12 + 35 = 47." A document repeats a sub-pattern several times. To predict the *next* token in the middle of such a passage with low loss, the model has no choice but to figure out, from the preceding tokens alone, what regular relationship is governing this stretch of text, and then apply it to continue. Inferring "what's the rule here, given the examples so far" and applying it — that is exactly learning from a few examples. It's happening *inside the forward pass*, with no weight update, purely as a byproduct of being good at next-token prediction.

So next-token prediction over a diverse corpus isn't just teaching the model facts and grammar; it's implicitly forcing it to become an in-context learner, because so many sub-sequences in the corpus *are* "here are some examples of a pattern, now continue it." And if that's true, then the quality of in-context learning is just another capability that should ride the same scaling curve as the loss — it was weak before because the model was small, not because the idea is broken.

There are two nested loops here. There's a slow outer loop: stochastic gradient descent over the whole corpus, which updates the weights and, over the course of training, installs a broad repertoire of skills and pattern-recognition abilities. And there's a fast inner loop: a single forward pass over one sequence, which updates *nothing* — no weights move — but which, reading the prefix, recognizes the task implied by that prefix and adapts its predictions accordingly, using only the activations flowing across positions. This is the inner-loop/outer-loop signature of meta-learning. The unusual part is where the inner loop lives: not in a few steps of gradient descent on a support set, but in the network's own activations as it consumes the context.

That's not unprecedented when I think about it. A recurrent network can implement an entire learning algorithm in its hidden-state dynamics — the weights, trained across many tasks, encode a procedure that the activations execute on a new task at run time. The reinforcement-learning version makes it vivid: train an agent across a distribution of environments, and it learns to adapt to a fresh environment purely through its recurrent activations over a trajectory, weights frozen at test time; the outer loop (training across environments) taught it *how to learn*, and the inner loop (one episode's activations) does the learning. The autoregressive language model is the same shape, with the corpus playing the role of the task distribution. Predicting text, averaged over a corpus diverse enough to contain a vast implicit distribution of tasks, *is* a meta-learning objective. Nobody has to assemble the task distribution by hand — it's already latent in "predict the next word."

This reframes the whole problem. I do not need a new training objective. I do not need a new architecture for adaptation. The few-shot setting is not a mechanism at all — it's a way of *evaluating* a model I already know how to train. A few-shot prompt is K pairs of (context, completion), followed by a final context, fed to the same next-token model; the model's job is to produce the completion. It's just p(completion | context, demonstrations) — the demonstrations sharpen the model's estimate of which task it's in, and the completion drops out of the same conditional distribution it was always modeling. Zero-shot, one-shot, few-shot are points on a single axis — how much task-specific text I put in the context — running parallel to fine-tuning, except with zero gradient updates at inference. Fine-tuning sits at one extreme (a whole dataset, plus weight updates); few-shot puts a handful of examples in the context with no updates; one-shot, one example; zero-shot, just an instruction. They're not competitors, they're a trade-off between how much task data you spend and how cleanly you're comparing to a human.

The diagnostic I need is direct. Train a family of models with the same objective and the same evaluation interface, then compare zero demonstrations against a few demonstrations as model size changes. If the few-demonstration curve rises faster than the zero-demonstration curve, that would be exactly the signature of a better inner loop: a small model barely uses the examples in its context, while a large one extracts the task from them. So the way to test the hypothesis is not to invent anything — it's to take the existing language-modeling recipe and scale it hard, training a family of sizes, and watch whether few-shot in-context learning crosses from "toy" into "useful."

So the plan is settled in outline: keep the autoregressive objective, keep the decoder-only Transformer, and push the scale by about two orders of magnitude over the largest model anyone has trained this way, while also training a ladder of smaller models so I can *see* the scaling curve of in-context learning, not just one point. Now the engineering: how big, shaped how, trained on what, and how exactly do I run the evaluation. Every one of these has to be pinned down, and I want a reason for each, not a guess.

Start with the architecture, because here the right move is restraint. The scaling measurements told me loss is dominated by total scale N and is nearly indifferent to architectural shape over a broad range. So I should not be clever. I take the proven decoder-only Transformer language model and scale it, changing as little as possible — partly to keep the in-context-learning story clean (if I bolt on a fancier objective and it works, I won't know whether scale or the gadget did it), and partly because at the sizes I'm targeting, the implementation has to actually run on a cluster, and complexity is the enemy.

But "scale a Transformer to dozens of billions of parameters and 96 layers" surfaces problems that don't bite at small scale, and I have to handle each. The first is depth stability. The original Transformer puts LayerNorm *after* each sublayer, on the residual sum. Stack that very deep and the gradients get unstable — the normalization sits on the main path and the signal has to fight through it at every layer. The fix is to move the normalization to *before* each sublayer, so each block computes x + Sublayer(LN(x)). Now the residual stream is a clean additive identity path from input to output, and the LayerNorm only conditions the input to each sublayer; gradients flow straight down the residuals. Add one final LayerNorm at the very top to normalize before the output projection. With pre-normalization I can train 96 layers without the training dynamics falling apart.

The second problem is variance growth down the residual stream, and it's worth doing the arithmetic. Each block adds its sublayer's output back into the stream: x ← x + δ. If I initialize every projection the standard way, each δ has roughly unit-scale variance, and the additions accumulate: after L blocks the residual stream's variance has grown roughly in proportion to the number of additions. With each block contributing two additions (one from attention, one from the MLP), that's about 2L additions, and at L on the order of a hundred the activations at the top are enormous compared to the bottom — which wrecks the conditioning and the stability of training. I want to keep the residual stream's scale roughly constant with depth. So I down-scale the weights of the projection that writes back into the residual stream: initialize them with standard deviation scaled by 1/√(number of residual additions) ≈ 1/√(2L). Then each δ has variance ∝ 1/(2L), the 2L of them sum back to order one, and the stream stays well-behaved no matter how deep I go. Concretely, the residual-path output projections get init std = 0.02/√(2·n_layers), while the other weights keep std ≈ 0.02. This is the GPT-2 initialization, and the reason is exactly this variance accounting.

The input channel has to tolerate whatever a task can be written as: arithmetic with arbitrary digit strings, words with letters scrambled, freshly invented nonsense words, code, multiple languages. A word-level vocabulary would choke on some of those with out-of-vocabulary tokens, and a language-specific tokenizer would smuggle in assumptions about what the task input is allowed to look like. A byte-level scheme can represent *any* string, needs no language-specific preprocessing, and is reversible so I can decode exactly what the model emits. That pushes me to byte-level byte-pair encoding — BPE built over bytes — with a vocabulary around fifty thousand merges.

A few shape parameters I won't overthink, precisely because the scaling work says loss doesn't care much. The feed-forward width inside each block I set to four times the model width — the standard ratio; there's no measured reason to deviate. The attention is multi-head with the head dimension fixed (128 in the large models), and the number of heads chosen so heads × head-dim equals the model width; among the near-equivalent options I pick whatever maps most cleanly onto how the model is sharded across GPUs, since loss is insensitive within a broad band and load-balancing the parallel layout is the real constraint.

One parameter I *do* care about: the context window. GPT-2-scale models used a thousand-ish tokens. But the context window is now load-bearing in a new way — it's literally the budget for how many demonstrations I can put in a few-shot prompt. If I want to fit ten to a hundred (context, completion) pairs before the final query, I need room. So I double the window to 2048 tokens. The number of in-context examples K I can use is bounded by exactly this — K runs from zero up to however many demonstrations fit in 2048 tokens, which for typical tasks is tens.

A 2048-token window makes the quadratic attention cost visible: every position attends to every earlier position, so the deepest models spend real compute just moving information across a long prefix. Full dense attention everywhere is wasteful when much of the useful mixing is local, but making every layer local would cut off global communication. The compromise is to alternate: some layers use full dense attention for global mixing, and the others use a locally-banded sparse pattern where each position attends within a band, in the style of the Sparse Transformer. That cuts the attention cost substantially while preserving enough long-range mixing across the dense layers. It's an efficiency change, not a capability change — in fact attention is a small enough fraction of total compute that I'll ignore it entirely when I do the compute accounting. For the compact code below I write the dense causal-attention form; the alternating sparse layers are an efficiency variant on the same block.

Now sizes. The whole point is to *trace the curve*, so I train a ladder — eight models spanning three orders of magnitude, from around 125 million parameters up to 175 billion. The smallest matches a GPT-2-scale config (12 layers, width 768, 12 heads); each step up grows depth and width together (24 layers and wider, then 32, then 40, then 96 layers at width 12288 for the largest). Eight points let me check whether in-context learning scales smoothly the way loss does, and whether the zero/one/few-shot gap genuinely widens with size, rather than betting everything on a single giant model.

The training hyperparameters have to scale *with* the model, and there's a clean reason for the pattern. The critical batch size is set by the gradient noise scale: below it, averaging more examples buys a real reduction in stochastic noise; above it, extra examples mostly waste compute. Larger models and later stages of training tend to have a larger critical batch, so bigger models can use bigger batches before those diminishing returns kick in. I ramp the batch from half a million tokens at the small end up to over three million tokens at the 175B model, and I actually *measure* the gradient noise scale during training to set it rather than guessing. The flip side is the learning rate: a bigger model is more delicate, so the peak learning rate comes *down* as size goes up — from about 6×10⁻⁴ at the smallest to about 0.6×10⁻⁴ at the largest. Bigger batch, smaller learning rate, both monotone in size.

The optimizer is Adam, but with a tweak: β₁ = 0.9 as usual, and β₂ = 0.95 rather than the customary 0.999. The second-moment average with 0.999 has a very long memory; at this scale and batch size a shorter memory (0.95) tracks the curvature more responsively and trains more stably. ε = 10⁻⁸. I clip the global gradient norm at 1.0 — a single occasional bad batch at this scale can blow things up, and the clip caps the damage. And I add a small decoupled weight decay of 0.1 for mild regularization — decoupled meaning it's applied as a direct pull toward zero on the weights, not folded into the gradient through Adam's denominator, and I apply it only to the matrix weights, not to biases, LayerNorm gains, or embeddings, since decaying those just hurts.

The learning-rate schedule needs a warmup, and here's why it isn't optional with Adam. Early in training Adam's second-moment estimate v is built from very few gradients and is unreliable; the update divides by √v, so a bad early estimate produces wildly mis-scaled steps right when the model is most fragile. So I warm the learning rate up linearly over the first chunk of training (the first few hundred million tokens) to let v stabilize, then cosine-decay it down to a tenth of its peak over the bulk of training (a couple hundred billion tokens), and hold it flat at that tenth afterward. I also warm up the *batch size* — start small (tens of thousands of tokens) and ramp to full over the first several billion tokens — for a parallel reason: early on the gradient noise scale is small, so a small batch is the compute-efficient choice; grow it as the optimal batch grows.

Data is the other half, and the scale here changes what "good data" means. I need on the order of hundreds of billions of tokens, which points at Common Crawl — about a trillion words, big enough to train the largest model without ever revisiting a sequence. But raw Common Crawl is low quality, and a high-capacity model will faithfully learn the junk. So I clean it. To filter, I train a simple classifier — logistic regression on hashed-token features — to tell curated text (an existing high-quality web corpus, Wikipedia, books) from raw Common Crawl, and I use it to score every Common Crawl document. The natural move would be a hard threshold on the score, but I don't want to lose all diversity by keeping only the most "Wikipedia-like" pages. So I keep a document if np.random.pareto(α) > 1 − score: high-scoring documents almost always pass, low-scoring ones usually don't, but a heavy-tailed Pareto draw lets some lower-scoring, out-of-distribution documents through, preserving variety. I set α = 9 so the kept-document score distribution matches that of the curated reference corpus.

Then deduplication, because memorization is the enemy at this capacity. I fuzzily deduplicate with MinHash locality-sensitive hashing — documents that overlap heavily get collapsed — both within and across datasets, and I also scrub the curated web corpus out of Common Crawl so it isn't double-counted. Duplicates do two bad things: they secretly raise the number of times the model sees the same content (inflating effective epochs on whatever happens to be duplicated), and they leak between training and the held-out validation set, which would make my overfitting signal a lie. Dedup costs about ten percent of the data and is worth it.

I don't sample the datasets in proportion to their size. Common Crawl is by far the largest but the lowest quality; the curated corpora are small but excellent. So I up-weight quality: over a training run of a few hundred billion tokens, Common Crawl gets seen less than once through, while the small high-quality sets get seen two or three times. That deliberately accepts a touch of overfitting on the good data in exchange for a higher average quality of every token the model trains on. For efficiency I always train on full 2048-token sequences, packing several short documents into one sequence separated by a special end-of-text token — no padding waste — and I don't bother with cross-document attention masking, because the end-of-text token itself tells the model that text on either side of it is unrelated, so it learns to not bleed context across the boundary.

How long to train, and how big given a compute budget — this is where the scaling work pays off in a non-obvious way. The forward pass costs about 2 flops per active parameter per token, one multiply and one add. The backward pass needs work of the same order for gradients with respect to parameters and activations, so forward plus backward is about 3 times the forward cost, or 6 flops per parameter per token. Total training compute is about 6·N·D for N parameters and D tokens — and at 174.6 billion parameters over 300 billion tokens that's 3.14×10²³ flops. The non-obvious part is the *allocation*. The compute-optimal prescription from the scaling measurements is to put a fixed budget mostly into model size and *not* train to convergence — the optimal model size grows with compute roughly as C^0.73, so as I get more compute I should buy a much bigger model and feed it comparatively few tokens, stopping well before the loss plateaus. That's why I train these enormous models on "only" a few hundred billion tokens rather than grinding each to convergence: a model ten times larger than a heavily-trained smaller one can use the *same* compute because it sees far fewer tokens, and it ends up with lower loss. Counterintuitive, but it's what the power laws say, and it's the whole reason the 175B model is feasible at all.

Now the evaluation interface — the slots I left empty: how do I turn an arbitrary task into model input, and how do I read an answer back out, without ever training on the task. Constructing the prompt is straightforward once I commit to "text is the task spec": for a K-shot evaluation I draw K examples from the task's training set, write each as its context followed by its completion, concatenate them (delimited by newlines), and append the final query's context with the completion left blank for the model to fill. Optionally I prepend a natural-language instruction — and for zero-shot, the instruction *is* the whole prompt, since there are no demonstrations. K is whatever fits in 2048 tokens; more is usually better, so I tune K on a dev set and report the best on test.

Reading the answer out splits by task type, and each split has a reason. For multiple-choice tasks I don't ask the model to generate — I score. I build the prompt as the K demonstrations plus the query context, then for each candidate completion I compute the model's likelihood of that completion continuing the prompt, and pick the highest. The subtlety is normalization. Raw joint likelihood favors *shorter* completions, since each extra token multiplies in another probability less than one, so I normalize by length — compare per-token likelihood. For a few datasets even that biases toward completions that are intrinsically common strings regardless of the question; there I divide by the completion's *unconditional* likelihood given a generic neutral lead-in like "Answer:", i.e. score P(completion | context) / P(completion | "Answer:"), which cancels the part of the completion's probability that has nothing to do with the actual question. Binary classification I just turn into multiple choice by giving the two classes meaningful word names ("True"/"False" instead of 0/1) and comparing their likelihoods, since the model has priors over real words but not over bare integers. Free-form generation tasks I actually decode, with beam search (width 4, length penalty 0.6 to stop it preferring trivially short outputs), and score with whatever the dataset uses — F1, exact match, or BLEU.

One methodological worry I have to take seriously precisely *because* I'm training on a giant web scrape: contamination. Many benchmark test sets are themselves on the web, so Common Crawl may contain them, and a high-capacity model could simply have memorized the answers, which would make few-shot "learning" a mirage on those tasks. So I have to build tooling to measure train/test overlap, search for benchmark text in the training data, and either drop the affected datasets or flag them — otherwise I can't trust my own numbers. This isn't a feature of the method; it's the price of training at this scale on uncurated text, and I have to pay it to believe the results.

I should also be honest about where this design is structurally limited, since I'm choosing it deliberately. By committing to a pure autoregressive, left-to-right model I'm giving up bidirectionality. Tasks that hinge on comparing or re-reading two spans — does this sentence imply that one, is this word used the same way in both sentences, fill in a blank with full surrounding context — are exactly where bidirectional/denoising models have an edge, and I should expect my few-shot performance to lag there. I'm accepting that cost because the autoregressive model is the one I can both sample from and score likelihoods with cleanly, which is what the whole in-context evaluation needs. And at a more fundamental level, the objective weights every token equally — it has no notion of which predictions matter — so pure next-token prediction will eventually hit a ceiling regardless of scale; but testing how far this particular lever goes is the point of the exercise.

The code should keep the separation clean: the model is a decoder-only pre-LN Transformer with the scaled residual init, the byte-level vocabulary, the 2048 context, and (in the efficient variant) alternating dense/sparse attention; the in-context-learning layer is only a prompt builder plus likelihood/generation scorer around the fixed model.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- the architecture: decoder-only, pre-LN Transformer LM (dense reference form) ----

class CausalSelfAttention(nn.Module):
    # multi-head masked self-attention. In the large models, half the layers
    # instead use a locally-banded sparse mask for O(n*band) cost; the dense
    # form is the heart of it and is what I write here.
    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)   # q,k,v in one matmul
        self.c_proj = nn.Linear(d_model, d_model)       # writes back into the residual stream
        self.n_head = n_head
        # causal mask: position t may attend only to <= t (left-to-right)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))   # 1/sqrt(d_head) keeps logits in range
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # enforce causality
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    # pre-normalization: LN before each sublayer, clean additive residual path
    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_ctx)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(d_model, 4 * d_model),       # FFN width = 4 * d_model
            c_proj=nn.Linear(4 * d_model, d_model),     # also writes back into the residual stream
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))                 # residual add #1
        m = self.mlp
        x = x + m.c_proj(F.gelu(m.c_fc(self.ln_2(x))))  # residual add #2
        return x

class SequenceModel(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_layer, d_model, n_head):
        super().__init__()
        self.n_ctx = n_ctx
        self.wte = nn.Embedding(vocab_size, d_model)    # byte-level BPE token embeddings
        self.wpe = nn.Embedding(n_ctx, d_model)         # learned positional embeddings
        self.h = nn.ModuleList([Block(d_model, n_head, n_ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)               # final pre-output norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight            # GPT-style tied token/output embeddings
        # base init
        self.apply(self._init)
        # scale residual-path projections by 1/sqrt(2*n_layer): ~2*n_layer residual
        # adds accumulate variance, so shrink each write-back to keep the stream ~unit-scale
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
            # the entire training signal: next-token cross-entropy, every token weighted equally
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.n_ctx else idx[:, -self.n_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                nxt = torch.multinomial(probs, 1)
            else:
                _, nxt = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# ---- training: scale batch up / LR down with model size; warmup; cosine decay ----

def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    # decay matrix weights only; never decay biases, LayerNorm gains, or embeddings
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.endswith("weight") and ("wte" not in name and "wpe" not in name and "ln" not in name):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)  # beta2=0.95: shorter 2nd-moment memory

# ---- the in-context-learning harness: fixed-model prompt + scorer ----

def build_prompt(tokenizer, instruction, demonstrations, query_context):
    # K-shot: optional instruction, then K (context -> completion) demos, then the query
    parts = []
    if instruction:
        parts.append(instruction)
    for ctx, completion in demonstrations:          # K of these; K bounded by n_ctx
        parts.append(ctx + " " + completion)
    parts.append(query_context)                     # completion left blank for the model
    return tokenizer.encode("\n".join(parts))

@torch.no_grad()
def _completion_logprob(model, prefix_ids, completion_ids):
    # logits at position j predict token j+1, so score completion tokens using full[:-1]
    if not prefix_ids:
        raise ValueError("completion likelihood needs at least one prefix token")
    if not completion_ids:
        device = next(model.parameters()).device
        return torch.tensor(float("-inf"), device=device)
    device = next(model.parameters()).device
    full = prefix_ids + completion_ids
    idx = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    logits, _ = model(idx)
    logp = F.log_softmax(logits[0], dim=-1)
    start = len(prefix_ids) - 1
    token_logps = [logp[start + i, tok] for i, tok in enumerate(completion_ids)]
    return torch.stack(token_logps).mean()

@torch.no_grad()
def score_choice(model, tokenizer, prompt_ids, completion, answer_context=None):
    # multiple choice: rank candidates by the model's likelihood of the completion,
    # length-normalized; optionally divide out the unconditional likelihood.
    comp = tokenizer.encode(completion)
    cond = _completion_logprob(model, prompt_ids, comp)
    if answer_context is None:
        return cond                                 # per-token conditional log-likelihood
    base_ids = tokenizer.encode(answer_context)     # e.g. "Answer:" — generic lead-in
    uncond = _completion_logprob(model, base_ids, comp)
    return cond - uncond                            # divide out completion's generic likelihood

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

I end up with a simple chain: fine-tuning needs per-task data and updates and overfits narrow distributions; prompting a language model needs neither but was too weak — and the diagnosis is that it may be too small, not wrong, because loss scales smoothly with size and in-context learning is one of the capabilities packed in by next-token prediction over a corpus riddled with implicit demonstrations; that makes pre-training an outer loop and the forward pass an inner loop, i.e. meta-learning, so the few-shot setting is an *evaluation* of the existing model, not a new mechanism. The experiment is to scale the unchanged autoregressive Transformer recipe hard — pre-norm and scaled init for depth, doubled context for the demonstrations, alternating sparse attention for efficiency, batch up / LR down with size, compute-optimal under-training of very large models per the power laws, quality-filtered deduplicated web data — and wrap it in a prompt-builder-plus-likelihood/generation harness, then read off whether few-shot in-context learning crosses from toy into useful as the model grows.
