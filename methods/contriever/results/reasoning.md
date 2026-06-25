OK, let me think about this from the pain, not the architecture. I want a retriever that, given a query, pulls the relevant documents out of a collection of millions. The thing I keep running into is that the system everyone actually deploys — BM25 — needs no training data at all, and it is *good*. It scores a document by how many query terms appear in it, weighted by how rare each term is (inverse document frequency), with some length normalization and term-frequency saturation. No labels, robust, fast. And whenever I train a fancy neural retriever on a big labeled set and then point it at a new domain it has never seen, it loses to BM25. That is the humbling fact I have to take seriously.

Why does BM25 lose anything at all, then? Because it matches surface tokens. "Car" and "automobile" share no characters; an Arabic query and the English document that answers it share nothing; a question phrased differently from the passage that answers it can score near zero. That is the lexical gap. A dense model that embeds query and document into a shared vector space and scores by dot product closes that gap — semantic matches light up even with zero shared tokens. So the dense approach is clearly the right *representation*. The problem is purely how to *learn* that space without labels.

And I do need labels in the standard recipe. DPR's whole training signal is human-matched (question, gold-passage) pairs — pull the gold passage's embedding toward the question, push everything else away. Those pairs exist for a few English benchmarks and basically nowhere else. To make them for a new collection I'd have to hand-match queries against millions of documents. For non-English languages there's essentially nothing. So "just label more data" is a dead end for the thing I actually want, which is a retriever that works *anywhere*, including languages with no retrieval supervision. I need a training signal that comes from raw text alone.

What signal lives in raw text? Here's the only thing I'm sure of: every document is, in some way, unique. If I have no labels, the one piece of structure I can lean on is *identity* — this chunk of text came from that document and not from any other. So suppose I could manufacture, from a single document, two pieces that I'm willing to *declare* a matched pair — a synthetic "query" and a synthetic "relevant document" — and then train the encoder so that this pair scores higher than the same query paired with pieces of *other* documents. That's discrimination by identity, and it needs no annotation. The encoder is forced to map the two pieces of the same source near each other and far from everything else, which is exactly the geometry a retriever needs.

Is there prior art shaped like this? Yes — ICT, the inverse Cloze task. Take a passage, pull out one sentence as the "query," use the surrounding sentences as the "document," and train the model to retrieve that context from that sentence. It helped as pre-training, but as a zero-shot retriever it still sat below BM25. So the *shape* is right and the *result* is not good enough yet. Why might that be? I'll come back to it — first let me notice that this shape is precisely instance discrimination from vision: two "views" of one instance form a positive, every other instance is a negative, and you learn by picking the positive out of the crowd. Vision has pushed that idea hard in the last couple of years. Maybe ICT underperformed not because the idea is wrong but because it wasn't run with the modern contrastive machinery. Let me try to build the strongest possible version of "discriminate one document against many" and see how far it goes.

So the objective. I have a query representation q and its positive key k₊, and a pool of negative keys k₁…k_K. I want q·k₊ large and q·k_i small. The natural loss is the softmax that has to assign all its mass to the positive — InfoNCE. Score by dot product s(q,k) = ⟨q,k⟩, divide by a temperature τ, softmax over {positive, negatives}, and take negative log-likelihood of the positive:

  L = −log [ exp(s(q,k₊)/τ) / ( exp(s(q,k₊)/τ) + Σ_i exp(s(q,k_i)/τ) ) ].

Before I trust this I should actually differentiate it and see which way the scores move, because it's easy to get a sign backwards and never notice. Take one example with τ = 0.05 and four scores — the positive at 0.8 and three negatives at 0.6, 0.7, 0.2 — and read off ∂L/∂s. The positive's derivative comes out to −2.66 and the three negatives' come out to +0.32, +2.35, +0.0001. A gradient-descent step moves each score by −(learning rate)·(derivative), so the positive score rises and the negative scores fall. That's the geometry I want, and it's not just asserted now: I watched the sign flip the right way. The other thing that falls out of those numbers is telling. The negative at 0.7 — the one whose score is closest to the positive's 0.8 — got a gradient of +2.35, roughly seven times the negative at 0.6 and thousands of times the one at 0.2. So at τ = 0.05 the loss spends almost all its force on the single hardest negative and basically ignores the easy ones. That is exactly the τ behavior I'd want for retrieval, where I need the positive to beat the near-miss documents, not the obviously-irrelevant ones; a larger τ would flatten that concentration. I'll fix τ around 0.05 and keep it as a knob.

Now, two design questions are doing all the work here, and ICT's mediocre zero-shot result must trace to one of them: (1) how do I build the positive pair from a single document, and (2) how do I get enough negatives. Let me take them in turn.

Negatives first, because contrastive learning lives or dies on having many. With InfoNCE, the positive has to beat *every* negative in the pool; the more negatives, the harder and more informative each step, and the closer the softmax approximates "find the one right document among the whole collection," which is literally the retrieval task. The simplest source of negatives is the rest of the batch: encode 2N views for N documents, and for each query use the other documents' keys as negatives — in-batch negatives, backprop through both sides. That's SimCLR's setup. The catch is that the number of negatives equals the batch size, so to get thousands of negatives I need batches of thousands, and the memory blows up because I'm backpropagating through every key. I keep hitting that wall: I want, say, 100k negatives, and I cannot fit a 100k batch.

So decouple the number of negatives from the batch size. Keep a *queue*: a rolling buffer of key vectors from recent batches, and use the whole buffer as negatives for the current query. Now negatives ≫ batch size, cheaply, because the queued vectors are just stored embeddings — I don't backprop into them. But there's a problem the moment I write that down: the vectors in the queue were produced by *older* versions of the encoder. The encoder is moving every step; a key I computed 500 steps ago was made by a noticeably different network, so comparing today's query to a stale key is comparing against a representation from a different space. When the encoder changes fast, those stale negatives are inconsistent and training degrades. That's the wall with a naive queue.

The fix is to slow the keys down so the queue stays self-consistent. Use a *second* encoder for the keys — the momentum encoder — whose weights are not trained by gradient but are an exponential moving average of the trainable query encoder:

  θ_k ← m·θ_k + (1−m)·θ_q,

with m close to 1. Now the key encoder drifts slowly and smoothly, so a key computed many steps ago is still in nearly the same space as a key computed now; the whole queue is mutually consistent. This is MoCo. It buys me a huge negative pool (queue size in the tens of thousands to ~131k) at small batch size, with keys that don't lurch around. The price: the loss becomes asymmetric — gradients flow only through the query side; keys are detached. I'll accept that; the consistency is worth more than the symmetric gradient.

What value of m, though? "Close to 1" is vague, and the queue is only consistent if a key still living in it was made by roughly today's network. The EMA retains a fraction m of its old weights each step, so a perturbation to the query encoder is only m^H absorbed into the key encoder H steps later; the half-life — steps for the key encoder to move halfway toward current query weights — is ln(0.5)/ln(m). Plug in m = 0.9995 and that's about 1386 steps; a key encoded 500 steps ago has caught up only ~22% of the way, 100 steps ago only ~5%. So with a queue in the tens of thousands and a batch in the low thousands, the oldest keys in the queue are at most a handful of half-lives old and the encoder that made them is genuinely close to the current one. Contrast m = 0.99: half-life ~69 steps, so a key a few hundred steps old was made by a substantially different network — that's the inconsistency I was trying to escape. And pushing m the other way, too close to 1, the half-life runs to many thousands of steps and the key encoder barely tracks the query encoder at all, so the keys stop reflecting what the model has learned. m = 0.9995 sits in the band where the queue is self-consistent but the keys still follow — and now I can see *why* that band, not just assert it. Bigger queues should help up to a point — more negatives, harder discrimination — so I'll treat queue size as something to scale.

Now the harder question: how do I make the positive pair from one document with no labels? This is where ICT's choice deserves a second look, because I suspect it's why ICT lagged. ICT takes a sentence span as the query and its *complement* — everything except that span — as the key. Two issues jump out. First, the two views are mutually exclusive by construction: the query tokens are deliberately removed from the key. So there is *no lexical overlap* between query and key. But part of what makes BM25 so strong is exactly lexical overlap — when the query and the relevant document literally share rare terms, that's a powerful, reliable signal. If I train my positives to *never* share tokens, I'm teaching the model to ignore the one signal that lexical retrieval nails. Second, the query (a single sentence) and the key (a long context) follow completely different length and content distributions, so the task is asymmetric in a way real retrieval isn't necessarily, and the model can exploit that asymmetry instead of learning a clean similarity.

So what's the alternative? The vision recipe makes two *independent* views by random cropping. In text, a "crop" is just a randomly sampled contiguous span of tokens — sample two spans independently from the same document and call that the positive pair. Two things change relative to ICT. The two views are now *symmetric* — both are spans drawn from the same distribution, so query-side and key-side look alike, which matches a setting where one shared encoder embeds both. And because the spans are sampled independently rather than carved to be disjoint, they *can* overlap and share tokens, instead of being forbidden to.

But "can overlap" is doing a lot of work in that argument, and I don't actually know how often it happens — if independent crops almost never touch, then I've gained symmetry but still thrown away the lexical signal, and the whole case for cropping over ICT collapses. So let me just compute it. Take a 256-token document, draw each crop's length as a fraction uniform in [0.05, 0.50] of the document and place it at a uniform start; two intervals overlap iff max(start) < min(end). Simulating a few hundred thousand pairs, two independent crops share at least one token about **60%** of the time (the mean crop is ~70 tokens, ~27% of the document). That number is the actual justification: in the majority of positive pairs the model sees literal token overlap and is rewarded for noticing exact matches — BM25's strength — while the remaining ~40% have no overlap and force it to learn semantic matching from context alone. ICT's complement construction drives that overlap fraction to exactly 0 by deleting the query tokens from the key, so it never gets the lexical half. That asymmetry in the overlap statistic — 60% versus 0% — is the concrete reason I'd expect cropping to make a better retriever than ICT, and it's checkable rather than just plausible.

Let me nail the cropping down concretely. Take documents of a fixed working length — say 256 tokens. For each, sample two spans; let the span length be a fraction of the document, sampled in some range (something like 5% to 50% of the length) so I get a mix of short and long views and a mix of high- and low-overlap pairs. After cropping, delete each token with probability 10%. That gives the two views a little extra noise without changing the basic same-document signal.

Now the encoder itself. I want one vector per text so I can pre-compute the document index and search it with approximate nearest neighbors (FAISS) — that's non-negotiable for scaling to millions of documents, and it's why a cross-encoder is off the table: a cross-encoder would have to re-encode the query with every document, which can't search a large collection. So: bi-encoder, single vector each, dot-product scoring. Initialize from BERT base — I'm not learning language from scratch, I'm reshaping a pre-trained model's space for retrieval. Should I use one shared encoder for both query and document, or two separate ones as DPR does? Two encoders double the parameters and, more importantly, can specialize the query tower and document tower to whatever quirks the training distribution has — which is fine in-domain but brittle zero-shot. A single shared encoder is forced to put queries and documents in the *same* space symmetrically, which should transfer more robustly across domains and languages. And with random cropping my two views already come from the same distribution, so a shared encoder is the natural match. One encoder it is.

How do I collapse BERT's sequence of token vectors into one embedding? The obvious candidate is the [CLS] token, since that's what BERT's pre-training shapes for sentence-level tasks. But [CLS] is tuned for next-sentence-prediction-style objectives, not for capturing the content of a span uniformly. The alternative is to *average* the last-layer hidden states over the (non-padding) tokens — mean pooling. Mean pooling treats every token's contribution evenly, which for retrieval (where any part of a long passage might match the query) is a better inductive bias than betting everything on one special token. I'll mean-pool: sum the last hidden states, masking out padding, and divide by the number of real tokens. Concretely, zero out padded positions, sum over the sequence, divide by the attention-mask sum.

One more representation choice: do I L2-normalize the embeddings before the dot product? Normalizing turns the dot product into cosine similarity and keeps the logit scale bounded, which interacts cleanly with the temperature τ — without it, embedding norms could drift and silently rescale the effective temperature. I'll keep normalization as an optional switch on both query and key sides, and to stay consistent I initialize the random queue as unit vectors too.

Let me also be honest about MoCo vs. just using a big batch (SimCLR-style in-batch negatives). If I could afford huge batches, in-batch negatives with symmetric gradients might be just as good — the real reason I'm reaching for MoCo isn't that the momentum encoder is magic, it's that the queue lets me scale negatives *without* scaling the batch, and scaling negatives is the lever that matters. So MoCo is the pragmatic choice for getting many negatives at fixed memory, and the momentum encoder is the thing that makes the queue's stale negatives usable. If the queue equals the batch size, MoCo and in-batch should behave similarly; the win shows up as I grow the queue past what a batch could hold.

Now let me assemble the training step precisely, because the queue bookkeeping is fiddly and easy to get wrong. Encode the query view with the trainable encoder: q = f_q(query). Under no-grad, first do the momentum update of the key encoder, then encode the key view: k = f_k(key) — detached, no gradient. Build the logits: the positive logit is the per-example dot product ⟨q, k⟩ (one number per example), and the negative logits are q against every column of the queue, ⟨q, queue⟩ (K numbers per example). Concatenate them as [positive | negatives] so the positive sits at column 0, divide the whole thing by τ, and apply cross-entropy with the target label fixed to 0 for every example. I want to be sure this actually computes the InfoNCE loss I wrote at the start and not something subtly different, so let me check it on the same four scores. Cross-entropy with target class 0 evaluates to −log of the softmax probability of column 0, which is −log[ exp(s₀/τ) / Σⱼ exp(sⱼ/τ) ]. With s = (0.8, 0.6, 0.7, 0.2) and τ = 0.05, `F.cross_entropy` returns 0.14293706, and computing −log(exp(s₀/τ)/Σ exp(sⱼ/τ)) by hand gives 0.14293696 — equal to seven decimals (the tiny gap is float roundoff). So the framework cross-entropy at index 0 *is* the InfoNCE objective from line one, positive at column 0, no hidden discrepancy. Then enqueue the current batch's keys into the queue and dequeue the oldest, advancing a ring pointer; the queue size must be a multiple of the batch size so the pointer wraps cleanly. Initialize the queue with random unit vectors so it is well-defined before stored key embeddings replace them.

Why fix the label to index 0 rather than scatter positives around? Because I built the logits with the positive deliberately first; the label is just "the positive is column 0." Cross-entropy then pushes mass onto column 0 and away from all the queue columns, which is the contrastive objective whose gradient I checked above — positive score up, negative scores down, force concentrated on the hardest negative. I can also fold in a touch of label smoothing on that cross-entropy as a mild regularizer.

Optimization is unremarkable: AdamW, a learning rate around 5e-5, a long schedule (hundreds of thousands of steps), batch in the low thousands, on a mix of raw corpora — Wikipedia for clean encyclopedic text and CCNet for breadth and multilinguality, sampled so neither dominates. For the multilingual model I'd initialize from multilingual BERT instead, sample languages uniformly, and add a warmup-then-linear-decay schedule for stability, since multilinguality adds variance. And critically, none of this touches a single labeled pair.

Putting the pieces together, the causal chain is: I can't get labels, so I lean on the one label-free signal — document identity — and discriminate one document against many via InfoNCE; "many negatives at fixed memory" forces a momentum-encoder queue (MoCo) rather than huge batches; "label-free positive pairs that don't throw away the lexical signal BM25 exploits" forces independent random cropping rather than ICT's mutually-exclusive complement; "must search millions of documents" forces a single-vector shared bi-encoder with mean pooling and dot-product scoring, indexed with FAISS. Here is the code those decisions land on.

```python
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class Encoder(transformers.BertModel):
    """Bi-encoder tower: BERT -> one vector via mean pooling. Shared by query and key."""
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, normalize=False):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        last = out["last_hidden_state"]
        # mean pooling: zero the padding, sum over tokens, divide by #real tokens
        last = last.masked_fill(~attention_mask[..., None].bool(), 0.0)
        emb = last.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if normalize:                      # optional cosine scoring, bounded logits for tau
            emb = F.normalize(emb, dim=-1)
        return emb


def token_delete(tokens, p=0.10):
    if p <= 0.0 or len(tokens) <= 1:
        return tokens
    kept = [tok for tok in tokens if random.random() > p]
    return kept if kept else [tokens[random.randrange(len(tokens))]]


def build_positive_pair(tokens, low=0.05, high=0.5, delete_prob=0.10):
    """Two independent crops of one document = a label-free positive pair.
    Overlap between crops teaches lexical matching; non-overlap teaches semantics."""
    n = len(tokens)
    if n == 0:
        return [], []

    def crop():
        length = max(1, min(n, int(round(n * random.uniform(low, high)))))
        start = random.randint(0, n - length)
        return token_delete(tokens[start:start + length], delete_prob)
    return crop(), crop()


class MoCoTrainer(nn.Module):
    """InfoNCE with a momentum-encoder queue: many negatives at fixed batch size."""
    def __init__(self, opt):
        super().__init__()
        self.temperature = opt.temperature          # ~0.05
        self.momentum = opt.momentum                # ~0.9995
        self.queue_size = opt.queue_size            # up to ~131072
        self.label_smoothing = opt.label_smoothing  # optional, mild regularizer
        self.norm = opt.normalize                   # L2-normalize embeddings (optional)
        self.encoder_q = Encoder.from_pretrained(opt.model_id)
        self.encoder_k = copy.deepcopy(self.encoder_q)   # momentum (key) encoder
        for p in self.encoder_k.parameters():
            p.requires_grad = False                 # keys get no gradient
        self.register_buffer("queue", F.normalize(torch.randn(opt.dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.momentum + pq.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bsz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bsz == 0            # ring pointer wraps cleanly
        self.queue[:, ptr:ptr + bsz] = keys.T
        self.queue_ptr[0] = (ptr + bsz) % self.queue_size

    def forward(self, q_tokens, q_mask, k_tokens, k_mask):
        q = self.encoder_q(q_tokens, q_mask, normalize=self.norm)
        with torch.no_grad():
            self._momentum_update()                  # slow EMA before encoding keys
            k = self.encoder_k(k_tokens, k_mask, normalize=self.norm)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)     # positive at column 0
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)  # InfoNCE
        self._dequeue_and_enqueue(k)
        return loss


def train(trainer, loader, opt):
    optim = torch.optim.AdamW((p for p in trainer.parameters() if p.requires_grad), lr=opt.lr)
    for batch in loader:                             # batches mix Wikipedia + CCNet
        loss = trainer(batch["q_tokens"], batch["q_mask"], batch["k_tokens"], batch["k_mask"])
        loss.backward(); optim.step(); optim.zero_grad()
```

The whole thing trains on raw text with no query–document labels: identity gives the supervision, InfoNCE turns it into a loss, MoCo's momentum-queue supplies the many consistent negatives that make the discrimination informative, random cropping makes positives that keep both lexical and semantic signal, and a shared mean-pooled BERT bi-encoder gives the single-vector index that scales — a dense retriever I can build, and point at any domain or language, without ever collecting a label.
