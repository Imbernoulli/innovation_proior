Let me start from the uncomfortable observation. A masked language model like BERT, asked to fill "The ___ is the currency of the United Kingdom," answers "pound." So it *knows* a fact about the world. But where is that fact? It's smeared across the weights — implicit, undelimited, impossible to point at. And that implicitness costs me three things at once. I can't inspect what the model knows or where. I can't grow the knowledge without growing the network, because the only place facts can live is in parameters, so more facts means a bigger, slower, more expensive model. And I can't revise a fact — if the world changes, I retrain. The thread tying all three together is the same: the model's only memory is its parameters. So the fix can't be "train a bigger BERT." The fix has to *move the knowledge out of the parameters* into something explicit and modular.

What would explicit knowledge look like? The obvious candidate is text — a corpus, say all of Wikipedia, sitting outside the network. If the model, before predicting a masked token, could go *retrieve* the relevant passage from that corpus and read it, then the knowledge lives in the corpus, not the weights: I can read what was retrieved (interpretable), I can add documents (capacity isn't bounded by network size), and I can swap the corpus to update facts (modular). So the architecture I want is: input x → retrieve documents z from a corpus Z → predict y by attending over x and z. Retrieve, then predict.

Now the hard question. How do I *learn* the retriever? I have no labels saying "for this masked sentence, document #4,381,002 is the right one." All I have is the language-modeling signal: did the model predict the masked token correctly? So the retriever has to be trained purely from whether its retrievals *help prediction*. A retrieval that lowers the model's perplexity on the masked token was helpful and should be reinforced; a useless retrieval should be discouraged. I need to turn that intuition into a differentiable objective.

The clean way is to make the retrieved document a *latent variable*. I don't observe z; I observe that x produced y. So write the likelihood of the observed thing, marginalizing over the unobserved document:

  p(y | x) = Σ_{z ∈ Z} p(y | z, x) · p(z | x).

Here p(z|x) is the retriever — how likely it is to pick document z for input x — and p(y|z,x) is the predictor — how likely the correct output is once we condition on document z. Maximize log p(y|x) by gradient descent over both the retriever's parameters and the predictor's. That's the entire training principle. But before I trust it, I want to know *why* maximizing this marginal makes the retriever learn anything sensible — because the retrieval is latent, it's genuinely not obvious that the objective rewards good retrievals rather than, say, collapsing to retrieve nothing. So let me actually compute the gradient with respect to the retriever and see what it does.

Let θ be the retriever's parameters. Start from the log of the marginal:

  ∇ log p(y|x) = (1/p(y|x)) ∇ p(y|x).

Now ∇ p(y|x) = ∇ Σ_z p(y|z,x) p(z|x). The predictor p(y|z,x) doesn't depend on θ (it's the encoder's job, parameter φ), so only p(z|x) gets differentiated:

  ∇ p(y|x) = Σ_z p(y|z,x) ∇ p(z|x).

Use the log-derivative trick, ∇ p(z|x) = p(z|x) ∇ log p(z|x):

  ∇ log p(y|x) = (1/p(y|x)) Σ_z p(y|z,x) p(z|x) ∇ log p(z|x).

Look at the coefficient p(y|z,x) p(z|x) / p(y|x). By Bayes' rule that's exactly p(z|y,x) — the posterior over which document was used, given that we got the right answer. So

  ∇ log p(y|x) = Σ_z p(z|y,x) ∇ log p(z|x).

That's already telling: it's an expectation over the posterior p(z|y,x) of the score-gradient. Now expand ∇ log p(z|x). The retriever is a softmax over relevance scores f(x,z):

  log p(z|x) = f(x,z) − log Σ_{z'} exp f(x,z'),

so

  ∇ log p(z|x) = ∇ f(x,z) − Σ_{z'} p(z'|x) ∇ f(x,z').

Substitute back:

  ∇ log p(y|x) = Σ_z p(z|y,x) [ ∇ f(x,z) − Σ_{z'} p(z'|x) ∇ f(x,z') ].

The first term is Σ_z p(z|y,x) ∇ f(x,z). The second term: Σ_z p(z|y,x) [Σ_{z'} p(z'|x) ∇ f(x,z')] — the inner sum doesn't depend on z, and Σ_z p(z|y,x) = 1, so it's just Σ_{z'} p(z'|x) ∇ f(x,z'). Both sums run over the whole corpus, so rename z' to z and combine:

  ∇ log p(y|x) = Σ_z [ p(z|y,x) − p(z|x) ] ∇ f(x,z).

So the retriever moves each document's score f(x,z) in proportion to [posterior − prior] = [p(z|y,x) − p(z|x)]. A document's score gets pushed *up* exactly when its posterior probability (given the correct answer) exceeds its prior probability (before seeing the answer) — i.e. when conditioning on the answer makes the document look *more* likely than the retriever currently thinks. Read aloud, that's "this document helped explain the right answer, retrieve it more." Let me make the multiplier more interpretable. Expand the posterior with Bayes again, p(z|y,x) = p(y|z,x) p(z|x) / p(y|x):

  p(z|y,x) − p(z|x) = [ p(y|z,x) / p(y|x) − 1 ] p(z|x).

So define r(z) = [ p(y|z,x) / p(y|x) − 1 ] p(z|x), and ∇ log p(y|x) = Σ_z r(z) ∇ f(x,z). The sign of r(z) should be the sign of p(y|z,x) − p(y|x): p(y|z,x) is "how well does the correct answer score *with* document z," and p(y|x) = Σ_z p(z|x) p(y|z,x) is the *expected* score over a random draw from the retriever, so r(z) > 0 would mean document z makes the prediction better than the retriever's average document does.

I've rearranged this twice now, so before I lean on it I should actually put numbers through it rather than trust the algebra. Take a toy corpus of three documents with prior p(z|x) = (0.5, 0.3, 0.2) and per-document answer likelihoods p(y|z,x) = (0.8, 0.1, 0.3). Then p(y|x) = 0.5·0.8 + 0.3·0.1 + 0.2·0.3 = 0.40 + 0.03 + 0.06 = 0.49. The posterior is p(z|y,x) = p(y|z,x)p(z|x)/0.49 = (0.40, 0.03, 0.06)/0.49 = (0.8163, 0.0612, 0.1224); it sums to 1.000, which it must. Now the coefficients: posterior − prior = (0.8163−0.5, 0.0612−0.3, 0.1224−0.2) = (+0.3163, −0.2388, −0.0776). The other form had better agree: [p(y|z,x)/0.49 − 1]·p(z|x) = ([1.633−1]·0.5, [0.204−1]·0.3, [0.612−1]·0.2) = (+0.3163, −0.2388, −0.0776). It matches to the digit. And the signs land where the interpretation predicted: doc 0 has p(y|z,x)=0.8 > 0.49 and gets a positive coefficient; docs 1 and 2 have 0.1 and 0.3, both below 0.49, and both get negative coefficients. One more thing I didn't expect but should have: the three coefficients sum to +0.3163 − 0.2388 − 0.0776 = 0. Of course — Σ_z p(z|y,x) = Σ_z p(z|x) = 1, so the differences cancel. That means the gradient never uniformly inflates all scores; it can only *redistribute* score mass from below-average documents to above-average ones, which is exactly the behavior a softmax retriever should have (a uniform push would do nothing after normalization anyway). So the latent objective rewards retrievals that improve prediction — that's the performance-based signal I wanted, and it's a redistribution, not a runaway.

One more thing this gradient tells me, a useful sanity check on the limit. Suppose there's one document z* that gives perfect prediction, p(y|z*,x) = 1, and every other document gives zero, p(y|z',x) = 0. Then p(z*|y,x) = 1 and the gradient collapses to ∇ f(x,z*) − Σ_z p(z|x) ∇ f(x,z) = ∇ log p(z*|x). That's just supervised maximum likelihood of retrieving the gold document z*. So the latent-variable objective smoothly interpolates to ordinary supervised retrieval training in the limit where one document carries all the value. Good — it's not some exotic objective, it's the natural generalization.

The retriever has to be fast enough to score against a corpus and smooth enough to train, so I want the probability of a document to come from a softmax over a cheap relevance score. A dense inner product gives me both:

  p(z|x) = exp f(x,z) / Σ_{z'} exp f(x,z'),  f(x,z) = Embed_input(x)ᵀ Embed_doc(z).

For the embeddings I'll use BERT-style Transformers: tokenize, prepend [CLS], join spans with [SEP], run the Transformer, take the [CLS] vector as the pooled representation, then a linear projection W to a lower dimension. So Embed_input(x) = W_input · BERT_CLS([CLS] x [SEP]) and Embed_doc(z) = W_doc · BERT_CLS([CLS] z_title [SEP] z_body [SEP]) — the document gets its title and body, since titles carry disambiguating signal. Call all the retriever parameters θ.

The predictor — the knowledge-augmented encoder — models p(y|z,x). Here I want *rich* interaction between the question and the retrieved document, so I join x and z into one sequence and run a *separate* Transformer over it, letting x and z cross-attend before predicting y. This is the place I can afford cross-attention, because by the time the predictor runs I've already narrowed to a handful of documents. For pre-training the task is MLM: for each masked position j, predict the original token with p(y_j|z,x) ∝ exp(w_jᵀ BERT_MASK(j)(join(x, z_body))), product over the masked positions. For Open-QA fine-tuning the task is span extraction: assume the answer y appears as a contiguous span somewhere in z, let S(z,y) be the set of matching spans, and score p(y|z,x) ∝ Σ_{s∈S(z,y)} exp(MLP([h_START(s); h_END(s)])), where h_START, h_END are the Transformer's start/end token vectors over join(x, z_body). Call the predictor parameters φ.

Now the wall I've been circling. The marginal p(y|x) = Σ_{z∈Z} sums over the *entire* corpus — 13 million documents. I cannot compute that sum, nor its gradient, over 13 million documents per training step. First approximation: most documents have essentially zero p(z|x), so truncate the sum to a small candidate set with the highest retrieval probability. That's reasonable if the retrieval distribution is peaked, which it should be once trained. So I marginalize over the retrieved candidates rather than the full corpus; during pre-training that set has 8 total candidates including the null document. But that just moves the problem: how do I *find* the high-probability candidates out of 13 million without scoring all of them? This is where the inner-product form pays off — the ranking by p(z|x) is the ranking by f(x,z) = Embed_input(x)ᵀ Embed_doc(z), a pure inner product. So finding the candidate set is Maximum Inner Product Search, and MIPS over millions of vectors runs in sub-linear time with the right index. So: precompute Embed_doc(z) for all 13M documents, build a MIPS index, and at each step embed the query and retrieve the candidate documents.

But here's the catch that almost kills the whole approach. The MIPS index is built from Embed_doc(z), which depends on θ. The instant I take a gradient step on θ, every document embedding changes, and the index is *stale* — it no longer reflects the current retriever. If I rebuild the index every step, I'm re-embedding 13 million documents per step, which is hopeless. If I never rebuild it, the retriever and its index drift apart and training is incoherent.

The resolution: separate the two roles of the index. The index is *only* used to *select* which documents to look at. Once I have that small set, I recompute the candidate scores, the candidate-set softmax, and the gradients using the *fresh* θ on those documents. So a slightly stale index only means I might select a slightly suboptimal candidate set — but the loss I train on is evaluated with current query and document embeddings for the selected documents. That tolerance is what makes it work: I can let the index lag. So I refresh it *asynchronously*. Run two jobs in parallel — a trainer that does gradient updates, and an index-builder that, every several hundred steps, takes a snapshot θ' of the current parameters, re-embeds the whole corpus, and ships back a new index while the trainer keeps going. In practice that's about one refresh per ~500 steps. The trainer never blocks on re-indexing, the index is never more than a few hundred steps stale, and the masked-LM gradient still updates the retriever scores and the document encoder on the retrieved candidates. Updating the document encoder during pre-training is the thing prior latent-retrieval Open-QA didn't do; it froze the index after a heuristic warm-start, so the document side of retrieval was never refined by the language-modeling signal. (For *fine-tuning* I'll actually simplify: build the index once from the pre-trained θ and stop updating Embed_doc — pre-training already gives a good doc encoder, and I keep fine-tuning Embed_input so the query side still adapts. The expensive asynchronous refresh is reserved for pre-training, where it matters most.)

Now I have a working objective and a tractable system. But the gradient I just derived has a sharp dependence on the *kind* of masked token — r(z) only moves when documents differ in how well they predict y — and that makes me want to walk through the cases where a token gives the retriever nothing to learn from, before I burn compute on a run that quietly learns nothing. Each such case suggests a fix.

First failure: most masked tokens don't need world knowledge at all. If I mask "of" in "the currency ___ the UK," local context predicts it — no document helps, the retriever gets no useful signal, and worse, the predictor learns it can ignore retrievals. The whole point is knowledge, so I should *mask the tokens that require knowledge*. Those are the salient spans — named entities like "United Kingdom" and dates like "July 1969." So instead of BERT's random token masking, I do salient span masking: run a named-entity tagger and a date regex over the sentence, and mask one salient span. That concentrates the training signal on exactly the cases where retrieval can help. I'd expect this to matter a lot — it's the difference between the retriever being needed and being decoration.

Second failure: even among salient spans, some are predictable without retrieval, or the world knowledge is so common it's already baked into the predictor. For those, *forcing* a retrieval distorts the credit assignment — the retriever gets rewarded or punished for documents on a prediction it could make alone. So I add a *null document* z_null — an empty document — to the candidate set. When no real retrieval is needed, probability mass and credit flow to this consistent sink instead of being misattributed to some arbitrary real document. Comparing log p(y|z,x) against log p(y|z_null,x) also gives a clean retrieval utility for document z: how much z helped over retrieving nothing.

Third failure, and it's a sneaky one, present only when the pre-training corpus X and the knowledge corpus Z are the same (both Wikipedia). Then for a masked sentence x drawn from document z, that very document z sits in the corpus *unmasked* — it literally contains the answer in plain text. Retrieving it makes prediction trivial: the encoder just copies the unmasked token. That produces a huge positive gradient for p(z|x) on that exact document, and if it happens often the retriever degenerates into learning *exact string match* — "find the document I came from" — which captures none of the semantic relevance I actually want. So during pre-training I must *prohibit this trivial candidate*: exclude the document the masked sentence came from from the retrieval set.

Fourth failure: cold start. At initialization the embeddings are meaningless, so the top-k retrievals are random junk unrelated to x. The predictor quickly learns those documents are useless and learns to *ignore* them. But once the predictor ignores retrievals, p(y|z,x) is the same constant for every z, and my coefficient formula has a clean consequence I can check directly. Put p(y|z,x) = c for all z back through the toy corpus: p(y|x) = Σ_z p(z|x)·c = c, the posterior becomes p(z|y,x) = c·p(z|x)/c = p(z|x), and so r(z) = p(z|y,x) − p(z|x) = 0 for *every* document, exactly. Not approximately — the retriever's gradient is identically zero the moment the predictor stops distinguishing documents. A vicious cycle: bad retrieval → ignored retrieval → exactly-zero retriever gradient → bad retrieval forever. I need a warm start so the retriever is already *somewhat* sensible before joint training begins. The Inverse Cloze Task does exactly this: given a sentence, train the retriever to recover the document it came from. That's a cheap self-supervised proxy for relevance that gives Embed_input and Embed_doc a meaningful starting geometry, breaking the cold-start cycle. And the predictor I warm-start with ordinary BERT-base pre-training (12 layers, 768 hidden, 12 heads). After that warm start, the full retrieve-then-predict marginal objective takes over and refines everything jointly.

Put it together. Pre-training: take a sentence from X, mask a salient span, retrieve a small candidate set through the asynchronously refreshed MIPS index, exclude the trivial self-document, include the null document so the total set is 8 candidates, recompute fresh candidate scores and the marginal MLM likelihood over that set, and gradient-step both θ and φ. Fine-tuning for Open-QA: same retrieve-then-marginalize structure, but the predictor scores answer spans, the index is built once and Embed_doc frozen (Embed_input still trained), and k is small (≈5) so the whole thing runs on a single 12 GB GPU. Since the model reads from the corpus at inference time, changing the corpus gives me a direct path to changing accessible facts, even though the encoder can still remember common facts in its parameters.

The implementation lands on the marginal, the gradient-justified scoring, and the asynchronous-refresh structure.

```python
import copy
import torch, torch.nn as nn, torch.nn.functional as F

# --- Retriever: dense inner-product model, p(z|x) = softmax_z f(x,z) ---
class Retriever(nn.Module):
    def __init__(self, bert_q, bert_d, dim):
        super().__init__()
        self.bert_q, self.bert_d = bert_q, bert_d
        self.W_input = nn.Linear(bert_q.config.hidden_size, dim, bias=False)
        self.W_doc   = nn.Linear(bert_d.config.hidden_size, dim, bias=False)
    def embed_input(self, x):                      # W_input . CLS(BERT([CLS] x [SEP]))
        return self.W_input(self.bert_q(**x).last_hidden_state[:, 0])
    def embed_doc(self, z):                         # W_doc . CLS(BERT([CLS] title [SEP] body [SEP]))
        return self.W_doc(self.bert_d(**join_title_body(z)).last_hidden_state[:, 0])
    def score(self, qvec, dvecs):                   # f(x,z) = inner product
        return qvec @ dvecs.t()

# --- Knowledge-augmented encoder: p(y|z,x) via cross-attention over join(x, z) ---
class KnowledgeAugmentedEncoder(nn.Module):
    def __init__(self, bert):                       # SEPARATE Transformer from the retriever
        super().__init__(); self.bert = bert
        self.mlm_head  = nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
        self.span_mlp  = nn.Sequential(nn.Linear(2*bert.config.hidden_size, 1))
    def mlm_logprob(self, x, z, target):            # pre-training: predict masked tokens
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        logits = self.mlm_head(h[target.masked_pos])
        return F.log_softmax(logits, -1).gather(-1, target.tokens).sum()  # product over masks
    def span_logprob(self, x, z, target):           # fine-tuning: answer span score
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        s = torch.stack([self.span_mlp(torch.cat([h[a], h[b]])) for a, b in target.matching_spans(z)])
        return torch.logsumexp(s, 0)                                # sum over matching spans
    def logprob(self, x, z, target, task="mlm"):
        return self.mlm_logprob(x, z, target) if task == "mlm" else self.span_logprob(x, z, target)

# --- MIPS index over ALL doc embeddings; only used to SELECT the top-k ---
class MIPSIndex:
    def build(self, doc_embeddings): self.E = doc_embeddings        # re-embed whole corpus
    def search(self, qvec, k):       return topk_inner_product(self.E, qvec, k)
    def replace(self, other):        self.E = other.E

# --- Marginal log-likelihood over the retrieved candidate set, with fresh theta ---
def marginal_logprob(x, y, retriever, predictor, index, docs, K, null_doc, task="mlm"):
    qvec = retriever.embed_input(x)
    cand_ids = index.search(qvec.detach(), K + 8)     # stale index just PICKS candidates
    cand = [docs[i] for i in cand_ids if not is_trivial(i, x)][:K - 1] + [null_doc]
    dvecs = retriever.embed_doc(batch(cand))         # recompute embeddings with current theta
    log_pz = F.log_softmax(retriever.score(qvec, dvecs), -1).squeeze(0)  # top-K p(z|x)
    log_pyz = torch.stack([predictor.logprob(x, z, y, task) for z in cand])
    return torch.logsumexp(log_pz + log_pyz, 0)      # log sum_z p(z|x) p(y|z,x)

# --- Asynchronous index refresh: trainer never blocks; index lags a few hundred steps ---
class AsyncIndexRefresh:
    def __init__(self, every=500):
        self.every, self.pending = every, None
    def maybe_refresh(self, retriever, index, corpus, step):
        if self.pending is not None and self.pending.done():
            index.replace(self.pending.result())      # install finished background index
            self.pending = None
        if self.pending is None and step % self.every == 0:
            theta = copy.deepcopy(retriever).eval()   # theta' snapshot for the builder
            self.pending = launch_background(lambda: build_mips_index(theta, corpus))

def train(retriever, predictor, index, corpus, docs, data, steps, k, null_doc):
    refresher = AsyncIndexRefresh(every=500)
    opt = torch.optim.Adam(list(retriever.parameters()) + list(predictor.parameters()), lr=3e-5)
    for step, (x, y) in enumerate(data):             # x = salient-span-masked sentence
        loss = -marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc)
        opt.zero_grad(); loss.backward(); opt.step()
        refresher.maybe_refresh(retriever, index, corpus, step)
```

Knowledge welded into parameters is opaque, capacity-bounded, and unrevisable, so I move it into an external text corpus the model learns to retrieve from; with no retrieval labels, I make the document a latent variable and maximize the marginal likelihood of the prediction, whose retriever gradient equals Σ_z [p(z|y,x) − p(z|x)] ∇f(x,z), rewarding exactly the documents that beat the retriever's average; the inner-product score turns candidate selection into MIPS over millions of documents, and the index-only-selects insight lets an asynchronously refreshed stale selector coexist with fresh gradients on the retrieved few; salient-span masking, a null document, prohibiting the trivial self-retrieval, and an ICT warm start keep the latent retriever from collapsing into uselessness or exact match.
