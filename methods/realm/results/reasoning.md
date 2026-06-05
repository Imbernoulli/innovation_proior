Let me start from the uncomfortable observation. A masked language model like BERT, asked to fill "The ___ is the currency of the United Kingdom," answers "pound." So it *knows* a fact about the world. But where is that fact? It's smeared across the weights — implicit, undelimited, impossible to point at. And that implicitness costs me three things at once. I can't inspect what the model knows or where. I can't grow the knowledge without growing the network, because the only place facts can live is in parameters, so more facts means a bigger, slower, more expensive model. And I can't revise a fact — if the world changes, I retrain. The thread tying all three together is the same: the model's only memory is its parameters. So the fix can't be "train a bigger BERT." The fix has to *move the knowledge out of the parameters* into something explicit and modular.

What would explicit knowledge look like? The obvious candidate is text — a corpus, say all of Wikipedia, sitting outside the network. If the model, before predicting a masked token, could go *retrieve* the relevant passage from that corpus and read it, then the knowledge lives in the corpus, not the weights: I can read what was retrieved (interpretable), I can add documents (capacity isn't bounded by network size), and I can swap the corpus to update facts (modular). So the architecture I want is: input x → retrieve documents z from a corpus Z → predict y by attending over x and z. Retrieve, then predict.

Now the hard question, and it's the whole paper. How do I *learn* the retriever? I have no labels saying "for this masked sentence, document #4,381,002 is the right one." All I have is the language-modeling signal: did the model predict the masked token correctly? So the retriever has to be trained purely from whether its retrievals *help prediction*. A retrieval that lowers the model's perplexity on the masked token was helpful and should be reinforced; a useless retrieval should be discouraged. I need to turn that intuition into a differentiable objective.

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

This is beautiful, and it's exactly what I hoped for. The retriever moves each document's score f(x,z) in proportion to [posterior − prior] = [p(z|y,x) − p(z|x)]. A document gets its score pushed *up* exactly when its posterior probability (given the correct answer) exceeds its prior probability (before seeing the answer) — i.e. when conditioning on the answer makes the document look *more* likely than the retriever currently thinks. That's precisely "this document helped explain the right answer, retrieve it more." Let me make the multiplier even more interpretable. Expand the posterior with Bayes again, p(z|y,x) = p(y|z,x) p(z|x) / p(y|x):

  p(z|y,x) − p(z|x) = [ p(y|z,x) / p(y|x) − 1 ] p(z|x).

So define r(z) = [ p(y|z,x) / p(y|x) − 1 ] p(z|x), and ∇ log p(y|x) = Σ_z r(z) ∇ f(x,z). The sign of r(z) is the sign of p(y|z,x) − p(y|x). Now p(y|z,x) is "how well does the correct answer score *with* document z," and p(y|x) = Σ_z p(z|x) p(y|z,x) is the *expected* score over a random draw from the retriever. So r(z) > 0 exactly when document z makes the prediction better than the retriever's average document does. Documents that beat expectation get reinforced; documents that underperform get suppressed. The latent objective rewards retrievals that improve prediction — exactly the performance-based signal I wanted, and now I've proven it falls out of the math rather than assumed it.

One more thing this gradient tells me, a useful sanity check on the limit. Suppose there's one document z* that gives perfect prediction, p(y|z*,x) = 1, and every other document gives zero, p(y|z',x) = 0. Then p(z*|y,x) = 1 and the gradient collapses to ∇ f(x,z*) − Σ_z p(z|x) ∇ f(x,z) = ∇ log p(z*|x). That's just supervised maximum likelihood of retrieving the gold document z*. So the latent-variable objective smoothly interpolates to ordinary supervised retrieval training in the limit where one document carries all the value. Good — it's not some exotic objective, it's the natural generalization.

Now the components. The retriever is a dense inner-product model:

  p(z|x) = exp f(x,z) / Σ_{z'} exp f(x,z'),  f(x,z) = Embed_input(x)ᵀ Embed_doc(z).

For the embeddings I'll use BERT-style Transformers: tokenize, prepend [CLS], join spans with [SEP], run the Transformer, take the [CLS] vector as the pooled representation, then a linear projection W to a lower dimension. So Embed_input(x) = W_input · BERT_CLS([CLS] x [SEP]) and Embed_doc(z) = W_doc · BERT_CLS([CLS] z_title [SEP] z_body [SEP]) — the document gets its title and body, since titles carry disambiguating signal. Call all the retriever parameters θ.

The predictor — the knowledge-augmented encoder — models p(y|z,x). Here I want *rich* interaction between the question and the retrieved document, so I join x and z into one sequence and run a *separate* Transformer over it, letting x and z cross-attend before predicting y. This is the place I can afford cross-attention, because by the time the predictor runs I've already narrowed to a handful of documents. For pre-training the task is MLM: for each masked position j, predict the original token with p(y_j|z,x) ∝ exp(w_jᵀ BERT_MASK(j)(join(x, z_body))), product over the masked positions. For Open-QA fine-tuning the task is span extraction: assume the answer y appears as a contiguous span somewhere in z, let S(z,y) be the set of matching spans, and score p(y|z,x) ∝ Σ_{s∈S(z,y)} exp(MLP([h_START(s); h_END(s)])), where h_START, h_END are the Transformer's start/end token vectors over join(x, z_body). Call the predictor parameters φ.

Now the wall I've been circling. The marginal p(y|x) = Σ_{z∈Z} sums over the *entire* corpus — 13 million documents. I cannot compute that sum, nor its gradient, over 13 million documents per training step. First approximation: most documents have essentially zero p(z|x), so truncate the sum to the top-k documents under p(z|x). That's reasonable if the retrieval distribution is peaked, which it should be once trained. So I marginalize over only the top-k (small k, like 8). But that just moves the problem: how do I *find* the top-k out of 13 million without scoring all of them? This is where the inner-product form pays off — the ranking by p(z|x) is the ranking by f(x,z) = Embed_input(x)ᵀ Embed_doc(z), a pure inner product. So finding the top-k is Maximum Inner Product Search, and MIPS over millions of vectors runs in sub-linear time with the right index. So: precompute Embed_doc(z) for all 13M documents, build a MIPS index, and at each step embed the query and retrieve the top-k.

But here's the catch that almost kills the whole approach. The MIPS index is built from Embed_doc(z), which depends on θ. The instant I take a gradient step on θ, every document embedding changes, and the index is *stale* — it no longer reflects the current retriever. If I rebuild the index every step, I'm re-embedding 13 million documents per step, which is hopeless. If I never rebuild it, the retriever and its index drift apart and training is incoherent.

The resolution: separate the two roles of the index. The index is *only* used to *select* which top-k documents to look at. Once I have those k documents, I recompute f(x,z) and p(z|x) and their gradients using the *fresh* θ, exactly, for just those k. So a slightly stale index only means I might select a slightly suboptimal candidate set — but the scores and gradients I actually train on are always current. That tolerance is what makes it work: I can let the index lag. So I refresh it *asynchronously*. Run two jobs in parallel — a trainer that does gradient updates, and an index-builder that, every several hundred steps, takes a snapshot θ' of the current parameters, re-embeds the whole corpus, and ships back a new index while the trainer keeps going. In practice that's about one refresh per ~500 steps. The trainer never blocks on re-indexing; the index is never more than a few hundred steps stale; and the gradients are exact on the retrieved top-k. Backpropagating into the index — refreshing the doc encoder — is the thing prior latent-retrieval Open-QA didn't do; they froze the index after a heuristic warm-start, so their retriever never got refined by the language-modeling signal. (For *fine-tuning* I'll actually simplify: build the index once from the pre-trained θ and stop updating Embed_doc — pre-training already gives a good doc encoder, and I keep fine-tuning Embed_input so the query side still adapts. The expensive asynchronous refresh is reserved for pre-training, where it matters most.)

Now I have a working objective and a tractable system, but if I just turn it on I expect it to fail or learn nothing useful, because of how the gradient interacts with the *kind* of masked token. Let me think about what could go wrong, because each failure mode suggests a fix.

First failure: most masked tokens don't need world knowledge at all. If I mask "of" in "the currency ___ the UK," local context predicts it — no document helps, the retriever gets no useful signal, and worse, the predictor learns it can ignore retrievals. The whole point is knowledge, so I should *mask the tokens that require knowledge*. Those are the salient spans — named entities like "United Kingdom" and dates like "July 1969." So instead of BERT's random token masking, I do salient span masking: run a named-entity tagger and a date regex over the sentence, and mask one salient span. That concentrates the training signal on exactly the cases where retrieval can help. I'd expect this to matter a lot — it's the difference between the retriever being needed and being decoration.

Second failure: even among salient spans, some are predictable without retrieval, or the world knowledge is so common it's already baked into the predictor. For those, *forcing* a retrieval distorts the credit assignment — the retriever gets rewarded or punished for documents on a prediction it could make alone. So I add a *null document* z_∅ — an empty document — to the top-k set. When no real retrieval is needed, probability mass and credit flow to this consistent sink instead of being misattributed to some arbitrary real document. (As a bonus, comparing log p(y|z,x) against log p(y|z_∅,x) gives me a clean "retrieval utility" of document z — how much z actually helped over retrieving nothing.)

Third failure, and it's a sneaky one, present only when the pre-training corpus X and the knowledge corpus Z are the same (both Wikipedia). Then for a masked sentence x drawn from document z, that very document z sits in the corpus *unmasked* — it literally contains the answer in plain text. Retrieving it makes prediction trivial: the encoder just copies the unmasked token. That produces a huge positive gradient for p(z|x) on that exact document, and if it happens often the retriever degenerates into learning *exact string match* — "find the document I came from" — which captures none of the semantic relevance I actually want. So during pre-training I must *prohibit this trivial candidate*: exclude the document the masked sentence came from from the retrieval set.

Fourth failure: cold start. At initialization the embeddings are meaningless, so the top-k retrievals are random junk unrelated to x. The predictor quickly learns those documents are useless and learns to *ignore* them. But once the predictor ignores retrievals, p(y|z,x) ≈ p(y|x) for all z, so r(z) ≈ 0 — the retriever gets no gradient and can never improve. A vicious cycle: bad retrieval → ignored retrieval → no retriever gradient → bad retrieval forever. I need a warm start so the retriever is already *somewhat* sensible before joint training begins. The Inverse Cloze Task does exactly this: given a sentence, train the retriever to recover the document it came from. That's a cheap self-supervised proxy for relevance that gives Embed_input and Embed_doc a meaningful starting geometry, breaking the cold-start cycle. And the predictor I warm-start with ordinary BERT-base pre-training (12 layers, 768 hidden, 12 heads). After that warm start, the full retrieve-then-predict marginal objective takes over and refines everything jointly.

Put it together. Pre-training: take a sentence from X, mask a salient span, retrieve top-k (≈8) documents via the asynchronously-refreshed MIPS index plus the null document, exclude the trivial self-document, recompute exact scores and the marginal MLM likelihood over those k, and gradient-step both θ and φ. Fine-tuning for Open-QA: same retrieve-then-marginalize structure, but the predictor scores answer spans, the index is built once and Embed_doc frozen (Embed_input still trained), and k is small (≈5) so the whole thing runs on a single 12 GB GPU. Because the knowledge is now in the corpus, swapping in a newer Wikipedia changes the model's answers without any retraining — the modularity I wanted at the very start.

Now the code, grounding the marginal, the gradient-justified scoring, and the asynchronous-refresh structure.

```python
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
        return self.W_doc(self.bert_d(**z).last_hidden_state[:, 0])
    def score(self, qvec, dvecs):                   # f(x,z) = inner product
        return qvec @ dvecs.t()

# --- Knowledge-augmented encoder: p(y|z,x) via cross-attention over join(x, z) ---
class KnowledgeAugmentedEncoder(nn.Module):
    def __init__(self, bert):                       # SEPARATE Transformer from the retriever
        super().__init__(); self.bert = bert
        self.mlm_head  = nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
        self.span_mlp  = nn.Sequential(nn.Linear(2*bert.config.hidden_size, 1))
    def mlm_logprob(self, x, z, masked_pos, y):     # pre-training: predict masked tokens
        h = self.bert(**join(x, z)).last_hidden_state
        logits = self.mlm_head(h[masked_pos])
        return F.log_softmax(logits, -1).gather(-1, y).sum()       # prod over masks
    def span_logprob(self, x, z, y_spans):          # fine-tuning: answer span score
        h = self.bert(**join(x, z)).last_hidden_state
        s = torch.stack([self.span_mlp(torch.cat([h[a], h[b]])) for a, b in y_spans])
        return torch.logsumexp(s, 0)                                # sum over matching spans

# --- MIPS index over ALL doc embeddings; only used to SELECT the top-k ---
class MIPSIndex:
    def build(self, doc_embeddings): self.E = doc_embeddings        # re-embed whole corpus
    def search(self, qvec, k):       return topk_inner_product(self.E, qvec, k)

# --- Marginal log-likelihood over top-k (+ null doc), exact scores on fresh theta ---
def marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc):
    qvec = retriever.embed_input(x)
    cand_ids = index.search(qvec, k)                 # stale index just PICKS candidates
    cand = [docs[i] for i in cand_ids if not is_trivial(i, x)] + [null_doc]
    dvecs = retriever.embed_doc(batch(cand))         # recompute embeddings with current theta
    log_pz = F.log_softmax(retriever.score(qvec, dvecs), -1)        # log p(z|x), exact
    log_pyz = torch.stack([predictor.logprob(x, z, y) for z in cand])  # log p(y|z,x)
    return torch.logsumexp(log_pz + log_pyz, 0)      # log sum_z p(z|x) p(y|z,x)

# --- Asynchronous index refresh: trainer never blocks; index lags a few hundred steps ---
def maybe_refresh_index(retriever, index, corpus, step, every=500):
    if step % every == 0:                            # secondary builder uses a theta snapshot
        index.build(embed_all_docs(retriever, corpus))   # ~13M docs, in the background

def train(retriever, predictor, index, corpus, docs, data, steps, k, null_doc):
    opt = torch.optim.Adam(list(retriever.parameters()) + list(predictor.parameters()), lr=3e-5)
    for step, (x, y) in enumerate(data):             # x = salient-span-masked sentence
        loss = -marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc)
        opt.zero_grad(); loss.backward(); opt.step()
        maybe_refresh_index(retriever, index, corpus, step)
```

The causal chain: knowledge welded into parameters is opaque, capacity-bounded, and unrevisable, so move it into an external text corpus the model learns to retrieve from; with no retrieval labels, make the document a latent variable and maximize the marginal likelihood of the prediction, whose retriever gradient provably equals Σ_z [p(z|y,x) − p(z|x)] ∇f(x,z) — rewarding exactly the documents that beat the retriever's average; the inner-product score turns top-k selection into MIPS over millions of documents, and the index-only-selects insight lets an asynchronously-refreshed stale index coexist with exact, current gradients on the retrieved few; and salient-span masking, a null document, prohibiting the trivial self-retrieval, and an ICT warm start are the four inductive biases that keep the latent retriever from collapsing into uselessness or exact-match.
