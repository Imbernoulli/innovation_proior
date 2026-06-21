A masked language model like BERT, asked to fill "The ___ is the currency of the United Kingdom," answers "pound," so it plainly *knows* a fact about the world. But that knowledge is smeared across millions of weights — implicit, undelimited, impossible to point at — and that implicitness costs three things at once. The model is opaque: I cannot inspect what it knows or where. It is capacity-bounded: the only place facts can live is in parameters, so holding more facts means a larger, slower, more expensive network. And it is unmodular: I cannot revise or swap out a single fact without retraining. The common thread is that the model's only memory is its parameters, so the fix cannot be "train a bigger BERT" — it has to move the knowledge *out* of the parameters into something explicit. The natural candidate for explicit knowledge is text: a corpus, say all of Wikipedia, sitting outside the network. If the model, before predicting a masked token, could go *retrieve* the relevant passage and read it, then knowledge lives in the corpus rather than the weights — I can read what was retrieved (interpretable), add documents (capacity unbounded by network size), and swap the corpus to update facts (modular). The prior options all fall short of this. BERT-style MLM keeps knowledge welded into weights, the very problem. T5-style generation-based Open-QA scales knowledge only by scaling parameters and remains implicit and uneditable. Heuristic-retrieval-and-read systems (DrQA, BM25/TF-IDF pipelines) cap coverage at whatever a *non-learned* retriever surfaces and never optimize the retriever for the end task. The closest prior, ORQA, does learn a MIPS retriever as a latent-variable Open-QA model, but it warm-starts the retriever only heuristically with the Inverse Cloze Task and then *freezes* the document index, so the document encoder is never refined by a language-modeling signal and there is no knowledge-rich pre-training stage built to make retrieval useful.

I propose REALM (Retrieval-Augmented Language Model pre-training): augment LM pre-training with a learned textual knowledge retriever, train it end-to-end from the masked-LM signal alone, and keep a search index over millions of documents usable while the encoders train. The architecture is retrieve-then-predict: input $x$, retrieve documents $z$ from a corpus $Z$, predict $y$ by attending over $x$ and $z$. The defining difficulty is that there are no retrieval labels — nothing tells me which of 13 million documents is right for a given masked sentence — so the retriever must be trained purely from whether its retrievals *help prediction*. I make this precise by treating the retrieved document as a latent variable and writing the likelihood of the observed output, marginalizing over the unobserved document:
$$p(y \mid x) = \sum_{z \in Z} p(y \mid z, x)\, p(z \mid x).$$
Here $p(z\mid x)$ is the retriever and $p(y\mid z,x)$ is the predictor; maximizing $\log p(y\mid x)$ by gradient descent over both trains the retriever without labels. What makes this work is the form of the retriever gradient. Writing $\nabla\log p(y\mid x) = \tfrac{1}{p(y\mid x)}\sum_z p(y\mid z,x)\nabla p(z\mid x)$, applying the log-derivative trick $\nabla p(z\mid x)=p(z\mid x)\nabla\log p(z\mid x)$, recognizing the coefficient $p(y\mid z,x)p(z\mid x)/p(y\mid x)$ as the posterior $p(z\mid y,x)$ by Bayes' rule, and expanding $\nabla\log p(z\mid x)=\nabla f(x,z)-\sum_{z'}p(z'\mid x)\nabla f(x,z')$ for the softmax, the two sums collapse into
$$\nabla\log p(y\mid x) = \sum_{z}\big[\,p(z\mid y,x)-p(z\mid x)\,\big]\,\nabla f(x,z) = \sum_z\Big[\tfrac{p(y\mid z,x)}{p(y\mid x)}-1\Big]p(z\mid x)\,\nabla f(x,z).$$
Each document's score $f(x,z)$ is pushed up exactly when $p(y\mid z,x) > p(y\mid x)$ — when document $z$ predicts the correct answer better than the retriever's *average* document does. Helpful retrievals are reinforced, useless ones suppressed, and this is proven to fall out of the marginal rather than assumed. As a sanity check on the limit, if one $z^*$ gives perfect prediction and all others give zero, then $p(z^*\mid y,x)=1$ and the gradient collapses to $\nabla\log p(z^*\mid x)$, ordinary supervised maximum-likelihood retrieval of the gold document — so the latent objective is the natural generalization of supervised retrieval, not an exotic one.

The retriever must be fast to score and smooth to train, which is why $p(z\mid x)$ comes from a softmax over a cheap dense inner-product relevance score, $f(x,z)=\text{Embed}_\text{input}(x)^\top\text{Embed}_\text{doc}(z)$, with each embedding a BERT [CLS] vector linearly projected to dimension $d$ (the document encoder reads title *and* body, since titles carry disambiguating signal). The predictor models $p(y\mid z,x)$ with a *separate* Transformer over the joined input and document body so that $x$ and $z$ can cross-attend before $y$ is produced — I can afford this richer interaction because by the time the predictor runs I have already narrowed to a handful of documents. For pre-training the task is MLM: $p(y\mid z,x)=\prod_j p(y_j\mid z,x)$ with $p(y_j\mid z,x)\propto\exp\!\big(w_j^\top\text{BERT}_{\text{MASK}(j)}(\text{join}(x,z_\text{body}))\big)$. For Open-QA fine-tuning it is span extraction, $p(y\mid z,x)\propto\sum_{s\in S(z,y)}\exp\!\big(\text{MLP}([h_\text{START}(s);h_\text{END}(s)])\big)$ over the spans of $z$ matching the answer string.

The wall is that the marginal sums over the entire 13-million-document corpus, which is intractable per step. The first move is top-$k$ truncation: a peaked retrieval distribution means I can replace the full sum by a small candidate set of highest-probability documents — 8 total candidates during pre-training (including the null document), top-5 at inference. Finding those candidates is where the inner-product form pays off: ranking by $p(z\mid x)$ is ranking by $f(x,z)$, so candidate selection is Maximum Inner Product Search over precomputed document embeddings, sub-linear in corpus size. The catch that almost kills the approach is that the MIPS index is built from $\text{Embed}_\text{doc}(z)$, which depends on $\theta$; one gradient step on $\theta$ makes every embedding — and thus the whole index — stale, yet re-embedding 13 million documents per step is hopeless. The resolution is to separate the index's two roles: the index is used *only to select* which documents to look at, and once that small set is chosen, the candidate scores, the candidate-set softmax, and the gradients are all recomputed on those documents with the *current* $\theta$. A slightly stale index therefore only risks a slightly suboptimal selection, while the loss is always evaluated with fresh embeddings — so I can let the index lag and refresh it asynchronously. A trainer does gradient updates uninterrupted while a parallel index-builder periodically snapshots $\theta'$, re-embeds the whole corpus, and ships back a new index, about once per 500 steps; the trainer never blocks and the index is never more than a few hundred steps behind. Updating the document encoder during pre-training this way is exactly what frozen-index latent-retrieval Open-QA never did. For fine-tuning I simplify: build the index once from the pre-trained $\theta$ and freeze $\text{Embed}_\text{doc}$ while still training $\text{Embed}_\text{input}$, so the query side adapts and the expensive asynchronous refresh is reserved for pre-training where it matters.

Several inductive biases keep the latent retriever from collapsing, each fixing a concrete failure. Most masked tokens (function words like "of") need only local context, give the retriever no signal, and teach the predictor to ignore retrievals — so I use *salient span masking*, masking named entities and dates found by an NER tagger plus a date regex, concentrating the signal where retrieval can help. Even among salient spans some need no retrieval, and forcing one distorts credit assignment — so I add a *null document* $z_\text{null}$ to the candidate set as a consistent sink for probability mass and credit when nothing real is needed (and a clean baseline for a document's retrieval utility). When the pre-training corpus and knowledge corpus are the same Wikipedia, the very document a masked sentence came from sits in the corpus unmasked and makes prediction trivial, which would degenerate the retriever into exact string match — so during pre-training I *prohibit this trivial candidate*, excluding the source document. And at initialization the embeddings are meaningless, so retrievals are random, the predictor learns to ignore them, $p(y\mid z,x)\approx p(y\mid x)$, the gradient multiplier vanishes, and the retriever can never improve — a cold-start cycle I break with an *ICT warm start* (predict the passage a sentence came from), giving the embeddings a meaningful starting geometry, with the encoder warm-started from BERT-base (12 layers, 768 hidden, 12 heads). After the warm start the full marginal objective refines everything jointly. Because the model reads the corpus at inference time, swapping the corpus changes accessible facts without changing weights, even though common facts can still live in the encoder.

```python
import copy
import torch, torch.nn as nn, torch.nn.functional as F

class Retriever(nn.Module):
    def __init__(self, bert_q, bert_d, dim):
        super().__init__(); self.bert_q, self.bert_d = bert_q, bert_d
        self.W_input = nn.Linear(bert_q.config.hidden_size, dim, bias=False)
        self.W_doc   = nn.Linear(bert_d.config.hidden_size, dim, bias=False)
    def embed_input(self, x): return self.W_input(self.bert_q(**x).last_hidden_state[:, 0])
    def embed_doc(self, z):   return self.W_doc(self.bert_d(**join_title_body(z)).last_hidden_state[:, 0])
    def score(self, q, d):    return q @ d.t()          # f(x,z) = inner product

class KnowledgeAugmentedEncoder(nn.Module):             # separate Transformer over join(x,z)
    def __init__(self, bert):
        super().__init__(); self.bert = bert
        self.mlm_head = nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
        self.span_mlp = nn.Linear(2 * bert.config.hidden_size, 1)
    def mlm_logprob(self, x, z, target):
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        logits = self.mlm_head(h[target.masked_pos])
        return F.log_softmax(logits, -1).gather(-1, target.tokens).sum()
    def span_logprob(self, x, z, target):
        h = self.bert(**join_input_and_body(x, z)).last_hidden_state
        scores = [self.span_mlp(torch.cat([h[a], h[b]])) for a, b in target.matching_spans(z)]
        return torch.logsumexp(torch.stack(scores), 0)
    def logprob(self, x, z, target, task="mlm"):
        return self.mlm_logprob(x, z, target) if task == "mlm" else self.span_logprob(x, z, target)

class MIPSIndex:                                        # only SELECTS the top-k
    def build(self, doc_embeddings): self.E = doc_embeddings
    def search(self, q, k):          return topk_inner_product(self.E, q, k)
    def replace(self, other):        self.E = other.E

def marginal_logprob(x, y, retriever, predictor, index, docs, K, null_doc, task="mlm"):
    q = retriever.embed_input(x)
    ids = index.search(q.detach(), K + 8)               # fetch slack, then drop trivial self-docs
    cand = [docs[i] for i in ids if not is_trivial(i, x)][:K - 1] + [null_doc]
    d = retriever.embed_doc(batch(cand))                # recompute with CURRENT theta
    log_pz  = F.log_softmax(retriever.score(q, d), -1).squeeze(0)   # top-K softmax p(z|x)
    log_pyz = torch.stack([predictor.logprob(x, z, y, task) for z in cand])
    return torch.logsumexp(log_pz + log_pyz, 0)         # log sum_z p(z|x) p(y|z,x)

class AsyncIndexRefresh:
    def __init__(self, every=500):
        self.every, self.pending = every, None
    def maybe_refresh(self, retriever, index, corpus, step):
        if self.pending is not None and self.pending.done():
            index.replace(self.pending.result())         # install finished background index
            self.pending = None
        if self.pending is None and step % self.every == 0:
            theta = copy.deepcopy(retriever).eval()      # snapshot theta' for index builder
            self.pending = launch_background(lambda: build_mips_index(theta, corpus))

def train(retriever, predictor, index, corpus, docs, data, steps, k, null_doc):
    refresher = AsyncIndexRefresh(every=500)
    opt = torch.optim.Adam(list(retriever.parameters()) + list(predictor.parameters()), lr=3e-5)
    for step, (x, y) in enumerate(data):                # x: salient-span-masked sentence
        loss = -marginal_logprob(x, y, retriever, predictor, index, docs, k, null_doc)
        opt.zero_grad(); loss.backward(); opt.step()
        refresher.maybe_refresh(retriever, index, corpus, step)
```
