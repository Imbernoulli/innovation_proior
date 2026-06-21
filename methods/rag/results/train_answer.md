Probe a large pre-trained language model and it recalls world facts like a knowledge base, but every fact it knows is welded into its weights, and that welding is the real defect. The knowledge cannot be revised: if a fact about the world changes, there is no edit button, only retraining or fine-tuning. It cannot be inspected: when the model answers, it hands me no passage I can check against. And when its parametric memory is thin or wrong it does not degrade gracefully — it hallucinates, confidently. These three failures share one root cause: the only memory the model has *is* its parameters. So the fix cannot be "make the model bigger"; it has to change *where the knowledge lives*. The hybrid approaches that already exist point in the right direction but stop short. ORQA and REALM pair a retriever with a *masked* language model and aim at *extractive* open-domain QA, where the answer must be a span you select out of a retrieved passage — they cannot rephrase an answer, cannot fuse evidence from two passages into one fluent sentence, cannot generate free-form text at all. REALM goes further and trains the retriever end-to-end while also updating the *document* encoder, which forces periodic re-embedding of the whole corpus and rebuilding of the search index mid-training: heavy machinery. A closed-book seq2seq model (BART, T5) is general but suffers exactly the parametric limits above, and the strongest closed-book QA baselines lean on enormous parameter counts. What is missing is a single, general-purpose generator given an explicit non-parametric memory, trained from ordinary input/output pairs with no supervision on which document to fetch.

I propose Retrieval-Augmented Generation (RAG): equip a pre-trained encoder–decoder generator — BART-large, roughly $400$M parameters, the *parametric* memory — with a second, *non-parametric* memory consisting of raw retrievable text, namely a dense vector index of Wikipedia (each article split into disjoint $100$-word chunks, $21$M documents) accessed by a pre-trained DPR bi-encoder. Choosing raw text rather than learned fact-embeddings is deliberate and load-bearing: raw text is human-readable, which is the interpretability I want, and human-writable, which is the editability — to update the model's knowledge I re-embed a newer Wikipedia with the same fixed document encoder, swap in the new index, and not a single weight changes. The retriever is a dense bi-encoder with a BERT-base document encoder $d(z)$ and a BERT-base query encoder $q(x)$ scoring by inner product,
$$p_\eta(z\mid x) \propto \exp\!\big(d(z)^\top q(x)\big),$$
and selecting the top documents is a maximum-inner-product search over the precomputed document vectors, done sub-linearly with a FAISS index (HNSW). I initialize this retriever from one already trained to fetch answer-bearing passages and use it to build the index; the generator gives $p_\theta(y_i \mid x, z, y_{<i})$, and feeding a retrieved document into BART needs nothing clever — concatenate $x$ and $z$ and let the encoder read both.

The central difficulty is that the training data is input/output pairs $(x,y)$ — a question and its answer, a claim and its label — with *no* label on which document should have been retrieved. The way through is to stop supervising the document and treat it as a *latent variable*: I never observe $z$, only that $x$ produced $y$, so I write the probability of what I do observe by marginalizing over what I do not,
$$p(y\mid x) = \sum_z p_\eta(z\mid x)\,p_\theta(y\mid x,z),$$
and maximize the marginal log-likelihood of the observed pairs, minimizing $\sum_j -\log p(y_j\mid x_j)$. This is what makes the retriever trainable without retrieval labels: the gradient of the marginal flows into *both* factors, so documents that, when retrieved, make the correct output more likely have their retrieval probability pushed up — the retriever learns which documents help purely from the downstream generation signal. Since I cannot sum over all $21$M chunks, the marginal is a top-$k$ approximation: retrieve the top $k$ under the inner-product score, then softmax-normalize their raw scores into $p_\eta(z\mid x)$ over just that retrieved set.

There are genuinely two latent structures, and which is right depends on the task, so I build both. The first treats the document as a single latent for the *whole* output — one document responsible for the entire sequence, which is natural for a short factoid drawn from one passage:
$$p(y\mid x) \approx \sum_{z\in\text{top-}k} p_\eta(z\mid x)\,\prod_i p_\theta(y_i\mid x,z,y_{<i}),$$
the RAG-Sequence model, with the sum *outside* the product. The second draws a fresh document at every token and marginalizes *inside* the product,
$$p(y\mid x) \approx \prod_i \sum_{z\in\text{top-}k} p_\eta(z\mid x)\,p_\theta(y_i\mid x,z,y_{<i}),$$
the RAG-Token model, with the sum *inside*: at each step the next-token distribution mixes the $k$ documents' predictions, so token $i$ can lean on a different document than token $i{+}1$, which is what lets one answer stitch content from several passages. These are different functions of $y$ — shared document choice versus a fresh draw per step — and they coincide only in the degenerate case $N=1$, where the product collapses and both reduce to the same one-step mixture. That is exactly the one-token classification formulation (the FEVER variant regenerates the claim, classifies from the final hidden state, then marginalizes class probabilities over documents), and its consistency is a check that the math is right.

Computation is in log-space. Turn the raw inner-product scores into $\log p_\eta(z\mid x)$ with a log-softmax over the retrieved set, and take the generator's per-token vocabulary log-probs from a log-softmax. For RAG-Token I need $\log\prod_i\sum_z p_\eta p_\theta$; a sum of products in probability space is a logsumexp in log-space, so at each token position I add the document's log-prob to that document's vocabulary log-probs (broadcasting it across vocabulary and positions) and logsumexp over the document axis, collapsing the $k$ documents into one marginal next-token distribution, then sum the chosen target's log-prob over positions. Because every transition is a proper distribution $p'_\theta(y_i\mid x,y_{<i}) = \sum_z p_\eta(z\mid x)\,p_\theta(y_i\mid x,z,y_{<i})$, RAG-Token decodes like an ordinary autoregressive seq2seq plugged straight into beam search. For RAG-Sequence the order is reversed: compute the full-sequence log-prob $\ell\ell_z = \sum_i \log p_\theta(y_i\mid x,z,y_{<i})$ per document, then combine once, $\log p(y\mid x) = \operatorname{logsumexp}_z[\ell\ell_z + \log p_\eta(z\mid x)]$. The document log-prob is a per-sequence quantity and must be added *once*, after the token log-probs are summed — adding it at every token would raise it to the $N$-th power and give a different, wrong model. Decoding RAG-Sequence is harder because the sum over $z$ sits outside the whole product, so there is no single per-token distribution to beam over: I run beam search separately for each document to get a candidate pool $Y$, and because a hypothesis from one document's beam may be missing from another's, I run an extra forward pass to score each candidate under every document, weight by $p_\eta(z\mid x)$, and sum — "thorough decoding," fine and cheap for short QA answers. Assuming $p_\theta(y\mid x,z)\approx 0$ for any $y$ absent from $z$'s own beam skips those passes — "fast decoding" — for long outputs.

One practical choice keeps the whole thing cheap: I do *not* fine-tune the document encoder. Updating $d(z)$ would stale every one of the $21$M index vectors the instant I take a step, forcing the same expensive asynchronous re-indexing REALM needed. The query encoder $q(x)$ can already move queries around the fixed document-vector space and BART can already learn to use whatever text returns, so I freeze the document encoder *and the index* and fine-tune only the query encoder ($\sim 110$M) plus BART ($\sim 406$M); the MIPS index never needs rebuilding. I also resist adding a null/empty-document branch — the generator sees $x$ directly and can learn to ignore unhelpful retrieved text, so the marginal stays over the retrieved passages only and the model stays simpler. The result is one architecture, trained with Adam on the negative marginal log-likelihood, that applies across knowledge-intensive QA, abstractive generation, and classification, is grounded in inspectable evidence, and has a memory I can edit by swapping the index.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DenseRetriever:                  # fixed doc encoder + index; trainable query encoder
    def __init__(self, query_encoder, doc_index):
        self.q_enc, self.index = query_encoder, doc_index   # FAISS MIPS over fixed doc vectors
    def retrieve(self, x, k):
        q = self.q_enc(x)
        docs, doc_scores = self.index.search(q, k)          # raw q(x)·d(z) scores
        return docs, doc_scores

class Seq2SeqGenerator(nn.Module):     # BART-large; doc enters by concatenation
    def __init__(self, bart): super().__init__(); self.bart = bart
    def forward(self, x, z, y):
        # one encoder/decoder row per retrieved document
        return self.bart(input_ids=concat_each(x, z), labels=repeat_for_docs(y, z)).logits

def _view_by_doc(seq_logits, doc_scores, n_docs):
    seq_lp = F.log_softmax(seq_logits, -1).reshape(
        seq_logits.size(0)//n_docs, n_docs, seq_logits.size(1), seq_logits.size(-1))
    doc_lp = F.log_softmax(doc_scores, 1)
    return seq_lp, doc_lp

def gather_target(token_logprobs, target, ignore_index=-100):
    mask = target.ne(ignore_index)
    safe_target = target.masked_fill(~mask, 0)
    if token_logprobs.dim() == 4:
        safe_target = safe_target.unsqueeze(1).expand(-1, token_logprobs.size(1), -1)
        mask = mask.unsqueeze(1).expand(-1, token_logprobs.size(1), -1)
    gathered = token_logprobs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return gathered.masked_fill(~mask, 0.0)

def ragtoken_nll(seq_logits, target, doc_scores, n_docs):       # marginalize per token
    seq_lp, doc_lp = _view_by_doc(seq_logits, doc_scores, n_docs)
    token_lp = torch.logsumexp(seq_lp + doc_lp.unsqueeze(-1).unsqueeze(-1), dim=1)
    return -gather_target(token_lp, target).sum(1)

def ragsequence_nll(seq_logits, target, doc_scores, n_docs):    # marginalize per sequence
    seq_lp, doc_lp = _view_by_doc(seq_logits, doc_scores, n_docs)
    ll = gather_target(seq_lp, target).sum(2)                   # token log-likelihood per doc
    return -torch.logsumexp(ll + doc_lp, dim=1)                 # add p_eta(z|x) once

def train(retriever, generator, data, n_docs, mode="sequence"):
    opt = torch.optim.Adam(list(retriever.q_enc.parameters()) + list(generator.parameters()))
    for x, y in data:
        docs, doc_scores = retriever.retrieve(x, n_docs)
        logits = generator(x, docs, y)
        if mode == "token":
            loss = ragtoken_nll(logits, y, doc_scores, n_docs).mean()
        else:
            loss = ragsequence_nll(logits, y, doc_scores, n_docs).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```
