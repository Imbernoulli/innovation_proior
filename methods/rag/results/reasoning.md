Let me start from what's actually broken, because it's easy to be dazzled by how much a big pre-trained language model knows. Probe one and it recalls facts like a knowledge base — but the knowledge is welded into the weights, and that welding is the problem. I can't revise it: if a fact about the world changes, there's no edit button, I'd have to retrain or fine-tune. I can't inspect it: when the model answers "Sauron," there's no source it can hand me, no passage I can check. And when its parametric memory is thin or wrong, it doesn't fail gracefully — it hallucinates, confidently. These three failures all have the same root cause: the only memory the model has *is* its parameters. So the fix has to change *where the knowledge lives*, not just how big the model is.

The shape of the fix seems clear in the abstract: give the generator a second memory that is *non-parametric* — an external store of raw text it can pull from at inference time. If knowledge lives in retrievable text, then I can revise it by editing the text, I can show the retrieved passage as provenance, and the generation can be grounded in real evidence instead of invented. Raw text specifically — not learned embeddings of facts the way memory networks do it — because raw text is human-readable (that's the interpretability) and human-writable (that's the editability). The commitment is important: the extra knowledge stays in a separate text store instead of being hidden back inside weights.

This isn't a wholly new idea, and I should be honest about where it already works and where it doesn't, because the gap is exactly my opening. Hybrid models that combine a parametric reader with a retriever already exist for open-domain QA — ORQA, REALM — and they're good. But look closely: both pair a *masked language model* with the retriever, and both are aimed at *extractive* QA — the answer has to be a span you select out of a retrieved passage. That's a real limitation. Extraction can't synthesize an answer that's phrased differently from the source, can't combine evidence from two passages into one fluent sentence, can't do free-form generation at all. The workhorse of NLP isn't a span-selector; it's the seq2seq generator — it can do QA, summarization, fact generation, even classification (a class is just a length-one target). Nobody has brought the non-parametric memory to a *general-purpose generator*. That's the move: replace the extractive reader with a pre-trained seq2seq generator and let it *generate* the answer conditioned on retrieved text.

Now I have to actually wire a retriever to a generator and train the thing. And the hard constraint is: I have input/output pairs (x, y) — a question and its answer, a claim and its label — but I have *no labels on which document should be retrieved*. So I can't train the retriever with supervised retrieval targets. How do I train a retriever when I never observe its output?

The way out is to stop treating the retrieved document as something I supervise and start treating it as a *latent variable*. I don't observe z; I only observe that x produced y. So I should write the probability of the thing I *do* observe, marginalizing over the thing I don't:

  p(y|x) = Σ_z p(z|x) · p(y|x, z),

where p(z|x) is the retriever's distribution over documents and p(y|x, z) is the generator's probability of the output given the input and that document. Then I just maximize the marginal log-likelihood of the observed (x, y) pairs — minimize Σ_j −log p(y_j | x_j). The beautiful thing about this is that the gradient of the marginal flows back into *both* p(y|x,z) and p(z|x): documents that, when retrieved, make the correct output more likely get their retrieval probability pushed up. The retriever learns *which documents help* purely from the downstream signal, no retrieval labels needed. That's the whole reason to phrase it as a latent variable.

For the components, I don't want to reinvent either half — the entire point is that pre-trained access mechanisms already carry knowledge. So the retriever is a dense bi-encoder: a BERT-base document encoder d(z) and a BERT-base query encoder q(x), with

  p_η(z|x) ∝ exp( d(z)ᵀ q(x) ),

and finding the top documents is a maximum inner product search over the precomputed document vectors, solvable sub-linearly with a FAISS index. I'll initialize this from a retriever already trained to fetch answer-bearing passages for Natural Questions and TriviaQA, and use it to build the document index over Wikipedia — call that index the non-parametric memory. The generator is a pre-trained encoder–decoder, BART-large (~400M params) — call its weights the parametric memory. To feed a retrieved document into BART I don't need anything clever: just concatenate x and z and let the encoder read both. That's it. Both halves arrive pre-loaded with knowledge; my job is to fine-tune them to cooperate.

There is no way I can sum over all 21M Wikipedia chunks in the likelihood. The retriever gives me the top k under the inner-product score, so every practical marginal has to be a top-k approximation. Once those k documents are selected, I should renormalize their raw scores with a softmax and treat that truncated distribution as p_η(z|x) inside the marginal.

Now the real modeling question, and it's subtle. The marginal Σ_z p(z|x) p(y|x,z) treats the document as a *single* latent for the whole output y. Is that what I want? It says: one document is responsible for generating the entire sequence. That's natural when the answer is a short factoid drawn from one passage. But think about a longer generation — a Jeopardy clue, a multi-fact answer. The first clause might come from one Wikipedia article and the second clause from another. If I force a single z for the whole sequence, the model can never blend content across documents within one answer. So there are genuinely two different latent structures here, and I should build both and see which fits which task.

First structure — one document per sequence. The document is a single latent variable, marginalized once over the whole output:

  p(y|x) ≈ Σ_{z ∈ top-k} p_η(z|x) · p_θ(y|x, z) = Σ_{z ∈ top-k} p_η(z|x) · ∏_i p_θ(y_i | x, z, y_<i).

The inner product is just the generator's usual autoregressive factorization, conditioned on the *same* z at every step. The marginalization is over the whole sequence: generate y under each of the k documents, weight each by its retrieval probability, sum. Call this the per-sequence model.

Second structure — a different document allowed at each token. Now the latent isn't one z for the sequence; it's a fresh draw for every token, and I marginalize *inside* the product:

  p(y|x) ≈ ∏_i Σ_{z ∈ top-k} p_η(z|x) · p_θ(y_i | x, z, y_<i).

Read it carefully: at each step i, I form the next-token distribution by mixing the k documents' predictions, weighted by p_η(z|x); then I take the product over steps. Because the marginalization happens per token, token i can effectively lean on a different document than token i+1. Call this the per-token model. This is the one that can stitch content from several documents into a single answer.

Let me sanity-check the two against each other. They are genuinely different functions of y — in the per-sequence model the document choice is shared across all tokens (the sum is outside the product), while in the per-token model it's resampled each step (the sum is inside the product). They only coincide in a degenerate case: when the output is a single token (N=1), the product collapses and both reduce to the same one-step mixture. That's exactly the one-token classification formulation: if the target is just the class symbol, both models are identical. Good, that's a consistency check that the math is right, not a coincidence.

How do I compute these stably and train them? Everything has to be in log-space, and the retriever scores need normalizing over the retrieved set. So first turn the raw inner-product scores over the top-k documents into a log-distribution: doc_logprobs = log_softmax(doc_scores) — that's log p_η(z|x) restricted to the k retrieved docs. The generator gives me per-token log-probs log p_θ(y_i|x,z,y_<i) = log_softmax over the vocabulary.

For the per-token model, I want log ∏_i Σ_z p_η(z|x) p_θ(y_i|x,z,y_<i). The sum-over-z of products-of-probabilities is a logsumexp in log space: at each token position, take the generator's vocab log-probs for each of the k documents, add the document's log_prob (broadcasting it across vocabulary and positions), and logsumexp over the document axis. That collapses the k documents into one marginal next-token distribution. Then the sequence log-prob is the sum over token positions of the chosen target's log-prob — exactly a standard autoregressive likelihood, except each transition is the document-marginalized mixture. Concretely:

  log p(y_i | x, y_<i) = logsumexp_z [ log p_θ(y_i|x,z,y_<i) + log p_η(z|x) ],

and sum that over i. Because each transition is a proper distribution, this model decodes like an ordinary autoregressive seq2seq: define the marginalized transition probability p'_θ(y_i|x,y_<i) = Σ_z p_η(z|x) p_θ(y_i|x,z,y_<i) and plug it straight into a standard beam search.

For the per-sequence model it's the other order: marginalize *after* summing over tokens. Compute, for each document z, the full-sequence log-prob ll_z = Σ_i log p_θ(y_i|x,z,y_<i), then combine the documents once: log p(y|x) = logsumexp_z [ ll_z + log p_η(z|x) ]. The document log-prob is a per-sequence quantity; add it after the token log-probs have been summed, or equivalently add it once into one token position. If I add log p_η(z|x) at every token, I have changed the model into p_η(z|x)^N times the generator likelihood, which is wrong. The loss in both cases is −log p(y|x) summed over training pairs, optimized with Adam.

Decoding the per-sequence model is harder, and I should face it rather than wave at it. p(y|x) for this model does *not* factor into a per-token transition probability — the sum over z sits outside the whole product, so there's no single distribution to beam over. So I can't run one beam search. Instead I run beam search separately *for each document z*, scoring hypotheses with p_θ(·|x,z); that gives me a pool of candidate sequences Y. The trouble is a hypothesis y that surfaced in document z₁'s beam might not have appeared in document z₂'s beam, yet I need p_θ(y|x,z₂) to compute the marginal score for that same candidate. So for each y and each z where y didn't appear, I run an extra forward pass to score it, multiply by p_η(z|x), and sum across documents. That's thorough decoding. It's expensive when Y is large, so a cheaper approximation is to assume p_θ(y|x,z) ≈ 0 for any y not generated in z's own beam, which skips all the extra forward passes — fast decoding. For short outputs like QA answers, thorough decoding is fine and cheap; for long outputs, fast decoding keeps it tractable.

Now a practical decision matters: do I fine-tune the *document* encoder during training? If I update d(z), then every document vector in my 21M-document index goes stale the instant I take a gradient step, and I'd have to periodically re-embed all of Wikipedia and rebuild the MIPS index mid-training — the heavy asynchronous re-indexing machinery that the masked-LM hybrid needed during pretraining. That's a huge cost. The query encoder q(x) and the generator θ are fully trainable and cheap to update: q(x) can move queries around the fixed document-vector space, and BART can learn how to use whatever text comes back. So the conservative design is to keep the document encoder *and the index fixed*, and only fine-tune the query encoder and BART. If that is enough, I avoid re-indexing entirely; if it is not, I can revisit the cost. The fine-tuned parameters are the BERT-base query encoder (~110M) plus BART-large (~406M), while the document encoder's ~110M parameters are used to build the memory but are not optimized during fine-tuning.

I should also resist adding a null-document branch unless the model needs it. I could append an empty document to the retrieved set and learn a special logit for "nothing useful was retrieved"; that would make the marginal sum over k+1 choices. But it adds another mechanism and another way to route probability mass away from evidence. The generator still sees x directly and can learn not to rely on irrelevant retrieved text, so the core model stays simpler if the marginal only covers retrieved passages. If missing-evidence cases dominate, the empty-document path can be revisited.

Let me make sure the end-to-end loop is coherent. Take a pair (x, y). Encode x, MIPS the fixed index for the top-k documents and their raw scores. Normalize the scores to log p_η(z|x) over those k. For each retrieved z, concatenate (x, z), run BART, get per-token vocab log-probs for the target y. Then either logsumexp-over-z per token then sum over real target tokens, ignoring padding labels (per-token model), or sum real-token log-probs per document then logsumexp-over-z (per-sequence model). Negative of that is the loss; backprop flows into the query encoder (raising/lowering p_η for helpful/unhelpful docs) and into BART (learning to use the retrieved text). No retrieval labels anywhere — the latent-variable marginalization supplies the retrieval gradient. During training I retrieve a small top-k set, and at test time I can choose k on development data for the task and decoding procedure.

The whole thing is general: the only task-specific input is the supervised target, whether that target is an answer string, an abstractive generation, a Jeopardy clue, or a class prediction produced either as a short target sequence or from a classifier head before the document marginalization. And the non-parametric memory is swappable: re-embed a *newer* Wikipedia with the same fixed document encoder, drop in the new index, and the model's knowledge updates without touching a single weight. That's the editability I wanted at the start, falling straight out of "the memory is a separate, raw-text, non-parametric store."

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- Retriever: fixed BERT-base doc encoder + index, trainable BERT-base query encoder ---
class DenseRetriever:
    def __init__(self, query_encoder, doc_index):
        self.q_enc = query_encoder        # trainable
        self.index = doc_index            # FAISS MIPS over fixed DPR document vectors
    def retrieve(self, x, k):
        q = self.q_enc(x)                              # query vector
        docs, doc_scores = self.index.search(q, k)     # top-k docs + raw q.d scores
        return docs, doc_scores                        # scores are UNNORMALIZED q(x)·d(z)

# --- Generator: pretrained BART-large; retrieved doc enters by concatenation with x ---
class Seq2SeqGenerator(nn.Module):
    def __init__(self, bart):
        super().__init__(); self.bart = bart           # pretrained, trainable
    def forward(self, x, z, y):
        # concat_each/repeat_for_docs create one (x, z_j, y) row for each retrieved doc.
        # encode concat(x, z); teacher-force y; return per-token logits (B*K, T, V)
        return self.bart(input_ids=concat_each(x, z), labels=repeat_for_docs(y, z)).logits

# p_eta(z|x) over the retrieved set, in log space
def retriever_logprobs(doc_scores):
    return F.log_softmax(doc_scores, dim=1)             # (batch, n_docs)

def _view_by_doc(seq_logits, doc_scores, n_docs):
    seq_logprobs = F.log_softmax(seq_logits, dim=-1).reshape(
        seq_logits.size(0) // n_docs, n_docs, seq_logits.size(1), seq_logits.size(-1))
    doc_logprobs = F.log_softmax(doc_scores, dim=1)                       # (B, K)
    return seq_logprobs, doc_logprobs

def gather_target(token_logprobs, target, ignore_index=-100):
    mask = target.ne(ignore_index)
    safe_target = target.masked_fill(~mask, 0)
    if token_logprobs.dim() == 4:                                          # (B, K, T, V)
        safe_target = safe_target.unsqueeze(1).expand(-1, token_logprobs.size(1), -1)
        mask = mask.unsqueeze(1).expand(-1, token_logprobs.size(1), -1)
    gathered = token_logprobs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return gathered.masked_fill(~mask, 0.0)

# --- RAG-Token: marginalize the document PER TOKEN (sum inside the product) ---
def ragtoken_nll(seq_logits, target, doc_scores, n_docs):
    seq_logprobs, doc_logprobs = _view_by_doc(seq_logits, doc_scores, n_docs)
    # add doc log-prob to every token's vocab log-probs, then logsumexp over docs
    log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
    token_logprobs = torch.logsumexp(log_prob_sum, dim=1)                 # (B, T, V)
    ll = gather_target(token_logprobs, target).sum(dim=1)                 # sum over tokens -> (B,)
    return -ll

# --- RAG-Sequence: marginalize the document PER SEQUENCE (sum outside the product) ---
def ragsequence_nll(seq_logits, target, doc_scores, n_docs):
    seq_logprobs, doc_logprobs = _view_by_doc(seq_logits, doc_scores, n_docs)
    token_ll = gather_target(seq_logprobs, target).sum(dim=2)             # (B, K)
    return -torch.logsumexp(token_ll + doc_logprobs, dim=1)               # logsumexp over docs

def train(retriever, generator, data, steps, n_docs, mode="sequence"):
    opt = torch.optim.Adam(list(retriever.q_enc.parameters()) +
                           list(generator.parameters()))   # doc encoder + index stay fixed
    for x, y in data:
        docs, doc_scores = retriever.retrieve(x, n_docs)
        seq_logits = generator(x, docs, y)                 # (B*K, T, V)
        if mode == "token":
            loss = ragtoken_nll(seq_logits, y, doc_scores, n_docs).mean()
        else:
            loss = ragsequence_nll(seq_logits, y, doc_scores, n_docs).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

The causal chain: a closed-book parametric model can't revise, can't cite, and hallucinates because its only memory is its weights; so give a general-purpose seq2seq generator a second, non-parametric memory of raw retrievable text, which makes knowledge editable and inspectable; with no labels on which document to use, treat the document as a latent variable and maximize the marginal likelihood of the output, which trains the retriever purely from the generation signal; build the dense retriever and the index from off-the-shelf pre-trained pieces and freeze the document encoder so the index never needs rebuilding; and offer two marginalizations — one document per sequence for short focused answers, a different document per token for blending evidence — that collapse to the same distribution whenever the target has one generated token.
