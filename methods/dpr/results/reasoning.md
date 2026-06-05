Let me start from the number that keeps bothering me. A good reading-comprehension model, handed the gold paragraph, scores above 80% exact match on SQuAD. Take that same reader and drop it into open-domain QA — now it has to answer against all of Wikipedia, relying on a retriever to hand it the paragraph — and exact match falls below 40%. The reader didn't get worse. So whatever is being lost, it's lost upstream, in the retriever. The whole open-domain answer is capped by one question: is the passage that contains the answer even in the top-k that the reader gets to read? If it isn't, the reader has no chance, no matter how good it is. So I'm going to stop thinking about reading and put everything into retrieval.

What does retrieval mean here concretely. The corpus is ~21 million 100-word passages from Wikipedia. A question comes in; I have to return maybe 20 to 100 passages, and I want the answer-bearing one to be among them. And it has to be fast — real-time fast — because this is meant to serve users. That speed requirement is not a footnote, it's a structural constraint that's going to decide the entire design. With 21 million passages I cannot, at query time, run any neural network that looks at the (question, passage) pair jointly, because that's 21 million forward passes per question. So whatever scoring function I pick, it has to let me do all the heavy work on the passage side *offline*, once, and reduce the online step to something an index can answer in milliseconds.

The incumbent is BM25, and I should be honest about why it's hard to beat. It represents the question and each passage as sparse, weighted bags of terms, and scores relevance by weighted term overlap through an inverted index. It's fast, needs no training, and it's genuinely strong whenever the answer hinges on a rare, selective keyword — a name like "Thoros of Myr" gets matched exactly and precisely. But it's semantically blind. Take "Who is the bad guy in Lord of the Rings?" and the passage "Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy." That passage answers the question. But "bad guy" and "villain" share zero tokens, so a term-matching system will not connect them. Synonyms, paraphrases — anything where the meaning matches but the surface form doesn't — falls straight through BM25's fingers. That gap is exactly where a learned, meaning-based representation should win.

So the obvious idea: embed the question and the passage as dense vectors in some low-dimensional space, learned so that semantically matching pairs land near each other. "bad guy" and "villain" don't share tokens but they should share a region of vector space. This isn't a new dream — dense semantic representations go back to Latent Semantic Analysis, and people have trained dense encoders for web search and ad matching for years. But there's a wall, and everybody knows it: dense retrieval has *never* been shown to beat BM25 on open-domain QA. The folklore reason is that you need a huge number of labeled question–context pairs to learn a good dense space, and we don't have that many. Only one approach has crossed the line so far, and it crossed it by leaning on heavy self-supervised pretraining — an Inverse Cloze Task, where you mask a sentence out of a passage and train the encoder to recover which block it came from, as a stand-in for the question→passage matching signal. That gets you above BM25, but it bothers me on two counts. First, that pretraining is expensive, and I'm not convinced a random held-out sentence is really a good surrogate for the kind of question a person actually asks. Second — and this is the part that nags more — in that scheme the *passage* encoder is frozen after pretraining; only the question side gets fine-tuned on real (question, answer) data. So the passage vectors are stuck at whatever the surrogate task produced. If the passage representation is suboptimal, the whole retrieval is suboptimal, and no amount of question-side tuning fully fixes it.

That points me at the actual hypothesis I want to test. Is the expensive pretraining really necessary? Or is the field's belief — "you need ICT, you need millions of pairs" — just an artifact of nobody having gotten the *training setup* right? We do have, in every QA dataset, pairs of questions and the passages (or answers) that go with them. Maybe a few thousand to tens of thousands per dataset. The question I want to answer is sharp: can I train a dense retriever that beats BM25 using *only* those question–passage pairs, with no extra pretraining at all, just by fine-tuning a standard pretrained encoder the right way? If yes, the whole "you need ICT" story is wrong, and the real lever was always the training recipe.

Let me fix the architecture first, because the speed constraint already dictates most of it. I need the passage representation to be computable offline and independent of the question — otherwise I can't precompute and index. So the question and passage must be encoded by *separate* towers, each producing one vector, and the only thing that couples them at query time is a cheap vector operation. That's a dual-encoder. Two encoders, E_Q for questions and E_P for passages. For the encoder itself, BERT is sitting right there — a strong pretrained Transformer — and the natural fixed-size summary is the `[CLS]` token representation, 768-dimensional for BERT-base. So E_Q(q) and E_P(p) are each a BERT, take `[CLS]`, get a 768-d vector.

Now the scoring function, and here the constraint bites hardest. I need sim(q, p) to be *decomposable* — it has to factor into "something about q" and "something about p" so that the p-part is precomputable. The most expressive thing would be to let the question and passage attend to each other through many layers of cross-attention. That would clearly score relevance more accurately. But cross-attention is non-decomposable by construction: the passage representation depends on the question, so I'd have to recompute it for every query against every passage. 21 million forward passes. Dead on arrival for first-stage retrieval. I'll hold that thought — a cross-attention scorer is too expensive to *retrieve* with, but it might be perfect later as a reranker over a tiny candidate set, where 100 forward passes is fine. For retrieval, though, I need a decomposable score.

So what decomposable scores are on the table. The simplest is the inner product: sim(q, p) = E_Q(q)ᵀ E_P(p). There's cosine, which is just inner product on unit-normalized vectors. There's negative L2 distance. And the nice fact is these are all close cousins — cosine is inner product for unit vectors, and Mahalanobis distance is L2 in a transformed space; the relationships between inner product, cosine, and L2 are well studied. More importantly, inner product is exactly the thing that maximum inner-product search is built to serve: given the precomputed passage vectors, finding the top-k by inner product is a MIPS query, and MIPS over billions of vectors is a solved systems problem — FAISS and friends do it sub-linearly with in-memory indices. So inner product isn't just simple, it lines up perfectly with the serving infrastructure I'll need. My instinct is to pick the plainest one, get the *training* right, and only complicate the similarity if an ablation shows it matters. (Spoiler to myself: when I do later sweep dot product vs. L2 vs. cosine, they come out comparable, which is a relief — it means the lever really is the encoder, not the metric, so I keep the simplest, the dot product.)

Good. Architecture and score settled: two BERTs, `[CLS]` vectors, inner-product similarity, FAISS index over the precomputed passage vectors. Inference is then: encode all 21M passages offline with E_P, build the index; at query time encode q to v_q = E_Q(q), MIPS the index for the top-k. The entire question of whether this beats BM25 now collapses to one thing — the training objective. Can I shape the vector space, from question–passage pairs alone, so that relevant pairs have higher inner product than irrelevant ones?

This is a metric-learning problem. I want a space where the right passage is closer (higher inner product) to the question than the wrong ones. So for a question q with its known positive passage p⁺, and some set of negative passages p⁻₁,…,p⁻ₙ, I want sim(q, p⁺) to come out on top. The clean way to say "p⁺ should beat all the negatives" probabilistically is a softmax over the candidates and a negative-log-likelihood on the positive:

  L(q, p⁺, p⁻₁,…,p⁻ₙ) = −log [ exp(sim(q, p⁺)) / ( exp(sim(q, p⁺)) + Σⱼ exp(sim(q, p⁻ⱼ)) ) ].

This is just cross-entropy treating retrieval as "pick the positive out of {positive} ∪ {negatives}." Minimizing it pushes the positive's inner product up relative to the negatives'. Simple, standard, differentiable. Fine.

But now the question that the prior work said was "often overlooked but could be decisive," and I think they're right: where do the negatives come from? The positives are easy — the dataset gives them, or I find a passage containing the answer. But "irrelevant passage" means essentially the entire 21-million-passage corpus minus a few. I have to *choose* which negatives go in the denominator, and that choice shapes what the encoder learns. Let me think through the options.

Option one: random passages from the corpus. Easy to get. But a uniformly random Wikipedia passage is, almost always, wildly off-topic — different subject entirely — so the encoder learns to separate the positive from garbage, which is easy and doesn't teach it the fine distinctions that actually matter at retrieval time. Option two: hard negatives — passages returned high by BM25 that match many question tokens but don't contain the answer. These are exactly the confusable cases: lexically similar, topically close, but wrong. Training against those should force the encoder to learn the semantic difference rather than just topicality. Option three: "gold" negatives — the positive passages that belong to *other* questions in the training set. Those are real, well-formed passages, each genuinely relevant to *some* question, just not this one.

Here's where I get a useful idea, and it comes from a computational worry rather than a statistical one. If I want lots of negatives per question — and more negatives in that softmax denominator should give a richer, harder contrastive signal — then naively I have to encode all those extra negative passages, for every question, which is a lot of wasted forward passes. But look at what I'm already computing. In a mini-batch I have B questions, each with its own positive passage. I've already run E_P on all B of those positive passages to get their vectors. For question i, the positive is passage i — but passages 1…B excluding i are all perfectly good "gold negatives": real passages, relevant to *their* questions, irrelevant to question i. They cost nothing extra. I've already encoded them.

So let me make that concrete as matrices. Stack the B question vectors into a matrix **Q** ∈ ℝ^{B×d} and the B passage vectors into **P** ∈ ℝ^{B×d}. Then **S** = **Q P**ᵀ is a B×B matrix where S_{ij} = sim(qᵢ, pⱼ) — every question scored against every passage in the batch, in one matrix multiply. Row i is question i's scores against all B passages. The diagonal entry S_{ii} is the positive; the B−1 off-diagonal entries in that row are negatives. So from a single batch I get B training instances, each a (B−1)-way contrastive problem, and the loss is just a row-wise softmax cross-entropy where the target for row i is column i. Effectively I'm training on B² question/passage pairs for the price of encoding B questions and B passages. This in-batch-negatives trick isn't something I invented from nothing — it's been used in the full-batch setting and more recently for mini-batches in dual-encoder training — but it fits this problem beautifully, because the bigger I make the batch, the more negatives every question sees for free, and I'd expect accuracy to climb with batch size. That's a falsifiable prediction I can check: scale B and watch top-k accuracy. (When I do, it does climb — which confirms the negatives, not just the architecture, are doing real work.)

Now, are in-batch gold negatives *enough*? They're real passages but they're essentially random with respect to topic — passage j is about whatever question j was about, probably unrelated to question i. So the in-batch negatives behave a lot like random negatives: they teach topical separation but not the razor's-edge distinctions. I'm missing the hard, BM25-confusable negatives — the passages that look right and aren't. So combine them: keep the in-batch gold negatives (free, plentiful), and add a small number of explicit BM25 hard negatives. Concretely, for each question I pull one passage that BM25 ranks highly, that matches lots of question tokens, but that does *not* contain the answer string. And here's another free lunch: that hard negative, encoded for question i, can also serve as a negative for every *other* question in the batch — so one BM25 negative per question augments the whole B×(B+B) score matrix. How many hard negatives? My guess is the first one carries almost all the value — it converts the easy "separate from random junk" problem into "separate from a topically-matched decoy," which is the real skill. A second one probably adds little, because once the encoder can beat one strong decoy it's mostly learned the distinction. So I'll bet on: in-batch gold negatives plus exactly one BM25 hard negative per question. (Sweeping this later: one BM25 negative helps a lot, two doesn't help further. Good — the bet holds.)

Let me also sanity-check the loss choice against the obvious alternative, the triplet/margin loss, which compares one positive and one negative directly with a margin. Triplet loss only looks at pairs, so it doesn't naturally exploit "one positive against many negatives at once," and it needs margin tuning. The softmax NLL handles an arbitrary number of negatives in one shot and is exactly the in-batch matrix story. I'll keep NLL as the default and treat L2 / triplet / cosine as ablations to confirm they don't beat it. (They come out comparable or worse — so the simplest, dot-product NLL, stays.)

One detail on the data side that's easy to get wrong: a passage is a *disjoint* 100-word block, and I prepend the article title with a `[SEP]` so the encoder knows the topic context — titles carry real disambiguating signal ("the 8th Dalai Lama" vs. some other Dalai Lama). And for datasets that only give (question, answer) with no gold paragraph, I manufacture a positive by running BM25 with the answer and taking the top passage that actually contains the answer string; if none of the top 100 does, I drop the question rather than train on a false positive. That's distant supervision, and it's slightly noisier than a true gold context, but the gap turns out small.

Now stand back and check the whole thing is internally consistent and actually serves at scale. Training: BERT-base for both towers, `[CLS]` vectors, batch of B questions each with a positive and one BM25 negative, S = QPᵀ over the whole batch, row-wise softmax NLL targeting the diagonal. Optimize with Adam at a small learning rate (~1e-5) with linear warmup and a little dropout — standard BERT fine-tuning, nothing exotic, which is the point: the claim is that the *recipe* is simple. Inference: run E_P over all 21M passages once (expensive but embarrassingly parallel — a handful of GPU-hours on a few machines), dump the vectors into a FAISS index. At query time, one BERT forward for E_Q(q), then a MIPS lookup for the top-k. With an HNSW index this answers hundreds of questions per second — fast enough to be real.

So I've gone from "dense retrieval needs heavy pretraining" to a dual-encoder fine-tuned only on question–passage pairs, where the entire trick that makes it work is the negatives: free in-batch gold negatives scaled up by batch size, plus one BM25 hard negative to teach the hard distinctions. If this beats BM25 by a wide margin in top-k accuracy, then the field's belief about needing ICT pretraining was wrong, and the lever was always the training setup. And the second thing I want to confirm is the premise I started from — that higher retrieval accuracy actually buys higher end-to-end answer accuracy — so I need a reader on top.

The reader is the second stage, and now I'm allowed to be expensive because I'm reading only the top-k (≤100) passages, not millions. This is exactly where the cross-attention model I shelved earlier earns its place. Encode each retrieved passage *with* the question through a full BERT — passage i gives token representations **P**ᵢ ∈ ℝ^{L×h}. I need three things from the reader: where the answer span starts, where it ends, and which of the k passages to trust. So put three small linear heads on top. For start and end positions, project each token's representation to a scalar logit and softmax over the L token positions:

  P_start,i(s) = softmax(**P**ᵢ **w**_start)_s,
  P_end,i(t)  = softmax(**P**ᵢ **w**_end)_t,

with learnable vectors **w**_start, **w**_end ∈ ℝ^h. The span score for words s..t in passage i is the product P_start,i(s) · P_end,i(t). For passage selection, collect the `[CLS]` representations of all k passages into **P̂** = [P₁^[CLS], …, P_k^[CLS]] ∈ ℝ^{h×k} and softmax a selection logit across the k passages:

  P_selected(i) = softmax(**P̂**ᵀ **w**_selected)_i,

with **w**_selected ∈ ℝ^h. The reader's selection head acts as a reranker — it's the cross-attention scorer I couldn't afford for retrieval, now affordable because k is tiny. The final answer is the best span from the passage with the highest selection score. Train it by sampling, per question, one positive passage and m̃−1 negatives from the retriever's top 100 (I'll use m̃ = 24), maximizing the marginal log-likelihood of the correct answer spans in the positive passage (the answer string can occur several times, so marginalize over occurrences) plus the log-likelihood that the positive passage is the selected one. Because all k passage representations are independent given the question, the whole top-100 set fits in one batch on a single 32GB GPU, so reading 100 passages costs about the same latency as reading one — around 20ms. That keeps the two-stage system real-time end to end.

Now the code. Two encoders that are just BERT pooled at `[CLS]`; a dot-product similarity that's literally a matmul of the batch's question and passage matrices; a loss that's row-wise log-softmax NLL with the diagonal as the target — that single matmul *is* the in-batch-negatives mechanism; an offline FAISS index for serving; and the extractive reader with three linear heads.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel

# --- Encoder: BERT pooled at [CLS] -> one d=768 vector. Same class, two instances. ---
class BertEncoder(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)

    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        # take the [CLS] position (index 0) as the fixed-size representation
        return out.last_hidden_state[:, 0, :]            # (B, d)

# --- Decomposable similarity: just the dot product, so passages precompute. ---
def dot_product_scores(q_vecs, p_vecs):                  # (Bq, d), (Bp, d)
    return torch.matmul(q_vecs, p_vecs.transpose(0, 1))  # (Bq, Bp) = Q P^T

# --- In-batch-negatives NLL. The B x B(+hard) score matrix IS the negatives trick:
#     row i's positive is column i; every other column is a free negative. ---
def biencoder_nll(q_vecs, p_vecs, positive_idx):
    scores = dot_product_scores(q_vecs, p_vecs)          # (B, B [+ hard negatives])
    log_p  = F.log_softmax(scores, dim=1)                # softmax over candidates per question
    return F.nll_loss(log_p, positive_idx)               # -log of the positive column

def train_step(eq, ep, q_ids, q_mask, p_ids, p_mask, positive_idx, opt):
    # p_ids packs, for each question, its positive passage then (optionally) one BM25
    # hard negative; positives sit at the diagonal positions -> positive_idx.
    q = eq(q_ids, q_mask)                                # (B, d)
    p = ep(p_ids, p_mask)                                # (B + #hard, d)
    loss = biencoder_nll(q, p, positive_idx)
    opt.zero_grad(); loss.backward(); opt.step()         # Adam, lr~1e-5, linear warmup
    return loss.item()

# --- Offline indexing + online MIPS retrieval ---
import faiss
class PassageIndex:
    def __init__(self, dim): self.index = faiss.IndexHNSWFlat(dim, 512)  # inner-product MIPS
    def add(self, passage_vecs): self.index.add(passage_vecs)            # all 21M, encoded offline
    def search(self, q_vec, k): return self.index.search(q_vec, k)       # top-k by dot product

# --- Extractive reader over the top-k retrieved passages (the affordable cross-attention) ---
class Reader(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        h = self.bert.config.hidden_size
        self.qa_outputs   = nn.Linear(h, 2)              # w_start, w_end as one (h x 2)
        self.qa_classifier = nn.Linear(h, 1)             # w_selected

    def forward(self, passage_ids, passage_mask):        # (k, L)
        seq = self.bert(input_ids=passage_ids,
                        attention_mask=passage_mask).last_hidden_state   # (k, L, h)
        start_logits, end_logits = self.qa_outputs(seq).split(1, dim=-1) # each (k, L, 1)
        select_logits = self.qa_classifier(seq[:, 0, :])                 # (k, 1) from [CLS]
        return start_logits.squeeze(-1), end_logits.squeeze(-1), select_logits.squeeze(-1)
        # span score = softmax(start)_s * softmax(end)_t ; pick passage by softmax(select)
```

The causal chain: the open-domain accuracy collapse traces entirely to retrieval, so I attack retrieval; real-time serving over 21M passages forces a decomposable score, which forces a dual-encoder with an inner-product similarity served by MIPS; the only remaining degree of freedom is the training objective, and there the decisive move is the negatives — a row-wise softmax NLL over a B×B in-batch score matrix turns every other passage in the batch into a free gold negative (so accuracy scales with batch size), and one BM25 hard negative per question supplies the topically-confusable decoy that random negatives never provide. Fine-tuning two BERTs this way, on question–passage pairs alone with no special pretraining, is enough to beat BM25 by a wide margin; a cross-attention reader over the cheap top-k then turns that retrieval gain into end-to-end answer accuracy.
