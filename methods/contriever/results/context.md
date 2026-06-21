# Context: Unsupervised Dense Retrieval with Contrastive Learning

## Research question

A retriever takes a query and a large collection of documents and must return the documents
relevant to the query. The dominant production systems are lexical: TF-IDF and BM25 score a
document by how many query terms it contains, weighted by term specificity. They need no training
data. They match on surface tokens, so a query and a relevant document that say the same thing with
different words ("car" vs. "automobile", or an English query against an Arabic document) share no
terms.

Neural bi-encoders map query and document into a shared dense vector space and score by dot
product, so semantic matches can score high even with no shared tokens. To learn that space they
are trained on sets of human-labeled (query, relevant-document) pairs. Such labels exist for a
handful of English benchmarks; collecting them means matching a query against a collection of
millions of documents by hand, and in non-English languages there are essentially no large labeled
retrieval sets.

So the setting: **train a dense retriever from raw, unlabeled text — with no labeled
query–document pairs — and evaluate it against the unsupervised lexical baseline (BM25), across
domains and languages.** The retriever should produce a single embedding per text (so the document
index can be pre-computed and searched with fast approximate nearest neighbors), be applicable
zero-shot to unseen domains, and extend to languages where no retrieval supervision exists.

## Background

**The lexical gap and term-frequency retrieval.** TF-IDF weights a term by its frequency in a
document times its inverse document frequency (term specificity). BM25 extends this with document
length normalization and term-frequency saturation. These represent each document as a sparse
bag-of-words vector over the vocabulary; relevance is a weighted count of shared terms. They are
strong, training-free, and the standard yardstick; relevance is measured by shared surface terms.

**Bi-encoders vs. cross-encoders.** Two neural architectures map text into relevance scores. A
*cross-encoder* feeds the concatenated (query, document) pair through one network and reads off a
relevance score; it captures fine-grained token interactions and re-runs the network for every
(query, document) pair, so it is used to *re-rank* a small candidate set. A *bi-encoder* (the
deep-structured-semantic-model line, Huang et al. 2013) encodes query and document *independently*
into single vectors and scores by dot product. The document vectors can be computed once, indexed,
and searched in sublinear time with a nearest-neighbor library such as FAISS, so bi-encoders scale
to retrieval over large collections.

**Supervised dense retrieval.** DPR (Karpukhin et al. 2020) is the canonical supervised
bi-encoder: initialize from BERT, train discriminatively on (question, gold-passage) pairs with
in-batch negatives plus BM25-mined hard negatives. ANCE (Xiong et al. 2020) mines hard negatives
from the model itself during training. Both are trained on large labeled sets.

**Self-supervised objectives for retrieval.** The Inverse Cloze Task (ICT, Lee et al. 2019)
manufactures pseudo (query, document) pairs from raw text: sample a sentence-length span as the
"query" and use its surrounding context as the "document". A bi-encoder pre-trained this way is
used as a retriever. The masked-salient-spans variant (Guu et al. 2020, REALM) uses named-entity
annotations to select spans.

**Contrastive learning (the import from vision).** In computer vision, instance discrimination
(Wu et al. 2018) trains a network so that two augmented "views" of the same image are close and
views of different images are far apart, using the InfoNCE loss — a softmax that must pick the one
positive out of many negatives. Two frameworks supply the negatives. SimCLR (Chen et al. 2020)
uses the other examples in the same large batch as negatives, and backpropagates through both
views; the number of negatives is the batch size. MoCo (He et al. 2020) instead keeps a *queue* of
key vectors from recent batches and encodes keys with a separate *momentum encoder* (a second copy
of the network updated without gradient), decoupling the negative count from the batch size. These
methods learn features used for retrieval (Caron et al. 2021).

## Baselines

- **BM25** (Robertson et al. 1995). Sparse term-frequency relevance with IDF weighting, TF
  saturation, and length normalization. Unsupervised, strong, the baseline to beat. Matches on
  shared surface terms.
- **DPR** (Karpukhin et al. 2020). Supervised BERT bi-encoder, separate query/document encoders,
  in-batch + BM25 hard negatives.
- **ANCE** (Xiong et al. 2020). DPR plus self-mined hard negatives refreshed during training.
- **ICT pre-training** (Lee et al. 2019). Self-supervised pseudo pairs built by taking a sentence
  span as the "query" and its surrounding complement as the "document".
- **REALM / masked-salient-spans** (Guu et al. 2020). Uses entity annotations to select masked
  spans.
- **SimCSE** (Gao et al. 2021), as an unsupervised dense baseline; **SBERT** (Reimers & Gurevych
  2019), a Siamese sentence encoder that uses *aligned* sentence pairs to form positives.

## Evaluation settings

- **BEIR** (Thakur et al. 2021): a heterogeneous suite of retrieval datasets across domains (fact
  checking, citation prediction, QA, argument retrieval, scientific/biomedical) built to measure
  *zero-shot* transfer. Metrics: nDCG@10 (ranking quality of the top results, for human-facing
  search) and Recall@100 (whether the gold document is anywhere in the top 100, the relevant metric
  when a downstream reader will consume many passages).
- **Open-domain QA retrieval**: NaturalQuestions and TriviaQA (open versions of Lee et al. 2019),
  retrieving from a Wikipedia dump; metric is top-k retrieval accuracy (R@5/20/100 — does any of the
  top-k passages contain the answer).
- **MS MARCO** (Bajaj et al. 2016): large labeled passage-retrieval set, the standard supervised
  fine-tuning target.
- **Multilingual retrieval**: Mr. TyDi, reporting MRR@100 and Recall@100; queries and documents in
  many languages, plus cross-lingual settings (e.g. retrieving English documents from Arabic
  queries) where lexical matching is structurally impossible.
- **Indexing**: document embeddings are pre-computed and searched with FAISS approximate
  nearest-neighbor search.
- **Pre-training corpora**: raw, unlabeled text — English Wikipedia and CCNet (a large
  Common-Crawl-derived multilingual corpus, Wenzek et al. 2020).

## Code framework

The primitives that already exist: a pre-trained Transformer encoder (BERT base, via the
`transformers` library), the InfoNCE / cross-entropy loss, AdamW, and FAISS for indexing. The slot
the method will fill is everything about *how to make a positive pair from raw text* and *how to
supply many negatives without labels*.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import transformers

# --- Encoder: a pre-trained Transformer turned into a single-vector embedder ---
class Encoder(transformers.BertModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        last = out["last_hidden_state"]
        emb = pool(last, attention_mask)   # TODO: how to collapse a sequence to one vector?
        return emb

def pool(last_hidden, attention_mask):
    # TODO: which pooling turns token states into one retrieval embedding?
    pass

# --- Turning a raw document into a (query, key) positive pair, with NO labels ---
def build_positive_pair(text_tokens):
    # TODO: the core question — how do we synthesize a positive pair
    #       from a single unlabeled document so the learned space is good for retrieval?
    pass

# --- Supplying many negatives without labels ---
class ContrastiveTrainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temperature = opt.temperature
        self.encoder = Encoder.from_pretrained(opt.model_id)
        # TODO: how do we get a large pool of negatives without huge batches?

    def forward(self, q_tokens, q_mask, k_tokens, k_mask):
        # TODO: encode query and key, score against positive + negatives,
        #       and apply the contrastive loss
        pass

# --- Training loop (already standard) ---
def train(trainer, loader, opt):
    optim = torch.optim.AdamW(trainer.parameters(), lr=opt.lr)
    for batch in loader:
        loss = trainer(**batch)
        loss.backward(); optim.step(); optim.zero_grad()
```
</content>
</invoke>
