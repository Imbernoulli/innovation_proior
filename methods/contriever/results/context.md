# Context: Unsupervised Dense Retrieval with Contrastive Learning

## Research question

A retriever takes a query and a large collection of documents and must return the documents
relevant to the query. The dominant production systems are lexical: TF-IDF and BM25 score a
document by how many query terms it contains, weighted by term specificity. They need no training
data and they are shockingly hard to beat out of the box. But they match on surface tokens, so
they suffer from the *lexical gap*: a query and a relevant document that say the same thing with
different words ("car" vs. "automobile", or an English query against an Arabic document) score near
zero.

Neural bi-encoders fix the lexical gap by mapping query and document into a shared dense vector
space and scoring by dot product, so semantic matches score high even with no shared tokens. But
they pay for it with supervision: to learn that space they need large sets of human-labeled
(query, relevant-document) pairs. Such labels exist for a handful of English benchmarks and almost
nowhere else — collecting them means matching a query against a collection of millions of
documents by hand. The painful empirical fact: a dense retriever trained on one large labeled set
and applied zero-shot to a new domain is *outperformed by BM25*, which used no labels at all. And
in non-English languages there are essentially no large labeled retrieval sets, so the supervised
recipe cannot even be run.

So the precise problem: **train a dense retriever with no labeled query–document pairs, and reach
or beat the unsupervised lexical baseline (BM25), across domains and languages.** A solution must
produce a single embedding per text (so the document index can be pre-computed and searched with
fast approximate nearest neighbors), generalize zero-shot to unseen domains, and extend to
languages where no retrieval supervision exists.

## Background

**The lexical gap and term-frequency retrieval.** TF-IDF weights a term by its frequency in a
document times its inverse document frequency (term specificity). BM25 extends this with document
length normalization and term-frequency saturation. These represent each document as a sparse
bag-of-words vector over the vocabulary; relevance is a weighted count of shared terms. They are
strong, training-free, and the standard yardstick — but blind to synonymy and to cross-script
matching.

**Bi-encoders vs. cross-encoders.** Two neural architectures address the lexical gap. A
*cross-encoder* feeds the concatenated (query, document) pair through one network and reads off a
relevance score; it captures fine-grained token interactions and is very accurate, but it must
re-run the network for every (query, document) pair, so it cannot search a large collection — it is
only usable to *re-rank* a small candidate set. A *bi-encoder* (the deep-structured-semantic-model
line, Huang et al. 2013) encodes query and document *independently* into single vectors and scores
by dot product. The document vectors can be computed once, indexed, and searched in sublinear time
with a nearest-neighbor library such as FAISS — so bi-encoders are the only neural option that
scales to retrieval. Their weakness is exactly the single-vector bottleneck: no token-level
interaction.

**Supervised dense retrieval.** DPR (Karpukhin et al. 2020) is the canonical supervised
bi-encoder: initialize from BERT, train discriminatively on (question, gold-passage) pairs with
in-batch negatives plus BM25-mined hard negatives. ANCE (Xiong et al. 2020) improves it by mining
hard negatives from the model itself during training. Both need large labeled sets and both
transfer poorly to new domains.

**Self-supervised objectives for retrieval.** The Inverse Cloze Task (ICT, Lee et al. 2019)
manufactures pseudo (query, document) pairs from raw text: sample a sentence-length span as the
"query" and use its surrounding context as the "document". Pre-training a bi-encoder this way helps,
but as a zero-shot retriever it still trails BM25. The masked-salient-spans variant (Guu et al.
2020, REALM) uses named-entity annotations and so is not fully unsupervised.

**Contrastive learning (the import from vision).** In computer vision, instance discrimination
(Wu et al. 2018) trains a network so that two augmented "views" of the same image are close and
views of different images are far apart, using the InfoNCE loss — a softmax that must pick the one
positive out of many negatives. Two frameworks supply the negatives. SimCLR (Chen et al. 2020)
uses the other examples in the same large batch as negatives, and backpropagates through both
views; it needs very large batches (thousands) to provide enough negatives. MoCo (He et al. 2020)
instead keeps a *queue* of key vectors from recent batches and encodes keys with a separate
*momentum encoder* (a second copy of the network updated without gradient). These methods were
shown to learn features well suited to retrieval (Caron et al. 2021).

**Motivating observation.** ICT's self-supervised pre-training gains were modest: as a zero-shot
retriever it still trailed BM25. Meanwhile, contrastive learning in vision had advanced
substantially over the same period — larger negative pools, stronger augmentations — though those
advances had been demonstrated on images rather than on text retrieval.

## Baselines

- **BM25** (Robertson et al. 1995). Sparse term-frequency relevance with IDF weighting, TF
  saturation, and length normalization. Unsupervised, strong, the baseline to beat. Gap: lexical
  matching only — no synonymy, no cross-script retrieval.
- **DPR** (Karpukhin et al. 2020). Supervised BERT bi-encoder, separate query/document encoders,
  in-batch + BM25 hard negatives. Gap: needs large labeled sets; transfers poorly zero-shot.
- **ANCE** (Xiong et al. 2020). DPR plus self-mined hard negatives refreshed during training. Gap:
  same supervision requirement; even more training machinery.
- **ICT pre-training** (Lee et al. 2019). Self-supervised pseudo pairs built by taking a sentence
  span as the "query" and its surrounding complement as the "document". Gap: zero-shot retrieval
  still below BM25.
- **REALM / masked-salient-spans** (Guu et al. 2020). Uses entity annotations, so not fully
  unsupervised; still below BM25 zero-shot.
- **SimCSE** (Gao et al. 2021), as an unsupervised dense baseline; **SBERT** (Reimers & Gurevych
  2019), a Siamese sentence encoder that requires *aligned* sentence pairs to form positives.

## Evaluation settings

- **BEIR** (Thakur et al. 2021): a heterogeneous suite of retrieval datasets across domains (fact
  checking, citation prediction, QA, argument retrieval, scientific/biomedical) explicitly built to
  measure *zero-shot* transfer. Metrics: nDCG@10 (ranking quality of the top results, for
  human-facing search) and Recall@100 (whether the gold document is anywhere in the top 100, the
  relevant metric when a downstream reader will consume many passages).
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
