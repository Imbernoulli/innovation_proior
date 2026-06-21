# Context

## Research question

Large pre-trained language models store world knowledge implicitly in their weights — you can probe them like a knowledge base and they recall facts they were never explicitly taught to recall. That knowledge lives entirely *inside the parameters*, so the model cannot easily expand or revise what it knows, cannot point at why it produced an answer, and can produce confident but wrong text when its parametric memory is incomplete. These behaviors are most apparent on knowledge-intensive tasks — open-domain QA, fact verification, fact completion, long-form generation — those a person could not do without consulting an external source.

The question, then: can a *generative* model — a sequence-to-sequence model that produces open-ended text — be given access to an explicit, external, *non-parametric* memory it can read from at inference time, and can the access mechanism and the generator be made to work together on a range of knowledge-intensive tasks?

## Background

**Parametric knowledge in pre-trained models.** A pre-trained seq2seq or LM acts as an implicit, parameterized knowledge base. Fine-tuning adapts the weights to a task but leaves all knowledge inside the parameters. "Knowledge-intensive" tasks — those a person could not do without consulting an external source — are the natural arena for studying these models.

**Hybrid parametric + non-parametric memory.** The alternative is to pair the parametric model with a *non-parametric* memory: an external store of raw text that the model retrieves from. Two recent systems showed this works for masked-LM-based *extractive* open-domain QA. ORQA (Lee et al., 2019) introduced a latent-variable retriever pretrained with the Inverse Cloze Task and fine-tuned jointly with a reader. REALM (Guu et al., 2020) made the retriever fully differentiable and trained it end-to-end with the masked-LM objective during pretraining, even updating the *document* encoder — which forces periodic re-indexing of the whole corpus (the document vectors drift, so the MIPS index must be rebuilt). Both were demonstrated on *extractive* QA (selecting answer spans).

**Dense retrieval is now a reusable component.** Dense Passage Retrieval (DPR; Karpukhin et al., 2020) showed a bi-encoder — a BERT-base query encoder and a BERT-base document encoder, scoring by inner product p(z|x) ∝ exp(d(z)ᵀq(x)) — beats BM25 on open-domain QA when trained on question–passage pairs. Retrieval over a precomputed document index is a Maximum Inner Product Search (MIPS) problem, solvable approximately in sub-linear time (FAISS; Johnson et al., 2017, with HNSW). DPR provides a *pre-trained* retriever and a way to build a dense Wikipedia index off the shelf.

**General-purpose pre-trained generators.** On the generation side, BART (Lewis et al., 2019) and T5 (Raffel et al., 2019) are pre-trained encoder–decoder Transformers (BART via a denoising objective) that fine-tune to strong performance across both discriminative and generative tasks. BART-large is ~400M parameters. These are the natural "parametric memory" to augment — a single model fine-tunable on any seq2seq task.

**Memory-augmented neural networks.** Older work attached external differentiable memory to networks trained from scratch for one task: memory networks (Weston et al., 2015; Sukhbaatar et al., 2015), stack-augmented nets, large memory layers. Those memories are *learned distributed representations*, trained per task; a memory made of *raw retrievable text* is human-readable (interpretable) and human-writable (editable to update knowledge).

**No supervision on which document to use.** In the target setting the training data is input/output pairs (x, y) only — a question and its answer, a claim and its label — with no annotation of which document *should* have been retrieved.

## Baselines

- **Closed-book parametric models (T5, BART fine-tuned alone).** A single pre-trained seq2seq model fine-tuned to answer/generate from parameters only. Knowledge is stored in weights; the strongest closed-book QA baselines rely on very large parameter counts (e.g. T5-11B).

- **Extractive retrieve-then-read (DPR + reader).** Retrieve passages with DPR, then a reader *extracts* an answer span from a retrieved passage.

- **ORQA / REALM.** Latent-retrieval models combining a masked LM with a differentiable retriever, trained end-to-end. REALM updates the document encoder, requiring expensive asynchronous re-indexing during training. Demonstrated on *extractive* open-domain QA.

- **Task-specific retrieval-augmented systems.** Retrieval has separately been applied to open-domain QA, fact checking, dialogue, translation, language modeling, Wikipedia generation — each with a bespoke architecture and optimization (search, RL, or latent variables).

## Evaluation settings

- **Non-parametric memory.** A single December 2018 Wikipedia dump, each article split into disjoint **100-word chunks**, totaling **21M documents**. Each document is embedded by the (DPR) document encoder; a single FAISS MIPS index with HNSW is built for fast approximate retrieval. The index sits on CPU (~100 GB, compressible to ~36 GB); vectors are 768-d (21M × 768 ≈ 15.3B values).
- **Tasks / datasets.** Open-domain QA: Natural Questions, TriviaQA, WebQuestions, CuratedTREC (extractive answers, but the system may generate them); same train/dev/test splits as prior open-domain work. Abstractive generation: MS-MARCO (answer generation) and Jeopardy question generation. Classification: FEVER fact verification (Supported / Refuted / Not Enough Info), where a class label can be treated as the supervised target like any other output.
- **Metrics.** Exact match for open-domain QA; generation quality (factuality, specificity, diversity; standard generation metrics like BLEU/ROUGE-style scores and Q-BLEU for Jeopardy) for the abstractive tasks; classification accuracy for FEVER.
- **Decoding / retrieval depth.** During training, each query retrieves the top k documents, with k chosen from {5, 10}. At test time the retrieval depth k is selected on development data per task (values up to ~50 have been used); QA is decoded greedily, while abstractive generation (MS-MARCO, Jeopardy) uses beam size 4.

## Code framework

Available pieces: a pre-trained dense bi-encoder retriever with a prebuilt MIPS index, a pre-trained encoder-decoder generator, and an Adam optimizer.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- Pre-trained dense retriever over a prebuilt document index ---
class DenseRetriever:
    def __init__(self, query_encoder, doc_index):
        self.q_enc = query_encoder        # BERT-base query encoder (pretrained)
        self.index = doc_index            # FAISS MIPS index over precomputed doc vectors
    def retrieve(self, x, k):
        # returns top-k docs z and their UNNORMALIZED scores q(x).d(z)
        # TODO: encode query, MIPS the index, return (docs, scores)
        pass

# --- Pre-trained encoder-decoder generator ---
class Seq2SeqGenerator(nn.Module):
    def __init__(self, pretrained="bart-large"):
        super().__init__()
        # TODO: load pretrained seq2seq; how does a retrieved doc enter the input?
        pass
    def forward(self, x, z, y):
        # per-token generator log-probs log p(y_i | x, z, y_<i)
        pass

# --- tie retriever + generator into one model trained on (x, y) ---
def retriever_logprobs(doc_scores):
    # turn raw q.d scores over the top-k docs into a distribution over the retrieved set
    # TODO
    pass

def loss(x, y, retriever, generator, k):
    # the document is retrieved, not labelled: define a single training loss on (x, y).
    # TODO
    pass

def decode(x, retriever, generator, k):
    # produce the model's output for x. TODO
    pass

# --- training loop scaffold ---
def train(retriever, generator, data, steps):
    opt = torch.optim.Adam(...)           # TODO: choose trainable parameter set
    for x, y in data:
        l = loss(x, y, retriever, generator, k)
        opt.zero_grad(); l.backward(); opt.step()
```
