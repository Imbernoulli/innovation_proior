# Context

## Research question

Large pre-trained language models store a startling amount of world knowledge implicitly in their weights — you can probe them like a knowledge base and they recall facts they were never explicitly taught to recall. That is exciting, but the knowledge lives entirely *inside the parameters*, and that has hard consequences. The model cannot easily expand or revise what it knows: if a fact changes in the world, you must retrain or fine-tune to update it. It cannot point at *why* it produced an answer — there is no source to inspect. And it confidently fabricates ("hallucinates") when its parametric memory is wrong or thin. These are not edge cases; they are structural limits of a closed-book parametric model.

The question, then: can we give a *generative* model — a sequence-to-sequence model, the workhorse that can produce open-ended text for QA, generation, classification — access to an explicit, external, *non-parametric* memory it can read from at inference time, so that (1) knowledge can be revised and expanded by editing the memory rather than retraining, (2) the evidence behind an output is inspectable, and (3) generation is grounded in retrieved text instead of hallucinated? And crucially: can the access mechanism (the retriever) and the generator be combined into one model that is trained *end-to-end* on ordinary input/output pairs, with no supervision on which document should be retrieved — yet work as a single, general-purpose architecture across many knowledge-intensive tasks?

## Background

**Parametric knowledge and its limits.** A pre-trained seq2seq or LM acts as an implicit, parameterized knowledge base. The downsides are concrete and were already documented: knowledge cannot be cheaply edited or grown, predictions are not provenance-bearing, and the model hallucinates. "Knowledge-intensive" tasks — those a person could not do without consulting an external source (open-domain QA, fact verification, fact completion, long-form QA) — expose these limits most sharply.

**Hybrid parametric + non-parametric memory.** The alternative is to pair the parametric model with a *non-parametric* memory: an external store of raw text that the model retrieves from. Two recent systems showed this works for masked-LM-based *extractive* open-domain QA. ORQA (Lee et al., 2019) introduced a latent-variable retriever pretrained with the Inverse Cloze Task and fine-tuned jointly with a reader. REALM (Guu et al., 2020) made the retriever fully differentiable and trained it end-to-end with the masked-LM objective during pretraining, even updating the *document* encoder — which forces periodic re-indexing of the whole corpus (the document vectors drift, so the MIPS index must be rebuilt). Both were limited to *extractive* QA (selecting answer spans), not free-form generation.

**Dense retrieval is now a solved, reusable component.** Dense Passage Retrieval (DPR; Karpukhin et al., 2020) showed a bi-encoder — a BERT-base query encoder and a BERT-base document encoder, scoring by inner product p(z|x) ∝ exp(d(z)ᵀq(x)) — beats BM25 on open-domain QA when trained on question–passage pairs. Retrieval over a precomputed document index is a Maximum Inner Product Search (MIPS) problem, solvable approximately in sub-linear time (FAISS; Johnson et al., 2017, with HNSW). Critically, DPR provides a *pre-trained* retriever and a way to build a dense Wikipedia index off the shelf — an access mechanism that already carries knowledge without further training.

**General-purpose pre-trained generators.** On the generation side, BART (Lewis et al., 2019) and T5 (Raffel et al., 2019) are pre-trained encoder–decoder Transformers (BART via a denoising objective) that fine-tune to strong performance across both discriminative and generative tasks. BART-large is ~400M parameters. These are the natural "parametric memory" to augment — a single model fine-tunable on any seq2seq task.

**Memory-augmented neural networks.** Older work attached external differentiable memory to networks trained from scratch for one task: memory networks (Weston et al., 2015; Sukhbaatar et al., 2015), stack-augmented nets, large memory layers. The contrast that matters here: those memories are *learned distributed representations*, trained per task; a memory made of *raw retrievable text* is human-readable (interpretable) and human-writable (editable to update knowledge), and pre-trained access means knowledge is available without task-specific training.

**Latent-variable view.** If "which document was used" is unobserved, the document is a latent variable and the output likelihood is obtained by marginalizing over it — p(y|x) = Σ_z p(z|x) p(y|x,z). This is the probabilistic frame that lets retrieval be trained without document-level labels, by maximizing the marginal likelihood of the observed output.

## Baselines

- **Closed-book parametric models (T5, BART fine-tuned alone).** A single pre-trained seq2seq model fine-tuned to answer/generate from parameters only. **Gap:** knowledge is frozen in weights — uneditable, un-inspectable, prone to hallucination; matching retrieval-grounded accuracy needs enormous parameter counts (e.g. T5-11B for top closed-book open-domain QA).

- **Extractive retrieve-then-read (DPR + reader).** Retrieve passages with DPR, then a reader *extracts* an answer span. **Gap:** the answer must appear verbatim as a span in a retrieved passage; cannot synthesize across passages or generate free-form text; tied to extractive tasks.

- **ORQA / REALM.** Latent-retrieval models combining a masked LM with a differentiable retriever, trained end-to-end. REALM updates the document encoder, requiring expensive asynchronous re-indexing during training. **Gap:** demonstrated only on *extractive* open-domain QA; not a general seq2seq architecture; the index-refresh machinery is heavy.

- **Task-specific retrieval-augmented systems.** Retrieval has separately been bolted onto open-domain QA, fact checking, dialogue, translation, language modeling, Wikipedia generation — each with a bespoke architecture and optimization (search, RL, or latent variables). **Gap:** one architecture per task; no single retrieval-augmented model shown to transfer across many knowledge-intensive tasks.

## Evaluation settings

- **Non-parametric memory.** A single December 2018 Wikipedia dump, each article split into disjoint **100-word chunks**, totaling **21M documents**. Each document is embedded by the (DPR) document encoder; a single FAISS MIPS index with HNSW is built for fast approximate retrieval. The index sits on CPU (~100 GB, compressible to ~36 GB); vectors are 768-d (21M × 768 ≈ 15.3B values).
- **Tasks / datasets.** Open-domain QA: Natural Questions, TriviaQA, WebQuestions, CuratedTREC (extractive answers, but the system may generate them); same train/dev/test splits as prior open-domain work. Abstractive generation: MS-MARCO (answer generation) and Jeopardy question generation. Classification: FEVER fact verification (Supported / Refuted / Not Enough Info), cast as a one-token target sequence.
- **Metrics.** Exact match for open-domain QA; generation quality (factuality, specificity, diversity; standard generation metrics like BLEU/ROUGE-style scores and Q-BLEU for Jeopardy) for the abstractive tasks; classification accuracy for FEVER.
- **Decoding / retrieval depth at test time.** Number of retrieved documents k is a knob (e.g. on the order of 5–50). Two model variants need different decoding: a per-token-marginalized model plugs into a standard beam decoder; a per-sequence-marginalized model runs beam search per document and then combines, either exactly ("thorough") or with an approximation ("fast").

## Code framework

The components that already exist before the method: a pre-trained dense bi-encoder retriever and its prebuilt MIPS index (the non-parametric memory), a pre-trained encoder–decoder generator (the parametric memory), and an Adam optimizer. The method must define how a retrieved document conditions generation, how to combine the retriever's document distribution with the generator's token distribution into one likelihood, and the end-to-end training loss with the document as a latent variable.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- Pre-trained dense retriever over a prebuilt document index (already exists) ---
class DenseRetriever:
    def __init__(self, query_encoder, doc_index):
        self.q_enc = query_encoder        # BERT-base query encoder (pretrained)
        self.index = doc_index            # FAISS MIPS index over precomputed doc vectors
    def retrieve(self, x, k):
        # returns top-k docs z and their UNNORMALIZED scores q(x).d(z)
        # TODO: encode query, MIPS the index, return (docs, scores)
        pass

# --- Pre-trained encoder-decoder generator (already exists) ---
class Seq2SeqGenerator(nn.Module):
    def __init__(self, pretrained="bart-large"):
        super().__init__()
        # TODO: load pretrained seq2seq; how does a retrieved doc enter the input?
        pass
    def forward(self, x, z, y):
        # per-token generator log-probs log p(y_i | x, z, y_<i)
        pass

# --- The contribution: tie retriever + generator into one likelihood over y ---
def retriever_logprobs(doc_scores):
    # turn raw q.d scores over the top-k docs into log p(z|x)
    # TODO
    pass

def marginal_loss(x, y, retriever, generator, k):
    # treat the retrieved doc as a latent variable; marginalize to get p(y|x);
    # return -log p(y|x). TODO: per-sequence vs per-token marginalization?
    pass

def decode(x, retriever, generator, k):
    # produce argmax_y p(y|x) under the chosen marginalization. TODO
    pass

# --- training loop scaffold ---
def train(retriever, generator, data, steps):
    opt = torch.optim.Adam(...)           # fine-tune query encoder + generator
    for x, y in data:
        loss = marginal_loss(x, y, retriever, generator, k)
        opt.zero_grad(); loss.backward(); opt.step()
```
