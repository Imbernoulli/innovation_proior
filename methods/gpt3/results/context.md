## Research question

Across NLP, the way to get a model to do a task has converged on a two-stage recipe: pre-train a Transformer language model on a large corpus, then fine-tune it on a supervised dataset for the specific task. The architecture is task-agnostic, but the *adaptation* is not — reaching strong performance on a desired task still takes a labeled set of thousands to hundreds of thousands of examples plus a round of gradient descent on that set.

The question is whether that second stage can be removed. Can a single fixed model perform a wide range of new tasks given only a natural-language instruction and/or a handful of demonstrations supplied at inference time, with no gradient updates? Three pressures make this worth solving:

- **Coverage.** The space of useful language tasks is effectively unbounded (correct this grammar, give an example of this concept, critique this story). Building and maintaining a labeled dataset for each one does not scale.
- **Robustness.** A large model fine-tuned on a narrow task distribution can latch onto spurious, distribution-specific correlations; its score on a benchmark can sit at nominally "human" levels while its true competence on the underlying task is lower, and its out-of-distribution behavior can degrade. The narrower the fine-tuning distribution relative to the model's capacity, the worse this gets.
- **The human comparison.** People pick up a new language task from a short instruction or one or two examples, and switch fluidly between tasks. A system that needs ten thousand labeled examples and a training run per task is qualitatively unlike that, and a system closer to the human mode would be both more general and more useful.

A solution would have to: (i) carry enough general competence that no per-task gradient step is needed; (ii) accept the task specification as ordinary text — an instruction, some demonstrations, or both; and (iii) hold up across a broad spread of tasks rather than one.

## Background

**The pre-train/fine-tune lineage.** Representation learning in NLP moved from static word vectors (word2vec, GloVe) fed to bespoke per-task architectures, to contextual representations from multi-layer RNNs (CoVe, ELMo), to directly fine-tuning a pre-trained recurrent or Transformer language model so that no task-specific architecture is needed at all (GPT-1, Radford 2018; BERT, Devlin 2018; ULMFiT, Howard 2018), and onward through RoBERTa, XLNet, ALBERT, and the text-to-text framing of T5. The consistent win is that one pre-trained body transfers everywhere; the consistent cost is that each task still needs its own labeled set and its own fine-tuning run.

**Evidence that fine-tuning's generalization is fragile.** A high-capacity model fine-tuned on a narrow distribution can exploit annotation artifacts and surface heuristics rather than the intended capability; diagnostic work shows that models passing a benchmark often fail on minimally perturbed or adversarial versions, and that larger models do not automatically generalize better out of distribution. This is a diagnostic fact about the *existing* paradigm: expressiveness times distribution-narrowness is exactly the regime where spurious correlations are most available.

**Task specification as text.** A separate observation is that a plain autoregressive language model, prompted with a natural-language framing of a task, can perform that task with no fine-tuning at all — the input text itself encodes the task. Conditioning the model on an instruction and/or a few demonstrations and asking it to continue is a way to specify a task entirely through the text channel. This had been shown to work in principle, but the measured quality lagged far behind fine-tuned systems (for instance, on the order of single-digit accuracy on open-domain question answering, and a reading-comprehension score tens of points behind the state of the art).

**Smooth power-law scaling of language-model loss.** A body of measurement (Kaplan 2020; Hestness 2017; Rosenfeld 2019) finds that the test cross-entropy loss of autoregressive Transformer LMs falls as a clean power law in three quantities, over many orders of magnitude:

- in non-embedding parameter count N: L(N) ≈ (N_c / N)^α_N, with α_N ≈ 0.076, N_c ≈ 8.8×10¹³;
- in dataset size D (tokens): L(D) ≈ (D_c / D)^α_D, with α_D ≈ 0.095, D_c ≈ 5.4×10¹³;
- in compute C_min (optimally allocated): L(C_min) ≈ (C_c^min / C_min)^α_C^min, with α_C^min ≈ 0.050 and C_c^min ≈ 3.1×10⁸ PF-days.

Two further findings matter. First, loss is only weakly sensitive to architectural *shape* (depth/width aspect ratio, number of heads, feed-forward multiplier) over a broad range — total scale N dominates. Second, the compute-optimal prescription is counterintuitive: at a fixed compute budget you should train a *very large* model on a relatively *modest* number of tokens and stop well before convergence; the optimal model size grows with compute roughly as N_opt ∝ C^0.73, so most additional compute should buy a bigger model rather than more data or more steps. Capability on many downstream tasks tracks this loss.

**Meta-learning structure.** Learning-to-learn has an inner-loop / outer-loop form: an outer process (slow, across many tasks) installs a capability that an inner process (fast, on a single new task) then exploits. Classic instances place the inner loop inside a model's own activations rather than in weight updates — a recurrent network can implement a learning algorithm in its hidden-state dynamics (Hochreiter 2001), and RL² (Duan 2016) lets an agent adapt to a new task purely through the recurrent activations across a trajectory while the weights, trained across a task distribution, stay fixed at test time. Matching networks, MAML, and learning-to-optimize are other points in this literature.

## Baselines

**Autoregressive language modeling (decoder-only Transformer).** Maximize Σ_t log p(x_t | x_<t) over a corpus with a causal-masked Transformer decoder (GPT-1; GPT-2, Radford 2019). The pre-trained model is a general next-token predictor. Used zero-shot via a text prompt, it can attempt downstream tasks with no fine-tuning, but at quality far below fine-tuned systems; used as a fine-tuning backbone, it is strong but needs per-task data and updates. Gap: as a no-fine-tuning route it was, at the scales tried, not competitive.

**Masked / denoising language models (BERT).** Predict masked tokens using bidirectional context; fine-tune with a task head. Strong on many benchmarks because bidirectionality helps tasks that compare or re-read spans. Gaps: still requires per-task fine-tuning data and updates; the masked objective does not give a clean autoregressive generator, so it is awkward for free-form generation and for likelihood-based "continue the text" task specification.

**Text-to-text transfer (T5) and natural-language task framing.** Cast every task as text-in/text-out and prepend a textual task descriptor (McCann 2018; T5, Raffel 2019). This unifies tasks under one format and one model, but the descriptor is used for multi-task *fine-tuning* with weight updates, not for adaptation at inference time without updates. Gap: still trains on the task.

**Parameter-count scaling lines.** One line scales dense Transformers in parameters and compute together — from the original Transformer through GPT-2, Megatron, T5-11B, and a 17B model. A second line (mixture-of-experts / conditional computation) scales parameter count without proportional compute, but only a fraction of parameters fire per token. A third (adaptive computation time, universal transformer) scales compute without parameters. Each increase improved synthesis and/or downstream metrics, establishing that scale buys capability; none of them, at the sizes reached, made fine-tuning-free task adaptation competitive.

**Smaller-is-better and distillation.** ALBERT, knowledge distillation, task-specific distillation push toward strong-but-small models. Orthogonal to the question here: complementary for deployment, but not aimed at fine-tuning-free generality.

## Evaluation settings

The yardstick is the standard NLP benchmark suite that already exists, scored under whatever metric is conventional for each dataset:

- Language modeling and completion: LAMBADA, HellaSwag, StoryCloze; raw LM perplexity on held-out corpora.
- Closed-book question answering: TriviaQA, Natural Questions, WebQuestions; open-ended QA: CoQA, DROP.
- Reading comprehension: SQuAD 2.0, RACE, QuAC.
- Coreference / common sense: Winograd, Winogrande, PIQA, ARC, OpenBookQA.
- Aggregate suites: SuperGLUE; natural-language inference: ANLI, RTE, WIC.
- Translation: WMT pairs (En↔Fr, En↔De, En↔Ro).
- Synthetic probes of on-the-fly adaptation: multi-digit arithmetic, word unscrambling and manipulation, SAT analogies, using a freshly defined nonsense word in a sentence.

Metrics are dataset-conventional: accuracy or per-token-normalized likelihood for multiple choice; F1 / exact match for QA and reading comprehension; BLEU for translation; perplexity for raw LM. For multiple-choice scoring, candidate completions are compared by the model's likelihood, optionally normalized by length or by an unconditional baseline likelihood to remove length and frequency bias. For free-form generation, beam search (width 4, length penalty 0.6). Test-set numbers are reported where the test set is public; otherwise development-set numbers.

## Code framework

The reusable pieces are a byte-level BPE tokenizer, an embedding table, a neural sequence-model slot, an output projection over the vocabulary, next-token cross-entropy, an Adam-style training loop, and autoregressive decoding. The model body and the task interface stay empty here: the code only fixes the signatures that later code must fill.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- byte-level BPE tokenizer (reversible, no UNK) ---
class BPETokenizer:
    def encode(self, text: str) -> list[int]: ...   # text -> token ids
    def decode(self, ids: list[int]) -> str: ...    # token ids -> text

# --- the slot the architecture will fill ---
class SequenceModel(nn.Module):
    """A causal next-token predictor over a vocabulary.
    The internal architecture is exactly what we have to design."""
    def __init__(self, vocab_size, n_ctx, **arch):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, arch["d_model"])
        # TODO: position information
        # TODO: the body of the model (the architecture we will design)
        self.to_logits = nn.Linear(arch["d_model"], vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # TODO: embed -> body -> logits over next token
        # TODO: if targets given, next-token cross-entropy
        raise NotImplementedError

# --- maximum-likelihood next-token training loop ---
def train(model, data, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # exact variant/schedule TBD
    for _ in range(steps):
        x, y = data.next_batch()                        # y = x shifted by one
        _, loss = model(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()

# --- autoregressive decoding from a trained model ---
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx

# --- the slots for "do a task without training on it" ---
def specify_task(tokenizer, task_description, demonstrations, query):
    """Turn an instruction + some example (input, output) pairs + a query
    into a single token sequence the model can continue.
    How to do this without per-task training is open."""
    pass  # TODO

def score(model, tokenizer, prompt, candidates=None):
    """Read an answer out of the model given the constructed prompt:
    either compare likelihoods of candidate completions, or generate.
    The mechanism is open."""
    pass  # TODO
```
