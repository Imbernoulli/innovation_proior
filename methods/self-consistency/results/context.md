# Context

The goal is to improve the reasoning accuracy of a frozen large language model that is already being prompted to reason step by step — without any training, fine-tuning, extra annotation, or auxiliary model. The lever is the **decoding strategy**: how answers are extracted from the model. This is the landscape as it stands in early 2022, just after chain-of-thought prompting appears, before the method below.

## Research question

Chain-of-thought prompting (Wei et al., 2022) gets a large language model to generate a series of intermediate natural-language reasoning steps before its answer, substantially improving multi-step reasoning. But it is decoded **greedily**: the model produces a *single* reasoning path, taking the most-probable token at each step, and reads the answer off that one path.

That single-path greedy decode is fragile. Greedy decoding is prone to repetition and to getting stuck in a locally-optimal-but-globally-wrong line of reasoning; and any one sampled generation is at the mercy of a single mistake in a single step. A reasoning problem usually has a *fixed* correct answer but *many* valid ways to reach it — and a model that makes an error on one path may well reach the right answer on another. Committing to one path throws away that redundancy.

The precise problem: given a frozen model and chain-of-thought prompting, replace the single greedy decode with a decoding procedure that exploits the multiplicity of reasoning paths to land on the right answer more often — while staying entirely unsupervised, training-free, and usable off the shelf, with no verifier or re-ranker to train.

What a solution must achieve:
- **No additional training, annotation, verifier, or fine-tuning** — a pure decoding-time method on top of an existing prompted model.
- **Robustness to a single faulty reasoning step** — one bad path should not determine the answer.
- **A principled way to combine many paths** that does not require knowing which path is "correct."
- Compatibility with the **fixed-answer** structure of reasoning tasks (the final answer comes from a small set), and with standard sampling algorithms.

## Background

**Chain-of-thought prompting (the immediate ancestor).** Prepend a few `⟨question, chain of thought, answer⟩` exemplars to a frozen model; it then generates its own intermediate reasoning steps before answering. This unlocks multi-step reasoning at large model scale and needs no training. As originally used, it decodes **greedily** — one reasoning path per question. *Limitation that opens the door:* a single path is fragile; the method does nothing to exploit the fact that the same answer can be reached many ways.

**Why a problem admits many reasoning paths.** A well-known idea about human problem-solving: tasks that require deliberate, analytical thinking typically admit *several* distinct lines of reasoning that all reach the same correct conclusion, and the harder the problem, the more such diverse paths exist. If multiple different ways of thinking converge on the same answer, one is more confident the answer is right. This is a fact about the structure of reasoning problems, not about any model — it is the intuition a decoding method could exploit.

**Sampling vs. greedy decoding.** Reasoning tasks have fixed answers, so the field defaults to greedy (or beam) decoding, treating generation as a search for the single best output. Open-ended text generation, by contrast, routinely uses stochastic decoding — temperature sampling, top-k sampling, nucleus (top-p) sampling — to produce *diverse* outputs. These samplers are standard, drop-in tools. The unexploited observation: even when the desired answer is fixed, *diversity in the reasoning process* (not the answer) may be useful.

**A latent-variable view of generation.** A chain-of-thought generation can be read as a reasoning path `r` (a sequence of tokens) followed by a final answer `a` that the path produces, `r → a`. The model defines a joint distribution `P(r, a | prompt, question)`. If `r` is treated as a latent nuisance variable that exists only to reach `a`, then the quantity of interest is the marginal over reasoning paths, `P(a | prompt, question) = Σ_r P(r, a | prompt, question)` — and that marginal can be estimated by sampling `(r, a)` pairs and aggregating their answers. This marginalization framing is the conceptual seed for combining paths.

**Calibration of these models.** A relevant empirical property: large language models are not well-calibrated on these reasoning generations — the model's own probability for a correct full solution is often close to its probability for a wrong one, which is precisely why prior work resorted to *training* separate verifiers/re-rankers to judge solution quality. Any aggregation that leans on the model's raw sequence probabilities has to contend with this.

## Baselines

**Greedy chain-of-thought decoding (the direct comparison).** CoT prompting with a single greedy decode; the answer is read from the one generated path. *Gap:* fragile to a single error, prone to repetition and local optima, and ignores the redundancy of multiple valid paths.

**Sample-and-rank.** Sample multiple generations, then pick the one with the highest model-assigned (sequence) probability. *Gap:* relies on the model's poorly-calibrated probabilities to identify the best solution; ranking by likelihood does not track correctness well.

**Beam search.** Decode multiple high-probability sequences via beam search and take the top. *Gap:* still a likelihood-driven search for *one* output; beams tend to be near-duplicates, giving little genuine reasoning diversity, and on these tasks it can degrade fluency/diversity.

**Trained verifier / re-ranker.** Train an auxiliary model on labeled data to score candidate solutions and re-rank them (used in prior math-word-problem work). *Gap:* requires additional human annotation and a separate training run — exactly the supervised machinery a frozen prompting method wants to avoid.

**Ensembling multiple models.** Train several models (or several prompts) and aggregate their outputs. *Gap:* needs multiple trained models or carefully engineered diverse prompts; heavier than a method working over a single frozen model.

## Evaluation settings

- **Arithmetic reasoning:** AddSub, MultiArith, ASDiv, AQuA, GSM8K, SVAMP. Metric: exact-match answer accuracy.
- **Commonsense reasoning:** CommonsenseQA, StrategyQA, ARC (easy/challenge).
- **Symbolic reasoning:** last-letter concatenation, coin-flip state tracking.
- **Models (frozen, varied scale):** UL2-20B, GPT-3 175B (Codex engines), LaMDA-137B, PaLM-540B.
- **Prompting:** the same chain-of-thought exemplars as the prior work (e.g. eight manually written exemplars for arithmetic; a few per commonsense task).
- **Sampling schemes for diversity:** temperature sampling with top-k truncation — e.g. T=0.5, k=40 for UL2/LaMDA; T=0.7, k=40 for PaLM; T=0.7 without truncation for GPT-3 — following standard open-text-generation practice; results shown to be robust to these settings.
- **Few-shot setting** throughout: no training or fine-tuning of the language models.

## Code framework

The pre-method scaffold is the chain-of-thought few-shot harness extended with a *decoding* slot and an *aggregation* slot. The model client (with sampling), the CoT prompt builder, and the answer parser already exist; how to decode multiple paths and how to combine their answers are the empty slots.

```python
from typing import List

def llm_generate(prompt: str, temperature: float = 0.0, top_k: int = 0) -> str:
    """Frozen LM; supports greedy (temperature=0) and stochastic sampling."""
    ...

def build_cot_prompt(exemplars, question: str) -> str:
    """Chain-of-thought few-shot prompt: ⟨question, chain of thought, answer⟩ exemplars + question."""
    ...

def parse_answer(generation: str) -> str:
    """Extract the final answer a_i from a reasoning generation (task-specific rule)."""
    ...

def decode_paths(prompt: str) -> List[str]:
    # TODO: how to obtain reasoning paths from the model -- the decoding strategy we will design
    raise NotImplementedError

def aggregate(generations: List[str]) -> str:
    # TODO: how to combine the answers of several reasoning paths into one final answer
    raise NotImplementedError

def answer_question(exemplars, question: str) -> str:
    prompt = build_cot_prompt(exemplars, question)
    return aggregate(decode_paths(prompt))
```
