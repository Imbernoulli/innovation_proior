# Context

The goal is to make large language models solve **multi-step reasoning** problems — arithmetic word problems, commonsense questions, symbolic manipulation — that they reliably get wrong, using only a small prompt rather than training or fine-tuning a new model.

## Research question

Scaling language models bought broad gains — better perplexity, better sample efficiency, strong few-shot performance on many tasks — but scaling alone did *not* solve reasoning. On challenging arithmetic and symbolic-reasoning benchmarks, standard few-shot prompting often stays weak and improves little with scale; on commonsense benchmarks, scale can help more, but a direct-answer prompt still leaves multi-step inference under-served.

The precise problem: get a large language model to correctly answer a problem that requires several dependent intermediate steps (e.g. a grade-school math word problem where you must compute one quantity, use it to compute another, and only then answer). Standard few-shot prompting — show the model a handful of `⟨question, answer⟩` exemplars and ask for the answer to a new question — produces a single, immediate answer and fails badly on these tasks, because it asks the model to leap from question to final answer in one shot when the computation genuinely has depth.

What a solution would have to achieve:
- **No model training or large rationale dataset.** Creating large sets of high-quality worked solutions to fine-tune on is expensive and brittle; a method that only changes a few in-context exemplars keeps a single frozen model usable across task families.
- **Cope with problems of varying depth.** A one-shot answer gives every problem the same fixed amount of "thinking" regardless of how many dependent steps it actually requires.
- **Work across reasoning types** — arithmetic, commonsense, symbolic — not just one benchmark.
- Ideally, produce an **interpretable trace** of how the answer was reached.

## Background

Two separate lines of work each get *part* of the way, and each has its own limitation.

**Intermediate-step / rationale generation.** A body of work shows that having a model produce natural-language (or formal) intermediate steps before the answer helps on reasoning. Early systems trained from scratch to emit natural-language rationales for math problems; later ones fine-tune a pretrained model to generate a step-by-step solution (e.g. training on the worked solutions in a math-word-problem corpus, sometimes with a separately trained verifier that re-ranks candidate solutions). Neuro-symbolic approaches instead emit a formal expression (an equation or program) that an external solver executes. The common finding: *making the steps explicit, rather than demanding the final answer directly, improves multi-step accuracy.* **Limitation:** all of these require either training from scratch or fine-tuning on a large set of hand-written rationales — costly to produce and far more involved than simple input–output supervision — and the formal-language variants need an executor and lose the generality of free natural language.

**In-context few-shot prompting.** A frozen large language model can be "programmed" for a new task purely through its context: prepend a few `⟨input, output⟩` exemplars demonstrating the task, then the test input, and read off the continuation. No gradient updates, one model for many tasks. This was strikingly effective for many simple question-answering and classification tasks. **Limitation:** on tasks that require genuine reasoning it works poorly; for arithmetic and symbolic settings in particular, bigger models often do not rescue standard direct-answer prompting.

**The shape of the prompting curve, and why scale matters here.** A relevant empirical fact about these models: simple arithmetic without semantic grounding, faithful symbol mapping, semantic understanding, staying on topic, and producing a parseable final answer all strengthen with model scale. Any prompt-only method that asks the model to generate its own multi-step working inherits those prerequisites: if the model cannot reliably add, track symbols, stay coherent, and finish in the requested answer format, extra generated text is likely to be fluent text rather than dependable computation.

**How the targeted tasks look.** Math word problems (e.g. GSM8K-style): a few sentences of natural-language setup, the answer a single number reached through several dependent arithmetic steps. Commonsense questions: multiple-choice or short-answer questions needing world knowledge plus a small inference chain. Symbolic tasks: e.g. concatenate the last letters of words, or track a coin's heads/tails state through a sequence of flips — trivial in structure but requiring faithful step-by-step execution and generalization to longer inputs than the exemplars.

## Baselines

**Standard few-shot prompting.** The direct comparison. The prompt is a handful of `⟨question, answer⟩` pairs; the model emits the answer immediately for the test question. *Gap:* asks for a one-step leap on problems with multi-step structure; flat or weak across model scale on reasoning benchmarks.

**Rationale-augmented training / fine-tuning.** Train or fine-tune a model to output intermediate reasoning steps, learned from a corpus of worked solutions (optionally with a trained solution-verifier/re-ranker). *Gap:* needs expensive hand-written rationale data and a training run per task; not a frozen-model, prompt-only method.

**Neuro-symbolic / formal-expression methods.** Map the problem to a formal equation or program and execute it with an external solver. *Gap:* requires the formal-language supervision and an executor; less general than free-form natural language and unable to express the messy semantic reasoning some word problems need.

**Task-specific fine-tuned SOTA.** For each benchmark, a model fine-tuned on that benchmark's training set is the standing state of the art a prompting method would be measured against. *Gap (by construction):* task-specific, data-hungry, one model per task.

## Evaluation settings

- **Arithmetic-reasoning benchmarks:** GSM8K (grade-school math word problems), SVAMP, ASDiv, AQuA (algebraic, multiple-choice), MAWPS. Metric: exact-match answer accuracy (the final number/choice).
- **Commonsense-reasoning benchmarks:** CommonsenseQA, StrategyQA, BIG-bench Date Understanding, BIG-bench Sports Understanding, and SayCan robot-planning tasks.
- **Symbolic-reasoning tasks:** last-letter concatenation and coin-flip state tracking, evaluated both in-domain and on inputs *longer* than the few-shot exemplars (out-of-distribution length generalization).
- **Models** spanning a wide scale range, evaluated frozen: GPT-3 / InstructGPT (350M–175B), LaMDA (422M–137B), PaLM (8B–540B), UL2-20B, Codex.
- **Decoding:** greedy decoding from the model.
- **Protocol:** a small set of few-shot exemplars for each task format: eight manually composed exemplars reused for all free-response math word-problem datasets except AQuA; four AQuA exemplars with solutions from its training set; randomly selected training-set exemplars for CSQA and StrategyQA; first evaluation-set examples for BIG-bench Date/Sports where no training set exists; six SayCan training examples; manually composed symbolic-task exemplars. Answers are parsed from the model's generation by a simple task-specific rule.

## Code framework

A frozen-LM few-shot evaluation harness already has the model client, exemplar store, answer parser, and accuracy loop: build a prompt from exemplars, call the model, parse an answer, score it. How each exemplar is *formatted* is the empty slot.

```python
from typing import List, Tuple

def llm_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Frozen large language model; greedy decoding by default."""
    ...

def parse_answer(generation: str) -> str:
    """Extract the final answer from a generation (task-specific rule)."""
    ...

def format_exemplar(question: str, answer: str) -> str:
    # TODO: how to lay out a single in-context demonstration -- the design we will choose
    raise NotImplementedError

def build_prompt(exemplars: List[Tuple[str, str]], test_question: str) -> str:
    # concatenate formatted exemplars, then the test question
    body = "".join(format_exemplar(q, a) for q, a in exemplars)
    return body + f"Q: {test_question}\nA:"

def answer_question(exemplars, test_question) -> str:
    gen = llm_generate(build_prompt(exemplars, test_question))
    return parse_answer(gen)

def evaluate(dataset, exemplars) -> float:
    correct = sum(answer_question(exemplars, ex.question) == ex.gold for ex in dataset)
    return correct / len(dataset)
```
