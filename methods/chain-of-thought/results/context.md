# Context

The goal is to make large language models solve **multi-step reasoning** problems — arithmetic word problems, commonsense questions, symbolic manipulation — using only a small prompt rather than training or fine-tuning a new model.

## Research question

Scaling language models has bought broad gains — better perplexity, better sample efficiency, strong few-shot performance on many tasks. On challenging arithmetic and symbolic-reasoning benchmarks, standard few-shot prompting stays weak and improves little with scale; on commonsense benchmarks, scale helps more.

The setting: get a frozen large language model to answer a problem that requires several dependent intermediate steps (e.g. a grade-school math word problem where you compute one quantity, use it to compute another, and only then answer), by choosing only what goes in the in-context prompt. Standard few-shot prompting shows the model a handful of `⟨question, answer⟩` exemplars and asks for the answer to a new question; the model emits a single, immediate answer. The question is how to format the in-context exemplars so that one frozen model handles arithmetic, commonsense, and symbolic problems of varying depth, without model training or a large dataset of worked solutions.

## Background

Two separate lines of work each address part of this setting.

**Intermediate-step / rationale generation.** A body of work has a model produce natural-language (or formal) intermediate steps before the answer. Early systems trained from scratch to emit natural-language rationales for math problems; later ones fine-tune a pretrained model to generate a step-by-step solution (e.g. training on the worked solutions in a math-word-problem corpus, sometimes with a separately trained verifier that re-ranks candidate solutions). Neuro-symbolic approaches instead emit a formal expression (an equation or program) that an external solver executes. These methods make the intermediate steps explicit rather than demanding the final answer directly, and require either training from scratch or fine-tuning on a set of hand-written rationales; the formal-language variants use an executor.

**In-context few-shot prompting.** A frozen large language model is "programmed" for a new task purely through its context: prepend a few `⟨input, output⟩` exemplars demonstrating the task, then the test input, and read off the continuation. No gradient updates, one model for many tasks. This is effective for many question-answering and classification tasks.

**The shape of the prompting curve, and how scale enters.** An empirical fact about these models: simple arithmetic without semantic grounding, faithful symbol mapping, semantic understanding, staying on topic, and producing a parseable final answer all strengthen with model scale. A prompt-only method that asks the model to generate its own multi-step working draws on those capabilities: it relies on the model adding reliably, tracking symbols, staying coherent, and finishing in the requested answer format.

**How the targeted tasks look.** Math word problems (e.g. GSM8K-style): a few sentences of natural-language setup, the answer a single number reached through several dependent arithmetic steps. Commonsense questions: multiple-choice or short-answer questions needing world knowledge plus a small inference chain. Symbolic tasks: e.g. concatenate the last letters of words, or track a coin's heads/tails state through a sequence of flips — trivial in structure but requiring step-by-step execution and generalization to longer inputs than the exemplars.

## Baselines

**Standard few-shot prompting.** The direct comparison. The prompt is a handful of `⟨question, answer⟩` pairs; the model emits the answer immediately for the test question.

**Rationale-augmented training / fine-tuning.** Train or fine-tune a model to output intermediate reasoning steps, learned from a corpus of worked solutions (optionally with a trained solution-verifier/re-ranker).

**Neuro-symbolic / formal-expression methods.** Map the problem to a formal equation or program and execute it with an external solver.

**Task-specific fine-tuned SOTA.** For each benchmark, a model fine-tuned on that benchmark's training set is the standing state of the art a prompting method is measured against.

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
