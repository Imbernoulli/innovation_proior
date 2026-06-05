# Context

The goal is to make large language models solve **multi-step reasoning** problems — arithmetic word problems, commonsense questions, symbolic manipulation — that they reliably get wrong, *without training, fine-tuning, or any task-specific data*. This is the landscape as it stands in early 2022, before the method below.

## Research question

Scaling language models bought broad gains — better perplexity, better sample efficiency, strong few-shot performance on many tasks — but it did *not* solve reasoning. On challenging arithmetic, commonsense, and symbolic-reasoning benchmarks, even the largest models stay weak, and performance often barely improves with scale: a curve nearly flat in model size. So raw scale is necessary but visibly not sufficient for reasoning.

The precise problem: get a large language model to correctly answer a problem that requires several dependent intermediate steps (e.g. a grade-school math word problem where you must compute one quantity, use it to compute another, and only then answer). Standard few-shot prompting — show the model a handful of `⟨question, answer⟩` exemplars and ask for the answer to a new question — produces a single, immediate answer and fails badly on these tasks, because it asks the model to leap from question to final answer in one shot when the computation genuinely has depth.

What a solution would have to achieve:
- **No training and no task-specific data.** Creating large sets of high-quality worked solutions to fine-tune on is expensive and brittle; a method that only changes the *prompt* keeps a single frozen model usable for many tasks.
- **Allocate more computation to harder problems.** A one-shot answer gives every problem the same fixed amount of "thinking"; a harder, deeper problem should be able to use more.
- **Work across reasoning types** — arithmetic, commonsense, symbolic — not just one benchmark.
- Ideally, produce an **interpretable trace** of how the answer was reached.

## Background

Two separate lines of work each get *part* of the way, and each has a limitation that the other seems poised to fix.

**Intermediate-step / rationale generation.** A body of work shows that having a model produce natural-language (or formal) intermediate steps before the answer helps on reasoning. Early systems trained from scratch to emit natural-language rationales for math problems; later ones fine-tune a pretrained model to generate a step-by-step solution (e.g. training on the worked solutions in a math-word-problem corpus, sometimes with a separately trained verifier that re-ranks candidate solutions). Neuro-symbolic approaches instead emit a formal expression (an equation or program) that an external solver executes. The common finding: *making the steps explicit, rather than demanding the final answer directly, improves multi-step accuracy.* **Limitation:** all of these require either training from scratch or fine-tuning on a large set of hand-written rationales — costly to produce and far more involved than simple input–output supervision — and the formal-language variants need an executor and lose the generality of free natural language.

**In-context few-shot prompting.** A frozen large language model can be "programmed" for a new task purely through its context: prepend a few `⟨input, output⟩` exemplars demonstrating the task, then the test input, and read off the continuation. No gradient updates, one model for many tasks. This was strikingly effective for many simple question-answering and classification tasks. **Limitation:** on tasks that require genuine reasoning it works poorly, and — critically — it *does not improve much with scale*. Bigger models do not rescue standard few-shot prompting on multi-step reasoning.

**The shape of the prompting curve, and why scale matters here.** A relevant empirical fact about these models: the ability to do simple arithmetic without semantic grounding, to map symbols consistently, and to stay coherent over a long generation all themselves strengthen with model scale. Small models, even when shown step-by-step demonstrations, tend to produce fluent-but-illogical chains and fail at elementary symbol mapping and arithmetic — so any method built on the model *generating its own multi-step working* would be expected to help only once the model is large enough to execute the steps it writes. This is a pre-method observation about how model capabilities scale, knowable from prior analyses of these models.

**How the targeted tasks look.** Math word problems (e.g. GSM8K-style): a few sentences of natural-language setup, the answer a single number reached through 2–8 dependent arithmetic steps. Commonsense questions: multiple-choice or short-answer questions needing world knowledge plus a small inference chain. Symbolic tasks: e.g. concatenate the last letters of words, or track a coin's heads/tails state through a sequence of flips — trivial in structure but requiring faithful step-by-step execution and generalization to longer inputs than the exemplars.

## Baselines

**Standard few-shot prompting.** The direct comparison. The prompt is a handful of `⟨question, answer⟩` pairs; the model emits the answer immediately for the test question. *Gap:* asks for a one-step leap on problems with multi-step structure; flat or weak across model scale on reasoning benchmarks.

**Rationale-augmented training / fine-tuning.** Train or fine-tune a model to output intermediate reasoning steps, learned from a corpus of worked solutions (optionally with a trained solution-verifier/re-ranker). *Gap:* needs expensive hand-written rationale data and a training run per task; not a frozen-model, prompt-only method.

**Neuro-symbolic / formal-expression methods.** Map the problem to a formal equation or program and execute it with an external solver. *Gap:* requires the formal-language supervision and an executor; less general than free-form natural language and unable to express the messy semantic reasoning some word problems need.

**Task-specific fine-tuned SOTA.** For each benchmark, a model fine-tuned on that benchmark's training set is the standing state of the art a prompting method would be measured against. *Gap (by construction):* task-specific, data-hungry, one model per task.

## Evaluation settings

- **Arithmetic-reasoning benchmarks:** GSM8K (grade-school math word problems), SVAMP, ASDiv, AQuA (algebraic, multiple-choice), MAWPS. Metric: exact-match answer accuracy (the final number/choice).
- **Commonsense-reasoning benchmarks:** CommonsenseQA, StrategyQA, the AI2 Reasoning Challenge (ARC), plus date/sports understanding tasks.
- **Symbolic-reasoning tasks:** last-letter concatenation and coin-flip state tracking, evaluated both in-domain and on inputs *longer* than the few-shot exemplars (out-of-distribution length generalization).
- **Models** spanning a wide scale range, evaluated frozen: GPT-3 / InstructGPT (350M–175B), LaMDA (422M–137B), PaLM (8B–540B), UL2-20B, Codex.
- **Decoding:** greedy decoding from the model.
- **Protocol:** a single small set of manually written few-shot exemplars reused across benchmarks of the same type (e.g. eight exemplars for the free-response math sets; a few exemplars for AQuA's multiple-choice format). Answers parsed from the model's generation by a simple task-specific rule.

## Code framework

The pre-method scaffold is a frozen-LM few-shot evaluation harness: build a prompt from exemplars, call the model, parse an answer, score it. The model client, the exemplar store, the answer parser, and the accuracy loop already exist; how each exemplar is *formatted* — what goes between the question and the final answer — is the empty slot.

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
