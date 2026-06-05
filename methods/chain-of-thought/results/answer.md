# Chain-of-Thought Prompting, distilled

Chain-of-thought (CoT) prompting elicits multi-step reasoning from a frozen large language model by changing only the format of the in-context exemplars: instead of `⟨question, answer⟩` pairs, each exemplar is a `⟨question, chain of thought, answer⟩` triple, where the chain of thought is a natural-language series of intermediate reasoning steps leading to the answer. Shown such demonstrations, the model imitates them — producing its own step-by-step reasoning before answering. No training, no fine-tuning, no task-specific data.

## The problem

Scaling language models did not unlock reasoning: on arithmetic, commonsense, and symbolic tasks, performance stays weak and barely improves with size. Standard few-shot prompting asks for the final answer immediately, forcing a one-step leap on problems with genuine multi-step structure, and gives every problem the same fixed amount of computation.

## The key idea

Combine the two partial solutions while keeping the best of each:
- **Explicit intermediate steps** help multi-step accuracy — but prior methods required training/fine-tuning on hand-written rationales.
- **Few-shot prompting** needs no training and reuses one frozen model — but fails on reasoning.

CoT injects intermediate steps *through the prompt*: demonstrate the reasoning in the exemplars, and the frozen model reproduces it. This (1) lets the model allocate more computation (more generated tokens) to deeper problems, (2) requires only reformatting exemplars, working across task types with a single model, and (3) yields an interpretable reasoning trace.

## Method

- Hand-write a small set of `⟨question, chain of thought, answer⟩` exemplars (eight for the free-response math benchmarks; a few matched exemplars for multiple-choice AQuA). No prompt engineering.
- Prepend them to each test question; generate greedily; parse the final answer (e.g. the number after "The answer is").
- Expected behavior (validated by analysis): the gain is an **emergent ability of scale** — CoT does little or hurts below ~10–100B parameters (small models write fluent but illogical chains) and helps only for sufficiently large models; gains are largest on the hardest, deepest problems; the effect is robust to exemplar annotator, style, choice, order, and count.

## Why it works (ablations rule out alternatives)

- **Equation only** (emit the bare equation, no prose): helps on 1–2 step problems but not on semantically harder sets like GSM8K → the natural-language steps do real work, not just the equation.
- **Variable compute only** (emit dots matching the equation length, no content): ≈ baseline → it is not merely the extra token budget; the step *content* matters.
- **Chain of thought after the answer**: ≈ baseline → the model depends on the *preceding* sequential reasoning to compute the answer, not on mere knowledge activation.

## Working code

```python
from typing import List, Tuple

def llm_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Frozen large language model; greedy decoding."""
    ...

def parse_answer(generation: str) -> str:
    """Task-specific: e.g. the first number after 'The answer is' for arithmetic."""
    ...

def format_exemplar(question: str, chain_of_thought: str, answer: str) -> str:
    # the entire method: the chain of thought sits between question and answer
    return f"Q: {question}\nA: {chain_of_thought} The answer is {answer}.\n\n"

def build_prompt(exemplars: List[Tuple[str, str, str]], test_question: str) -> str:
    body = "".join(format_exemplar(q, cot, a) for q, cot, a in exemplars)
    return body + f"Q: {test_question}\nA:"

def answer_question(exemplars, test_question) -> str:
    return parse_answer(llm_generate(build_prompt(exemplars, test_question)))

def evaluate(dataset, exemplars) -> float:
    return sum(answer_question(exemplars, ex.question) == ex.gold
               for ex in dataset) / len(dataset)

# eight manually written triples, reused across the math benchmarks
COT_EXEMPLARS = [
    ("Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
     "Each can has 3 tennis balls. How many tennis balls does he have now?",
     "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.",
     "11"),
    # ... seven more ...
]

# ablation variants
def format_equation_only(q, equation, a):  return f"Q: {q}\nA: {equation} The answer is {a}.\n\n"
def format_dots_only(q, equation, a):      return f"Q: {q}\nA: {'.' * len(equation)} The answer is {a}.\n\n"
def format_cot_after(q, cot, a):           return f"Q: {q}\nA: The answer is {a}. {cot}\n\n"
```
