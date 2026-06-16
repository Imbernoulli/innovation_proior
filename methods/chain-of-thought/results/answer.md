# Chain-of-Thought Prompting, distilled

Chain-of-thought (CoT) prompting elicits multi-step reasoning from a frozen large language model by changing the format of the in-context exemplars: instead of `⟨question, answer⟩` pairs, each exemplar is a `⟨question, chain of thought, answer⟩` triple, where the chain of thought is a natural-language series of intermediate reasoning steps leading to the answer. Shown such demonstrations, the model imitates them — producing its own step-by-step reasoning before answering. There is no model training or fine-tuning; task-specific few-shot exemplars are still supplied in the prompt.

## The problem

Scaling language models did not unlock reasoning by itself: on many arithmetic and symbolic tasks, standard few-shot prompting stays weak and improves little with size, while commonsense tasks still leave multi-step inference implicit even when scale helps. Standard few-shot prompting asks for the final answer immediately, forcing a one-step leap on problems with genuine multi-step structure, and gives every problem the same fixed amount of generated computation.

## The key idea

Combine the two partial solutions while keeping the best of each:
- **Explicit intermediate steps** help multi-step accuracy — but prior methods required training/fine-tuning on hand-written rationales.
- **Few-shot prompting** needs no training and reuses one frozen model — but fails on reasoning.

CoT injects intermediate steps *through the prompt*: demonstrate the reasoning in the exemplars, and the frozen model reproduces it. This (1) lets the model allocate more computation (more generated tokens) to deeper problems, (2) uses natural language as a shared scratchpad for arithmetic, commonsense, and symbolic state tracking, (3) requires only a small number of exemplars rather than a large rationale dataset, and (4) yields an interpretable reasoning trace.

## Method

- Use a small set of `⟨question, chain of thought, answer⟩` exemplars: eight manually composed exemplars reused for all free-response math word-problem benchmarks except AQuA; four AQuA training-set exemplars with solutions for multiple choice; small task-format-specific prompts for commonsense and symbolic tasks.
- Prepend them to each test question; generate greedily; parse the final answer (e.g. the number after "The answer is").
- Empirical behavior: the gain is an **emergent ability of scale** — CoT hurts or does not help most models below 10B parameters, begins to help only at sufficiently large scale, and is clearest around 100B+ parameters; smaller models often write fluent but illogical chains. Gains are largest on harder multi-step problems and smaller on one-step problems; arithmetic gains are robust to exemplar annotator, style, choice, order, and count, though prompt engineering still matters for some tasks.

## What the ablations show

- **Equation only** (emit the bare equation, no prose): helps on one- or two-step datasets but not much on semantically harder GSM8K, suggesting the natural-language decomposition is doing work beyond writing an equation.
- **Variable compute only** (emit dots matching the equation length, no content): about baseline, suggesting that extra generated characters alone are not enough.
- **Chain of thought after the answer**: about baseline, supporting the interpretation that the useful steps must precede the answer rather than merely activate relevant knowledge.

## Working code

```python
import re
from typing import Callable, Iterable, List, Protocol, Tuple

Exemplar = Tuple[str, str, str]

class Example(Protocol):
    question: str
    gold: str

def format_standard_exemplar(question: str, answer: str,
                             answer_prefix: str = "The answer is") -> str:
    return f"Q: {question}\nA: {answer_prefix} {answer}.\n\n"

def format_cot_exemplar(question: str, chain_of_thought: str, answer: str,
                        answer_prefix: str = "The answer is") -> str:
    return f"Q: {question}\nA: {chain_of_thought} {answer_prefix} {answer}.\n\n"

def build_standard_prompt(exemplars: Iterable[Tuple[str, str]], test_question: str,
                          answer_prefix: str = "The answer is") -> str:
    body = "".join(format_standard_exemplar(q, a, answer_prefix) for q, a in exemplars)
    return body + f"Q: {test_question}\nA:"

def build_cot_prompt(exemplars: Iterable[Exemplar], test_question: str,
                     answer_prefix: str = "The answer is") -> str:
    body = "".join(format_cot_exemplar(q, cot, a, answer_prefix) for q, cot, a in exemplars)
    return body + f"Q: {test_question}\nA:"

def parse_arithmetic_answer(generation: str) -> str:
    matches = re.findall(r"The answer is\s+(-?\d+(?:\.\d+)?)", generation)
    return matches[-1] if matches else ""

def answer_question(generate: Callable[[str], str],
                    exemplars: Iterable[Exemplar],
                    test_question: str) -> str:
    generation = generate(build_cot_prompt(exemplars, test_question))
    return parse_arithmetic_answer(generation)

def evaluate(generate: Callable[[str], str],
             dataset: Iterable[Example],
             exemplars: Iterable[Exemplar]) -> float:
    items = list(dataset)
    prompt_examples = list(exemplars)
    correct = sum(answer_question(generate, prompt_examples, ex.question) == ex.gold for ex in items)
    return correct / len(items)

MATH_WORD_PROBLEM_EXEMPLARS: List[Exemplar] = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6.",
     "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, "
     "how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
     "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, "
     "how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
     "32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
     "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8.",
     "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
     "then that is 4 more toys. 5 + 4 = 9.",
     "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
     "So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
     "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
     "After losing 2 more, he had 35 - 2 = 33 golf balls.",
     "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 dollars left. 23 - 15 is 8.",
     "8"),
]

def format_equation_only(question: str, equation: str, answer: str) -> str:
    return f"Q: {question}\nA: {equation} The answer is {answer}.\n\n"

def format_dots_only(question: str, equation: str, answer: str) -> str:
    return f"Q: {question}\nA: {'.' * len(equation)} The answer is {answer}.\n\n"

def format_cot_after_answer(question: str, chain_of_thought: str, answer: str) -> str:
    return f"Q: {question}\nA: The answer is {answer}. {chain_of_thought}\n\n"
```
