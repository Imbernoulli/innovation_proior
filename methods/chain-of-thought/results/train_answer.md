The problem is how to make a frozen large language model solve multi-step reasoning tasks—grade-school math word problems, commonsense chains, and small symbolic manipulations—using only a prompt, without retraining or fine-tuning. Scaling model size has improved many language tasks, but on arithmetic and symbolic reasoning it has left standard few-shot prompting nearly flat. The reason is that ordinary few-shot prompting presents the model with question-answer pairs and asks it to jump directly to the final answer. That format forces a single leap on problems that really require several dependent intermediate steps, and it gives every problem the same fixed amount of generated computation whether the underlying reasoning has depth one or depth eight.

Two partial fixes are known. Letting models generate explicit intermediate steps before the final answer improves multi-step accuracy, but earlier approaches require training or fine-tuning on costly hand-written rationales, and neuro-symbolic variants sacrifice the generality of free natural language for a formal executor. Few-shot prompting is cheap and keeps the model frozen, but it fails on reasoning because it never shows the model how to work through a problem. The natural move is to keep the frozen-model, prompt-only setup and inject the intermediate-step benefit through the in-context demonstrations themselves.

The method is called chain-of-thought prompting. Instead of formatting each exemplar as a question-answer pair, format it as a triple: question, a short natural-language chain of intermediate reasoning steps, and then the final answer. For example, the answer side of an exemplar says, "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6." Because the model imitates the structure it sees in context, when it is shown several demonstrations that narrate a reasoning chain, it generates its own chain for the test question before committing to an answer.

This buys several things at once. It gives the model a variable compute budget: deeper problems naturally produce longer chains of generated tokens, while shallow problems stay short. It uses plain natural language as a shared scratchpad, so the same idea applies to arithmetic, commonsense, and symbolic state tracking without a special formalism or external solver. It stays cheap, since it only changes the content and format of a handful of in-context exemplars rather than the model weights. And it produces an interpretable trace that shows roughly how the answer was reached.

The method is expected to be scale-emergent. Small models often lack the sub-skills needed to execute the steps they write, so chain-of-thought can hurt or do nothing for them; the gains appear once the model is large enough to reliably carry out the working it narrates, becoming clearest around 100B parameters and larger. Gains should be largest on harder multi-step problems like GSM8K and smaller on one-step problems. The effect is also not a mere artifact of extra tokens: ablations show that emitting dots of the same length does not help, that emitting only a bare equation helps on simple problems but not on semantically tangled ones, and that placing the chain after the answer does not help, which together suggest the intermediate steps must be meaningful and must precede the answer.

```python
import re
from typing import Callable, Iterable, List, Protocol, Tuple

Exemplar = Tuple[str, str, str]

class Example(Protocol):
    question: str
    gold: str

def format_cot_exemplar(question: str, chain_of_thought: str, answer: str,
                        answer_prefix: str = "The answer is") -> str:
    return f"Q: {question}\nA: {chain_of_thought} {answer_prefix} {answer}.\n\n"

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
```
