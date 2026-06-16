Let me start from the fact that's been nagging me: scaling these language models bought us almost everything except reasoning. Perplexity, sample efficiency, few-shot performance on a pile of tasks — all improved smoothly as the models got bigger. But on the things that actually require multi-step reasoning — grade-school math word problems, commonsense chains, symbolic manipulation — direct-answer few-shot prompting is still too weak. On the hardest arithmetic and symbolic settings, the curve can be nearly flat in model size; on commonsense, scale helps more, but the prompt is still asking for an answer without making the inference path visible. So scale is necessary but plainly not sufficient, and I want to understand why, and whether there's a cheap intervention that unlocks it.

Look at the two things the field already knows, because each gets partway and each has a wound the other seems able to heal. On one side, intermediate-step generation: if you train or fine-tune a model to emit a worked solution — the natural-language steps, or a formal equation an external solver runs — before the answer, multi-step accuracy goes up. Making the working *explicit* helps. But the cost is brutal: you need a big corpus of hand-written rationales to train or fine-tune on, which is far more expensive to produce than plain input-output pairs, and the formal-language versions need an executor and lose the generality of free text. On the other side, in-context few-shot prompting: prepend a few question-answer exemplars to a frozen model and it does the task with no gradient updates, one model for many tasks. Beautiful and cheap. But on reasoning tasks it works poorly, and the damning part — it doesn't improve much with scale. Bigger models don't rescue plain few-shot prompting on multi-step problems.

So I have one idea that helps reasoning but needs training, and one idea that needs no training but doesn't help reasoning. The obvious move is to take their intersection: keep the frozen-model, prompt-only setup, and somehow get the *intermediate-steps* benefit without any fine-tuning. The question is how you inject "produce the working" into a model you're not allowed to train.

Let me think about why standard few-shot prompting fails on a math word problem specifically, because the failure mode is the clue. A standard exemplar is `Q: ... A: 6`. The model sees a few of those and then a new question, and it's being asked to emit the final number *immediately*. But the answer to "There are 15 trees in the grove, workers plant some more, and afterward there are 21 trees" is not a thing you read off — it is the end of a little computation: the missing number is 21 - 15 = 6. By formatting the exemplar as question-then-answer, I've literally instructed the model, by demonstration, to skip the computation and jump straight to a number. And there's a structural problem hiding underneath: whatever the model does between reading the question and emitting the answer token, it's a *fixed* amount of computation — the same handful of forward-pass layers regardless of whether the problem takes one step or eight. A one-step problem and an eight-step problem get the same budget. That can't be right for something with genuine depth.

So what if the exemplar itself demonstrated the working? Instead of `⟨question, answer⟩`, make each demonstration a triple: `⟨question, a series of intermediate reasoning steps in natural language, answer⟩`. The model is an imitator of its context — that's the entire premise of few-shot prompting — so if every exemplar shows the question being decomposed into steps that lead to the answer, the model should, by imitation, decompose the *test* question into its own steps before answering. Think about how a person solves "Jane had 12 flowers, gives 2 to mom, then 3 to dad" — you say "after giving 2 to mom she has 10... then after 3 to dad she has 7... so the answer is 7." You narrate a chain of intermediate states. I want the model to narrate the same kind of chain. Call it a chain of thought: a coherent series of intermediate natural-language reasoning steps that lead to the final answer.

And look at what this buys, mechanically, that plain prompting couldn't. First, the fixed-computation problem dissolves: by generating intermediate tokens, the model spends *more* forward passes — more actual computation — on a problem that needs more steps, and fewer on a shallow one. The amount of "thinking" now scales with the problem, because the chain is longer when the problem is deeper. Second, the same format should not be locked to arithmetic. A commonsense question also has hidden substeps: identify what fact is relevant, reject choices that do not satisfy it, or build the multi-hop bridge before committing to yes/no. A symbolic question has the same need for state tracking: name the last letter of each word before concatenating, or count which people actually flipped the coin before deciding heads/tails. Natural language is the common intermediate representation across all of these, where a bare equation or a formal program would be too narrow. Third, it stays cheap: I am changing the in-context demonstrations, not the model weights; I need a handful of exemplars in the right format rather than a large rationale dataset and a training run. And as a side benefit, the chain is an *interpretable window* — I can read it and see roughly how the model got there, and where it went wrong when it's wrong. The first of these — variable computation allocated through generated intermediate steps — is the one I think actually does part of the work, but it cannot be the whole story; the ablations need to separate extra tokens from meaningful intermediate content.

Now I have to be honest that this might just not work, and think about *when* it would and wouldn't. The whole mechanism rests on the model being able to actually *execute* the steps it writes — do the little arithmetic, map the symbols consistently, stay coherent over a long generation. Those sub-abilities, from what's known about how these models scale, themselves only come in at large scale: small models can't reliably add two numbers without semantic grounding, can't keep a symbol mapping straight, drift off topic over a long output. So I'd predict chain-of-thought prompting to do nothing — or even *hurt* — for small models, which will produce fluent but illogical chains, plausible-sounding sentences that don't compute, and land on worse answers than they'd get by guessing directly. And I'd predict it to kick in only past some scale, maybe around 100B parameters, where the model can carry out the working it narrates. If that's the pattern, it means the capability was *latent* in the large model all along and the chain-of-thought format is what lets it surface — the reasoning ability is there, standard prompting just never gave it room to run. I'd also expect the gains to be largest exactly where the baseline is worst — the hardest, deepest problems like GSM8K — and small or absent on one-step problems where there's nothing to decompose.

Let me pin down the method concretely so I can test it. For free-response math word problems, I manually compose eight exemplars, each being a question, a natural-language chain that works through it, and the final answer, and I reuse that same eight-example prompt across the math word-problem benchmarks except AQuA. AQuA is multiple choice, so it needs four exemplars in the same format but with answer choices and final letter choices. Commonsense and symbolic tasks need their own small prompts because the output formats differ, but the intervention is the same: the answer side of each exemplar contains the intermediate steps before the final answer. I let the model generate greedily and parse the final answer out of the generation — for arithmetic, the number after "The answer is"; for multiple choice, the final option; for yes/no tasks, the final yes or no. That's the whole method. It's almost embarrassingly simple, which is the point: the contribution is the *format* of the demonstration, not any new model or training.

I should worry that any apparent gain is an artifact of something other than genuine step-by-step reasoning, so let me design the ablations I'd run before believing it. Three alternative explanations, three controls.

One: maybe the benefit is just that the chain produces the *equation*, and having the equation written out is what helps — the natural language is incidental. Control: prompt the model to output only the bare mathematical expression (e.g. "5 + 2 * 3 =") before the answer, no prose. My prediction: this helps on one- or two-step problems where the equation is easy to read straight off the question, but does little on GSM8K, because GSM8K's questions are semantically tangled enough that you can't get to the right equation without the natural-language reasoning that decomposes the situation first. If equation-only fails on the hard set, then the prose steps are doing real work, not just dressing up an equation.

Two: maybe the benefit is purely the *extra computation* — the model just needs more forward passes / more intermediate tokens, and the content of those tokens is irrelevant. Control: prompt the model to emit a sequence of dots ("...") matching the length of the equation it would need, giving it the same extra token budget but no actual reasoning content. My prediction: this performs about the same as the baseline. If pure variable-compute-with-no-content doesn't help, then it isn't merely the number of tokens — the *semantic content* of the intermediate steps matters.

Three: maybe the chain just helps the model *retrieve relevant pretraining knowledge*, and it doesn't actually depend on the produced steps to compute the answer — the steps are a knowledge-priming ritual. Control: put the chain of thought *after* the answer instead of before. If the model only needed the chain to activate knowledge, post-answer reasoning would help just as much; if it actually *uses* the sequential steps to reach the answer, post-answer reasoning should not, since the answer is already committed before the steps are written. My prediction: chain-after-answer performs about like the baseline, confirming that the model depends on the *preceding* sequential reasoning to produce the answer, not on mere knowledge activation.

If all three controls come out as predicted — equation-only fails on hard problems, dots don't help, post-answer doesn't help — then the pressure points line up: extra characters are not enough, bare equations are not enough for semantically tangled math, and reasoning placed after the answer is too late. The live hypothesis becomes much narrower: the model needs meaningful natural-language intermediate steps before the final answer.

One more robustness concern, because prompting methods are notoriously fragile: maybe I just got lucky with the particular sentences I wrote. So I'd want different annotators to independently write their own chains for the same exemplars, write a more concise style, sample entirely different exemplars from a training set, and vary the order and number of exemplars. If chain-of-thought beats the baseline by a wide margin under *all* of those, then the effect is about the step-by-step format itself, not about one person's phrasing — which is what I'd need to believe to trust it.

The code is the point's mirror image of how little changes: the entire intervention lives in how one exemplar is laid out. Everything else — the model call, the answer parser, the accuracy loop — is the standard frozen-LM few-shot harness.

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

Scale alone didn't unlock multi-step reasoning, and the two partial fixes — explicit intermediate steps that help but usually require training, and few-shot prompting that is cheap but asks for one-shot answers — combine cleanly if I reformat each in-context exemplar to demonstrate a natural-language chain of intermediate steps before the answer. By imitation the frozen model can produce its own chain, spend computation proportional to the problem's depth, and use language as a shared scratchpad for arithmetic, commonsense, and symbolic state tracking. I expect the effect to emerge only when the model is large enough to execute the steps it writes, to help most on harder multi-step problems, and to survive the equation-only, dots-only, reasoning-after-answer, annotator, exemplar, order, and exemplar-count checks only if the intermediate step content is actually doing the work.
