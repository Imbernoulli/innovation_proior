Let me start from the fact that's been nagging me: scaling these language models bought us almost everything except reasoning. Perplexity, sample efficiency, few-shot performance on a pile of tasks — all improved smoothly as the models got bigger. But on the things that actually require multi-step reasoning — grade-school math word problems, commonsense chains, symbolic manipulation — the curve is nearly flat in model size. A 137B model is barely better than a much smaller one at GSM8K. So scale is necessary but plainly not sufficient, and I want to understand why, and whether there's a cheap intervention that unlocks it.

Look at the two things the field already knows, because each gets partway and each has a wound the other seems able to heal. On one side, intermediate-step generation: if you train or fine-tune a model to emit a worked solution — the natural-language steps, or a formal equation an external solver runs — before the answer, multi-step accuracy goes up. Making the working *explicit* helps. But the cost is brutal: you need a big corpus of hand-written rationales to train or fine-tune on, which is far more expensive to produce than plain input-output pairs, and the formal-language versions need an executor and lose the generality of free text. On the other side, in-context few-shot prompting: prepend a few question-answer exemplars to a frozen model and it does the task with no gradient updates, one model for many tasks. Beautiful and cheap. But on reasoning tasks it works poorly, and the damning part — it doesn't improve much with scale. Bigger models don't rescue plain few-shot prompting on multi-step problems.

So I have one idea that helps reasoning but needs training, and one idea that needs no training but doesn't help reasoning. The obvious move is to take their intersection: keep the frozen-model, prompt-only setup, and somehow get the *intermediate-steps* benefit without any fine-tuning. The question is how you inject "produce the working" into a model you're not allowed to train.

Let me think about why standard few-shot prompting fails on a math word problem specifically, because the failure mode is the clue. A standard exemplar is `Q: ... A: 5`. The model sees a few of those and then a new question, and it's being asked to emit the final number *immediately*. But the answer to "Roger has 5 balls, buys 2 cans of 3, how many now?" isn't a thing you read off — it's the end of a little computation: 2 cans times 3 is 6, plus 5 is 11. By formatting the exemplar as question-then-answer, I've literally instructed the model, by demonstration, to skip the computation and jump straight to a number. And there's a structural problem hiding underneath: whatever the model does between reading the question and emitting the answer token, it's a *fixed* amount of computation — the same handful of forward-pass layers regardless of whether the problem takes one step or eight. A one-step problem and an eight-step problem get the same budget. That can't be right for something with genuine depth.

So what if the exemplar itself demonstrated the working? Instead of `⟨question, answer⟩`, make each demonstration a triple: `⟨question, a series of intermediate reasoning steps in natural language, answer⟩`. The model is an imitator of its context — that's the entire premise of few-shot prompting — so if every exemplar shows the question being decomposed into steps that lead to the answer, the model should, by imitation, decompose the *test* question into its own steps before answering. Think about how a person solves "Jane had 12 flowers, gives 2 to mom, then 3 to dad" — you say "after giving 2 to mom she has 10... then after 3 to dad she has 7... so the answer is 7." You narrate a chain of intermediate states. I want the model to narrate the same kind of chain. Call it a chain of thought: a coherent series of intermediate natural-language reasoning steps that lead to the final answer.

And look at what this buys, mechanically, that plain prompting couldn't. First, the fixed-computation problem dissolves: by generating intermediate tokens, the model spends *more* forward passes — more actual computation — on a problem that needs more steps, and fewer on a shallow one. The amount of "thinking" now scales with the problem, because the chain is longer when the problem is deeper. Second, it's free and general: I'm only changing how I format the exemplars, so a single frozen off-the-shelf model handles arithmetic, commonsense, and symbolic tasks just by being shown the right kind of demonstration; no training data, no fine-tuning, no per-task model. Third, as a side benefit, the chain is an *interpretable window* — I can read it and see roughly how the model got there, and where it went wrong when it's wrong. The first of these — variable computation allocated through generated intermediate steps — is the one I think actually does the work; I'll want to test that.

Now I have to be honest that this might just not work, and think about *when* it would and wouldn't. The whole mechanism rests on the model being able to actually *execute* the steps it writes — do the little arithmetic, map the symbols consistently, stay coherent over a long generation. Those sub-abilities, from what's known about how these models scale, themselves only come in at large scale: small models can't reliably add two numbers without semantic grounding, can't keep a symbol mapping straight, drift off topic over a long output. So I'd predict chain-of-thought prompting to do nothing — or even *hurt* — for small models, which will produce fluent but illogical chains, plausible-sounding sentences that don't compute, and land on worse answers than they'd get by guessing directly. And I'd predict it to kick in only past some scale, maybe around 100B parameters, where the model can carry out the working it narrates. If that's the pattern, it means the capability was *latent* in the large model all along and the chain-of-thought format is what lets it surface — the reasoning ability is there, standard prompting just never gave it room to run. I'd also expect the gains to be largest exactly where the baseline is worst — the hardest, deepest problems like GSM8K — and small or absent on one-step problems where there's nothing to decompose.

Let me pin down the method concretely so I can test it. I manually write a small set of exemplars — eight of them for the free-response math benchmarks — each being a question, a natural-language chain of thought that works through it, and the final answer. I don't engineer these carefully or tune them; I just write reasonable step-by-step solutions. I prepend the same eight to every test question (for a multiple-choice set like AQuA I use a few exemplars matched to that format), let the model generate greedily, and parse the final answer out of the generation — for arithmetic, take the number after "The answer is". That's the whole method. It's almost embarrassingly simple, which is the point: the contribution is the *format* of the demonstration, not any new model or training.

I should worry that any apparent gain is an artifact of something other than genuine step-by-step reasoning, so let me design the ablations I'd run before believing it. Three alternative explanations, three controls.

One: maybe the benefit is just that the chain produces the *equation*, and having the equation written out is what helps — the natural language is incidental. Control: prompt the model to output only the bare mathematical expression (e.g. "5 + 2 * 3 =") before the answer, no prose. My prediction: this helps on one- or two-step problems where the equation is easy to read straight off the question, but does little on GSM8K, because GSM8K's questions are semantically tangled enough that you can't get to the right equation without the natural-language reasoning that decomposes the situation first. If equation-only fails on the hard set, then the prose steps are doing real work, not just dressing up an equation.

Two: maybe the benefit is purely the *extra computation* — the model just needs more forward passes / more intermediate tokens, and the content of those tokens is irrelevant. Control: prompt the model to emit a sequence of dots ("...") matching the length of the equation it would need, giving it the same extra token budget but no actual reasoning content. My prediction: this performs about the same as the baseline. If pure variable-compute-with-no-content doesn't help, then it isn't merely the number of tokens — the *semantic content* of the intermediate steps matters.

Three: maybe the chain just helps the model *retrieve relevant pretraining knowledge*, and it doesn't actually depend on the produced steps to compute the answer — the steps are a knowledge-priming ritual. Control: put the chain of thought *after* the answer instead of before. If the model only needed the chain to activate knowledge, post-answer reasoning would help just as much; if it actually *uses* the sequential steps to reach the answer, post-answer reasoning should not, since the answer is already committed before the steps are written. My prediction: chain-after-answer performs about like the baseline, confirming that the model depends on the *preceding* sequential reasoning to produce the answer, not on mere knowledge activation.

If all three controls come out as predicted — equation-only fails on hard problems, dots don't help, post-answer doesn't help — then the only explanation left standing is that the model is genuinely using the natural-language intermediate steps, in sequence, before the answer, to do multi-step reasoning. That's the claim I'd be willing to make.

One more robustness concern, because prompting methods are notoriously fragile: maybe I just got lucky with the particular sentences I wrote. So I'd want different annotators to independently write their own chains for the same exemplars, write a more concise style, sample entirely different exemplars from a training set, and vary the order and number of exemplars. If chain-of-thought beats the baseline by a wide margin under *all* of those, then the effect is about the step-by-step format itself, not about one person's phrasing — which is what I'd need to believe to trust it.

The code is the point's mirror image of how little changes: the entire intervention lives in how one exemplar is laid out. Everything else — the model call, the answer parser, the accuracy loop — is the standard frozen-LM few-shot harness.

```python
from typing import List, Tuple

def llm_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    # frozen LM, greedy decoding
    ...

def parse_answer(generation: str) -> str:
    # e.g. for arithmetic, the number after "The answer is"
    ...

# --- the whole method: an exemplar carries the chain of thought, not just the answer ---
def format_exemplar(question: str, chain_of_thought: str, answer: str) -> str:
    # standard prompting would be:  f"Q: {question}\nA: {answer}\n\n"
    # chain-of-thought prompting inserts the natural-language reasoning before the answer:
    return f"Q: {question}\nA: {chain_of_thought} The answer is {answer}.\n\n"

def build_prompt(exemplars: List[Tuple[str, str, str]], test_question: str) -> str:
    body = "".join(format_exemplar(q, cot, a) for q, cot, a in exemplars)
    return body + f"Q: {test_question}\nA:"

# eight hand-written ⟨question, chain of thought, answer⟩ triples, reused across math benchmarks
COT_EXEMPLARS = [
    ("Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
     "Each can has 3 tennis balls. How many tennis balls does he have now?",
     "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.",
     "11"),
    # ... seven more, written without prompt engineering ...
]

def answer_question(exemplars, test_question) -> str:
    gen = llm_generate(build_prompt(exemplars, test_question))
    return parse_answer(gen)

# --- ablation prompt variants used to rule out the alternative explanations ---
def format_equation_only(question, equation, answer):      # control 1: bare equation, no prose
    return f"Q: {question}\nA: {equation} The answer is {answer}.\n\n"

def format_dots_only(question, equation, answer):          # control 2: same token budget, no content
    return f"Q: {question}\nA: {'.' * len(equation)} The answer is {answer}.\n\n"

def format_cot_after(question, chain_of_thought, answer):  # control 3: reasoning after the answer
    return f"Q: {question}\nA: The answer is {answer}. {chain_of_thought}\n\n"
```

The whole chain of reasoning: scale alone didn't unlock multi-step reasoning, and the two partial fixes — explicit intermediate steps (helps, but needs training) and few-shot prompting (no training, but doesn't help reasoning) — combine cleanly if I just reformat each in-context exemplar to demonstrate a natural-language chain of intermediate steps before the answer. By imitation the frozen model then produces its own chain, which lets it spend computation proportional to the problem's depth and surfaces a reasoning ability that was latent but unreachable through one-shot answering. I expect it to emerge only at large scale, to help most on the hardest problems, and — ruling out the equation-only, dots-only, and reasoning-after alternatives — to be doing genuine sequential natural-language reasoning, robustly across who wrote the exemplars.
