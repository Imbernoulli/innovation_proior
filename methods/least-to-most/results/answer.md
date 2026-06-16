# Least-to-Most Prompting

## The problem

Few-shot chain-of-thought (CoT) prompting raises a frozen large language model's accuracy on multi-step reasoning by showing it a few worked examples, each with a full natural-language rationale, and letting it generate its own rationale before answering. But it has a sharp failure mode: it does well only when the test problem is about as hard as the demonstrations, and degrades as the test problem grows *harder* than anything in the prompt. If the exemplars solve 2-step problems, accuracy slides as the gold solution needs more steps; if the exemplars use short inputs, accuracy collapses on longer inputs. This is *easy-to-hard generalization* — the regime where humans excel (solve small instances, then break large ones down) and where CoT, which tries to solve the whole instance in one continuous rationale, cannot stretch. The most pointed version is *compositional generalization*: a model shown short compositions of primitives fails to generalize to longer or novel compositions (the length split of the command→action benchmark SCAN; the last-letter-concatenation task; multi-step GSM8K and the numerical subset of DROP). The goal is a pure prompting strategy — no architecture change, no training, no symbolic machinery — that lets a frozen LM solve instances substantially harder (more steps, longer inputs) than its exemplars.

## The key idea

Least-to-most prompting mimics how a person handles a hard instance: don't solve it in one shot — decompose it into an ordered list of simpler subproblems, then solve them one at a time, each building on the answers to the earlier ones. It runs in two stages, both implemented purely by few-shot prompting of the same frozen LM (no training in either stage):

1. **Decompose.** Prompt the LM with exemplars that demonstrate breaking a complex problem into an ordered list of simpler subproblems, then append the actual problem and read off its decomposition.
2. **Sequentially solve.** Normalize that ordered list so the original problem itself is the final subproblem, appending it if the decomposition only returned prerequisite components. Then walk the list in order. For each subproblem, build a solving prompt = constant solving exemplars + the previously answered (subproblem, answer) pairs so far + the current subproblem; query the LM; then append this (subproblem, answer) pair to the running context before moving on. The answer to the original problem is the final solution.

The decisive design point is in stage 2: the solving exemplars are deliberately *dependent*, demonstrating a base case plus a recursive step that **reuses the previous answer** rather than recomputing from scratch. Because the accumulated prior (subproblem, answer) pairs sit in the prompt, each step the LM faces is only as hard as a demonstration — one letter to append, one short command to map, one arithmetic step — regardless of how long the overall instance is. The instance's difficulty is absorbed entirely by the *number* of steps, which is what lets a frozen model generalize from easy exemplars to hard instances.

**Contrast with CoT.** CoT and least-to-most can use the *same* worked pairs, but CoT's exemplars are *independent*: its rationale for the longer instance is built from scratch, ignoring the shorter exemplar above it, and the whole problem is solved in one pass. Least-to-most's are dependent — the longer exemplar starts from the shorter one's answer and extends it. That single difference (does the later answer reuse the earlier one) is what buys the length/step generalization. Least-to-most is orthogonal to and composes with CoT and self-consistency, but does not need them; and for some math prompts, decomposition and subproblem solving are folded into one generated response, followed by a short final-answer request.

## The two stages, concretely

**Last-letter concatenation** (symbolic; difficulty axis = list length). Decomposition is just expanding a list into its chain of growing prefixes; the solve loop starts at the first prefix supported by the solving exemplars (the two-word base case) and then extends one word at a time. One decomposition shape:

```
Q: "think, machine, learning, reasoning"
A: "think, machine", "think, machine, learning", "think, machine, learning, reasoning"
```

The solving prompt uses two exemplars that form a base case and a recursive step — the second starts from the first's answer:

```
Q: "think, machine"
A: The last letter of "think" is "k". The last letter of "machine" is "e".
   Concatenating "k", "e" leads to "ke". So, "think, machine" outputs "ke".

Q: "think, machine, learning"
A: "think, machine" outputs "ke". The last letter of "learning" is "g".
   Concatenating "ke", "g" leads to "keg". So, "think, machine, learning" outputs "keg".
```

The CoT prompt uses the same two lists but the rationale for the 3-word list recomputes all three last letters independently — exemplars are independent.

**SCAN** (compositional; difficulty axis = action-sequence length). Decomposition parses a long command into its constituent shorter commands in dependency order; the solve loop then appends the original command so the last mapping step composes the already translated components into the full action expression. Two separate prompts are used: a command-decomposition prompt (8 exemplars) and a command-mapping prompt (14 exemplars chosen to cover the command semantics). Intermediate representations use a compact Python-ish notation (`"LOOK" * 3` rather than `LOOK LOOK LOOK`) to fit the LM's ~2048-token context limit; a postprocessing script expands these expressions at the end. A decomposition exemplar:

```
Q: "look opposite right thrice after walk"
A: "look opposite right thrice" can be solved by: "look opposite right", "look opposite right thrice".
   "walk" can be solved by "walk". So, "look opposite right thrice after walk" can be solved by:
   "look opposite right", "look opposite right thrice", "walk".
```

**Math word problems (GSM8K / DROP)** (difficulty axis = number of solving steps). Here the prompt can fold decomposition and subproblem solving into one generated response: one exemplar first lists the subquestions ("Let's break down this problem: 1. … 2. …") and then answers them in sequence, each numbered answer using the earlier ones. In the revised GSM8K one-shot setup, the prompt prefix ends after "Let's break down this problem:", the model's initial reply is appended to the prompt, and a second short request ending in "The answer is:" elicits the final answer. The CoT baseline is this exemplar with the decomposition part removed.

## Runnable orchestration

This is a prompting method, so the "code" is the orchestration: a `decompose()` call followed by a sequential `solve()` loop that threads each prior (subproblem, answer) pair into the next prompt. The model is a frozen LM completion endpoint.

```python
def llm(prompt, stop=None):
    """Frozen LM completion endpoint (e.g. a GPT-3-class code/text model). Exists already."""
    ...

# --- Stage-1 exemplars: how to break a problem into an ordered list of subproblems. ---
# (last-letter: a list -> growing prefixes, starting at the two-word base case)
DECOMPOSITION_EXEMPLARS = '''Q: "think, machine, learning, reasoning"
A: "think, machine", "think, machine, learning", "think, machine, learning, reasoning"
'''

# --- Stage-2 exemplars: a BASE CASE + a RECURSIVE STEP that builds on the previous answer. ---
# Note the second exemplar starts from the first's answer ("ke") instead of recomputing it.
SOLUTION_EXEMPLARS = '''Q: "think, machine"
A: The last letter of "think" is "k". The last letter of "machine" is "e". \
Concatenating "k", "e" leads to "ke". So, "think, machine" outputs "ke".

Q: "think, machine, learning"
A: "think, machine" outputs "ke". The last letter of "learning" is "g". \
Concatenating "ke", "g" leads to "keg". So, "think, machine, learning" outputs "keg".
'''


def decompose(question):
    """Stage 1: prompt the LM to break `question` into an ordered list of easier subproblems."""
    completion = llm(DECOMPOSITION_EXEMPLARS + f'Q: {question}\nA:')
    return parse_subproblems(completion)        # -> ordered list of sub-instances


def ensure_final_subproblem(subproblems, question):
    """Make the target problem the final step; SCAN-style decompositions omit it."""
    return subproblems if subproblems and subproblems[-1] == question else [*subproblems, question]


def solve(question):
    subproblems = ensure_final_subproblem(
        decompose(question), question
    )                                          # stage 1: ordered prerequisites + original problem
    history = ""                                # accumulated (subproblem, answer) pairs = building blocks
    answer = None
    for sub in subproblems:                     # stage 2: solve in order, each on top of prior answers
        prompt = SOLUTION_EXEMPLARS + history + f'Q: {sub}\nA:'
        answer = llm(prompt).strip()
        history += f'Q: {sub}\nA: {answer}\n\n'  # thread this answer into the next subproblem's prompt
    return answer                               # answer to the final subproblem = the solution
```

For SCAN the two banks are separate prompts (8 decomposition + 14 mapping exemplars), the original command is appended after the decomposed components for the final composition step, and a postprocessing step expands the Python-notation answers. For GSM8K-style tasks, decomposition and subproblem solving can be folded into one generated response, with the revised one-shot prompt using a follow-up `The answer is:` request to extract the final answer. The invariant is identical in every case: expose easier subproblems first, then make later answers depend on earlier answers.

## Verification

- **Two stages, both prompt-only, no training.** Grounded in the method section: stage 1 = decomposition prompt (constant exemplars + the question to decompose); stage 2 = subproblem-solving prompt = (constant solving exemplars) + (a possibly empty list of previously answered subquestions and their solutions) + (the next question); "Both stages are implemented by few-shot prompting, so that there is no training or finetuning in either stage." The original problem is appended or retained as the final subproblem in stage 2.
- **Answer-threading.** Stage 2 constructs the next prompt by appending the generated answer to the previous prompt before the next subproblem — "solving a given subproblem is facilitated by the answers to previously solved subproblems." The `history` string in the loop is exactly this accumulation.
- **Base-case + recursive-step solving exemplars.** Matches the last-letter solving prompt: the 3-word exemplar opens with `"think, machine" outputs "ke"` and extends it rather than recomputing — "The two exemplars together illustrate a base case and a recursive step."
- **CoT contrast.** Both use the same lists; in the CoT prompt the response to the longer list is "built from scratch, instead of using the output of the first list," and "the exemplars in the chain-of-thought prompt are independent of each other." Least-to-most's are dependent. Faithfully reflected in the DECOMPOSITION/SOLUTION exemplars and the contrast paragraph.
- **SCAN specifics.** 8 decomposition exemplars + 14 command-mapping exemplars; decomposition peels a long command into ordered shorter commands; Python notation (`"LOOK" * 2`) used to fit the ~2048-token limit with a postprocessing expansion step — all grounded in the compositional-generalization section.
- **GSM8K merged prompt.** "for some tasks, the two stages in least-to-most prompting can be merged to form a single-pass prompt"; the GSM8K exemplar first lists subquestions ("Let's break down this problem") then answers them in sequence. The revised one-shot appendix adds the operational detail that the initial reply is appended and followed by `The answer is:` in a second request.
- **Easy-to-hard framing.** The motivating limitation (CoT "performs poorly on tasks that require generalization of solving problems harder than the demonstration examples, such as compositional generalization") and the per-task difficulty axes (list length / action-sequence length / number of solving steps) are pre-method motivation, used here only to frame the problem — no proposed-method result numbers are reported.
- **In-frame.** The method is named "Least-to-Most Prompting" as the thing being built; prior-art ancestors (chain-of-thought, self-consistency, SCAN, GSM8K, DROP) are referenced as the landscape. No source-paper citation line, authors, venue, or arXiv id; no result tables.
