# Context

## Research question

A large language model generates text autoregressively: one token at a time, left to right, each token sampled from the conditional distribution over the next token given everything so far. This single mechanism, scaled up, turns out to solve a startling range of problems — arithmetic, commonsense, symbolic manipulation — purely through prompting. But it is, by construction, a committed left-to-right walk: once a token is emitted it is never revisited, and the model never explicitly considers an alternative continuation, never measures how promising a partial solution is, never backs up from a dead end.

The question is whether this is enough to build a general problem solver, and if not, exactly which problems break it. The suspicious cases are the ones where (i) the first few tokens are pivotal — an early commitment can foreclose all solutions — and (ii) reaching the answer requires trying something, recognizing it leads nowhere, and trying something else. A toy that exposes both: take four numbers and combine them with `+ - * /` to make 24. The arithmetic itself is trivial for a capable model, yet a left-to-right generator that writes "4 + 9 = 13" as its first step may have already doomed the attempt, with no way to notice or undo it. What a solution would have to provide is a way to (1) hold several candidate continuations at once instead of betting on one, and (2) evaluate partial progress and decide where to keep searching, including the option to abandon a branch and return to an earlier choice.

## Background

**Autoregressive decoding.** A pretrained LM with parameters θ factorizes the probability of a sequence as p_θ(x) = ∏_i p_θ(x[i] | x[1..i-1]). Generation samples each token from this conditional. Decoding strategies — greedy, beam search, top-k, top-p (nucleus) sampling — all operate at the token level: they choose among likely next *tokens*, scored by the model's own probability (or perplexity). Beam search keeps several partial token-sequences alive, but it ranks them by likelihood, a low-level signal that need not align with whether a partial answer is on track toward the goal.

**Prompting as problem solving.** The dominant way to apply an LM to a task is in-context: wrap the input x with instructions and/or a few input–output demonstrations, then read off the completion. This input–output (IO) prompting works when the map x → y is shallow.

**Intermediate reasoning.** When the map is non-trivial (a math word problem to a number), inserting a chain of intermediate steps before the answer raises accuracy substantially — the model produces a sequence of coherent intermediate sequences (call each one a *thought*) that bridge input and answer, then the answer. In practice the whole chain plus answer is sampled as one continuous string, and *how* the chain is segmented into steps is left implicit. Ensembling helps too: sample several independent chains and take the majority answer, which is more robust because there are usually many valid routes to the same answer.

**Problem solving as search (classical AI / cognitive science).** Work going back to Newell, Shaw, and Simon in the 1950s framed human problem solving as search through a combinatorial *problem space*: a tree whose nodes are partial solutions and whose branches are operators that extend them. The solver is guided by *heuristics* — cheap assessments that suggest which branch to take, rule out classes of solutions, or distinguish likely from unlikely possibilities. Search procedures over such trees — breadth-first search, depth-first search, A*, Monte Carlo tree search — are standard. The persistent open question is where the heuristic comes from: historically it was either hand-programmed (the evaluation function in a chess engine) or learned from data (a value network). Both are expensive or brittle, and neither is available for an open-ended, hard-to-formalize task stated in natural language.

**Dual-process framing.** A recurring lens from cognitive science contrasts a fast, automatic, associative mode of cognition with a slow, deliberate, sequential one. Plain autoregressive token generation resembles the former; the kind of explicit, branch-and-evaluate planning above resembles the latter. The diagnostic observation that motivates everything below is concrete and measurable: on the make-24 task, a strong model with chain-of-thought prompting fails the large majority of instances, and an error breakdown shows the failure is overwhelmingly committed at the *first* step — the opening move is already wrong, and left-to-right decoding has no recourse.

## Baselines

**Input–output (IO) prompting.** y ~ p_θ(y | prompt_IO(x)), where prompt_IO wraps x with task instructions and/or few-shot examples. One forward pass, one answer. Gap: no intermediate computation; the entire solution must materialize in one shot, so any task whose solution needs working steps is at the mercy of a single sample.

**Chain-of-thought (CoT) prompting** (Wei et al., 2022). Introduce thoughts z_1, …, z_n bridging x and y: each z_i ~ p_θ^CoT(z_i | x, z_{1..i-1}) and y ~ p_θ^CoT(y | x, z_{1..n}); in practice [z_{1..n}, y] is sampled as one continuous string. Gap: a single linear chain. There is no branching at a step, no evaluation of whether a partial chain is promising, and no way to back up — an early mistake propagates to the end.

**Self-consistency with CoT (CoT-SC)** (Wang et al., 2022). Sample k i.i.d. chains [z_{1..n}^(i), y^(i)] and return the most frequent answer, argmax_y #{i : y^(i) = y}. More robust than a single chain because it explores a richer set of complete reasoning paths. Gaps: (i) it explores only *complete* chains — there is no local exploration of alternatives *within* a chain at a given step; (ii) the majority heuristic only applies when the output space is small enough for answers to repeat (e.g. multiple-choice), and says nothing for open-ended outputs.

**Self-refinement / iterative revision.** Generate a full answer, then condition on it (and possibly external feedback) to produce a revised answer, repeating a few times. Gap: it revises whole solutions globally and may miss the specific early decision point that went wrong; it also typically needs an external correctness signal.

## Evaluation settings

Natural yardsticks are tasks where left-to-right generation is expected to struggle — search/planning problems statable in language, where partial progress is meaningful and early commitments matter.

- **Game of 24.** Inputs are four numbers; a valid output uses each number exactly once with `+ - * /` to reach 24. Data: 1,362 puzzles ordered by human solving time; a hard slice (indices 901–1000) used for testing. Metric: success rate over 100 puzzles, where success = a valid equation equal to 24 using each input once.
- **Creative writing.** Input: 4 random sentences; output: a coherent 4-paragraph passage ending in those 4 sentences (one per paragraph). No ground-truth passage. Coherence judged by an LM-assigned 1–10 score (averaged over several samples) and by blind human pairwise preference over 100 inputs.
- **Mini crosswords (5×5).** Input: 5 horizontal and 5 vertical clues; output: the 25-letter board. Three success levels: fraction of correct letters (25/game), words (10/game), games. Data scraped into a small held-out test set, with a few games reserved for prompting.

Base model: a strong chat LM queried via its completion API at sampling temperature 0.7. The protocol fixes few-shot example counts per task and a generation budget, and compares against the IO/CoT/CoT-SC/refine baselines above under matched-ish token budgets.

## Code framework

Pre-method primitives that already exist: a wrapper around the LM completion endpoint that returns `n` samples for a prompt, and a per-task object holding the data, the step structure, and the prompt-wrapping/parsing logic. The contribution will fill the inference loop and the per-task generate/evaluate hooks.

```python
import os, openai, backoff

# --- LM access (exists) ---
@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    res = completions_with_backoff(model=model, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens,
                                   n=n, stop=stop)
    return [c.message.content for c in res.choices]

# --- per-task interface (exists as an abstraction; bodies are task-specific) ---
class Task:
    def get_input(self, idx): pass         # the problem instance
    def test_output(self, idx, output): pass  # success check (e.g. equals 24)
    # prompt wrappers / parsers the inference will call:
    def standard_prompt_wrap(self, x, y): pass
    def cot_prompt_wrap(self, x, y): pass
    # --- slots the method will define ---
    # how to propose / value / vote at a step, and the step structure

# --- the inference procedure the method will design ---
def solve(args, task, idx):
    x = task.get_input(idx)
    # TODO: maintain a set of partial solutions, and at each step
    #       (1) extend each into candidates, (2) score the candidates,
    #       (3) keep the promising ones / decide where to search next.
    pass
```
