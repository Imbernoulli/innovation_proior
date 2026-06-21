# Context

## Research question

A large language model generates text autoregressively: one token at a time, left to right, each token sampled from the conditional distribution over the next token given everything so far. That single mechanism, scaled up, handles arithmetic, commonsense, and symbolic tasks through prompting. The question is how to extend language-model problem solving to tasks where partial progress is meaningful and early commitments matter — such as Game of 24, where four numbers and the operations `+ - * /` must be combined in sequence to equal 24, and where an early equation step determines which numbers remain.

## Background

**Autoregressive decoding.** A pretrained LM with parameters θ factorizes a sequence as p_θ(x) = ∏_i p_θ(x[i] | x[1..i-1]). Greedy decoding, beam search, top-k sampling, and top-p sampling all operate at the token level: they choose among likely next tokens, scored by the model's own probability or a variant of it. Beam search keeps several token prefixes alive, ranked by likelihood.

**Prompting as problem solving.** The standard in-context setup wraps an input x with instructions and/or a few input-output examples, then samples y from p_θ(y | prompt_IO(x)). This works when the input-output map is shallow enough to materialize in one completion.

**Intermediate reasoning.** Chain-of-thought prompting introduces intermediate language sequences z_1, ..., z_n between x and y. Each z_i is a coherent step toward the answer, but in practice the model samples the whole chain plus answer as one continuous string, and the decomposition into steps is implicit. Self-consistency improves robustness by sampling several complete chains and returning the most frequent answer.

**Problem solving as search.** Newell, Shaw, and Simon framed problem solving as search through a combinatorial problem space: nodes are partial solutions, branches are operators that extend them, and heuristics guide which branches to take, prune, or revisit. Search procedures such as breadth-first search, depth-first search, A*, and Monte Carlo tree search are standard, but their heuristic functions have usually been hand-programmed or learned separately.

**Dual-process framing.** Cognitive science often contrasts fast associative decisions with slower deliberate planning. Token-by-token generation resembles the fast mode; deliberate planning involves evaluating partial progress and selecting among alternatives.

## Baselines

**Input-output prompting.** y ~ p_θ(y | prompt_IO(x)). One forward pass produces one answer.

**Chain-of-thought prompting.** z_i ~ p_θ^CoT(z_i | x, z_{1..i-1}) and y ~ p_θ^CoT(y | x, z_{1..n}); in implementation, [z_{1..n}, y] is sampled as a continuous chain. It provides intermediate steps in a single linear path.

**Self-consistency with chain-of-thought.** Sample k independent complete chains [z_{1..n}^{(i)}, y^{(i)}] and return argmax_y #{i : y^{(i)} = y}. This explores a set of full solutions and uses the frequency of final answers as a selection heuristic.

**Self-refinement.** Generate a full answer, then condition on it and feedback to produce a revised answer, repeating for a small number of rounds.

## Evaluation settings

The natural yardsticks are language-stated problems where partial progress is meaningful and early commitments matter.

- **Game of 24.** Input: four numbers. Output: an equation using each number exactly once with `+ - * /` to equal 24. Data: 1,362 puzzles ordered by human solving time, with the hard slice at indices 901-1000 used for testing. Metric: success over 100 puzzles, where success means the equation is valid and equals 24.
- **Creative writing.** Input: four random sentences. Output: a coherent four-paragraph passage ending with those four sentences, one per paragraph. There is no ground-truth passage. Coherence can be assessed by an LM-assigned 1-10 score averaged over samples and by blind human pairwise preference.
- **Mini crosswords.** Input: five horizontal and five vertical clues for a 5x5 grid. Output: the 25-letter board. Success can be measured at the letter, word, and full-game levels. A small held-out test set is separated from a few games reserved for prompting.

The base setup uses a strong chat LM through a completion API at sampling temperature 0.7, fixes the few-shot examples per task, and records generation budgets for comparisons among prompting and search-style procedures.

## Code framework

Available primitives: an LM wrapper that returns `n` completions for a prompt, and a task object that owns the data, stopping rules, success check, and prompt-wrapping/parsing logic. The inference procedure on top of these primitives is left open.

```python
import os, openai, backoff

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    res = completions_with_backoff(model=model, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens,
                                   n=n, stop=stop)
    return [c.message.content for c in res.choices]

class Task:
    def get_input(self, idx): pass
    def test_output(self, idx, output): pass
    def standard_prompt_wrap(self, x, y): pass
    def cot_prompt_wrap(self, x, y): pass

def solve(args, task, idx):
    x = task.get_input(idx)
    # TODO: produce an output for x using the LM wrapper and task primitives.
    pass
```
