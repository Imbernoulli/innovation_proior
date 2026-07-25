The puzzle that exposed the limitation is Game of 24: given four numbers, combine them with the four basic operations using each number exactly once to make 24. A capable language model knows arithmetic well, yet when it generates left-to-right it often fails because it commits to a bad first equation and has no way to undo it. Once the model writes something like "4 + 9 = 13 (left: 13 10 13)" it has already lost, because no combination of 13, 10, and 13 equals 24, but the autoregressive pass cannot revisit that first step. This is not a knowledge failure; it is a search failure. Autoregressive decoding samples a single committed walk, and for problems where early decisions dominate the outcome, a single walk is insufficient.

Chain-of-thought prompting helps when a task needs intermediate steps, but it remains one linear chain sampled as a continuous string. It offers no branching, no evaluation of partial progress, and no backtracking. Self-consistency samples many complete chains and majority-votes, which averages noise but still never explores alternatives at the pivotal step. Self-refinement revises a finished answer globally, which tends to miss the specific early branch that went wrong. What is missing is a middle ground between the token, which is too small to evaluate, and the whole solution, which is too large to find reliably. I propose Tree of Thoughts, or ToT, which makes that missing middle explicit.

Tree of Thoughts treats language-model problem solving as heuristic search over partial solutions. A node in the tree is a state consisting of the input plus the sequence of thoughts generated so far. A thought is one coherent intermediate step, sized so that several candidates can be generated and each can be judged for promise. For Game of 24, a thought is one equation that consumes two numbers and produces one, and a state is the multiset of remaining numbers. For creative writing, a thought might be a short paragraph-level plan. For crosswords, a thought might be filling in one word. The decomposition is task-specific, but the structure is general.

The framework has four modular choices. First, thought granularity: the step must be small enough to allow diverse candidates and large enough to permit meaningful evaluation. Second, thought generation: when the thought space is rich and unconstrained, such as paragraph plans, independent chain-of-thought samples naturally produce diversity; when the space is constrained, such as single equations, the model should propose several distinct candidates in one prompt to avoid duplicates. Third, state evaluation: when progress can be scored directly, each state is valued independently, for instance by asking the model to classify reachability as sure, likely, or impossible; when absolute scoring is unreliable, candidate states are compared side-by-side and the model votes on the most promising. Fourth, search procedure: for shallow trees, breadth-first search with a beam of the best states works well; for deeper trees, depth-first search with threshold pruning and backtracking is more appropriate. Beam search over thoughts differs from beam search over tokens because the kept objects are thought-level states and the scoring signal is deliberated value, not token likelihood.

For Game of 24, the concrete instantiation is small but instructive. There are three equation steps plus a final step that writes the answer expression, so the solver runs four steps. Generation uses a propose prompt because equations are constrained and short. Evaluation uses a value prompt with three classes. A single "sure" vote should dominate many "likely" votes, and "impossible" should nearly zero a state out, so the labels are mapped to numeric scores such as impossible to 0.001, likely to 1, and sure to 20. The beam keeps the top five states by aggregated value. This simple machinery turns a left-to-right guesser into a deliberate search procedure.

The following is the actual solver, built directly on a language-model wrapper `gpt(prompt, n=...)` that returns several completions for a prompt. A task object owns the data, the step count and stop tokens, and the prompt wrappers/parsers that turn a partial solution `y` into a propose/value/vote prompt and back; the solver itself just runs the generate-evaluate-select loop over a beam of partial solutions `ys`.

```python
import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values, local = [], {}
    for y in ys:
        if y in local:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    return task.vote_outputs_unwrap(vote_outputs, len(ys))

def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + p + '\n' for p in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + s for s in samples]

def solve(args, task, idx):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    ys = ['']
    for step in range(task.steps):
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample,
                                  prompt_sample=args.prompt_sample,
                                  stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:args.n_select_sample]
        ys = [new_ys[i] for i in select_ids]
    return ys, {}
```

The Game of 24 task object pins the constants that make the loop above concrete: four steps (three equations plus the final answer line), each stopped at a newline; a `propose` prompt that lists candidate next equations over whatever numbers remain, falling back to a chain-of-thought prompt once only "24" is left; and a value prompt whose sure/likely/impossible labels are unwrapped into the 20/1/0.001 scores discussed above.

```python
import os
import re
import pandas as pd
import sympy
from tot.prompts.game24 import *
from tot.tasks.base import DATA_PATH, Task

def get_current_numbers(y):
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    def __init__(self, file='24.csv'):
        super().__init__()
        self.data = list(pd.read_csv(os.path.join(DATA_PATH, '24', file))['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def get_input(self, idx):
        return self.data[idx]

    def test_output(self, idx, output):
        expr = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        if sorted(re.findall(r'\d+', expr)) != sorted(re.findall(r'\d+', self.data[idx])):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expr) == 24)}
        except Exception:
            return {'r': 0}

    @staticmethod
    def propose_prompt_wrap(x, y=''):
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            return cot_prompt.format(input=x) + 'Steps:' + y
        return propose_prompt.format(input=current_numbers)

    @staticmethod
    def value_prompt_wrap(x, y):
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:
            answer = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=answer)
        return value_prompt.format(input=get_current_numbers(y))

    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [output.split('\n')[-1] for output in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        return sum(value * value_names.count(name) for name, value in value_map.items())
```

`solve` is the beam search made literal: at each step it generates candidates for every kept partial solution (`get_proposals` for a constrained space like Game of 24 equations, `get_samples` for a rich space that tolerates independent chain-of-thought draws), evaluates the whole expanded frontier (`get_values` scoring each state on its own, `get_votes` comparing states against each other), and keeps only the top `n_select_sample` before moving to the next step. `Game24Task.value_outputs_unwrap` is exactly the class-to-score map from the earlier discussion: three sampled labels are counted and summed against `{impossible: 0.001, likely: 1, sure: 20}`, so one confident "sure" already outweighs a pile of "likely" votes, and the caching in `get_value` means a state that recurs across branches of the search is only judged once.

Tree of Thoughts generalizes the simpler prompting strategies. Input-output prompting is a tree of depth one and breadth one. Chain-of-thought prompting is a single deep path of breadth one. Self-consistency is a depth-one tree of many complete chains with a final-answer vote. Self-refinement is a chain whose new thoughts revise old ones. By lifting the unit of reasoning to an intermediate thought and using the model itself to guide search, ToT makes deliberate planning available to a pretrained language model without any task-specific training.
