# Tree of Thoughts (ToT)

Tree of Thoughts (ToT) turns language-model problem solving into heuristic search over partial solutions. A node is a state s = [x, z_{1..i}], the input plus the thoughts generated so far. A thought is not a token and not a whole answer; it is a coherent intermediate step large enough to evaluate and small enough to branch over, such as one equation step, one paragraph plan, or one crossword word placement.

An instantiation fixes four choices. First, choose the thought granularity. Second, choose a thought generator G(p_θ, s, k): sample k independent thoughts from a chain-of-thought-style prompt when the thought space is rich, or ask the model to propose k distinct thoughts in one call when the space is constrained and duplicate samples are likely. Third, choose a state evaluator V(p_θ, S): value each state independently when progress can be judged on its own, or vote across candidate states when relative comparison is more reliable than an absolute score. Fourth, choose a search procedure. Breadth-first search expands the current frontier, evaluates the expanded states, and keeps the b best states at each step:

S_0 = {x}; for t = 1..T, S'_t = { [s, z] : s in S_{t-1}, z in G(p_θ, s, k) }, V_t = V(p_θ, S'_t), and S_t is the size-b subset of S'_t maximizing the summed values; after T, return G(p_θ, argmax_{s in S_T} V_T(s), 1). Depth-first search explores the most promising children first, records an output once t > T, prunes a subtree when V(p_θ, {s})(s) falls below v_th, and backtracks to the parent to try the next alternative.

This differs from beam search in both unit and score: the maintained objects are thought-level states, not token prefixes, and the pruning signal is a prompted state evaluation, not token likelihood. IO prompting, chain-of-thought prompting, self-consistency, and self-refinement are limited tree shapes inside the same view: one-shot output, one linear chain, many complete chains with final-answer voting, and revision chains.

## Code

The core breadth-first implementation maintains a beam of partial strings, expands each kept state by sampling or proposing thoughts, evaluates the expanded frontier by voting or value scoring, and selects the next beam.

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

For Game of 24, the task object fixes four newline-stopped steps: three equation thoughts and one final answer line. It uses proposal generation for constrained equation steps, switches to a chain-of-thought prompt when the remaining number is already 24, and evaluates states with sure/likely/impossible labels mapped to numeric values so a confident reachable state dominates uncertain ones while impossible states contribute almost nothing.

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
