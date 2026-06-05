# Tree of Thoughts (ToT)

## Problem

Autoregressive LMs decode left-to-right, one token at a time, with no way to hold alternatives, evaluate partial progress, or backtrack. This breaks on problems where early decisions are pivotal or where reaching the answer needs exploration and revision (e.g. Game of 24, where a wrong first equation dooms the attempt). Chain-of-thought adds a single linear chain of intermediate thoughts but still cannot branch, score a partial solution, or back up; self-consistency only explores complete chains and majority-votes the answer.

## Key idea

Frame problem solving as search over a tree whose nodes are *states* s = [x, z_{1..i}] (input plus the thoughts so far) and whose edges add one more *thought* — a coherent intermediate step sized to be both generable in diverse variants and evaluable for progress. Use the LM itself, prompted to deliberate in language, as the search heuristic that classical methods had to hand-program or train. Instantiating ToT for a task means answering four questions:

1. **Thought decomposition** — choose a step granularity (one equation; one paragraph plan; one filled word).
2. **Thought generation** G(p_θ, s, k):
   - **Sample** i.i.d. from a CoT prompt, z^(j) ~ p_θ^CoT(z_{i+1} | s) — for rich spaces (paragraphs), where independent samples give diversity.
   - **Propose** several distinct steps in one call, [z^(1..k)] ~ p_θ^propose(· | s) — for constrained spaces (a word/equation), to avoid duplicate samples.
3. **State evaluation** V(p_θ, S):
   - **Value** each state independently, v ~ p_θ^value(v | s), as a scalar (1–10) or class (sure/likely/impossible) via lookahead + commonsense — when progress is directly scorable.
   - **Vote** across states, s* ~ p_θ^vote(s* | S), V(s) = 1[s = s*] — when only relative comparison is reliable (e.g. coherence). Aggregate several samples for robustness.
4. **Search algorithm** over the tree:
   - **BFS** (shallow trees): keep a beam of the b best states per step.
   - **DFS** (deep trees): descend most-promising-first; prune a subtree when its value falls below threshold v_th and **backtrack** to the parent.

**BFS.** S_0 = {x}; for t = 1..T: S'_t = { [s, z] : s ∈ S_{t-1}, z ∈ G(p_θ, s, k) }; V_t = V(p_θ, S'_t); S_t = argmax_{S ⊆ S'_t, |S| = b} ∑_{s∈S} V_t(s). Return G(p_θ, argmax_{s∈S_T} V_T(s), 1).

**DFS(s, t).** If t > T: record G(p_θ, s, 1). Else for each s' ∈ G(p_θ, s, k) (sorted by value): if V(p_θ, {s'}) > v_th: DFS(s', t+1) — pruning + backtracking otherwise.

Unlike beam search (keeps top-b by token likelihood), ToT keeps states by deliberated value. IO, CoT, CoT-SC, and self-refine are special cases (trees of limited depth/breadth). No training; modular; adaptable to compute budget.

## Code

```python
import itertools, numpy as np
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
        if y in local:                  # duplicate state -> 0
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_outputs = gpt(task.vote_prompt_wrap(x, ys), n=n_evaluate_sample, stop=None)
    return task.vote_outputs_unwrap(vote_outputs, len(ys))

def get_proposals(task, x, y):
    proposals = gpt(task.propose_prompt_wrap(x, y), n=1, stop=None)[0].split('\n')
    return [y + p + '\n' for p in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    return [y + s for s in gpt(prompt, n=n_generate_sample, stop=stop)]

def solve(args, task, idx):                      # BFS instantiation
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    ys = ['']
    for step in range(task.steps):
        # generate
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample,
                                  prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluate
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        # select (keep best b)
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:args.n_select_sample]
        ys = [new_ys[i] for i in select_ids]
    return ys, {}
```

Game of 24 task object (3 equation steps + 1 answer step; propose-generate; value-evaluate with the class→score map where `sure` dominates and `impossible` nearly kills a state):

```python
import re, os, sympy, pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import *

def get_current_numbers(y):
    return y.strip().split('\n')[-1].split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    def __init__(self, file='24.csv'):
        super().__init__()
        self.data = list(pd.read_csv(os.path.join(DATA_PATH, '24', file))['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def get_input(self, idx): return self.data[idx]

    def test_output(self, idx, output):
        expr = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        if sorted(re.findall(r'\d+', expr)) != sorted(re.findall(r'\d+', self.data[idx])):
            return {'r': 0}
        try:    return {'r': int(sympy.simplify(expr) == 24)}
        except: return {'r': 0}

    @staticmethod
    def propose_prompt_wrap(x, y=''):
        current = get_current_numbers(y if y else x)
        if current == '24':
            return cot_prompt.format(input=x) + 'Steps:' + y
        return propose_prompt.format(input=current)

    @staticmethod
    def value_prompt_wrap(x, y):
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:
            return value_last_step_prompt.format(input=x, answer=last_line.lower().replace('answer: ', ''))
        return value_prompt.format(input=get_current_numbers(y))

    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        names = [o.split('\n')[-1] for o in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        return sum(v * names.count(name) for name, v in value_map.items())
```
