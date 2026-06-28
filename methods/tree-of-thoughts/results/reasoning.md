Let me start from the thing that's actually bothering me, which is a tiny puzzle that a capable model fails far more often than it should. Four numbers, combine them with plus minus times divide, each used once, hit 24. Give it "4 9 10 13". When I ask it to just produce the answer, or even to think step by step, it often comes back with a wrong equation — and yet the model can do any of the individual arithmetic in its sleep. Before I read too much into that I should make sure I'm not blaming the model for an unsolvable instance. Is "4 9 10 13" even reachable? Let me search by hand: I want a pair whose combination plays well with the rest. 4 and 10 differ by 6; 9 and 13 differ by 4; and 6 × 4 = 24. So (4 − 10) × (9 − 13) = (−6) × (−4) = 24. It uses each number once. So the puzzle *has* a solution, and a fairly slick one. The model isn't failing because the answer doesn't exist; it's failing to find an answer that does. That narrows the problem: the arithmetic knowledge is there, and so is at least one valid route. What's broken is the *search* for that route.

So what's it doing wrong. Looking at the failed chains, the failure pattern is local: many of them are already dead at the first step. It writes "4 + 9 = 13 (left 13 10 13)" and now it's stuck. Is it actually stuck, or does it just look stuck? Let me check whether 13, 10, 13 can reach 24 at all. Two of them have to combine first. 13 + 10 = 23, then 23 and 13: 23+13, 23−13, 13−23, 23×13, 23/13, 13/23 — none is 24. 13 − 10 = 3, then 3 and 13: 16, −10, 10, 39, none. 13 × 10 = 130 with a 13 left — hopeless. 13 / 10 = 1.3 with a 13 left — no. And starting from 13 + 13 = 26 then with 10: 36, 16, −16, 260, 2.6, no; 13 × 13 = 169 — no; 13 − 13 = 0 then 0 and 10 gives at best 10 — no. Every first combination of {13, 10, 13} leads to a pair or single that can't make 24. So the state really is dead. The model has committed to "4 + 9" and walked into a corner with no exit, but it doesn't recognize the corner — it just plows forward producing a wrong final equation. The error isn't in the arithmetic. The error is the *opening move*, and once the opening move is made, the left-to-right generator has no mechanism to say "that was bad, let me undo it."

That's the crux. Autoregressive decoding is a single committed walk. p(x) = ∏ p(x[i] | x[1..i-1]); I draw token after token, never looking back. For most text that's fine — there's no single pivotal token, the sentence can recover. But for a problem where the first three tokens decide everything, a one-shot left-to-right pass is betting the whole task on getting the start right with no information about whether the start is right. Chain of thought doesn't fix this. It inserts intermediate thoughts z1, z2, … between input and answer, which helps when the map needs working steps, but it's still one linear chain sampled as one continuous string. One opening move, no alternatives, no evaluation of partial progress, no backtracking. Self-consistency samples k whole chains and majority-votes the answer — that's better, it explores more *complete* routes and averages out noise — but it still never explores *within* a chain at the pivotal step, and majority vote only means anything when answers can repeat, which they can't for an open-ended output. Self-refinement revises a whole finished answer, which is global and tends to miss the one early decision that actually went wrong.

What do all of these share. They treat the unit of generation and selection as either a single token (decoding) or a whole solution (sampling and voting). The token is too small to evaluate — "is starting with 8 good?" is unanswerable in isolation. The whole solution is too big — to find a good one I might have to sample a hundred. There's a missing middle. What I actually want is to reason at the level of a *step* — one equation, one paragraph plan, one filled-in word — and at that level do something the autoregressive walk can't: hold several candidate steps at once, judge which are promising, and walk forward only on the good ones, abandoning the bad.

The instant I phrase it that way I'm describing search through a tree. This is the oldest idea in the field — Newell and Simon, problem solving as search through a combinatorial problem space: nodes are partial solutions, branches are operators that extend them, and a heuristic tells you which branch to take, which to prune, when to back up. So let me just adopt that frame wholesale. A node is a *state* s = [x, z_{1..i}]: the input plus the thoughts so far. An edge generates one more thought. The root is the bare input; a leaf with all steps done yields the answer. Game of 24: three steps, each an equation that consumes two numbers and produces one, so the state is "the numbers I have left." Now the early-commitment problem dissolves, because I'm not committed — at the root I can branch into several first moves, evaluate each, and only descend into the ones that look reachable.

But hold on, classical tree search needs a heuristic, and that's exactly the part that historically is hard. In a chess engine the evaluation function is hand-programmed; in a game-playing net it's a value network trained on millions of positions. I have neither. I have no hand-written rule for "is this partial Game of 24 state promising," and I certainly don't want to train a value network for every new task I dream up. So is this idea dead before it starts?

No — because the model can supply the missing heuristic without a new trained evaluator. The same LM that generates the thoughts can *evaluate* them, in language. I can hand it a partial state and ask "can these numbers still reach 24?" and it'll reason — "5, 5, 14: 5 + 5 + 14 = 24, sure" or "11, 12: 11 + 12 = 23, 12 − 11 = 1, 11 × 12 = 132, none of these is 24 and they can't combine to it, impossible." That's a heuristic that is neither programmed nor learned-with-gradients; it's *deliberated*, instantiated in natural language by the same model. It doesn't need to be perfect — a search heuristic never is — it only needs to be approximately helpful for deciding where to look next.

Now the pieces I have to choose are concrete: how to break the process into thought steps, how to generate candidate next thoughts, how to evaluate states, and what search algorithm to run over the tree.

Take the first — what *is* a thought, how big is one step. There's a real tradeoff here and it sets everything else. If a thought is one token, I'm back to a tree of tokens: enormously deep, and the LM can't meaningfully evaluate "is the token '8' promising." If a thought is the entire solution, I'm back to the bandit-of-solutions setup where I have to sample a ton to get a good one. The right size is in between: small enough that the model can produce several genuinely different, promising candidates for the next step, and big enough that the model can say something useful about whether that step is on track. For Game of 24 that's one equation step (three steps total). For the writing task, a thought is a short paragraph-level plan (a single intermediate step). For crosswords, a thought is filling in one word clue (five to ten steps, depending). The decomposition is task-shaped, and that's fine — it's the one place I lean on the structure of the problem.

Second — generating candidate thoughts from a state. Two regimes, and which one I pick depends on the thought space. If the space is rich — each thought is a paragraph, like a writing plan — then the natural move is to sample several thoughts independently and identically from a chain-of-thought-style prompt: z^(j) ~ p^CoT(z_{i+1} | s) for j = 1..k. Independent samples of a paragraph naturally come out diverse, which is what I want for breadth. But if the space is constrained — each thought is a single equation, or a single word — then i.i.d. sampling is a bad idea, because independent samples of a short, constrained object collide; I'll draw the same "4 + 9 = 13" five times. So instead I write a *propose* prompt that asks the model, in one shot, to list several distinct next steps: [z^(1), …, z^(k)] ~ p^propose(· | s). Proposing them together in the same context lets the model deliberately avoid duplicates. Rich space → sample for diversity; constrained space → propose to avoid collision. That's the rule.

Third — evaluating states. Again two regimes, driven by whether progress is directly scorable. When I *can* judge a state on its own — "can these numbers reach 24" — I value each state independently: v ~ p^value(v | s), prompting the model to reason and emit either a scalar (1–10) or a class. For Game of 24 the natural classes are sure / likely / impossible, where the model does a couple of quick lookahead trials ("5 5 14 → 5+5+14 = 24, sure") to promote good states, and applies commonsense ("1 2 3 are all too small," "10 10 10 are all too big → impossible") to kill dead ones. The valuation is heuristic and allowed to be wrong sometimes; it just has to bias the search. But for something like passage coherence, there's no clean per-state scalar — "how coherent is this plan, absolutely?" is noisy and hard. There, what's reliable is *relative* judgment: put the candidate states side by side and ask which is most promising. So I vote: s* ~ p^vote(s* | S), and V(s) = 1[s = s*]. This is essentially self-consistency lifted to the step level — instead of voting on a final answer, I cast "which state should I explore next?" as a multiple-choice question and let several samples vote. Directly scorable → value each state; hard-to-score, comparison is easier → vote across states. And for either one I can sample the evaluation several times and aggregate, trading compute for a steadier heuristic, since any single judgment is noisy.

Fourth — the search algorithm itself, which I can now plug in over the generate/evaluate machinery. For Game of 24 and the writing task the tree is shallow (at most three steps) and at each step I can score the frontier and prune to a small promising set, so breadth-first search fits: keep a beam of the b best states per level. Concretely — start with S0 = {x}. At step t, expand: S'_t = { [s, z] : s ∈ S_{t-1}, z ∈ G(p, s, k) }, every kept state extended by each of its k generated thoughts. Evaluate the whole expanded frontier, V_t = V(p, S'_t). Then keep the best b: S_t = argmax over size-b subsets S ⊆ S'_t of ∑_{s∈S} V_t(s). After T steps, choose the surviving state with the largest value and call the generator once more for the final output. Note how this differs from beam search: beam search keeps the b sequences of highest *likelihood*, scoring by the model's token probabilities; here the b states are kept by *deliberated value*, the model's own reasoning about progress, and the unit being kept is a thought-state, not a token-prefix.

Crosswords is different — the tree is deeper (up to ten word-placements) and I really want to commit to a line, discover it's hopeless, and back up. That's depth-first search. Descend into the most promising child first; keep going until either I've reached the end (record the output) or the evaluator deems the current state hopeless — for crosswords, it translates the filled letters into constraints on the remaining clues and, if any remaining clue is judged impossible to fill, the value falls below a threshold v_th. When that happens, prune: cut the subtree, trading exploration for exploitation, and *backtrack* to the parent to try the next-best alternative thought there. Backtracking is the whole point — it's the mechanism the left-to-right walk never had. A greedy fill-in path with no return has removed the exact operation this search needs: return to an earlier choice and take a different branch.

Now I notice the older methods aren't competitors so much as corners of this same space. IO prompting is a tree of depth one, breadth one. CoT is a single deep chain, breadth one. CoT-SC is a depth-one tree with breadth k and a majority-vote evaluator. Self-refinement is a chain where new thoughts come from revising old ones. They're all trees of limited depth and breadth; the framework generalizes over them. And it's modular — the base model, the decomposition, the generator, the evaluator, and the search can each be swapped independently — adaptable to a problem's structure and a compute budget, and it needs no training at all, just a prompted pretrained model.

Let me make the Game of 24 instantiation concrete because the constants matter. State is the remaining numbers, parsed out of a "(left: a b c)" annotation. Three equation steps, then a fourth "step" that writes the final answer expression — so the loop runs four steps, each generation stopping at a newline. Generation is *propose*, because the space (one equation over the current numbers) is constrained: one call to a propose prompt, split on newlines, each line a candidate next state. Once the remaining numbers are just "24," I switch to the CoT prompt to emit the final boxed equation. Evaluation is *value*: ask the model to judge the current numbers sure / likely / impossible. I still have to turn three sampled class-labels into a number. I want "sure" to dominate, "impossible" to essentially zero a state out, and "likely" to count as a faint positive. The shape I'm reaching for is a steep map: impossible → 0.001, likely → 1, sure → 20, and for a state I sum the map over however many of the three samples fell in each class. Before I trust that, let me actually push a few label-triples through it and see whether the ordering comes out the way I want. Three "likely" gives 3·1 = 3. One "sure" plus two "impossible" gives 20 + 2·0.001 = 20.002. So a state the model is unanimously lukewarm about (3 likely → 3) ranks *below* a state where a single sample said "sure" even if the other two screamed "impossible" (20.002). That's the asymmetry I wanted — one confident "sure" outweighs a pile of "likely"s — and it falls straight out of the 20-vs-1 gap, no tuning needed. Check the other end: three "impossible" gives 3·0.001 = 0.003, near zero so the state is all but killed, but not a hard zero (which keeps it from breaking a normalization if I ever select by sampling proportional to value). And a clean state, three "sure", tops out at 60. So the map spreads the frontier across roughly 0.003 … 60, with the live/dead boundary sitting near 1. Good — the numbers behave. Selection then keeps the top-b by this aggregated value — for Game of 24, b = 5.

It's worth tracing one expansion end to end to be sure the dead opening move actually gets demoted rather than just deprioritized in my head. Back at the root "4 9 10 13", propose hands me several first equations. Say among them are "4+9=13 (left 13 10 13)", "4−10=−6 (left −6 9 13)", "9−13=−4 (left 4 10 −4)", and "13−9=4 (left 4 4 10)". I just showed {13,10,13} is dead, so its three value samples should land near "impossible" — aggregate ≈ 0.003. The "4−10=−6" branch keeps −6, 9, 13 alive, and that's exactly the front half of the slick solution I found ((4−10)×(9−13)), so the model should call it "sure" at least once — aggregate up in the 20s. Sorting the frontier descending by aggregated value, the live branches sit at the top and "4+9=13" falls to the very bottom of the beam at 0.003. With b = 5 it might technically survive into the next layer, but it sits dead last and contributes nothing once its own children all evaluate impossible. The point that matters is the reversal: under a plain left-to-right pass "4 + 9" was a fully committed prefix; here it's just the lowest-scored leaf on a frontier, and the search spends its expansion budget on the branches that scored high. The commitment is gone.

So the code is just this loop made literal. The LM wrapper returns n samples for a prompt; a task object holds the data, the step count and stop tokens, and the prompt wrappers/parsers; the solver maintains the beam.

```python
import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)         # "(left: ...)" -> sure/likely/impossible prompt
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)  # counts classes -> {impossible:.001, likely:1, sure:20}
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values, local = [], {}
    for y in ys:
        if y in local:            # duplicate candidate -> 0, never explore the same state twice
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    return task.vote_outputs_unwrap(vote_outputs, len(ys))   # tally votes into per-state scores

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
    ys = ['']                                   # current beam of partial solutions
    for step in range(task.steps):
        # 1. generate candidates from each kept state
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample,
                                  prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # 2. evaluate the whole frontier (value each, or vote across)
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        # 3. keep the promising ones (greedy top-b, or sample proportional to value)
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:args.n_select_sample]
        ys = [new_ys[i] for i in select_ids]    # the beam for the next step
    return ys, {}
```

And the Game of 24 task object pins the constants: three equation steps plus a final-answer step, propose-generation, value-evaluation with the sure/likely/impossible map.

```python
import re, os, sympy, pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import *

def get_current_numbers(y):                     # parse the "(left: a b c)" tail
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    def __init__(self, file='24.csv'):
        super().__init__()
        self.data = list(pd.read_csv(os.path.join(DATA_PATH, '24', file))['Puzzles'])
        self.value_cache = {}
        self.steps = 4                          # 3 equation steps + 1 final answer
        self.stops = ['\n'] * 4

    def get_input(self, idx): return self.data[idx]

    def test_output(self, idx, output):         # success = valid expr using each input once, equals 24
        expr = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        if sorted(re.findall(r'\d+', expr)) != sorted(re.findall(r'\d+', self.data[idx])):
            return {'r': 0}
        try:    return {'r': int(sympy.simplify(expr) == 24)}
        except: return {'r': 0}

    @staticmethod
    def propose_prompt_wrap(x, y=''):
        current = get_current_numbers(y if y else x)
        if current == '24':                     # numbers exhausted -> ask for the final equation
            return cot_prompt.format(input=x) + 'Steps:' + y
        return propose_prompt.format(input=current)   # else propose next equations

    @staticmethod
    def value_prompt_wrap(x, y):
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:           # last step -> judge the final answer
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        return value_prompt.format(input=get_current_numbers(y))

    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        names = [o.split('\n')[-1] for o in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}   # sure dominates; impossible ~kills
        return sum(v * names.count(name) for name, v in value_map.items())
```

So the chain is: a left-to-right LM commits to pivotal early tokens and can't revise, which kills it on search-flavored problems; lift the unit of reasoning from token (too small to judge) or whole solution (too big to find) to a *thought* (one step, sized to be both generable-in-variety and evaluable); arrange thoughts as a tree of states; use the LM itself, prompted to deliberate, as the search heuristic — proposing or sampling next thoughts depending on whether the step-space is constrained or rich, and valuing or voting on states depending on whether progress is absolutely or only relatively scorable; and run BFS with a value-pruned beam where the tree is shallow, DFS with threshold-pruning and backtracking where it's deep. No training, just a prompted model deliberately searching its own thoughts.
