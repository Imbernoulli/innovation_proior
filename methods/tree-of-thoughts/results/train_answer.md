The puzzle that exposed the limitation is Game of 24: given four numbers, combine them with the four basic operations using each number exactly once to make 24. A capable language model knows arithmetic well, yet when it generates left-to-right it often fails because it commits to a bad first equation and has no way to undo it. Once the model writes something like "4 + 9 = 13 (left: 13 10 13)" it has already lost, because no combination of 13, 10, and 13 equals 24, but the autoregressive pass cannot revisit that first step. This is not a knowledge failure; it is a search failure. Autoregressive decoding samples a single committed walk, and for problems where early decisions dominate the outcome, a single walk is insufficient.

Chain-of-thought prompting helps when a task needs intermediate steps, but it remains one linear chain sampled as a continuous string. It offers no branching, no evaluation of partial progress, and no backtracking. Self-consistency samples many complete chains and majority-votes, which averages noise but still never explores alternatives at the pivotal step. Self-refinement revises a finished answer globally, which tends to miss the specific early branch that went wrong. What is missing is a middle ground between the token, which is too small to evaluate, and the whole solution, which is too large to find reliably. I propose Tree of Thoughts, or ToT, which makes that missing middle explicit.

Tree of Thoughts treats language-model problem solving as heuristic search over partial solutions. A node in the tree is a state consisting of the input plus the sequence of thoughts generated so far. A thought is one coherent intermediate step, sized so that several candidates can be generated and each can be judged for promise. For Game of 24, a thought is one equation that consumes two numbers and produces one, and a state is the multiset of remaining numbers. For creative writing, a thought might be a short paragraph-level plan. For crosswords, a thought might be filling in one word. The decomposition is task-specific, but the structure is general.

The framework has four modular choices. First, thought granularity: the step must be small enough to allow diverse candidates and large enough to permit meaningful evaluation. Second, thought generation: when the thought space is rich and unconstrained, such as paragraph plans, independent chain-of-thought samples naturally produce diversity; when the space is constrained, such as single equations, the model should propose several distinct candidates in one prompt to avoid duplicates. Third, state evaluation: when progress can be scored directly, each state is valued independently, for instance by asking the model to classify reachability as sure, likely, or impossible; when absolute scoring is unreliable, candidate states are compared side-by-side and the model votes on the most promising. Fourth, search procedure: for shallow trees, breadth-first search with a beam of the best states works well; for deeper trees, depth-first search with threshold pruning and backtracking is more appropriate. Beam search over thoughts differs from beam search over tokens because the kept objects are thought-level states and the scoring signal is deliberated value, not token likelihood.

For Game of 24, the concrete instantiation is small but instructive. There are three equation steps plus a final step that writes the answer expression, so the solver runs four steps. Generation uses a propose prompt because equations are constrained and short. Evaluation uses a value prompt with three classes. A single "sure" vote should dominate many "likely" votes, and "impossible" should nearly zero a state out, so the labels are mapped to numeric scores such as impossible to 0.001, likely to 1, and sure to 20. The beam keeps the top five states by aggregated value. This simple machinery turns a left-to-right guesser into a deliberate search procedure.

The following Python script is a self-contained, runnable illustration of the same idea. It enumerates legal equation steps for a Game of 24 puzzle, scores partial states with a small handcrafted heuristic, and runs a breadth-first beam search to find a valid expression. It does not call an external language model; instead it simulates the generator and evaluator with deterministic rules so the whole pipeline can be executed and verified.

```python
import itertools
from fractions import Fraction

def parse_numbers(state):
    if not state:
        return []
    return [Fraction(x) for x in state.split()]

def format_numbers(nums):
    nums = sorted(nums, key=lambda x: (float(x), x.denominator))
    return ' '.join(str(n) for n in nums)

def apply(a, b, op):
    try:
        if op == '+':
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if b == 0:
                return None
            return a / b
    except Exception:
        return None

def legal_equations(nums):
    res = []
    for (i, a), (j, b) in itertools.combinations(enumerate(nums), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        for op in ['+', '-', '*', '/']:
            val = apply(a, b, op)
            if val is None:
                continue
            new_nums = rest + [val]
            expr = f"({a} {op} {b})"
            res.append((new_nums, expr))
    return res

def heuristic(nums):
    # Prefer states whose numbers can still combine toward 24.
    if len(nums) == 1:
        return 1.0 if nums[0] == 24 else 0.0
    if 24 in nums:
        return 0.8
    # Simple signal: existence of a pair close to 24 by basic ops.
    best = 0.0
    for a, b in itertools.combinations(nums, 2):
        for v in [a + b, abs(a - b), a * b]:
            try:
                if v != 0:
                    best = max(best, 1.0 / (1.0 + abs(float(v - 24))))
            except Exception:
                continue
        for num, den in [(a, b), (b, a)]:
            try:
                if den != 0:
                    best = max(best, 1.0 / (1.0 + abs(float(num / den - 24))))
            except Exception:
                continue
    return best

def solve_tot(input_numbers, beam_width=10, max_depth=3):
    initial = format_numbers([Fraction(n) for n in input_numbers])
    beams = {initial: []}  # state -> list of expression steps
    for depth in range(max_depth):
        candidates = []
        for state, history in beams.items():
            nums = parse_numbers(state)
            for new_nums, expr in legal_equations(nums):
                new_state = format_numbers(new_nums)
                new_history = history + [expr]
                score = heuristic(new_nums)
                candidates.append((score, new_state, new_history))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = {}
        for score, state, history in candidates[:beam_width]:
            beams[state] = history
    for state, history in beams.items():
        nums = parse_numbers(state)
        if len(nums) == 1 and nums[0] == 24:
            return history
    return None

if __name__ == "__main__":
    puzzle = [4, 9, 10, 13]
    solution = solve_tot(puzzle)
    print("Puzzle:", puzzle)
    if solution:
        print("Solution steps:")
        for step in solution:
            print(step)
    else:
        print("No solution found in the beam width.")
```

This script captures the spirit of Tree of Thoughts: maintain a beam of partial states, generate legal next thoughts, score each partial state, keep only the most promising ones, and return a complete solution when a leaf state satisfies the goal. The real method replaces the deterministic generator and heuristic with prompted calls to the same language model, but the search skeleton is identical.

Tree of Thoughts generalizes the simpler prompting strategies. Input-output prompting is a tree of depth one and breadth one. Chain-of-thought prompting is a single deep path of breadth one. Self-consistency is a depth-one tree of many complete chains with a final-answer vote. Self-refinement is a chain whose new thoughts revise old ones. By lifting the unit of reasoning to an intermediate thought and using the model itself to guide search, ToT makes deliberate planning available to a pretrained language model without any task-specific training.
