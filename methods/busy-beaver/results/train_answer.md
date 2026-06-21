The question is how to make the boundary of computability concrete. The halting problem tells us that no algorithm can decide whether every Turing machine halts, but that negative statement does not give us a shape or scale for the boundary. A natural first idea is to simulate machines and record how long they run, but simulation alone never certifies non-halting, so it can only produce lower bounds, not the true maximum. Another idea is to search for the longest-running machine among all machines of a given size, but the search space is finite yet the halting predicate is undecidable, so exhaustive enumeration cannot be completed by any algorithm. Existing approaches therefore leave us with either abstract impossibility theorems or partial empirical records, neither of which turns undecidability into a comparable extremal object.

Busy Beaver resolves this by fixing a tiny description budget and asking for the largest finite behavior achievable inside that budget. Concretely, fix a Turing-machine convention, such as two symbols and n non-halting states. There are only finitely many such machines. Among those that do halt, let Sigma(n) be the maximum number of 1s left on the tape and S(n) be the maximum number of steps taken before halting. Each value is a specific natural number, and the function grows faster than any computable function. If S(n) were computable, we could decide halting for every n-state machine by simulating it for exactly S(n) steps: any machine that has not halted by then would never halt. That would solve the halting problem, which is impossible, so S(n) and Sigma(n) cannot be computable. Busy Beaver therefore turns the halting barrier into a named growth curve whose unknown entries mark where finite search, proof, and computation become inseparable.

Busy Beaver is the canonical name of this method. It treats minimal machine size as a measuring stick for maximal finite computation. Instead of asking whether an arbitrary program halts, it asks how much output or time can be squeezed out of the smallest programs that halt. This reframing makes undecidability comparable: exact values for very small n become finite classification results, discovered champion machines become lower bounds, and the open gaps become explicit points where mathematics currently cannot close the search.

The method works in three stages. First, fix a precise machine convention so that the set of n-state machines is finite and unambiguous. Second, enumerate or sample machines of that size, simulate each one with a step limit, and separate the confirmed halters from the unresolved cases. Third, report Sigma(n) and S(n) as the maxima over confirmed halters, while acknowledging that unresolved machines may raise the bound if they ever halt. Proving a value exact requires both a champion machine that attains it and a proof that no other machine of that size can surpass it, with the latter often being the hardest part.

```python
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass(frozen=True)
class TMState:
    """A transition: write symbol, move direction, next state."""
    write: int
    move: int   # +1 for right, -1 for left
    next_state: int  # -1 means halt

class BusyBeaverExplorer:
    """
    Exhaustive Busy Beaver explorer for 2-symbol, n-state machines
    with a single bidirectional tape.  States are numbered 1..n;
    state 0 is the halt state.  On startup the machine is in state 1
    with the tape head over a blank cell (symbol 0).
    """
    def __init__(self, num_states: int, step_limit: int = 10_000):
        self.n = num_states
        self.step_limit = step_limit

    def run(self, transitions: List[List[TMState]]) -> Optional[Tuple[int, int]]:
        """
        Simulate a single machine.  transitions[s][sym] gives the action
        in state s (1-indexed) reading symbol sym (0 or 1).
        Returns (steps, ones) if the machine halts within the step limit,
        otherwise None.
        """
        tape = {0: 0}
        head = 0
        state = 1
        steps = 0
        while state != 0:
            if steps >= self.step_limit:
                return None
            sym = tape.get(head, 0)
            action = transitions[state][sym]
            tape[head] = action.write
            head += action.move
            state = action.next_state
            steps += 1
        ones = sum(1 for v in tape.values() if v == 1)
        return steps, ones

    def _enumerate_transitions(self):
        """Yield every possible transition table for the fixed convention."""
        writes = [0, 1]
        moves = [-1, 1]
        next_states = list(range(-1, self.n + 1))  # -1 halt, 0 unused placeholder
        # We remap -1 to 0 for internal halt state; callers should treat 0 as halt.
        # Build all actions.
        actions = [
            TMState(write=w, move=m, next_state=(ns if ns != -1 else 0))
            for w in writes for m in moves for ns in next_states
        ]
        # A transition table has n states, each with 2 actions (read 0 and read 1).
        # Use iterative product to avoid huge intermediate lists.
        def helper(state_index):
            if state_index == 0:
                yield [[]]  # sentinel; will be filled from 1..n
                return
            for rest in helper(state_index - 1):
                for a0 in actions:
                    for a1 in actions:
                        table = [None] + [None] * self.n
                        for s in range(1, state_index):
                            table[s] = rest[s]
                        table[state_index] = [a0, a1]
                        yield table
        yield from helper(self.n)

    def find_champions(self) -> Tuple[int, int, int, int]:
        """
        Brute-force search over all n-state machines up to the step limit.
        Returns (best_steps, best_ones, machines_examined, unresolved).
        """
        best_steps = 0
        best_ones = 0
        examined = 0
        unresolved = 0
        for transitions in self._enumerate_transitions():
            examined += 1
            result = self.run(transitions)
            if result is None:
                unresolved += 1
                continue
            steps, ones = result
            if steps > best_steps:
                best_steps = steps
            if ones > best_ones:
                best_ones = ones
        return best_steps, best_ones, examined, unresolved


if __name__ == "__main__":
    # Small example: n=2 states with a modest step limit.
    explorer = BusyBeaverExplorer(num_states=2, step_limit=1000)
    best_steps, best_ones, examined, unresolved = explorer.find_champions()
    print(f"Machines examined: {examined}")
    print(f"Unresolved within step limit: {unresolved}")
    print(f"Best steps S(2) lower bound:   {best_steps}")
    print(f"Best ones Sigma(2) lower bound: {best_ones}")
```
