# Tournament home/away scheduling as a satisfiability instance

## Research question

A round-robin tournament is being scheduled. In each round every team is tagged either **home** (`1`)
or **away** (`0`). A list of pairwise requirements has been collected from venue availability, TV
slots, and fairness rules. Every requirement is a disjunctive constraint of the form

> "team `i` is **home** in this round, **or** team `j` is **away** in this round"

(more generally, each clause fixes a desired value `a` for team `i` *or* a desired value `b` for team
`j`; either side satisfying it is enough). Some requirements pin a single team by repeating it on both
sides; some directly contradict one another.

Given `n` teams and `m` such two-team requirements, decide whether there exists a single assignment of
home/away to all teams that satisfies **every** requirement simultaneously. If one exists, output it.
The interesting part is not any one constraint but the *interaction*: a long tangle of two-team
"either/or" rules can force chains of consequences ("if team 3 is away then team 5 must be home then
team 8 must be away ...") that eventually loop back and contradict themselves, and the question is
whether the whole web is jointly satisfiable.

## Input / output contract

- Input (stdin):
  - line 1: two integers `n` and `m` — the number of boolean variables (teams) and the number of
    clauses (requirements). `0 <= n <= 10^5`, `0 <= m <= 10^5`.
  - next `m` lines: four integers `i a j b` describing the clause `(team i == a) OR (team j == b)`,
    with `0 <= i, j < n` and `a, b in {0, 1}` (`1` = home, `0` = away). `i` and `j` may be equal.
- Output (stdout):
  - if no satisfying assignment exists, a single line `NO`.
  - otherwise, line 1 is `YES`, and line 2 contains `n` space-separated values in `{0, 1}`, the
    home/away tag of teams `0..n-1`. If `n = 0`, the second line is empty. **Any** assignment that
    satisfies all clauses is accepted (a checker validates the clauses, not a fixed answer string).
- Time limit: 1 second. Memory: 256 MB.

Example: with `n = 3` and the clauses `(0,1)|(1,1)`, `(1,0)|(2,1)`, `(2,0)|(0,0)`, the output

```
YES
1 0 0
```

is valid: team 0 home, teams 1 and 2 away — check each clause is satisfied by at least one side.

## Background

Each clause has the shape `lit_a OR lit_b`, where a *literal* asserts a specific boolean value of one
variable. A formula that is a conjunction of two-literal clauses is a **2-CNF** formula, and deciding
its satisfiability is the classic **2-SAT** problem. Two angles are visible before committing:

- **Search / propagation.** Pick a variable, guess a value, unit-propagate the forced consequences,
  backtrack on conflict. This is correct and is what a generic SAT solver does, but the worst case is
  exponential, and an adversarial chain of constraints can trigger it.
- **Structure of implications.** A clause `lit_a OR lit_b` is logically equivalent to the pair of
  implications `(NOT lit_a) -> lit_b` and `(NOT lit_b) -> lit_a`. Collecting all such implications
  turns the formula into a directed graph on the `2n` literals, and the satisfiability question becomes
  a purely graph-theoretic one about that graph's strongly connected components.

## Evaluation settings

Judged on hidden tests covering: tiny instances (`n = 0`, `n = 1`), free variables with no clauses,
unit-style clauses that pin one team, directly contradictory pairs (forcing `NO`), long implication
chains that stress recursion depth, duplicate clauses, self-referential clauses (`i == j`), dense
random instances that are usually unsatisfiable, and full-size instances `n = m = 10^5`. The returned
assignment is re-checked clause by clause, so an arbitrary valid assignment is accepted.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int c = 0; c < m; c++) {
        int i, a, j, b;
        cin >> i >> a >> j >> b;   // clause (var i == a) OR (var j == b)
        // TODO: record the clause / build whatever structure the method needs.
    }

    // TODO: decide satisfiability; if satisfiable, recover one assignment.
    bool satisfiable = false;

    if (!satisfiable) {
        cout << "NO\n";
    } else {
        cout << "YES\n";
        // TODO: print n space-separated 0/1 values.
    }
    return 0;
}
```
