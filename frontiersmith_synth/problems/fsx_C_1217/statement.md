# One Edit to Rule Them All: Repairing a Broken Matcher

## Problem

You are given a small pattern-matching automaton (an NFA with epsilon
transitions) over the alphabet `{0, 1}`: a set of states `0..n-1`, a start
state `0`, a set of accepting states, epsilon edges (no symbol consumed),
and symbol edges (each `(state, symbol) -> state`). This automaton was
*supposed* to implement some intended matcher, but it has been damaged:
somewhere in its structure, exactly **one** thing is wrong — a needed
epsilon edge is missing, a transition points to the wrong place, or one
extra state was incorrectly marked accepting. You do not know which, or
where.

You are also given a handful of example strings the automaton currently
gets wrong: for each, whether the intended matcher should **accept** (1)
or **reject** (0) it, which differs from what the given automaton actually
does on it right now.

Your job: submit a list of **edits** to the automaton (an edit budget
bounds how many you may use) that repairs it. A held-out suite of further
strings — generated from the *same* intended matcher, disjoint from the
examples you were shown — is used to grade how well your edited automaton
matches the intended behavior. Beware: patching each visible example
individually (e.g. "whatever this string does when it stops, mark that
state as accepting") will make the visible examples pass but generalizes
badly, since the real defect is a single shared structural cause, not many
independent ones.

## Input (stdin)

```
testId
n
k_accept
<k_accept accepting state ids>
m_eps
<m_eps lines: a b>            epsilon edge a -> b
m_trans
<m_trans lines: a c b>        on symbol c ('0' or '1'), a -> b
budget
k_examples
<k_examples lines: string label>   label 1=should accept, 0=should reject
(these are strings the automaton currently gets WRONG)
```

States are `0..n-1`; `0` is always the start state. Every `(state, symbol)`
pair has at most one outgoing transition; a state may have several
outgoing epsilon edges.

## Output (stdout)

```
K
<K edit lines, one of:>
TOGGLE s          flip whether state s is accepting
ADDEPS a b        add epsilon edge a -> b (no-op if already present)
DELEPS a b        remove epsilon edge a -> b (no-op if absent)
SETTRANS a c b    set the transition on symbol c from a to be b (overwrites)
```

Each edit costs 1. `K` must not exceed `budget`.

## Feasibility

All state ids referenced must be in `[0, n)`; symbols must be `0`/`1`; the
total edit cost must not exceed `budget`. Any violation, or malformed
output, scores 0.

## Objective & Scoring

Apply your edits to the given automaton, then classify a large held-out
suite of strings with it. Let `F` be your held-out classification accuracy
(fraction correctly labeled accept/reject, matching the *intended* matcher
— not the automaton you were shown). The checker also has its own fixed
calibration baseline `B = 0.30`. Score:

```
Ratio = min(1.0, 0.1 * F / B)
```

Submitting zero edits reproduces the damaged automaton's own raw accuracy
(usually well below `B`). Finding and fixing the true shared defect with a
single well-chosen edit typically drives `F` close to 1.0.

## Constraints

- `1 <= n <= 40`, `0 <= budget <= 12`, `1 <= k_examples <= 8`
- strings in the examples have length `<= 40`
- time limit 5s, memory 512MB

## Example (worked score, illustrative shape only)

Suppose after your edits the held-out suite has 100 strings and your
automaton correctly classifies 95 of them: `F = 0.95`. Then
`Ratio = min(1.0, 0.1 * 0.95 / 0.30) = min(1.0, 0.3167) = 0.3167`. If
instead you submit zero edits and the damaged automaton only gets 20 of
100 right, `F = 0.20`, `Ratio = min(1.0, 0.0667) = 0.0667`.
