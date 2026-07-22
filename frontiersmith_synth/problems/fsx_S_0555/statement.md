# State Golf: Smallest NFA Separating Two String Samples

## Problem
You are given two disjoint finite sets of strings over an alphabet: a positive
sample **P** (must be accepted) and a negative sample **N** (must be rejected).
Output a **nondeterministic finite automaton (NFA)** that accepts every string in
**P** and rejects every string in **N**, using as **few states as possible**.

An NFA is `(states, start set, accept set, transitions)`. It **accepts** a string
`x` if there exists at least one path that starts in a start state, follows
transitions labelled by the symbols of `x` in order (a state may have zero, one,
or several outgoing edges on the same symbol), and ends in an accept state after
consuming all of `x`. There are no epsilon transitions.

Because a state may fire on many paths at once, one state can serve many roles:
the minimum NFA is a **covering** problem, not deterministic-automaton
minimization. A minimal DFA that classifies P exactly is always valid but is
usually far larger than the best NFA.

## Input (stdin)
```
alphabet                (a string of the distinct usable symbols, e.g. 01)
nP nN                   (sizes of P and N)
p_1 ... p_nP            (the nP positive strings, one per line, each non-empty)
n_1 ... n_nN            (the nN negative strings, one per line, each non-empty)
```
All sample strings use only alphabet symbols; P and N are disjoint.

## Output (stdout)
```
S                       (number of states; states are 0..S-1)
k  q_1 ... q_k          (k start states, then their indices)
m  a_1 ... a_m          (m accept states, then their indices)
T                       (number of transitions)
f sym t                 (T lines: an edge from state f to state t on symbol sym)
```
Multiple edges from the same state on the same symbol are allowed (that is the
nondeterminism). `1 <= S <= 5000`.

## Feasibility
Your NFA is feasible iff it accepts **all** of P and rejects **all** of N, with
every index in range and every transition symbol in the alphabet. Any violation
(unparsable output, out-of-range index, a rejected positive, an accepted
negative) scores **0**.

## Objective
Minimize `S`, the number of states.

## Scoring
Let `B` be the number of states of the checker's internal baseline: the
**prefix-trie acceptor of P** (one state per distinct prefix of the strings in
P; it accepts exactly P). For a feasible NFA with `S` states the score is
```
Ratio = min(1.0, 0.1 * B / S)
```
So the trie baseline scores `0.1`, and a 10x smaller automaton reaches `1.0`.
Fewer states is strictly better; the reference constructions do not reach the
cap, so smaller separating automata still gain.

## Constraints
- Deterministic scoring; same submission always earns the same score.
- Time limit 5 s, memory 512 MB. Each instance is small.

## Example
Suppose `alphabet = 01`, `P = {01, 0011, 1}`, `N = {10, 00}`. The prefix-trie of
P has 7 states. One feasible NFA uses the predicate "the last symbol is `1`": a
looping start state (self-loop on `0` and `1`) with an edge on `1` to a single
accept state (2 states total). It accepts all three positives (each ends in `1`)
and rejects both negatives (each ends in `0`), scoring `0.1 * 7 / 2 = 0.35`,
above the baseline. (Illustrative only; your instance differs.)
