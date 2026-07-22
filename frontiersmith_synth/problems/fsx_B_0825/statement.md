# Night Custodian: Emptying Rows Between Acts

## Problem
A theater has `N` numbered rows. Row `i` currently holds some patrons; patron
records for row `i` are given as a list of `k_i` **checkout rounds** — the round
at which that particular patron will get up and leave on their own (or `-1` if
they will stay for the whole run). These checkout times are exactly known in
advance (ticketing data). A patron who has not yet checked out by round `t` is
still occupying their seat at round `t`.

The theater also has a stock of spare rows in the back room; the stock starts
at `F0` rows and must never drop below a hard floor `Fmin`. At certain known
future rounds (the **demand rounds**), a fresh group of patrons arrives and the
box office must open one brand-new row from the stock to seat them, consuming
one stock row at that exact round.

Between acts (each round `1..T` hosts at most one intermission), the night
custodian may **sweep** at most one row. Sweeping row `i` at round `t` personally
escorts out every patron still seated there at that moment (any patron whose
checkout round is `-1` or `> t`) — this costs one unit of effort per escorted
patron — and afterwards the *entire* row (every seat, occupied or not) is
returned to the back-room stock, regardless of how many patrons needed
escorting. A row may be swept at most once; a row that is never swept is left
alone. Waiting to sweep a row never increases its cost (patrons only ever
leave, never arrive back), but every round the stock stays too low risks a
future demand round breaching the floor.

## Input (stdin)
```
N T F0 Fmin
k_1 d_{1,1} d_{1,2} ... d_{1,k_1}
...
k_N d_{N,1} ... d_{N,k_N}
D
e_1 e_2 ... e_D
```
`d_{i,j}` is a checkout round in `[1,T]`, or `-1` for "stays all run". `e_1<...<e_D`
are the strictly increasing demand rounds (each in `[1,T]`, each occurs once).

## Output (stdout)
```
K
t_1 g_1
...
t_K g_K
```
`K` sweep actions (`0 <= K <= N`); each `(t_j, g_j)` sweeps row `g_j`
(`1<=g_j<=N`) at round `t_j` (`1<=t_j<=T`). Every `t_j` must be distinct
(one sweep per round) and every `g_j` must be distinct (each row swept once).

## Feasibility
Replay rounds `1..T` in order. At round `t`: if the plan sweeps a row there,
add its currently-seated-patron count to the cost and add 1 to the stock;
then, if `t` is a demand round, subtract 1 from the stock. The plan is
feasible iff the stock never drops below `Fmin` at any round, and the output
is syntactically well formed (integers only, ranges respected, no repeats).
Any violation, or any non-finite/malformed token, scores `0`.

## Objective
Minimize the total escorted-patron cost across all sweeps.

## Scoring
The checker builds its own baseline plan `B`: sweep rows in plain numeric
order `1,2,3,...`, each exactly at the earliest round a sweep becomes
unavoidable (ignoring who is actually seated there). Let `F` be your feasible
plan's total cost. Then
```
Ratio = min(1, 0.1 * B / F)
```
Matching the naive baseline scores `0.1`; halving it doubles the ratio.

## Constraints
- `1 <= N <= 11`, `1 <= T <= 22`, `1 <= F0,Fmin <= 5`.
- `0 <= k_i <= 20`, each seat count and demand count small; input `<= 5MB`.
- Deterministic integer scoring; no timing.

## Example
Rows 1 and 2 both hold live patrons; a demand round forces one sweep before
either empties on its own. Sweeping row 1 (2 live patrons) costs 2; sweeping
row 2 (5 live now, but 4 check out together two rounds later) costs 5 if
swept immediately, but only 1 if the custodian sweeps row 1 now and defers
row 2 past that checkout cluster — banking the stock slack needed to wait.
"Sweep whichever row is emptiest right now" gets this round right, but on a
later, tighter round it is forced to sweep a still-full clustering row
before it empties, losing exactly the gap a lookahead schedule avoids.
