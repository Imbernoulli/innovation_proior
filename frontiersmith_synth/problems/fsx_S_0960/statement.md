# Commit-Cut: Streaming Panel Orders on a Kerf-Locked Sheet

## Story
A panel shop cuts custom rectangles from one big sheet of stock, `W` cells
wide and `H` cells tall. Orders arrive one at a time, in a fixed sequence,
each a panel `w x h` worth `value`. Your program is handed the **whole order
book up front** — nothing about future orders is hidden — but every decision
is still graded as if made on arrival and **irrevocably**: decision `i` is
validated against a sheet that has already absorbed decisions `0..i-1` and
can never be revised. The challenge is not "guess what's coming" — it is the
geometric/combinatorial one of committing cuts, in causal order, that a real
saw could execute without ever backtracking. At its turn, each order is
either cut out now, or refused for a small fixed forfeiture `discard_cost`.
No reconsidering later, no combining discarded orders.

Cutting a panel is not free of side effects: the saw needs **kerf clearance**
`k` on every side of the cut that passes through the *interior* of the
remaining stock (a cut flush against the sheet's own outer edge needs no
clearance there). Once placed, both the panel and its clearance buffer are
**permanently locked**, even if that later turns out to have been a poor use
of space.

Your goal: maximize total value collected, net of forfeiture costs.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE
JSON answer to `stdout`. It runs in an isolated subprocess and sees only the
public instance below.

### Public instance (stdin)
```json
{ "W":22, "H":16, "k":1, "discard_cost":0.5,
  "n":18, "items":[ {"w":2,"h":4,"value":8.1}, ... ] }
```
`items` is given in **arrival order**; your answer's decisions correspond to
it **positionally**. No identifying name or seed is sent — only raw geometry —
so a policy has to actually read `items` rather than special-case a known id.

### Answer (stdout)
```json
{ "decisions": [
    {"action":"place","x":3,"y":0,"rot":0},
    {"action":"discard"},
    ...
] }
```
Exactly `n` decisions, one per order, in arrival order. `"place"` gives the
panel's lower-left cell `(x,y)` on a `W x H` integer grid; `rot=0` keeps
`(w,h)`, `rot=1` uses `(h,w)`.

## Replay (deterministic, causal)
The evaluator starts with an empty `W x H` grid and processes your decisions
**in order**. For a `"place"` decision on order `i` with rotated size
`(dw,dh)`: the footprint `[x,x+dw) x [y,y+dh)` must lie in bounds, and the
footprint **inflated by `k` on every side (clipped to the sheet's own
boundary)** must not overlap any cell locked by an earlier decision. If legal,
both the footprint and its clipped buffer lock and `value` is added to your
score. If illegal — out of bounds, overlap, bad `rot`, wrong types — **the
entire instance scores 0**: a plan is either fully executable or it is not a
real plan. `"discard"` simply subtracts `discard_cost` and moves on. A
malformed `decisions` list, a crash, a timeout, or non-JSON output also
scores 0.

## Scoring (deterministic)
Per instance the evaluator computes:
```
gained = sum(value of placed orders) - discard_cost * (# discarded orders)
relax  = sum(value_i for i where the panel fits SOME orientation on an
             *empty* sheet, i.e. (w<=W and h<=H) or (h<=W and w<=H))
```
`relax` is a valid upper bound — an order that cannot fit alone on an empty
sheet can never be accepted under any policy — but it is deliberately loose:
it ignores that orders compete for the same sheet, that kerf eats real space
between neighbours, and that arrival order can make otherwise-compatible
orders mutually unreachable. Your score on the instance is
`clamp(gained / relax, 0, 1)`. The final score is the mean over **10** fixed
seeded instances.

## Why it is open-ended
Filling every legal spot the instant an order arrives (bottom-left-fill,
never refuse) is the obvious policy — but it fragments the sheet with
kerf-locked slivers on cheap early orders, and by the time a rare, high-value
order shows up, no contiguous room of the right shape is left. Reserving a
generous corridor for the value tail is *also* a trap: reserve too much and
you refuse orders you could easily have kept; too little and the value tail
still cannot land. Genuine policies size a reservation to the order book's
own mixture and treat every discard as a deliberate purchase against orders
not yet placed.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the
public instance above; `relax` and all grading happen in the evaluator
process.
