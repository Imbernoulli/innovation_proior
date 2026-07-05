# Greenway Depot: Two-Constraint Collection Routes

## Story
The Greenway municipal recycling depot dispatches identical collection trucks to
clear a stream of curbside containers. Every container has **two independent limits**
that matter at once:

- **mass** — kilograms of compacted recyclables (crushed cans are dense), and
- **bulk** — litres of loose volume (foam and flattened cardboard are light but huge).

Each truck can carry at most **W kilograms** *and* at most **V litres**; a load is
legal only when **both** limits hold. Sending a truck out on a route costs one
dispatch (a full loop of the depot circuit plus a tipping fee). You want to collect
**every** container using **as few dispatched trucks as possible**.

This is 2-D *vector* bin packing: containers are items with a `(mass, bulk)` vector,
a truck is a bin with capacity vector `(W, V)`, and "trucks dispatched" is the number
of non-empty bins — which you minimize.

## Task
You write a **standalone program**: read ONE JSON instance from `stdin`, write ONE
JSON answer to `stdout`.

### Public instance (stdin)
```json
{
  "name": "route701",
  "W": 100,
  "V": 100,
  "n": 26,
  "mass": [ ... n integers, each 1 <= m_i <= W ... ],
  "bulk": [ ... n integers, each 1 <= b_i <= V ... ]
}
```

### Answer (stdout)
```json
{ "assign": [ t_0, t_1, ..., t_{n-1} ] }
```
`t_i >= 0` is the truck index that container `i` is loaded onto. Truck indices need
not be contiguous; a truck **exists** iff at least one container loads onto it, and
the number of **distinct non-empty trucks** is your dispatch count.

### Validity
A plan is valid iff `assign` is a list of exactly `n` non-negative integers and, for
every truck, the total mass `<= W` **and** the total bulk `<= V`. Any violation —
wrong length, a non-integer / negative index, an overloaded truck, a crash, a
timeout, or non-JSON output — scores **0.0** on that instance.

## Objective & scoring (deterministic)
Let `q_cand` be the number of trucks your plan dispatches. The evaluator computes,
per instance:

- `q_lb  = max( ceil(sum(mass)/W), ceil(sum(bulk)/V) )` — the L1 *vector* lower bound
  (generally unreachable), and
- `q_base` — the trucks used by a weak internal **next-fit** operator (fill the
  current truck until a container breaks either limit, then open a new one).

Your per-instance score is
```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```
so matching next-fit scores ~0.1, reaching the L1 bound scores 1.0, and doing worse
than next-fit scores below 0.1. The final score is the mean of `r` over a fixed
family of 12 seeded instances (mass/bulk distributions vary: balanced,
anti-correlated, mass-bound, bulk-bound, and spiky, plus larger held-out cases).

Because the L1 vector bound is loose — packing tightly on mass can strand bulk and
vice versa — even strong decreasing-order packers stay below 1.0 on most instances.
There is no easy optimum; multiple strategies (first-fit, best-fit-decreasing on
various keys, local search) are viable and trade off differently across the family.

## Isolation
Your program is run in a fresh sandboxed subprocess and only ever sees the public
instance above. The lower bound and baseline are computed by the evaluator process,
which your program cannot reach.
