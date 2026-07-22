# Slipstream Couriers: Convoy Formation and Lead Rotation

A courier company runs `K` cyclists across a city street network. You plan a **timed route**
for every courier. Riding in a shared slipstream (a *peloton*) saves energy, but someone has to
sit in front and pull — and pulling for too long is punishing. Minimize the **total energy** the
whole fleet burns while every courier still meets its deadline.

## Model

The city is an undirected weighted graph on `N` nodes with `M` edges; edge `(u,v)` has a positive
base energy `w`. Time advances in integer **ticks** `0,1,2,...`. At each tick a courier either
**waits** at its current node (0 energy) or **traverses** one incident edge (one tick). Multiple
couriers may occupy the same node.

When a courier traverses a directed edge at a tick it declares a role, **L**ead or **D**raft.
For a fixed directed edge `e` and tick `t`, let the *convoy* be the set of couriers traversing `e`
at `t`. Energy charged for that edge, per courier:

- **Solo** (convoy size 1): `w` (the role is ignored).
- **Convoy** (size ≥ 2):
  - each **leader** pays `w · (1 + α · s)`, where `s` is that courier's current **lead streak** —
    the number of *consecutive* ticks it has been a convoy leader, counting this one;
  - each **drafter** pays `w · β` if the convoy has at least one leader, otherwise `w`.

A courier's lead streak resets to 0 on any tick where it waits, rides solo, or drafts. The
constants `α` (streak penalty) and `β` (draft factor, `0<β<1`) are given in the input. Because the
streak grows, a convoy whose front never changes can cost **more** than riding solo; rotating the
lead keeps every streak at 1.

## Input (stdin)

```
N M K
α β
u v w        (M lines, undirected edge, w > 0)
s g d        (K lines: start node, target node, deadline)
```

## Output (stdout)

`K` route blocks (any order), each:

```
i  T  a_1 n_1  a_2 n_2  ...  a_T n_T
```

`i` = courier index (`0..K-1`, each exactly once). `T` = number of steps (`T ≤ d_i`). Step `j` is a
letter `a_j ∈ {W,L,D}` and the node `n_j` occupied after it. The courier starts at `s_i`; for `W`,
`n_j` must equal the current node; for `L`/`D`, `(current, n_j)` must be an edge. After step `T`
the courier must be at `g_i`.

## Feasibility

Any of: a missing/duplicate courier, `T > d_i`, an illegal wait, a move along a non-edge, a node id
out of range, or not ending at the target ⇒ the whole submission scores **0**.

## Objective & Scoring

Let `F` be the exact replayed total energy (sum over all couriers, all traversals; waits are free).
The checker builds `B` = the total energy of the independent shortest-energy routes (every courier
alone). Score:

```
Ratio = min(1000, 100 · B / F) / 1000
```

Independent shortest paths give `F = B ⇒ Ratio ≈ 0.1`. Lower `F` scores higher; a 10× reduction
caps at 1.0. Drafting is capped per edge (`β`), so the achievable ceiling is well below 1.0 —
headroom is deliberate.

## Example (illustrative)

Two couriers whose shortest routes overlap for 4 blocks: riding independently they cross those
blocks at different ticks (their spur delays differ), so both pay full energy. If instead one waits
a couple of ticks so they enter the shared blocks together and they swap the lead each block, the
follower drafts and neither builds a streak — total energy drops below `B`, so `Ratio > 0.1`.

## Constraints

`α, β` and all weights are given per instance. Time limit 5 s, memory 512 MB.
