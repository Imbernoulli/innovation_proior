# Staged Trainer: Curriculum Ordering for Fastest Convergence

You are the training scheduler for a **fixed** toy "student" model. You do not
change the model, the loss, or the learning rate — you only decide the
**curriculum**: the exact sequence in which training examples are shown. Reach the
target loss in as **few** updates as possible.

## The corpus and the concept DAG

There are `K` atomic **concepts** arranged in a dependency DAG. Every training
example teaches exactly one concept and carries that concept's **prerequisite**
set (concepts from shallower layers). Root concepts (layer 0) have no
prerequisites; deeper concepts stack on shallower ones. Each concept has several
example "reps", and the corpus arrives **shuffled**.

## The frozen training dynamics (deterministic)

Every concept `k` has a mastery `m_k ∈ [0,1]`, all starting at 0. Showing an
example that teaches concept `c` with prerequisite set `P` applies **one** update:

```
readiness = 1.0                    if P is empty
          = min(m_p for p in P)    otherwise
m_c      += LR * readiness * (1 - m_c)
```

An update on a concept whose prerequisites are unmastered barely moves it
(`readiness ≈ 0`); a well-prepared concept jumps toward mastery with diminishing
returns. After each update the **loss** is the mean immaturity:

```
loss = (1/K) * sum_k (1 - m_k)
```

Training halts at the **first** update whose loss `<= target`. That 1-based update
index is your convergence step count. You may re-show any example any number of
times (like revisiting a topic across epochs).

## Candidate program contract (stdin → stdout)

Read ONE JSON object (the public instance) from stdin and write ONE JSON object to
stdout.

**Input (stdin):**
```json
{"name": str, "K": int, "n_examples": N, "LR": float, "target": float, "cap": int,
 "examples": [{"concept": c, "prereqs": [ ... ], "layer": l}, ...]}
```
`examples[i]` is the example with id `i` (the shuffled arrival order). `LR`,
`target`, and the full DAG are given, so you may simulate the dynamics yourself.

**Output (stdout):**
```json
{"schedule": [e_0, e_1, ...]}
```
`schedule` is a non-empty list of at most `cap` integers, each a valid example id
in `[0, N-1]`; repetition is allowed. The model is shown
`examples[e_0], examples[e_1], ...` in order.

A non-list schedule, wrong element type/range, over-length (`> cap`), a crash, a
timeout, or non-JSON makes that instance score **0.0**. A valid schedule that
never reaches the target within its length (or the cap) also scores 0.0.

## Objective and scoring (minimize convergence steps)

For each instance the evaluator computes, under the same frozen dynamics:

- `q_lb` — an **optimistic lower bound**: ignore prerequisites (`readiness = 1`
  always) and greedily update the least-mastered concept each step. No real
  curriculum can beat it — an unreachable ideal.
- `q_base` — the natural **as-shipped** order cycled `0,1,...,N-1,0,1,...` (weak
  reference).
- `q_cand` — your schedule's convergence step count.

Your normalized per-instance score is
```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```
Reproducing the as-shipped order scores ≈ 0.1; matching the (generally
unreachable) optimistic bound scores 1.0; doing worse than the natural order
scores below 0.1. The final score is the mean of `r` over all instances. Because
`q_lb` ignores the dependency tax every real schedule must pay, even strong
simulation-driven curricula stay well below 1.0 — there is headroom.

## Strategy ladder

- **As-shipped**: cycle the shuffled arrival order (≈ 0.1 baseline).
- **Static difficulty sort**: order easy-to-hard by concept layer and cycle it, so
  prerequisites precede dependents within each pass.
- **State-aware greedy**: simulate the student and always show the example with the
  largest immediate loss reduction `LR * readiness * (1 - m_c)`.
- **Search / lookahead**: perturb a good schedule (swap/insert re-shows, beam or
  rollout over the deterministic simulator) to shave updates toward — but never
  reaching — the optimistic bound.

The candidate runs in an isolated sandboxed subprocess and only ever sees the
public instance; the references and the authoritative simulation run in the
evaluator.
