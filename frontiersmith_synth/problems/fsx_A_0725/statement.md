# The Frontier Ring: Quarantine on a Contact Chain

A contact network of `N` people is organised as a **chain of clusters** (workplaces or
villages), each internally dense: every cluster has one **hub** person connected to
everyone else in that cluster, so once a cluster's hub is infected the whole cluster is
exposed almost immediately. Consecutive clusters in the chain are linked by only a
handful of **gate** edges — a narrow bridge — and a few extra long-range **shortcut**
edges connect distant clusters directly, so the network is a small-world graph, not a
pure line. One large, densely-connected **decoy** cluster hangs off the far end of the
chain.

The outbreak follows deterministic SIR dynamics: an infected node is **infectious for
exactly `D` rounds**, then recovers and can never be infected again. Each round, every
susceptible node adjacent to at least one currently-infectious node becomes infected.
The outbreak starts already active at a fixed set of `seeds` (in the first cluster).

Each round you may **immunize** up to `rate_cap` currently-susceptible nodes (immunizing
a node that turns out to already be infected or recovered by then simply has no effect —
wasted budget, not an error). You have a fixed **total budget** `total_budget`, strictly
less than `rate_cap * T`, that must be **reallocated across the `T` rounds**: you cannot
spend the maximum every round, so you must decide *when*, not just *where*, to intervene
as the outbreak's frontier moves through the chain. Your goal is to **minimize the total
number of people ever infected** over the whole run.

## Program contract

Your solution is a standalone program. It reads ONE JSON object (the public instance)
from **stdin** and writes ONE JSON object (your plan) to **stdout**.

### Input (stdin)

```json
{
  "name": "outbreak4001", "N": 138,
  "edges": [[0,1], [0,2], ...],
  "seeds": [0, 3],
  "T": 16, "D": 2, "rate_cap": 3, "total_budget": 16
}
```

`edges` are undirected 0-indexed node pairs. `seeds` are the nodes infectious from round 0.
Since everything is deterministic and fully known up front, you can simulate the whole
outbreak yourself to plan your entire intervention.

### Output (stdout)

```json
{"schedule": [[ids immunized at round 0], [round 1], ..., [round T-1]]}
```

`schedule` must be a list of exactly `T` lists of integer node ids. A schedule is
**valid** iff: every id lies in `[0, N)`; no round's list contains a duplicate id; no
round's list exceeds length `rate_cap`; and the total number of ids across **all**
rounds combined does not exceed `total_budget`. Any violation (wrong shape, bad type,
out-of-range id, in-round duplicate, or a per-round/total budget overrun), a crash, a
timeout, or non-JSON output scores that instance **0.0**.

## Simulation and scoring

Per round `t = 0..T-1`, synchronously: (1) any infectious node whose `D`-round window has
elapsed recovers; (2) your immunizations for round `t` are applied to any still-susceptible
targets; (3) every susceptible node adjacent to a currently-infectious node becomes
infected (infectious for the next `D` rounds). Your objective is the total number of nodes
ever infected across the whole run — lower is better.

You are graded on **10** fixed, deterministically-generated instances. For each, the
evaluator computes:

- `noint` — total ever infected if you spend no budget at all (a weak reference),
- `ub` — the number of seed nodes (the value if the outbreak were contained the instant
  it started — an idealised, generally unreachable bound given your limited budget),
- `cand` — the total ever infected your schedule achieves,

and scores `r = clamp(0.1 + 0.9 * (noint - cand) / (noint - ub), 0, 1)`. Doing nothing
scores about `0.1`; instant containment (essentially unreachable) scores `1.0`. Your
final score is the mean of `r` over all instances.

## Notes

- Everything is deterministic; seed any randomness in your own solver for reproducible
  output. Your program runs isolated and only sees the public instance above.
- The single highest-degree node in the graph sits in the decoy cluster — think about
  whether protecting it is actually useful given where the outbreak currently is.
