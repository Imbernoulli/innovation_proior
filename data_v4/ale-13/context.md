# String Reassembly

## Research question

We are given a bag of `n` short text **fragments** and must reassemble them into a
single string `T` — a **common superstring** — that contains *every* fragment as a
contiguous substring. Among all such superstrings, we want `T` to be as **short**
as possible. This is the **Shortest Common Superstring (SCS)** problem: it models
shotgun fragment assembly (the fragments are overlapping reads of one hidden
source) and data-compression by overlap. SCS is NP-hard and APX-hard, so there is
no known efficient exact solution at this scale; the benchmark scores a submission
by *how short* a feasible superstring it produces rather than by matching a unique
optimum.

Phrased structurally: a superstring is determined (up to which overlaps are taken)
by an **order** in which the fragments are laid down left-to-right; placing
fragment `b` right after fragment `a` lets `b` start inside `a` by exactly the
length of the longest suffix of `a` that is also a prefix of `b` (their
**overlap**). The length of the superstring built from an order `p` is

```
len(T) = (sum of all fragment lengths) - (sum of overlaps of consecutive pairs in p).
```

The first term is a constant, so **minimizing the superstring length is exactly
maximizing the total overlap of consecutive fragments** — a maximum-weight
Hamiltonian path on the (asymmetric) overlap graph. That reformulation is the lever
the whole solver turns on.

## Input / output contract

- **Input (stdin):** the first line is `n s`, where `n` is the number of fragments
  (`60 ≤ n ≤ 400`) and `s` is the number of distinct lowercase-letter symbols that
  occur across all fragments (`2 ≤ s ≤ 4`). Then `n` lines follow, each a non-empty
  fragment over the first `s` lowercase letters; fragment lengths lie in `[12, 40]`.
  No fragment is a substring of another (redundant fragments are filtered out at
  generation time), and the fragments are presented in shuffled order.
- **Output (stdout):** a single line holding the superstring `T` (the reassembly).
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff `T` uses only symbols that appear in the input
fragments **and** every input fragment is a contiguous substring of `T`. Anything
else — a missing fragment, a stray symbol outside the alphabet, an empty line, a
missing file — is **infeasible**.

## Background

Because the order alone determines the assembly, several approaches sit on the
table before committing:

- **Trivial concatenation.** Just print all fragments back-to-back. Always
  feasible, length `= Σ|fragment|`, takes zero overlap — the obvious but very loose
  baseline (the scorer's reference point).
- **Greedy max-overlap merge.** Repeatedly join the two currently *open* chain ends
  with the largest pairwise overlap, refusing any join that would close a cycle,
  until a single chain remains. This is the **classic greedy SCS algorithm**; it is
  a constant-factor approximation (a 3.5-approximation is proven and a factor-2 is
  conjectured) and in the assembly regime it recovers most of the achievable
  overlap immediately. The overlaps are precomputed once as a dense `n × n` table.
- **Reordering local search (the decisive lever).** Greedy fixes adjacencies
  early and can strand a few fragments behind a slightly-better-looking merge. Since
  length `= const − Σ consecutive-overlap`, we treat the greedy result as an
  *initial order* and run local search to push the total consecutive overlap up:
  **Or-opt** (relocate a run of 1–3 fragments to a better slot) and a bounded
  **2-opt** segment reversal. Each move changes only a handful of adjacencies, so
  its effect on the total overlap is an **O(1) incremental delta** over exactly the
  broken and created edges — the superstring is never rebuilt to evaluate a
  candidate. When local search stalls we apply a small **double-bridge-style kick**
  and continue, keeping the best order seen, all under a wall-clock budget.

The engineering lever is precisely this incremental delta plus the order
reformulation: feasibility is automatic for *any* order (merging consecutive
fragments by their overlap keeps every fragment a contiguous substring), so the
search can move freely while always holding a valid answer.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`)
  builds a hidden low-entropy "source" string from a handful of repeated motifs
  interleaved with random filler over a small alphabet, then samples many
  overlapping substrings of it as the fragments, de-duplicates them, and removes
  any fragment that is a substring of another. The source is never revealed; the
  solver sees only the shuffled fragments. Small alphabets and motif repeats make
  overlaps plentiful — exactly the regime where greedy-plus-reordering pays off.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the
  submitted superstring `T`.
  - **Feasibility floor:** if `T` contains a symbol outside the fragment alphabet,
    or if any input fragment is not a contiguous substring of `T` (or the file is
    missing/empty), the score is **`0`**.
  - Otherwise let `Lsol = len(T)` and let `Lbase = Σ|fragment|` be the length of
    the deterministic trivial-concatenation reference superstring (recomputed
    inside the scorer, so the reference is reproducible and independent of the
    solver). The score is

    ```
    score = round( 1 000 000 × Lbase / Lsol )      (feasible, Lsol > 0)
    score = 0                                       (infeasible)
    ```

    A higher score is better. The trivial concatenation scores exactly
    `1 000 000`; any shorter feasible superstring scores strictly more.
- **Reported metric.** The mean score over a fixed seed set. A genuine
  greedy-plus-reordering solver lands far above `1 000 000` (≈ 2.0–4.0× on these
  instances); the trivial concatenation is the `1 000 000` floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a
feasible superstring to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, s;
    if (scanf("%d %d", &n, &s) != 2) return 0;
    { int c; while ((c = getchar()) != '\n' && c != EOF) {} }
    vector<string> frag(n);
    for (int i = 0; i < n; i++) {
        string line; int c;
        while ((c = getchar()) != EOF && c != '\n') if (c != '\r') line.push_back((char)c);
        frag[i] = line;
    }

    // A feasible answer is ANY string that contains every fragment as a substring.
    // The trivial concatenation is always feasible, so start from it.
    string T;
    for (auto& f : frag) T += f;

    // TODO heuristic: drop redundant fragments, precompute pairwise overlaps,
    // greedy max-overlap merge to an initial ORDER, then Or-opt + bounded 2-opt
    // local search that MAXIMIZES total consecutive overlap (== minimizes length)
    // using O(1) incremental deltas, with double-bridge kicks under a ~2s budget;
    // finally rebuild T by overlap-merging the fragments in the chosen order.

    fputs(T.c_str(), stdout);
    fputc('\n', stdout);
    return 0;
}
```
