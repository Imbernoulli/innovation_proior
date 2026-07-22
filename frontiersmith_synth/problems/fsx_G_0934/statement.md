# Small Automaton, Big Generalization: Inducing a DFA from Labeled Strings

## Story

A protocol-analysis team recorded traces of interactions with several black-box,
binary-alphabet devices. For each device they logged a finite **training** sample:
strings over `{0,1}` together with a label — did the device end the interaction in
an "accept" state (`1`) or a "reject" state (`0`)? Every device is internally driven
by a small, deterministic finite automaton, but the team never sees it, only the
labeled traces. Your job: from the training sample alone, output a DFA that
reproduces the device's true accept/reject rule well enough to classify a fresh batch
of traces from the same device that you were never given labels for — drawn from the
same distribution as the training log, but for several devices skewed toward traces
considerably longer than anything you trained on.

Memorizing the training strings exactly is easy and gets every training example
right. But it teaches you nothing about strings that diverge from the training log
even by one early symbol, and it needs a state for nearly every distinct prefix you
observed. These devices are actually driven by *small* automata; an over-large
memorized automaton generalizes badly the moment a held-out trace runs off the edge
of what you memorized. The strategy that works: treat the training sample as
**evidence** for which prefixes secretly reach the same underlying device state, and
merge them — accepting a merge only when it stays consistent with every labeled
example. Compressing correctly recovers a small automaton that predicts the unseen
traces, not just the seen ones.

## Input (stdin): ONE JSON object — the PUBLIC training sample

```json
{"name": "device_02", "alphabet": ["0", "1"],
 "train": [{"s": "0110", "label": 1}, {"s": "1", "label": 0}, ...]}
```

`train` is a list of records; `s` is a string over `"0"`/`"1"` (possibly empty),
`label` is `0` or `1`. The held-out test traces, and the true device automaton, are
never sent to you. The held-out set is sampled fresh from the same distribution as
`train` and is not specially scrubbed of literal repeats — for the shorter devices a
training string can legitimately recur — but it is never told to you, and several
devices weight it toward lengths far beyond anything you trained on, where a repeat
is essentially impossible.

## Output (stdout): ONE JSON object — your induced DFA

```json
{"delta": [[1, 2], [1, 3], ...], "start": 0, "accept": [3, 5]}
```

- `delta` is a list of `n` rows (`n` = the number of states you declare), each row
  `[next_on_"0", next_on_"1"]` — your DFA must be **complete** (both transitions
  defined for every state).
- `start` is the index (`0..n-1`) of the initial state.
- `accept` is the list of accepting state indices.

A submission is **valid** iff `delta` is a non-empty list of 2-element integer rows
whose entries are all valid state indices, `start` is a valid index, and every entry
of `accept` is a valid index. Anything else — a crash, a timeout, non-JSON, `null`,
a wrong type, or an out-of-range index — scores **0.0** on that device.

## Objective — MAXIMIZE

For each device the evaluator computes, on the SAME held-out trace set:

- `a_triv`   — accuracy of the majority-class rule (always predict the more common
  training label) — the weak anchor,
- `a_oracle` — accuracy of the true device automaton — the strong anchor (always
  very close to `1.0`, by construction),
- `a_cand`   — accuracy of **your** submitted DFA,

and combines that with a compactness term. Let `s` be the number of your states that
are actually **reachable** from `start` (padding with unreachable states buys
nothing), and let `m` be the true device's minimal state count (never revealed).
Then:

```
acc_r  = clamp( (a_cand - a_triv) / max(a_oracle - a_triv, 0.05),  0, 1 )
size_r = m / max(s, m)                # 1.0 once s <= m; shrinks as s grows past m
r      = clamp( 0.1 + 0.75 * acc_r * (0.5 + 0.5 * size_r),  0, 1 )
```

Matching the majority-class baseline scores `r ≈ 0.1` on that device; recovering
something behaviorally close to the true device *and* close to its minimal size
pushes `r` toward `1.0`. A DFA that classifies well but carries hundreds of
superfluous states is capped well below the score a compact, equally-accurate DFA
would earn — correctness alone is not enough, the automaton must also be small. The
reported **Ratio** is the mean of `r` over the whole device bank (10 devices), so a
heuristic must generalize across the *whole* bank, not just the easy devices.

## Determinism & isolation

Scoring is fully deterministic — every device is seeded, and the model has no
randomness or wall-time dependence. Your program runs isolated in a fresh sandboxed
subprocess and only ever sees the public training sample; the held-out traces and the
true device automaton stay in the evaluator process.
