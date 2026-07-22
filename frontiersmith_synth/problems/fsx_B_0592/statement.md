# Drift-Aware Heavy-Hitter Budget

You run a **fixed-memory streaming summarizer**. A long token stream is processed
online, but your memory holds at most **K** token ids. When the observation window
closes you must commit the set of tokens you are still "holding". The stream then
**continues in a part you never see**, and the *eventual* heavy hitters — the top-**M**
tokens by total frequency over the whole stream — are decided partly by that unseen
continuation. A token you dropped is gone forever. Your goal: keep the tokens that
will turn out to be the heavy hitters.

The catch is **drift**. Somewhere in the observed prefix the distribution changes.
Early on, a set of *leader* tokens dominates — then they die off. After the change,
new tokens begin to appear. Some are **decoys**: they spike early after the drift,
then fizzle in the unseen tail. Others are the **true risers**: still small when the
window closes, ramping late and accelerating, they surge in the hidden continuation to
become the real heavy hitters. Counting alone cannot tell them apart — at window close
the stale leaders and the fizzling decoys have far larger counts than the true risers.

## Input (one public instance, JSON on stdin)
```
{
  "K":      <int>,           # memory budget: your kept set may hold at most K token ids
  "M":      <int>,           # number of heavy hitters that will be graded
  "stream": [<int>, ...]     # the OBSERVED PREFIX of the stream, in arrival order
}
```
The stream elements are integer token ids. The hidden suffix, and therefore the true
global top-M, are **not** provided — you must infer which tokens will matter from the
prefix alone.

## Output (JSON on stdout)
```
{ "keep": [<int>, ...] }     # the token ids you retain in memory (order irrelevant)
```
Feasibility (any violation scores 0 for that instance):
- `len(keep) <= K`;
- every id in `keep` is a **distinct** token that **actually appears** in `stream`
  (bounded memory cannot hold a token you never observed).

## Scoring — MAXIMIZE
Let `TrueTopM` be the M tokens with the highest total count over the full
(prefix + hidden suffix) stream. Your per-instance score is the **recall**
```
        | keep ∩ TrueTopM |
score = --------------------              in [0, 1]
                 M
```
The final Ratio is the mean over 10 seeded instances. Everything is deterministic:
the same `keep` always yields the same score.

## What shapes the score
- **The stale-leader trap.** Keeping the K most-frequent-so-far tokens burns the whole
  budget on pre-drift leaders and early decoy spikes — tokens with big prefix counts and
  no future — so the true risers, still small at window close, are never retained.
- **Reading the drift.** The change point is observable in the prefix (the leader set
  loses its grip; new tokens arrive). Down-weighting what happened *before* the drift, and
  favouring tokens whose mass is **recent and rising**, separates the accelerating true
  risers from the already-fizzling decoys.
- **Budget under uncertainty.** Because the surge magnitudes live in the hidden suffix, the
  exact top-M cannot be read off the prefix. Spending part of the budget as a reserve for
  freshly-emerging, up-trending tokens hedges against that uncertainty; committing every
  slot to the current front-runners does not.

Time limit 2–5 s, memory 512 MB.
