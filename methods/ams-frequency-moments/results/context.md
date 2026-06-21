## A Stream Becomes a Frequency Vector

A data set arrives as a sequence `a_1, a_2, ..., a_m`, each item drawn from a large universe `{1, ..., n}`. After the stream ends, item `i` has appeared `m_i` times, so the stream defines a frequency vector `(m_1, ..., m_n)`. For `k > 0`, its moments are

`F_k = sum_i m_i^k`.

The first few moments have different meanings. `F_0` is the number of distinct values, `F_1` is the stream length, and `F_2` is the sum of squared frequencies. Higher moments increasingly emphasize heavy values.

## Why The Squared Moment Matters

The squared moment measures skew. If a relation has join-key value `i` appearing `m_i` times, then a self-join on that key contributes `m_i^2` pairs for that value, and the whole self-join size is `sum_i m_i^2`. Query optimizers and parallel data systems use this number because skew changes partitioning decisions, join plans, and load balance.

The estimate is most useful while records are being inserted or scanned, rather than after a full materialization. The model is therefore one left-to-right pass, with a small working memory that updates as each record arrives.

## The Histogram Baseline

With enough memory, the problem is direct: keep a counter for every universe value, increment `m_i` when item `i` arrives, and sum powers at the end. This costs `Theta(n log m)` bits for exact counters. Compressing each counter probabilistically still keeps one register per possible item, so the dependence on `n` is linear.

## Earlier Small Random Summaries

Two earlier successes show that randomness can replace exact state in special cases. Approximate event counters store a small logarithmic-scale register and randomize increments, giving controlled relative error for the stream length using far fewer bits than an exact counter. Distinct-counting methods hash each value and watch rare bit patterns, so duplicates do not change the summary and the number of distinct values can be inferred from the largest observed rarity level.

These methods address `F_1` (total length) and `F_0` (distinctness). The squared moment `F_2` depends on how often equal values collide with themselves, not merely on total length or distinctness.

## The Estimation Question

The target is a one-pass randomized relative-error estimate: output `Y` such that `|Y - F_k| <= lambda F_k` with failure probability at most `delta`, using space below the full histogram. The central case is the squared moment, and the surrounding question asks which moments can be estimated cheaply and which require polynomial or linear space.
