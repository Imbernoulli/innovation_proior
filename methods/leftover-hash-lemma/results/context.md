## Fragile Perfect-Randomness Assumption

Randomized algorithms and cryptographic protocols are usually proved under an ideal input model: the random bits are uniform, independent, and hidden from any party that should not know them. Physical sources can be biased or correlated, and cryptographic secrets can be partly exposed before they are used.

The operational need is not merely to assign a large entropy number to the source. The output must be close enough to uniform that replacing ideal random bits by the processed source changes every downstream event by only a small amount.

## One-Sample Weak Sources

The source model should allow only partial knowledge of the distribution. A realistic procedure cannot be tuned to the exact probabilities of every possible source value; it should work from a lower bound on worst-case uncertainty.

Min-entropy captures that worst case. If no value occurs with probability above `2^-k`, then no single guess succeeds too often. This is stricter than Shannon entropy, which can remain large even when one value carries a large mass.

## Deterministic Extractors

A fundamental result shows that a single deterministic function cannot turn every high-min-entropy source into a nearly uniform output. For any fixed output bit, one of its preimages is large. A source uniform on that preimage can still have substantial min-entropy while making the output constant. This impossibility is a known theorem in the randomness extraction literature.

One avenue is to allow a short public seed that selects among a family of processing functions. The combined output—seed and processed value—should then look close to uniform even when the seed is known.

## Public Randomness in Extraction

Families of hash functions provide a structured way to implement this public-seed approach. A family is called 2-universal if for all distinct inputs `x, x'`, a uniformly chosen member `H` satisfies

```text
Pr_H[H(x) = H(x')] <= 1/|range(H)|.
```

Such families can be described compactly and evaluated efficiently, and their collision probabilities are well understood analytically.

## Statistical Distance and the Evaluation Target

The target guarantee for any extraction procedure is stated in terms of statistical distance. If `S` denotes the public seed and `Y` the processed output, then `(S, Y)` should have small statistical distance from `(S, U_m)`, where `U_m` is uniform over `{0,1}^m` and independent of `S`.

The parameters to balance are the source min-entropy `k`, the output length `m`, and the tolerated statistical error `epsilon`. Applications such as privacy amplification require the public seed to remain visible without reducing the quality of the extracted output.
