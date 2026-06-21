# Channel Coding Theorem

For a discrete memoryless channel `W(y|x)`, define the mutual information induced by input distribution `p(x)` as

`I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)`.

The channel capacity is

`C = max_{p(x)} I(X;Y)`.

A rate `R` is reliably achievable if there are block codes of length `n` with about `2^{nR}` messages whose decoding error probability tends to zero as `n` grows.

The theorem states:

- Every `R < C` is reliably achievable.
- No `R > C` is reliably achievable.

The achievability proof uses random coding. Fix `p(x)`, draw `2^{nR}` independent length-`n` codewords, and decode by the unique codeword jointly typical with the received block. The transmitted codeword is jointly typical with the output with probability tending to one. Any wrong codeword is independent of that output and is jointly typical with probability about `2^{-n I(X;Y)}`. The union bound over wrong codewords vanishes when `R < I(X;Y)`. Therefore some deterministic codebook in the random ensemble has small error. Maximizing over `p(x)` gives all rates below `C`.

The converse uses the fact that reliable decoding leaves vanishing message uncertainty:

`H(M|Y^n) = o(n)`.

Thus

`nR = H(M) <= I(M;Y^n) + o(n) <= I(X^n;Y^n) + o(n) <= nC + o(n)`,

so `R <= C` in the limit.

For a band-limited additive white Gaussian noise channel with bandwidth `W`, signal power `P`, and noise power `N`, the corresponding capacity is

`C = W log_2(1 + P/N)` bits per second.

The result is an existence theorem: it proves that suitable long codes exist and that higher reliable rates are impossible, but it does not by itself give a practical low-complexity code construction.
