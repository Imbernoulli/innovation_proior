# Context — the ground a theory of lossy compression stands on

## Research question

When a source must be reproduced *approximately* rather than exactly, how few bits per symbol does it fundamentally take? Lossless source coding answers the exact-reproduction question — a discrete memoryless source needs and suffices with `H(X)` bits per symbol. But two large classes of source defeat exact reproduction outright. A continuously distributed quantity carries an actual infinity of bits in its decimal expansion, so reproducing it exactly needs an infinite-capacity channel; any real (noisy, finite-capacity) channel cannot do it. And even for a discrete source, exact reproduction is often more than anyone wants to pay for: a photograph, a voice waveform, a sensor reading need only be reproduced *closely enough* for the use at hand.

So the question is not "how many bits to reproduce `X` exactly" but "how many bits to reproduce `X` to within a stated tolerance." We need (i) a way to *measure* tolerance — a fidelity criterion that says how bad a given reproduction is — and (ii) a theorem of the same character as the lossless source-coding theorem: a single number `R` that is the true rate of the source *at that tolerance*, achievable by some encoding and impossible below.

## Background

The field at this point is built on a small, sharp set of tools, all from the theory of communication established a decade earlier.

**Entropy and the lossless source-coding theorem.** A discrete memoryless source emitting letter `i` with probability `P_i` has entropy `H(X) = −Σ_i P_i log₂ P_i`. The source-coding theorem says `H(X)` is simultaneously the floor and the achievable rate of lossless coding: long blocks can be coded at `H(X)+ε` bits/symbol with vanishing error, and not below `H(X)`. The engine behind it is the **asymptotic equipartition property**: a length-`n` block is, with probability `→1`, "typical," and there are only about `2^{nH(X)}` typical sequences, each of probability about `2^{−nH(X)}`. Coding then amounts to indexing the typical set.

**Mutual information.** For a joint distribution `p(x,y)`, the mutual information `I(X;Y) = Σ p(x,y) log₂ [p(x,y)/(p(x)p(y))] = H(X) − H(X|Y) = H(Y) − H(Y|X)` measures how many bits one variable carries about the other. It is non-negative, zero exactly when `X⊥Y`, and it is a **convex (∪) function of the channel `p(y|x)`** for fixed input `p(x)`, and a concave (∩) function of the input `p(x)` for fixed channel.

**Channel capacity as a max of mutual information.** A memoryless channel with transition law `p(y|x)` has capacity `C = max_{p(x)} I(X;Y)`, and the channel-coding theorem says rates below `C` are reliably achievable and rates above `C` are not. The proof is a **random-coding** argument: draw `2^{nR}` codewords i.i.d.; a received word is, with high probability, jointly typical with the true codeword and with no other, provided `R<C`. Geometrically this is **sphere packing** — fitting noise-balls disjointly. Capacity is a *maximization* over inputs: nature fixes the channel, the coder chooses the input distribution best matched to it.

**The seed already planted in the continuous theory.** The original communication theory did not stop at the discrete lossless case. It also asked, for a *continuous* source, how to "assign a definite rate when we require only a certain fidelity of recovery." It argued that any reasonable fidelity valuation `v(P(x,y))` reduces (for ergodic sources and large blocks) to an *average of a distance function* `ρ(x,y)` over the joint distribution, `v = ∫∫ P(x,y) ρ(x,y) dx dy` — a per-letter "cost" of reproducing `x` as `y`. It then *defined* the rate of the source at fidelity `v₁` as

```
R₁ = min_{P_x(y)}  I(X;Y)    subject to   ∫∫ P(x,y) ρ(x,y) dx dy ≤ v₁,
```

the minimum mutual information over all systems meeting the fidelity budget, and stated (Theorem 21) that this `R₁` is exactly the channel capacity needed: a channel of capacity `C ≥ R₁` can carry the source at fidelity `→ v₁`, and `C < R₁` cannot. A partial Lagrange-multiplier solution was given, `P_y(x) = B(x) e^{−λ ρ(x,y)}`, showing the optimal backward conditional declines exponentially in the distance. It is stated for the continuous case with only a sketched argument; the discrete case, the properties of this quantity, how to compute it, and concrete worked examples remain to be developed.

## Baselines

The prior art a theory of lossy rate would be measured against:

- **Lossless source coding / entropy (the `H(X)` theorem).** Core idea: index the `≈2^{nH(X)}` typical sequences; rate `→ H(X)`, error `→ 0`; converse `R ≥ H(X)`. The actual math: AEP + counting the typical set.

- **Scalar quantization / PCM.** Core idea: partition the source range into cells, send a cell index, reproduce by the cell's representative (e.g. its centroid). For one bit on a `N(0,σ²)` source, the two half-lines with centroid reproduction give expected squared error `(π−2)/π · σ² ≈ 0.363 σ²`.

- **Channel capacity and the channel-coding theorem `C = max I`.** Core idea: random codebook, joint-typicality decoding, sphere packing; rates `<C` reliable, `>C` not. The math: `2^{nR}` random codewords, the probability of a confusable second codeword is `≈2^{−n(C−R)}`.

- **The continuous-source fidelity definition (`R₁ = min I` under budget `v₁`) and Theorem 21.** Core idea and math: as quoted in Background — the min-mutual-information-under-fidelity-budget definition, the claim that it equals the required channel capacity, and the exponential variational solution `P_y(x) = B(x) e^{−λ ρ(x,y)}`.

## Evaluation settings

The natural yardsticks for a theory of lossy rate are analytic source/distortion pairs whose rate one can compute in closed form and check against simple schemes:

- **Binary source with error-frequency (Hamming) distortion.** Source `Bernoulli(p)` (and the symmetric `p=½`), reproduction alphabet `{0,1}`; distortion `d_{ij}=1−δ_{ij}` so average distortion is the per-digit error probability. The metric is bits/symbol vs. tolerated error rate `D`.
- **`b`-ary equiprobable source, error-probability distortion.** Alphabets of size `2,3,4,5,10,100`; same error-frequency distortion. Metric: rate vs. `D`.
- **Gaussian source with mean-squared-error distortion.** `X ~ N(0,σ²)`, `d(x,x̂)=(x−x̂)²`, average distortion `D = E[(X−X̂)²]`. Metric: bits/symbol vs. `D` (equivalently distortion vs. rate).
- **Independent / parallel Gaussian components.** A vector of independent `N(0,σ_i²)` with summed squared-error distortion — the natural setting for asking how a fixed distortion budget should be split across components.
- **Reference schemes to compare the limit against**: one-bit scalar quantization of the Gaussian; sending raw binary digits over a noiseless channel; using short single-error-correcting codes (block lengths 3,7,15,31) "backwards" as source codes; transmission over a binary symmetric channel of given crossover. These give concrete points to hold up against the computed `R(D)` curve.
