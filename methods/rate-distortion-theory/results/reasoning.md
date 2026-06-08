I start from the thing that bothers me. I have a clean theorem for lossless coding: a discrete memoryless source with letter probabilities `P_i` has entropy `H(X) = −Σ P_i log₂ P_i`, and that number is both the floor and the achievable rate — code long blocks, index the roughly `2^{nH}` typical sequences, done. It is a beautiful closed answer. And it is useless the moment I stop demanding exactness. A continuous source — a voltage, a sample of a waveform — has a real number in it, an actual infinity of bits in the tail of its decimal expansion. To reproduce that exactly I would need a channel of infinite capacity. Every real channel is finite. So the lossless theorem hands me `+∞` and walks away. Even for a discrete source, nobody actually wants exact reproduction of a photograph or a voice; they want it *close enough*. The honest question was never "how many bits to reproduce `X` exactly." It is "how many bits to reproduce `X` to within a tolerance I get to name."

So I need two things I don't yet have. I need a way to *say* "within a tolerance" — a fidelity criterion. And I need a theorem of the same shape as the lossless one: a single number that is the true rate of the source at that tolerance, achievable above and impossible below. For a discrete source where zero distortion means exact reproduction, the lossless `H(X)` should fall out as the zero-tolerance corner.

Let me get the fidelity criterion right first, because everything rests on it. I have a source word `m = m_1 m_2 … m_t` and a reproduction `Z = z_1 z_2 … z_t`. I want a number saying how bad `Z` is as a stand-in for `m`. The most general thing I could write is some functional `v(P(m,Z))` of the whole joint distribution that at least orders systems — given two systems I can say which is more faithful. That's too loose to compute with. But watch what happens if I assume the source is ergodic and the criterion is "reasonable" in the sense that I could *estimate* it from a long sample and the estimate sharpens as the sample grows. Then for a long block the empirical fidelity concentrates, and the only functionals that behave this way are *averages of a per-pair cost*:

```
v = Σ_{m,Z} P(m,Z) d(m,Z),     d(m,Z) = (1/t) Σ_k d(m_k, z_k).
```

The whole criterion collapses to a single non-negative matrix `d_{ij}` — the cost of reproducing letter `i` as letter `j` — averaged over the word and over the joint statistics. That's the move that makes this tractable: a vague "fidelity" becomes a single-letter distortion matrix. The overall distortion of a system is then `d = Σ_{i,j} P_{ij} d_{ij}` where `P_{ij}` is the probability that `i` is produced and reproduced as `j`. Sanity check: error-probability fidelity is `d_{ij} = 1 − δ_{ij}`, and then `d` is exactly the per-symbol probability of error — good, that's the discrete case I'd expect; squared error is `d(x,x̂)=(x−x̂)²`. The matrix is general enough to weight some confusions more than others (mistaking "emergency" for "all's well" can cost ten times the reverse).

Now, the rate. What is the rate of *this* source at distortion budget `D`? I have to resist writing down the lossless instinct, which would be to count reproduction sequences. Let me think about what's actually free to me. I am the system designer. I choose how `m` gets mapped to `Z` — that means I am free to choose the joint distribution `p(m,Z)`, or equivalently the conditional `q_i(j) = p(\text{reproduce } j \mid \text{source } i)`. The only thing I'm not free in is the constraint: I must keep `Σ_{ij} P_i q_i(j) d_{ij} ≤ D`. Among all the joints I could build, which is "cheapest," and what does "cheap" even mean in bits?

Let me reason about what a code costs. A reproduction `Z` only has to carry whatever information about `m` is needed to pin it down to within distortion `D`. The relevant quantity is therefore how much `Z` tells me about `m` — and the bit-measure of that is the average mutual information

```
I(X;X̂) = Σ_{ij} P_i q_i(j) log₂ [ q_i(j) / Σ_k P_k q_k(j) ] = H(X) − H(X|X̂).
```

Why this and not, say, `H(X̂)`? I'll convince myself properly when I do the converse, but the intuition is: `H(X̂)` counts the bits in the reproduction, which I could inflate with useless randomness; `I(X;X̂)` counts only the bits of the reproduction that are *about the source*, which is what a code must actually transmit. So a candidate definition: among all test channels `q_i(j)` meeting the distortion budget, take the one with the *least* mutual information,

```
R(D) = min_{q: Σ P_i q_i(j) d_{ij} ≤ D}  I(X;X̂).
```

Notice the shape of this. Capacity was `C = max_{P(x)} I(X;Y)` — a *maximization* over input distributions for a *fixed* channel; nature gives you the channel, you find the input best matched to it. Here I have a *minimization* over channels `q_i(j)` for a *fixed* source; the source is given, and I find the channel — the test channel — best matched to it. The two problems are mirror images. That mirror is going to keep paying off.

Let me get a feel for the curve `R(D)` before I try to prove anything, because if its shape is wrong my whole plan collapses. As `D` grows the constraint set of allowed channels only gets larger, so the minimum can only drop: `R(D)` is non-increasing. Where does it bottom out? `I(X;X̂)=0` exactly when `X̂⊥X`, i.e. `q_i(j)=Q_j` independent of `i`. With such a constant assignment the distortion is `Σ_i P_i Σ_j Q_j d_{ij}`, minimized by putting all of `Q` on the column `j*` with the smallest `Σ_i P_i d_{ij}`. So `R` hits zero at `d_max = min_j Σ_i P_i d_{ij}` — beyond that tolerance you need send nothing, just output the single best constant letter. And the top end: the smallest achievable distortion is `d_min = Σ_i P_i min_j d_{ij}`, reproducing each letter by its individually cheapest target. If zero distortion forces exact reproduction, then `d_min=0`, the test channel must be the identity, `q_i(j)=δ_{ij}`, and `I = H(X)`. So the lossless entropy reappears as the left endpoint in the exact-reproduction case. That's the special case I demanded, falling out for free.

I need the curve to behave under mixing. Take two points on the curve, `(R,d)` from a channel `q` and `(R',d')` from `q'`. Mix them: `q'' = λq + (1−λ)q'`. Distortion is *linear* in the channel, so `q''` has distortion exactly `λd + (1−λ)d'`. And mutual information is a convex-`∪` function of the channel for a fixed source, so `I(q'') ≤ λI(q) + (1−λ)I(q') = λR + (1−λ)R'`. The minimizing channel at distortion `λd+(1−λ)d'` can only do better than this particular `q''`. Hence `R(λd+(1−λ)d') ≤ λR(d) + (1−λ)R(d')`. The curve is convex in the usual `∪` sense, because distortion is linear and information is convex.

Time for the real test. A definition is just a definition; I have to prove that `R(D)` is the *operational* rate — that no code can beat it, and that codes approaching it exist. I start with the converse, because if `R(D)` weren't a genuine floor the whole edifice is worthless.

Suppose I have any code: an encoder mapping a length-`n` source block `X^n` to an index from `{1,…,2^{nR}}`, and a decoder producing a reproduction `X̂^n`, achieving average distortion `D = E[d(X^n,X̂^n)]`. I want to force `R ≥ R(D)`. The number of distinct indices is at most `2^{nR}`, so the entropy of the index — and hence of the reproduction `X̂^n`, which is a function of the index — is at most `nR`. Let me chase the inequalities:

```
nR ≥ H(X̂^n)
   ≥ H(X̂^n) − H(X̂^n | X^n)        (subtract a non-negative quantity; in fact = 0 since X̂^n is a function of X^n)
   = I(X^n; X̂^n)
   = H(X^n) − H(X^n | X̂^n).
```

Now single-letterize. The source is i.i.d., so `H(X^n) = Σ_i H(X_i)`. For the conditional, the chain rule gives `H(X^n|X̂^n) = Σ_i H(X_i | X̂^n, X_{i−1},…,X_1)`, and conditioning only reduces entropy, so each term is `≤ H(X_i | X̂_i)`. Therefore

```
nR ≥ Σ_i [ H(X_i) − H(X_i | X̂_i) ] = Σ_i I(X_i ; X̂_i).
```

Each `(X_i, X̂_i)` is a joint distribution with some per-letter expected distortion `D_i = E[d(X_i,X̂_i)]`. By the very definition of `R(·)` as the *minimum* mutual information over all channels meeting a distortion, `I(X_i;X̂_i) ≥ R(D_i)`. So

```
nR ≥ Σ_i R(D_i) = n · (1/n) Σ_i R(D_i).
```

And now the convexity I just proved closes it. By Jensen, the average of `R(D_i)` is at least `R` of the average distortion:

```
(1/n) Σ_i R(D_i) ≥ R( (1/n) Σ_i D_i ) = R(D),
```

since `(1/n)Σ_i D_i` is exactly the overall average distortion `D`. Hence `R ≥ R(D)`. The converse is done, and notice *which* properties it used: mutual information is what `H(X̂^n)` lower-bounds and what single-letterizes; the min-over-channels definition is exactly what turns `I(X_i;X̂_i)` into `R(D_i)`; and convexity is exactly what assembles the per-letter bounds into `R(D)`. Every piece of the definition was forced. That retroactively answers why mutual information and not `H(X̂)` — `H(X̂)` is where the chain *starts*, but the load-bearing object the whole argument flows through is `I`, the part of the reproduction that is about the source.

Let me also confirm I can run this through a *channel* rather than a pure source code, because the duality lives there. Put a memoryless channel of capacity `C` between coder and decoder, `n` channel uses for `t` source letters. The reproduction `Z` is a function of the channel output `Y`, and `X` is a function of `m`. I want `H(m|Z) ≥ H(m) − nC`. Because `Z` is a function of `Y`, `H(m|Z) ≥ H(m|Y)`; and tracking how `H(X|Y_k)` changes as each channel letter arrives, the memoryless channel lets the conditional entropy drop by at most `C` per letter, so over `n` uses `H(X|Y)` falls by at most `nC` from `H(X)`, giving `H(m|Z) ≥ H(m) − nC`. On the other side I overbound `H(m|Z)`: single-letterizing as above, `H(m|Z) ≤ Σ_i H(m_i) − Σ_i R(D_i) ≤ Σ_i H(m_i) − t R(D)` by convexity, and `Σ H(m_i)=H(m)`. Combine the two: `H(m) − nC ≤ H(m) − tR(D)`, i.e.

```
nC ≥ t R(D).
```

So a channel used `n` times to carry `t` source letters at distortion `D` must supply at least `t R(D)` bits of capacity. `R(D)` really is the equivalent rate of the source at distortion `D`. And this isn't asymptotic hand-waving — it holds for *any* `n`, any block or variable-length code, as long as `t` letters get written down after `n` uses.

The converse says I can't beat `R(D)`. Now I have to show I can *reach* it, and this is where I expect the channel-coding machinery to come back, run in reverse. There, a random codebook of `2^{nR}` codewords with `R<C` worked because each received word was jointly typical with its own codeword and essentially no other — sphere *packing*, balls kept disjoint. Here I want the opposite: I want a *small* set of reproduction words such that *every* source word has *some* reproduction word close to it. That's sphere *covering*. Packing asks the balls not to overlap; covering asks them to leave no gap. Same typicality counting, opposite goal.

Let me set it up. Fix the minimizing test channel `q_i(j)` that achieves `R(D)`, so `I(X;X̂)=R(D)` and the induced reproduction marginal is `Q(j) = Σ_i P_i q_i(j)`. Generate a codebook of `M = 2^{nR}` reproduction words `X̂^n`, each drawn i.i.d. from this marginal `Q` — crucially, drawn from the *output* marginal of the optimal channel, not uniformly, because those are the reproduction words that are "shaped right" for this source. Encoding: given a source word `x^n`, look for a codeword `x̂^n(w)` that is *distortion-typical* with it — jointly typical in the usual `−(1/n)log p` senses *and* with empirical distortion within `ε` of `E[d]=D`. If one exists, send its index `w` (that costs `nR` bits); if several, send the least; if none, send `1` and eat the distortion. Decoding is trivial: output `x̂^n(w)`.

The only thing to control is the probability that *no* codeword is distortion-typical with `x^n` — the covering failure. For a fixed typical source word, the relevant number is the `Q`-measure of its fan of distortion-typical reproduction words. For a distortion-typical pair, the definitions of typicality give

```
p(x̂^n) ≥ p(x̂^n | x^n) 2^{−n(I(X;X̂)+3ε)},
```

which says the `Q`-probability that a single random codeword lands in that fan is at least `2^{−n(I+3ε)}` times the conditional probability of the fan under `p(x̂^n|x^n)`. That conditional probability tends to one for typical `x^n`. Using `(1−ab)^M ≤ 1−a+e^{−bM}` with `M=2^{nR}`, the all-miss probability is bounded by the conditional-typicality miss probability plus

```
≈ exp( − 2^{nR} · 2^{−n(I+3ε)} ) = exp( − 2^{n(R − I − 3ε)} ),
```

plus the vanishing probability that `x^n` wasn't typical at all. The exponent `2^{n(R−I−3ε)}` blows up — driving the failure to zero — precisely when `R > I(X;X̂) + 3ε`. Choose the test channel that achieves `R(D)`, so `I=R(D)`; then any `R > R(D)` leaves room to choose `ε` small enough for covering to succeed for large `n`. When covering succeeds the empirical distortion is within `ε` of `D`; the rare failures contribute at most `P_e · d_max → 0`. So the expected distortion over the random codebook is `≤ D + ε + P_e d_max ≤ D + δ`. The expectation being that small means some particular codebook achieves it. There exists a code of rate `R(D)+δ` with distortion `≤ D+δ`. Push `δ→0`: `R(D)` is achievable.

So the two halves meet. No code beats `R(D)` (converse), and codes approach `R(D)` (achievability). `R(D) = min_{q:E[d]≤D} I(X;X̂)` is the operational rate of lossy compression. And the packing/covering symmetry is now explicit: channel coding fits `2^{nC}` disjoint noise-balls *inside* the output space (packing, `R<C`); rate-distortion covers the source space with `2^{nR(D)}` distortion-balls (covering, `R>R(D)`). Same typical-set counting, dual extremal problems — `max I` against `min I`.

Now I want to *compute* this thing in cases I can check, because a min over all channels is not obviously something I can evaluate. Take the binary source, `X ~ Bernoulli(p)` with `p ≤ ½`, Hamming distortion — average distortion is the per-digit error rate. I can't minimize `I(X;X̂)` over channels by staring at it, so let me lower-bound and then try to hit the bound. Write

```
I(X;X̂) = H(X) − H(X|X̂) = H(p) − H(X ⊕ X̂ | X̂),
```

using that knowing `X̂`, the residual uncertainty in `X` is the uncertainty in the error bit `X⊕X̂`. Conditioning reduces entropy, so `H(X⊕X̂|X̂) ≤ H(X⊕X̂)`. The error bit has `Pr(X≠X̂) ≤ D`, and binary entropy increases on `[0,½]`, so `H(X⊕X̂) ≤ H(D)`. Therefore

```
I(X;X̂) ≥ H(p) − H(D),    i.e.   R(D) ≥ H(p) − H(D)   for 0 ≤ D ≤ p.
```

Can I achieve equality? I need a joint where the error bit is independent of `X̂` and has weight exactly `D` — that's a binary symmetric channel with crossover `D`, but oriented from `X̂` to `X` (the test channel runs *backward*). Feed it an input `X̂ ~ Bernoulli(r)` chosen so that the *output* `X` comes out `Bernoulli(p)`: I need `r(1−D) + (1−r)D = p`, i.e. `r = (p−D)/(1−2D)`. This is a valid probability for `0≤D≤p`. For that joint, the error bit is independent of `X̂` with weight `D`, so every inequality above is tight and `I = H(p) − H(D)`. Hence

```
R(D) = H(p) − H(D),   0 ≤ D ≤ min(p, 1−p);   R(D) = 0,   D ≥ min(p,1−p).
```

For `p=½` this is `R(D) = 1 + D log₂ D + (1−D) log₂(1−D)` — which I recognize as exactly the capacity of a BSC with crossover `D`. The duality again: the rate-distortion function of the symmetric binary source *is* a channel capacity. The general `b`-ary equiprobable case works the same way and gives `R(D,b) = log₂ b + D log₂ D + (1−D) log₂[(1−D)/(b−1)]`. The pattern I'd guessed — that the minimizing assignment for a symmetric source is the symmetric channel — holds.

The example I most want is the continuous one, because that's where the whole motivation started: a source that can't be reproduced exactly at all. `X ~ N(0,σ²)`, squared-error distortion `d(x,x̂)=(x−x̂)²`, budget `D = E[(X−X̂)²]`. Same strategy: lower-bound `I`, then construct a joint that meets it. Now I'm in differential entropies:

```
I(X;X̂) = h(X) − h(X|X̂) = ½ log₂(2πe σ²) − h(X − X̂ | X̂).
```

Conditioning reduces entropy: `h(X−X̂|X̂) ≤ h(X−X̂)`. And among all variables with a fixed second moment, the Gaussian maximizes differential entropy, so `h(X−X̂) ≤ h(N(0, E(X−X̂)²)) = ½ log₂(2πe · E(X−X̂)²) ≤ ½ log₂(2πe D)` since `E(X−X̂)² ≤ D`. Chaining,

```
I(X;X̂) ≥ ½ log₂(2πe σ²) − ½ log₂(2πe D) = ½ log₂(σ²/D),    so   R(D) ≥ ½ log₂(σ²/D).
```

Now achieve it. I need the error `X−X̂` to be Gaussian, of variance exactly `D`, and *independent of `X̂`*. If I tried the naive forward channel `X̂ = X + \text{noise}`, the reproduction would be noisier than the source — wrong marginal, and the error wouldn't be independent of `X̂`. The construction that works runs backward: write the source as reproduction-plus-error,

```
X = X̂ + Z,    X̂ ~ N(0, σ² − D),   Z ~ N(0, D),   X̂ ⊥ Z   (for 0 < D ≤ σ²).
```

Then `X ~ N(0, σ²)` as required (variances add), `E(X−X̂)² = E Z² = D`, and because `Z` is independent of `X̂` of variance `D`, the lower bound is met with equality: `I(X;X̂) = ½ log₂(σ²/D)`. For `D ≥ σ²` just output `X̂=0` and pay distortion `σ²`, giving `R=0`. At `D=0`, exact reproduction of a continuous Gaussian costs infinite rate. So

```
R(D) = ½ log₂(σ²/D)   for 0 < D < σ²,    R(0)=+∞,    R(D) = 0   for D ≥ σ².
```

This is exactly the closed-form lossy rate the lossless theory could never produce — it was returning `+∞` at exact reproduction, and here is the finite curve for every positive distortion. Rewrite it as `D(R) = σ² 2^{−2R}`: every bit of description cuts the squared error by a factor of four, i.e. about `6.02` dB per bit. That gives the scalar quantizer an honest yardstick: one-bit quantization of `N(0,σ²)` gives `≈0.363 σ²`, but `R=1` permits `D = σ²/4 = 0.25 σ²`. The quantizer is leaving distortion on the table — and the reason it can't reach `0.25σ²` while the theorem can is that the theorem codes a long *block* of independent samples jointly: the typical source words concentrate on a thin spherical shell that `2^{nR(D)}` distortion-balls cover efficiently, whereas scalar quantization treats each coordinate in isolation and wastes rate. That long-block gain over independent symbols is exactly the covering argument made flesh.

Both worked examples used the same trick: lower-bound `I` by "conditioning reduces entropy" plus a maximum-entropy fact, then build a *backward* test channel that makes the error independent of the reproduction so the bound is tight. That generalizes. For any difference distortion `d(x,x̂)=ρ(x−x̂)`, let `φ(D)` be the maximum entropy of a variable whose mean distortion is `D` (a concave function of `D`). Then

```
R(D) = h(X) − h(X|X̂) ≥ h(X) − h(X−X̂) ≥ h(X) − φ(D),
```

a clean lower bound on `R(D)` for any difference distortion. For the Gaussian-plus-squared-error pair it is tight — `φ(D)=½log₂(2πeD)` and `h(X)=½log₂(2πeσ²)` recover `½log₂(σ²/D)`. For a fixed variance this bound is largest when `X` is Gaussian, and the Gaussian case meets it, making the Gaussian curve the natural worst-case benchmark for squared error.

One more property I should pin down, because real sources are multidimensional: how to spend a fixed distortion budget across independent components. Take independent Gaussians `X_i ~ N(0, σ_i²)` with summed squared-error distortion. Each component contributes `R_i(D_i) = ½ log₂(σ_i²/D_i)` (or `0` if `D_i ≥ σ_i²`), and I minimize `Σ R_i` subject to `Σ D_i = D`. Lagrange on `Σ ½ log₂(σ_i²/D_i) + λ Σ D_i` gives `−1/(2 ln 2 · D_i) + λ = 0`, so every active component gets the same distortion `D_i = θ`, capped at its own variance: `D_i = min(θ, σ_i²)`, with `θ` set so `Σ D_i = D`. That's reverse water-filling — pour distortion `θ` into every component, and any component with variance below the water level `θ` gets thrown away entirely (`X̂_i = 0`, spend no bits on it). This is the precise rule for how the product source splits a budget; in the single-letter language it says the `R(D)` of a sum-distortion product source is gotten by adding the component curves at points of equal slope.

Let me step back and make sure the whole chain holds together, because the duality I keep noticing is the deepest part. The lossless theorem gave one number `H(X)` for exact reproduction. I relaxed exactness into a single-letter distortion matrix; defined the rate at distortion `D` as the *minimum* mutual information over all test channels meeting the budget, `R(D) = min_{q:E[d]≤D} I(X;X̂)`; proved this minimum is convex and runs from the exact-reproduction entropy corner down to `0` at `d_max`; proved no code can beat it by single-letterizing `nR ≥ I(X^n;X̂^n)` and using the min-definition plus convexity (the converse); proved codes approach it by random covering with a codebook drawn from the optimal output marginal (achievability); and evaluated it in closed form — `H(p)−H(D)` for the binary source, `½ log₂(σ²/D)` for the Gaussian — checking each against the lower bound and against simple schemes. Throughout, the single object is mutual information, minimized over channels for a fixed source. Capacity was the same object *maximized* over inputs for a fixed channel: `C = max_P I` gives a concave curve and finds the source matched to a channel; `R(D) = min_q I` gives a convex curve and finds the channel matched to a source. Put them in series — source coded to `R(D)` bits, then channel-coded under capacity `C` — and lossy transmission at distortion `D` is possible exactly when `C ≥ R(D)`. For a discrete source whose zero-distortion condition is exact reproduction, the lossless `H(X)` is just `R(0)`; for a continuous Gaussian, the same curve correctly blows up as `D↓0`. One quantity, two extremal problems, and the whole lossy theory hangs off the single inequality `C/R(D) ≥ 1`.

I can now write the theorem and the examples in the form the whole argument has forced:

```
Rate–distortion function.  For an i.i.d. source X ~ P with single-letter
distortion d(x, x̂) ≥ 0,

        R(D) = min_{ q(x̂|x) : E[d(X,X̂)] ≤ D }  I(X; X̂).

Theorem (operational meaning).  R(D) is the least rate, in bits per symbol,
at which X can be reproduced within average distortion D:
  • Converse:  any (2^{nR}, n) code with E d(X^n, X̂^n) ≤ D has R ≥ R(D).
       nR ≥ H(X̂^n) ≥ I(X^n;X̂^n) = ΣH(X_i) − ΣH(X_i|X̂^n,X_{<i})
          ≥ Σ[H(X_i) − H(X_i|X̂_i)] = Σ I(X_i;X̂_i) ≥ Σ R(D_i)
          ≥ n R( (1/n)ΣD_i ) = n R(D).     [min-MI def + convexity/Jensen]
  • Achievability:  for any R > R(D), a codebook of 2^{nR} reproduction words
       drawn i.i.d. from the optimal output marginal Q(x̂)=Σ_x P(x) q*(x̂|x)
       distortion-covers the source: the covering failure term is
          exp(−2^{n(R − I(X;X̂) − 3ε)}) → 0,
       so some code attains distortion ≤ D+δ.   [random covering, dual to packing]

R(D) is non-increasing and convex.  When zero distortion means exact
reproduction for a discrete source, R(0)=H(X); and R(D)=0 for
D ≥ d_max = min_x̂ E[d(X,x̂)].

Gaussian source, squared error.  X ~ N(0,σ²), d=(x−x̂)²:
       R(D) = ½ log₂(σ²/D),  0 < D < σ²;  R(0)=+∞;  R(D)=0 for D≥σ².
   Lower bound:  I = h(X) − h(X−X̂|X̂) ≥ ½log₂(2πeσ²) − ½log₂(2πeD) = ½log₂(σ²/D).
   Achieved by the backward channel X = X̂ + Z, X̂~N(0,σ²−D) ⊥ Z~N(0,D).
   Equivalently D(R) = σ² 2^{−2R}:  6.02 dB per bit.

Binary source, Hamming distortion.  X ~ Bernoulli(p), p ≤ ½:
       R(D) = H(p) − H(D),  0 ≤ D ≤ p;      R(D) = 0,  D ≥ p.
   Lower bound H(p)−H(D) achieved by a backward BSC test channel of crossover D
   feeding input Bernoulli(r), r = (p−D)/(1−2D).

Duality.  C = max_P I(X;Y) (concave, channel fixed) and R(D) = min_q I(X;X̂)
(convex, source fixed) are mirror extremal problems; lossy transmission at
distortion D over a capacity-C channel is possible iff C ≥ R(D).
```
