Let me start from the thing that actually annoys me, which is Morse code. I have a source — call it a random variable `X` over an alphabet `{x_1,…,x_m}`, with probabilities `p_i` — and I want to write each symbol as a string over a `D`-ary channel alphabet `{0,1,…,D−1}`, short strings for frequent symbols, long for rare. That part is easy and old. What is not easy is that I am sending a *stream*: I glue the codewords together with nothing between them, and the receiver gets one long `D`-ary string and has to peel the original symbols back out. Morse cheats here — it inserts a space between letters. That space is a whole extra symbol of the alphabet spent purely on punctuation. It feels wasteful. I'd like the code itself to tell the receiver where one codeword ends and the next begins, with no separator. And ideally I'd like to know the symbol the *instant* its codeword finishes, without reading ahead.

So before I optimize anything, I need to understand what's even *possible*. If I just hand out very short codewords to everybody, I'll obviously create ambiguity — two different symbol sequences will produce the same channel string. There has to be a budget on the lengths `ℓ_i`. What is it? And does it depend on how demanding I am about decoding — instant decoding versus merely "you can eventually figure it out"?

Let me pin down the decoding conditions, because I keep conflating them. The weakest useful one: the code is *uniquely decodable* — the extension (concatenation) map from symbol-sequences to channel-strings is injective. No two distinct messages collide. But decoding might be horrible: I might have to read the entire received string before I can commit to even the first symbol. The stronger one I actually want: *prefix-free*, no codeword is a prefix of any other. Then the moment I finish reading a codeword I know it's a codeword — nothing it could be the start of is also legal — so I decode left to right with zero lookahead. Self-punctuating. That's the dream, no wasted comma symbol.

These nest: every prefix code is uniquely decodable (if I can decode instantly I can certainly decode), every uniquely decodable code is nonsingular. And the nesting is strict — there are uniquely decodable codes that are not prefix-free. Quick sanity check on that, because it matters for the whole story. Over `{0,1}` take codewords `{10, 00, 11, 110}`. The word `110` has `11` as a prefix, so this is not prefix-free. But the Sardinas-Patterson dangling-suffix check starts with the suffix `0` and then only reproduces `0`; it never produces the empty suffix and never produces a codeword, so the code is uniquely decodable. Operationally, after seeing an initial `11`, I may have to read through a run of following `0`s before the first boundary is forced. So prefix-free is a *real* restriction. Which means I should worry: by insisting on instant decoding, am I giving up length profiles that a cleverer non-instantaneous-but-still-decodable code could have used? Because if mere unique decodability lets me use shorter codewords, I'd want those.

Hold that worry. First let me just nail the prefix case, because it feels the most tractable.

Picture the code on a tree. The channel alphabet has `D` symbols, so build the `D`-ary tree: each node has `D` children, the `D` edges out of a node are the `D` channel symbols, and a codeword is the sequence of edge-labels from the root down to some node. So every codeword is a node, at depth equal to its length. Now what does prefix-free mean on this tree? Codeword `a` is a prefix of codeword `b` exactly when the node for `a` is an *ancestor* of the node for `b`. Prefix-free = no codeword is an ancestor of another = the codewords are an *antichain*, they sit at leaves of the subtree they span. Good, that's a clean geometric condition.

Now I want a budget. I look at the bottom. Let `ℓ_max` be the longest codeword length. Go down to depth `ℓ_max` — there are exactly `D^{ℓ_max}` nodes at that level, that's my finite resource. Each codeword, sitting at depth `ℓ_i`, "owns" all its descendants at depth `ℓ_max`. How many are those? From a node at depth `ℓ_i`, every additional level multiplies by `D`, so at depth `ℓ_max` it has `D^{ℓ_max − ℓ_i}` descendants. And because no codeword is an ancestor of another, these descendant-sets are *disjoint* — a depth-`ℓ_max` node can't be a descendant of two different codewords, or one codeword would be on the path to the other. Disjoint subsets of a set of size `D^{ℓ_max}`:

`Σ_i D^{ℓ_max − ℓ_i} ≤ D^{ℓ_max}`.

Divide both sides by `D^{ℓ_max}`:

`Σ_i D^{−ℓ_i} ≤ 1`.

There's the budget. The lengths of any prefix code must satisfy `Σ_i D^{−ℓ_i} ≤ 1`. It's almost embarrassingly simple once it's a counting statement about a leaf budget. Let me re-examine the exponent to be sure: a node at depth `ℓ_i`, descendants at depth `ℓ_max`, that's `ℓ_max − ℓ_i` levels below it, each level `×D`, so `D^{ℓ_max−ℓ_i}` — yes. And summing the owned leaves can't exceed the total leaves `D^{ℓ_max}` — yes. Divide through — yes. The `D^{ℓ_max}` cancels and `ℓ_max` vanishes from the answer, which is reassuring: the budget can't depend on an arbitrary truncation level.

And I notice something the inequality is whispering: the quantities `D^{−ℓ_i}` are nonnegative and sum to at most 1. They behave like a sub-probability distribution. Short codeword ⇒ large `D^{−ℓ_i}` ⇒ expensive, eats more of the unit budget. That smells like it will connect to probabilities `p_i` later. Park that.

Now the converse, because a budget is only half the story — I want it to be *sufficient*, not just necessary. Suppose someone hands me positive integer lengths `ℓ_1 ≤ ℓ_2 ≤ ⋯ ≤ ℓ_m` with `Σ D^{−ℓ_i} ≤ 1`. Can I always build a prefix code with exactly these lengths? I take the full tree down to depth `ℓ_max` and number the depth-`ℓ_max` leaves from `0` to `D^{ℓ_max}-1` in lexicographic order. A node at depth `ℓ` corresponds to one aligned block of `D^{ℓ_max-ℓ}` consecutive bottom leaves. At step `j`, the previous codewords have reserved

`U_{j-1} = Σ_{i<j} D^{ℓ_max-ℓ_i}`

bottom leaves. Since the earlier lengths satisfy `ℓ_i ≤ ℓ_j`, every previous block size `D^{ℓ_max-ℓ_i}` is a multiple of the current block size `D^{ℓ_max-ℓ_j}`, so `U_{j-1}` is aligned to a depth-`ℓ_j` node. And the total budget says

`U_{j-1} + D^{ℓ_max-ℓ_j} ≤ Σ_i D^{ℓ_max-ℓ_i} ≤ D^{ℓ_max}`.

So the aligned block beginning at `U_{j-1}` exists and is not inside any earlier reserved block. I choose the depth-`ℓ_j` node corresponding to that block as the next codeword. Repeating this plants every codeword, and the reserved bottom-leaf blocks are disjoint, so no chosen node can be an ancestor of another. Prefix code exists.

(There's an equivalent way to see the converse that I like even better because it makes the "sub-probability" feeling literal. Map each codeword `y_1 y_2 ⋯ y_{ℓ_i}` to the real number `0.y_1 y_2 ⋯ y_{ℓ_i}` in base `D`, and to the half-open interval `[0.y_1⋯y_{ℓ_i}, 0.y_1⋯y_{ℓ_i} + D^{−ℓ_i})` — all reals in `[0,1]` whose `D`-ary expansion *starts* with that codeword. Prefix-free ⇔ these intervals are disjoint. Their lengths are exactly `D^{−ℓ_i}`, disjoint sub-intervals of `[0,1]`, so they sum to `≤ 1` — that re-proves necessity — and conversely, sorting the lengths and laying intervals `[Σ_{j<i} D^{−ℓ_j}, Σ_{j≤i} D^{−ℓ_j})` end to end inside `[0,1]` builds the code. The budget `≤1` guarantees they fit, and the sorted order matters: before I place length `ℓ_i`, the cumulative left endpoint is a sum of powers `D^{−ℓ_j}` with `ℓ_j ≤ ℓ_i`, hence an integer multiple of `D^{−ℓ_i}`. So that left endpoint is aligned on a `D`-adic interval of width `D^{−ℓ_i}`, and its first `ℓ_i` base-`D` digits really do name the next codeword.)

So for prefix codes I have a clean iff: a prefix code with lengths `(ℓ_i)` exists **if and only if** `Σ D^{−ℓ_i} ≤ 1`. And the same leaf-budget argument extends verbatim to a countably infinite set of codewords — the interval picture is cleanest there, since the intervals are still disjoint sub-intervals of `[0,1]`, so `Σ_{i=1}^∞ D^{−ℓ_i} ≤ 1`, and the converse follows by applying the same aligned-interval placement to each finite initial collection and taking the nested limit.

Now back to the worry I parked, and it's the real question. I proved the budget for *prefix* codes. But prefix is a strong restriction — I showed there exist uniquely decodable codes that aren't prefix-free. So maybe, by allowing the full class of uniquely decodable codes (decode the whole string if you must, just be unambiguous), I can *beat* the budget — use a length profile with `Σ D^{−ℓ_i} > 1` that's still decodable, just not instantly. If so, then insisting on prefix codes is genuinely costing me, and I should chase those non-instantaneous codes for the extra room.

Let me try to construct a counterexample to convince myself there's slack. I want lengths violating `Σ D^{−ℓ_i} ≤ 1` but still uniquely decodable... and I keep failing. Every uniquely decodable code I write down, when I tally `Σ D^{−ℓ_i}`, comes out `≤ 1`. That's suspicious. Maybe there *is* no slack. Maybe the budget is the same for the bigger class. That would be a genuinely surprising thing — it would say the instant-decoding restriction is *free*, costs you nothing in achievable lengths.

How would I even prove that? For prefix codes I had the tree, the ancestor structure, a place to hang the counting. A uniquely decodable code has no tree structure — codewords can be ancestors of each other, the whole geometric picture is gone. All I'm given is: the *extension* map is injective. Concatenations of codewords don't collide. That's the only handle. Let me see what I can squeeze out of injectivity alone.

Let me write `S = Σ_i D^{−ℓ_i}` for the quantity I'm trying to bound by 1, and `ℓ(x)` for the length of codeword `x`. In this stream setting a zero-length codeword cannot appear in a uniquely decodable code: I could insert that symbol into any source sequence and the channel string would not change. So `ℓ_min ≥ 1`. The injectivity is about *sequences* of symbols, not single symbols. So I should look at sequences. Concatenate `k` codewords: a length-`k` source string `x_1 x_2 ⋯ x_k` encodes to a channel string of length `ℓ(x_1)+ℓ(x_2)+⋯+ℓ(x_k)`. Unique decodability says this map — from length-`k` source strings to channel strings — is injective.

How do I get `S` to talk to length-`k` strings? Raise it to the `k`-th power. That's the gamble: a single codeword gives me nothing, but `S^k` is a sum over all `k`-tuples of codewords, which is exactly the object injectivity controls. Let me just expand it and see.

`S^k = (Σ_{x} D^{−ℓ(x)})^k = Σ_{x_1} Σ_{x_2} ⋯ Σ_{x_k} D^{−ℓ(x_1)} D^{−ℓ(x_2)} ⋯ D^{−ℓ(x_k)}`.

Each product `D^{−ℓ(x_1)}⋯D^{−ℓ(x_k)} = D^{−(ℓ(x_1)+⋯+ℓ(x_k))}`, and `ℓ(x_1)+⋯+ℓ(x_k)` is precisely the total channel-length of the encoded string for the source `k`-tuple `(x_1,…,x_k)`. So

`S^k = Σ_{(x_1,…,x_k)} D^{−(\text{total length of that } k\text{-tuple's encoding})}`.

This is a sum over all `m^k` source `k`-tuples, weighted by `D^{−(\text{encoded length})}`. Now group the terms by the encoded length. Let `a(m')` be the number of source `k`-tuples whose encoding has total length exactly `m'`. The shortest possible total length is `k·ℓ_min`, the longest is `k·ℓ_max`. So

`S^k = Σ_{m'=k·ℓ_min}^{k·ℓ_max} a(m') · D^{−m'}`.

Now the only fact I have about uniquely decodable codes finally pays. `a(m')` counts source `k`-tuples encoding to a *given* total length `m'`. By unique decodability, distinct source `k`-tuples encode to distinct channel strings. So the encodings counted by `a(m')` are distinct channel strings, each of length `m'`. But there are only `D^{m'}` distinct `D`-ary strings of length `m'` to go around. Therefore

`a(m') ≤ D^{m'}`.

That's the whole power of injectivity, dropped right in. Substitute:

`S^k = Σ_{m'=k·ℓ_min}^{k·ℓ_max} a(m') D^{−m'} ≤ Σ_{m'=k·ℓ_min}^{k·ℓ_max} D^{m'} D^{−m'} = Σ_{m'=k·ℓ_min}^{k·ℓ_max} 1`.

The `D^{m'}` and `D^{−m'}` cancel exactly — every term collapses to `1` — and I'm just *counting the terms*. The number of integers `m'` from `k·ℓ_min` to `k·ℓ_max` is `k(ℓ_max − ℓ_min) + 1`. Since I am in the nontrivial case `ℓ_min ≥ 1`, this is at most `k·ℓ_max`. So

`S^k ≤ k·ℓ_max`.

Stare at that. The left side is `S^k`. If `S > 1`, the left side grows *exponentially* in `k`. The right side grows *linearly* in `k`. An exponential cannot stay under a line forever. So `S > 1` is impossible — for large enough `k` it would break the inequality. The only way `S^k ≤ k·ℓ_max` can hold for *every* `k` is `S ≤ 1`. Let me make that airtight by taking the `k`-th root:

`S ≤ (k·ℓ_max)^{1/k}`.

As `k → ∞`, `(k·ℓ_max)^{1/k} → 1` (any polynomial-in-`k` quantity has `k`-th root tending to 1 — `log` of it is `(log k + log ℓ_max)/k → 0`). And `S` doesn't depend on `k` at all; the inequality holds for all `k`; so `S ≤ \lim (k·ℓ_max)^{1/k} = 1`. Done:

`Σ_i D^{−ℓ_i} ≤ 1` — for **every uniquely decodable code.**

That's the surprise, and it took a second to register. The *same* budget, `Σ D^{−ℓ_i} ≤ 1`, that I derived for the instantaneous codes using their tree structure, holds for the entire larger class of uniquely decodable codes — even the ones that need unbounded lookahead, even though they have no tree structure at all to hang an argument on. I never used a tree here; I used only that concatenations don't collide, raised to a power, and counted strings. The exponential-versus-linear mismatch did all the work, and the `k`-th root squeezed the polynomial slack down to exactly 1.

And the consequence is what I was actually after. The set of length profiles `(ℓ_i)` achievable by uniquely decodable codes is *the same* as the set achievable by prefix codes — both are exactly `{(ℓ_i) : Σ D^{−ℓ_i} ≤ 1}`. (One direction I just proved: UD ⇒ the budget holds. The other is free: given lengths satisfying the budget, my earlier tree construction builds a prefix code with those lengths, and a prefix code is uniquely decodable.) So allowing non-instantaneous decoding buys me *nothing* in terms of which lengths I can use. The comma-free, lookahead-free, self-punctuating prefix codes lose nothing. My earlier worry is dead: I can always restrict to prefix codes with a clear conscience. That's a relief and it's the whole justification for ever caring about prefix codes.

(I should record that the counting argument is the same thing as a generating function, which is the cleaner way to see why powering was the right instinct. Set `F(x) = Σ_i x^{−|s_i|}` — collect by length, `F(x) = Σ_ℓ p_ℓ x^{−ℓ}` where `p_ℓ` is the number of codewords of length `ℓ`. Then `F(x)^k = Σ q_ℓ x^{−ℓ}` and `q_ℓ` is the number of length-`k` source strings encoding to total length `ℓ` — and unique decodability is exactly the statement `q_ℓ ≤ D^ℓ`, since the map into length-`ℓ` channel strings is injective. Evaluate at `x = D`: `F(D)^k ≤ Σ_ℓ D^ℓ D^{−ℓ} = (\text{number of possible total lengths}) ≤ k·ℓ_max`, exponential `≤` linear, so `F(D) = Σ D^{−ℓ_i} ≤ 1`. Same argument, the polynomial-coefficients language just makes the "raise to the `k`-th and read off coefficients" move obvious. And it shows the infinite-alphabet case is fine too: any finite sub-collection of an infinite UD code is itself uniquely decodable, so every partial sum is `≤ 1`, hence the infinite sum is `≤ 1`.)

Now I can finally optimize, because I have an exact characterization of the feasible set. I want to minimize the expected length

`L = Σ_i p_i ℓ_i`

over all integer length-profiles `(ℓ_i)` subject to `Σ_i D^{−ℓ_i} ≤ 1`. The feasible region is the same whether I think prefix or uniquely decodable — so whatever `L` I can't beat with a prefix code, I can't beat with *any* decodable code. That's why the budget is the right object: it turns "code design" into "constrained minimization over lengths."

Let me first ignore that the `ℓ_i` must be integers and just feel out the shape. Minimize `Σ p_i ℓ_i` subject to `Σ D^{−ℓ_i} = 1` (it'll want to use the whole budget). Lagrangian `J = Σ p_i ℓ_i + λ(Σ D^{−ℓ_i} - 1)`. Differentiate in `ℓ_i`: `∂J/∂ℓ_i = p_i − λ D^{−ℓ_i} ln D`. Set to zero: `D^{−ℓ_i} = p_i/(λ ln D)`. Plug into `Σ D^{−ℓ_i} = 1`: `Σ p_i/(λ ln D) = 1`, and `Σ p_i = 1`, so `λ ln D = 1`, giving `D^{−ℓ_i} = p_i`, i.e.

`ℓ_i* = −log_D p_i = log_D(1/p_i)`.

There it is — the "lengths look like a distribution" hunch from way back, made exact. The optimal (real-valued) length of symbol `i` is `log_D(1/p_i)`, and the resulting expected length is

`L* = Σ p_i (−log_D p_i) = H_D(X)`.

The entropy. The optimal `D^{−ℓ_i}` *is* `p_i` exactly, the budget is filled exactly, and the cost is the entropy. So I should *conjecture* `L ≥ H_D(X)` always, and prove it directly without leaning on calculus, because the real `ℓ_i` are integers and I want a clean lower bound that holds for every feasible integer profile.

Take any uniquely decodable code with lengths `ℓ_i`. Look at the gap:

`L − H_D(X) = Σ p_i ℓ_i − Σ p_i log_D(1/p_i) = Σ p_i ℓ_i + Σ p_i log_D p_i`.

Write `ℓ_i = log_D(D^{ℓ_i}) = −log_D(D^{−ℓ_i})`, so `Σ p_i ℓ_i = −Σ p_i log_D(D^{−ℓ_i})`. Then

`L − H_D = −Σ p_i log_D D^{−ℓ_i} + Σ p_i log_D p_i = Σ p_i log_D ( p_i / D^{−ℓ_i} )`.

I want this to be a relative entropy, but `D^{−ℓ_i}` need not sum to 1 — it sums to `c := Σ_j D^{−ℓ_j} ≤ 1` (that's exactly the budget, and exactly where unique decodability re-enters). So normalize: let `r_i = D^{−ℓ_i}/c`, a genuine probability distribution. Then `D^{−ℓ_i} = c·r_i` and

`L − H_D = Σ p_i log_D ( p_i / (c r_i) ) = Σ p_i log_D (p_i/r_i) − Σ p_i log_D c = D(p‖r) − log_D c = D(p‖r) + log_D(1/c)`.

Now read off signs. `D(p‖r) ≥ 0` by Gibbs' inequality — relative entropy is nonnegative, that's the convexity of `−log` / the standard fact, equality iff `p = r`. And `c ≤ 1` (the budget!) so `1/c ≥ 1` so `log_D(1/c) ≥ 0`. Two nonnegative pieces. Therefore

`L − H_D(X) = D(p‖r) + log_D(1/c) ≥ 0`, i.e. `L ≥ H_D(X)`.

The expected length of *any* uniquely decodable `D`-ary code is at least the `D`-ary entropy. And the budget `c ≤ 1` is doing essential work — without `Σ D^{−ℓ_i} ≤ 1` the `log_D(1/c)` term could be negative and the bound would fail. Because the budget holds even for the full uniquely-decodable class, the entropy lower bound holds for the full class too, not just prefix codes. Equality requires both `D(p‖r)=0` (so `p_i = r_i`) and `c=1`, which together force `D^{−ℓ_i} = p_i` — exactly the calculus solution. After I restrict the alphabet to symbols of positive probability, equality means every `p_i` is `D^{-n}` for an integer `n`. Otherwise we pay a strict premium.

So the floor is the entropy. Can I get close to it with *integer* lengths? The real optimum wants `ℓ_i = log_D(1/p_i)`, which generally isn't an integer. Round up — take the smallest integer at least that big:

`ℓ_i = ⌈ log_D(1/p_i) ⌉`.

For a non-degenerate source, after I drop zero-probability symbols, every remaining `p_i` lies in `(0,1)`, so these rounded lengths are positive. Two things to check: that this is feasible (obeys the budget, so a code actually exists), and how much the rounding costs. Feasibility: `⌈log_D(1/p_i)⌉ ≥ log_D(1/p_i)`, so `D^{−ℓ_i} ≤ D^{−log_D(1/p_i)} = p_i`. Sum: `Σ_i D^{−ℓ_i} ≤ Σ_i p_i = 1`. The budget holds, so by my converse construction a prefix code with these lengths exists — these "Shannon lengths" are realizable. Cost of rounding: the ceiling satisfies

`log_D(1/p_i) ≤ ℓ_i < log_D(1/p_i) + 1`.

Multiply through by `p_i` and sum over `i`:

`Σ p_i log_D(1/p_i) ≤ Σ p_i ℓ_i < Σ p_i log_D(1/p_i) + Σ p_i`,

which is exactly

`H_D(X) ≤ L < H_D(X) + 1`.

So the Shannon-length code lands within one `D`-ary digit per symbol of the entropy floor. Combined with the lower bound `L ≥ H_D(X)`, the *optimal* code `L*` is squeezed:

`H_D(X) ≤ L* < H_D(X) + 1`.

That `+1` is one whole digit of overhead, annoying when `H_D` is small (if `H_D < 1`, paying nearly a full extra digit per symbol could nearly double the rate). But it's a *per-symbol* overhead, and it comes only from the integer-rounding ceiling — so spread it out. Encode `n` source symbols at once as a single super-symbol from the alphabet `X^n`. The same bound applied to the super-source gives `H_D(X_1,…,X_n) ≤ E[ℓ(X^n)] < H_D(X_1,…,X_n) + 1`. Define the per-symbol length `L_n = (1/n) E[ℓ(X^n)]`. Divide by `n`:

`(1/n) H_D(X_1,…,X_n) ≤ L_n < (1/n) H_D(X_1,…,X_n) + 1/n`.

For an i.i.d. source `H_D(X_1,…,X_n) = n H_D(X)`, so this is

`H_D(X) ≤ L_n < H_D(X) + 1/n`,

and the overhead `1/n → 0`. By blocking long enough I drive the per-symbol length arbitrarily close to the entropy. (And for a stationary — not necessarily i.i.d. — source the same chain gives `H_D(X_1,…,X_n)/n ≤ L_n < H_D(X_1,…,X_n)/n + 1/n`, and `H_D(X_1,…,X_n)/n` tends to the `D`-ary entropy rate `H_D(𝒳)`, so `L_n → H_D(𝒳)`.) The entropy isn't just a lower bound I can't beat — it's a limit I can approach. That closes the loop: the budget `Σ D^{−ℓ_i} ≤ 1` is the exact feasibility condition, it holds for every decodable code (not just instantaneous ones), it forces `L ≥ H_D(X)`, and ceiling-rounding plus blocking achieves `H_D(X)` in the limit.

Let me lay out the whole chain so I trust it end to end. A `D`-ary code lives on a `D`-ary tree; prefix-free means codewords are an antichain, so their depth-`ℓ_max` descendant sets are disjoint inside a budget of `D^{ℓ_max}` leaves, giving `Σ D^{−ℓ_i} ≤ 1` — and that condition is also sufficient, by placing aligned bottom-leaf blocks or equivalently aligned `D`-adic intervals in `[0,1]`. Then the worry that the larger uniquely-decodable class might escape this budget is killed by raising `S = Σ D^{−ℓ_i}` to the `k`-th power: `S^k` sums `D^{−(\text{total length})}` over all source `k`-tuples, unique decodability bounds the count of `k`-tuples per total length `m'` by `D^{m'}`, the `D`-powers cancel, and `S^k ≤ k·ℓ_max` — exponential trapped under linear forces `S ≤ 1`. So the achievable length sets for prefix and uniquely-decodable codes coincide; prefix codes lose nothing. Treating `D^{−ℓ_i}` as a sub-probability, the gap `L − H_D = D(p‖r) + log_D(1/c)` is a sum of two nonnegative terms (relative entropy `≥ 0`, and `c = Σ D^{−ℓ_i} ≤ 1` so `log_D(1/c) ≥ 0`), giving `L ≥ H_D(X)` with equality iff, after restricting to the positive-probability alphabet, `p_i = D^{−ℓ_i}` for every symbol; and `ℓ_i = ⌈log_D(1/p_i)⌉` is feasible and achieves `H_D ≤ L < H_D + 1`, which blocking drives to `H_D` per symbol. The lengths are a probability distribution in disguise, and the entropy is exactly the price of admission.

To make it concrete, the smallest possible "code" — really just a verifier and a builder for the two theorems:

```python
from fractions import Fraction
from math import log, ceil

def kraft_sum(lengths, D):
    # S = sum_i D^{-l_i}; exact rational arithmetic keeps the budget exact.
    if D < 2 or any(not isinstance(l, int) or l <= 0 for l in lengths):
        raise ValueError("use D >= 2 and positive integer lengths")
    return sum(Fraction(1, D ** l) for l in lengths)

def kraft_feasible(lengths, D):
    # Necessary AND sufficient (prefix OR uniquely decodable -- same set):
    # a decodable D-ary code with these lengths exists iff S <= 1.
    return kraft_sum(lengths, D) <= 1

def base_D_digits(index, length, D):
    digits = [0] * length
    for pos in range(length - 1, -1, -1):
        digits[pos] = index % D
        index //= D
    return tuple(digits)

def build_prefix_code(lengths, D):
    # Constructive converse: sorted lengths give aligned depth-l_max leaf blocks.
    if not kraft_feasible(lengths, D):
        raise ValueError("lengths violate the Kraft budget")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    code = [None] * len(lengths)
    l_max = max(lengths, default=0)
    next_leaf = 0
    for i in order:
        l = lengths[i]
        block = D ** (l_max - l)
        assert next_leaf % block == 0      # sorted lengths make the block aligned
        code[i] = base_D_digits(next_leaf // block, l, D)
        next_leaf += block
    return code

def shannon_lengths(p, D):
    # l_i = ceil(log_D(1/p_i)): feasible (sum D^{-l_i} <= sum p_i = 1) and
    # achieves H_D(X) <= L < H_D(X) + 1.
    if any(pi <= 0 or pi >= 1 for pi in p):
        raise ValueError("use a non-degenerate positive-probability support")
    return [ceil(log(1.0 / pi, D)) for pi in p]

def entropy_D(p, D):
    return sum(pi * log(1.0 / pi, D) for pi in p if pi > 0)

def expected_length(lengths, p):
    return sum(pi * l for pi, l in zip(p, lengths))

# sanity: D=2, p = (1/2, 1/4, 1/8, 1/8) is dyadic -> L = H exactly.
# lengths (1,2,3,3): kraft_sum = 1/2+1/4+1/8+1/8 = 1 (budget filled),
# build_prefix_code -> [(0,), (1, 0), (1, 1, 0), (1, 1, 1)],
# expected_length = 1.75 = H_2(X).
```
