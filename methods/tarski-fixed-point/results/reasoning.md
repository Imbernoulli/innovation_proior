I want a fixed point, but I do not want to sneak in a metric. If I ask for a contraction, I have already changed the problem: I need distances, I need a shrinking constant, and I will probably get uniqueness. That is too much structure and the wrong kind of structure. In an ordered space, many fixed points can be meaningful. A recursive definition can have many self-consistent solutions, and the least one is the one generated without adding extra facts. A game with strategic complements can have a lowest and a highest equilibrium. So uniqueness is not the prize. Existence plus order structure is the prize.

Topology has the same problem from another direction. A Brouwer-style proof wants compactness, convexity, and continuity. Those are powerful, but they do not explain why a map that only respects `<=` should stabilize. The data I actually have are these: every subset has a join and a meet, and whenever `x <= y`, monotonicity gives `f(x) <= f(y)`. I should stare at what fixed points mean in that language.

A fixed point is an element where the map neither pushes above nor below. Before equality, there are two one-sided notions. Some elements sit above their images: `f(x) <= x`. These are upper stable bounds. If I am below one of them and apply `f`, monotonicity helps keep me below it. Dually, some elements sit below their images: `x <= f(x)`. These are lower stable bounds. The complete lattice can take the meet of all upper stable bounds and the join of all lower stable bounds. Maybe equality is hiding exactly at those extremal bounds.

Before reaching for extremal bounds, the obvious move is to iterate: start at `bot` and climb, `bot`, `f(bot)`, `f(f(bot))`, and so on, hoping the increasing chain settles on a fixed point. Monotonicity does make the chain increasing, since `bot <= f(bot)` trivially and applying `f` to both sides of any `x <= f(x)` preserves the inequality. On a finite lattice this chain cannot climb forever, so it must stop, and it stops exactly at a fixed point. But `L` need not be finite, and nothing in the hypotheses says `f` preserves the join of an infinite chain. Completeness hands me a candidate limit, the join of the whole chain, but without an assumption that `f` carries chain-suprema to chain-suprema, applying `f` to that join need not land back on it; the honest way to push the construction through in general is to keep iterating past every finite stage, transfinitely along the ordinals, which is a far heavier and less clean argument than I want, and it quietly readmits the very kind of continuity structure I set out to avoid by working with order alone. If plain iteration cannot be trusted to reach a fixed point for an arbitrary monotone map, the fixed point has to be pinned down by a condition on the whole lattice at once, not assembled one step at a time.

Let me try the upper side first, because least fixed points are usually the inductive ones. Define

`P = {x in L | f(x) <= x}`.

This set is not empty: `top` is in it, since nothing is above `top`, so `f(top) <= top`. Completeness lets me form

`a = meet P`.

Now I need `a = f(a)`. Take any `p in P`. Since `a <= p`, monotonicity gives `f(a) <= f(p)`. Since `p` is in `P`, `f(p) <= p`. Therefore `f(a) <= p` for every `p in P`. So `f(a)` is a lower bound of `P`, and `a` is the greatest lower bound. Hence `f(a) <= a`.

That proves `a` itself belongs to `P`. But if `a in P`, and `a` is the meet of all elements of `P`, then `a <= f(a)` because `f(a)` is one of the elements of `P`. I have both inequalities:

`f(a) <= a` and `a <= f(a)`.

So `f(a) = a`. The meet of all upper stable bounds is not merely another upper bound; monotonicity makes its image an upper stable bound too, and completeness squeezes equality out of the two inequalities. Also, every fixed point `z` satisfies `f(z) = z`, so `z in P`, and therefore `a <= z`. This `a` is the least fixed point.

There is something worth pausing on in that argument. I did not build `a` by exhibiting it as an explicit term, an iterate of `f` starting from `bot`, or any finite recipe in `f` and the lattice operations. I built it by quantifying over the whole set `P` and taking its meet, and only afterward discovered that `a` itself is a member of the very set I quantified over. The object is picked out by a totality it belongs to, not assembled from below. That is what makes this an existence argument rather than a construction: I know `a` exists and is extremal, but there is no sequence of approximations converging up to it the way the chain `bot, f(bot), f(f(bot)), ...` would have supplied one if chain-continuity had been available. Giving up the continuity assumption bought generality at the price of that explicit recipe.

The lower side is the same argument turned upside down. Define

`Q = {x in L | x <= f(x)}`.

It is nonempty because `bot in Q`. Let

`b = join Q`.

For any `q in Q`, `q <= b`, so `f(q) <= f(b)`. Since `q <= f(q)`, I get `q <= f(b)` for every `q in Q`. Thus `f(b)` is an upper bound of `Q`, and because `b` is the least upper bound, `b <= f(b)`. That puts `b` in `Q`. Since `b` is the join of all elements of `Q`, and `f(b) in Q`, I also have `f(b) <= b`. Hence `b = f(b)`. Every fixed point lies in `Q`, so every fixed point is below `b`. This is the greatest fixed point.

This already gives existence, but the stronger statement is still waiting. The fixed points should themselves form a complete lattice. I cannot simply say that the join in `L` of a family of fixed points is fixed. It need not be. I need to close the join using the same extremal-bound argument, but now inside an interval.

Take any set `A` of fixed points and let `s = join_L A`. For each `x in A`, `x = f(x)` and `x <= s`, so `x = f(x) <= f(s)`. Since `f(s)` is an upper bound of `A`, the least upper bound satisfies `s <= f(s)`. So `s` is below its image. That is exactly enough to make the interval `[s, top]` stable under `f`: if `y` is in `[s, top]`, then `s <= y`, hence `f(s) <= f(y)`, and since `s <= f(s)`, I get `s <= f(y)`. Also `f(y) <= top`. Thus `f` maps `[s, top]` into itself.

The interval `[s, top]` is a complete lattice: for any subset of it, its join in `L` still lies above `s`, and its meet in `L` still lies below `top` and above `s` because `s` is a lower bound of the subset; empty joins and meets give `s` and `top`. On this interval, the least-fixed-point argument applies. Let `j` be the least fixed point of `f` in `[s, top]`. Then `j` is a fixed point in `L`, and it is above every element of `A` because it is above `s`. If `u` is any fixed point in `L` that is above every element of `A`, then `s <= u`, so `u` lies in `[s, top]`; since `j` is the least fixed point there, `j <= u`. Therefore `j` is the join of `A` inside the fixed-point set.

Meets are dual. If `t = meet_L A`, then for every `x in A`, `f(t) <= f(x) = x`, so `f(t) <= t`. The interval `[bot, t]` is stable under `f`: if `y <= t`, then `f(y) <= f(t) <= t`. Applying the greatest-fixed-point argument inside `[bot, t]` gives the greatest fixed point below `t`, and that element is the meet of `A` inside the fixed-point set. Empty `A` is included in the same reasoning: the join of the empty family in the fixed-point lattice is the least fixed point, and the meet is the greatest fixed point.

So the mechanism is very clean. The map does not have to pull points closer together. It does not have to move continuously. The complete lattice lets me collect all one-sided stable bounds at once; monotonicity makes those collections stable under `f`; and the meet/join operations turn one-sided stability into equality. Order replaces metric contraction and topology.

The final theorem is now unavoidable: every monotone self-map of a complete lattice has a complete lattice of fixed points; its least fixed point is `meet {x | f(x) <= x}`, and its greatest fixed point is `join {x | x <= f(x)}`. The least one is the inductive solution constrained by all pre-fixed upper bounds. The greatest one is the coinductive solution supported by all post-fixed lower bounds. In a recursive program semantics, this explains why the least self-consistent meaning is canonical. In a supermodular game, it explains why monotone best responses do not merely have an equilibrium, but have ordered extremal equilibria.

One instance is worth checking by hand, because it shows the abstraction is not empty formalism. Take two sets `A` and `B` with injections `f : A -> B` and `h : B -> A`, neither assumed onto. The power set of `A`, ordered by inclusion, is a complete lattice, with union and intersection as join and meet. Define `g(X) = A \ h(B \ f(X))` for `X subseteq A`. If `X <= Y`, then `f(X) <= f(Y)`, so `B \ f(Y) <= B \ f(X)`, so `h(B \ f(Y)) <= h(B \ f(X))`, so `A \ h(B \ f(X)) <= A \ h(B \ f(Y))`: two order-reversing complementations, one inside `B` and one inside `A`, cancel back to order-preserving, so `g` is monotone on a complete lattice and has a fixed point `C`, with `C = A \ h(B \ f(C))`. That single equality already builds a bijection between `A` and `B`: map `x in C` to `f(x)`, and map `x in A \ C` to the `h`-preimage of `x`. The preimage exists because `A \ C = A \ (A \ h(B \ f(C))) = h(B \ f(C))`, so every point of `A \ C` is an `h`-image of some point of `B \ f(C)`, and `h` injective makes that point unique. This is the Schroder-Bernstein theorem, obtained as a byproduct of an existence statement about a monotone map on a power set, with no cardinal arithmetic and no explicit description of the bijection beyond locating the fixed set `C`. Turning a pair of injections into a fixed point of a map on subsets is exactly the kind of leverage the lattice-only argument was built to reach.
