I want a fixed point, but I do not want to sneak in a metric. If I ask for a contraction, I have already changed the problem: I need distances, I need a shrinking constant, and I will probably get uniqueness. That is too much structure and the wrong kind of structure. In an ordered space, many fixed points can be meaningful. A recursive definition can have many self-consistent solutions, and the least one is the one generated without adding extra facts. A game with strategic complements can have a lowest and a highest equilibrium. So uniqueness is not the prize. Existence plus order structure is the prize.

Topology has the same problem from another direction. A Brouwer-style proof wants compactness, convexity, and continuity. Those are powerful, but they do not explain why a map that only respects `<=` should stabilize. The data I actually have are these: every subset has a join and a meet, and whenever `x <= y`, monotonicity gives `f(x) <= f(y)`. I should stare at what fixed points mean in that language.

A fixed point is an element where the map neither pushes above nor below. Before equality, there are two one-sided notions. Some elements sit above their images: `f(x) <= x`. These are upper stable bounds. If I am below one of them and apply `f`, monotonicity helps keep me below it. Dually, some elements sit below their images: `x <= f(x)`. These are lower stable bounds. The complete lattice can take the meet of all upper stable bounds and the join of all lower stable bounds. Maybe equality is hiding exactly at those extremal bounds.

Let me try the upper side first, because least fixed points are usually the inductive ones. Define

`P = {x in L | f(x) <= x}`.

This set is not empty: `top` is in it, since nothing is above `top`, so `f(top) <= top`. Completeness lets me form

`a = meet P`.

Now I need `a = f(a)`. Take any `p in P`. Since `a <= p`, monotonicity gives `f(a) <= f(p)`. Since `p` is in `P`, `f(p) <= p`. Therefore `f(a) <= p` for every `p in P`. So `f(a)` is a lower bound of `P`, and `a` is the greatest lower bound. Hence `f(a) <= a`.

That proves `a` itself belongs to `P`. But if `a in P`, and `a` is the meet of all elements of `P`, then `a <= f(a)` because `f(a)` is one of the elements of `P`. I have both inequalities:

`f(a) <= a` and `a <= f(a)`.

So `f(a) = a`. The meet of all upper stable bounds is not merely another upper bound; monotonicity makes its image an upper stable bound too, and completeness squeezes equality out of the two inequalities. Also, every fixed point `z` satisfies `f(z) = z`, so `z in P`, and therefore `a <= z`. This `a` is the least fixed point.

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
