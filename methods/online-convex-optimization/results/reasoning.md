What is actually bothering me is that I keep proving the *same theorem* over and over in different costumes. I have the experts result: pick a distribution over `n` experts each round, an adversary hands me a cost vector, and weighted majority — multiply each expert's weight by `exp(−η·cost)`, renormalize — gets me regret `O(√(T log n))` against the best single expert, with no assumption whatsoever on the cost sequence. I have the online-regression result: predict with a weight vector, take a gradient step on the round's squared error, and a relative-loss bound falls out against the best fixed predictor. And I have the games result, infinitesimal gradient ascent: in a two-player two-action matrix game each player nudges its mixed strategy up its own payoff gradient, and even when the strategies cycle the *average* payoffs converge to a Nash equilibrium. Three settings, three algorithms, three potential-function arguments — entropic here, a Bregman divergence there, an eigenvalue analysis of a `2×2` linear system over there. They smell like the same theorem. I want to find the one object they are all special cases of, and the one algorithm and one proof that covers all of them.

So what do they share? In every one of them I commit to *something* before I see the round's cost. In experts I commit to a distribution `x` over the simplex; in regression I commit to a weight vector; in the game I commit to a mixed strategy in the unit square. In every one the thing I commit to lives in a *convex* set: the simplex is convex, the box is convex, all of `ℝⁿ` is convex. And in every one the cost, as a function of the thing I committed to, is *convex*: experts cost `c^t·x` is linear hence convex; squared error is convex; the game payoff is bilinear, so for a fixed opponent move it is linear in my strategy. And in every one I do not get to assume anything about how the costs are generated — the adversary, or the adapting opponent, can do what it likes. That is the common skeleton. Let me just *name* it and refuse to specialize: a fixed convex feasible set `F ⊆ ℝⁿ`, known in advance; a sequence of convex cost functions `c¹, c², …`, each revealed only *after* I commit; on round `t` I pick `x^t ∈ F`, then `c^t` arrives. No distribution on the `c^t`, no relationship between `c^t` and `c^{t+1}`. That is the whole setting.

Now, what is the *goal*? I have to get this right, because it is where naïve formulations die. I cannot ask to minimize `c^t(x^t)` on each round — I commit `x^t` before I see `c^t`, and `c^t` can be anything, so the round's own optimum is simply unknowable in advance; demanding it is incoherent. So I need a comparator that is also blind in the right way but allowed hindsight. The honest comparator is the *single best fixed point chosen with full hindsight of the whole cost sequence*: an offline player who sees `c¹, …, c^T` all at once but must commit to one `x* ∈ F` for all rounds. My excess cost over that player is the regret,

`R(T) = Σ_{t=1}^T c^t(x^t) − min_{x*∈F} Σ_{t=1}^T c^t(x*)`,

and what I want is for the *average* regret `R(T)/T` to go to zero as `T` grows. That is the right bar: it is exactly "Hannan consistency" / the no-regret guarantee from the game-theory side, and it is exactly what weighted majority delivers in the experts case, and it asks for nothing about the cost distribution. Good — the setting and the goal already unify experts and games and regression. The question is whether one simple algorithm hits this bar for *arbitrary* convex `F` and *arbitrary* convex `c^t`.

The simplest thing I know how to do with a convex cost is descend its gradient. Offline, to minimize a fixed convex `f` over `F` I would iterate `x ← P(x − η∇f(x))` — step against the gradient, then project back into `F`. Here the cost is different every round, but the move suggests itself anyway: when `c^t` arrives, take a gradient step on *that* loss from where I currently am, project back, and that projected point is what I play next.

`x^{t+1} = P(x^t − η_t ∇c^t(x^t))`.

That is it — that is the entire algorithm. Pick any feasible `x^1`, and from then on each revealed cost just pushes me one projected-gradient step. It is almost suspiciously plain. Whether it is any *good* is entirely a question of whether I can bound its regret, and the worry is real: the `c^t` are unrelated and adversarial, so I am descending a *different* hill every round. There is no fixed landscape I am converging on. Why on earth should chasing a moving gradient track the best *fixed* point?

Let me try to bound it and see where it breaks. The comparator term `min_{x*∈F} Σ c^t(x*)` has `c^t` in it, and `c^t` could be any convex function — I do not want my proof to depend on its shape. The first move has to be to get rid of the function and keep only its gradient at the point I actually played, because the gradient is all my algorithm ever looks at. Convexity hands me exactly that. For a convex `c^t`, the tangent plane at `x^t` lies below the function:

`c^t(x) ≥ c^t(x^t) + ∇c^t(x^t)·(x − x^t)` for all `x`.

Write `g^t = ∇c^t(x^t)`. Put `x = x*`:

`c^t(x*) ≥ c^t(x^t) + g^t·(x* − x^t)`, i.e. `c^t(x^t) − c^t(x*) ≤ g^t·(x^t − x*)`.

So my per-round regret against `x*` is upper-bounded by the *linear* quantity `g^t·(x^t − x*)`. Summing,

`R(T) = Σ_t [c^t(x^t) − c^t(x*)] ≤ Σ_t g^t·(x^t − x*)`.

This is the moment the whole problem collapses to something I can handle: I never again need the curved `c^t`; it suffices to control the regret of the *linear* losses `g^t·x`. This also explains why "convex" was the right class to insist on — convexity is precisely what lets the curved loss be replaced from below by a linear one, so that bounding against linear costs bounds against all convex costs. It also tells me the worst case for this algorithm is when the `c^t` are themselves linear: a linear function equals its own tangent, so the inequality is tight there, and the gradient is all the information about `c^t` that my move uses anyway. I will reason as if the losses were `g^t·x`.

Now I need to bound `Σ_t g^t·(x^t − x*)`. I have to bring in the only structure my update created, which is the projected gradient step, and I need a quantity that connects "how I move" to "how far I am from `x*`." The natural thing to watch is my squared distance to the comparator, `‖x^t − x*‖²`. Let me see how it changes in one step. Before projecting, let `y^{t+1} = x^t − η_t g^t`. Then

`‖y^{t+1} − x*‖² = ‖(x^t − x*) − η_t g^t‖² = ‖x^t − x*‖² − 2η_t g^t·(x^t − x*) + η_t² ‖g^t‖²`.

That middle term is exactly the linear quantity I am trying to control — `g^t·(x^t − x*)` — sitting inside a perfect square. So expanding the squared distance *generates* the thing I want to sum, as a difference of consecutive potentials plus an error. Now I just have to deal with the projection, because what I actually play is `x^{t+1} = P(y^{t+1})`, not `y^{t+1}`.

And here is where I would worry the projection ruins everything — it moves my point, so why should `‖x^{t+1} − x*‖²` relate cleanly to `‖y^{t+1} − x*‖²`? But projection onto a convex set is non-expansive *toward any point already in the set*: `x*` is in `F`, so projecting `y^{t+1}` to its closest feasible point cannot increase its distance to `x*`. That is the Pythagorean fact — for `x = P(y)` and any `z ∈ F`, `‖y − z‖ ≥ ‖x − z‖`. So

`‖x^{t+1} − x*‖² = ‖P(y^{t+1}) − x*‖² ≤ ‖y^{t+1} − x*‖²`.

The projection only helps me — it shrinks the potential, never grows it. So the constraint set is *free*: I get to handle an arbitrary convex `F` and pay nothing in the bound for the projection. That is the second thing convexity bought me, and it is why the move is "step then project" rather than anything cleverer. Chaining,

`‖x^{t+1} − x*‖² ≤ ‖x^t − x*‖² − 2η_t g^t·(x^t − x*) + η_t² ‖g^t‖²`.

Solve for the term I want:

`g^t·(x^t − x*) ≤ (‖x^t − x*‖² − ‖x^{t+1} − x*‖²) / (2η_t) + (η_t/2) ‖g^t‖²`.

The squared distance is a *potential*; `(‖x^t − x*‖² − ‖x^{t+1} − x*‖²)/(2η_t)` is a difference of potentials I am hoping will telescope; and `(η_t/2)‖g^t‖²` is the *error I pay for responding only after seeing the cost* — it is the overshoot from taking a step of size `η_t` along a gradient that the adversary got to pick. Let me bound the gradient norm uniformly: assume `‖∇c^t(x)‖ ≤ G` over `F` (this is just `G`-Lipschitz costs), so `‖g^t‖ ≤ G`. Then

`g^t·(x^t − x*) ≤ (‖x^t − x*‖² − ‖x^{t+1} − x*‖²)/(2η_t) + (η_t/2) G²`.

Sum from `t = 1` to `T`:

`Σ_t g^t·(x^t − x*) ≤ Σ_t (‖x^t − x*‖² − ‖x^{t+1} − x*‖²)/(2η_t) + (G²/2) Σ_t η_t`.

The second sum is clean. The first I have to telescope, and I have to be careful, because the `1/(2η_t)` factor changes with `t` — it is not a plain telescope. Let me write `a_t = ‖x^t − x*‖²` and do the Abel rearrangement honestly:

`Σ_{t=1}^T (a_t − a_{t+1})/(2η_t) = a_1/(2η_1) + Σ_{t=2}^T a_t (1/(2η_t) − 1/(2η_{t-1})) − a_{T+1}/(2η_T)`.

The last term `−a_{T+1}/(2η_T) = −‖x^{T+1} − x*‖²/(2η_T) ≤ 0`, so I drop it. Now I want to bound each `a_t = ‖x^t − x*‖²` by the squared diameter `D² = (max_{x,y∈F} ‖x − y‖)²` — but I can only do that to coefficients that are *non-negative*, or I would be turning the inequality the wrong way. So I need `1/(2η_t) − 1/(2η_{t-1}) ≥ 0`, i.e. the step sizes `η_t` must be *non-increasing*. Fine — I will use a decreasing schedule, which I want anyway: late in the game I am more settled and should take smaller steps. With `η_t` non-increasing every coefficient is `≥ 0`, so

`Σ_t (a_t − a_{t+1})/(2η_t) ≤ D² [ 1/(2η_1) + Σ_{t=2}^T (1/(2η_t) − 1/(2η_{t-1})) ] = D²/(2η_T)`,

because that bracket telescopes — `1/(2η_1)` plus all the increments `1/(2η_t) − 1/(2η_{t-1})` collapses to `1/(2η_T)`. So the running regret bound is

`R(T) ≤ Σ_t g^t·(x^t − x*) ≤ D²/(2η_T) + (G²/2) Σ_{t=1}^T η_t`.

Now I can *see* the tension that fixes the step size, and I want to read it off the two terms rather than guess. The first term `D²/(2η_T)` is the price of geometry — it is essentially the cost of possibly having started `x^1` on the wrong side of `F`, the full squared diameter divided by the final step size; it *shrinks* as `η_T` grows, so it wants *large* steps. The second term `(G²/2) Σ η_t` is the accumulated overshoot from responding a beat late on every round; it *grows* with the step sizes, so it wants *small* steps. A fixed positive step that does not depend on the horizon makes the second term grow like `η T` — linear in `T`, fatal. A step shrinking too fast, like `1/t`, makes `Σ η_t` converge but `1/η_T = T` blow up the first term linearly — also fatal. I want the schedule that balances them, and the balance point is visible: I want `1/η_T` and `Σ_{t} η_t` to grow at the *same* rate in `T`. Take `η_t = 1/√t`. Then `1/η_T = √T`, and

`Σ_{t=1}^T 1/√t ≤ 1 + ∫_1^T dt/√t = 1 + [2√t]_1^T = 1 + 2√T − 2 = 2√T − 1`.

Both pieces are `Θ(√T)` — that is the schedule that equalizes the start-up cost against the cumulative overshoot. Plugging `η_t = t^{-1/2}` in:

`R(T) ≤ (D²/2)·√T + (G²/2)·(2√T − 1) = (D²/2)√T + (√T − 1/2) G²`.

There it is. The regret of greedy projection — step against the current gradient, project back, with `η_t = 1/√t` — is `O(√T)`, every term, with *no* assumption on the cost sequence. So `R(T)/T = O(1/√T) → 0`: average regret vanishes. And nothing in this argument touched the shape of `F` beyond convexity, or the shape of `c^t` beyond convexity and a gradient bound. One algorithm, one telescoping potential argument, for the whole class.

Let me sanity-check the constant by tuning more carefully, because if I know the horizon `T` ahead of time I can pick the *single best constant* step `η` rather than the anytime schedule, and the balance becomes exact. With a constant `η`, the same sum gives `R(T) ≤ D²/(2η) + (G²/2) η T` (now the telescope is just `a_1/(2η) ≤ D²/(2η)`). Minimize the right side over `η`: derivative `−D²/(2η²) + G² T/2 = 0` gives `η² = D²/(G² T)`, so `η* = D/(G√T)`, and the value is `D²/(2·D/(G√T)) + (G²/2)·(D/(G√T))·T = (DG√T)/2 + (DG√T)/2 = DG√T`. So with the horizon-tuned constant step the bound is exactly `R(T) ≤ DG√T`. If instead I want the anytime version and set `η_t = D/(G√t)` round by round, the same telescoping with `1/η_T = G√T/D` and `Σ 1/√t ≤ 2√T` gives `2R(T) ≤ D²·(G√T/D) + G²·(D/G)·2√T = DG√T + 2DG√T = 3DG√T`, i.e. `R(T) ≤ (3/2) DG√T` — same `√T` rate, a slightly looser constant, the small price for not knowing `T` in advance. Either way it is `Θ(DG√T)`, governed by the diameter of the set and the Lipschitz constant of the losses, and nothing else.

Is `√T` actually the best possible, or am I just not being clever enough? Let me try to *force* large regret on any algorithm and see how big it has to be. Take `F` the cube `‖x‖_∞ ≤ 1` in `ℝⁿ`, and each round draw the cost `f_t(x) = v_t·x` with `v_t` a uniformly random `±1` sign vector. Whatever point `x_t` I commit before seeing `v_t`, the sign vector is independent of it, so `E[f_t(x_t)] = E[v_t]·x_t = 0` — in expectation I incur zero, every round. But the *comparator* gets to wait and then pick the best fixed corner: `E[min_{x∈F} Σ_t v_t·x] = E[ −Σ_i |Σ_t v_t(i)| ]`, and each `Σ_t v_t(i)` is a `±1` random walk of length `T`, whose expected absolute value is `Θ(√T)`. Over `n` coordinates that is `−Θ(n√T)`. So the expected regret of *any* algorithm on this instance is `Ω(n√T)`. Here the diameter is `D = 2√n` and the gradient bound is `G = √n`, so `DG = 2n`, and `Ω(n√T) = Ω(DG√T)`. The lower bound matches my upper bound up to a constant: the `√T` rate is not an artifact of a lazy proof, and no online algorithm can improve the worst-case order. The plain projected-gradient method has the right minimax scale.

Now I want to watch the old results fall out, because that was the whole motivation. Take the experts problem: `F` is the probability simplex, the cost is linear, `c^t(x) = c^t·x`. Greedy projection becomes: step the distribution along `−η_t c^t`, project back onto the simplex. My bound gives `O(√T)` regret against the best fixed distribution — hence against the best single expert, since the best fixed point of a linear cost on the simplex is a vertex. Interesting: weighted majority gets `O(√(T log n))` here and my bound has *no* `log n` — instead it carries the simplex's diameter. The two are genuinely different and incomparable: weighted majority runs the *entropic* potential `Σ w_i` and pays `log n` (the number of experts); I run the *Euclidean* potential `‖x − x*‖²` and pay the diameter, which is unrelated to how many experts there are. Same setting, two different geometries of the same proof template — and that is exactly the unification I was after: weighted majority is not a different *kind* of result, it is the simplex-with-entropic-potential instance of "linearize by convexity, telescope a potential, balance the step size."

Now the games case, which is where I started and which IGA only solved for `2×2`. In a repeated game, before each round I commit a mixed strategy — a distribution over my actions — which is a point in the simplex `F`; once the opponent's action is revealed, my expected utility `u` is a *linear* function of my strategy. I have a utility to *maximize* rather than a cost to minimize, so I ascend instead of descend: `y^{t+1} = x^t + η_t ∇u(·, h_{t,2})`, then `x^{t+1} = P(y^{t+1})` back onto the simplex. This is exactly greedy projection on the loss `−u`, so the same `O(√T)` regret bound holds — and crucially it holds for *any number of actions*, because nothing in my telescoping argument cared about the dimension of `F` or its being the `2×2` box; the simplex of any size is just a convex set with some diameter. So this *generalizes* infinitesimal gradient ascent off of the `2×2` case entirely — call it generalized infinitesimal gradient ascent. And with `η_t = 1/√t`, average regret against the best fixed action goes to zero against *any* environment, adversarial or adaptive: the strategy is universally consistent. IGA needed an eigenvalue analysis of a `2×2` linear system and got stuck there; the convex-analysis argument needs none of that and has no dimension ceiling.

One more comparator is worth chasing, because "best *fixed* point in hindsight" is sometimes too weak — what if the environment is slowly drifting and even the offline player should be allowed to *move* a little? Let me let the comparator be a whole sequence `z^1, …, z^T` with path length `Σ_{t=2}^T ‖z^{t-1} − z^t‖ ≤ L`. The per-step inequality is unchanged, but now the comparator changes each round, so with fixed `η` the potential differences are `‖x^t − z^t‖² − ‖x^{t+1} − z^t‖²`. I need to expand them before summing, because a moving `z^t` will not telescope by itself:

`Σ_t [‖x^t − z^t‖² − ‖x^{t+1} − z^t‖²]/(2η)`

`= [‖x^1‖² − ‖x^{T+1}‖²]/(2η) + [x^{T+1}·z^T − x^1·z^1]/η + Σ_{t=2}^T (z^{t-1} − z^t)·x^t/η`.

I can translate the coordinate system so `0 ∈ F` without changing distances. Then every feasible point has norm at most the diameter `D = ‖F‖`, and any two feasible points have inner product at least `−D²/4` and at most `D²`: the lower bound is the worst case when two feasible points point in opposite directions and their distance is still at most `D`. The first boundary term contributes at most `D²/(2η)`, the terminal dot product at most `D²/η`, and `−x^1·z^1/η` at most `D²/(4η)`, so the fixed boundary pieces collect to `7D²/(4η)`. The moving-target sum is bounded by Cauchy-Schwarz:

`Σ_{t=2}^T (z^{t-1} − z^t)·x^t/η ≤ Σ_{t=2}^T ‖z^{t-1} − z^t‖ D/η ≤ LD/η`.

Keeping the same gradient-error term `TηG²/2`, the dynamic regret comes out as

`R(T, L) ≤ 7D²/(4η) + L D/η + T η G²/2`,

with `D = ‖F‖` the diameter and `G = ‖∇c‖`. The shape is the lesson: if the target moves only a little (`L` small), tracking it still costs `O(√T)` with `η ∼ 1/√T`, plus a price `L D/η` that is *linear in how far the target travels*. A static target is the `L = 0` case and has the same static-regret scale, with a looser boundary constant from this moving-target proof. So the same potential argument, barely perturbed, also handles a drifting comparator — which is exactly the regime (slowly-changing environments) the fixed-point comparator was too crude for.

So the causal chain is short and it closes. I was proving one theorem three times because experts, online regression, and gradient-ascent-in-games are all the same thing: committing a point in a fixed convex set against an adversarially chosen convex loss, judged by regret to the best fixed point. Once I name that setting, the algorithm is forced — descend the only thing my move can see, the gradient of the revealed loss, and project back into the set. Convexity lets me replace each curved loss by its tangent from below, so it suffices to beat *linear* losses; the squared distance to the comparator is a potential whose one-step change *is* the linear regret term plus an `η²G²` overshoot; projection is non-expansive so the constraint costs nothing; the potential telescopes against the diameter; and the two surviving terms — start-up `D²/(2η_T)` versus accumulated overshoot `(G²/2)Σ η_t` — are balanced by `η_t = 1/√t` (or the horizon-tuned `η = D/(G√T)`), giving regret `O(DG√T)`, tight up to constants by the random-sign lower bound. Experts and IGA drop out as special cases; the games case becomes universally consistent for any number of actions; and a moving comparator adds the `LD/η` movement term.

```python
import numpy as np

def online_gradient_descent(project, reveal_cost, x1, T, step_size=None):
    """Greedy Projection / Online Gradient Descent.

    project(y) -> closest point in the fixed convex set F to y (non-expansive).
    reveal_cost(t, x) -> round t's convex cost after x has been committed.
    x1                -> arbitrary feasible starting point in F.
    step_size(t)      -> e.g. 1/sqrt(t), D/(G*sqrt(t)), or D/(G*sqrt(T)).
    Regret vs the best fixed x* in F is O(D G sqrt(T)), no assumptions on costs.
    """
    x = np.array(x1, dtype=float)
    history = []
    for t in range(1, T + 1):
        # Commit x^t BEFORE the round's cost is revealed.
        play = x.copy()

        # Adversary reveals c^t only after the commitment. The update uses only
        # the gradient at the played point.
        c_t = reveal_cost(t, play)
        g = np.asarray(c_t.grad(play), dtype=float)  # g^t = grad c^t(x^t)

        # Step size balances the start-up term D^2/(2 eta_T) against the
        # accumulated overshoot (G^2/2) sum eta_t.
        eta = step_size(t) if step_size is not None else 1.0 / np.sqrt(t)

        # Gradient step, then project back into F. Projection is non-expansive
        # toward any feasible point, so the constraint set is handled for free
        # and the potential ||x - x*||^2 only shrinks under it.
        y = play - eta * g                      # y^{t+1} = x^t - eta_t g^t
        x = project(y)                          # x^{t+1} = P(y^{t+1})

        history.append(play)
    return history
```
