I want to solve `X = U(X)`, but the equation is sitting in a space where "solve" may not mean isolate `X`. If `X` is a function and `U` is an integral operation, direct algebra is the wrong reflex. The analytic thing I can actually do is repeat the operation: choose some `Y`, set `X_1 = Y`, then `X_{n+1} = U(X_n)`. This gives me a concrete sequence. The whole question becomes whether this sequence has any reason to settle, and whether the point it settles on is forced to satisfy the equation.

Continuity alone is too soft. If I know only that nearby points go to nearby points, that tells me what happens after the sequence is already converging; it does not create convergence. Compact topology gives another kind of answer: a continuous self-map of a filled compact region may have a fixed point because the region has the right shape. But that answer is silent about this orbit I can compute. It also allows multiple fixed points, and iteration can behave badly. I need a hypothesis that talks directly about the motion under repeated application.

Distances are the only quantities that survive in this abstract setting. If `U` makes every pair of points no farther apart, `||U(X)-U(Y)|| <= ||X-Y||`, I still do not have enough. Equal distances can preserve oscillations, and even strict shrinking without a uniform margin can waste all its shrinking at infinity. So let me try to demand the same loss of distance every time and everywhere: a number `M` with `0 <= M < 1` such that

`||U(X') - U(X'')|| <= M ||X' - X''||`

for all points of the space. If `M = 0`, the map collapses every point to the same image and the second iterate is already fixed. The interesting case is `0 < M < 1`. This single inequality is stronger than continuity, and it is exactly about dynamics: after one application all separations are multiplied by at most `M`. Whether it is enough is the thing to find out, not to assume.

Let me see what it does to the orbit. The first useful distance is not from `X_n` to the unknown fixed point, because I do not have that point yet. It is the distance between two successive approximations. Since `X_{n+1}=U(X_n)`,

`||X_{n+1}-X_n|| = ||U(X_n)-U(X_{n-1})|| <= M ||X_n-X_{n-1}||`.

Apply the same estimate repeatedly:

`||X_{n+1}-X_n|| <= M^{n-1} ||X_2-X_1||`

if the indexing starts with `X_1=Y`. So the jumps do not merely tend to zero; their upper bounds form a geometric series. That is more than I asked for, and it is the part that might rescue convergence.

A sequence is not Cauchy just because consecutive jumps go to zero, so I have to use the triangle inequality across a whole tail. If `p>q`, then

`||X_p-X_q|| <= ||X_p-X_{p-1}|| + ... + ||X_{q+1}-X_q||`.

Each term is controlled by the first jump:

`||X_p-X_q|| <= (M^{p-2}+...+M^{q-1}) ||X_2-X_1||`.

The finite geometric tail is at most `M^{q-1}/(1-M)`, so

`||X_p-X_q|| <= M^{q-1} ||X_2-X_1|| / (1-M)`.

As `q` goes to infinity this bound goes to zero, uniformly for all `p>q`. So the orbit is Cauchy. Now I want to know what role completeness is really playing here, because so far I have only used the distance inequality and the triangle inequality. The Cauchy estimate manufactures tail collapse; it does not by itself produce a limit that lives in the space. To test whether that gap matters, take the simplest possible contraction and put it on a space with a hole. On the half-open interval `(0,1]` let `U(x)=x/2`, which is a contraction with `M=1/2`. Starting from `x=1` the orbit is `1, 1/2, 1/4, 1/8, ...`, marching steadily toward `0`. The fixed-point equation `p=p/2` has the unique solution `p=0`, and `0` is not in `(0,1]`. So here is a genuine contraction whose orbit is Cauchy and yet has no fixed point in the space at all. That settles the question: the distance inequality alone is not enough, and completeness is doing real work, not bookkeeping. It is exactly the axiom that turns the manufactured Cauchy behavior into an actual element `X` of the same space.

So assume completeness, and now `X_n -> X`. I still need `X = U(X)`. I could invoke continuity, since the distance inequality makes `U` Lipschitz and hence continuous. Then `U(X_n) -> U(X)`, but `U(X_n)=X_{n+1}` and the shifted sequence has the same limit `X`, so `U(X)=X`. There is also a direct metric check that avoids treating continuity as a separate ingredient:

`||U(X)-X|| <= ||U(X)-U(X_n)|| + ||X_{n+1}-X|| <= M||X-X_n|| + ||X_{n+1}-X||`,

and both terms go to zero. Therefore `||U(X)-X||=0`, so `U(X)=X`.

Existence is done, and the proof has not used linearity. The normed setting gives the language and completeness, but the argument itself only asks for a distance, the triangle inequality, and Cauchy limits. That suggests the theorem should belong to complete metric spaces, not just linear spaces — and indeed nothing above mentioned addition or scaling once the inequality was in hand.

Uniqueness is even cleaner. Suppose `P` and `Q` are both fixed. Then

`d(P,Q) = d(U(P),U(Q)) <= M d(P,Q)`.

Since `M<1`, the only nonnegative number that can be at most `M` times itself is zero. Thus `d(P,Q)=0`, and `P=Q`. So the fixed point this orbit finds is also the only fixed point the map can have.

The estimates also tell me how much the process knows about its own error. Once the limit `X` exists, let `p -> infinity` in the tail estimate. With indexing `X_0=x_0`, `X_{n+1}=U(X_n)`, the same calculation gives

`d(X,X_n) <= M^n d(X_1,X_0)/(1-M)`.

That is an a priori bound from the first observed jump. There is a sharper local version after step `n`: the future tail from `X_{n+1}` is bounded by

`d(X,X_{n+1}) <= d(X_{n+2},X_{n+1}) + d(X_{n+3},X_{n+2}) + ...`.

Because each later jump is at most another factor `M` of the previous one,

`d(X,X_{n+1}) <= d(X_{n+2},X_{n+1})/(1-M) <= M d(X_{n+1},X_n)/(1-M)`.

Before I trust either bound I should watch the machine run on a case where I know the answer, including a case where the orbit does not crawl monotonically toward the limit but overshoots it. Take `U(x)=-x/2+3` on the real line, a contraction with `M=1/2`; its fixed point solves `x=-x/2+3`, i.e. `(3/2)x=3`, so `x*=2`. Starting at `x_0=0` the orbit is

`0, 3, 1.5, 2.25, 1.875, 2.0625, 1.96875, ...`,

oscillating to either side of `2`. The first jump is `d(x_1,x_0)=3`. Take `n=3`, where `x_3=2.25`, so the true error is `d(x*,x_3)=0.25`. The a priori bound predicts `M^3 d(x_1,x_0)/(1-M) = (1/8)(3)/(1/2)=0.75`, and `0.25 <= 0.75` holds. The a posteriori bound uses the last jump `d(x_3,x_2)=0.75`: it predicts `M d(x_3,x_2)/(1-M) = (1/2)(0.75)/(1/2)=0.75`, and again `0.25 <= 0.75` holds. Both bounds are satisfied and both are loose by a factor of three here, which is the expected behavior — the orbit alternates sides of `x*`, so consecutive jumps overcount the genuine distance to the limit. If I instead run the monotone contraction `U(x)=x/2+1` from `0` (fixed point `2`, orbit `0,1,1.5,1.75,...`), the same two bounds come out equal to the true error at every step, the saturated case. So the inequalities hold in both regimes and tighten exactly when the orbit approaches from one side. Nothing in the run contradicts the derived estimates, and the loose-vs-saturated split is exactly what the geometric-series derivation should produce.

So the iteration is not just a proof of existence. It is an approximation scheme with a geometric error certificate. The constant `M` is the whole clock: close to zero means rapid collapse, close to one means slow convergence, and at one the argument loses its geometric summability.

This also clarifies the difference from topological fixed-point existence. A topological theorem uses the shape of a domain to prevent every point from being moved away from itself; it can work with arbitrary continuous maps on compact convex sets, but the conclusion is qualitative. Here the map may live in a function space and the domain need not be compact. The price is a strong metric hypothesis. The reward is existence, uniqueness, convergence from every start, and explicit bounds, all from the same inequality.

The final theorem is therefore: in a nonempty complete metric space `(X,d)`, any self-map `T:X->X` for which there is a constant `q` with `0 <= q < 1` and `d(Tx,Ty) <= q d(x,y)` for all `x,y` has a unique fixed point `x*`. For any starting point `x_0`, the iterates `x_{n+1}=T x_n` converge to `x*`, and for `m>n`,

`d(x_m,x_n) <= q^n (1-q^{m-n}) d(x_1,x_0)/(1-q) <= q^n d(x_1,x_0)/(1-q)`.

Letting `m` tend to infinity gives `d(x*,x_n) <= q^n d(x_1,x_0)/(1-q)`, and comparing to the most recent jump gives `d(x*,x_{n+1}) <= q d(x_{n+1},x_n)/(1-q)`. The causal chain is the one the orbit on `(0,1]` and the run on `x*=2` together exhibited: a uniform contraction constant makes successive jumps geometrically summable; geometric summability makes the Picard orbit Cauchy; completeness — without which the `(0,1]` orbit had nowhere to land — supplies the limit; the contraction inequality passes the limit through the map; and the same inequality shrinks the distance between any two fixed points to zero.
