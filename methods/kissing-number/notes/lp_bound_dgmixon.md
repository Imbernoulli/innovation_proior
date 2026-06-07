# Delsarte LP bound (from dustingmixon explainer)

Spherical code X = finite set of unit vectors in R^d. A(X)={<x,y>: x!=y}.

Gegenbauer (ultraspherical) polynomials Q_k (normalized Q_k(1)=1), dimension d:
KEY positivity: for ANY finite X,  sum_{x,y in X} Q_k(<x,y>) >= 0  for all k>=0.
(Q_0 term gives |X|^2.)

LP bound theorem: For A-code X (A(X) subset A subset [-1,1)),
|X| <= inf{ F(1) : F = sum_k f_k Q_k, f_0 = 1, f_k>=0 all k, F(alpha)<=0 for all alpha in A }.

Proof: S := sum_{x,y in X} F(<x,y>).
 Upper: S = |X|F(1) + sum_{x!=y} F(<x,y>) <= |X|F(1) since F<=0 on A(X).
 Lower: S = sum_k f_k sum_{x,y} Q_k(<x,y>) >= f_0 * |X|^2 = |X|^2 (drop k>=1 nonneg terms).
 => |X|^2 <= |X| F(1) => |X| <= F(1).

E8 / dim 8 / tau=240: 240 min vectors of E8 on S^7, A(X8)={-1,-1/2,0,1/2}.
Magic polynomial:
 F(t) = (320/3)(t+1)(t+1/2)^2 t^2 (t-1/2).
 F = sum_{k=0}^6 f_k Q_k with f_0=1, f_k>=0. F(1)=240. => optimal.

Leech / dim 24 / tau=196560: analogous degree-10 polynomial, A={-1,-1/2,-1/4,0,1/4,1/2}.
