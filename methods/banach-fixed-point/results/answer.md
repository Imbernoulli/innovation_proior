# Banach Fixed-Point Theorem

Let `(X,d)` be a nonempty complete metric space. Let `T:X->X` satisfy a global contraction estimate: there is a constant `q` with `0 <= q < 1` such that

`d(Tx,Ty) <= q d(x,y)`

for every `x,y in X`.

Then `T` has a unique fixed point `x*`, and for every starting point `x_0 in X`, the Picard iterates

`x_{n+1}=T(x_n)`

converge to `x*`.

## Proof

If `q=0`, then `T` sends all points to one point, say `a=T(x_0)`. Since `T(a)=T(T(x_0))=a`, this point is fixed. It is unique because all images are equal. Assume now `0<q<1`.

For the iterates, the contraction estimate gives

`d(x_{n+1},x_n)=d(Tx_n,Tx_{n-1}) <= q d(x_n,x_{n-1})`.

By induction,

`d(x_{n+1},x_n) <= q^n d(x_1,x_0)`.

For `m>n`, the triangle inequality gives

`d(x_m,x_n) <= sum_{k=n}^{m-1} d(x_{k+1},x_k)`

and hence

`d(x_m,x_n) <= d(x_1,x_0) sum_{k=n}^{m-1} q^k`

`= q^n (1-q^{m-n}) d(x_1,x_0)/(1-q)`

`<= q^n d(x_1,x_0)/(1-q)`.

The right side tends to `0` as `n->infinity`, uniformly in `m>n`, so `(x_n)` is Cauchy. Completeness gives a point `x* in X` with `x_n -> x*`.

Now

`d(Tx*,x*) <= d(Tx*,Tx_n)+d(x_{n+1},x*)`

`<= q d(x*,x_n)+d(x_{n+1},x*)`.

Letting `n->infinity` gives `d(Tx*,x*)=0`, so `Tx*=x*`.

If `u` and `v` are fixed points, then

`d(u,v)=d(Tu,Tv) <= q d(u,v)`.

Since `q<1`, this forces `d(u,v)=0`, hence `u=v`. The fixed point is unique.

The same tail estimate gives the geometric error bound

`d(x*,x_n) <= q^n d(x_1,x_0)/(1-q)`.

A useful a posteriori form is

`d(x*,x_{n+1}) <= q d(x_{n+1},x_n)/(1-q)`.

This is the full mechanism: the contraction constant turns iteration into a Cauchy process, completeness supplies the limit, and the contraction inequality makes the limit both fixed and unique.
