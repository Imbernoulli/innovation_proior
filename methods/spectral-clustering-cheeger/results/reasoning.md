I want to cut a graph into two well-separated clusters, and I want to do it well, not heuristically. The way I keep coming back to scoring a candidate set $S\subseteq V$ is: count the edges that leave it, $|\partial(S)|$, and divide by how big $S$ is, because a single edge dangling off a huge blob shouldn't count the same as a single edge holding two halves together. So the thing I actually care about is $\phi(S)=|\partial(S)|/\min(d(S),d(V\setminus S))$, the conductance, and the best nontrivial cut is $\phi_G=\min_{\varnothing\ne S\subsetneq V}\phi(S)$. Small means a real bottleneck; large means the graph is an expander with no good cut. And the trouble is staring at me immediately: that minimum is over all $2^{|V|}$ subsets. I can't enumerate them; the sparsest-cut / minimum-conductance problem is NP-hard. So I'm not going to find $\phi_G$ by searching. What I want is something I *can* compute in polynomial time, and — this is the part I'll have to actually earn, not assume — a way to tie whatever I compute back to $\phi_G$ with an honest inequality, so that a cut I pull out of it comes with a certificate rather than a hope.

Let me look at what makes the problem hard and where it might soften. The hardness is the discreteness — $S$ is a yes/no choice per vertex, an integer object. The classic move when an integer program is intractable is to relax the integrality and hope the continuous relaxation both is tractable and stays close to the integer optimum. So I want to write the cut as something that *has* a continuous version.

Pick the indicator vector $\chi_S$, with $\chi_S(u)=1$ if $u\in S$ and $0$ otherwise. What is $|\partial(S)|$ in terms of $\chi_S$? An edge $(u,v)$ is cut exactly when one endpoint is in $S$ and the other isn't, i.e. when $\chi_S(u)\ne\chi_S(v)$, i.e. when $(\chi_S(u)-\chi_S(v))^2=1$; otherwise that term is $0$. So

$$\sum_{(u,v)\in E}(\chi_S(u)-\chi_S(v))^2 = |\partial(S)|.$$

That sum of squared differences across edges is exactly the Laplacian quadratic form. With $L=D-A$, for any real vector $x$,

$$x^{T}Lx=\sum_{(u,v)\in E}w_{u,v}\,(x(u)-x(v))^2.$$

I can see why: write $L=\sum_{(u,v)\in E}w_{u,v}(\delta_u-\delta_v)(\delta_u-\delta_v)^{T}$, one rank-one term per edge, and $x^{T}(\delta_u-\delta_v)(\delta_u-\delta_v)^{T}x=(x(u)-x(v))^2$. So $\chi_S^{T}L\chi_S=|\partial(S)|$. The combinatorial cut size is a *quadratic form* in the indicator. That's the bridge I wanted: cut size lives in linear algebra, and $L$ is positive semidefinite with $L\mathbf 1=0$, so its smallest eigenvalue is $0$ on the constant vector.

Now the conductance is $\chi_S^{T}L\chi_S$ over a balance term. If I just minimized $\chi_S^{T}L\chi_S$ over indicators with no constraint, the answer is $S=\varnothing$ or $S=V$: no edges cut, zero, useless. The constant vector — everyone on the same side — is the trivial "cut," and it's exactly the kernel direction $\mathbf 1$ of $L$. So the real optimization is over $x$ that are *not* constant; I need to forbid the $\mathbf 1$ direction.

Drop the demand that $x$ be a $0/1$ indicator and let $x$ range over $\mathbb R^n$, but keep it orthogonal to $\mathbf 1$ to kill the trivial solution, and fix the scale. Minimize $x^{T}Lx/x^{T}x$ over $x\perp\mathbf 1$. But that is precisely the Courant–Fischer characterization of the second-smallest eigenvalue:

$$\lambda_2=\min_{x\perp\mathbf 1}\frac{x^{T}Lx}{x^{T}x}.$$

So $\lambda_2$ is the continuous relaxation of (a normalized version of) the sparsest-cut objective. The integer minimum is at least the relaxed minimum, so $\lambda_2$ *lower-bounds* the cut quantity. That's the reason the *spectrum sees cuts*: the second eigenvalue is the relaxed price of separating the graph, and the constraint $x\perp\mathbf 1$ is exactly "don't put everyone on one side." This isn't an accident of definitions — it's the same relaxation Hall used to draw a graph on a line, minimizing $\sum_{(u,v)}(x(u)-x(v))^2$ with $\|x\|=1$ and $\mathbf 1^{T}x=0$, whose answer is the $\lambda_2$-eigenvector. The drawing pulls adjacent vertices together, so the coordinate *separates* loosely-connected regions; that coordinate *is* a soft cut.

And I already have a strong hint that $\lambda_2$ tracks connectivity, because Fiedler showed $a(G)=\lambda_2$ — he called it the algebraic connectivity — is $0$ exactly when the graph is disconnected, with the multiplicity of $0$ counting components, and that $a(G)\le v(G)\le e(G)$, the algebraic connectivity sitting below the vertex connectivity below the edge connectivity. So a small $\lambda_2$ is *necessary* for a sparse cut. But that's only the qualitative, one-sided story: $a(G)\le v(G)$ tells me a sparse vertex cut forces small $\lambda_2$, not that small $\lambda_2$ *delivers* a good conductance cut, and it doesn't hand me the cut. I want a two-sided, quantitative bracket on $\phi_G$, and I want the eigenvector to actually produce a cut.

Let me nail the easy direction first, because it should fall right out of the relaxation. I want to show $\lambda_2$ is genuinely below the conductance, using the *optimal* cut as a test vector — feed a cut into the Rayleigh quotient and the minimum can only be smaller. Take the optimal $S$ and its indicator $\chi_S$. The catch is $\chi_S$ isn't orthogonal to $\mathbf 1$. So shift it: set $x=\chi_S-s\,\mathbf 1$ with $s=|S|/|V|$, so $x(u)=1-s$ on $S$ and $-s$ off it, and $\mathbf 1^{T}x=|S|-s|V|=0$. Subtracting a constant doesn't change any difference $x(u)-x(v)$, so

$$x^{T}Lx=\sum_{(u,v)\in E}\big((\chi_S(u)-s)-(\chi_S(v)-s)\big)^2=\sum_{(u,v)\in E}(\chi_S(u)-\chi_S(v))^2=|\partial(S)|.$$

And the denominator: $x^{T}x=|S|(1-s)^2+(|V|-|S|)s^2=|S|(1-2s+s^2)+|S|s-|S|s^2=|S|(1-s)$. Since $x\perp\mathbf 1$, the relaxation gives $\lambda_2\le x^{T}Lx/x^{T}x=|\partial(S)|/(|S|(1-s))=\theta(S)/(1-s)$, i.e. $\theta(S)\ge\lambda_2(1-s)$. For $|S|\le n/2$ we have $1-s\ge 1/2$, so $\theta_G\ge\lambda_2/2$. The lower-bound half is done, and it is just "the optimal cut is one feasible point of the relaxation."

This is cleaner if I switch from the vertex-count denominator to the degree (volume) denominator, which is the natural one for edges and for random walks. Define conductance with $\phi(S)=|\partial(S)|/\min(d(S),d(V\setminus S))$. Then the right matrix is the *normalized* Laplacian. Look at the generalized Rayleigh quotient $y^{T}Ly/y^{T}Dy$ — the $D$ in the denominator is exactly what turns counting vertices into counting volume. Substitute $x=D^{1/2}y$:

$$\frac{y^{T}Ly}{y^{T}Dy}=\frac{x^{T}D^{-1/2}LD^{-1/2}x}{x^{T}x},$$

an ordinary Rayleigh quotient of $N=D^{-1/2}LD^{-1/2}$, the normalized Laplacian. Its smallest eigenvalue is $0$ with eigenvector $D^{1/2}\mathbf 1=d^{1/2}$ (since $D^{-1/2}LD^{-1/2}d^{1/2}=D^{-1/2}L\mathbf 1=0$). So the informative one is

$$\nu_2=\min_{x\perp d^{1/2}}\frac{x^{T}Nx}{x^{T}x}=\min_{y\perp d}\frac{y^{T}Ly}{y^{T}Dy},$$

where the orthogonality $x\perp d^{1/2}$ becomes $y\perp d$ after $x^{T}d^{1/2}=y^{T}D^{1/2}d^{1/2}=y^{T}d$. Redo the easy direction in this language: take the optimal $S$ and $y=\chi_S-\sigma\mathbf 1$ with $\sigma=d(S)/d(V)$, so $y^{T}d=\chi_S^{T}d-\sigma\mathbf 1^{T}d=d(S)-(d(S)/d(V))d(V)=0$, good, $y\perp d$. The numerator is still $y^{T}Ly=|\partial(S)|$ since shifting by a constant leaves differences alone. The denominator,

$$y^{T}Dy=\sum_{u\in S}d(u)(1-\sigma)^2+\sum_{u\notin S}d(u)\sigma^2=d(S)(1-\sigma)^2+d(V\setminus S)\sigma^2,$$

and expanding, $=d(S)-2d(S)\sigma+d(V)\sigma^2=d(S)-d(S)\sigma=d(S)d(V\setminus S)/d(V)$. So

$$\nu_2\le\frac{y^{T}Ly}{y^{T}Dy}=\frac{|\partial(S)|\,d(V)}{d(S)\,d(V\setminus S)}.$$

The larger of $d(S),d(V\setminus S)$ is at least $d(V)/2$, so $d(S)d(V\setminus S)/d(V)\ge\frac12\min(d(S),d(V\setminus S))$, giving $\nu_2\le 2|\partial(S)|/\min(d(S),d(V\setminus S))=2\phi(S)$, hence $\nu_2/2\le\phi_G$. Same structure, cleaner constant.

Before I trust this I want to see the three identities ($y\perp d$, $y^{T}Ly=|\partial(S)|$, $y^{T}Dy=d(S)d(V\setminus S)/d(V)$) hit real numbers, because I derived them by symbol-pushing and that's exactly where a sign or a factor goes wrong. Take two triangles $\{0,1,2\}$ and $\{3,4,5\}$ joined by a single bridge edge $(2,3)$ — an obvious one-edge bottleneck. Degrees are $d=(2,2,3,3,2,2)$, $d(V)=14$. For $S=\{0,1,2\}$: $d(S)=7$, $\sigma=7/14=1/2$, and $y=\chi_S-\tfrac12\mathbf 1=(\tfrac12,\tfrac12,\tfrac12,-\tfrac12,-\tfrac12,-\tfrac12)$. Then $y^{T}d=\tfrac12(2+2+3)-\tfrac12(3+2+2)=\tfrac72-\tfrac72=0$, orthogonal as claimed. Only the bridge edge $(2,3)$ has endpoints differing in $y$ (they differ by $1$, all other edges by $0$), so $y^{T}Ly=1^2=1$, which equals $|\partial(S)|=1$. And $y^{T}Dy=\sum d(u)y(u)^2=\tfrac14\sum d(u)=\tfrac{14}{4}=3.5$, against $d(S)d(V\setminus S)/d(V)=7\cdot7/14=3.5$. All three land. So $\nu_2\le 1/3.5\approx0.286$, and indeed the bottleneck is real: $\phi(S)=1/7\approx0.143$ here, comfortably above $\nu_2/2$. The lower bound is the trivial direction because relaxing can only help; the *content* is going to be the other way.

Now I hit the wall. The relaxation gives me a *lower* bound and a *real-valued* eigenvector, the Fiedler vector $y$. It does not give me a cut. A small $\nu_2$ tells me a good cut might exist, but maybe the relaxation is loose — maybe the eigenvector is some smeared-out function with no clean place to cut it, and every actual $0/1$ cut I extract has conductance far above $\sqrt{\nu_2}$ or whatever. The whole thing is worthless unless I can *round* the eigenvector back to a vertex set and *control* the conductance of that set by $\nu_2$. So the real question: given any $y\perp d$ with small Rayleigh quotient $\rho=y^{T}Ly/y^{T}Dy$, can I extract a cut $S$ with $\phi(S)$ bounded by a function of $\rho$? If yes, then in particular the eigenvector ($\rho=\nu_2$) gives a cut with conductance bounded by the same function of $\nu_2$, and the relaxation is *certified* near-optimal.

How do I round a real vector into a cut? The dumb thing is to threshold: pick a number $t$ and let $S_t=\{u:y(u)\le t\}$. The vertices with small eigenvector coordinate go on one side. That's the sweep. The question is which $t$, and whether *some* threshold cut is good. Notice I get to try all of them — sort the vertices by $y$ value, there are only $n-1$ distinct level-set cuts, sweep through and keep the best. If I can prove that the *best* sweep cut has conductance $\le$ something$(\rho)$, I'm done, and computing it is cheap. So I need: there exists $t$ with $\phi(S_t)\le f(\rho)$.

Let me set this up carefully, because the bound should come out as a square root and I want to see *why*. First, I don't need $y$ to be an eigenvector — the argument should work for any $y\perp d$, which is a nice robustness, but the cleanest path is to normalize. Sort so $y(1)\le y(2)\le\cdots\le y(n)$. I want to center $y$ at a balance point so that "small side" corresponds to one sign. Let $j$ be the least index with $\sum_{u\le j}d(u)\ge d(V)/2$, the half-volume median, and set $z=y-y(j)\mathbf 1$. Shifting by a constant doesn't change $y^{T}Ly$ (differences are preserved). It changes the denominator in the helpful direction: for any shift $v_t=y+t\mathbf 1$,

$$v_t^{T}Dv_t=y^{T}Dy+2t\,y^{T}d+t^2d(V)=y^{T}Dy+t^2d(V),$$

because $y\perp d$. So the volume-orthogonal representative $y$ is the minimum-denominator shift, $z^{T}Dz\ge y^{T}Dy$, and therefore

$$\frac{z^{T}Lz}{z^{T}Dz}\le\frac{y^{T}Ly}{y^{T}Dy}=\rho.$$

And $z(j)=0$. By the choice of $j$, the total volume with $z<0$ is below $d(V)/2$, while the total volume with $z\le0$ is at least $d(V)/2$, even if several vertices tie at zero. So a negative threshold makes $S_t$ the smaller side, and a nonnegative threshold makes $V\setminus S_t$ the smaller side. Finally rescale $z$ so that $z(1)^2+z(n)^2=1$; this fixes the range of the threshold I'm about to randomize over. Now $\phi(S_t)=|\partial(S_t)|/\min(d(S_t),d(V\setminus S_t))$, and I want to bound this for a good $t$.

The move I'll try is to not fix $t$ at all but *randomize* it, and bound the expectations of numerator and denominator separately, so that some realized $t$ does at least as well as the ratio of expectations. The question is which density. Uniform is the obvious first guess; let me hold off on committing and instead pick the density by what I need downstream. Whatever density $p(t)$ I use, an edge $(u,v)$ with $z(u)\le z(v)$ is cut precisely when $t\in[z(u),z(v))$, so its cut probability is $\int_{z(u)}^{z(v)}p$, and a vertex $u$ lands on the smaller side with probability $\int p$ over the interval from $0$ out to $z(u)$. I want the *volume* expectation to come out as something I already control — ideally $z^{T}Dz$, the Rayleigh denominator — because then numerator-over-denominator will be a Rayleigh quotient I can bound by $\rho$. A vertex contributes $d(u)\int_0^{z(u)}p$ to that volume; for this to equal $d(u)z(u)^2$ I'd need $\int_0^{s}p=s^2$, i.e. $p(t)=2|t|$. So the density isn't a clever guess, it's forced by demanding the volume expectation reduce to $z^{T}Dz$. Take $p(t)=2|t|$ on $[z(1),z(n)]$ and first check it's a distribution: $\int_{z(1)}^{z(n)}2|t|\,dt=\int_{z(1)}^{0}2|t|\,dt+\int_{0}^{z(n)}2|t|\,dt=z(1)^2+z(n)^2=1$, using $z(1)\le z(j)=0\le z(n)$ and the scaling $z(1)^2+z(n)^2=1$ I just imposed.

The probability that $t$ falls in an interval $[a,b]$ is $\int_a^b 2|t|\,dt=\operatorname{sgn}(b)b^2-\operatorname{sgn}(a)a^2$. Now the numerator. An edge $(u,v)$ with $z(u)\le z(v)$ is on the boundary of $S_t$ exactly when $z(u)\le t<z(v)$, so

$$\Pr[(u,v)\in\partial(S_t)]=\operatorname{sgn}(z(v))z(v)^2-\operatorname{sgn}(z(u))z(u)^2
=\begin{cases}|z(u)^2-z(v)^2| & \operatorname{sgn}(z(u))=\operatorname{sgn}(z(v)),\\ z(u)^2+z(v)^2 & \operatorname{sgn}(z(u))\ne\operatorname{sgn}(z(v)).\end{cases}$$

I claim both cases are $\le|z(u)-z(v)|\,(|z(u)|+|z(v)|)$. Same sign: $|z(u)^2-z(v)^2|=|(z(u)-z(v))(z(u)+z(v))|\le|z(u)-z(v)|\,(|z(u)|+|z(v)|)$ since $|z(u)+z(v)|\le|z(u)|+|z(v)|$. Opposite sign: then $z(u),z(v)$ have opposite signs so $(z(u)-z(v))^2=z(u)^2+z(v)^2-2z(u)z(v)\ge z(u)^2+z(v)^2$ (the cross term $-2z(u)z(v)\ge 0$), and $|z(u)-z(v)|=|z(u)|+|z(v)|$, hence $z(u)^2+z(v)^2\le|z(u)-z(v)|(|z(u)|+|z(v)|)$. So summing over weighted edges,

$$\mathbb E[\,|\partial(S_t)|\,]=\sum_{(u,v)\in E}w_{u,v}\Pr[(u,v)\in\partial(S_t)]\le\sum_{(u,v)\in E}w_{u,v}|z(u)-z(v)|\,(|z(u)|+|z(v)|).$$

Now the denominator. I want $\mathbb E[\min(d(S_t),d(V\setminus S_t))]$, and this is exactly where centering at $j$ pays off. If $t<0$, the smaller side is $S_t$, and a vertex contributes to it exactly when $z(u)\le t<0$. If $t\ge0$, the smaller side is $V\setminus S_t$, and a vertex contributes to it exactly when $0\le t<z(u)$. Vertices with $z(u)=0$ contribute zero either way. So

$$\mathbb E[\min(d(S_t),d(V\setminus S_t))]=\sum_{z(u)<0}\Pr[z(u)\le t<0]\,d(u)+\sum_{z(u)>0}\Pr[0\le t<z(u)]\,d(u).$$

For $z(u)<0$, $\Pr[z(u)\le t<0]=\int_{z(u)}^{0}2|t|\,dt=z(u)^2$; for $z(u)>0$, $\Pr[0\le t<z(u)]=\int_0^{z(u)}2|t|\,dt=z(u)^2$. So both sums collapse and

$$\mathbb E[\min(d(S_t),d(V\setminus S_t))]=\sum_u z(u)^2 d(u)=z^{T}Dz.$$

This is the volume expectation I engineered the density to produce — $\int_0^{s}2|t|\,dt=s^2$ is exactly what turns each vertex's contribution into $d(u)z(u)^2$ — so it lands on $z^{T}Dz$ as intended. Now I want $\mathbb E[|\partial(S_t)|]\le\sqrt{2\rho}\,\mathbb E[\min(d(S_t),d(V\setminus S_t))]$, because that would imply some realized $t$ has $|\partial(S_t)|\le\sqrt{2\rho}\,\min(d(S_t),d(V\setminus S_t))$, i.e. $\phi(S_t)\le\sqrt{2\rho}$. (If $\mathbb E[A]\le c\,\mathbb E[B]$ then $\mathbb E[cB-A]\ge0$, so $cB-A\ge0$ for some outcome.) So I need to push the edge sum down to $\sqrt{2\rho}\,z^{T}Dz$.

Take the edge sum and apply Cauchy–Schwarz, splitting the product term:

$$\sum_{(u,v)\in E}w_{u,v}|z(u)-z(v)|\,(|z(u)|+|z(v)|)\le\sqrt{\sum_{(u,v)\in E}w_{u,v}(z(u)-z(v))^2}\;\sqrt{\sum_{(u,v)\in E}w_{u,v}(|z(u)|+|z(v)|)^2}.$$

The left factor under the root is $z^{T}Lz$. By the definition of $\rho$ and the shift inequality, $z^{T}Lz\le\rho\,z^{T}Dz=\rho\sum_u z(u)^2 d(u)$. The right factor: $(|z(u)|+|z(v)|)^2\le 2(z(u)^2+z(v)^2)$, and summing over weighted edges counts each vertex $u$ with total weight $d(u)$, so $\sum_{(u,v)\in E}w_{u,v}(|z(u)|+|z(v)|)^2\le 2\sum_{(u,v)\in E}w_{u,v}(z(u)^2+z(v)^2)=2\sum_u z(u)^2 d(u)=2\,z^{T}Dz$. Putting the two factors together,

$$\mathbb E[|\partial(S_t)|]\le\sqrt{\rho\,z^{T}Dz}\,\sqrt{2\,z^{T}Dz}=\sqrt{2\rho}\;z^{T}Dz=\sqrt{2\rho}\;\mathbb E[\min(d(S_t),d(V\setminus S_t))].$$

So there is a threshold $t$ with $\phi(S_t)\le\sqrt{2\rho}$. Feeding in the eigenvector, $\rho=\nu_2$, the best sweep cut over the Fiedler vector satisfies $\phi_G\le\phi(S_t)\le\sqrt{2\nu_2}$. Combined with the easy direction,

$$\frac{\nu_2}{2}\le\phi_G\le\sqrt{2\nu_2}.$$

Now I'm uneasy about that square root. Is it real, or just slack in my Cauchy–Schwarz that a sharper argument would remove and replace with a clean $\phi_G\le c\,\nu_2$? If the bound could be made linear in $\nu_2$ I'd be leaving a lot on the table. The way to settle it is to find a graph where I can pin down both $\nu_2$ and $\phi_G$ exactly and see how they actually scale. A cycle $C_n$ is perfect: it's vertex-transitive and degree-$2$, so the normalized Laplacian is $\tfrac12 L$ on the unit-degree cycle, whose eigenvalues are $1-\cos(2\pi k/n)$, giving $\nu_2=1-\cos(2\pi/n)$. And the sparsest cut is transparent — any cut into two arcs removes exactly two edges, smaller-side volume maximized at the balanced split with volume $\approx n$ (each of $n/2$ vertices has degree $2$), so $\phi_G=2/n$ for even $n$. Let me put numbers on it rather than wave at $\Theta$'s. For $n=8$: $\nu_2=1-\cos(\pi/4)=1-0.7071=0.2929$, so my bracket predicts $0.1464\le\phi_G\le\sqrt{2(0.2929)}=0.7654$, while the true $\phi_G=2/8=0.25$ — inside the bracket, and the upper bound is loose by about $3\times$ but the *lower* bound $\nu_2/2=0.146$ is the same order as $\phi_G$. Now stretch it: $n=20$ gives $\nu_2=1-\cos(\pi/10)=0.0489$, bracket $0.0245\le\phi_G\le0.313$, true $\phi_G=2/20=0.10$. Watch the ratios as $n$ grows: $\phi_G/\nu_2$ goes $0.25/0.293=0.85$ at $n=8$ to $0.10/0.0489=2.04$ at $n=20$ — it's *growing*, so no inequality of the form $\phi_G\le c\,\nu_2$ with a fixed $c$ can survive. But $\phi_G/\sqrt{\nu_2}$ goes $0.25/0.541=0.46$ to $0.10/0.221=0.45$ — essentially constant. So on the cycle $\phi_G\asymp\sqrt{\nu_2}$, not $\nu_2$: the square root is not slack I failed to remove, it's the true scaling, forced by the geometry of long thin graphs. The eigenvector genuinely loses a square root here, and the sweep recovers exactly that order.

Two things I want to register, because they make the result more than a one-off. The rounding argument never used that $y$ is an eigenvector — only $y\perp d$ and the value of its Rayleigh quotient. So *any* test vector orthogonal to $d$ with small $y^{T}Ly/y^{T}Dy$ yields a cut of conductance $\le\sqrt{2\,(y^{T}Ly/y^{T}Dy)}$; the eigenvector is just the optimal such vector. And the whole proof goes through for weighted graphs essentially unchanged: replace edge counts by edge weights, so the numerator becomes $\sum_{(u,v)}w_{u,v}|z(u)-z(v)|(|z(u)|+|z(v)|)$ and Cauchy–Schwarz now produces $\sqrt{\sum w_{u,v}(z(u)-z(v))^2}=\sqrt{z^{T}Lz}$ with the weighted $L$, and $\sum_{(u,v)}w_{u,v}(|z(u)|+|z(v)|)^2\le 2z^{T}Dz$ with weighted degrees. Same $\sqrt{2\nu_2}$.

This is the discrete shadow of the manifold statement. On a Riemannian manifold the first nonzero eigenvalue of the Laplace–Beltrami operator and the isoperimetric constant $h(M)=\inf_E S(E)/\min(V(A),V(B))$ satisfy $\lambda_1\ge h^2/4$ — a small spectral gap forces a geometric bottleneck. The graph Laplacian *is* the discrete analogue of that operator, conductance the discrete analogue of $h$, and here I've got both directions, with the rounding making the hard direction constructive: the bottleneck isn't just asserted to exist, the sweep over the eigenvector hands it to me.

So let me write down the algorithm the proof has been describing, assuming the graph has positive degrees so the normalized Laplacian is defined. Form $N=D^{-1/2}LD^{-1/2}$, compute its second eigenvector (equivalently the generalized eigenvector $y$ of $Ly=\nu_2 Dy$ orthogonal to $d$ — the Fiedler vector), sort the vertices by $y$, sweep through the $n-1$ threshold cuts $S_t=\{u:y(u)\le t\}$, and return the one of least conductance.

Before I believe the sweep actually finds the bottleneck I derived it to find, let me run the whole pipeline on the two-triangle bridge graph from before and watch each step. Computing $N$ and its second eigenpair gives $\nu_2=0.2047$ and a Fiedler vector $y\approx(0.315,0.315,0.186,-0.186,-0.315,-0.315)$ — and that's already telling: the two triangles get opposite signs, with the bridge endpoints $2,3$ pulled toward zero, exactly the "loosely-connected regions separate along the coordinate" picture. Sorting by $y$ gives the order $(5,4,3,2,0,1)$, and sweeping prefixes, the cut after the first three vertices is $\{5,4,3\}=\{3,4,5\}$ — the second triangle — with one boundary edge (the bridge) and smaller-side volume $7$, so $\phi=1/7=0.1429$, which is the least over the whole sweep. So the sweep returns the intended bottleneck, not some smeared-out non-cut. Checking the bracket on this instance: $\nu_2/2=0.1023\le\phi=0.1429\le\sqrt{2\nu_2}=0.6398$ — both inequalities hold, the cut sits between them. That's the certificate working end to end: the eigenvector found the cut, and the two computed bounds sandwich its conductance.

In general the lower bound $\nu_2/2$ is needed by any cut and the sweep delivers a cut with $\phi\le\sqrt{2\nu_2}$, so this polynomial-time eigenvector-and-sweep is provably within a square-root factor of the NP-hard optimum.

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def build_laplacian(adj):
    # L = D - A; the quadratic form x^T L x = sum_(u,v) w(u,v)(x(u)-x(v))^2
    # is exactly the cut size of an indicator, the bridge to linear algebra.
    d = np.asarray(adj.sum(axis=1)).ravel()
    D = sp.diags(d)
    return (D - adj).tocsr(), d

def conductance(adj, d, S_mask):
    volS = d[S_mask].sum()
    volC = d.sum() - volS
    if min(volS, volC) == 0:
        return np.inf
    cross = adj[S_mask][:, ~S_mask].sum()           # |boundary(S)|
    return cross / min(volS, volC)

def fiedler_vector(adj):
    # Relaxation: min over y orthogonal to d of y^T L y / y^T D y is nu_2 of the normalized
    # Laplacian. Solve via N = D^{-1/2} L D^{-1/2}; its eigenvector of nu_2,
    # mapped back by D^{-1/2}, is the y orthogonal to d (the Fiedler vector).
    L, d = build_laplacian(adj)
    if np.any(d <= 0):
        raise ValueError("normalized Laplacian requires positive degrees")
    dinv_half = sp.diags(1.0 / np.sqrt(d))
    N = (dinv_half @ L @ dinv_half).tocsr()
    vals, vecs = eigsh(N.astype(float), k=2, which="SM")
    order = np.argsort(vals)
    x2 = vecs[:, order[1]]                            # eigenvector of nu_2 of N
    y = np.asarray(dinv_half @ x2).ravel()
    return y, max(float(vals[order[1]]), 0.0)          # y orthogonal to d, and nu_2

def relax_and_round(adj):
    # Round the eigenvector by sweeping: try all n-1 threshold cuts of the
    # sorted Fiedler vector, keep the least-conductance one. The Cheeger proof
    # certifies the best of these has conductance <= sqrt(2 * nu_2).
    _, d = build_laplacian(adj)
    y, nu2 = fiedler_vector(adj)
    order = np.argsort(y)
    n = len(y)
    best_S, best_phi = None, np.inf
    in_S = np.zeros(n, dtype=bool)
    for k in range(n - 1):                            # prefixes of the sorted order
        in_S[order[k]] = True
        phi = conductance(adj, d, in_S)
        if phi < best_phi:
            best_phi, best_S = phi, in_S.copy()
    # Cheeger bracket in exact arithmetic: nu2/2 <= phi_G <= best_phi <= sqrt(2*nu2).
    return best_S, best_phi, nu2
```

The causal chain, start to finish: the sparsest cut is NP-hard because $S$ is an integer object; cut size equals the Laplacian quadratic form of the indicator, so the cut lives in linear algebra; relaxing the indicator to a real vector orthogonal to the constant kernel turns the conductance into the second eigenvalue's Rayleigh quotient, which therefore lower-bounds it ($\nu_2/2\le\phi_G$); the eigenvector by itself is only a lower bound, so I round it by sweeping all threshold cuts, and a random threshold with density $2|t|$ makes the expected boundary and the expected volume line up through Cauchy–Schwarz into $\phi\le\sqrt{2\nu_2}$; hence a polynomial-time eigenvector-and-sweep computes a cut provably within a square-root factor of the combinatorial optimum, the gap being genuinely a square root as the cycle shows.
