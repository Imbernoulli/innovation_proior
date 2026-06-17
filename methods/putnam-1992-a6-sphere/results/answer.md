# Four random points on a sphere enclose the center with probability 1/8

## The result

Choose four points independently and uniformly on the surface of a sphere. The probability that the center lies inside the tetrahedron they span is

$$\boxed{\dfrac{1}{8}}.$$

## The key idea

Put the center at the origin $O$. Sampling a uniform point on the sphere is the same as sampling a uniform **diameter** (a line through $O$) and then flipping an independent **fair coin** to keep one of its two endpoints — legitimate because the uniform law is invariant under the antipodal map $v\mapsto -v$. Hold one of the four points fixed and reparametrize the other three this way: three random diameters and three coins. The three coins generate $2^3=8$ equally likely tetrahedra over the same diameters and the same fixed point. **Exactly one** of these eight contains $O$, for *every* choice of the diameters and the fixed point — so the conditional probability is the constant $\tfrac18$, and the answer is $\tfrac18$ with no integral.

## The circle check

For three independent uniform points on a circle, condition on $P_1,P_2$ and let $\alpha\in[0,\pi]$ be their smaller central separation. The triangle contains the center exactly when $P_3$ lies on the opposite arc between $-P_1$ and $-P_2$, whose angular length is also $\alpha$. Thus the conditional probability is $\alpha/(2\pi)$. After fixing $P_1$, the smaller central separation to uniform $P_2$ is uniform on $[0,\pi]$, so the average is

$$\mathbb{E}\frac{\alpha}{2\pi}=\frac{1}{\pi}\int_0^\pi \frac{\alpha}{2\pi}\,d\alpha=\frac14.$$

Equivalently, two random diameters plus two fair endpoint coins give four equally likely triangles, and exactly one sign pattern places the center inside. The sphere argument below is the same count with three diameters.

## The containment criterion (the decisive lemma)

**Lemma.** Let $v_1,v_2,v_3,v_4\in\mathbb{R}^3$ be in general position: they are affinely independent, and any three are linearly independent. Their linear dependence $\sum_i c_i v_i=0$ is unique up to scale, with all $c_i\neq 0$. Then the origin is strictly inside their convex hull **iff all $c_i$ have the same sign.**

*Proof.* The origin is strictly interior iff $O=\sum_i\lambda_i v_i$ with all $\lambda_i>0$ and $\sum_i\lambda_i=1$. The equation $\sum_i\lambda_i v_i=0$ says $(\lambda_i)$ is a dependence among the $v_i$; but the dependence space of four generic vectors in $\mathbb{R}^3$ is one-dimensional, spanned by $(c_i)$, so $\lambda_i=t\,c_i$ for some scalar $t$. The constraint $\sum_i\lambda_i=1$ forces $t=1/\sum_j c_j$ (requiring $\sum_j c_j\neq0$), giving the unique candidate weights

$$\lambda_i=\frac{c_i}{\sum_j c_j}.$$

These are all positive iff all $c_i$ share one sign. With mixed signs, either $\sum_j c_j=0$ and there are no normalized barycentric weights for $O$, or the ratios have mixed signs, so $O$ is not interior. $\qquad\blacksquare$

The dependence vector $(c_1,c_2,c_3,c_4)$ is the right object because it is **normalization-free**: replacing $v_i$ by $-v_i$ simply multiplies $c_i$ by $-1$ and leaves the other coefficients unchanged. (The normalized barycentric weights $\sum\lambda_i=1$ do *not* survive a sign flip, which is why one works with the raw dependence.)

## The derivation

Place $O$ at the origin; antipode of $v$ is $-v$. Fix the diameters: let the four points of a candidate tetrahedron be the fixed point $v_1=p$ and the chosen endpoints $v_2=\varepsilon_2 q$, $v_3=\varepsilon_3 r$, and $v_4=\varepsilon_4 s$, where $q,r,s$ are reference endpoints of the three diameters and $(\varepsilon_2,\varepsilon_3,\varepsilon_4)\in\{\pm1\}^3$ are the coins. Let the base dependence be

$$c_1\,p + c_2\,q + c_3\,r + c_4\,s = 0,\qquad c_i\neq 0.$$

Flipping endpoint $i$ sends $c_i\mapsto\varepsilon_i c_i$ and fixes the rest, so the dependence for the coin pattern $(\varepsilon_2,\varepsilon_3,\varepsilon_4)$ is

$$(c_1,\ \varepsilon_2 c_2,\ \varepsilon_3 c_3,\ \varepsilon_4 c_4).$$

By the Lemma, this tetrahedron contains $O$ iff these four share a sign. With $c_1$ fixed as the reference, each coin must satisfy $\varepsilon_i=\operatorname{sign}(c_1)/\operatorname{sign}(c_i)$ — one forced value apiece. Hence **exactly one** of the $2^3=8$ patterns encloses $O$.

Since the eight patterns are equally likely (fair, independent coins) and exactly one works regardless of the diameters and the fixed point, the conditional probability is $\tfrac18$, and therefore

$$P(\,O\in\text{tetrahedron}\,)=\frac18.$$

(Degenerate configurations — four points coplanar with $O$, or $O$ on a face, where some $c_i=0$ — form a measure-zero set and contribute probability $0$.)
