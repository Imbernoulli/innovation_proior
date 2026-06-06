OK, let me start from the thing that's actually bothering me. I have $y = Ax + w$ with $A \in \mathbb{R}^{m\times n}$ and $m \ll n$ — far fewer measurements than unknowns. As a linear-algebra object this is hopeless: the set $\{x : Ax = y\}$ is an affine subspace of dimension at least $n-m$, so there are infinitely many exact explanations of the data and nothing in the equations themselves prefers one over another. The minimum-$\ell_2$-norm solution, $A^{\top}(AA^{\top})^{-1}y$, is what you reach for reflexively, but stare at what it does: it spreads the energy across every one of the $n$ coordinates — it's the *least* sparse answer you could possibly pick — and when $A$ is ill-conditioned its norm blows up and it's meaningless. So the linear-algebra answer is the wrong answer.

But I have a piece of side information that the equations don't know about: the signal I care about is *sparse*. Maybe $x$ itself has only $k \ll n$ nonzero entries, or maybe it's sparse after a known transform — a natural image is a few big wavelet coefficients plus a sea of near-zeros, a tone is a few Fourier spikes. So the real question is: among the infinitely many $x$ with $Ax = y$, is the *sparsest* one unique, and can I find it? That reframes everything. I'm not solving a linear system; I'm searching the solution subspace for its sparsest point.

Let me write that down honestly. The sparsest consistent signal is

$$ \min_x \ \|x\|_0 \quad \text{s.t.}\quad Ax = y, \qquad \|x\|_0 = \#\{i : x_i \neq 0\}. $$

This is exactly what I want. And it's exactly what I can't have. $\|\cdot\|_0$ isn't a norm — it's a count, it's nonconvex, it's discontinuous. To minimize it I'd have to decide *which* coordinates are allowed to be nonzero, i.e. choose a support $T \subseteq \{1,\dots,n\}$, and then check whether some $x$ supported on $T$ explains $y$. There are $\binom{n}{k}$ supports of size $k$ and exponentially many overall — for a Fourier-sampling instance with about half the frequencies observed the count of supports to check scales like $4^N 3^{-3N/4}$, an astronomical number. And this isn't just "looks hard": minimizing $\|x\|_0$ subject to $\|Ax - b\| \le \varepsilon$ is provably NP-hard (Natarajan, 1995). So $(P_0)$ states precisely the problem and is precisely intractable. I need a surrogate I can actually optimize, and the surrogate has to keep the sparsity-seeking behavior while being something a convex solver can chew on.

What's the right relaxation? The natural move is to replace the count $\|x\|_0$ by some norm of $x$ that I *can* minimize. Try $\ell_2$ first, just to see it fail: $\min \|x\|_2$ s.t. $Ax=y$ is the pseudoinverse again — round level sets, energy spread everywhere, dense. So $\ell_2$ relaxation actively fights sparsity. Now slide down the family of $\ell_p$ "norms" toward $\ell_0$. For $p<1$ the $\ell_p$ ball gets spiky and pinched toward the axes — great for sparsity — but it's nonconvex, so I'm back to local minima and intractability. The boundary case is $p=1$: $\|x\|_1 = \sum_i |x_i|$. It's the *smallest* exponent for which $\|x\|_p$ is still convex. And geometrically the $\ell_1$ ball is a cross-polytope — a diamond in 2-D, an octahedron in 3-D — whose vertices and low-dimensional edges sit exactly *on the coordinate axes*. That's the whole trick: if I inflate the data-consistent set until it first touches the $\ell_1$ ball, it generically touches at a vertex or an edge, and at those points most coordinates are exactly zero. The round $\ell_2$ ball has no corners, which is exactly why it never zeroes anything. So $\ell_1$ is convex *and* it puts its pressure where I want it — toward sparse points.

So I'll relax to

$$ (P_1)\qquad \min_x \ \|x\|_1 \quad \text{s.t.}\quad Ax = y. $$

This is convex — in fact a linear program. But why should its solution have anything to do with the $\ell_0$ solution I actually wanted? The geometry has to be backed by a property of $A$: it should act as a near-isometry on sparse vectors, so that every small subset of columns is approximately orthonormal and distinct sparse signals stay separated after measurement. Candès, Romberg, and Tao make that precise in the partial-Fourier setting: for an overwhelming fraction of supports and random frequency sets, the $\ell_1$ and $\ell_0$ solutions coincide when the support size is below a threshold of the form $|T| \lesssim |\Omega|/\log N$. Random and partial-Fourier matrices have this near-isometry behavior with high probability. So the convex relaxation isn't just a heuristic that's sometimes close; under the right measurement geometry it recovers the true sparse signal *exactly*. Good — that's enough to commit to $\ell_1$ as the objective and turn my attention to *computing* it.

Now, the measurements are noisy ($y = Ax + w$), and the hard equality $Ax=y$ is wrong in that case — I shouldn't insist on fitting the noise. The clean way to trade fit against sparsity is the penalized form, which also matches the lasso: instead of "minimize $\|x\|_1$ subject to exact consistency," minimize a weighted sum,

$$ \min_x \ F(x) = \tfrac12\|Ax - y\|_2^2 + \lambda \|x\|_1. $$

(By Lagrangian duality this is equivalent to the constrained $\|x\|_1 \le t$ form for a matching $t$; the penalized version is more convenient.) The knob $\lambda > 0$ sets the tradeoff — large $\lambda$ buys more sparsity at the cost of fit. The $\tfrac12$ on the least-squares term is just my bookkeeping convention; it makes the gradient come out clean. This is the object I'm going to minimize. It's convex. But it is *not* smooth, because of the $\|x\|_1$ term — and that nonsmoothness is going to dictate the whole algorithm.

So how do I minimize it? The obvious thought: cast it as a second-order cone / linear program and call an interior-point method. That works, and it's accurate, but it's the wrong tool for *my* problem. In the applications I care about $n$ is in the millions and $A$ is dense — a blur convolution, a partial-Fourier operator. Interior-point methods take Newton steps; a Newton step needs me to form and factor a system involving $A^{\top}A$, and that's flatly impossible at this scale. The only operations I can afford are applying $A$ and $A^{\top}$ — cheap matrix–vector products. That single constraint — *first order only, $A$ and $A^{\top}$ products and nothing heavier* — pins me to gradient-style methods. Note $\nabla\big(\tfrac12\|Ax-y\|_2^2\big) = A^{\top}(Ax - y)$, which is exactly one application of $A$ then one of $A^{\top}$. Perfect, cheap. But gradient descent needs a differentiable objective, and mine isn't, because of $\|x\|_1$.

Let me not panic at the nonsmoothness. The objective has a very particular structure: it's the sum of a *nice smooth* convex part $f(x) = \tfrac12\|Ax-y\|_2^2$ — differentiable, with a Lipschitz gradient — and a *simple but nonsmooth* convex part $g(x) = \lambda\|x\|_1$. The $\ell_1$ part is ugly to differentiate but it's *separable* across coordinates and dead simple in isolation. So maybe I shouldn't treat the whole thing as one monolithic nonsmooth function. The honest first try — the subgradient method — would do exactly that: pick a subgradient of $F$ and step along it. But subgradient descent crawls at $O(1/\sqrt{k})$, far worse than gradient descent's $O(1/k)$, and worse, it has no built-in dead-zone that deliberately snaps small coordinates to zero. I'd lose the very thing I'm chasing. So treating $f+g$ as one undifferentiated blob throws away the structure. I want to handle $f$ with its gradient and handle $g$ *exactly*, with whatever its own simple structure allows.

How do I even take a "gradient step" that respects the nonsmooth $g$? Let me go back to what a plain gradient step *is*, because there's a reading of it that will generalize. For smooth $f$, the step $x_{k} = x_{k-1} - t\nabla f(x_{k-1})$ is the same as

$$ x_k = \arg\min_x \Big\{ f(x_{k-1}) + \langle x - x_{k-1}, \nabla f(x_{k-1})\rangle + \tfrac{1}{2t}\|x - x_{k-1}\|_2^2 \Big\}. $$

Read that: I replace $f$ by its linear (first-order) model at $x_{k-1}$, add a quadratic "trust" term $\tfrac{1}{2t}\|x-x_{k-1}\|^2$ that penalizes moving too far, and minimize. Setting the gradient to zero gives back $x_k = x_{k-1} - t\nabla f(x_{k-1})$. So a gradient step is "minimize a simple local model of $f$." Now I have a place to put $g$: linearize the *smooth* part $f$, keep the quadratic trust term, and tack the *exact* nonsmooth $g$ on as-is:

$$ x_k = \arg\min_x \Big\{ f(x_{k-1}) + \langle x - x_{k-1}, \nabla f(x_{k-1})\rangle + \tfrac{1}{2t}\|x - x_{k-1}\|_2^2 + g(x)\Big\}. $$

The first term is constant in $x$; drop it. Complete the square on the linear-plus-quadratic part: $\langle x - x_{k-1}, \nabla f\rangle + \tfrac{1}{2t}\|x-x_{k-1}\|^2 = \tfrac{1}{2t}\|x - (x_{k-1} - t\nabla f(x_{k-1}))\|^2 + \text{const}$. So the whole thing collapses to

$$ x_k = \arg\min_x \Big\{ g(x) + \tfrac{1}{2t}\big\|x - \big(x_{k-1} - t\nabla f(x_{k-1})\big)\big\|_2^2 \Big\}. $$

Look at the shape of that minimization: I take an ordinary gradient step on the smooth part, landing at $v = x_{k-1} - t\nabla f(x_{k-1})$, and then I solve "find the $x$ that is close to $v$ in Euclidean norm but also makes $g(x)$ small." That second operation is a self-contained object — it depends only on $g$ and on $v$. It's Moreau's *proximal mapping*,

$$ \operatorname{prox}_{t g}(v) = \arg\min_x \Big\{ g(x) + \tfrac{1}{2t}\|x - v\|_2^2 \Big\}, $$

the resolvent that generalizes Euclidean projection (for an indicator function of a convex set it *is* the projection onto that set). So my iteration is

$$ x_k = \operatorname{prox}_{t g}\big(x_{k-1} - t\nabla f(x_{k-1})\big) $$

— gradient step on the smooth part, then prox of the nonsmooth part. This is exactly the split I wanted: $f$ enters only through its gradient (cheap, $A^{\top}(Ax-y)$), and $g$ is handled exactly through its prox. The whole method is only useful, though, if that prox is *cheap* to evaluate. So everything now hinges on: what is $\operatorname{prox}_{t g}$ when $g = \lambda\|\cdot\|_1$?

Because $\|x\|_1 = \sum_i |x_i|$ is *separable*, the prox decouples completely across coordinates: the minimization $\min_x \lambda\|x\|_1 + \tfrac{1}{2t}\|x-v\|^2$ splits into $n$ independent scalar problems

$$ \min_{u}\ \lambda|u| + \tfrac{1}{2t}(u - v_i)^2 \qquad\text{for each } i. $$

Let me actually solve one of these. Write $\tau = t\lambda$ to keep it clean; I'm minimizing $\phi(u) = \lambda|u| + \tfrac{1}{2t}(u-v)^2$. For $u > 0$, $|u|=u$ is differentiable: $\phi'(u) = \lambda + \tfrac1t(u - v) = 0 \Rightarrow u = v - t\lambda = v - \tau$, and this is a valid positive solution only when $v > \tau$. Symmetrically for $u < 0$: $\phi'(u) = -\lambda + \tfrac1t(u-v) = 0 \Rightarrow u = v + \tau$, valid only when $v < -\tau$. And at $u=0$ the function has a kink; $0$ is the minimizer exactly when the subdifferential contains zero, i.e. $0 \in \lambda[-1,1] + \tfrac1t(0 - v)$, which rearranges to $|v| \le t\lambda = \tau$. Stitch the three regimes together and the per-coordinate prox is

$$ \big(\operatorname{prox}_{t\lambda\|\cdot\|_1}(v)\big)_i = \begin{cases} v_i - \tau, & v_i > \tau \\ 0, & |v_i| \le \tau \\ v_i + \tau, & v_i < -\tau \end{cases} \;=\; \operatorname{sign}(v_i)\,\max(|v_i| - \tau,\, 0), \qquad \tau = t\lambda. $$

This is **soft-thresholding** (the shrinkage operator). And now I can see *why* the $\ell_1$ relaxation does what I built it to do at the algorithmic level: the prox doesn't merely shrink coordinates toward zero, it has a flat dead-zone $[-\tau,\tau]$ where it sets them *exactly* to zero, and shrinks the survivors by a constant $\tau$. Every iteration produces an exactly sparse vector. The kink in $|\cdot|$ that made the objective nonsmooth — the thing I was tempted to be afraid of — is *precisely* the feature that snaps small coordinates to zero. Compare to $\ell_2$: the prox of $\tfrac{\mu}{2}\|\cdot\|_2^2$ is just multiplicative shrinkage $v/(1+t\mu)$, no dead-zone, nothing ever hits zero. The nonsmoothness is the point.

Putting it together with $f(x)=\tfrac12\|Ax-y\|_2^2$, so $\nabla f(x) = A^{\top}(Ax-y)$, the iteration is

$$ x_k = \operatorname{soft}\Big(x_{k-1} - t\,A^{\top}(Ax_{k-1} - y),\ t\lambda\Big). $$

A gradient step (one $A$, one $A^{\top}$), then a coordinatewise shrink. This is **ISTA** — iterative shrinkage-thresholding. Cheap, simple, exactly the first-order/prox method the scale demanded.

Now I have to pin down two things: what step size $t$ keeps it converging, and how fast it actually goes. For the step size, go back to the quadratic-model picture. The reason linearizing $f$ and adding $\tfrac{1}{2t}\|x-x_{k-1}\|^2$ is *safe* is that, if $\tfrac1t$ is large enough, that quadratic model sits *above* $f$ everywhere, so minimizing the model can only decrease $F$. The relevant fact about a smooth convex $f$ with $L$-Lipschitz gradient ($\|\nabla f(x)-\nabla f(z)\| \le L\|x-z\|$) is the descent lemma:

$$ f(x) \le f(z) + \langle x - z, \nabla f(z)\rangle + \tfrac{L}{2}\|x-z\|_2^2 \qquad \forall x,z. $$

Let me actually prove this so I trust it. By the fundamental theorem of calculus along the segment $z\to x$, $f(x) - f(z) = \int_0^1 \langle \nabla f(z + s(x-z)),\, x-z\rangle\, ds$. Subtract the linear term: $f(x) - f(z) - \langle\nabla f(z), x-z\rangle = \int_0^1 \langle \nabla f(z+s(x-z)) - \nabla f(z),\, x-z\rangle\, ds \le \int_0^1 \|\nabla f(z+s(x-z)) - \nabla f(z)\|\,\|x-z\|\,ds \le \int_0^1 Ls\|x-z\|^2 ds = \tfrac{L}{2}\|x-z\|^2$, using Cauchy–Schwarz and Lipschitzness. That's the descent lemma. So with $\tfrac1t \ge L$, i.e. $t \le 1/L$, the quadratic model $Q_t(x,z) = f(z) + \langle x-z,\nabla f(z)\rangle + \tfrac{1}{2t}\|x-z\|^2 + g(x)$ majorizes $F(x) = f(x)+g(x)$, and minimizing it (which is exactly the prox step) is a majorization–minimization step: $F(x_k) \le Q_t(x_k, x_{k-1}) \le Q_t(x_{k-1}, x_{k-1}) = F(x_{k-1})$. The objective is monotone nonincreasing. For my $f$, the Lipschitz constant of $\nabla f$ is the largest eigenvalue of the Hessian $A^{\top}A$, so $L = \lambda_{\max}(A^{\top}A) = \|A\|^2$, and the safe step is $t = 1/L$. (If $L$ is unknown or expensive, I can *backtrack*: start with a guess $L$ and inflate it by a factor $\eta>1$ until the majorization inequality $F(p_L(x_{k-1})) \le Q_L(p_L(x_{k-1}), x_{k-1})$ holds; the descent lemma guarantees it holds once $L \ge L(f)$, so backtracking stops, with $L_k \le \eta L(f)$.)

Now the rate. To get it I need one master inequality that relates $F$ at the new point to $F$ anywhere, and it's worth deriving carefully because the *same* inequality will carry the accelerated method too. Let $p_L(z) = \operatorname{prox}_{(1/L)g}(z - \tfrac1L\nabla f(z))$ be one prox-gradient step from $z$ with stepsize $1/L$. I claim: if $L$ is large enough that $F(p_L(z)) \le Q_L(p_L(z), z)$ (true for $L \ge L(f)$), then for *every* $x$,

$$ F(x) - F(p_L(z)) \ \ge\ \tfrac{L}{2}\|p_L(z) - z\|^2 + L\langle z - x,\ p_L(z) - z\rangle. $$

Let me prove it. Abbreviate $z^+ = p_L(z)$. The optimality condition for the prox: $z^+$ minimizes $Q_L(\cdot, z)$, and that strongly convex problem has the stationarity $\nabla f(z) + L(z^+ - z) + \gamma = 0$ for some subgradient $\gamma \in \partial g(z^+)$. Now use convexity of $f$ and $g$ to lower-bound them by their (sub)tangents at the right points: $f(x) \ge f(z) + \langle x-z, \nabla f(z)\rangle$ and $g(x) \ge g(z^+) + \langle x - z^+, \gamma\rangle$. Add: $F(x) \ge f(z) + \langle x-z,\nabla f(z)\rangle + g(z^+) + \langle x - z^+, \gamma\rangle$. On the other side, by the assumed majorization, $F(z^+) \le Q_L(z^+,z) = f(z) + \langle z^+ - z, \nabla f(z)\rangle + \tfrac{L}{2}\|z^+-z\|^2 + g(z^+)$. Subtract this upper bound on $F(z^+)$ from the lower bound on $F(x)$; the $f(z)$ and $g(z^+)$ cancel:

$$ F(x) - F(z^+) \ \ge\ \langle x - z^+,\ \nabla f(z) + \gamma\rangle - \tfrac{L}{2}\|z^+ - z\|^2. $$

Now substitute $\nabla f(z) + \gamma = -L(z^+ - z)$ from stationarity: $\langle x - z^+, \nabla f(z)+\gamma\rangle = -L\langle x - z^+, z^+ - z\rangle = L\langle x - z^+,\ z - z^+\rangle$. Rewrite $x - z^+ = (x - z) + (z - z^+)$, so $\langle x - z^+, z - z^+\rangle = \langle x-z, z-z^+\rangle + \|z-z^+\|^2$. Hence $F(x) - F(z^+) \ge L\|z-z^+\|^2 + L\langle x-z, z-z^+\rangle - \tfrac{L}{2}\|z^+-z\|^2 = \tfrac{L}{2}\|z^+-z\|^2 + L\langle x - z, z - z^+\rangle$. Flip the sign inside the inner product ($z - z^+ = -(z^+-z)$ and swap $x-z$ for $z-x$) to land it in the form I claimed:

$$ F(x) - F(z^+) \ \ge\ \tfrac{L}{2}\|z^+ - z\|^2 + L\langle z - x,\ z^+ - z\rangle. $$

Good. Now ISTA's $O(1/k)$ rate falls out by using this twice per step and telescoping. Take constant stepsize $L = L(f)$, so $x_{n+1} = p_L(x_n)$. First apply the master inequality with $x = x^\star$ (a minimizer), $z = x_n$, $z^+ = x_{n+1}$, and multiply by $2/L$:

$$ \tfrac{2}{L}\big(F(x^\star) - F(x_{n+1})\big) \ \ge\ \|x_{n+1}-x_n\|^2 + 2\langle x_n - x^\star,\ x_{n+1}-x_n\rangle. $$

The right side is exactly $\|x^\star - x_{n+1}\|^2 - \|x^\star - x_n\|^2$ (expand $\|x^\star - x_{n+1}\|^2 = \|(x^\star - x_n) - (x_{n+1}-x_n)\|^2 = \|x^\star - x_n\|^2 - 2\langle x^\star - x_n, x_{n+1}-x_n\rangle + \|x_{n+1}-x_n\|^2$; the cross term sign matches because $\langle x_n - x^\star, \cdot\rangle = -\langle x^\star - x_n, \cdot\rangle$). So with $v_n := F(x_n) - F(x^\star) \ge 0$,

$$ -\tfrac{2}{L} v_{n+1} \ \ge\ \|x^\star - x_{n+1}\|^2 - \|x^\star - x_n\|^2. $$

Sum over $n = 0,\dots,k-1$. The right side telescopes to $\|x^\star - x_k\|^2 - \|x^\star - x_0\|^2 \ge -\|x^\star-x_0\|^2$, so $\tfrac{2}{L}\sum_{n=1}^{k} v_n \le \|x^\star - x_0\|^2$. And because the objective is monotone nonincreasing, $v_k \le v_n$ for all $n \le k$, so $k\,v_k \le \sum_{n=1}^k v_n \le \tfrac{L}{2}\|x^\star - x_0\|^2$. Therefore

$$ F(x_k) - F(x^\star) \ \le\ \frac{L\,\|x_0 - x^\star\|^2}{2k} = O(1/k). $$

So ISTA matches plain gradient descent's sublinear rate — which makes sense, since with $g\equiv 0$ ISTA *is* gradient descent. And that's the problem. $O(1/k)$ means to drive the error down by another digit I need $10\times$ the iterations; in practice ISTA can be agonizingly slow, and rigorous worst-case examples make its asymptotic rate arbitrarily bad. I've got a correct, scalable method that's too slow. I need to go faster without giving up the one-cheap-prox-per-iteration budget.

Where could the speed come from? I know that for *smooth* convex minimization, plain gradient descent's $O(1/k)$ is **not** the best a first-order method can do. Nesterov (1983) showed there's a gradient method achieving $O(1/k^2)$ — and Nemirovsky–Yudin had already shown $O(1/k^2)$ is the *best possible* rate for any method that only sees function values and gradients on this class, so it's optimal, not merely better. The remarkable thing about Nesterov's scheme is that it costs no more than ordinary gradient descent: still *one* gradient per iteration. The extra ingredient is just a cleverly chosen auxiliary point. Instead of taking the gradient/prox step from the previous iterate $x_{k-1}$, take it from an *extrapolated* point $y_k$ — a specific linear combination of the last two iterates that "looks ahead" along the direction of recent progress, a momentum-like overshoot. Since ISTA reduces to gradient descent when $g\equiv 0$, and the prox-gradient step $p_L$ is the natural generalization of a gradient step, the natural conjecture is that I can graft Nesterov's extrapolation directly onto ISTA: keep the exact same prox step, but evaluate it at $y_k$ rather than at $x_{k-1}$.

So propose:

$$ x_k = p_L(y_k), \qquad y_{k+1} = x_k + \beta_k (x_k - x_{k-1}), $$

with $y_1 = x_0$, and some momentum weights $\beta_k$ to be determined. The per-iteration cost is identical to ISTA — one prox, one $A$, one $A^{\top}$ — plus a couple of vector axpys, which is nothing. The only question is what the $\beta_k$ must be for the rate to improve, and *why those values*. I don't want to guess; I want the analysis to *force* the weights. Let me carry the master inequality through and see what the algebra demands.

Introduce a scalar sequence $t_k \ge 1$ (to be pinned down) and set the momentum weight $\beta_k = \tfrac{t_k - 1}{t_{k+1}}$ — I'm writing it this way because the convergence proof for the smooth case naturally produces weights of this form, and I'll let the proof tell me the recursion $t_k$ must satisfy. For the clean constant-step derivation, apply the master inequality twice at the *extrapolation* point $z = y_{k+1}$ with the same $L$ each time (so $x_{k+1} = p_{L}(y_{k+1})$): once with $x = x_k$, once with $x = x^\star$. Writing $v_k = F(x_k) - F(x^\star)$:

$$ \tfrac{2}{L}(v_k - v_{k+1}) \ \ge\ \|x_{k+1} - y_{k+1}\|^2 + 2\langle x_{k+1}-y_{k+1},\ y_{k+1} - x_k\rangle, $$
$$ -\tfrac{2}{L}v_{k+1} \ \ge\ \|x_{k+1} - y_{k+1}\|^2 + 2\langle x_{k+1}-y_{k+1},\ y_{k+1} - x^\star\rangle. $$

To produce a relation that telescopes in $v_k$, take the weighted combination that keeps the coefficient of $v_{k+1}$ negative: multiply the first by $(t_{k+1} - 1)$ and add the second:

$$ \tfrac{2}{L}\big((t_{k+1}-1)v_k - t_{k+1}v_{k+1}\big) \ \ge\ t_{k+1}\|x_{k+1}-y_{k+1}\|^2 + 2\langle x_{k+1}-y_{k+1},\ t_{k+1}y_{k+1} - (t_{k+1}-1)x_k - x^\star\rangle. $$

Now multiply through by $t_{k+1}$. On the left I get $\tfrac{2}{L}\big(t_{k+1}(t_{k+1}-1)v_k - t_{k+1}^2 v_{k+1}\big)$. For this to become a telescoping energy, the coefficient of $v_k$ has to be $t_k^2$, so I need $t_{k+1}(t_{k+1}-1) = t_k^2$, i.e.

$$ t_{k+1}^2 - t_{k+1} = t_k^2, $$

because then the left side becomes $\tfrac{2}{L}\big(t_k^2 v_k - t_{k+1}^2 v_{k+1}\big)$ — a clean telescoping difference of $t_k^2 v_k$. That requirement *is* the definition of the $t$ recursion; solving the quadratic $t_{k+1}^2 - t_{k+1} - t_k^2 = 0$ for the positive root gives

$$ t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad t_1 = 1. $$

So the seemingly arbitrary "$\tfrac{1+\sqrt{1+4t_k^2}}{2}$" isn't arbitrary at all — it's exactly the recursion that makes the energy $t_k^2 v_k$ telescope. With that choice, multiplying the right side by $t_{k+1}$ and grouping, the quadratic terms assemble (via the identity $\|b-a\|^2 + 2\langle b-a, a-c\rangle = \|b-c\|^2 - \|a-c\|^2$, applied with $a = t_{k+1}y_{k+1}$, $b = t_{k+1}x_{k+1}$, $c = (t_{k+1}-1)x_k + x^\star$) into

$$ \tfrac{2}{L}\big(t_k^2 v_k - t_{k+1}^2 v_{k+1}\big) \ \ge\ \|t_{k+1}x_{k+1} - (t_{k+1}-1)x_k - x^\star\|^2 - \|t_{k+1}y_{k+1} - (t_{k+1}-1)x_k - x^\star\|^2. $$

Define $u_k := t_k x_k - (t_k - 1)x_{k-1} - x^\star$. The first bracket on the right is $\|u_{k+1}\|^2$ by definition. For the second bracket to be $\|u_k\|^2$, I need $t_{k+1}y_{k+1} - (t_{k+1}-1)x_k = t_k x_k - (t_k-1)x_{k-1}$, i.e. $t_{k+1}y_{k+1} = t_{k+1}x_k + (t_k - 1)(x_k - x_{k-1})$, i.e.

$$ y_{k+1} = x_k + \frac{t_k - 1}{t_{k+1}}\,(x_k - x_{k-1}). $$

There it is — the extrapolation rule, with exactly the weight $\beta_k = \tfrac{t_k-1}{t_{k+1}}$ I wrote down on faith, now *derived* as the unique choice that turns the second bracket into $\|u_k\|^2$. The momentum formula and the $t$-recursion are a matched pair: each is forced by making the analysis telescope. With both in place,

$$ \frac{2}{L}\,t_k^2 v_k - \frac{2}{L}\,t_{k+1}^2 v_{k+1} \ \ge\ \|u_{k+1}\|^2 - \|u_k\|^2. $$

Now set $a_k = \tfrac{2}{L}t_k^2 v_k$ and $b_k = \|u_k\|^2$. The inequality reads $a_k - a_{k+1} \ge b_{k+1} - b_k$, i.e. $a_{k+1} + b_{k+1} \le a_k + b_k$ — the quantity $a_k + b_k$ is nonincreasing, so $a_k + b_k \le a_1 + b_1$ for all $k$. I need to check the base case $a_1 + b_1 \le \|x_0 - x^\star\|^2$: with $t_1 = 1$, $a_1 = \tfrac{2}{L}v_1$ and $b_1 = \|u_1\|^2 = \|x_1 - x^\star\|^2$ (since $u_1 = t_1 x_1 - 0 - x^\star$). Apply the master inequality once at $z = y_1 = x_0$, $x = x^\star$: $\tfrac{2}{L}v_1 = \tfrac{2}{L}(F(x_1)-F(x^\star)) \le \|x_0 - x^\star\|^2 - \|x_1 - x^\star\|^2$, which rearranges to $a_1 + b_1 \le \|x_0 - x^\star\|^2$. So $\tfrac{2}{L}t_k^2 v_k \le a_k + b_k \le \|x_0 - x^\star\|^2$, giving

$$ v_k = F(x_k) - F(x^\star) \ \le\ \frac{L\,\|x_0 - x^\star\|^2}{2\,t_k^2}. $$

The last step is to show $t_k$ grows linearly, so $t_k^2 \sim k^2$. From $t_{k+1}^2 = t_{k+1} + t_k^2$ and $t_{k+1} \ge 1$, an induction gives $t_k \ge (k+1)/2$: it holds at $k=1$ ($t_1 = 1 \ge 1$), and if $t_k \ge (k+1)/2$ then $t_{k+1} = \tfrac{1+\sqrt{1+4t_k^2}}{2} \ge \tfrac{1 + \sqrt{1 + (k+1)^2}}{2} \ge \tfrac{1 + (k+1)}{2} = \tfrac{k+2}{2}$. Plug in $t_k^2 \ge (k+1)^2/4$:

$$ F(x_k) - F(x^\star) \ \le\ \frac{2L\,\|x_0 - x^\star\|^2}{(k+1)^2} = O(1/k^2). $$

So extrapolating the prox step to the look-ahead point $y_k$ — at no extra per-iteration cost beyond two vector additions — turns ISTA's $O(1/k)$ into $O(1/k^2)$, the optimal first-order rate. To hit accuracy $\varepsilon$ I now need $O(1/\sqrt\varepsilon)$ iterations instead of $O(1/\varepsilon)$, an enormous practical gain. This is **FISTA** — the fast iterative shrinkage-thresholding algorithm. And note the analysis never used anything special about $\ell_1$: $g$ could be any convex function with a cheap prox, and $f$ any smooth convex function; the soft-threshold is just the instance $g = \lambda\|\cdot\|_1$.

Let me also sanity-check the momentum from a different angle, because the $t_k$ recursion is opaque. For large $k$, $t_{k+1} \approx t_k + \tfrac12$ (from $t_{k+1}^2 - t_k^2 = t_{k+1}$ and $t_{k+1}+t_k \approx 2t_k$), so $t_k \approx k/2$, and the weight $\beta_k = \tfrac{t_k - 1}{t_{k+1}} \approx \tfrac{k/2 - 1}{k/2 + 1/2} \to \tfrac{k}{k+3}$. So asymptotically FISTA is "add about $\tfrac{k}{k+3}$ of the last step as momentum" — momentum that starts near zero and climbs toward 1, exactly the increasing-overshoot profile of Nesterov acceleration. That's a useful equivalent form of the weight.

Now let me land this in code, grounded in the same structure a standard proximal-gradient solver uses. The smooth part contributes $\nabla f(x) = A^{\top}(Ax - y)$; the fixed step is $t = 1/L$ with $L = \lambda_{\max}(A^{\top}A) = \|A\|^2$, which I can estimate by power iteration; if I do not know a safe step, backtracking starts from a trial step and shrinks it until the quadratic model majorizes the smooth part. The nonsmooth part enters only through its prox, and the acceleration switch only changes the extrapolation point.

```python
import numpy as np


def smooth_value(A, y, x):
    r = A @ x - y
    return 0.5 * float(np.real(np.vdot(r, r)))


def grad_smooth(A, y, x):
    return A.conj().T @ (A @ x - y)


def soft_threshold(v, thresh):
    # Prox of lam*||.||_1 at step t, with thresh = t*lam.
    mag = np.abs(v)
    if np.iscomplexobj(v):
        return np.maximum(mag - thresh, 0.0) * np.exp(1j * np.angle(v))
    return np.sign(v) * np.maximum(mag - thresh, 0.0)


def lipschitz_constant(A, n_iter=100, seed=0):
    # L = ||A||^2 = lambda_max(A.conj().T @ A), so the largest fixed step is 1/L.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise ValueError("power iteration initialized at zero")
    x = x / norm
    for _ in range(n_iter):
        x = A.conj().T @ (A @ x)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            return 0.0
        x = x / norm
    return float(np.real(np.vdot(x, A.conj().T @ (A @ x))))


def _backtracking(point, step, A, y, lam, beta=0.5, n_back=100):
    grad = grad_smooth(A, y, point)
    f_point = smooth_value(A, y, point)
    for _ in range(n_back):
        cand = soft_threshold(point - step * grad, step * lam)
        d = cand - point
        q = f_point + np.real(np.vdot(grad, d)) + (0.5 / step) * np.real(np.vdot(d, d))
        if smooth_value(A, y, cand) <= q:
            break
        step *= beta
    return cand, step


def proximal_gradient_l1(A, y, lam, x0=None, step=None, n_iter=500,
                         backtracking=False, beta=0.5, n_back=100,
                         acceleration=None):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    if step is None:
        backtracking = True
        step = 1.0
    if acceleration not in (None, "fista", "vandenberghe"):
        raise ValueError("acceleration must be None, 'fista', or 'vandenberghe'")

    x = x0.copy()
    z = x.copy()
    tk = 1.0
    for i in range(n_iter):
        x_old = x.copy()
        if backtracking:
            x, step = _backtracking(z, step, A, y, lam, beta=beta, n_back=n_back)
        else:
            x = soft_threshold(z - step * grad_smooth(A, y, z), step * lam)

        if acceleration == "fista":
            tk_old = tk
            tk = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
            omega = (tk_old - 1.0) / tk
        elif acceleration == "vandenberghe":
            omega = i / (i + 3.0)
        else:
            omega = 0.0
        z = x + omega * (x - x_old)
    return x


def ista(A, y, lam, n_iter=500, L=None):
    if L is None:
        L = lipschitz_constant(A)
    if L <= 0.0:
        raise ValueError("L must be positive")
    step = 1.0 / L
    return proximal_gradient_l1(A, y, lam, step=step, n_iter=n_iter)


def fista(A, y, lam, n_iter=500, L=None):
    if L is None:
        L = lipschitz_constant(A)
    if L <= 0.0:
        raise ValueError("L must be positive")
    step = 1.0 / L
    return proximal_gradient_l1(A, y, lam, step=step, n_iter=n_iter, acceleration="fista")
```

The causal chain, start to finish: an underdetermined $y=Ax$ has no unique solution, but the signal is sparse, so I want the sparsest consistent $x$; that's $\ell_0$ minimization, which is NP-hard and combinatorial; I relax it to the convex $\ell_1$ — the tightest convex surrogate, whose cross-polytope geometry pushes solutions onto the axes — and under a near-isometry condition on $A$ the $\ell_1$ solution provably *is* the $\ell_0$ solution; with noise I use the penalized form $\tfrac12\|Ax-y\|^2 + \lambda\|x\|_1$, convex but nonsmooth; the scale forbids interior-point methods and demands a first-order, $A/A^{\top}$-only algorithm, so I split the objective into a smooth $f$ I treat by its gradient and a simple nonsmooth $g$ I treat by its prox; minimizing the quadratic-model-plus-$g$ collapses each step into a gradient step followed by $\operatorname{prox}_{tg}$, and for $g=\lambda\|\cdot\|_1$ that prox is soft-thresholding, whose dead-zone delivers exact sparsity — giving ISTA at $O(1/k)$; ISTA is too slow, so I evaluate the same prox step at a Nesterov-extrapolated point, and forcing the convergence inequality to telescope pins down both the $t_k = \tfrac{1+\sqrt{1+4t_{k-1}^2}}{2}$ recursion and the momentum weight $\tfrac{t_{k-1}-1}{t_k}$, yielding FISTA at the optimal $O(1/k^2)$ — all at one cheap prox per iteration.
