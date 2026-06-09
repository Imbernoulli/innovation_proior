# Context: local optimality in nonconvex-nonconcave minimax optimization

## Research question

Many of the most prominent objectives in modern machine learning are not minimizations but
*minimax* problems
$$\min_{\mathbf{x}\in\mathcal X}\ \max_{\mathbf{y}\in\mathcal Y}\ f(\mathbf{x},\mathbf{y}),$$
where one agent drives $f$ down and the other drives it up. Generative adversarial networks
(generator vs. discriminator), adversarial training (classifier vs. perturbation), and multi-agent
reinforcement learning all have this shape, and in every one of them $f$ is *nonconvex–nonconcave*:
$f(\cdot,\mathbf y)$ need not be convex and $f(\mathbf x,\cdot)$ need not be concave. In practice these
problems are attacked with gradient descent ascent (GDA) — descend on $\mathbf x$, ascend on $\mathbf y$.

The pain point is foundational rather than algorithmic: **there is no agreed-upon notion of what a
*local* solution even is.** For ordinary nonconvex minimization the answer is settled — a local minimum,
characterized by $\nabla f=0$ and $\nabla^2 f\succeq 0$ — and gradient descent provably finds it. For
minimax no such clean object is in place. The classical global notions (a saddle point / Nash equilibrium)
need not exist when $f$ is nonconvex–nonconcave, and the global sequential solution is NP-hard to find. A
candidate local notion borrowed from simultaneous games (local Nash) exists in the literature, but it has
visible defects: it can fail to exist on trivially simple functions, it ignores the *order* of play that
$\min\max$ explicitly imposes, and — most tellingly — it does not match the points that GDA actually
converges to. A satisfactory answer would be a definition of local optimality that (i) reflects the
asymmetry between the two players that $\min\max$ imposes (unlike a symmetric notion), (ii) is *genuinely local* (determinable from $f$ in an
infinitesimal neighborhood), (iii) comes with first- and second-order characterizations like local minima
do, (iv) exists more often than local Nash, and (v) explains the asymptotic behavior of GDA.

## Background

**The minimax theorem and saddle points (the convex–concave world).** Game theory began with
von Neumann's 1928 minimax theorem (*Zur Theorie der Gesellschaftsspiele*): for a bilinear/convex–concave
payoff on compact convex domains,
$$\min_{\mathbf x}\max_{\mathbf y} f(\mathbf x,\mathbf y)=\max_{\mathbf y}\min_{\mathbf x} f(\mathbf x,\mathbf y),$$
so a saddle point exists and the game has a unique value. Sion (1958) extended this to quasiconvex–
quasiconcave $f$. The equality is what makes the *order of play irrelevant* in this regime: whether
$\mathbf x$ or $\mathbf y$ commits first, the value is the same. In the convex–concave setting a saddle
point and a Nash equilibrium coincide, and averaged GDA iterates converge to it. Everything downstream
depends on this equality — and it is exactly what breaks for nonconvex–nonconcave $f$, where in general
$\min\max \neq \max\min$.

**Simultaneous vs. sequential games.** A *Nash equilibrium* $(\mathbf x^\star,\mathbf y^\star)$ of a
two-player zero-sum game is a point where neither player gains by unilaterally deviating:
$f(\mathbf x^\star,\mathbf y)\le f(\mathbf x^\star,\mathbf y^\star)\le f(\mathbf x,\mathbf y^\star)$ for all
$\mathbf x,\mathbf y$ — i.e. $\mathbf x^\star$ globally minimizes $f(\cdot,\mathbf y^\star)$ and
$\mathbf y^\star$ globally maximizes $f(\mathbf x^\star,\cdot)$. This is a *simultaneous*-game notion: both
players choose at once, with no order between them. A *Stackelberg* (sequential) game is different: a
leader commits to $\mathbf x$, the follower observes it and best-responds with
$\mathrm{BR}(\mathbf x)=\arg\max_{\mathbf y} f(\mathbf x,\mathbf y)$, and the leader, anticipating this,
minimizes the *worst-case* value $\phi(\mathbf x):=\max_{\mathbf y} f(\mathbf x,\mathbf y)$. The
machine-learning applications have this asymmetric flavor: a GAN trains a generator
*against* a discriminator that gets to react to it; adversarial training seeks a classifier robust to a
perturbation chosen *after* the classifier. When $\min\max\neq\max\min$, which player goes first changes the
solution.

**Existence is fragile.** Even for the smooth $f(x,y)=\sin(x+y)$ there is no Nash equilibrium, local or
global: every stationary point has $\nabla^2_{xx}f=\nabla^2_{yy}f=\pm 1$ with the *same* sign, so it can
never simultaneously be a local min in $x$ and a local max in $y$. By contrast the global sequential
solution — the minimizer of $\phi(\mathbf x)=\max_{\mathbf y} f$ — always exists on a compact domain by the
extreme-value theorem, so at the global level existence is not fragile. But computing it is NP-hard
(it contains nonconvex minimization), and gradient methods are inherently local; the gap is that no local
optimality notion is in place to say what such methods are entitled to reach.

**A subtle, load-bearing phenomenon: global $\ne$ local for minimax.** In ordinary nonconvex
minimization a global minimum is automatically a local minimum, so local minima always exist and local
search at least targets the right kind of point. This is *not* true for minimax. Because the global
sequential solution minimizes a *global* inner-max while any local notion can only minimize a *local*
inner-max, a global minimax point can be neither locally optimal nor even stationary. The function
$f(x,y)=0.2xy-\cos y$ on $[-1,1]\times[-2\pi,2\pi]$ is the diagnostic example: its global sequential
solutions are $(0,\pm\pi)$, where the gradient $(0.2y,\,0.2x+\sin y)$ equals $(\pm0.2\pi,0)\neq 0$. The
global solution sits at a non-stationary point. This says minimax is genuinely harder than nonconvex
minimization and deserves its own local theory.

**The asymptotics of GDA, and the gap it leaves.** Treating GDA as a discrete dynamical system
$\mathbf z_{t+1}=\mathbf w(\mathbf z_t)$, its Jacobian at a fixed point need not be symmetric (the update is
not the gradient of any scalar), so the linearized dynamics can rotate and even cycle rather than converge.
Daskalakis & Panageas (2018) analyzed the *limit* behavior: with random initialization GDA almost surely
avoids *linearly unstable* fixed points (via the center–stable manifold theorem), so the relevant question
is which *stable* fixed points it admits. They proved that the set of strict local Nash equilibria is
*strictly contained* in the set of GDA-stable fixed points — GDA stably converges to points that are *not*
local Nash. Mazumdar & Ratliff (2018) reached the same conclusion in a broader game setting. The stable
points outside local Nash had no game-theoretic interpretation; their existence was treated as a defect of
GDA. This is the diagnostic finding the field was sitting on: GDA's stable limit set is the *wrong* shape
for local Nash, and nobody could name what the extra points were.

## Baselines

**Local Nash equilibrium (local saddle).** The notion studied by Daskalakis & Panageas (2018),
Mazumdar & Ratliff (2018), and Adolphs et al. (2018). $(\mathbf x^\star,\mathbf y^\star)$ is a local Nash
equilibrium if there is $\delta>0$ such that for all $(\mathbf x,\mathbf y)$ within $\delta$,
$$f(\mathbf x^\star,\mathbf y)\le f(\mathbf x^\star,\mathbf y^\star)\le f(\mathbf x,\mathbf y^\star).$$
First order it requires $\nabla_{\mathbf x}f=\nabla_{\mathbf y}f=0$; second order, necessarily
$\nabla^2_{\mathbf y\mathbf y}f\preceq 0$ and $\nabla^2_{\mathbf x\mathbf x}f\succeq 0$; a stationary point
with $\nabla^2_{\mathbf y\mathbf y}f\prec 0$ and $\nabla^2_{\mathbf x\mathbf x}f\succ 0$ is a *strict* local
Nash. *Gaps:* (1) It is a simultaneous-game concept — perfectly symmetric in the two players (block-diagonal
Hessian conditions on $\mathbf x$ and $\mathbf y$ separately, no coupling term $\nabla^2_{\mathbf x\mathbf y}f$)
— so it cannot reflect the leader/follower order that $\min\max$ imposes. (2) It can fail to exist
($\sin(x+y)$). (3) It is *strictly smaller* than the GDA-stable set, so it does not characterize what GDA
finds. Adolphs et al. (2018) and Mazumdar et al. (2019) instead modify the algorithm — Hessian/curvature-
based updates whose stable points are exactly local Nash — but keep local Nash as the goal, inheriting (1)
and (2).

**Stable fixed points of GDA (Daskalakis & Panageas 2018; Mazumdar & Ratliff 2018).** Define the GDA
dynamics $\mathbf x_{t+1}=\mathbf x_t-\alpha\nabla_{\mathbf x}f$, $\mathbf y_{t+1}=\mathbf y_t+\alpha\nabla_{\mathbf y}f$;
a fixed point is a stationary point of $f$; it is linearly stable iff the Jacobian
$\mathbf I+\alpha\mathbf H$ with
$\mathbf H=\begin{pmatrix}-\nabla^2_{xx}f & -\nabla^2_{xy}f\\ \nabla^2_{yx}f & \nabla^2_{yy}f\end{pmatrix}$
has spectral radius $\le 1$, i.e. (as $\alpha\to0$) all eigenvalues of $\mathbf H$ have negative real part.
They prove $\textsf{Local-Nash}\subsetneq\textsf{GDA-stable}\subsetneq\textsf{OGDA-stable}$. *Gap:* the
middle and right sets contain points with *no known meaning* — the central open question is what those
stable-but-not-Nash points are. This is the question a correct local notion must answer.

**Nonconvex-but-concave and variational-inequality relaxations.** Rafique et al. (2018) and
Nouiehed et al. (2019) handle the easier case where $f(\mathbf x,\cdot)$ is concave for every $\mathbf x$,
combining approximate inner maximization with proximal/gradient steps to reach stationary points of
$\phi(\mathbf x)=\max_{\mathbf y} f$. Lin et al. (2018) assume a variational-inequality structure; Hsieh et
al. (2018) target *mixed* (distributional) Nash. *Gap:* none of these defines local optimality for the
fully nonconvex–nonconcave *pure-strategy* sequential problem, which is the regime of GANs.

**Evtushenko (1974) "local" minimax.** An older candidate: $(\mathbf x^\star,\mathbf y^\star)$ is a
local solution if there is *some* neighborhood $\mathcal W$ in which it is the global sequential solution of
$f|_{\mathcal W}$. *Gap:* this is not a *truly local* property — whether a point qualifies can depend on the
values of $f$ far away (enlarging $\mathcal W$ can flip the verdict), so it does not satisfy first- or
second-order necessary conditions and cannot be read off the local Hessian.

## Evaluation settings

The natural yardstick for a local-optimality notion is not a benchmark number but a set of
*mathematical tests* a candidate should pass, applied to small closed-form $f$ where everything is computable:
- *Existence/non-existence diagnostics* — $f(x,y)=\sin(x+y)$ (no Nash anywhere); $f(x,y)=y^2-2xy$ on a box
  (a candidate where local solutions may not exist); $f(x,y)=0.2xy-\cos y$ (global solution non-stationary).
- *Separation diagnostics* — $f(x,y)=x^2-y^2$ (origin passes the local Nash test);
  $f(x,y)=-x^2+5xy-y^2$ (origin fails local Nash but has a strong cross-term), to test whether a sequential
  local notion can relax the blockwise Nash condition without ignoring the follower's reaction.
- *Algorithmic-correspondence diagnostics* — small quadratics $f=\pm x^2+2\sqrt\epsilon\,xy\pm(\epsilon/2)y^2$,
  whose GDA Jacobian eigenvalues can be computed exactly, to test how fixed points of GDA relate to any
  candidate sequential local notion.
- *Regularity classes* under which the global sequential solution is guaranteed to be locally optimal:
  $f(\mathbf x,\cdot)$ strongly concave near its maxima, or all-local-maxima-are-global (a property
  established in matrix-completion / Burer–Monteiro landscapes by Ge et al. 2017 and Boumal et al. 2016).
The "metrics" are correctness of first/second-order conditions, set inclusions between the candidate notion
and the GDA-stable set, and existence guarantees — not held-out performance.

## Code framework

The available computational harness is a payoff $f$ with autodiff for its gradients and Hessian blocks, the
bare GDA loop, a generic fixed-point/stability check on a Jacobian, and an open slot for a candidate
local-optimality test.

```python
import numpy as np

# --- payoff and its derivatives (autodiff / closed form) --------------------
def f(x, y):            # the nonconvex-nonconcave payoff, min over x, max over y
    raise NotImplementedError

def grads(x, y):        # returns (grad_x f, grad_y f)
    raise NotImplementedError

def hess_blocks(x, y):  # returns A = d2f/dxx, B = d2f/dyy, C = d2f/dxy
    raise NotImplementedError

# --- vanilla local-search dynamics ------------------------------------------
def gda_step(x, y, eta):
    gx, gy = grads(x, y)
    return x - eta * gx, y + eta * gy            # descend x, ascend y

# --- generic dynamical-system fixed point / linear stability ----------------
def is_stationary(x, y, tol=1e-8):
    gx, gy = grads(x, y)
    return np.linalg.norm(gx) < tol and np.linalg.norm(gy) < tol

def linearly_stable(jacobian):
    # stable iff spectral radius <= 1 (discrete) / all Re(eig) < 0 (flow)
    raise NotImplementedError

# --- TODO: local optimality slot --------------------------------------------
def is_local_optimum(x, y):
    # TODO: check the local optimality notion for min_x max_y f.
    pass
```
