## Research question

A great many of the methods we trust most touch the data in only one way: through inner products. The perceptron, in its mistake-driven dual form, never looks at a point except to take $\langle x_t, x\rangle$. Ridge regression and least squares solve normal equations whose data dependence is the Gram matrix $X X^\top$. Linear discriminant rules, principal components, the whole machinery of linear geometry — strip away the surface and what remains is dot products between examples.

This buys clean theory and fast computation, but it costs us expressiveness. A linear rule can only carve the input space with flat boundaries. Data that curls — concentric rings, crescents, anything where the right decision surface bends — defeats it. The classical escape is to engineer nonlinear features: map each point $x$ to $\phi(x) = (\phi_1(x), \phi_2(x), \dots)$ (squares, products, higher monomials), then run the linear method on $\phi(x)$. A flat boundary in $\phi$-space is a curved boundary back in $x$-space. The trouble is that to capture rich nonlinearity the feature vector $\phi(x)$ must be enormous — for degree-$d$ polynomials in $p$ variables it has on the order of $p^d$ coordinates — and for some desirable similarity notions the natural feature space is infinite-dimensional. Writing down and computing in $\phi(x)$ is then hopeless.

The precise question: **can we get the modelling power of an arbitrarily rich (even infinite-dimensional) feature space while paying only the cost of the inner products we were already computing — and, when we set up a learning problem in such a space, can we avoid actually optimizing over an infinite-dimensional object?** A solution would have to (i) supply a cheap-to-evaluate function $k(x,x')$ that secretly equals $\langle \phi(x), \phi(x')\rangle$ for some feature map we never build, with a usable criterion for which $k$ are legitimate; and (ii) guarantee that the function we are solving for, though it lives in an infinite-dimensional space, is pinned down by the finitely many training points.

## Background

**Linear methods are confined to flat boundaries and are written in inner products.** A linear classifier or regressor can only carve the input space with hyperplanes, so problems that are not linearly separable lie outside its reach. The way such methods are written down is in terms of inner products: the dual perceptron maintains weights $w = \sum_t \alpha_t y_t x_t$ and classifies by $\mathrm{sign}\,\langle w, x\rangle = \mathrm{sign}\sum_t \alpha_t y_t \langle x_t, x\rangle$; dual least-squares and discriminant analysis are similar, with fitting and prediction expressed via $\langle x_i, x_j\rangle$ and $\langle x_i, x\rangle$.

**Positive-definite functions and Gram matrices.** Given a symmetric function $k:\mathcal X\times\mathcal X\to\mathbb R$ and points $x_1,\dots,x_m$, the Gram matrix is $K=(k(x_i,x_j))$. $k$ is called *positive definite* if every such Gram matrix satisfies $\sum_{i,j} c_i c_j K_{ij}\ge 0$ for all real $c$. A real symmetric matrix is positive semidefinite exactly when all its eigenvalues are nonnegative; equivalently $c^\top K c\ge 0$ for all $c$. Two immediate consequences for a positive-definite $k$: $k(x,x)\ge 0$ (the $1\times1$ Gram matrix), and a Cauchy–Schwarz-type bound $k(x,x')^2\le k(x,x)\,k(x',x')$ (from positivity of the $2\times2$ Gram matrix).

**Mercer's theorem.** The word "kernel" comes from integral operators. A function $k$ defines $(T_k f)(x)=\int_{\mathcal X} k(x,x')f(x')\,dx'$, and $k$ is *the kernel of* $T_k$. Mercer's theorem states that a continuous symmetric positive-definite kernel on a compact domain admits a uniformly convergent eigen-expansion
$$ k(x,x') = \sum_{j} \lambda_j\, \psi_j(x)\,\psi_j(x'), \qquad \lambda_j\ge 0, $$
in eigenfunctions $\psi_j$ of $T_k$ (orthonormal in $L_2$) with nonnegative eigenvalues. Reading $\phi(x)=(\sqrt{\lambda_j}\,\psi_j(x))_j$, this says $k(x,x')=\sum_j \big(\sqrt{\lambda_j}\psi_j(x)\big)\big(\sqrt{\lambda_j}\psi_j(x')\big)$ — an inner product of (possibly infinitely many) features. Mercer is the analytic precedent that a positive-definite function *is* a dot product of features; the eigenvalues must be nonnegative, which is precisely positive definiteness.

**Reproducing kernel Hilbert spaces (Aronszajn 1950).** Aronszajn's *Theory of reproducing kernels* studies Hilbert spaces $\mathcal H$ of functions on a set $\mathcal X$ in which every point-evaluation $f\mapsto f(x)$ is a bounded linear functional. By the Riesz representation theorem each evaluation then has a representer $k_x\in\mathcal H$ with $\langle k_x, f\rangle = f(x)$ for all $f$; the bivariate function $k(x,x')=k_x(x')$ is the *reproducing kernel*, and it satisfies the reproducing property $\langle k(\cdot,x), f\rangle = f(x)$ and in particular $\langle k(\cdot,x),k(\cdot,x')\rangle=k(x,x')$. The Moore–Aronszajn theorem closes the loop: to every positive-definite kernel on $\mathcal X\times\mathcal X$ there corresponds a unique RKHS having it as reproducing kernel, and conversely. This is the bridge between the algebraic object (positive-definite $k$) and a concrete function space.

**Ill-posed function estimation and regularization.** Estimating a function $f$ from finite data in an infinite-dimensional model space is ill-posed; the standard cure is regularization — minimize a fit term plus a penalty,
$$ C(f\mid \text{data}) + \lambda\, J(f), $$
where $J(f)$ penalizes roughness (e.g. $J(f)=\int_a^b (Lf)^2$ for a linear differential operator $L$, with $f$ ranging over a Sobolev space $W_2^m[a,b]$ of functions whose first $m-1$ derivatives are absolutely continuous and $f^{(m)}\in L_2$). The tuning parameter $\lambda$ trades fit against smoothness. The open difficulty: the minimization is posed over an infinite-dimensional space, so it is not obvious it is even computable.

**The geometric picture behind lifting.** Data that is not linearly separable in its native coordinates often becomes separable after a nonlinear map into higher dimension: send $[x^{(1)},x^{(2)}]\mapsto[(x^{(1)})^2,(x^{(2)})^2,x^{(1)}x^{(2)}]$ and a parabola in the plane becomes a hyperplane upstairs. This is a documented, repeatedly observed phenomenon — the motivation for working in feature spaces at all. The cost of doing it explicitly is what motivates everything that follows.

## Baselines

**Explicit nonlinear feature expansion + linear method.** Choose a fixed dictionary of nonlinear features $\phi_1,\dots,\phi_D$, form $\phi(x)$, run a linear learner on it. Core idea is sound and the geometry works. *Gap:* $D$ explodes with the order of nonlinearity ($\sim p^d$ for degree-$d$ monomials), and the most natural similarity notions correspond to infinite $D$; both storing $\phi(x)$ and optimizing over the weight vector in $\mathbb R^D$ become infeasible. There is no escape as long as the features are materialized.

**The potential-function / kernel-perceptron method (Aizerman, Braverman & Rozonoer 1964).** To classify nonlinearly separable data, place a "potential" $K(x,x')$ at each example — a similarity that decays with distance, by analogy to the electrostatic field of a charge — and build the decision function online by a perceptron-like, mistake-driven rule, so that after processing the examples the rule has the form $h(x)=\sum_t \alpha_t y_t\, K(x_t, x)$, a combination over the training points. They argue (via a Mercer-type eigenfunction expansion $K(x,y)=\sum_i \lambda_i \varphi_i(x)\varphi_i(y)$) that $K$ acts as an inner product in a higher-dimensional "rectifying" space, so the nonlinear rule downstairs is a *linear* rule upstairs — obtained without ever computing the coordinates $\varphi_i$ — and prove the corrections terminate in finitely many steps when a separating potential exists. Core idea: lift via a kernel, separate linearly, never build the feature map. *Gap left open:* this is tied to a particular classification algorithm and its mistake-driven update; it does not, by itself, say which optimization problems in general admit a finite expansion, nor handle regularized risk with arbitrary loss and penalty.

**Interpolation and smoothing splines in an abstract Hilbert space.** Pose interpolation/smoothing as: among functions $u$ in a Hilbert space satisfying constraints $u(t_i)=y_i$ (or with a fit term), minimize $\int (Lu)^2$. Earlier work (Schoenberg and others) placed these problems in Hilbert-space terms and produced explicit spline solutions for specific operators $L=D^m$. Core idea: smoothness penalty as a Hilbert-space (semi)norm. *Gap:* the solutions were derived case by case for particular differential operators and particular (evaluation) constraints; what was missing was a single structural reason — independent of the operator and of the form of the data — for why the minimizer is a finite combination of data-determined functions.

**Support-vector machines (Boser, Guyon & Vapnik 1992; Vapnik).** Maximize the margin of a separating hyperplane in a feature space; the dual is a quadratic program whose objective depends on the data only through $\langle x_i, x_j\rangle$, which one replaces by $k(x_i,x_j)$. The optimal classifier comes out as $\sum_i \alpha_i y_i k(x_i,\cdot)+b$ — again a combination over training points. Core idea: kernelized convex margin maximization. *Gap:* the finite-expansion property here is read off from the *structure of that specific QP dual* and its Kuhn–Tucker conditions; it is not obvious it survives when the loss is not the hinge or the regularizer is not $\|f\|^2$ — e.g. when one wants to minimize a uniform-convergence risk bound, where there is no convenient QP to invoke.

## Evaluation settings

Natural yardsticks include nonlinear classification tasks where an explicit feature lift or a potential-function rule is the alternative (synthetic curved-boundary sets — concentric rings, crescents, spirals — and standard pattern-recognition tasks). For function estimation: scattered-data interpolation and smoothing on an interval, with the fit measured by squared error at the design points $t_i$ and smoothness by $\int (Lu)^2$; the polynomial-spline and thin-plate-spline settings ($\mathcal X=[a,b]$ or $\mathbb R^d$, Sobolev model space, $\lambda$ chosen by criteria such as generalized cross-validation) are the established testbeds. For regularized risk in an RKHS: penalized least squares, penalized likelihood, and margin losses, compared against the explicit-feature or QP-dual routes. The metrics and protocols are squared error, classification error, and the trade-off curve in $\lambda$.

## Code framework

A linear learner whose only contact with the data is an inner-product matrix already exists. The scaffold below leaves open the similarity between two points and the solve that fits the model from that matrix.

```python
import numpy as np

def similarity(x, z):
    # TODO: a cheap function of two points that gives richer geometry
    # than the raw dot product.
    pass

def gram_matrix(X):
    n = len(X)
    G = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = similarity(X[i], X[j])
    return G

def is_admissible(G):
    # TODO: test the algebraic condition that lets a similarity
    #       behave like an inner product.
    pass

def fit(X, y, reg):
    G = gram_matrix(X)
    # TODO: solve the learning problem from G, X, y, reg.
    coef = None
    return coef

def predict(coef, X_train, x):
    # A linear rule touches a new point only through its similarities to the
    # training points:
    return sum(coef[i] * similarity(X_train[i], x) for i in range(len(X_train)))
```
