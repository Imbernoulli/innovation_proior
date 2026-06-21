We are given a data matrix $X \in \mathbb{R}^{n \times d}$ of $n$ i.i.d. observations of $d$ variables, believed to come from a nonlinear structural equation model in which each variable is some unknown function of its parents plus independent noise, $X_j = f_j(X_{\mathrm{pa}(j)}) + z_j$, and we want to recover the directed acyclic graph behind them — who causes whom. The thing that makes this hard is the acyclicity requirement. Phrased as score-based learning — pick the graph that best fits the data under a sparsity-penalized score — the feasible set is the discrete space of DAGs, which is superexponential in $d$ and NP-hard to search. Every method I know therefore reaches for a combinatorial device to stay inside that space: add edges one at a time and check after each addition that no cycle appeared, or fix a topological order and search over the $d!$ orderings, fitting a regression within each. Those devices are exactly what make the existing methods slow, brittle, and a pain to implement. Linear NOTEARS already escaped the combinatorial trap for linear SEMs, but it assumes $f_j(X) = w_j^\top X$ and so can only fit a linear surrogate to nonlinear mechanisms, missing the very cause/effect asymmetry that nonlinearity exposes. The nonlinear alternatives each commit up front to one model class and one specialized search: CAM ties itself to an additive model with no parent interactions and runs a three-stage pipeline whose middle stage is still a combinatorial order search; kernel-based greedy equivalence search costs $O(n^3)$ per score and remains greedy edge-editing; DAG-GNN fixes a particular variational encoder/decoder and tends to predict too few edges; FGS is linear and returns only a Markov-equivalence CPDAG. None of them is a single, generic continuous program in which the model family is a plug-in and the optimizer is off-the-shelf. That is the gap I want to close: one continuous, sparsity-penalized objective over real-valued parameters, with acyclicity as a smooth constraint, that updates the whole graph at once.

Before building anything I want to be sure the problem is even solvable from observational data, because nonlinearity turns out to be on my side here. A jointly Gaussian linear SEM is not identifiable — a whole Markov equivalence class of DAGs induces the same Gaussian law, so no amount of data fixes an edge's direction. But nonlinear additive-noise models are identifiable under mild conditions (Hoyer et al. 2009; Peters et al. 2014, Corollary 31: if each $f_j$ is three-times differentiable and not linear in any argument, the DAG is generically recoverable). The intuition: regress $X_j$ on $X_k$ and inspect the residual; with a genuinely nonlinear mechanism $X_k \to X_j$ the residual is independent of $X_k$ in the correct direction but not in the reversed one, so the nonlinearity breaks the symmetry the Gaussian-linear case cannot break. Nonlinearity is thus not a modeling burden but the thing that makes direction visible. The problem is well-posed; I need a continuous machine to extract the answer.

I propose NOTEARS-MLP. The starting point is the smooth, exact algebraic characterization of acyclicity for the linear case (Zheng et al. 2018), which I want to understand part by part because everything hangs off it. In the binary world, $(B^k)_{ii}$ counts the length-$k$ closed walks from node $i$ back to itself, so a graph is acyclic iff $(B^k)_{ii} = 0$ for every $k \ge 1$ and every $i$, i.e. $\operatorname{tr}(B^k) = 0$ for all $k$. That is infinitely many conditions; I want one number. Collapsing the powers with the Neumann series gives $\operatorname{tr}((I-B)^{-1}) = d + \sum_{k\ge1}\operatorname{tr}(B^k)$, which equals $d$ iff acyclic — but it only converges when the spectral radius $r(B) < 1$, a restriction an arbitrary graph will not honor, and even then the walk counts $\operatorname{tr}(B^k)$ grow so fast that the series is wildly ill-conditioned. The finite version $\sum_{k=1}^{d}\operatorname{tr}(B^k) = 0$ drops the spectral condition (any cycle in $d$ nodes has length $\le d$) but the entries of $B^k$ overflow for even modest $d$. The requirement that crystallizes is a power series in $B$ that vanishes iff every $\operatorname{tr}(B^k)$ vanishes, converges for all matrices, and damps the explosive high powers — and the matrix exponential does all three at once:
$$\operatorname{tr}(e^{B}) - d = \sum_{k\ge1} \frac{\operatorname{tr}(B^k)}{k!},$$
zero iff every closed-walk count is zero, convergent for any square matrix, and numerically tame because the $1/k!$ weights crush exactly the long-walk terms that blew up. To allow real weights (which I need for gradients) without letting negative entries cancel walk counts, square entrywise via the Hadamard product so every edge weight is nonnegative, giving
$$h(W) = \operatorname{tr}\!\big(e^{\,W \circ W}\big) - d, \qquad [W \circ W]_{kj} = w_{kj}^2,$$
which is $0$ iff $G(W)$ — the graph with edge $k \to j$ whenever $w_{kj} \ne 0$ — is acyclic. Its value is a continuous "DAG-ness" dial counting weighted closed walks. Its gradient follows from $d\operatorname{tr}(e^{M}) = \operatorname{tr}(e^{M}\,dM)$ (from the series plus the cyclic property of trace) with $M = W \circ W$, $dM = 2W \circ dW$, giving $\nabla h(W) = (e^{\,W \circ W})^\top \circ 2W$, so both $h$ and its gradient cost one matrix exponential, $O(d^3)$.

The trouble is that this device lives entirely on a weighted matrix $W$, and a general nonlinear SEM $X_j = f_j(X) + z_j$ has no matrix of coefficients to feed it. The way past the wall is to ask what $h$ actually needs: for each ordered pair $(k,j)$, a single nonnegative number that is zero exactly when $f_j$ does not depend on $X_k$ and positive when it does. $h$ is agnostic about where that number comes from. A smooth function ignores a coordinate iff its partial derivative there vanishes everywhere, so working in $H^1$ the natural numeric stand-in for "$\partial_k f_j$ is the zero function" is its $L^2$ norm. Define the surrogate adjacency
$$[W(f)]_{kj} := \lVert \partial_k f_j \rVert_{L^2},$$
a genuine real, nonnegative $d \times d$ matrix that is zero exactly on the non-edges (the partial-derivative-as-dependence idea is the same one used for nonparametric variable selection, Rosasco et al. 2013), and drop it into the same $h$. As a sanity check, when $f_j(X) = w_j^\top X$ the dependence on coordinate $k$ disappears exactly when $w_{kj} = 0$, so the linear weighted adjacency falls out as a special case — the signature of a correct generalization rather than a bolted-on competitor.

This is still abstract: $f_j \in H^1$ is infinite-dimensional and $\lVert \partial_k f_j \rVert_{L^2}$ is an uncomputable integral. So I model each $f_j$ as a multilayer perceptron $\mathrm{MLP}(u; A^{(1)}, \dots, A^{(h)}) = \sigma(A^{(h)}\sigma(\cdots\sigma(A^{(1)}u)))$ with $A^{(1)} \in \mathbb{R}^{m_1 \times d}$ and a smooth activation $\sigma$: universal approximation captures any smooth $f_j$, and everything is differentiable in the weights. The first layer $A^{(1)}$ linearly mixes the $d$ inputs into $m_1$ pre-activations before any nonlinearity. The tempting dependence measure is "$f_j$ ignores $u_k$ iff the $k$-th column of $A^{(1)}$ is zero," but I have to settle whether forcing that column to zero throws away functions that are $u_k$-independent through some later-layer cancellation. So I prove the two classes coincide. Let $F$ be the MLPs (architecture fixed, weights free) independent of $u_k$ and $F_0$ those with the $k$-th column of $A^{(1)}$ all zero. The easy inclusion $F_0 \subseteq F$ is immediate: a zero column means $A^{(1)}u$ never sees $u_k$, so neither does anything downstream. For $F \subseteq F_0$, take $f \in F$ and let $\tilde u$ be $u$ with coordinate $k$ set to zero; since $f$ ignores $u_k$, $f(u) = f(\tilde u)$. Build $\tilde A^{(1)}$ from $A^{(1)}$ by zeroing only column $k$. Then for each hidden unit $s$,
$$(\tilde A^{(1)} u)_s = \sum_{k' \ne k} A^{(1)}_{sk'} u_{k'} = \sum_{k'} A^{(1)}_{sk'} \tilde u_{k'} = (A^{(1)} \tilde u)_s,$$
because the only dropped term on the left is $k'=k$, and on the right that term is already killed by $\tilde u_k = 0$. Hence $\tilde A^{(1)} u = A^{(1)} \tilde u$, and feeding both through the same later layers gives $f(u) = f(\tilde u) = \mathrm{MLP}(u; \tilde A^{(1)}, A^{(2)}, \dots, A^{(h)}) \in F_0$ (biases ride along as constants). So $F = F_0$: zeroing a first-layer column loses no expressivity. This proposition is doing real work beyond reassurance — it hands me the dependence summary and a cost guarantee. Independence on $X_k$ is controlled entirely by the $k$-th column of $A^{(1)}$, so the dependence scalar is its norm and, crucially, $h(W(\theta))$ depends on the network only through the $d$ first-layer columns, making the acyclicity computation independent of network depth no matter how many hidden layers I stack.

The concrete program then writes itself. With $\theta_j = (A_j^{(1)}, \dots, A_j^{(h)})$ the parameters of the $j$-th MLP and $\theta = (\theta_1, \dots, \theta_d)$,
$$\min_{\theta}\; \frac{1}{n}\sum_{j=1}^d \ell\big(x_j, \mathrm{MLP}(X; \theta_j)\big) + \lambda \sum_{j=1}^d \lVert A_j^{(1)} \rVert_{1,1} \quad \text{s.t.}\quad h(W(\theta)) = 0.$$
The fit term is least squares, which for additive Gaussian noise is the negative log-likelihood up to constants (any other differentiable loss or GLM link slots in identically). The $\ell_1$ on the first-layer weights drives whole columns to zero, which is exactly "drop a parent." For the constraint there is a tidy subtlety: $h$ wants $W \circ W$ with entries $w_{kj}^2$, and my $[W(\theta)]_{kj}$ is the column norm $\big(\sum_b (A_j^{(1)})_{bk}^2\big)^{1/2}$, so the squared column norm $[W(\theta)]_{kj}^2 = \sum_b (A_j^{(1)})_{bk}^2$ is already exactly $[W \circ W]_{kj}$. I therefore evaluate $h = \operatorname{tr}(e^{A}) - d$ directly on the sum-of-squares matrix $A_{kj} = \sum_b (A_j^{(1)})_{bk}^2$ — no square root inside the constraint, which keeps it smooth near zero; the $\sqrt{\,\cdot\,}$ appears only at the very end to report $W$ for thresholding.

To solve a smooth objective with the smooth equality constraint $h = 0$ over a nonconvex surface, I use the augmented Lagrangian,
$$L^\rho(\theta, \alpha) = L(\theta) + \frac{\rho}{2}\,\lvert h(W(\theta)) \rvert^2 + \alpha\, h(W(\theta)) + \lambda_1 \lVert \cdot \rVert_1 + \tfrac{\lambda_2}{2} R_2(\theta),$$
where the $\frac{\rho}{2} h^2$ quadratic penalty punishes constraint violation and the $\alpha h$ term carries the Lagrange multiplier. I keep both rather than cranking a pure penalty because a pure penalty needs $\rho \to \infty$ to drive $h$ exactly to zero, which makes the subproblem brutally ill-conditioned; the multiplier absorbs the residual so a finite $\rho$ recovers the constrained solution. The outer loop is dual ascent: since $\nabla D(\alpha) = h(\theta^*_\alpha)$, I update $\alpha \leftarrow \alpha + \rho h$, and I only stiffen the penalty when the constraint is stubborn — if after an inner solve $h$ has not shrunk to at most a quarter of its previous value, set $\rho \leftarrow 10\rho$ and resolve, otherwise accept the step — terminating when $h \le 10^{-8}$ or $\rho \ge 10^{16}$. Each inner subproblem is smooth-plus-$\ell_1$, the natural shape for L-BFGS-B (limited-memory BFGS with box bounds), but L-BFGS-B handles only box constraints, not an $\ell_1$ prox. The fix, inherited from the linear formulation, is variable splitting: write $A^{(1)} = A^{(1)}_+ - A^{(1)}_-$ with both parts bounded $\ge 0$, so $|A^{(1)}_{bk}| = (A^{(1)}_+)_{bk} + (A^{(1)}_-)_{bk}$ at the optimum and the $\ell_1$ norm becomes the linear function $\sum (A_+ + A_-)$ of nonnegative variables — exactly what L-BFGS-B optimizes. The same box bounds do double duty by pinning the diagonal entries to $(0,0)$, forbidding self-loops for free. For the architecture knobs, zero hidden units is just a linear model (and I expect a sharp jump in recovery the instant a few units appear, since that is the difference between seeing the nonlinear asymmetry and not), while too many units overfit each regressor at small $n$ and read spurious dependence into columns that should be zero; a single hidden layer of about ten sigmoid units sits in the sweet spot, sigmoid being smooth so the partials and the whole objective are differentiable. Finally, because the augmented Lagrangian leaves $h \le 10^{-8}$ rather than symbolically zero, the reported $W = \sqrt{A}$ may carry tiny numerical or weak spurious entries, so I hard-threshold $|W| < \omega$ with $\omega = 0.3$ as a fixed default; deleting edges can never create a cycle. The whole thing is one continuous program over real parameters, solved by a generic bound-constrained quasi-Newton solver, with every edge updated jointly and the model family a plug-in choice.

```python
import math
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
import torch.nn as nn


class TraceExpm(torch.autograd.Function):
    """f = tr(exp(A)).  d tr(exp(A)) = tr(exp(A) dA)  =>  grad is exp(A)^T."""
    @staticmethod
    def forward(ctx, input):
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (E,) = ctx.saved_tensors
        return grad_output * E.t()

trace_expm = TraceExpm.apply


class LocallyConnected(nn.Module):
    """One independent linear map per node: [n, d, m1] -> [n, d, m2]."""
    def __init__(self, num_linear, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(num_linear, out_features)) if bias else None
        k = 1.0 / in_features
        nn.init.uniform_(self.weight, -math.sqrt(k), math.sqrt(k))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):
        out = torch.matmul(x.unsqueeze(2), self.weight.unsqueeze(0)).squeeze(2)
        if self.bias is not None:
            out = out + self.bias
        return out


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):                 # dims = [d, m1, ..., 1]
        super().__init__()
        assert len(dims) >= 2 and dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)   # variable splitting for L1
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        self.fc2 = nn.ModuleList(
            [LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias)
             for l in range(len(dims) - 2)])

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for _ in range(self.dims[1]):
                for i in range(d):
                    bounds.append((0, 0) if i == j else (0, None))  # no self-loops
        return bounds

    def forward(self, x):                                # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)            # effective first layer
        x = x.view(-1, self.dims[0], self.dims[1])       # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(2)

    def _sq_colnorms(self):
        # [W o W]_{kj} = sum_b (A_j^{(1)})_{bk}^2 = squared k-th-column norm of node j's first layer
        d = self.dims[0]
        w1 = (self.fc1_pos.weight - self.fc1_neg.weight).view(d, -1, d)  # [j, m1, i]
        return torch.sum(w1 * w1, dim=1).t()             # [i, j]

    def h_func(self):
        return trace_expm(self._sq_colnorms()) - self.dims[0]   # tr(exp(W o W)) - d

    def l2_reg(self):
        reg = torch.sum((self.fc1_pos.weight - self.fc1_neg.weight) ** 2)
        for fc in self.fc2:
            reg = reg + torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        return torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)  # linear under >= 0 bounds

    @torch.no_grad()
    def fc1_to_adj(self):
        return torch.sqrt(self._sq_colnorms()).cpu().numpy()        # W = sqrt(W o W)


class LBFGSBScipy(torch.optim.Optimizer):
    """Wrap scipy L-BFGS-B; reads each parameter's `.bounds`."""
    def __init__(self, params):
        super().__init__(params, dict())
        self._params = self.param_groups[0]['params']
        self._numel = sum(p.numel() for p in self._params)

    def _flat_grad(self):
        return torch.cat([(p.grad.data.view(-1) if p.grad is not None
                           else p.data.new(p.numel()).zero_()) for p in self._params])

    def _flat_bounds(self):
        b = []
        for p in self._params:
            b += p.bounds if hasattr(p, 'bounds') else [(None, None)] * p.numel()
        return b

    def _flat_params(self):
        return torch.cat([p.data.view(-1) for p in self._params])

    def _set_params(self, flat):
        off = 0
        for p in self._params:
            n = p.numel()
            p.data = flat[off:off + n].view_as(p.data)
            off += n

    def step(self, closure):
        def wrapped(flat):
            self._set_params(torch.from_numpy(flat).to(torch.get_default_dtype()))
            loss = closure().item()
            return loss, self._flat_grad().cpu().detach().numpy().astype('float64')
        x0 = self._flat_params().cpu().detach().numpy()
        sol = sopt.minimize(wrapped, x0, method='L-BFGS-B', jac=True, bounds=self._flat_bounds())
        self._set_params(torch.from_numpy(sol.x).to(torch.get_default_dtype()))


def squared_loss(output, target):
    n = target.shape[0]
    return 0.5 / n * torch.sum((output - target) ** 2)


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h_prev, rho_max):
    optimizer = LBFGSBScipy(model.parameters())
    X_t = torch.from_numpy(X)
    h_new = None
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            loss = squared_loss(model(X_t), X_t)
            h = model.h_func()
            penalty = 0.5 * rho * h * h + alpha * h          # (rho/2) h^2 + alpha h
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            obj = loss + penalty + l2_reg + l1_reg
            obj.backward()
            return obj
        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h_prev:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model, X, lambda1=0., lambda2=0., max_iter=100,
                      h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W = model.fc1_to_adj()
    W[np.abs(W) < w_threshold] = 0
    return W


# Usage: W[k, j] != 0  means edge X_k -> X_j.
#   model = NotearsMLP(dims=[d, 10, 1], bias=True)
#   W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
```
