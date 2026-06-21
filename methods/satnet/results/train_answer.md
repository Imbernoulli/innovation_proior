A deep network can read a handwritten digit all day long, but the moment I ask it to *learn a rule* — a hard, discrete, all-or-nothing constraint such as "these nine cells must be a permutation of 1 through 9," or "the answer is the XOR of all these bits" — it falls apart. Not because the function is unrepresentable, but because gradient descent never finds it. Two small tasks make the gap concrete. Parity, the chained XOR of a bit string, is a known disaster for gradient-based learning: the gradient of the loss carries almost no information about *which* parity generated the data — its variance across the whole family of parities is essentially zero — so SGD wanders in a fog and an LSTM with a hundred hidden units sits at chance, around $0.476$ test error. Sudoku is the second: a large convolutional net memorizes the training boards and generalizes to roughly $0\%$ of held-out ones, and once I *permute* the bit representation so that no 2-D locality survives, it collapses even on the training set. It was learning the picture, never the logic.

So the real question is not "can a network apply a rule I hand it" — relational nets seeded with which cells may interact, and inductive-logic nets seeded with rule templates, already do that. The question is whether a network can *discover* the discrete relationships from data, end to end, with the rules unknown, and do so while sitting *inside* a larger network — behind a digit recognizer, say — so the whole stack trains together. The prior differentiable-optimization layers point the way but do not get there: OptNet's dense interior-point QP can learn the rules of a $4\times4$ Sudoku from examples, proving the concept, but its $O(n^3)$-ish solve makes negligible progress on the $9\times9$ problem even after days. And the natural object — the discrete logical primitive almost everything in symbolic AI reduces to — is satisfiability, whose objective is hostile to gradients at every turn.

I propose SATNet: a differentiable, smoothed MAXSAT layer whose learnable weight *is* the logical structure of the problem. The idea is to make a *learnable* MAXSAT instance the primitive — a clause matrix whose entries I can nudge by backprop — so that "learning the rules of Sudoku" becomes "learning the clauses." Writing variable $i$'s truth value as $\tilde v_i \in \{-1,1\}$ and its signed appearance in clause $j$ as $\tilde s_{ij} \in \{-1,0,1\}$, MAXSAT maximizes $\sum_j \bigvee_i \mathbf{1}\{\tilde s_{ij}\tilde v_i > 0\}$. Everything about that expression resists gradients: the variables are discrete, the indicator is a step, the disjunction is a max. I need a continuous, differentiable surrogate that still *means* MIN-UNSAT and that I can solve fast enough to run as a layer thousands of times.

The move is a low-rank semidefinite relaxation in the Goemans–Williamson spirit. Lift each binary $\tilde v_i \in \{\pm 1\}$ to a unit vector $v_i \in \mathbb{R}^k$, $\|v_i\| = 1$, and add one special unit "truth direction" $v_\top$; a variable is "true" by its angle to $v_\top$. The truth vector carries a coefficient $\tilde s_\top = \{-1\}^m$ in every clause, which absorbs the clause bias that a quadratic form over relative angles cannot otherwise see. The single identity that makes the whole thing differentiable is the randomized-rounding law: the probability that two unit vectors land on opposite sides of a uniformly random hyperplane is exactly proportional to the angle between them,
$$\Pr[\operatorname{sgn}(u_i^\top r) \neq \operatorname{sgn}(u_j^\top r)] = \frac{\arccos(u_i^\top u_j)}{\pi}.$$
This turns "which side of a hyperplane" into a smooth probability, and it is how I move between continuous vectors and probabilistic bits in both directions.

To check that the relaxation actually means MIN-UNSAT, take a clause of $|s_j|$ literals (counting the truth term) in the discrete case, where $V s_j$ is a scalar multiple of $v_\top$, and consider the candidate per-clause penalty $(\|V s_j\|^2 - (|s_j|-1)^2)/(4|s_j|)$. Enumerating confirms it: a single-literal clause gives penalty $-0.125$ on every satisfying assignment and $+0.375$ on the unique violating one (gap $0.5$); a two-literal clause gives $-0.25$ versus $+0.4167$ (gap $0.667$); a three-literal clause splits satisfying assignments into $-0.5625$ and $-0.3125$ but keeps the unique violator at $+0.4375$, strictly above all of them. So minimizing the sum of these penalties is, up to a per-clause additive constant and a positive scale, exactly counting violated clauses — a faithful relaxation, tight for clauses of arity $\le 2$, which is the MAX-2SAT regime. The additive constant $(|s_j|-1)^2$ does not depend on the assignment, so I drop it; folding the $1/(4|s_j|)$ weight into the clause vectors via the scaled clause matrix $S = [\,\tilde s_\top\ \tilde s_1\ \dots\ \tilde s_n\,]\cdot\mathrm{diag}(1/\sqrt{4|s_j|})$, the objective becomes $\langle S^\top S, V^\top V\rangle$. So my relaxed problem is a unit-diagonal SDP,
$$\min_{V \in \mathbb{R}^{k\times(n+1)}}\ \langle S^\top S,\ V^\top V\rangle\quad \text{subject to}\quad \|v_i\| = 1,\ i = \top,1,\dots,n,$$
and $S$ — the clause matrix — is exactly the thing I wanted to make learnable. Discovering the rules of Sudoku is now learning $S$ by gradient descent.

The dimension $k$ could have killed this, but a solvable SDP with $p$ linear constraints always has an optimal solution of rank at most $\lceil\sqrt{2p}\rceil$ (Barvinok; Pataki). With $n+1$ unit-norm constraints I may take $k \approx \sqrt{2n}$ and *still* hit the exact SDP optimum, so I set $k = \sqrt{2n}+1$: instead of $O(n^2)$ matrix entries I carry $O(nk) \approx O(n^{1.5})$ vector entries, and parameterizing $X = V^\top V$ directly makes the PSD constraint free. To solve it I use Mixing-method coordinate descent. The objective is $\operatorname{tr}(V^\top V S^\top S)$; the terms in a single column $v_i$ are $v_i^\top g_i$ plus the constant $\|s_i\|^2$, with
$$g_i = \sum_{j\neq i} s_j^\top s_i\, v_j = V S^\top s_i - \|s_i\|^2 v_i,$$
and the minimizer of an inner product against a fixed vector on the sphere is the unit vector pointing the opposite way, $v_i = -g_i/\|g_i\|$. That is the whole update — no step size, no line search, no free parameters. I maintain $\Omega = V S^\top$ and patch it after each change of one $v_i$ with a rank-one update $\Omega \mathrel{+}= (v_i^{\text{new}} - v_i^{\text{old}})\,s_i^\top$, so a full sweep is $O(nmk)$ and a handful of sweeps converge. For $k > \sqrt{2n}$ these updates provably reach the *global* SDP optimum: the bad critical points of the non-convex $V$-formulation are unstable and coordinate descent slides off them.

The layer takes shape around known indices $I$ (the clues) and unknown indices $O$ (the cells to fill). A clue probability $z_i \in [0,1]$ enters as a vector by requiring its truth-angle to match: I want $\Pr[i\ \text{true}] = \arccos(-v_i^\top v_\top)/\pi = z_i$, i.e. $v_i^\top v_\top = -\cos(\pi z_i)$, which I build explicitly as $v_i = -\cos(\pi z_i)\,v_\top + \sin(\pi z_i)\,(I_k - v_\top v_\top^\top)v_i^{\text{rand}}$ — unit norm because the two orthogonal pieces have squared norms $\cos^2$ and $\sin^2$. So $z_i=0$ points $v_i$ opposite $v_\top$ (false), $z_i=1$ along it (true), and intermediates interpolate the angle; a soft $0.7$ from a digit classifier flows straight in. Coordinate descent runs over the output columns only, with the clue and truth columns fixed, and each output is read back out by the same law, $z_o = \arccos(-v_o^\top v_\top)/\pi$. At test time I may instead round — draw random hyperplanes, set $\tilde v_o$ true iff $\operatorname{sgn}(r^\top v_o) = \operatorname{sgn}(r^\top v_\top)$, repeat, and keep the assignment satisfying the most clauses — but during training I never round, passing the smooth probability forward so everything stays differentiable.

The crux is backpropagation. The first hop, probability to vector, follows from $\cos(\pi z_o) = -v_o^\top v_\top$: differentiating gives $\partial \ell/\partial v_o = (\partial \ell/\partial z_o)\,v_\top/(\pi \sin(\pi z_o))$. For the dependence of $\ell$ on $S$ and the input vectors *through the SDP solution*, unrolling the sweeps is a wall — it stores every intermediate $V$ and Jacobian. Instead I differentiate the *fixed point*. At convergence $v_o = -g_o/\|g_o\|$, so multiplying through by $\|g_o\|$ and taking the total differential, with $d\|g_o\| = -v_o^\top dg_o$, yields $\|g_o\|\,dv_o = -(I_k - v_o v_o^\top)\,dg_o$. The projector $P_o = I_k - v_o v_o^\top$ onto the tangent space of the sphere appears naturally. Expanding $dg_o$ and collecting the coupled output differentials gives, for every $o\in O$, $(\|g_o\| I_k - \|s_o\|^2 P_o)\,dv_o + P_o\sum_{j\in O}(s_o^\top s_j)\,dv_j = -P_o\,\xi_o$, where $\xi_o$ gathers the input and $dS$ contributions. Stacking and vectorizing, with $C = S_O^\top S_O - \mathrm{diag}(\|s_o\|^2)$, $D = \mathrm{diag}(\|g_o\|)$, and the block projector $P = \mathrm{diag}(P_o)$, the system is $(D\otimes I_k + P C\otimes I_k)\,\mathrm{vec}(dV_O) = -P\,\mathrm{vec}(\xi_o)$. The raw operator is non-symmetric and $P$ is singular, but every $dv_o$ lies in the range of $P_o$, so substituting $\mathrm{vec}(dV_O) = P\,\mathrm{vec}(Y)$ and using $PP=P$ and the blockwise commutation of the diagonal $D$ collapses it to the symmetric, well-defined form $\mathrm{vec}(dV_O) = -\big(P((D+C)\otimes I_k)P\big)^\dagger \mathrm{vec}(\xi_o)$.

Because that operator is symmetric I never form it: I move it onto the upstream gradient. Define $U$ by $U_I = 0$ and $\mathrm{vec}(U_O) = \big(P((D+C)\otimes I_k)P\big)^\dagger \mathrm{vec}(\partial\ell/\partial V_O)$, and then, since $\xi_o$ is linear in the input perturbations and in $dS$, the gradients fall out by picking off coefficients:
$$\frac{\partial \ell}{\partial V_I} = -\Big(\sum_{o\in O} u_o s_o^\top\Big) S_I,\qquad \frac{\partial \ell}{\partial S} = -\Big(\sum_{o\in O} u_o s_o^\top\Big)^{\!\top} V \;-\; (S V^\top) U.$$
The input probabilities close the loop through $\partial v_i/\partial z_i = \pi\big(\sin(\pi z_i)v_\top + \cos(\pi z_i)(I_k - v_\top v_\top^\top)v_i^{\text{rand}}\big)$ and $\partial\ell/\partial z_i = \partial\ell/\partial z_i^{\text{direct}} + (\partial v_i/\partial z_i)^\top(\partial\ell/\partial v_i)$. I verified these against finite differences on a tiny instance — six-dimensional vectors, four variables, one clue, two outputs — and the analytic and numerical gradients agree to about $10^{-12}$, signs and all. The decisive structural fact is that the backward linear system shares the forward solve's $C$, projectors, and low-rank $S$, so I solve it with the *same* coordinate descent: maintaining $\Psi = U_O S_O^\top$, with $dg_o = \Psi s_o - \|s_o\|^2 u_o - \partial\ell/\partial v_o$, the closed-form update is $u_o = -P_o\,dg_o/(\|g_o\| + \lambda)$, again with rank-one patches to $\Psi$. The whole backward pass costs the same $O(nmk)$ per sweep as the forward, with no stored Jacobians. The small proximal $\lambda$ (the diagonal increment $\mathrm{prox\_lam}$) guards the case $\|g_o\|\to 0$, where $u_o$ and the readout $1/(\pi\sin(\pi z_o))$ would otherwise blow up; anything still non-finite I zero out rather than poison the batch.

Two design levers decide whether this *works* on Sudoku rather than merely existing. The first is that the number of clauses $m$ — the rank of $S$ — is the layer's capacity, and the instinct to set it large is exactly the ConvNet failure: too much capacity memorizes instead of finding the rule. Low rank is not just a speed trick, it is the regularizer; with few clauses the layer literally cannot encode a lookup table and is forced to discover a compact rule set, so for $9\times9$ Sudoku ($n = 9^3 = 729$ bit-variables) I keep $m = 600$. The second is auxiliary variables: extra latent columns of $V$ connected to neither inputs nor outputs act like CNF "register memory," exponentially shrinking the number of clauses needed to express a relation, so I add $\mathrm{aux} = 300$ to raise expressive power without raising $m$. They participate in the solve but are never read out. With this — $m = 600$, $\mathrm{aux} = 300$ — a single layer learns all the row, column, and block constraints and generalizes to $98\%$+ on held-out *and* permuted boards, the permuted case being the real test since no locality remains to cheat with.

The released solver implements `init / forward / backward` in C++/CUDA behind a `torch.autograd.Function`; the inner coordinate-descent kernel is shared between the forward pass (maintaining $W = V^\top S$) and the backward pass (maintaining $\Phi = U^\top S$). The Python interface is:

```python
import torch
import torch.nn as nn
from torch.autograd import Function

import satnet._cpp
if torch.cuda.is_available():
    import satnet._cuda


def get_k(n):
    return int((2 * n) ** 0.5 + 3) // 4 * 4          # rank >= sqrt(2n)+1, mult. of 4


class MixingFunc(Function):
    """Apply the Mixing method (low-rank SDP coordinate descent) to input probs.

    The SATNet module is a wrapper handling init and the auxiliary/truth columns.
    """
    @staticmethod
    def forward(ctx, S, z, is_input, max_iter, eps, prox_lam):
        B, n, m, k = z.size(0), S.size(0), S.size(1), 32
        ctx.prox_lam = prox_lam
        dev = 'cuda' if S.is_cuda else 'cpu'

        ctx.g, ctx.gnrm = torch.zeros(B, k, device=dev), torch.zeros(B, n, device=dev)
        ctx.index = torch.zeros(B, n, dtype=torch.int, device=dev)
        ctx.is_input = torch.zeros(B, n, dtype=torch.int, device=dev)
        ctx.V = torch.zeros(B, n, k, device=dev).normal_()       # random unit init
        ctx.W = torch.zeros(B, k, m, device=dev)                 # W = V^T S
        ctx.z = torch.zeros(B, n, device=dev)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=dev)
        ctx.S = torch.zeros(n, m, device=dev)
        ctx.Snrms = torch.zeros(n, device=dev)

        ctx.z[:] = z.data
        ctx.S[:] = S.data
        ctx.is_input[:] = is_input.data

        perm = torch.randperm(n - 1, dtype=torch.int, device=dev)
        impl = satnet._cuda if S.is_cuda else satnet._cpp
        # init: relax inputs into unit vectors (truth angle), set update order
        impl.init(perm, is_input, ctx.index, ctx.z, ctx.V)
        for b in range(B):
            ctx.W[b] = ctx.V[b].t().mm(ctx.S)
        ctx.Snrms[:] = S.norm(dim=1) ** 2

        # forward coordinate descent: v_o = -g_o/||g_o||, rank-1 update of W
        impl.forward(max_iter, eps,
                     ctx.index, ctx.niter, ctx.S, ctx.z,
                     ctx.V, ctx.W, ctx.gnrm, ctx.Snrms, ctx.g)
        return ctx.z.clone()

    @staticmethod
    def backward(ctx, dz):
        B, n, m, k = dz.size(0), ctx.S.size(0), ctx.S.size(1), 32
        dev = 'cuda' if ctx.S.is_cuda else 'cpu'
        ctx.dS = torch.zeros(B, n, m, device=dev)
        ctx.U = torch.zeros(B, n, k, device=dev)                 # backward vectors
        ctx.Phi = torch.zeros(B, k, m, device=dev)              # Phi = U^T S
        ctx.dz = torch.zeros(B, n, device=dev)
        ctx.dz[:] = dz.data

        impl = satnet._cuda if ctx.S.is_cuda else satnet._cpp
        # backward: solve the fixed-point linear system by the SAME coord. descent,
        # then assemble dS = U W + V Phi and the input-probability gradients.
        impl.backward(ctx.prox_lam,
                      ctx.is_input, ctx.index, ctx.niter, ctx.S, ctx.dS, ctx.z, ctx.dz,
                      ctx.V, ctx.U, ctx.W, ctx.Phi, ctx.gnrm, ctx.Snrms, ctx.g)
        ctx.dS = ctx.dS.sum(dim=0)
        return ctx.dS, ctx.dz, None, None, None, None


def insert_constants(x, pre, n_pre, app, n_app):
    one = x.new(x.size()[0], 1).fill_(1)
    seq = []
    if n_pre != 0:
        seq.append((pre * one).expand(-1, n_pre))
    seq.append(x)
    if n_app != 0:
        seq.append((app * one).expand(-1, n_app))
    r = torch.cat(seq, dim=1)
    r.requires_grad = False
    return r


class SATNet(nn.Module):
    """A differentiable MAXSAT layer; the clause matrix S is the learned logic.

    Args:
        n: number of input variables.
        m: rank of the clause matrix (number of clauses) -- low for generalization.
        aux: number of auxiliary ("register memory") variables.
        max_iter: max coordinate-descent iterations (default 40).
        eps: relative stopping threshold (default 1e-4).
        prox_lam: diagonal increment stabilizing the backward solve (default 1e-2).
    """
    def __init__(self, n, m, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2,
                 weight_normalize=True):
        super().__init__()
        S = torch.FloatTensor(n + 1 + aux, m).normal_()         # +1 = truth vector
        if weight_normalize:
            S = S * ((0.5 / (n + 1 + aux + m)) ** 0.5)          # small init
        self.S = nn.Parameter(S)
        self.aux = aux
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        B = z.size(0)
        dev = 'cuda' if self.S.is_cuda else 'cpu'
        m = self.S.shape[1]
        if dev == 'cpu' and m % 4 != 0:
            raise ValueError('m must be a multiple of 4 on CPU (SSE). Got ' + str(m))
        # prepend always-true truth variable; append unconstrained aux variables
        is_input = insert_constants(is_input.data, 1, 1, 0, self.aux)
        z = torch.cat([torch.ones(B, 1, device=dev), z,
                       torch.zeros(B, self.aux, device=dev)], dim=1)
        z = MixingFunc.apply(self.S, z, is_input,
                             self.max_iter, self.eps, self.prox_lam)
        return z[:, 1:self.S.size(0) - self.aux]                # strip truth + aux
```

To use it for learning $9\times9$ Sudoku from examples, instantiate one SATNet layer with $n = 9^3 = 729$, $m = 600$, $\mathrm{aux} = 300$, trained with Adam (lr $2\times10^{-3}$) under digit-wise NLL / binary cross-entropy. For visual Sudoku, feed the cell-wise softmax outputs of a LeNet digit classifier as the probabilistic input $z$ and train the whole stack end to end, with separate learning rates for the perception and logic parts. For parity of a length-$L$ string, chain $L-1$ tied SATNet layers — each computing one XOR — with single-bit supervision at the end.

```python
class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super().__init__()
        self.sat = SATNet(boardSz ** 6, m, aux)     # boardSz=3 -> n = 729

    def forward(self, y_in, mask):
        return self.sat(y_in, mask)
```
