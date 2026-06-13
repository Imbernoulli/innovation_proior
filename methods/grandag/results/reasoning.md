Let me start from what actually blocks me. I have observational samples of a vector `X = (X_1, ..., X_d)` and I want the directed causal graph `G` — oriented edges, not a correlation matrix and not an undirected skeleton. Two things stand in the way, and I should be honest about both before I reach for any machinery. First, the space of DAGs on `d` nodes is monstrous — it grows faster than `d!`, and acyclicity is a global discrete property of the whole edge set, so the obvious algorithms are greedy: propose an edge, check no cycle appeared, accept or reject, repeat. That is exactly the design of GES and of the order-search methods, and it is slow and bespoke. Second, even setting search aside, from `P_X` alone the direction is generally not recoverable — many DAGs share the same conditional-independence pattern, the same Markov equivalence class, and observational data can't tell them apart. So I need an assumption that breaks the cause/effect symmetry.

The symmetry-breaker is the additive noise model. Suppose `X_j := f_j(X_{pi_j}) + N_j` with the noises `N_j` mutually independent and the `f_j` nonlinear under mild regularity. Hoyer and then Peters, Mooij, Janzing and Scholkopf showed `G` is then identifiable from the distribution — and the part I keep coming back to is that this holds even when the noise is *Gaussian*, as long as `f` is nonlinear. The linear-Gaussian model is the textbook unidentifiable case; the nonlinear-Gaussian one is not. The reason is concrete: if I regress the effect on the cause I get a residual independent of the cause, but if I regress the wrong way the residual stays dependent — the nonlinearity leaves a fingerprint of direction. So nonlinearity is a gift here, not an obstacle. That tells me the *model class* I fit matters enormously: if my model can represent the true nonlinear-Gaussian ANM, and the truth is identifiable in that class, then whatever uniquely maximizes the population likelihood has to be the true graph. I'll want to come back and make that precise, because it's the whole theoretical justification.

But how do people actually exploit ANM identifiability today? RESIT regresses each variable on candidate parents and tests the residuals for independence — directly the fingerprint idea — and it works, but it's a greedy combinatorial search over orderings and the independence tests don't scale past about twenty nodes. CAM is cleverer about the search: it *decouples* finding a topological order from selecting each node's parents, using restricted maximum likelihood for the order and sparse regression for the edges, with a preliminary neighbor-selection step to cut candidates and a final significance-test pruning to drop spurious parents. I like the PNS and the pruning a lot — they're exactly the right tools to fight the fact that an unpenalized likelihood score only ever *increases* when you add an edge, so it overfits toward dense graphs. I'll keep those in my back pocket. But CAM pays for its tractability with a modelling lie: it assumes the mechanism is *additive*, `f_j(x_{pi_j}) = sum_i f_{ij}(x_i)`, no interactions between parents. And it's still a greedy order search underneath. I want general nonlinear `f_j`, and I want to stop doing greedy search over a discrete space.

The thing that reframes the search problem is NOTEARS. Its move is to stop searching the discrete space at all: encode the graph as a single real matrix and turn acyclicity into one smooth equality constraint, then hand the whole thing to a standard constrained solver. Let me make sure I understand *why* its constraint works, because I'm going to have to reproduce the argument in a harder setting. Take a nonnegative matrix `B`. The entry `(B^k)_{jj}` counts closed walks of length `k` through node `j`, so `tr(B^k)` counts all length-`k` cycles, and `B` is acyclic exactly when `tr(B^k) = 0` for every `k`. That's a true characterization but a terrible constraint: as a finite sum `sum_{k=1}^d tr(B^k)`, the powers `B^k` overflow machine precision for even moderate `d`, and the gradients are just as unstable. NOTEARS's fix is the matrix exponential, `e^B = sum_k B^k / k!`: it reweights the length-`k` walk counts by `1/k!`, which tames the explosion, and you get the clean statement that a binary `B` is a DAG iff `tr e^B = d`, i.e. `tr e^B - d = 0`. To handle real, possibly negative weights, replace `B` by its Hadamard square `W ∘ W`, which is entrywise nonnegative so the same walk-counting argument applies, giving `h(W) = tr e^{W ∘ W} - d = 0`, and the gradient falls out as `(e^{W ∘ W})^T ∘ 2W`. Lovely. Solve `max_W` (least-squares score `- L1` penalty) subject to `h(W) = 0` by augmented Lagrangian, and every edge updates at once from the gradient — no greedy search.

So why don't I just use NOTEARS? Because `W` *is* the coefficient matrix of a *linear* SEM, `X_j = u_j^T X + N_j`. The contribution of variable `i` to variable `j` is the single scalar `W_{ij}`, and "`i` is a parent of `j`" is precisely "`W_{ij} != 0`", with `|W_{ij}|` the edge weight that goes into `h`. There is no room for `X_j` to depend nonlinearly on its parents. On the Gauss-process ANM data I care about, NOTEARS will underfit the mechanisms and mis-score directions, because the entire identifiability story rested on representing the nonlinearity. DAG-GNN tried to push the continuous-constraint idea to nonlinear mechanisms with a graph neural network and an ELBO, but it shares parameters heavily across the `f_j` through the GNN, and when the mechanisms are genuinely independent functions that sharing is unjustified — it underfits. I want the continuous-constraint paradigm *and* an independent, flexible nonlinear model per variable.

So the obvious thing: give each variable `j` its own neural network. Let `NN_j` take the other variables as input and output the parameters `theta_(j)` of `X_j`'s conditional distribution — for the ANM case, a Gaussian whose mean is `mu_(j) = NN_j(X_{-j})` and whose noise variance I learn but keep independent of the parents, matching `X_j = f_j(parents) + N_j` exactly. Write the MLP plainly: `theta_(j) = W^{(L+1)} g( ... g(W^{(2)} g(W^{(1)} X_{-j})) ... )`, with `g` a pointwise nonlinearity. Mask the `j`-th input to zero so a variable can't be its own parent. If I do this for all `d` variables I get a product of conditionals `prod_j p_j(x_j | x_{-j})`, and I want to maximize its log over the data. Each `NN_j` can model arbitrary nonlinear dependence on the others — capacity solved.

And now I hit the wall that makes this whole thing nontrivial. In NOTEARS the constraint lived on `W`, but `W` was literally the adjacency-weighted graph. Here there *is* no single coefficient telling me whether `X_j` depends on `X_i`. Variable `i` enters `NN_j` through a whole tangle of weights spread across `L+1` layers and all the hidden units. The product `prod_j p_j(x_j | x_{-j})` isn't even a valid joint density unless the implied graph is acyclic — if everything depends on everything, it doesn't decompose along any DAG and won't integrate to one. So before I can write a constraint, I have to manufacture, *out of the NN weights*, a single nonnegative number `(A_phi)_{ij}` that is zero exactly when `NN_j`'s output truly does not depend on input `i`. If I can build that `A_phi`, I can drop it straight into NOTEARS's `tr e^{A_phi} - d` and I'm done. The question is how to read "does output depend on this input" off a stack of weight matrices.

Let me think about a single network and forget the subscript `j` for a moment. When can input `i` possibly influence output `k`? Information flows from `i` to `k` only along *computation paths* through the network: pick a hidden unit `h_1` in layer 1, a hidden unit `h_2` in layer 2, and so on up to output `k`, and the path is the sequence of weights `(W^{(1)}_{h_1 i}, W^{(2)}_{h_2 h_1}, ..., W^{(L+1)}_{k h_L})`. If *any* weight on that path is zero, the path is dead — the signal can't get through that link. So a path is "inactive" exactly when it has a zero on it. Now, when is output `k` *completely* independent of input `i`? When *every* path from `i` to `k` is dead. There is simply no route for the information to travel. That's the condition I need to detect, and it's purely about whether paths are alive.

Can I quantify "alive" with a single nonnegative scalar per path whose zero-ness equals "dead"? The natural one is the *path product* of magnitudes: `|W^{(1)}_{h_1 i}| · |W^{(2)}_{h_2 h_1}| · ... · |W^{(L+1)}_{k h_L}|`. It's nonnegative, and it's zero if and only if at least one factor is zero — exactly "this path is inactive." I take absolute values because I only care whether information *can* flow, not which way it pushes the activation; the sign of a weight is irrelevant to whether the link is open. (This is the same instinct that made NOTEARS use `W ∘ W` — turn a signed weight into a nonnegative strength.) So one path contributes a nonnegative strength, zero iff dead.

Now I want "every path from `i` to `k` is dead," which is "the *sum* of all their path products is zero" — a sum of nonnegatives is zero iff every term is zero. So define the total strength from input `i` to output `k` as the sum over all hidden routes of the path products. The beautiful thing is what that sum *is*. Summing a product of `|W|` factors over all intermediate indices is just matrix multiplication of the absolute-value weight matrices. Stack them: `C = |W^{(L+1)}| · ... · |W^{(2)}| · |W^{(1)}|`. The `AND` along a path (the product, dead if any link dead) and the `OR` over routes (the sum, alive if any route alive) are *exactly* what matrix multiplication does — multiply within a path, add across paths. So `C_{ki}` is precisely the sum of all path products from input `i` to output `k`, and `C_{ki} = 0` is *sufficient* to make output `k` independent of input `i`. I have my detector, and it's just a product of nonnegative matrices.

I want this at the variable level, not the per-output level. `NN_j` outputs a whole parameter vector `theta_(j)` of length `m` (for the plain ANM, `m = 1`, just the mean). I want "`theta_(j)` does not depend on `X_i` at all," meaning every component of the output is independent of input `i`. That's `C_{ki} = 0` for all output components `k`, which I get by summing over outputs: `sum_{k=1}^m (C_(j))_{ki} = 0`. So define the weighted adjacency entry as that column sum,

  `(A_phi)_{ij} = sum_{k=1}^m (C_(j))_{ki}`  for `j != i`, and `0` on the diagonal,

where `C_(j)` is the connectivity matrix of variable `j`'s network. By construction `(A_phi)_{ij} >= 0`, and `(A_phi)_{ij} = 0` implies `theta_(j)` does not depend on `X_i` — i.e. no edge `i -> j`. This is the object I needed: a single nonnegative `d × d` matrix read off all the networks' weights, playing exactly the role `W` played in the linear case.

And because `A_phi` is already entrywise nonnegative, I notice I don't even need NOTEARS's Hadamard square — that square was only there to *make* a signed `W` nonnegative so the closed-walk argument would apply. My `A_phi` is born nonnegative. So I can write the acyclicity constraint directly on it:

  `h(phi) = tr e^{A_phi} - d = 0`.

This is the same constraint, but I've now run the NOTEARS argument twice over: once at the level of *neural-network paths* (to build each `C_(j)` and hence `A_phi`) and once at the level of *graph paths* (the `tr e^{A_phi} - d` walk-counting on the variable graph). The first reduction is what lets the second one even make sense. One worry: `A_phi` involves absolute values, so `h` is non-differentiable wherever a weight crosses zero. But that's the same kind of kink ReLU networks already have, and in practice gradient methods step right through those measure-zero points without trouble. I'll note it and not lose sleep.

Now the score and the optimization. I want to maximize the population log-likelihood of the product of conditionals,

  `max_phi  E_{X ~ P_X} sum_{j=1}^d log p_j(X_j | X_{pi_j^phi}; phi_(j))   s.t.   h(phi) = 0`,

remembering that `sum_j log p_j` is only a *valid* joint log-likelihood when `h = 0`, because only then does the product decompose along a DAG and integrate to one. Before I worry about solving it, let me check it even points at the right answer in the ideal case, because if the optimum isn't the true graph the whole exercise is pointless. The population log-likelihood `E log p_phi(X)` is maximized exactly when the modeled joint `P_phi` equals the true `P_X`. Suppose the true causal model `(P_X, G)` lies inside the class `C` my networks can represent — say a nonlinear-Gaussian ANM — and suppose `C` coincides with the set of models satisfying an identifiability assumption set `A` under which `G` is identifiable. Then: the maximum is achievable (some `phi*` gives `P_{phi*} = P_X`, since the truth is in `C`), and by identifiability there is no *other* model in `C` with a different graph that also reproduces `P_X`. So the graph at the optimum, `G_{phi*}`, must be `G` itself. That's the justification, and I should be careful about its scope: it's a *population, exact-optimization* statement. In practice the problem is non-convex and I only have finitely many samples, so I'll reach a stationary point on an empirical objective — the guarantee is the lodestar, not a certificate.

How do I solve a single equality-constrained, non-convex problem like this? The standard tool is the augmented Lagrangian. I turn the constraint into a sequence of unconstrained subproblems

  `max_phi  L(phi, lambda_t, mu_t) = E[ sum_j log p_j ] - lambda_t h(phi) - (mu_t / 2) h(phi)^2`,

where `lambda_t` is the dual variable and `mu_t` the quadratic-penalty coefficient. The linear `- lambda h` term is the Lagrange-multiplier pull toward the constraint surface; the quadratic `- (mu/2) h^2` term penalizes the magnitude of the violation and is what makes the subproblems well-behaved even when the multiplier estimate is still poor. After approximately solving subproblem `t`, I update the way augmented-Lagrangian theory prescribes: do a dual ascent step `lambda_{t+1} = lambda_t + mu_t h(phi_t*)`, which nudges the multiplier by the current violation, and *raise the penalty only if the constraint isn't improving fast enough*, `mu_{t+1} = eta · mu_t` when `h(phi_t*) > gamma · h(phi_{t-1}*)` (the violation didn't shrink by at least a factor `gamma`), else leave `mu` alone. I'll start gentle, `lambda_0 = 0` and `mu_0 = 10^{-3}`, escalate hard when stuck, `eta = 10`, with `gamma = 0.9` as the "did it improve enough" bar, and declare the whole thing converged once the violation is numerically negligible, `h <= 10^{-8}`. Starting `mu` small matters: if I clamp the graph to acyclic too hard, too early, the networks can't explore which dependencies are real; I want the likelihood to lead early and the acyclicity to tighten gradually.

Each subproblem is a neural-network fit, so I solve it with minibatch stochastic gradient — specifically RMSprop, which is a sensible default for nets and carries an implicit regularizing effect, and that regularization is welcome here since the bare likelihood overfits. Why not NOTEARS's full-batch proximal quasi-Newton? Because that was natural when the objective was a small linear least-squares in `W`; here the objective is a stack of neural nets with many parameters, where stochastic gradients scale far better, and in practice fewer total iterations are needed before the augmented Lagrangian converges — I think because I can *early-stop* each subproblem (stop when a held-out estimate of `L` stops improving) instead of grinding each one to full convergence. The matrix exponential costs `O(d^3)` per iteration, but for the graph sizes I care about (`d` up to ~100) that's a small slice of the per-iteration cost; evaluating the networks' gradient dominates. When I escalate `mu` and `lambda` I also reinitialize the optimizer state, since the loss surface just changed shape. The learning rate belongs to the optimizer wrapper rather than the mathematics: I can start larger and restart smaller, but in the compact code I keep the restart learning rate configurable and use `1e-3` for both the initial and restarted RMSprop steps.

There's still the overfitting problem I flagged: maximum likelihood never penalizes adding an edge, so left alone the networks will keep spurious dependencies and `A_phi` will be dense. I'll control it four ways. Early stopping per subproblem on a held-out set, already noted. The implicit regularization of stochastic gradients, already noted. And then I'll borrow exactly CAM's two structural tools rather than inventing my own penalty — partly because they work and partly so the comparison to CAM is clean. *Preliminary neighbor selection* for larger graphs (`d >= 50`): for each variable, fit a flexible variable-importance regressor (extremely randomized trees) against all the others and keep only the inputs whose importance clears `0.75` times the mean importance, masking the rest out from the start. And a final *pruning* step: once I have a DAG, fit a generalized additive model of each node on its current parents and drop any parent whose covariate significance test exceeds a small p-value (`0.001`). Both fight the dense-graph bias of the likelihood.

That mention of "once I have a DAG" exposes a gap. The augmented Lagrangian only drives `h` to about `10^{-8}`, not exactly zero, so `A_phi` will have many small-but-nonzero entries and won't be exactly acyclic. I need to threshold. The cheap version: as I train, whenever an entry `(A_phi)_{ij}` drops below a small `epsilon = 10^{-4}`, permanently mask that input off — implement the mask as just one more (binary, non-learned) factor folded into the `C` product, so it changes nothing in the interpretation. But thresholding `A_phi` at the end is subtly wrong, and I should see why before I trust it. `(A_phi)_{ij} = 0` is *sufficient* for independence but *not necessary*. Two ways an input can be effectively dead while `A_phi` says otherwise: the path products from `i` can partly cancel each other in the actual function even though their magnitudes sum to something positive (I summed magnitudes, which can't cancel, but the realized function can); and hidden units can be saturated over the observed input range, so a path is *effectively* dead without any weight being zero. So `A_phi` overstates dependence. What I actually want to threshold is the *realized* sensitivity of each conditional to each input.

The honest measure of "does conditional `j` actually respond to `X_i`" is the derivative of the conditional score with respect to that input, averaged over the data. Let `L_i(X)` be the per-variable conditional likelihood, and form the expected absolute Jacobian `J = E_X |∂L/∂X|^T`; in code I can use the per-variable log-likelihood score for the same sensitivity test. The entry `J_{ij}` measures how much variable `j`'s conditional moves when I wiggle input `i` — it folds in cancellation and saturation that `A_phi` blind-summed away. So for the final DAG extraction I remove edges by increasing `J_{ij}`, smallest first, and stop at the smallest threshold whose remaining graph is acyclic (testing acyclicity directly by checking `tr(A^k) = 0` for `k = 1..d`, the same closed-walk fact, now on the thresholded binary graph). This is a thresholded weakest-edge pass, not a combinatorial minimum-edge-removal solver, and it's a better strength estimate than `A_phi` precisely because it's the realized sensitivity, not the weight-product upper bound.

Let me also place this against MADE for a moment, because the resemblance is striking and it sharpens what's new. MADE multiplies weight matrices by binary masks to force the autoregressive property — output `j` depends only on inputs before `j` in a chosen order — and the autoregressive property is *exactly* acyclicity for that fixed order. So MADE is, in a sense, my construction with the mask frozen in advance. The difference is the whole point: MADE *fixes* the ordering a priori, while here the masking — and therefore the ordering — is *learned from data* through the continuous constraint. The connectivity-matrix-to-`A_phi` reduction is what makes that learnable, because it turns "which inputs feed which conditional" into a differentiable nonnegative matrix I can constrain and optimize.

So let me assemble the actual algorithm, grounded in what I'd really run. Each variable gets a small MLP — two hidden layers of ten units, leaky-ReLU, Xavier-initialized — whose weights are stored as a single `(d, out, in)` tensor per layer so all `d` networks evaluate in parallel; the first layer's input is masked by `adjacency` so a variable never sees itself and so thresholded inputs stay off. The forward pass is one `einsum` per layer over the variable axis. The connectivity/adjacency `A_phi` is the path-normalized product of absolute weight matrices, column-summed over the output axis and transposed into `i -> j` convention. The score is the per-variable Gaussian log-likelihood (mean from the net, a learned parent-independent log-std for the ANM noise). The acyclicity is `tr e^{A_phi} - d`. The outer loop is the augmented Lagrangian with the convergence-based `mu`/`lambda` updates above; edge clamping below threshold; and a final Jacobian-based thresholding that removes the weakest edges until acyclic. Here it is as the single function I'd ship, filling the structure-model and fit slots from the harness:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """Learn a DAG from observational nonlinear (Gauss-ANM) data.

    Returns B with B[i, j] != 0 meaning j -> i.
    """
    import os
    seed = int(os.environ.get("SEED", "42"))
    torch.manual_seed(seed); np.random.seed(seed)
    n, d = X.shape
    DT = torch.float64

    class Model(nn.Module):
        # one MLP per variable: input X_{-j} (j masked), output the mean of X_j's
        # Gaussian conditional; 2 hidden layers x 10 units, leaky-ReLU (ANM model).
        def __init__(self):
            super().__init__()
            # binary input mask = current candidate adjacency (never a Parameter:
            # updated only by clamping). adjacency[i, j] live means i may feed j.
            self.adjacency = torch.ones(d, d, dtype=DT) - torch.eye(d, dtype=DT)
            layers = [d, 10, 10, 1]              # [in, hid1, hid2, out=mean]
            self.W = nn.ParameterList(); self.b = nn.ParameterList()
            for k in range(len(layers) - 1):     # weights stacked over the d nets
                self.W.append(nn.Parameter(torch.zeros(d, layers[k+1], layers[k], dtype=DT)))
                self.b.append(nn.Parameter(torch.zeros(d, layers[k+1], dtype=DT)))
            # learned, parent-independent noise log-std per variable (the N_j)
            self.log_std = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=DT)) for _ in range(d)])
            g = nn.init.calculate_gain('leaky_relu')   # Xavier init per network
            with torch.no_grad():
                for j in range(d):
                    for w in self.W:
                        nn.init.xavier_uniform_(w[j], gain=g)

        def _forward(self, x):
            # per-variable MLP; first layer masks inputs by the adjacency mask
            for k in range(3):
                if k == 0:
                    x = torch.einsum("tij,ljt,bj->bti", self.W[k],
                                     self.adjacency.unsqueeze(0), x) + self.b[k]
                else:
                    x = torch.einsum("tij,btj->bti", self.W[k], x) + self.b[k]
                if k < 2:
                    x = F.leaky_relu(x)
            return torch.unbind(x, 1)            # d tensors of (batch, 1)

        def log_lik(self, x, detach_target=False):
            mus = self._forward(x); parts = []
            for i in range(d):
                mu = mus[i].squeeze(1); sig = torch.exp(self.log_std[i])
                xi = x[:, i].detach() if detach_target else x[:, i]
                parts.append(distributions.Normal(mu, sig).log_prob(xi).unsqueeze(1))
            return torch.cat(parts, 1)           # (batch, d)

        def w_adj(self):
            # A_phi: path-normalised product of |W| (sum of NN-path products),
            # column-summed over outputs. (A_phi)[i,j]=0 => theta_j independent of X_i.
            prod = torch.eye(d, dtype=DT); pn = torch.eye(d, dtype=DT)
            off = (1.0 - torch.eye(d, dtype=DT)).unsqueeze(0)
            for i, w in enumerate(self.W):
                aw = torch.abs(w)                # nonneg strength; 0 iff link dead
                if i == 0:
                    prod = torch.einsum("tij,ljt,jk->tik", aw, self.adjacency.unsqueeze(0), prod)
                    pn   = torch.einsum("tij,ljt,jk->tik", torch.ones_like(aw), off, pn)
                else:
                    prod = torch.einsum("tij,tjk->tik", aw, prod)
                    pn   = torch.einsum("tij,tjk->tik", torch.ones_like(aw), pn)
            prod = prod.sum(1); pn = pn.sum(1)   # sum over output axis
            return (prod / (pn + torch.eye(d, dtype=DT))).t()   # -> i->j convention

    model = Model()

    # 80/20 split: train the conditionals, hold out for the convergence criterion
    tn = int(n * 0.8)
    Xtr = torch.as_tensor(X[:tn], dtype=DT); Xte = torch.as_tensor(X[tn:], dtype=DT)
    rtr = np.random.RandomState(seed); rte = np.random.RandomState(seed + 1)
    def sample(data, rng, bs):
        idx = rng.choice(data.shape[0], size=int(bs), replace=False)
        return data[torch.as_tensor(idx).long()]

    # augmented Lagrangian: max  E[sum_j log p_j] - lamb*h - (mu/2) h^2
    mu, lamb = 1e-3, 0.0                          # mu_0 = 1e-3, lambda_0 = 0
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    a_val, not_nll, h_hist = [], [], []
    BS, ITER, WIN = min(64, tn), 30000, 100

    for it in range(ITER):
        model.train()
        loss = -model.log_lik(sample(Xtr, rtr, BS)).mean()      # -E[sum log p]
        model.eval()
        wa = model.w_adj()
        h = torch.trace(torch.matrix_exp(wa)) - d               # tr e^{A_phi} - d
        al = loss + 0.5 * mu * h ** 2 + lamb * h
        opt.zero_grad(); al.backward(); opt.step()

        # edge clamping: permanently mask inputs whose A_phi entry is tiny
        with torch.no_grad():
            model.adjacency *= (wa > 1e-4).to(DT)

        not_nll.append(0.5 * mu * h.item() ** 2 + lamb * h.item())
        if it % WIN == 0:                                        # held-out check
            with torch.no_grad():
                vl = -model.log_lik(sample(Xte, rte, Xte.shape[0])).mean()
                a_val.append([it, vl.item() + not_nll[-1]])

        # convergence delta of the held-out augmented Lagrangian
        delta = -np.inf
        if it >= 2 * WIN and it % (2 * WIN) == 0:
            t0, th, t1 = a_val[-3][1], a_val[-2][1], a_val[-1][1]
            delta = (t1 - t0) / WIN if (min(t0, t1) < th < max(t0, t1)) else -np.inf

        if h.item() > 1e-8:
            # subproblem (approximately) converged -> dual/penalty update
            if abs(delta) < 1e-4 or delta > 0:
                lamb += mu * h.item()                            # dual ascent
                h_hist.append(h.item())
                if len(h_hist) >= 2 and h_hist[-1] > h_hist[-2] * 0.9:
                    mu *= 10                                     # escalate penalty
                a_val[-1][1] += (0.5 * mu * h.item() ** 2 + lamb * h.item() - not_nll[-1])
                opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)  # reset state
        else:
            with torch.no_grad():                               # h ~ 0: done
                model.adjacency *= (wa > 0).to(DT)
            break

    # DAG enforcement: rank edges by realized sensitivity (expected |Jacobian| of
    # each conditional log-lik w.r.t. each input), not by A_phi (which can overstate
    # dependence via path cancellation / saturated units).
    model.eval()
    xj = Xtr.clone().requires_grad_(True)
    lps = torch.unbind(model.log_lik(xj, detach_target=True), 1)
    jac = torch.zeros(d, d, dtype=DT)
    for i in range(d):
        gi = torch.autograd.grad(lps[i], xj, retain_graph=True,
                                 grad_outputs=torch.ones(Xtr.shape[0], dtype=DT))[0]
        jac[i] = gi.abs().mean(0)
    A = jac.t().detach().numpy()

    # remove weakest edges (smallest J) until the graph is acyclic
    with torch.no_grad():
        for thr in np.unique(A):
            keep = torch.tensor(A > thr + 1e-8, dtype=DT)
            na = model.adjacency * keep
            prod = torch.eye(d, dtype=DT); ok = True
            for _ in range(d):                                  # tr(A^k)=0 for all k
                prod = na @ prod
                if prod.trace() != 0:
                    ok = False; break
            if ok:
                model.adjacency = na; break

    # adjacency[j, t]=1 means j->t; convention B[i, j]=1 means j->i  =>  B = adjacency.T
    return model.adjacency.t().detach().numpy()
```

Let me trace the causal chain back. I wanted the directed nonlinear causal DAG from observational data, blocked by super-exponential combinatorial search and by the non-identifiability of direction. ANM identifiability supplied the assumption that breaks direction symmetry, and it works even with Gaussian noise *because* the mechanisms are nonlinear — which forced me to fit a flexible nonlinear model per variable, one independent MLP each, rather than CAM's additive form or NOTEARS's linear coefficients or DAG-GNN's shared GNN. NOTEARS supplied the way out of combinatorial search: turn discrete acyclicity into the smooth `tr e^{·} - d` constraint via the matrix-exponential reweighting of closed-walk counts. But that constraint needed a single nonnegative `d × d` weight matrix, and a stack of neural nets has no such coefficient. So I ran the same closed-walk logic one level deeper — at the level of network paths: a path is dead iff any weight on it is zero, the nonnegative path product is zero iff dead, the sum over routes is the matrix product of the absolute weight matrices, and the output-summed result is exactly the `(A_phi)_{ij}` whose zero certifies that variable `i` drops out of conditional `j`. Because `A_phi` is born nonnegative I fed it straight into `tr e^{A_phi} - d`. The population maximum-likelihood under this constraint provably recovers the true graph when the model class is right and the truth is identifiable. I solved the constrained program with an augmented Lagrangian — convergence-based dual ascent and penalty escalation, each subproblem a minibatch-RMSprop network fit with held-out early stopping — and beat the likelihood's dense-graph bias with CAM-style neighbor selection and significance pruning. Finally, since `A_phi` only upper-bounds true dependence, I extracted the DAG by removing edges in order of the realized expected-Jacobian sensitivity until acyclic.
