# Context: learning discrete logical structure inside a deep network

## Research question

Modern deep networks are excellent at perception and pattern completion but stumble badly on "hard," "global" constraints — the discrete, all-or-nothing relationships that define logical and combinatorial problems. A convolutional net can read a handwritten digit beautifully, yet cannot reliably learn the rules of a constraint puzzle from examples, because those rules are a *discrete* logical structure and the network only has continuous, locally-smooth machinery.

The sharper version of the question is not "can a network *apply* a known logical rule" — much prior work does that — but **can a network *discover* the discrete relationships that explain a set of observations, end to end, with the rules unknown and learnable from data?** A solution would have to:

- represent an *unknown* logical/constraint problem with parameters that are *continuous and differentiable*, so gradients can flow into them;
- be embeddable as a layer inside a larger network, so it can sit behind (say) a digit-recognition front-end and be trained jointly with it;
- learn the structure from weak supervision (bit labels, single-bit labels), not from a hand-specified rule template;
- and **scale** — a layer that takes a fraction of a second per forward/backward pass on realistic problem sizes, not minutes.

Two concrete tasks make the gap vivid. First, the **parity** of a bit string (1 if an odd number of ones): a pure logical function (chained XOR) that is notoriously hard to learn by gradient descent from input/output pairs alone. Second, **9×9 Sudoku** learned *from examples only* — the constraints (rows, columns, 3×3 blocks each a permutation of 1–9) are never given; the system must infer them. And a harder variant, **visual Sudoku**: the input is an *image* of a board (cells are handwritten digits), and the output must be a logical solution — so a perception network and a constraint-learning module must be trained together, end to end.

## Background

**Boolean satisfiability and its optimization analogue.** A SAT instance over `n` boolean variables and `m` clauses asks whether an assignment satisfies every clause (a clause is a disjunction of literals). MAXSAT, the optimization version, asks for the assignment maximizing the number of satisfied clauses; equivalently, minimizing the number violated (MIN-UNSAT). Writing variable `i`'s truth value as `ṽ_i ∈ {−1, 1}` and its signed appearance in clause `j` as `s̃_ij ∈ {−1, 0, 1}`, MAXSAT is

```
maximize_{ṽ ∈ {−1,1}^n}  Σ_j  ⋁_i  1{ s̃_ij ṽ_i > 0 }.
```

SAT/MAXSAT is the universal target of reductions across symbolic AI and constraint satisfaction, which makes it an attractive single primitive: a great many discrete logical problems can be *encoded* as MAXSAT, and the clause-sign data `S` is a natural place to put learnable parameters. But MAXSAT is discrete and NP-hard, and the indicator/disjunction objective is non-differentiable — exactly the property that has kept it out of neural architectures.

**Semidefinite relaxation and randomized rounding (Goemans & Williamson, 1995).** A landmark line of approximation algorithms relaxes such combinatorial problems to semidefinite programs. For MAXCUT and MAX-2SAT, Goemans and Williamson "lift" each binary `ṽ_i ∈ {±1}` to a unit vector `v_i ∈ R^k`, `‖v_i‖ = 1`, and optimize a quadratic form over the vectors — a semidefinite program in the Gram matrix `X = V^T V ⪰ 0`. They recover a discrete assignment by **randomized rounding**: draw a uniformly random hyperplane with normal `r` on the unit sphere and assign variable `i` by the sign of `r^T v_i`. Two facts from that analysis are load-bearing here:

- The probability that two vectors land on opposite sides of a random hyperplane is exactly proportional to the angle between them:
  `Pr[ sgn(u_i^T r) ≠ sgn(u_j^T r) ] = arccos(u_i^T u_j) / π.`
- For MAXCUT the scheme achieves the famous 0.878 approximation ratio in expectation, and empirically the SDP relaxation is substantially *tighter* than the linear-programming relaxation for MAXSAT (Gomes et al., 2006). Repeating the rounding and keeping the best assignment improves the realized ratio in practice.

**Low-rank structure of SDP solutions (Barvinok 1995; Pataki 1998).** A solvable SDP with `p` linear constraints always admits an optimal solution of rank at most `⌈√(2p)⌉`. For a unit-diagonal SDP with `n` unit-norm constraints, this means one can search over `V ∈ R^{k×n}` with `k ≈ √(2n)` instead of over the full `n×n` matrix `X`, while *still* recovering the exact SDP optimum. This collapses the variable count from `O(n²)` to `O(nk)` and replaces the `X ⪰ 0` constraint with a simple change of variables `X = V^T V` (semidefiniteness is then automatic).

**Coordinate descent for low-rank SDPs — the Mixing method (Wang, Chang & Kolter, 2017).** For the unit-diagonal SDP
`minimize ⟨C, V^T V⟩  subject to ‖v_i‖ = 1`,
the terms of the objective that depend on a single column `v_i` are `v_i^T (Σ_j c_ij v_j)`; since `‖v_i‖ = 1`, minimizing over `v_i` has a closed form,

```
v_i := −g_i / ‖g_i‖,    where  g_i = Σ_{j≠i} c_ij v_j.
```

Cycling this update over all `i` is the **Mixing method** (each `v_i` becomes a normalized "mixture" of the others). It has no step size and no free parameters; it is `O(k · #nonzeros)` per sweep; and, despite the non-convex `V`-formulation, for sufficient rank `k > √(2n)` it provably converges to the *global* optimum of the underlying SDP. The same work showed the method applies beyond MAXCUT — including to a general-MAXSAT SDP relaxation that generalizes the Goemans–Williamson MAX-2SAT construction and is solved by the very same coordinate-descent update, with a Goemans–Williamson hyperplane rounding to recover a binary assignment. Throughout, the Mixing method is a *solver*: it computes an assignment given a *fixed* coefficient matrix `C` (equivalently, fixed clauses); the coefficients are inputs, not something it learns.

**Optimization problems as differentiable layers (Amos & Kolter, 2017; Donti et al., 2017; and related).** A parallel line of work makes a *constrained optimization problem* itself a layer in a network. OptNet, for instance, places a small quadratic program

```
z_{i+1} = argmin_z  ½ z^T Q z + q^T z   s.t.  Az = b,  Gz ≤ h,
```

inside the network, where `(Q,q,A,b,G,h)` can depend differentiably on the previous layer. The gradient of the layer's solution with respect to its parameters is obtained not by unrolling the solver but by **differentiating the KKT optimality conditions at the solution** (matrix differentials / implicit differentiation), and the backward pass reuses the forward factorization so it is nearly free once the problem is solved. Related layers exist for submodular optimization and for equilibria of zero-sum games. The general recipe — *solve an optimization problem in the forward pass; differentiate its solution implicitly in the backward pass* — is the template.

**Diagnostic failures that frame the problem.**
- *Parity is hard for gradient-based learning.* Shalev-Shwartz, Shamir & Shammah (2017) show that for the family of parity functions the gradient of a network's loss carries almost no information about *which* parity generated the data — the gradient's variance across the family is essentially zero — so gradient descent has nothing to follow and fails to learn parity from input/output pairs.
- *Sudoku defeats convolutional nets.* A strong ConvNet (e.g. Park 2016, ten convolutional layers) can fit the training boards but fails to generalize, and collapses entirely on *permuted* boards where the spatial-locality crutch is removed: the network never recovers the underlying logical relations, only surface statistics.
- *Existing differentiable-optimization layers do not scale to these sizes.* The dense interior-point QP backward of the optimization-layer line is `O(n³)`-ish per solve and cannot exploit GPU parallelism over many small coordinate updates; it can handle a 4×4 toy Sudoku but stalls on the 9×9 problem.
- *Existing neuro-symbolic systems require the rules.* Most differentiable-logic systems (relational reasoning nets seeded with which variables may interact, inductive-logic-programming nets seeded with rule templates, probabilistic-logic-programming nets) tune the parameters of a *given* relational structure rather than discovering the structure itself.

## Baselines

- **Convolutional Sudoku solver (Park, 2016).** Interprets the bit board as image channels and stacks ~10 convolutional layers (512 3×3 filters each) to output the completed board; trained with MSE. *Limitation:* it fits training boards (e.g. ~73% train) but generalizes essentially not at all to held-out boards, and on permuted boards — where locality is destroyed — it makes little progress even on the training set. It learns surface spatial statistics, not the logical constraints. A variant that additionally receives a binary mask of which cells are unknown (ConvNetMask) helps only marginally.
- **OptNet (Amos & Kolter, 2017).** A differentiable QP layer; differentiates KKT conditions to get gradients w.r.t. the problem data. On 4×4 Sudoku it learns the rules from input/output examples alone, which is the proof of concept that a differentiable optimization layer *can* learn discrete structure. *Limitation:* the dense primal-dual interior-point solve does not scale — on 9×9 Sudoku it makes negligible progress even after days of training, and per-epoch wall-clock is far higher than a coordinate-descent solver of comparable accuracy. QP is also not the most natural relaxation of *logical* (clause) structure.
- **LSTM sequence classifier (for parity).** A recurrent classifier (e.g. 100 hidden units) trained to map a bit sequence to its parity. *Limitation:* with only single-bit (final-answer) supervision it cannot find a representation of the chained-XOR structure; test error stays near chance (≈0.5) regardless of architecture/learning-rate sweeps — consistent with the gradient-uninformativeness result for parities.
- **Recurrent relational networks (Palm et al., 2017).** A strong neural Sudoku solver, but one that is *given* hand-coded information about which variables are allowed to interact (the message-passing graph encodes the board's structure). *Limitation:* it does not *discover* the relations; the structure is built in.
- **Differentiable-logic systems with known rules** (DeepProbLog, ∂ILP/Evans–Grefenstette, lifted relational nets, semantic-loss, etc.). *Limitation:* each requires a pre-specified set of relationships / rule templates / groundings; they learn parameters *within* a fixed logical skeleton rather than learning the skeleton from data, so they cannot operate when the structure of the domain is unknown.

## Evaluation settings

- **Parity (chained XOR).** Input sequences of length `L = 20` and `L = 40`; 10K random examples each (9K train / 1K test); supervision is the *single* parity bit. A length-`L` model is built as a chain of tied logical sub-problems (each taking the next input bit plus the running result), so the system must coordinate a long series of sub-problems with no intermediate supervision. Metric: held-out classification error (chance = 0.5). Optimizer Adam, learning rate `10⁻¹`; LSTM baseline at `10⁻³`.
- **9×9 Sudoku (original and permuted).** 9K train / 1K test boards, given as a logical bit representation (one-hot per cell, 9×9×9 = 729 bits) plus a mask of unknown cells; the input is *vectorized* so no 2-D locality is available to the solver. The *permuted* variant applies a fixed permutation to the bit representation (and mask/labels), destroying any locality while preserving the logical relations modulo the permutation — a direct test of whether a method learns relations vs. spatial statistics. Metric: whole-board test accuracy (a board counts only if every cell is correct). Optimizer Adam; the constraint layer at lr `2×10⁻³`, ConvNet baselines at `10⁻⁴`.
- **Visual Sudoku.** Same boards, but each given cell is rendered as an MNIST digit image; the network must output the logical solution. A LeNet-style digit classifier feeds cell-wise probabilities into the constraint layer; the whole stack is trained end to end (cross-entropy), with separate learning rates for the perception front-end (`10⁻⁵`) and the logical layer (`2×10⁻³`). Metric: whole-board test accuracy, contextualized against the best achievable given the digit-classifier's per-digit accuracy.
- **4×4 reduced Sudoku.** A smaller instance used to compare against OptNet head-to-head (convergence speed, per-epoch wall-clock, final accuracy).

## Code framework

The pieces below already exist before the method does: an autograd framework with a custom-`Function` mechanism (forward + analytic backward), an Adam optimizer, standard losses (binary cross-entropy / NLL / MSE), a LeNet-style conv stack for digit recognition, and standard linear-algebra primitives (dot products, normalization, rank-one matrix updates). The one thing that does *not* yet exist is the constraint layer itself. The scaffold leaves it as a single empty slot whose forward and backward are to be designed.

```python
import torch
import torch.nn as nn
from torch.autograd import Function

# ---- The constraint layer: ONE empty slot to be designed --------------------

class ConstraintSolve(Function):
    """Custom autograd Function for the constraint layer.
    forward: given known variable assignments, produce assignments for the
             unknown variables under some learnable constraint parameters.
    backward: return gradients w.r.t. the inputs and the learnable parameters.
    """
    @staticmethod
    def forward(ctx, params, z, is_input):
        # TODO: the forward solve we will design.
        # Inputs: learnable params; z in [0,1] per variable; is_input mask.
        # Output: completed assignments z in [0,1].
        pass

    @staticmethod
    def backward(ctx, dz):
        # TODO: the backward we will design (gradients w.r.t. params and inputs).
        pass


class ConstraintLayer(nn.Module):
    """Wraps ConstraintSolve as a learnable layer."""
    def __init__(self, n, *hyperparams):
        super().__init__()
        # TODO: the learnable parameters of the constraint problem.
        self.params = nn.Parameter(...)  # shape to be decided

    def forward(self, z, is_input):
        # TODO: prepare inputs, call ConstraintSolve.apply, post-process outputs.
        return ConstraintSolve.apply(self.params, z, is_input)


# ---- Existing, generic harness ----------------------------------------------

class DigitConv(nn.Module):
    """LeNet-style digit classifier (already standard) for visual inputs."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1   = nn.Linear(4 * 4 * 50, 500)
        self.fc2   = nn.Linear(500, 10)
    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x)); x = self.fc2(x)
        return F.softmax(x, dim=1)[:, :9].contiguous()

def train_loop(model, loader, opt):
    import torch.nn.functional as F
    for data, is_input, label in loader:
        opt.zero_grad()
        pred = model(data, is_input)
        loss = F.binary_cross_entropy(pred, label)  # or NLL / MSE per task
        loss.backward()
        opt.step()
```
