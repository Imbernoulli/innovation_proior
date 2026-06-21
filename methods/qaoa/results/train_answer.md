The problem I want to solve is approximate combinatorial optimization on a quantum computer. An instance is $n$ bits and $m$ local clauses, with objective $C(z) = \sum_{\alpha=1}^m C_\alpha(z)$ counting how many constraints the string $z$ satisfies; MaxCut and bounded-occurrence Max E3LIN2 are my running examples. Exact MaxSat is NP-hard, so I am after something more honest: a quantum procedure with a single integer knob I can turn to spend more quantum effort and provably get a better answer, that compiles to gates no more nonlocal than the clauses themselves, and whose run parameters I can fix cheaply offline for fixed depth. The one quantum optimization idea really on the table is adiabatic evolution, and studying exactly where it breaks is what tells me what to build. Its encoding is right and I want to keep it: promote each bit $z_i$ to a qubit, so $C$ becomes diagonal in the computational basis with $C|z\rangle = C(z)|z\rangle$, and the optimum is literally the extremal eigenstate of a diagonal operator. Adiabatic evolution reaches that eigenstate by interpolating from an easy beginning Hamiltonian, the transverse field $H_B = \sum_i \tfrac12(1-\sigma^x_i)$, whose ground state is the uniform superposition $|s\rangle = 2^{-n/2}\sum_z |z\rangle = |+\rangle^{\otimes n}$ — one layer of Hadamards deep. One drags $H(t) = (1-t/T)H_B + (t/T)H_P$ slowly from $H_B$ into the problem Hamiltonian and, by the adiabatic theorem, ends in the answer provided the spectral gap stays open, which costs runtime governed by

$$T \gg \mathcal{E}/g_{\min}^2, \qquad g_{\min} = \min_s\big(E_1(s) - E_0(s)\big).$$

That $1/g_{\min}^2$ is the load-bearing failure. For hard instances the gap is exponentially small in $n$, so the guaranteed runtime is exponential; worse, this $T$ is the duration of a *single continuous coherent analog evolution*, which near-term gate-model hardware with a coherence window and a finite gate budget simply cannot sustain. And it is not even reliable: the adiabatic success probability is non-monotone in $T$ (it can climb then fall off a cliff, as on the 20-qubit Max2Sat instance of Crosson, Farhi, Lin, Lin and Shor), and symmetric Hamming-weight objectives trap it in a false minimum for any subexponential time (Farhi, Goldstone, Gutmann). So "just take $T$ large" is both physically out of reach and not safely correct.

I propose QAOA, the Quantum Approximate Optimization Algorithm. The move is to keep the encoding but replace the long analog run by a short *discrete* gate-model circuit, and then to break free of the schedule that made it long. Trotterization turns a continuous evolution under a sum of two terms into alternating short steps, $e^{-i(A+B)t} = (e^{-iAt/N}e^{-iBt/N})^N + O(t^2/N)$, and my evolving Hamiltonian is exactly two pieces: the cost operator $C$ and a transverse-field driver, for which I use $B = \sum_j \sigma^x_j$ (top eigenstate $|+\rangle^{\otimes n}$), differing from the shifted ground-state form only by a constant and a sign I absorb into the angles. So I alternate a cost factor and a mixer factor. The cost factor is

$$U(C,\gamma) = e^{-i\gamma C} = \prod_\alpha e^{-i\gamma C_\alpha},$$

which is *exact* — because $C = \sum_\alpha C_\alpha$ is diagonal all the clauses commute, so there is no Trotter error inside the cost layer — and each factor $e^{-i\gamma C_\alpha}$ acts only on the bits its clause touches, so circuit locality equals clause locality for free, a 2-bit clause giving a 2-qubit gate. Since $C$ has integer eigenvalues, $\gamma \in [0, 2\pi]$ suffices. The mixer factor is

$$U(B,\beta) = e^{-i\beta B} = \prod_j e^{-i\beta\sigma^x_j},$$

a single $RX(2\beta)$ per qubit, depth one, with $\beta \in [0,\pi]$. The driver $B = \sum_j\sigma^x_j$ is the right choice for two reasons: $|+\rangle^{\otimes n}$ is its extremal eigenstate (the easy state I can already prepare), and a single $\sigma^x$ flips one bit, so $B$ connects every basis string to its single-flip neighbors with non-negative off-diagonal entries — by Perron–Frobenius this forces a non-degenerate top eigenvalue with a gap, exactly the gap-positivity the adiabatic path needs. The two layers are complementary: the diagonal cost factor can only *rephase* a state, never move probability between strings, so without a mixer the circuit would never explore; $C$ imprints the objective as phases, $B$ converts those phases into amplitude flowing toward good strings.

The depth-$p$ ansatz is then $|\gamma,\beta\rangle = U(B,\beta_p)U(C,\gamma_p)\cdots U(B,\beta_1)U(C,\gamma_1)|s\rangle$, a circuit of depth at most $mp+p$. The crucial step is what to do with the angles. A *faithful* Trotterization of the adiabatic path pulls in two contradictory directions — small angles for low Trotter error, but a large total run time for the adiabatic theorem — forcing many steps and so large $p$, right back to a deep circuit. So I stop insisting on the schedule. The only thing I actually care about is the final expectation of $C$, so I cut all $2p$ angles loose and treat them as free variational parameters, defining

$$F_p(\gamma,\beta) = \langle\gamma,\beta|\,C\,|\gamma,\beta\rangle, \qquad M_p = \max_{\gamma,\beta} F_p(\gamma,\beta).$$

This is the variational principle put to work: $\langle\psi|C|\psi\rangle \le C_{\max}$ for any state, so $M_p$ can never exceed the optimum and every increase in $F_p$ is genuine progress, not a heuristic to apologize for. Freeing the angles can only help, since the adiabatic schedule's particular angle sequence is just one point in the box I now optimize over. Two structural facts close the loop. A depth-$(p-1)$ circuit is the special case of depth $p$ with the extra layer doing nothing, so $M_p \ge M_{p-1}$ — more layers never hurt, and $p$ is the knob I wanted. And $\lim_{p\to\infty} M_p = \max_z C(z)$, because among the angle choices available at large $p$ are exactly those that faithfully Trotterize an adiabatic path (small angles summing to a long run time): for that subfamily the adiabatic theorem drives $F_p$ to $\max_z C(z)$, and since $M_p$ is the max over all angles it is at least that and at most $C_{\max}$. So the dreaded faithful-adiabatic regime isn't gone — it sits inside the search space as a worst-case fallback while the optimizer finds something far shallower. This is genuinely *not* the adiabatic algorithm in disguise: on the ring at $p=1$ the best state gives a $3/4$ approximation ratio yet has exponentially small overlap with the optimal strings — it makes $\langle C\rangle$ large rather than approximating the ground state, and measuring such a state still hands me good strings often because the expectation governs the measured mean.

Choosing the angles is made cheap by locality. For MaxCut each edge contributes $C_{\langle jk\rangle} = \tfrac12(1 - \sigma^z_j\sigma^z_k)$, and conjugating one edge operator $U^\dagger\cdots C_{\langle jk\rangle}\cdots U$ leaves the mixer and cost factors on qubits beyond graph-distance $p$ to commute through and cancel against their daggers; only the distance-$p$ neighborhood survives. Edges with isomorphic neighborhood subgraphs contribute the same function, so $F_p(\gamma,\beta) = \sum_g w_g\, f_g(\gamma,\beta)$ with integer weights $w_g$ read straight off the graph and each $f_g$ living on an $n$-independent Hilbert space of size at most $2^{q_{\text{tree}}}$, $q_{\text{tree}} = 2\frac{(v-1)^{p+1}-1}{(v-1)-1}$. The *only* $n$-dependence is in the weights, so for fixed $p$ and bounded degree I optimize the $2p$ angles classically at cost independent of $n$. Working $p=1$ on triangle-free cubic graphs, conjugating $Z_jZ_k$ by the mixer rotates each $Z$ in the $Y$–$Z$ plane via $e^{i\beta\sigma^x}Ze^{-i\beta\sigma^x} = \cos2\beta\,Z + \sin2\beta\,Y$, and the cost layer's diagonal phases produce the per-edge expectation

$$\langle C_{\langle jk\rangle}\rangle = \tfrac12 + \tfrac14\sin(4\beta)\sin\gamma\,(\cos^{d_j}\gamma + \cos^{d_k}\gamma),$$

where $\sin4\beta$ is the mixer turning phase into population, $\sin\gamma$ is the cost layer's first imprint, and the $\cos^d\gamma$ factors are leakage through the $d$ neighboring edges. Maximizing gives $0.6924\ldots$ per edge near $\gamma\approx 0.616,\beta\approx0.393$. Bounding the optimal cut by $3T$ triangles and $4S$ crossed squares forcing uncut edges ($3T+4S\le n$), the worst-case ratio is $\min M_1(1,s,t)/(\tfrac32 - s - t)$ over densities with $4s+3t\le1$, minimized at $s=t=0$ to give a provable $0.6924$ on *any* 3-regular graph. On the ring, $M_p = n\frac{2p+1}{2p+2}$, ratio $\to 1$ at constant depth $3p$. And the measured value concentrates: $\mathrm{Var}(C) = O(m)$, so $O(m^2)$ shots land within $1$ of $F_p$ with probability $1-1/m$. The same machinery on Max E3LIN2, fixing $\beta=\pi/4$ so $Z\mapsto Y$ exactly and using a hypercontractivity tail bound plus a Chebyshev node argument, yields a $\gamma\in[-1/(10D^{1/2}),1/(10D^{1/2})]$ found among $5\ln D$ values giving $(\tfrac12 + 1/(101 D^{1/2}\ln D))m$ satisfied equations, and for the typical random-sign instance $\gamma = 1/(\sqrt3 D^{1/2})$ gives $(\tfrac12 + 1/(2\sqrt{3e}\,D^{1/2}))m$ — the $D^{1/2}$ scaling matching the inapproximability threshold.

So the recipe is settled: encode $C$ as a diagonal operator, build the $p$-layer alternating circuit on $|+\rangle^{\otimes n}$, treat the $2p$ angles as free variational parameters, maximize $F_p$ classically for fixed $p$ or by a classical optimizer querying the quantum computer, then run at the best angles, measure, and keep the best string. The implementation mirrors PennyLane's convention with a minimization Hamiltonian $H_C = \tfrac12\sum_{\text{edges}}(Z_iZ_j - I) = -C_{\text{cut}}$, so minimizing $\langle H_C\rangle$ maximizes the cut; the mixer is $H_B = \sum_i X_i$, each layer is $e^{-i\gamma H_C}$ then $e^{-i\beta H_B}$, the state starts with Hadamards, and the optimizer handles the angles while sampling reads candidate cuts.

```python
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

graph = nx.Graph([(0, 1), (0, 3), (1, 2), (2, 3)])
wires = list(graph.nodes)
n_wires = len(wires)

def build_objective_and_driver(graph):
    # PennyLane's MaxCut cost Hamiltonian is H_C = 1/2 sum_edges (Z_i Z_j - I) = -C_cut.
    return qml.qaoa.maxcut(graph)

def prepare_start(wires):
    for w in wires:
        qml.Hadamard(wires=w)

def state_preparation(gammas, betas, cost_h, mixer_h):
    for gamma, beta in zip(gammas, betas):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(beta, mixer_h)

cost_h, mixer_h = build_objective_and_driver(graph)
dev = qml.device("default.qubit", wires=n_wires)
shot_dev = qml.device("default.qubit", wires=n_wires, shots=100)

@qml.qnode(dev)
def expectation_circuit(gammas, betas):
    prepare_start(wires)
    state_preparation(gammas, betas, cost_h, mixer_h)
    return qml.expval(cost_h)                  # minimize H_C to maximize the cut

def objective(params):
    gammas, betas = params[0], params[1]
    return expectation_circuit(gammas, betas)

@qml.qnode(shot_dev)
def sampling_circuit(gammas, betas):
    prepare_start(wires)
    state_preparation(gammas, betas, cost_h, mixer_h)
    return qml.sample(wires=wires)

def cut_value(bitstring):
    assignment = dict(zip(wires, map(int, bitstring)))
    return sum(assignment[u] != assignment[v] for u, v in graph.edges)

def qaoa_maxcut(p=1, steps=30):
    params = np.array(0.01 * np.random.rand(2, p), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    for _ in range(steps):
        params = opt.step(objective, params)
    samples = sampling_circuit(params[0], params[1])
    best = max(samples, key=cut_value)
    return params, best, cut_value(best)
```
