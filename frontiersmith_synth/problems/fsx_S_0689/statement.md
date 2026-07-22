# Mirror Relay: Teleporting a Pulse Across a Defected Spin Chain

## Problem
A quantum relay is a path of `N` sites, numbered `1..N`. Site `1` is the *source* port and
site `N` is the *sink* port. Each site `i` carries an on-site energy `eps_i`; each edge
`e = 1..N-1` (connecting site `e` and site `e+1`) carries a coupling strength `J_e > 0`.
Together these form a real symmetric tridiagonal Hamiltonian matrix `A` (size `N x N`):
diagonal entries `A_ii = eps_i`, off-diagonal entries `A_{e,e+1} = A_{e+1,e} = J_e`.

A handful of sites are **defects**: their `eps_i` is fixed by the instance to a nonzero
value you cannot change (all other sites have `eps_i = 0`). A handful of edges are
**frozen**: their `J_e` is likewise fixed by the instance. You choose the remaining
(**free**) couplings, each a real number in `[J_LO, J_HI]`.

Injecting a pulse at site `1` and letting it evolve for a fixed time `T` under the
Schrodinger equation produces the complex amplitude vector `exp(-i*A*T) * e_1`. The
**transfer fidelity** is the magnitude of that vector's `N`-th entry,
`Fid = |[exp(-i*A*T)]_{1,N}|` — how much of the pulse arrives, intact, at the sink.

This is a continuous-time quantum walk: naive intuition ("stronger coupling moves the
packet faster") is misleading. Fidelity is governed by the *eigenvalues* of `A`, not by
any notion of a path: writing `A = V diag(lambda) V^T`, `Fid = |sum_k V_{1k} V_{Nk}
exp(-i*lambda_k*T)|`. This sum only refocuses to near 1 when the eigenvalue gaps are
close to **commensurate** with `T` (all near-integer multiples of a common base
frequency) — a spectral-arithmetic condition, not a propagation one. A uniform or
smoothly-varying coupling profile gives a spread-out (incommensurate) spectrum and the
sum destructively interferes, near zero. Defects and frozen edges break the profile you
would otherwise want to lay down, so you must adapt the free couplings around them.

## Input (stdin)
```
N
T
J_LO J_HI
D
s_1 eps_1
...
s_D eps_D
K
e_1 v_1
...
e_K v_K
```
`D` defect lines give a site index (`2 <= s <= N-1`; never a port) and its fixed energy.
`K` frozen-edge lines give an edge index (`1 <= e <= N-1`) and its fixed coupling. The
**free edges** are all `e` in `1..N-1` not listed among the `K` frozen edges, in
ascending order — there are `N-1-K` of them.

## Output (stdout)
```
F
J_free_1 J_free_2 ... J_free_F
```
Print `F = N-1-K` (the number of free edges), then that many reals — the coupling you
choose for each free edge, listed in the same ascending edge-index order used above.

## Feasibility
- The printed count must equal `N-1-K` exactly.
- Every value must parse as a finite real number within `[J_LO, J_HI]` (tolerance `1e-6`).
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `Fid`, the transfer fidelity defined above, over your choice of free couplings.

## Scoring
The checker builds its own reference: every free edge set to the midpoint `(J_LO+J_HI)/2`,
giving fidelity `B_raw`. The baseline used for normalization is `B = 1.4 * B_raw` (a 40%
margin so a strong solution never saturates the score). With your fidelity `F`:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Example: if `F = 0.55` and `B_raw = 0.10`, then `B = 0.14`,
`sc = min(1000, 100*0.55/0.14) = 392.86`, `Ratio = 0.39286`.

## Constraints
- `6 <= N <= 14`, `0.05 <= J_LO < J_HI <= 11`, `1 <= D,K <= 2`.
- Time limit 5s, memory 512m.
