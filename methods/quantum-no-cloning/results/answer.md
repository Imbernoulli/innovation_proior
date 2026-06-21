# The No-Cloning Theorem

## Problem

Can a single physical device copy an *unknown* quantum state — take one system in some state $|\psi\rangle$ the operator does not know, and output two systems each in state $|\psi\rangle$? Classically, copying is trivial. A quantum copier would be extraordinary: it would let you measure copies in many bases and reconstruct an unknown state without disturbing the original, and it would let an entangled pair carry a faster-than-light signal (the FLASH scheme: clone the collapsed far-half and read off which basis the distant party measured in). The theorem settles it in the negative and pins down exactly why.

## Key idea

A device that does not measure is a unitary $U$ on the system together with a target register and any machine ancilla. Cloning would require *one fixed* $U$ (the state is unknown, so $U$ cannot depend on it) to leave two visible registers in $|\psi\rangle|\psi\rangle$ for **every** $|\psi\rangle$. But $|\psi\rangle|\psi\rangle$ is *quadratic* in the state, and $U$ is *linear*. Linearity carries each superposition branch separately and cannot manufacture the cross term that "squaring" the state requires. Equivalently, unitarity preserves inner products, while perfect copying would square them — forcing every pair of clonable states to be the same ray or orthogonal. So no universal cloner exists; only a known, mutually orthogonal set of states can be copied.

## Theorem and proofs

**No-cloning theorem.** Let $\mathcal{H}$ be a Hilbert space, $|\Sigma\rangle$ a fixed blank state, and $|A_{\mathrm{ready}}\rangle$ any fixed machine ready state. There is no unitary $U$ on the system, target, and machine registers such that
$$U\big(|\psi\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle\big)=|\psi\rangle|\psi\rangle|A_\psi\rangle \quad\text{for all normalized } |\psi\rangle\in\mathcal{H},$$
even when the final machine state $|A_\psi\rangle$ is allowed to depend on $\psi$. The no-machine form $U(|\psi\rangle|\Sigma\rangle)=|\psi\rangle|\psi\rangle$ is the special case $|A_\psi\rangle$ absent.

**Proof 1 — linearity / superposition.** Suppose $U$ copies two orthonormal states,
$$U|0\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=|0\rangle|0\rangle|A_0\rangle,\qquad U|1\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=|1\rangle|1\rangle|A_1\rangle.$$
For $|\psi\rangle = a|0\rangle + b|1\rangle$, linearity of $U$ forces
$$U|\psi\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=a|0\rangle|0\rangle|A_0\rangle+b|1\rangle|1\rangle|A_1\rangle.$$
A genuine clone instead requires
$$|\psi\rangle|\psi\rangle|A_\psi\rangle=\big(a^2|0\rangle|0\rangle+ab\,|0\rangle|1\rangle+ab\,|1\rangle|0\rangle+b^2|1\rangle|1\rangle\big)|A_\psi\rangle.$$
For any genuine superposition $ab\ne0$, the true clone has visible $|0\rangle|1\rangle$ and $|1\rangle|0\rangle$ components, while the forced linear output has none. So no fixed $U$ copies the superpositions of a copied orthonormal pair. $\square$

**Proof 2 — inner-product preservation.** In the no-machine notation, suppose
$$U|\psi\rangle|\Sigma\rangle = |\psi\rangle|\psi\rangle,\qquad U|\phi\rangle|\Sigma\rangle = |\phi\rangle|\phi\rangle.$$
Unitarity preserves inner products. The inputs give $\langle\psi|\phi\rangle\langle\Sigma|\Sigma\rangle = \langle\psi|\phi\rangle$; the outputs give $\langle\psi|\phi\rangle\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2$. Equating,
$$\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2 \;\Longrightarrow\; \langle\psi|\phi\rangle\,(1-\langle\psi|\phi\rangle)=0 \;\Longrightarrow\; \langle\psi|\phi\rangle \in \{0,1\}.$$
With machine end states included, the equation becomes $\langle\psi|\phi\rangle=\langle\psi|\phi\rangle^2\langle A_\psi|A_\phi\rangle$. Taking magnitudes gives $s=s^2|\langle A_\psi|A_\phi\rangle|$ for $s=|\langle\psi|\phi\rangle|$, which is impossible for $0<s<1$ because $|\langle A_\psi|A_\phi\rangle|\le1$. So a single $U$ can clone at most a **mutually orthogonal** set, plus repeated descriptions of the same ray. Any non-orthogonal pair is unclonable, so no universal cloner exists. $\square$

## Consequences

- **Orthogonal states are clonable; unknown states are not.** A known orthonormal set *can* be copied — CNOT realizes $|x\rangle|0\rangle\mapsto|x\rangle|x\rangle$ for $x\in\{0,1\}$ — but fails on superpositions ($\tfrac1{\sqrt2}(|0\rangle+|1\rangle)|0\rangle \mapsto \tfrac1{\sqrt2}(|00\rangle+|11\rangle)\neq|+\rangle|+\rangle$). The prohibition is on *arbitrary/unknown* states only.
- **No conflict with classical copying.** Distinct classical symbols are perfectly distinguishable, i.e. mutually orthogonal quantum states, sitting in the $\langle\psi|\phi\rangle=0$ exception. Classical information copies freely precisely because it is orthogonal.
- **No superluminal signaling.** For the singlet, Bob's state before any ordinary message arrives is $\rho_B=\mathrm{Tr}_A(\rho_{AB})=I/2$. Alice's $z$ choice gives Bob the ensemble $\tfrac12|z_+\rangle\langle z_+|+\tfrac12|z_-\rangle\langle z_-|=I/2$; her $x$ choice gives $\tfrac12|x_+\rangle\langle x_+|+\tfrac12|x_-\rangle\langle x_-|=I/2$. Any allowed local operation $\mathcal{E}$ is linear and acts on $\rho_B$, so Bob gets $\mathcal{E}(I/2)$ either way. A perfect cloner would have to depend on the chosen pure-state decomposition of the same density matrix, which is nonlinear and not a physical quantum operation.
- **No-deleting and no-broadcasting (cousins).** The time-reverse — a fixed unitary erasing one of two identical unknown copies, $|\psi\rangle|\psi\rangle\to|\psi\rangle|\Sigma\rangle$ — is also forbidden: the inputs overlap as $\langle\psi|\phi\rangle^2$ but the targets as $\langle\psi|\phi\rangle$, irreconcilable by a unitary for a non-orthogonal pair (no-deleting). The mixed-state generalization: a set of density matrices can be broadcast iff they commute; non-commuting states cannot (no-broadcasting), the density-matrix echo of orthogonal vs. non-orthogonal.
- **Quantum key distribution.** Encoding bits in non-orthogonal states (e.g. rectilinear vs. diagonal polarizations) means an eavesdropper cannot clone the carrier to measure a copy while forwarding the original; any information gain disturbs the state and is detectable. No-cloning is the security primitive of QKD.

