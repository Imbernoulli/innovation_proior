There's a manuscript on my desk that should be wrong, and I can't immediately say where. It proposes faster-than-light signaling, so by relativity it must fail — but the argument never mentions relativity, so the relativistic objection is useless for finding the flaw. The error lives somewhere inside ordinary quantum mechanics, and I want to put my finger on the exact line.

Let me reconstruct the scheme so I'm arguing against the real thing and not a strawman. Two parties, far apart, share an entangled pair — take the clean Bohm version, a spin-zero source emitting two spin-$\tfrac12$ particles in the singlet. Alice has particle I, Bob particle II. Alice gets to choose what to measure on her particle: the $z$-component of spin, or the $x$-component. If she measures $s_z$, her outcome collapses Bob's particle into the corresponding opposite $z$-eigenstate, $|z_+\rangle$ or $|z_-\rangle$. If instead she measures $s_x$, Bob's particle lands in the corresponding opposite $x$-eigenstate, $|x_+\rangle$ or $|x_-\rangle$. So Alice's *choice of axis* determines which **basis** Bob's particle is now sharp in.

Now ordinarily this does Bob no good. He holds one particle. If he measures it, say along $x$, and Alice measured $z$, his particle was in $|z_\pm\rangle = \tfrac{1}{\sqrt2}(|x_+\rangle \pm |x_-\rangle)$, so he gets $x_+$ or $x_-$ with equal probability — a coin flip. And if Alice had measured $x$ instead, his particle is already $|x_\pm\rangle$ and he gets a definite result. But a single shot can't tell "definite" from "coin flip" — one outcome is one outcome. Averaged over Alice's random results, Bob's statistics are identical whichever axis she picked. That's the standard reason entanglement doesn't signal: Bob's marginal is fixed, independent of Alice's knob.

The manuscript's move is to break that symmetry by giving Bob *many copies* of his one particle. Insert a multiplier at Bob's end: feed in his single received particle, get out a burst of $N$ particles all in the same spin state. Then Bob measures all $N$ along $x$. If Alice measured $x$, every one of the $N$ comes out the same — all $x_+$ or all $x_-$. If Alice measured $z$, Bob's particle was an $x$-superposition, so the burst splits roughly half $x_+$, half $x_-$. Now a single run *does* distinguish the cases: "all aligned" versus "fifty-fifty." Bob reads Alice's axis off the histogram. Instantly. Across any distance.

So the entire scheme rests on one component: a box that takes an unknown one-particle state and emits two or more particles in that same state. Everything else is sound — entanglement, collapse, the statistics are all textbook. If the box exists, the signaling works and relativity is in trouble. The error, if there is one, is the box. The question I actually have to settle is therefore not about relativity at all. It's: **can quantum mechanics copy an unknown state?**

Let me strip it to the cleanest possible form and stop talking about bursts of $N$. If I can make *two* copies I can make $N$ by repeating, so the whole thing lives or dies on a two-copy machine. I want a device that does
$$|\psi\rangle \otimes |\text{blank}\rangle \;\longrightarrow\; |\psi\rangle \otimes |\psi\rangle.$$
The input is the unknown state next to a fresh target register in some standard ready state $|\text{blank}\rangle$ — copy onto blank paper. The output is two systems in the state $|\psi\rangle$.

And here is the constraint I must hold onto, because it's the heart of everything: the device cannot depend on $\psi$. Bob does not know which state collapsed out — that's the whole point, it depends on Alice's far-away result. So the box is *one fixed machine* that has to work for **every** input state simultaneously. I'm not asking "can I build a copier tuned to $|z_+\rangle$?" — of course I can, I just prepare a fresh $|z_+\rangle$. I'm asking whether one and the same machine copies $|z_+\rangle$ *and* $|x_+\rangle$ *and* every superposition in between, with no foreknowledge of which it's handed.

What kind of object is the machine? It's a physical process acting on the system together with the target and whatever internals it has, and as long as it doesn't itself perform a measurement it must be a unitary evolution — that's how closed quantum systems evolve, $|\Phi\rangle \mapsto U|\Phi\rangle$ with $U^\dagger U = I$. (If the box does measure partway through, I can defer the measurement and absorb the apparatus into a bigger unitary on a larger space; so unitary is the honest, fully general model.) Give the apparatus a ready state $|A_{\mathrm{ready}}\rangle$, and even allow its final state to depend on the input. A perfect copier still has to leave the two visible registers as a product of two copies:
$$U\big(|\psi\rangle|\text{blank}\rangle|A_{\mathrm{ready}}\rangle\big) = |\psi\rangle|\psi\rangle|A_\psi\rangle,\qquad \text{required for all } |\psi\rangle.$$

So I have a *single linear* operator $U$, and I'm demanding it produce $|\psi\rangle|\psi\rangle$ for every $\psi$. Linear operator. $|\psi\rangle|\psi\rangle$. Let me just stare at those two facts together, because something is already itching.

The output $|\psi\rangle|\psi\rangle$ is *quadratic* in $\psi$ — write $\psi$ in components and the output has products of components, $\psi_i \psi_j$. But $U$ is *linear* in its input. A linear map can't manufacture a quadratic dependence on the input out of nothing. That's the smell. Let me make it bite.

Suppose the machine works on two particular states — and let me pick the two that Bob actually cares about, the two basis vectors of one axis. Say it copies $|0\rangle$ and $|1\rangle$ (read these as $|z_+\rangle, |z_-\rangle$, or $|\!\updownarrow\rangle, |\!\leftrightarrow\rangle$ for photon polarization — doesn't matter):
$$U|0\rangle|\text{blank}\rangle|A_{\mathrm{ready}}\rangle = |0\rangle|0\rangle|A_0\rangle,\qquad U|1\rangle|\text{blank}\rangle|A_{\mathrm{ready}}\rangle = |1\rangle|1\rangle|A_1\rangle.$$
Fine, no contradiction yet — I've only constrained $U$ on two input vectors, and it's free to do this.

Now hand it the superposition Bob is faced with when Alice measured the *other* axis: $|\psi\rangle = a|0\rangle + b|1\rangle$. I don't get to choose how $U$ acts here freely, because $U$ is **linear**, and $|\psi\rangle|\text{blank}\rangle|A_{\mathrm{ready}}\rangle$ is just a linear combination of the two inputs I already pinned down. So linearity *forces* the output:
$$U\big((a|0\rangle+b|1\rangle)|\text{blank}\rangle|A_{\mathrm{ready}}\rangle\big)=a\,|0\rangle|0\rangle|A_0\rangle+b\,|1\rangle|1\rangle|A_1\rangle.$$
There's no freedom here. The machine that copies $|0\rangle$ and $|1\rangle$ has *no choice* but to send the superposition to that sum.

But what did I *want*? A genuine clone of $|\psi\rangle$, namely $|\psi\rangle|\psi\rangle$ in the two visible registers. Let me multiply that out:
$$|\psi\rangle|\psi\rangle = (a|0\rangle+b|1\rangle)(a|0\rangle+b|1\rangle) = a^2|0\rangle|0\rangle + ab\,|0\rangle|1\rangle + ab\,|1\rangle|0\rangle + b^2|1\rangle|1\rangle.$$
Put the two side by side. Linearity gave me
$$a\,|0\rangle|0\rangle|A_0\rangle + b\,|1\rangle|1\rangle|A_1\rangle,$$
and cloning demands
$$\big(a^2\,|0\rangle|0\rangle + ab\,|0\rangle|1\rangle + ab\,|1\rangle|0\rangle + b^2\,|1\rangle|1\rangle\big)|A_\psi\rangle.$$
These are flatly different vectors. The forced output has *no cross terms* in the copy registers — no $|0\rangle|1\rangle$, no $|1\rangle|0\rangle$ — while the true clone has both whenever $a$ and $b$ are nonzero. Whatever the apparatus does, it cannot hide those missing visible components. The condition $ab=0$ already kills every genuine superposition; up to the irrelevant phase of a basis vector, only the two basis states I started from survive.

And I can *see* what went wrong, not just that it went wrong. Cloning wants to square the state, and the cross term $ab\,(|01\rangle+|10\rangle)$ is exactly the part of "squaring" that a linear map cannot supply — linearity carries each branch $a|0\rangle$ and $b|1\rangle$ along its own track and never lets them multiply each other. In the no-apparatus shorthand the forced output is $a|00\rangle+b|11\rangle$, an entangled state of system-and-copy; the true clone $|\psi\rangle|\psi\rangle$ is a product. The machine can't even get the gross structure right. So there it is — the FLASH multiplier is the flaw. Linearity forbids it.

Let me redo this once in the polarization language to be sure it isn't an artifact of how I labeled things, since the manuscript I'm rebutting is about photons. A copier acting on polarization would do $|s\rangle|A_{\mathrm{ready}}\rangle \to |ss\rangle|A_s\rangle$, $A_{\mathrm{ready}}$ the apparatus before, $A_s$ after. Say it works on vertical and horizontal: $|\!\updownarrow\rangle \to |\!\updownarrow\updownarrow\rangle$, $|\!\leftrightarrow\rangle \to |\!\leftrightarrow\leftrightarrow\rangle$. Feed it $\alpha|\!\updownarrow\rangle + \beta|\!\leftrightarrow\rangle$. By linearity the output is
$$\alpha|\!\updownarrow\updownarrow\rangle|A_\updownarrow\rangle + \beta|\!\leftrightarrow\leftrightarrow\rangle|A_\leftrightarrow\rangle.$$
The honest clone would be
$$(\alpha|\!\updownarrow\rangle+\beta|\!\leftrightarrow\rangle)(\alpha|\!\updownarrow\rangle+\beta|\!\leftrightarrow\rangle) = \alpha^2|\!\updownarrow\updownarrow\rangle + \beta^2|\!\leftrightarrow\leftrightarrow\rangle + \sqrt2\,\alpha\beta\,|\!\updownarrow\leftrightarrow\rangle_{\mathrm{sym}},$$
where $|\!\updownarrow\leftrightarrow\rangle_{\mathrm{sym}}=(|\!\updownarrow\rangle|\!\leftrightarrow\rangle+|\!\leftrightarrow\rangle|\!\updownarrow\rangle)/\sqrt2$. The $\sqrt2$ is exactly the normalization of the symmetric one-vertical-one-horizontal two-photon component. Same story: the linearly-forced output lacks the $\alpha\beta$ cross term and has $\alpha,\beta$ where the clone needs $\alpha^2,\beta^2$. Identical obstruction. Good — it's not about spin versus polarization, it's about linearity versus squaring.

Now I'm uneasy in the opposite direction. I picked a basis $\{|0\rangle,|1\rangle\}$ and assumed the machine copies *those* two perfectly, then watched a superposition break it. But that's a slightly rigged setup — I chose which two states to grant it for free. I'd like an argument that doesn't privilege any basis and tells me, cleanly, *exactly which* pairs of states a single machine could ever copy. There ought to be a basis-free version, and it should fall out of the one property of $U$ I haven't fully used yet: not just linearity, but **unitarity** — the preservation of inner products.

Inner products are the invariants of unitary evolution: $\langle U\Phi | U\Phi'\rangle = \langle\Phi|\Phi'\rangle$ for any two states. So take *two arbitrary unknown* states $|\psi\rangle$ and $|\phi\rangle$ — no basis assumptions, nothing copied "for free" — and suppose first that the one machine copies both in the stripped notation with no leftover apparatus:
$$U|\psi\rangle|\text{blank}\rangle = |\psi\rangle|\psi\rangle, \qquad U|\phi\rangle|\text{blank}\rangle = |\phi\rangle|\phi\rangle.$$
Compute the inner product of the two *inputs*. The blanks are the same fixed register, $\langle\text{blank}|\text{blank}\rangle = 1$, so
$$\big(\langle\psi|\langle\text{blank}|\big)\big(|\phi\rangle|\text{blank}\rangle\big) = \langle\psi|\phi\rangle \cdot \langle\text{blank}|\text{blank}\rangle = \langle\psi|\phi\rangle.$$
Now compute the inner product of the two *outputs*:
$$\big(\langle\psi|\langle\psi|\big)\big(|\phi\rangle|\phi\rangle\big) = \langle\psi|\phi\rangle \cdot \langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2.$$
But $U$ is unitary, so the input inner product equals the output inner product. Therefore
$$\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2.$$
Write $x = \langle\psi|\phi\rangle$: $x = x^2$, i.e. $x(1-x) = 0$, so $x = 0$ or $x = 1$. In this phase-fixed stripped notation, either $\langle\psi|\phi\rangle = 1$, meaning the two kets are the same state vector, or $\langle\psi|\phi\rangle = 0$, meaning they're **orthogonal**. There is nothing in between.

If I keep the apparatus end states, the same calculation gives
$$\langle\psi|\phi\rangle=\langle\psi|\phi\rangle^2\langle A_\psi|A_\phi\rangle.$$
Taking magnitudes, $s=|\langle\psi|\phi\rangle|$ obeys $s=s^2|\langle A_\psi|A_\phi\rangle|$. Since $|\langle A_\psi|A_\phi\rangle|\le 1$, any $0<s<1$ would require $1=s|\langle A_\psi|A_\phi\rangle|\le s<1$, a contradiction. So the physical conclusion is $|\langle\psi|\phi\rangle|\in\{0,1\}$: same ray or orthogonal ray, and no non-orthogonal pair in between.

That's the clean statement I wanted, and it's sharper than the superposition argument. A single unitary machine can faithfully copy a set of states *only if those states are mutually orthogonal*. Hand it any two states that overlap without being identical — any non-orthogonal pair — and unitarity is violated; the machine can't exist. And the unknown state Bob receives is, generically, non-orthogonal to the basis he'd measure in. So no universal copier. The two arguments agree and illuminate each other: the superposition argument shows the breakage concretely in a chosen basis; the inner-product argument shows *why*, basis-free, and pins the exact boundary — overlap is the obstruction, orthogonality is the escape.

I should pause on that escape clause, because it looks at first like it threatens the whole conclusion. Orthogonal states *can* be cloned. Indeed they can — there's an explicit machine. The controlled-NOT, which sends $|x\rangle|0\rangle \to |x\rangle|x\rangle$ for $x \in \{0,1\}$, copies the computational basis perfectly; it just doesn't copy *superpositions* of it (apply it to $\tfrac{1}{\sqrt2}(|0\rangle+|1\rangle)|0\rangle$ and you get the entangled $\tfrac{1}{\sqrt2}(|00\rangle+|11\rangle)$, not $|+\rangle|+\rangle$ — exactly the forced-output failure again). So the theorem is not "nothing can be copied." It is: *an arbitrary unknown state cannot be copied; a known orthonormal set can.* The discriminating word is **unknown**.

And that finally dissolves a puzzle that should have been nagging the whole time. Classical information gets copied constantly — bits, files, letters — yet classical systems are made of quantum stuff, so how can they escape a quantum prohibition? The inner-product condition answers it precisely. Distinct classical symbols are perfectly distinguishable, which is the same as saying they are *mutually orthogonal* quantum states. They sit squarely in the $\langle\psi|\phi\rangle = 0$ exception. Copying classical information is copying an orthogonal set — always allowed. No clash. The prohibition bites only on *non-orthogonal* alternatives, which is exactly the regime with no classical analogue. A copier is forbidden precisely where it would be quantum-mechanically novel.

It's worth seeing why a copier would have been so extravagantly powerful, because that tells me what the no-go is really protecting. If I could clone an unknown $|\psi\rangle$, I'd defeat the disturbance of measurement: make a thousand copies, measure them in every basis, and reconstruct $a$ and $b$ to arbitrary precision — all while the original sits untouched. That would be a nondisturbing determination of an unknown state, which the structure of quantum measurement is supposed to forbid. So no-cloning is the flip side of "you cannot measure an unknown state without disturbing it." The two are the same prohibition wearing different clothes: cloning would buy you disturbance-free measurement, and disturbance-free measurement would let you signal. The linearity of $U$ is the single fact underneath all three.

Let me close the loop back to the manuscript and to relativity, since that's where I started. The reason FLASH cannot signal is *not* a separate relativistic miracle bolted on after the fact — it's this same linearity, viewed at Bob's end. Bob's machine is fixed (he doesn't know Alice's axis), so by the superposition argument it cannot clone the unknown collapsed state, and his burst is the wrong, entangled state rather than $N$ honest copies; the histogram he'd read Alice's choice from never materializes. I can also say it at the level of his density matrix. For the singlet, before Alice sends any ordinary message, Bob has
$$\rho_B=\mathrm{Tr}_A(\rho_{AB})=I/2.$$
If Alice measures $z$ and her result is not communicated, Bob's ensemble is $\tfrac12|z_+\rangle\langle z_+|+\tfrac12|z_-\rangle\langle z_-|=I/2$. If Alice measures $x$, it is $\tfrac12|x_+\rangle\langle x_+|+\tfrac12|x_-\rangle\langle x_-|=I/2$. Her choice changes only the decomposition of the same $\rho_B$, not $\rho_B$ itself. Any allowed local operation $\mathcal{E}$ on Bob's side is linear, so it produces $\mathcal{E}(\rho_B)$ in both cases. A perfect cloner would instead act as if it could map each pure component separately into $|\psi\rangle|\psi\rangle$ and thereby turn the $z$ and $x$ decompositions of $I/2$ into different two-particle mixtures. That dependence on the decomposition, rather than on the density matrix, is exactly nonlinearity. So no information crosses. Cloning is nonlinear on states (it squares them); no-signaling, no-cloning, and the linearity of quantum mechanics are one fact seen three ways. Relativity is safe because the dynamics is linear.

A couple of relatives suggest themselves once I see it this way. If I can't *create* a copy, can I *delete* one — given two identical copies $|\psi\rangle|\psi\rangle$, run a single fixed unitary that sends $|\psi\rangle|\psi\rangle \to |\psi\rangle|\Sigma\rangle$ for unknown $\psi$, dumping the second into the standard blank? That's the time-reverse of cloning, and the same inner-product bookkeeping run backwards forbids it: $|\psi\rangle|\psi\rangle$ has overlap $\langle\psi|\phi\rangle^2$ with $|\phi\rangle|\phi\rangle$, while the targets $|\psi\rangle|\Sigma\rangle$ and $|\phi\rangle|\Sigma\rangle$ have overlap $\langle\psi|\phi\rangle$, and a unitary can't reconcile $\langle\psi|\phi\rangle^2$ with $\langle\psi|\phi\rangle$ for a non-orthogonal pair — a single linear map can't un-square the state either. And if I weaken the demand from two pure copies to merely two parties each holding a *marginal* equal to the original — broadcasting rather than cloning — the obstruction survives in density-matrix form: a set of states can be broadcast only if they commute (are simultaneously diagonalizable, i.e. "classical"), and non-commuting states cannot. That commute/non-commute line is the exact density-matrix echo of orthogonal/non-orthogonal. The pure-state result is the sharp tip of this family.

There's also a payoff I can already see, the mirror image of the FLASH failure. Since non-orthogonal states resist copying, suppose I *deliberately* encode information in non-orthogonal alternatives — say bits carried by photons prepared in either the rectilinear pair $\{|\!\updownarrow\rangle,|\!\leftrightarrow\rangle\}$ or the diagonal pair, with the choice of pair kept secret. An eavesdropper who intercepts a carrier cannot clone it to keep a copy while forwarding the original — the cloner that would do that doesn't exist — and any attempt to read it in the wrong basis disturbs it, which is the same prohibition wearing its measurement face. So tampering is forced to leave a detectable trace. No-cloning is not just a restriction; it's a resource, the thing that makes a quantum channel's secrecy checkable. That is the foundation key distribution would rest on.

The precise result I land on is this: there is no unitary $U$ on the system, a target register, and any machine ancilla, with fixed blank $|\Sigma\rangle$ and fixed ready state $|A_{\mathrm{ready}}\rangle$, such that
$$U\big(|\psi\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle\big)=|\psi\rangle|\psi\rangle|A_\psi\rangle$$
for every normalized $|\psi\rangle$. The simpler no-machine notation is the special case $U(|\psi\rangle|\Sigma\rangle)=|\psi\rangle|\psi\rangle$.

The linearity proof is the forced-output calculation. If $U$ copies two orthonormal states, $U|0\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=|0\rangle|0\rangle|A_0\rangle$ and $U|1\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=|1\rangle|1\rangle|A_1\rangle$. For $|\psi\rangle=a|0\rangle+b|1\rangle$, linearity gives
$$U|\psi\rangle|\Sigma\rangle|A_{\mathrm{ready}}\rangle=a|0\rangle|0\rangle|A_0\rangle+b|1\rangle|1\rangle|A_1\rangle.$$
A true clone requires
$$|\psi\rangle|\psi\rangle|A_\psi\rangle=\big(a^2|0\rangle|0\rangle+ab|0\rangle|1\rangle+ab|1\rangle|0\rangle+b^2|1\rangle|1\rangle\big)|A_\psi\rangle.$$
For any genuine superposition, $ab\ne0$, the required $|0\rangle|1\rangle$ and $|1\rangle|0\rangle$ components are present on the right and absent on the left. So no single linear $U$ copies the superpositions of a copied orthonormal pair. $\square$

The inner-product proof is even shorter. In the stripped notation, suppose $U|\psi\rangle|\Sigma\rangle=|\psi\rangle|\psi\rangle$ and $U|\phi\rangle|\Sigma\rangle=|\phi\rangle|\phi\rangle$. Unitarity preserves inner products, so
$$\langle\psi|\phi\rangle\langle\Sigma|\Sigma\rangle=\langle\psi|\phi\rangle\langle\psi|\phi\rangle,$$
hence
$$\langle\psi|\phi\rangle=\langle\psi|\phi\rangle^2.$$
Thus $\langle\psi|\phi\rangle\in\{0,1\}$ in a fixed phase convention: the two kets are identical or orthogonal. With machine end states allowed, the equation is $\langle\psi|\phi\rangle=\langle\psi|\phi\rangle^2\langle A_\psi|A_\phi\rangle$, and taking magnitudes still forces $|\langle\psi|\phi\rangle|\in\{0,1\}$. A single $U$ can clone at most a mutually orthogonal set; no $U$ clones a non-orthogonal pair, so no universal cloner exists. $\square$

The causal chain, end to end: closed-system quantum evolution is linear and unitary; a copier would have to be a single such $U$ producing $|\psi\rangle|\psi\rangle$, which is quadratic in the state, for every $\psi$; linearity carries each superposition branch separately and cannot generate the cross term that squaring needs, so a machine that copies a basis fails on every superposition of it; equivalently, unitarity preserves the input inner product $\langle\psi|\phi\rangle$ while perfect copying would square it to $\langle\psi|\phi\rangle^2$, forcing $\langle\psi|\phi\rangle\in\{0,1\}$ in the stripped case and $|\langle\psi|\phi\rangle|\in\{0,1\}$ physically — only orthogonal, perfectly distinguishable states are copyable. The same linearity makes Bob's reduced density matrix independent of Alice's choice, so the would-be superluminal signal never appears.

```python
import numpy as np

# States are unit vectors in C^2; the unknown qubit is a|0> + b|1>.
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

# A machine that copies the basis {|0>,|1>} onto a blank |0> is CNOT:
#   |0>|0> -> |0>|0>,  |1>|0> -> |1>|1>.   It is the best a fixed linear box can do.
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
assert np.allclose(CNOT.conj().T @ CNOT, np.eye(4))

# Linearity argument: feed it a superposition; the output is FORCED, and it is
# not the clone.
a = b = 1 / np.sqrt(2)
psi   = a * ket0 + b * ket1                 # |+>
forced = CNOT @ np.kron(psi, ket0)          # a|00> + b|11>   (entangled)
clone  = np.kron(psi, psi)                  # a^2|00>+ab|01>+ab|10>+b^2|11>
assert not np.allclose(forced, clone)       # cross terms missing -> cloning fails

# Inner-product argument: a unitary cloner would need <psi|phi> = <psi|phi>^2,
# impossible for a non-orthogonal pair like |0> and |+>.
phi = (ket0 + ket1) / np.sqrt(2)            # <0|+> = 1/sqrt(2)
ip  = np.vdot(ket0, phi)
assert not np.isclose(ip, ip * ip)          # 1/sqrt(2) != 1/2 -> not clonable
```
