Let me start from the thing that actually bothers me. Shannon settled classical communication: a channel `p(y|x)` transmits at most `max_P I(X;Y)` bits per use, and a symbol with `m` clearly separated levels carries `log m` bits. I will measure every entropy in bits, so `log` is base 2. Clean, finished. But the signals we now push down a fiber are not classical symbols with `m` levels — they are quantum states. A single qubit, if I prepare it in a pure state, is a point on the Bloch sphere, and the Bloch sphere is a *continuum*. There are uncountably many directions a qubit can point. So naively I should be able to take a real number `θ ∈ [0,1)`, encode it as the angle of a qubit, send one qubit, and have transmitted infinitely many classical bits in one shot. That can't be right. But where exactly does it break?

It breaks at reading. To get classical information out of the carrier the receiver has to *measure* it, and quantum measurement is not a passive look — non-orthogonal states are not perfectly distinguishable. If I encode `θ` as `cos θ|0⟩ + sin θ|1⟩`, two nearby angles give nearly parallel states, and no measurement can reliably tell them apart. So the continuum is there in the *preparation* but it is not there in the *readout*. The honest quantity is operational: over all the ways the sender can encode `x` (with probability `p_x`) into states `ρ_x`, and over *all* measurements the receiver might apply, what is the largest classical mutual information `I(X;Y)` between the message `X` and the outcome `Y`? Is it bounded, and by what?

Let me get the setup exactly right, because the slipperiness is in the "all measurements." The sender has an ensemble `E = {p_x, ρ_x}`, `ρ_x` a density operator on a `d`-dimensional space. The receiver picks a measurement and gets a classical result `Y`. The most general measurement is a POVM: a set `{E_y}` of positive operators with `Σ_y E_y = I`, giving outcome `y` with probability `Tr(E_y ρ)`. So for a *fixed* measurement the carrier looks like an ordinary classical channel, `p(y|x) = Tr(E_y ρ_x)`, and Shannon's `I(X;Y)` is well-defined for it. The trouble is that `{E_y}` is the receiver's free choice — any number of outcomes, possibly even a collective measurement across many carriers — and I need a statement that holds for *all* of them at once. The accessible information is `Acc(E) = max over POVMs of I(X;Y)`. I want to upper-bound that maximum.

First let me pin the two extremes so I know what a correct answer must reproduce. If the `ρ_x` are mutually orthogonal — say `|0⟩` and `|1⟩` — the receiver projects onto each support, gets `y = x` with certainty, `H(X|Y) = 0`, so `I(X;Y) = H(X)`. Perfectly distinguishable: he learns the whole message. At the other end, if the states are non-orthogonal, no measurement nails them, `H(X|Y) > 0`, and `I(X;Y) < H(X)` strictly. And I already feel the mechanism of the "free lunch" failing: take a qubit and spread the alphabet over more and more points of the Bloch sphere. Each extra point makes the *encoding* richer, `H(X)` grows — but the points crowd together and become *less* distinguishable, so `H(X|Y)` grows too. The two effects fight. The whole question is whether they exactly cancel at some ceiling. I want to find that ceiling and prove the cancellation.

So what is the right quantity to compare `I(X;Y)` against? Let me try the obvious guess and watch it fail, because the failure tells me what to subtract. Guess: the ceiling is `S(ρ̄)`, the von Neumann entropy of the average carrier state `ρ̄ = Σ_x p_x ρ_x`. Motivation: `S(ρ̄)` is how much "entropy" the carrier holds, and information about `x` ought to live in that entropy. For *pure* signal states this feels right — if each `ρ_x = |φ_x⟩⟨φ_x|` is pure, all the entropy of `ρ̄` comes from the spread over `x`. But suppose each `ρ_x` is itself badly mixed — say every `ρ_x` is the *same* maximally mixed state `I/d`. Then the carrier tells you nothing about `x` (the receiver sees `I/d` no matter what was sent), so the accessible information is `0`. Yet `S(ρ̄) = S(I/d) = log d`, the maximum. So `S(ρ̄)` massively over-counts: it counts the internal noise inside each signal state as if it were information about the message. The fix is forced — I must subtract that internal noise off, on average. The internal entropy of signal `x` is `S(ρ_x)`, weighted by `p_x`. So the corrected candidate is

  `χ = S(ρ̄) − Σ_x p_x S(ρ_x) = S(Σ_x p_x ρ_x) − Σ_x p_x S(ρ_x)`.

In the all-same-mixed-state case this is `log d − log d = 0`. Good. And for pure signals the subtracted term is zero and `χ = S(ρ̄)`, matching the earlier intuition. The shape of `χ` is "entropy of the mixture minus average entropy of the parts," and that is exactly the von Neumann analogue of a mutual information. Let me make that precise, because if `χ` really is a mutual information then I have a chance of bounding `I(X;Y)` by it with general principles instead of brute force.

Here is the device that makes it precise: fold the classical message and the quantum carrier into a single quantum state. Introduce a register `X` that just records which message was sent — an orthonormal set `{|x⟩}` — and write the joint *classical-quantum* state

  `ρ_XA = Σ_x p_x |x⟩⟨x| ⊗ ρ_x`.

This is block-diagonal in `X`: the `x`-th block is `p_x ρ_x`. Its marginals and joint entropy are computable directly. `S(X) = S(Σ_x p_x |x⟩⟨x|) = H(p)`, the Shannon entropy of the priors, because `X` is just a classical distribution. `S(A) = S(Tr_X ρ_XA) = S(Σ_x p_x ρ_x) = S(ρ̄)`. For the joint entropy, the eigenvalues of a block-diagonal `Σ_x (p_x ρ_x)` are `{p_x λ}` where `λ` ranges over eigenvalues of `ρ_x`; so `S(XA) = −Σ_x Σ_λ p_x λ log(p_x λ) = −Σ_x p_x log p_x − Σ_x p_x Σ_λ λ log λ = H(p) + Σ_x p_x S(ρ_x)`. Therefore the quantum mutual information of this cq state is

  `I(X;A) = S(X) + S(A) − S(XA) = H(p) + S(ρ̄) − H(p) − Σ_x p_x S(ρ_x) = S(ρ̄) − Σ_x p_x S(ρ_x) = χ`.

So `χ` *is* the quantum mutual information `I(X;A)` between the message register and the carrier, before any measurement. That is a real foothold. As a mutual information it is automatically `≥ 0` — and I can double-check that against concavity of the entropy: `S(Σ p_x ρ_x) ≥ Σ p_x S(ρ_x)` is exactly concavity, so `χ ≥ 0` independently. Two routes to the same nonnegativity; the object is sound.

Now the claim I want is `I(X;Y) ≤ χ = I(X;A)`. Look at what those two quantities are. `I(X;A)` is the correlation between the message and the *quantum carrier as it is*. `I(X;Y)` is the correlation between the message and the *classical record the receiver produces by measuring the carrier*. The measurement sits *downstream* of the carrier — `Y` is manufactured from `A`. So morally this should be a "you can't get more correlation with `X` out of `A` by processing `A` than was already in `A`" statement. Data-processing. If I had a clean principle that says *acting on one half of a correlated pair with a channel cannot increase its mutual information with the other half*, I would be done in one line. Let me see whether I can lean on that — and whether I am even entitled to, because the carrier-to-outcome step has to be expressible as a channel.

It is. A POVM `{E_y}` can always be realized as a channel that writes the outcome into a fresh register. Factor each `E_y = M_y† M_y` (take `M_y = √E_y`), and define the map on the carrier

  `A → AY: ρ ↦ Σ_y M_y ρ M_y† ⊗ |y⟩⟨y|`.

This is completely positive and trace-preserving: trace-preserving because `Σ_y M_y† M_y = Σ_y E_y = I`. It does exactly what a measurement does — the diagonal of the `Y` register is the outcome distribution `Tr(E_y ρ)` — while keeping the post-measurement carrier around in `A`. Apply it to the `A` half of `ρ_XA`:

  `ρ'_XAY = Σ_{x,y} p_x |x⟩⟨x| ⊗ M_y ρ_x M_y† ⊗ |y⟩⟨y|`.

If I now trace out `A`, the `X`–`Y` marginal is `Σ_{x,y} p_x Tr(E_y ρ_x) |x⟩⟨x| ⊗ |y⟩⟨y|` — a *classical* joint distribution `p(x,y) = p_x Tr(E_y ρ_x)`, precisely the channel the receiver sees. So the receiver's information gain is the classical `I(X;Y)` read off the `X`–`Y` marginal of `ρ'`. Everything I want lives in these two states `ρ_XA` and `ρ'_XAY`.

The chain I want is

  `I(X;Y)_{ρ'} ≤ I(X;AY)_{ρ'} ≤ I(X;A)_{ρ} = χ`.

The right step `I(X;AY) ≤ I(X;A)` would say: going from `ρ` to `ρ'` I act on the carrier side by a channel `A → AY`, leaving `X` untouched, so that local processing does not raise the mutual information with `X`. The left step `I(X;Y) ≤ I(X;AY)` would say: inside `ρ'`, going from `AY` down to `Y` means discarding `A`, so throwing away part of the receiver's system does not increase its correlation with `X`. If both statements are available, they sandwich the bound.

Here is where I have to be honest with myself about what I am allowed to assume. The slick proof I just sketched leans on the monotonicity of quantum mutual information under channels — equivalently, the monotonicity of quantum *relative entropy*, `D(N(ρ)‖N(σ)) ≤ D(ρ‖σ)` — equivalently strong subadditivity, `S(ABC) + S(B) ≤ S(AB) + S(BC)`. Those are deep facts about von Neumann entropy, and as general theorems they are not in hand. The classical versions are one-liners from convexity, but the quantum statements are genuinely hard and not established. So I cannot just *invoke* them. I have to prove the *one instance I actually need* — the data-processing inequality for this classical-quantum situation — directly, from the structure of `ρ_XA` and the measurement channel, without the general machine.

So let me try to prove `I(X;Y) ≤ I(X;A)` by hand for the cq case, and see what's really needed. Write `I(X;A)` in its relative-entropy form. For the cq state `ρ_XA = Σ_x p_x |x⟩⟨x| ⊗ ρ_x`, the product of marginals is `ρ_X ⊗ ρ_A = Σ_x p_x |x⟩⟨x| ⊗ ρ̄`, and a short computation gives

  `I(X;A) = D(ρ_XA ‖ ρ_X ⊗ ρ_A) = Σ_x p_x D(ρ_x ‖ ρ̄)`.

That is a clean reduction: the message-carrier correlation is the average, over messages, of how far each signal state `ρ_x` sits from the average state `ρ̄`, measured in quantum relative entropy. Now the receiver's side. After the measurement the conditional outcome distribution given `x` is `q_x(y) = Tr(E_y ρ_x)` and the average outcome distribution is `q̄(y) = Tr(E_y ρ̄)`, and the classical mutual information is, in the very same shape,

  `I(X;Y) = Σ_x p_x D(q_x ‖ q̄)`,

a per-message average of *classical* relative entropies between the outcome distribution for `x` and the average outcome distribution. Set the two side by side, message by message: I need `D(q_x ‖ q̄) ≤ D(ρ_x ‖ ρ̄)` for each `x`. But `q_x` and `q̄` are obtained from `ρ_x` and `ρ̄` by *the same* measurement — `q_x(y) = Tr(E_y ρ_x)`, `q̄(y) = Tr(E_y ρ̄)`. So the whole bound reduces to a single, self-contained statement:

  *measuring two quantum states with the same POVM cannot make their outcome distributions more distinguishable than the states were* — `D(\{Tr(E_y ρ)\} ‖ \{Tr(E_y σ)\}) ≤ D(ρ ‖ σ)`.

This is exactly monotonicity of relative entropy, but only for the special map "apply a fixed POVM," and *that* I can attack directly. I do not need to appeal to any optimal-testing interpretation. I can use the variational form of quantum relative entropy, with the same base-2 logarithm convention,

  `D(ρ‖σ) = sup_K {Tr(ρK) − log Tr(2^K σ)}`,

where `K` ranges over Hermitian observables and the usual support convention is understood. First suppose the measurement is projective, `{P_y}`. Write `q_y = Tr(P_y ρ)` and `r_y = Tr(P_y σ)`, omitting zero-probability terms. If I choose

  `K = Σ_y log(q_y/r_y) P_y`,

then `Tr(ρK) = Σ_y q_y log(q_y/r_y) = D(q‖r)`, while

  `Tr(2^K σ) = Tr((Σ_y (q_y/r_y) P_y) σ) = Σ_y (q_y/r_y) r_y = 1`.

The variational formula therefore gives `D(ρ‖σ) ≥ D(q‖r)`. A general POVM is no harder: Naimark dilation gives an isometry `W` and projectors `{P_y}` on a larger space with `Tr(E_y ρ) = Tr(P_y WρW†)` and `Tr(E_y σ) = Tr(P_y WσW†)`, and an isometry preserves relative entropy because it preserves the nonzero spectrum and the logarithms on the support. So the same inequality holds for every POVM:

  `D(\{Tr(E_y ρ)\} ‖ \{Tr(E_y σ)\}) ≤ D(ρ ‖ σ)`.

Now apply this with `ρ = ρ_x` and `σ = ρ̄`. For each `x`, `D(q_x‖q̄) ≤ D(ρ_x‖ρ̄)`. Averaging with weights `p_x` gives `I(X;Y) ≤ I(X;A) = χ`. This is the cq data-processing instance I need, proved directly for the measurement, rather than borrowed from a general channel theorem.

The two-step chain is still useful because it checks the directions of the losses:

  `I(X;Y)_{ρ'} ≤ I(X;AY)_{ρ'} ≤ I(X;A)_{ρ} = χ`.

The left inequality is the discard-`A` step; in entropy language it is `I(X;AY) − I(X;Y) = I(X;A|Y) ≥ 0`, the strong-subadditivity form. The right inequality is the local data-processing step for the instrument `A → AY`. I am not using those two general statements as assumptions here; the direct POVM contraction above has already supplied the measured cq bound, but the labels keep the proof chain's direction straight.

The isometric bookkeeping gives the same check with no ambiguity about what is preserved. Regard the POVM as the classical-output channel `Φ(M) = Σ_y Tr(E_y M) |y⟩⟨y|` and take a Stinespring dilation: an isometry `V: A → Y⊗Z` with `Φ(M) = Tr_Z(V M V†)`. Build `σ = ρ_XA = Σ_x p_x |x⟩⟨x| ⊗ ρ_x` and `ξ = (1_X ⊗ V) σ (1_X ⊗ V)†`, a state on `X⊗Y⊗Z`. Because `V` is an isometry it changes nothing about the `X`-marginal or the joint spectrum:

  `S(ξ_X) = S(σ_X) = H(p)`,
  `S(ξ_{XYZ}) = S(σ_{XA}) = H(p) + Σ_x p_x S(ρ_x)`,
  `S(ξ_{YZ}) = S(σ_A) = S(Σ_x p_x ρ_x) = S(ρ̄)`.

Hence the mutual information of `X` with the *whole* output `YZ` is

  `I(X;YZ)_ξ = S(ξ_X) + S(ξ_{YZ}) − S(ξ_{XYZ}) = H(p) + S(ρ̄) − H(p) − Σ_x p_x S(ρ_x) = S(ρ̄) − Σ_x p_x S(ρ_x) = χ`.

That equals `χ` exactly, as it must, since the full isometric output `YZ` is just the old carrier written in a larger reversible form. The only comparison left after discarding `Z` is the three-system entropy inequality `I(X;YZ) ≥ I(X;Y)`, the same discard-a-system direction as before. For the present derivation, the direct POVM contraction already supplies this measured cq comparison, so `I(X;Y)_ξ ≤ I(X;YZ)_ξ = χ`. The `X`–`Y` marginal of `ξ` is `Σ_{x,y} p_x Tr(E_y ρ_x) |x⟩⟨x| ⊗ |y⟩⟨y|`, i.e. the classical joint `p(x,y) = p_x Tr(E_y ρ_x)`, whose mutual information is the receiver's information gain for this measurement. Since the measurement was arbitrary,

  `Acc(E) = max over POVMs of I(X;Y) ≤ χ(E) = S(Σ_x p_x ρ_x) − Σ_x p_x S(ρ_x)`.

There it is. The continuum of the Bloch sphere never enters the bound: `χ` is built from `d` finite-dimensional density operators, and whatever the receiver does, the classical correlation he extracts is capped by this single intrinsic number.

Now the payoff I started this for. Bound `χ` itself. The subtracted term `Σ_x p_x S(ρ_x) ≥ 0` since every `S(ρ_x) ≥ 0`, so `χ ≤ S(ρ̄)`. And `ρ̄` is a density operator on the carrier, so `S(ρ̄) ≤ log d`, with the maximum at the maximally mixed state. For `n` qubits, `d = 2^n`, `log_2 d = n`. Therefore

  `Acc(E) ≤ χ(E) ≤ S(ρ̄) ≤ log d = n`.

A qubit is not a free lunch. No matter how cleverly the sender exploits the continuous Hilbert space — a vast alphabet smeared over the Bloch sphere, mixed states, anything — and no matter how powerful the receiver's measurement — collective measurements across all `n` carriers at once — the message and the outcome can share at most `n` bits. One qubit, at most one bit. The continuum in the preparation is real, but it is not *accessible*; it drains away into indistinguishability at exactly the rate that keeps the extractable information at `n`.

Is the bound tight, or did I leave slack? Take the orthogonal case. If the `ρ_x` have orthogonal supports, the receiver projects onto those supports, learns `x` perfectly, and achieves `I(X;Y) = H(X)`. The average state is block diagonal across the same supports, so its entropy splits as `S(ρ̄) = H(p) + Σ_x p_x S(ρ_x)`. Therefore `χ = S(ρ̄) − Σ_x p_x S(ρ_x) = H(p) = H(X)`, and `Acc = χ = H(X)`. With `d` equally likely orthogonal pure states this reaches `log d`, so the dimensional ceiling itself is tight. For non-orthogonal alphabets the full-message equality disappears because the receiver cannot identify `x` with certainty; `χ` remains the computable ceiling even when the exact accessible information has no simple closed form.

So the whole chain, end to end. A qubit's Hilbert space is continuous, which *looks* like unbounded classical capacity, but reading requires measurement and non-orthogonal states are not perfectly distinguishable, so the operative quantity is the accessible information `Acc(E) = max over POVMs of I(X;Y)`. Comparing `I(X;Y)` to the carrier's own entropy `S(ρ̄)` over-counts the internal noise of mixed signals, which forces the corrected quantity `χ = S(ρ̄) − Σ_x p_x S(ρ_x)`; encoding the classical message in a register `X` and the carrier in `A` reveals `χ = I(X;A)`, the message-carrier mutual information. For a fixed measurement, `I(X;A) = Σ_x p_x D(ρ_x‖ρ̄)` and `I(X;Y) = Σ_x p_x D(q_x‖q̄)`, and the direct variational argument for POVMs proves `D(q_x‖q̄) ≤ D(ρ_x‖ρ̄)` message by message. That yields `Acc(E) ≤ χ(E)`, and since `χ ≤ S(ρ̄) ≤ log d = n`, an `n`-qubit carrier conveys at most `n` classical bits; orthogonal signal states attain the full-message equality `Acc = χ = H(X)`.

I keep a tiny numerical harness only to sanity-check the inequality on concrete ensembles — orthogonal states should saturate it, the three symmetric Bloch states with `ρ̄ = I/2` should sit strictly below `χ = 1` — never as a proof, only to catch an arithmetic slip in `χ`.

```python
import numpy as np

def von_neumann_entropy(rho):
    # S(rho) = -Tr rho log2 rho via eigenvalues; drop ~0 eigenvalues
    w = np.linalg.eigvalsh(rho).real
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))

def shannon_entropy(p):
    p = np.asarray(p, float); p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))

def holevo_chi(states, priors):
    # chi = S(sum p_x rho_x) - sum p_x S(rho_x)   [the ensemble's intrinsic ceiling]
    rho_bar = sum(p * r for p, r in zip(priors, states))
    return von_neumann_entropy(rho_bar) - sum(p * von_neumann_entropy(r)
                                              for p, r in zip(priors, states))

def measured_mutual_info(states, priors, povm):
    # joint p(x,y) = p_x Tr(E_y rho_x)
    px = np.asarray(priors, float)
    joint = np.array([[p * np.real(np.trace(E @ r)) for E in povm]
                      for p, r in zip(px, states)])          # rows x, cols y
    py = joint.sum(axis=0)
    info = 0.0
    for x in range(len(states)):
        for y in range(len(povm)):
            if joint[x, y] > 1e-12:
                info += joint[x, y] * np.log2(joint[x, y] / (px[x] * py[y]))
    return float(info)

def check(states, priors, povms):
    chi = holevo_chi(states, priors)
    for povm in povms:                                       # every measurement obeys the ceiling
        assert measured_mutual_info(states, priors, povm) <= chi + 1e-9
    return chi

def projector(v):
    return v @ v.conj().T

ket0 = np.array([[1.0], [0.0]])
ket1 = np.array([[0.0], [1.0]])
P0, P1 = projector(ket0), projector(ket1)
ortho = check([P0, P1], [0.5, 0.5], [[P0, P1]])              # chi == 1 bit, attained
assert abs(measured_mutual_info([P0, P1], [0.5, 0.5], [P0, P1]) - ortho) < 1e-9

def real_qubit(theta):
    return np.array([[np.cos(theta)], [np.sin(theta)]])

trine = [projector(real_qubit(t)) for t in (0, 2*np.pi/3, 4*np.pi/3)]
trine_povm = [(2.0/3.0) * r for r in trine]
trine_chi = check(trine, [1/3, 1/3, 1/3], [trine_povm])       # chi == 1 bit, not attained here
assert abs(trine_chi - 1.0) < 1e-9
assert abs(measured_mutual_info(trine, [1/3, 1/3, 1/3], trine_povm) - 1/3) < 1e-9
```
