# No-Cloning Theorem

There is no fixed quantum evolution that can realize, for any unknown pure state `|psi>` and fixed blank state `|b>`,

`|psi>|b> -> |psi>|psi>`.

More precisely, if a unitary operation `U` can simultaneously copy two candidate states `|psi>` and `|phi>`:

`U|psi>|b> = |psi>|psi>`

`U|phi>|b> = |phi>|phi>`,

then these two states must be orthogonal or identical.

## Proof by inner products

Unitary evolution preserves inner products. The inner product between the input states is

`(<psi| tensor <b|)(|phi> tensor |b>) = <psi|phi>`.

If copying succeeds, the inner product between the output states should be

`(<psi| tensor <psi|)(|phi> tensor |phi>) = <psi|phi>^2`.

Therefore we must have

`<psi|phi> = <psi|phi>^2`.

This equation only allows `<psi|phi> = 0` or `<psi|phi> = 1`. The former means the candidate states are orthogonal, the latter means the two states are identical. Any two distinct, non-orthogonal states have `0 < |<psi|phi>| < 1`, and cannot be simultaneously copied by the same perfect copier. Therefore a universal copier does not exist.

## Proof by linearity

Suppose a copier can copy the computational basis states:

`|0>|b> -> |0>|0>`

`|1>|b> -> |1>|1>`.

Since quantum evolution is linear, its action on the superposition `a|0>+b|1>` must give

`a|0>|0> + b|1>|1>`.

But genuinely copying this superposition requires the output

`(a|0>+b|1>) tensor (a|0>+b|1>)`

`= a^2|0>|0> + ab|0>|1> + ab|1>|0> + b^2|1>|1>`.

These are generally not equal. The copying map takes a state vector to the tensor square of itself, which is nonlinear in the amplitudes; but a legitimate quantum evolution must be linear.

## Interpretation

The distinctive insight of the No-Cloning Theorem is: the inability to copy an arbitrary unknown quantum state is not a technological limitation, but a limitation of the structure of the space of quantum states itself. Inner-product preservation shows that copying would take the overlap between states from `<psi|phi>` to `<psi|phi>^2`; the superposition principle shows that being able to copy a set of orthogonal basis states does not mean being able to copy arbitrary superpositions of them.

This marks a fundamental shift from classical-information intuition to quantum-information intuition. Classical information can be regarded as distinguishable labels, so copying is the default operation. Quantum information, by contrast, is jointly constrained by non-orthogonal states, superposition, and linear evolution; an unknown state is not a label that can be freely read out and rewritten. Only sets of orthogonal, distinguishable states can be copied the way classical information is; the entire space of quantum states cannot be universally broadcast or backed up.
