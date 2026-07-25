## Research question

A default intuition about classical information is: if I possess a piece of information, I can copy it onto a blank carrier. Bit strings, files, mathematical descriptions can all be rewritten onto another medium without altering the original. The question is: does this intuition carry over unchanged to quantum states?

The question is not whether some known states can be copied. Orthogonal basis states can be copied via controlled operations, e.g. copying the computational basis states `|0>` and `|1>` onto a blank bit. The real question is whether there exists a universal physical process that, for any unknown pure state `|psi>` and a fixed blank state `|b>`, realizes

`|psi>|b> -> |psi>|psi>`.

## Background

The reversible evolution of a closed quantum system is described by a unitary transformation. A unitary transformation is linear and preserves the inner product between state vectors. Linearity means that if a process acts on `|0>` and `|1>` separately, then its action on a superposition `a|0>+b|1>` must be the same linear combination of the corresponding outputs.

A copying map also looks natural at first: given a blank register, write the input state into the blank register while preserving the original input. If we only consider a set of mutually orthogonal candidate states, this can be done, because orthogonal states can be perfectly distinguished and can be rewritten using conditional operations.

An arbitrary unknown quantum state is different. Non-orthogonal states cannot be perfectly distinguished; copying an arbitrary state requires a linear, inner-product-preserving physical evolution realizing the map `|psi> -> |psi>|psi>`, which would take the inner product between states from `<psi|phi>` to `<psi|phi>^2`.

## Baselines

- **Classical copier.** For classical bit strings, copying is allowed, because the set of states can be regarded as discrete, mutually distinguishable labels. This model assumes by default that information can be read out of a carrier and rewritten onto another carrier.

- **Orthogonal-state copier.** For a known orthogonal set, one can construct a copying operation, e.g. `|0>|0> -> |0>|0>` and `|1>|0> -> |1>|1>`. This operation holds for that orthogonal set.

- **Measure-then-reprepare.** Measuring the unknown state first and then preparing a copy according to the outcome is one possible strategy. Measurement can yield a classical result, from which the state can be reprepared.

- **State estimation.** Multiple identically distributed samples can let one estimate the preparation process and use that to reproduce an approximate copy of the state.

- **Hypothetical nonlinear copier.** If arbitrary nonlinear maps were allowed, one could formally write `|psi> -> |psi>|psi>`.

## Evaluation settings

The core criterion should cover any unknown pure state, not just some fixed orthogonal basis. The minimal proof only needs two non-orthogonal states `|psi>` and `|phi>`: if the same unitary copier could copy both simultaneously, inner-product preservation gives

`<psi|phi> = <psi|phi>^2`.

This only allows `<psi|phi>` to be `0` or `1`. So two distinct, non-orthogonal states already suffice to rule out a universal copier.

Another equivalent stress test is superposition. If a copier can copy `|0>` and `|1>`, linearity requires that it output `a|00>+b|11>` on `a|0>+b|1>`; but genuine two copies should be

`(a|0>+b|1>)(a|0>+b|1>) = a^2|00>+ab|01>+ab|10>+b^2|11>`.

These generally differ. This directly exhibits the relationship between copying arbitrary states and the superposition principle.

## Proof target

The final argument should state: there is no fixed linear quantum evolution that can map any unknown pure state, together with a fixed blank state, into two identical copies of that pure state. If copying a set of candidate states is allowed, these candidate states must be pairwise orthogonal or identical; any two distinct, non-orthogonal states cannot be simultaneously perfectly copied.

The proof should exhibit two intuitions together. First, inner-product preservation shows that copying would square the overlap between states, thereby breaking the geometric structure. Second, the superposition principle shows that "copying the basis vectors" does not automatically mean "copying every superposition of them," because a copying map is inherently not linear.

The point of the report is to show how the structural constraints of quantum evolution limit what operations can be performed on unknown quantum states — unknown quantum information is not a label that can be freely read and rewritten, but a physical state jointly constrained by linearity, inner products, and distinguishability.
