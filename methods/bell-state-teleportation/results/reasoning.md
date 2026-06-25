I start from the obstruction rather than from the trick. If Alice has a single system in an unknown qubit state `|psi> = alpha|0> + beta|1>`, I cannot let her learn enough classical data to reconstruct it elsewhere. A measurement that identifies an arbitrary pair of amplitudes is not available on one copy, and a universal copier is ruled out by linearity. If the state is to reach Bob exactly, the original at Alice must stop being an independent copy. So the problem is not how to copy an unknown state across space, but how to move it while consuming some other resource.

I then look at what an EPR pair can and cannot provide. A shared maximally entangled pair does not contain Alice's unknown state before the protocol begins. Bob's reduced state is maximally mixed, so he has no local signal and no hidden copy of `alpha` and `beta`. That is a useful constraint, not a defect, because it keeps anything I build compatible with relativity. The pair can still be useful if Alice performs an operation that involves both the unknown qubit and her half of the pair, since the nonlocal correlation is already distributed and only the online part needs to travel.

Let me take the shared resource to be

`|Phi+>_23 = (|00>_23 + |11>_23) / sqrt(2)`

and try the most obvious thing first: Alice measures her input qubit `1` alone, in the computational basis, and tells Bob the bit. To see what this does to Bob, I expand `|psi>_1 |Phi+>_23` and group by the value of qubit `1`. Writing the three qubits in order `1,2,3`,

`(alpha|0> + beta|1>)_1 (|00> + |11>)_23 / sqrt(2)`
`= |0>_1 (alpha|0> + ... )` — more carefully, the cross terms separate as
`= (1/sqrt(2)) [ |0>_1 (alpha|00> + beta? ) ]`.

I should just do it term by term. The product is

`(1/sqrt2)[ alpha|0>_1|00>_23 + alpha|0>_1|11>_23 + beta|1>_1|00>_23 + beta|1>_1|11>_23 ]`.

If Alice's qubit `1` reads `0`, the surviving terms are `alpha|0>_1|00>_23 + alpha|0>_1|11>_23`, i.e. Bob's qubit `3` is in `alpha(|0>+|1>)`-flavored... no — qubits `2,3` collapse to `alpha|00>_23 + alpha|11>_23`, which after tracing out qubit `2` leaves Bob in a mixture of `|0>` and `|1>`, not a coherent superposition carrying `beta`. The `beta` amplitude has simply vanished from the `1`-reads-`0` branch. That already tells me this cannot work: a single-qubit computational measurement on `1` throws away exactly the phase/amplitude relation I need to transport.

I make this quantitative to be sure I am not fooling myself. Taking a concrete `alpha = 0.5 + 0.3i`, `beta = sqrt(1 - |alpha|^2)`, I compute Bob's conditional reduced state for each outcome of a computational-basis measurement on qubit `1`. The outcome `0` occurs with probability `0.340` and outcome `1` with probability `0.660` — already a tell, because those probabilities depend on `|alpha|` and `|beta|`, whereas anything fundamental about Bob's qubit must not. In both branches Bob's reduced state comes out as

`[[0.5, 0], [0, 0.5]] = I/2`,

with eigenvalues `(0.5, 0.5)`: maximally mixed. So measuring qubit `1` by itself destroys the input and leaves Bob with literally no information about `alpha, beta`. The classical bit Alice sends is useless. This route is dead, and the reason it is dead is instructive: I measured one of Alice's two qubits as a standalone carrier, which broke the entanglement that was supposed to ferry the coefficients.

The fix the failure points to is to measure both of Alice's qubits *jointly*, in a basis whose vectors entangle systems `1` and `2` rather than reading either one alone. I want a basis on qubits `1,2` whose four outcomes are equally likely (so no readable description of `alpha, beta` leaks) and each of which leaves Bob's qubit `3` as some fixed, `alpha,beta`-independent unitary applied to `|psi>`. The Bell basis is the natural candidate, since its four vectors pair Alice's two systems maximally:

`|Phi+> = (|00> + |11>) / sqrt(2)`, `|Phi-> = (|00> - |11>) / sqrt(2)`, `|Psi+> = (|01> + |10>) / sqrt(2)`, `|Psi-> = (|01> - |10>) / sqrt(2)`.

I don't yet know it works; I have to expand `|psi>_1 |Phi+>_23` in this basis on `(1,2)` and read off what lands on qubit `3`. Start from the four product terms above and invert the Bell definitions, `|00> = (|Phi+>+|Phi->)/sqrt2`, `|11> = (|Phi+>-|Phi->)/sqrt2`, `|01> = (|Psi+>+|Psi->)/sqrt2`, `|10> = (|Psi+>-|Psi->)/sqrt2`, applied to qubits `1,2`. Grouping by Bell outcome, the coefficient of `|Phi+>_12` collects `alpha|0>_3` from the `|00>` term and `beta|1>_3` from the `|11>` term, giving `(alpha|0> + beta|1>)_3 = |psi>_3`. The coefficient of `|Phi->_12` collects `alpha|0>_3` minus `beta|1>_3`, i.e. `Z|psi>_3`. The `|Psi+>_12` coefficient swaps which of Bob's basis states each amplitude rides on, giving `(beta|0> + alpha|1>)_3 = X|psi>_3`, and `|Psi->_12` gives `(beta|0> - alpha|1>)_3`, which is `X` after a sign, i.e. `XZ|psi>_3`. Collecting the `1/2` factors,

`|psi>_1 |Phi+>_23 = 1/2[ |Phi+>_12 |psi>_3 + |Phi->_12 Z|psi>_3 + |Psi+>_12 X|psi>_3 + |Psi->_12 XZ|psi>_3 ]`.

I want to confirm this is an exact identity, not just a plausible regrouping, so I evaluate both sides numerically for the same complex `alpha = 0.5 + 0.3i`. Projecting the full three-qubit vector onto each Bell state on `(1,2)` returns the residual on qubit `3`, and I get, branch by branch: `Phi+` -> `(0.25+0.15i, 0.406)` which is exactly `(1/2)|psi>`; `Phi-` -> `(0.25+0.15i, -0.406)` which is `(1/2)Z|psi>`; `Psi+` -> `(0.406, 0.25+0.15i)` which is `(1/2)X|psi>`; `Psi-` -> `(-0.406, 0.25+0.15i)` which is `(1/2)XZ|psi>`. All four match to machine precision, so the identity holds for a generic complex amplitude, not only for a real or cherry-picked one.

This expansion is the whole mechanism. Alice measures qubits `1,2` in the Bell basis. The measurement projects `1,2` into one Bell state, so the original free input is absorbed and no copy of `|psi>` is left at Alice — the no-cloning constraint is respected automatically rather than dodged. Simultaneously Bob's qubit becomes one of `|psi>`, `Z|psi>`, `X|psi>`, `XZ|psi>`. Because every term carries the same `1/2` amplitude and each conditional Bob state is normalized, each outcome occurs with probability `(1/2)^2 = 1/4`, independent of `alpha, beta`. So unlike the failed single-qubit route, the outcome statistics here leak nothing about the amplitudes; the two-bit result is purely a label naming which Pauli Bob must undo.

I now check the no-signaling property explicitly, since I do not want to merely assert it. Before Alice's message arrives, Bob's state is the equal mixture over the four conditional states. With `rho = |psi><psi|`,

`(rho + Z rho Z + X rho X + XZ rho ZX) / 4`.

Computing this for the same `rho` gives `[[0.5,0],[0,0.5]] = I/2`. To see it is not an accident of this particular `|psi>`, note the four operators `{I, Z, X, XZ}` are, up to phase, the full Pauli group on one qubit, and averaging any single-qubit `rho` over the Pauli twirl `(rho + X rho X + Y rho Y + Z rho Z)/4` always gives `I/2` for any input. So the result is `alpha,beta`-independent for every state, confirming Bob can extract nothing — not the input, not Alice's outcome — until the two classical bits arrive. The classical channel is therefore not decorative; it is the causal key that converts a locally useless conditional state into `|psi>`, and it is precisely what stops the pre-shared entanglement from carrying information faster than light.

Two small consistency points before I am satisfied. First, Bob's correction: for outcome `Psi-` his state is `XZ|psi>`, and applying `ZX` returns `ZX·XZ|psi>`. Since `ZX·XZ = Z·X·X·Z = Z·Z = I` up to the sign from reordering Paulis, this recovers `|psi>` up to a global phase, which is physically irrelevant. I verified numerically that applying `I, Z, X, ZX` to the four respective conditional states each returns `|psi>` up to a unit-modulus phase, so the correction table closes. Second, the single-qubit failure and the Bell success differ in exactly the way the algebra predicted: the bad route's outcome probabilities depended on `|alpha|^2` (`0.34/0.66`) while the good route's are flat at `1/4`, which is the signature that the joint Bell measurement, and only it, decouples the statistics from the unknown coefficients.

The tension is thus resolved in the right direction. The unknown state is never copied, since Alice's input is consumed by the Bell-basis measurement. The EPR pair is spent to make Bob's qubit the conditional carrier of the coefficients. The two classical bits are ordinary copyable information carrying only the correction label. Once Bob receives them and applies the matching Pauli inverse, he holds `|psi>` exactly.
