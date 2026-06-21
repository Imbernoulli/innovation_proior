# Borsuk-Ulam theorem

## Statement

For every continuous map `g : S^n -> R^n`, there exists a point `x in S^n` such that

`g(x) = g(-x)`.

Equivalently, there is no continuous odd map `S^n -> S^{n-1}`, where odd means `F(-x)=-F(x)`.

## Lemma: odd self-maps have odd degree

Let `m >= 1`. If `u : S^m -> S^m` is continuous and odd, then `deg(u)` is odd.

Proof. Work with `Z/2` coefficients and the double covering `p : S^m -> RP^m`. Since `u(-x)=-u(x)`, the map descends to a quotient map `bar u : RP^m -> RP^m`, and `u` and `bar u` form a map of the covering to itself.

For a two-sheeted covering there is a transfer exact sequence

`... -> H_i(RP^m; Z/2) -> H_i(S^m; Z/2) -> H_i(RP^m; Z/2) -> H_{i-1}(RP^m; Z/2) -> ...`.

Naturality gives a commutative map from this exact sequence to itself induced by `u` and `bar u`. The map in dimension `0` is an isomorphism. Since `H_i(S^m; Z/2)=0` for `0<i<m` and the nonzero groups in the transfer sequence are copies of `Z/2`, exactness carries this isomorphism inductively through the sequence and forces

`u_* : H_m(S^m; Z/2) -> H_m(S^m; Z/2)`

to be an isomorphism. But on top homology with `Z/2` coefficients, `u_*` is multiplication by `deg(u)` modulo `2`. Therefore `deg(u) = 1 mod 2`, so `deg(u)` is odd. QED

## Proof of the theorem

The case `n=0` is immediate, since `R^0` has one point. The case `n=1` is also elementary: if `g : S^1 -> R` and `h(x)=g(x)-g(-x)` never vanished, then `sign(h)` would be a continuous map from the connected circle to `{+1,-1}`. It would be constant, contradicting `h(-x)=-h(x)`.

Assume `n >= 2` and suppose, for contradiction, that `g(x) != g(-x)` for every `x in S^n`. Define

`h(x)=g(x)-g(-x)`.

Then `h` is continuous, nonzero, and odd. Hence

`F(x)=h(x)/|h(x)|`

is a continuous odd map `F : S^n -> S^{n-1}`.

Let `E ~= S^{n-1}` be an equator in `S^n`, and let `D^n` be one closed hemisphere with boundary `E`. The restriction

`u = F|_E : E -> S^{n-1}`

is an odd self-map of `S^{n-1}`, after identifying `E` with `S^{n-1}`. By the lemma, `deg(u)` is odd.

But `u` extends over the disk `D^n`, namely by `F|_{D^n}`. Therefore the induced map on top homology factors as

`H_{n-1}(S^{n-1}) -> H_{n-1}(D^n) -> H_{n-1}(S^{n-1})`.

Since `H_{n-1}(D^n)=0`, this induced map is zero, so `deg(u)=0`. This contradicts the oddness of `deg(u)`.

Thus the assumption was false. There exists `x in S^n` with `g(x)=g(-x)`. QED

## Core obstruction

If antipodal values were always distinct, the normalized difference would create an odd map `S^n -> S^{n-1}`. Its equatorial restriction would have odd degree because antipodal symmetry survives on the quotient by antipodes, but degree zero because it bounds across a hemisphere. The theorem is the impossibility of that symmetry-respecting collapse.
