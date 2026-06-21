# Grothendieck Schemes

Grothendieck's scheme concept rebuilds algebraic geometry around rings of functions rather than visible solution sets. For a commutative ring `A`, the affine geometric object is not the set of classical zeros of equations, but `Spec A`: the set of all prime ideals of `A`, equipped with the Zariski topology and the structure sheaf of local rings. A general scheme is a locally ringed space obtained by gluing such affine spectra.

The unique insight is that the right "points" of algebraic geometry are prime ideals, and the right object is not a point set but a point set plus local functions. Maximal ideals recover ordinary closed points in familiar algebraically closed settings, but prime ideals add generic points for irreducible subvarieties and encode specialization. The structure sheaf adds the information that the prime set alone loses, especially nilpotents and infinitesimal thickening.

This is why schemes make nilpotents, arithmetic fibers, and families part of one geometry:

- Nilpotents: `Spec k[epsilon]/(epsilon^2)` has the same underlying point as `Spec k`, but its local ring contains the nilpotent `epsilon`; the scheme remembers infinitesimal structure.
- Arithmetic fibers: `Spec Z` has a generic point `(0)` and closed points `(p)`. A scheme over `Spec Z` has one generic characteristic-zero fiber and closed characteristic-`p` fibers, all as fibers of a single morphism.
- Families: a morphism `X -> S` is a family over the base scheme `S`; fibers, generic members, special members, degeneration, and base change are handled by the same formal operation.

The conceptual shift is therefore radical. Classical geometry asks for the solution set of equations and then studies functions on that set. Scheme geometry starts with the function ring, studies all of its prime ideals as geometric probes, attaches the local rings of functions, and glues these affine models. The result is a category where geometry is recovered from algebra without discarding nilpotents, without separating number theory from geometry, and without treating families as an auxiliary construction.
