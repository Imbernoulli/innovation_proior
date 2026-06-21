# Cohen Forcing

Start with a countable transitive model `M` of ZF, usually chosen with `V=L` for the continuum application. Let `P` be a partial order of finite conditions. For adding many reals, a condition is a finite partial function

```text
p: aleph_tau^M x omega -> 2
```

ordered by extension. Equivalently, `p` makes finitely many decisions about membership in future subsets `a_delta subset omega`.

Build a filter `G subset P` meeting every dense requirement from `M`. The union of `G` gives total binary sequences `a_delta`; dense requirements make these sequences distinct and make every relevant statement decided. Define `M[G]` by interpreting `P`-names from `M` with `G`.

Define forcing in the ground model:

```text
p ||- phi
```

means that the finite condition `p` guarantees `phi` in every generic extension containing `p`. The forcing theorem is the control principle:

```text
M[G] |= phi  iff  some p in G satisfies p ||- phi.
```

The proof then verifies the ZF axioms in `M[G]`, not by assumption but through names and forcing. Replacement follows from boundedness of witnesses for definable functions. Power Set follows from the analysis of incompatible conditions and closure of index sets. Choice follows from a definable well-order of the generated objects. The relevant old cardinal comparisons are preserved.

For the continuum problem, choose `tau` so the added family has size at least `aleph_2^M`. In the resulting extension,

```text
aleph_tau <= 2^aleph_0 <= aleph_{tau+1}
```

and in the standard `tau = 2` case this yields a model of ZFC with

```text
2^aleph_0 = aleph_2
```

so CH fails. Goedel's constructible model gives the complementary model of ZFC plus GCH. Together the two constructions show that CH is independent of ZFC, assuming the relevant consistency hypothesis.
