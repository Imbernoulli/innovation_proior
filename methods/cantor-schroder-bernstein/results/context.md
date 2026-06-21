# Context: comparing the size of two sets by injections

## Research question

When may we say that two sets, possibly infinite, have *the same size*? The working definition in the new
theory of the transfinite is bijection: $A$ and $B$ have the same cardinality, written $A \approx B$, exactly
when there is a one-to-one correspondence (a bijection) between them. To compare sizes we use one-sided maps:
write $A \preceq B$ ("$A$ is at most the size of $B$") when there is an **injection** $f : A \to B$, an embedding
of $A$ into $B$.

We want to know whether $\preceq$ behaves like an order on sizes. Reflexivity ($A \preceq A$ via the
identity) and transitivity ($A \preceq B \preceq C$ via composition of injections) are immediate. The remaining
property is **antisymmetry**:

> If $A \preceq B$ and $B \preceq A$, is $A \approx B$?

Concretely: given two injections $f : A \to B$ and $g : B \to A$, must there exist a bijection $h : A \to B$?
For finite sets this is the pigeonhole fact that $a \le b$ and $b \le a$ imply $a = b$. For infinite sets the
question is open: the task is to start from the two injections $f, g$ and produce a bijection $h : A \to B$.

## Background

The field at this moment is the freshly built theory of infinite sets and their cardinalities. The load-bearing
concepts:

- **Cardinality by bijection; comparison by injection.** Two sets have the same power iff there is a bijection
  between them. The relation $A \preceq B := (\exists\,\text{injection } A \to B)$ is the available notion of
  "at most as large." Antisymmetry of $\preceq$ is the structural question above.

- **Infinite sets behave unlike finite ones.** A proper subset of an infinite set can have the same cardinality
  as the whole — $\mathbb N \approx 2\mathbb N$ via $n \mapsto 2n$; $\mathbb N \approx \mathbb Z$; the integer
  points of the plane are countable. An injection of a set into itself can fail to be surjective and yet
  still be a "same-size" map onto its image. An injection $A \to B$ carries $A$ onto only *part* of $B$.

- **Dedekind's theory of chains** (Dedekind, *Was sind und was sollen die Zahlen?*, 1888). For a map
  $\varphi : S \to S$, a subset $K \subseteq S$ is a **chain** (*Kette*) if it is closed under $\varphi$, i.e.
  $\varphi[K] \subseteq K$. For any seed $A \subseteq S$, the **chain of $A$** is the intersection of all chains
  containing $A$ — the *smallest* $\varphi$-closed set $\supseteq A$. Dedekind proves the union and intersection of
  chains are chains, so this intersection is itself a chain and is well defined. This machinery makes precise the
  idea "start from a seed and iterate $\varphi$ as many finite times as needed," and it does so
  *without presupposing the natural numbers* (Dedekind's aim was to *found* $\mathbb N$ on chains, so he
  could not assume $\mathbb N$ in building them). It is the rigorous form of "$A \cup \varphi[A] \cup
  \varphi^2[A] \cup \cdots$."

- **Monotone operators on the powerset.** $\wp(A)$ ordered by inclusion is a complete lattice: every family of
  subsets has a union (its join) and an intersection (its meet). A map $m : \wp(A) \to \wp(A)$ is *monotone* if
  $X \subseteq Y \Rightarrow m(X) \subseteq m(Y)$. Operators of the form $X \mapsto (\text{fixed set}) \cup
  (\text{image of } X \text{ under some map})$ are monotone, because both union-with-a-constant and direct image
  preserve inclusions. (The general fixed-point theory for monotone operators on a complete lattice — least and
  greatest fixed points obtained as $\bigcap\{X : m(X)\subseteq X\}$ and $\bigcup\{X : X \subseteq m(X)\}$ — sits
  one step beyond.)

- **Direct and inverse images, and what injectivity buys.** For $f : A \to B$ and $S \subseteq A$, the image
  $f[S] = \{f(x) : x \in S\}$. The inverse $f^{-1}$ is a genuine function *only on the image* $f[A]$, and there it
  is well defined **because** $f$ is injective — each $b \in f[A]$ has exactly one preimage. Injectivity also gives
  $f[S \cap T] = f[S] \cap f[T]$ and $f[A \setminus S] = f[A] \setminus f[S]$ (complements within the image), which
  fail for non-injective maps. These set-algebra facts are the special properties of $f, g$ available.

- **Choice principles.** The Axiom of Choice (equivalently, the Well-Ordering Theorem: every set
  can be well-ordered) lets one compare any two cardinals by well-ordering both and matching initial segments.
  A construction that calls Choice produces a bijection depending on an arbitrary choice function. A construction
  is **uniquely determined** by its inputs when, wherever a selection seems needed, the description pins down one
  and only one way to make it.

## Baselines

The approaches on the table for turning $A \preceq B \preceq A$ into $A \approx B$:

- **Use one injection alone.** $f : A \to B$ is a bijection onto its image $f[A]$, and $f[A]$ can be a proper
  subset of $B$ (take $A = B = \mathbb N$, $f(n) = n+1$: misses $0$). Symmetrically $g$ carries $B$ onto $g[B]
  \subseteq A$.

- **Use an inverse.** $g^{-1}$ is a bijection from $g[B]$ back to $B$, defined on $g[B] \subseteq A$.

- **Splice $f$ and $g^{-1}$.** Since $f$ is defined on all of $A$ and $g^{-1}$ is defined on $g[B]$, one may
  consider a map that is $f$ on some part of $A$ and $g^{-1}$ on the rest.

- **Comparability via well-ordering (Cantor's route, 1895).** Well-order $A$ and $B$ and compare order types;
  cardinal comparability then yields equality. This invokes the Well-Ordering Theorem, i.e. the Axiom of Choice,
  and the bijection it produces depends on the chosen well-orderings.

- **Add two correspondences.** If $A_1 \approx B_1$ and $A_2 \approx B_2$, the union of the two bijections is a
  bijection $A_1 \cup A_2 \to B_1 \cup B_2$ when the $A_i$ are pairwise disjoint *and* the $B_i$ are pairwise
  disjoint.

## Evaluation settings

The yardstick is purely logical, not numerical. A proposed construction is judged by:

- **Correctness of the produced map.** Is the candidate $h : A \to B$ *total* (defined on all of $A$),
  *well defined* (one output per input), *injective*, and *surjective onto $B$*? All four must be proved, for
  arbitrary sets, finite or infinite.
- **Generality.** It must hold for *all* sets $A, B$ and *all* injections $f, g$ — no countability or
  cardinality assumption, no structure on $A, B$ beyond being sets.
- **Logical economy / which axioms are used.** A construction is examined for which axioms it invokes and whether
  it is determined by $f$ and $g$. Canonical sanity checks: the finite case must reduce to ordinary counting; the
  motivating witnesses must come out right — $\mathbb N$ with $f(n)=n+1, g=\mathrm{id}$; $\mathbb N$ with
  $f(n)=2n$; $[0,1]$ vs $[0,1)$; pairs like $(0,1)$ vs $[0,1]$.

## Code framework

This is a pure existence theorem about sets; the natural "implementation" is the **proof itself**, and the
appropriate scaffold is a proof skeleton stated in the pre-method vocabulary above. The slots below are exactly
what the construction will fill in. (Pseudo-notation; $A, B$ arbitrary sets, $f, g$ the given injections.)

```text
Goal:  given injections f : A → B and g : B → A,  exhibit a bijection h : A → B.

Available primitives (all pre-method, already justified in Background):
  • images:        f[S] for S ⊆ A,   g[T] for T ⊆ B
  • partial inverse: g⁻¹ : g[B] → B   (total & well-defined on g[B] by injectivity of g)
  • inclusion lattice ℘(A): arbitrary ⋃, ⋂; complement A∖S
  • chain-of-a-seed (Dedekind): for ψ : A → A and seed S₀ ⊆ A,
        chain(S₀) := ⋂ { X ⊆ A : S₀ ⊆ X and ψ[X] ⊆ X }      # smallest ψ-closed set ⊇ S₀
        (well defined; intersection of chains is a chain)
  • gluing lemma:   if {Xᵢ} pairwise disjoint, {Yᵢ} pairwise disjoint, and each Xᵢ ≈ Yᵢ,
                    then ⋃Xᵢ ≈ ⋃Yᵢ.

# ----- the slot the method must fill -----
def partition_of_A(A, B, f, g):
    # TODO: determine, from f and g, the subset of A on which to use f
    #       versus the rest.
    pass

def h(x):
    # TODO: a single map A → B assembled from f and g⁻¹ using the partition above.
    pass

# ----- obligations the construction must then discharge -----
def prove_total_and_well_defined(h):   # h defined on all of A, one value each
    pass
def prove_injective(h):                # no two x collide
    pass
def prove_surjective(h):               # every b ∈ B is hit
    pass
def sanity_finite_and_witnesses(h):    # reduces to counting on finite sets; ℕ→ℕ examples come out right
    pass
```

The whole content of the method is what goes in `partition_of_A` and `h`, and the four proofs that the
resulting $h$ is a bijection — built only from $f$, $g$, images, the partial inverse, the chain-of-a-seed
device, and the gluing lemma.
