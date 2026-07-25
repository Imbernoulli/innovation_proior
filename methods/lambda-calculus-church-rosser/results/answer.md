# Church-Rosser theorem, distilled

The core conclusion of the Church-Rosser theorem is that beta-reduction in lambda calculus has confluence: if the same term `M` can be reduced separately to `N1` and `N2`, then there exists some `P` such that both `N1` and `N2` can further reduce to `P`.

This is exactly the idea of the diamond property: reduction paths can fork, but after forking they can still converge.

The most important consequence is that normal forms are independent of strategy. If a lambda term has a normal form, then it has at most one normal form. So different reduction orders may pass through different intermediate terms, take different numbers of steps, and some strategies may even fail to terminate; but as long as one succeeds in reaching a normal form, it will never arrive at an incompatible answer.

The distinctive insight of this theorem is not merely that "it doesn't much matter what you compute first," but that it lifts computation from a specific execution order to the global geometry of the rewriting system:

```text
term        = a node in the reduction graph
reduction   = a directed edge between nodes
strategy    = choosing a path through the graph
confluence  = forked paths can still converge downstream
normal form = if it exists, the unique terminal point that cannot be reduced further
```

So the Church-Rosser theorem separates the indeterminacy of local execution from the determinacy of global semantics. An interpreter can choose different redexes, a proof can adopt different reduction sequences, but the equational meaning of lambda calculus is not torn apart by these choices. It shows that computation does not have just one timeline, but is a reduction space constrained by confluence.
