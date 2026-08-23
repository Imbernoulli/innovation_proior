Compute is the silent constraint of structure learning at scale. The
per-network evaluation window is under an hour, and everything — loading,
testing, search, serialization — lives inside it; a search that is
comfortable on Cancer's five variables can spiral on Win95pts, where
conditioning-set enumeration and permutation moves explode
combinatorially. Timing out or thrashing yields the worst possible
outcome: no usable graph at all, on the very networks that dominate the
suite.

The discipline demanded here is anytime discovery under a self-imposed
compute envelope. Architect the procedure so that (a) a valid CPDAG
exists from very early in the run and is only ever refined, never
invalidated; (b) heavy computation is organized as cheap global triage
followed by budgeted refinement, spending marginal effort where the
expected metric gain is largest — ambiguous pairs, high-degree hubs,
orientation decisions with propagation leverage; and (c) the envelope is
enforced by the algorithm itself (wall-clock or operation-count
accounting), not by hoping the harness is patient. Adaptation to network
size must emerge from the budget arithmetic — more variables means
shallower per-pair effort — never from recognizing a specific dataset.

The evidence for success stays entirely inside the reported metrics:
SHD, adjacency precision and recall, and arrow precision and recall on
each of the five networks, achieved while the procedure demonstrably
respects its envelope (log the accounting). A frugal method that
matches a profligate one on these numbers is the thesis; quantify what
the last increments of budget bought, and show the large networks —
where the envelope binds hardest — did not collapse relative to the
small ones.
