The key problem is not whether particular graph classes have forbidden-minor descriptions. Many do.
The deeper question is whether minor-closed graph properties always have finite forbidden-minor
descriptions, even when the property has no obvious geometric definition.

A minor-closed family F has a natural obstruction set: the graphs not in F whose proper minors all lie
in F. These are the minimal counterexamples to membership in F. If this obstruction set is finite,
membership in F is equivalent to avoiding those finitely many minors. If it is infinite, then the
class cannot be summarized by a finite forbidden list.

Minimal obstructions are automatically pairwise incomparable under the minor relation. If one minimal
obstruction contained another as a proper minor, the larger one would not be minimal. Therefore an
infinite obstruction set would be an infinite antichain.

Robertson and Seymour's theorem rules out exactly that possibility. It says finite graphs are
well-quasi-ordered by the minor relation. In practical terms, every infinite list of graphs contains
two graphs G_i and G_j, with i < j, such that G_i is a minor of G_j. That statement forbids infinite
antichains. Once this global order theorem is known, finite obstruction sets for every minor-closed
class follow immediately.

The theorem is powerful because the proof does not try to enumerate obstruction sets. Enumeration
would be hopeless in general. Some obstruction sets are enormous, and for natural classes they may
remain unknown. The theorem instead proves a compactness-like structural fact: however complicated a
minor-closed property looks, its minimal failures cannot keep escaping into new incomparable shapes
forever.

Tree decompositions are the bridge between local graph shape and global order. They represent a graph
as bags of vertices arranged along a tree, so that interactions between distant parts are mediated by
bounded overlaps. In bounded-treewidth settings this turns graphs into tree-like objects, and
tree-like objects are precisely where well-quasi-ordering methods have strong leverage.

The full theorem has to handle graphs of unbounded treewidth. The structure theory developed in the
Graph Minors series says that if a fixed graph H is excluded as a minor, then the remaining graphs
have a controlled form. They can be decomposed along clique-sums into pieces that are almost embedded
in surfaces where H cannot be embedded, plus bounded exceptional features such as apices and
vortices. This is not a simple treewidth bound; it is a decomposition that isolates the reasons a
large graph can avoid H.

This is why the theorem is more than a forbidden-list result. Its real method is to replace
case-by-case classification with a structural grammar for graph families. Once graphs are expressed
through controlled decompositions, the well-quasi-ordering argument can propagate through the
decomposition instead of fighting arbitrary graphs directly.

The distinctive conceptual compression is therefore: infinite families are controlled by finite
minimal obstructions because the minor order is well-quasi-ordered, and the minor order is tractable
because graphs excluding a fixed minor have decomposable structure. Tree decompositions organize the
pieces, excluded-minor structure controls the pieces, and well-quasi-ordering turns that control into
finiteness.

This changes what it means to understand a graph class. The old question asks for the forbidden
graphs of one class. The Robertson-Seymour viewpoint asks whether the whole universe of finite graphs
has an ordering and decomposition theory strong enough to make every minor-closed class finitely
describable. The answer is yes, and that is the method's lasting insight.
