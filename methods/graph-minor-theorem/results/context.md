## Problem

Several graph classes are characterized by special forbidden-pattern theorems. Forests forbid cycles.
Planar graphs forbid K_5 and K_{3,3} as minors. Outerplanar graphs have their own small excluded-minor
list. A graph H is a minor of G when H can be obtained from G by deleting vertices, deleting edges, and
contracting edges. These examples raise a general question: if a graph property is preserved under all
three minor operations, is there always a finite list of minimal forbidden minors that describes it?

For a minor-closed class, the candidate obstructions are the graphs outside the class that are minimal
under taking minors. The obstruction set is finite exactly when these minor-minimal bad graphs cannot
form an infinite antichain, meaning an infinite collection in which no graph is a minor of another.

## Earlier View

Classification has so far been per-family. One chooses a graph class, finds its special geometric or
combinatorial description, and then proves the exact forbidden patterns for that class. This proceeds
cleanly for planarity, where topology supplies a visible obstruction theory. Each obstruction list is
treated as a separate target: the question asked is which individual graphs must be excluded for a
given class.

## Known Tools

Some machinery for analyzing graphs by their structure is available. Tree decompositions describe a
graph as pieces arranged along a tree, with a treewidth parameter measuring how tree-like the graph
is. Graphs of bounded treewidth admit a tree-like encoding. Kruskal's tree theorem gives a
well-quasi-ordering result for finite trees under a topological-embedding order: in any infinite
sequence of trees, one embeds into a later one. Clique-sums glue graphs together along shared cliques
and are used to build larger graphs from smaller controlled pieces.

## Research Question

The setting is the family of all minor-closed properties of finite graphs at once, rather than any
single class. The question is whether the minor relation on finite graphs has enough global regularity
to determine, for every such property, the structure of its forbidden patterns, and how the
order-theoretic behavior of minors relates to the structural decomposition of the graphs involved.
