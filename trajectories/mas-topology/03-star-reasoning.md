The layered numbers came back and they answered the question I closed step 2 on — is the win coming
from diversity or from synthesis? — in a way that points hard at the next move. On HumanEval the layered
shape did what I predicted: 0.697 on both backbones (23/33 each), a real lift over the chain's 0.6667 /
0.6061, and the qwen backbone — the chain's weakest, hostage to one source's approach — climbed the
most, exactly the bad-seed recovery I expected from a second independent draft plus a downstream
aggregation. So adding diversity and synthesis genuinely helped the closed-domain function task. But on
SRDD the result is the diagnostic one, and it is *not* the clean win I expected. `srdd_exec_rate` stayed
flat at 0.05 — still one runnable project out of twenty — while `mean_loc` collapsed from the chain's
150.7 all the way to 12.5. That collapse is the tell. The chain produced 150 lines and one of them ran;
the layered shape produced *twelve* lines and one of them ran. The synthesis did not fail by breaking
runnable projects — it failed by aggregating two parallel drafts into something hollow. When the
runtime forces a downstream node to fuse two divergent project structures, the path of least resistance
for an LLM aggregator is to keep only what the two drafts *agree* on — a skeletal intersection — and a
skeleton of twelve lines does not implement a working software project, so it does not run. The layered
shape over-invested in forced convergence, and on the open-ended task the merge step ate the very
diversity it was meant to synthesize, leaving a tiny, safe, non-functional stub.

So now I can read the two failures side by side and they triangulate the answer. The chain failed on
SRDD by *too much undiversified depth* — one long thread, 150 lines, rolled into something broken. The
layered shape failed on SRDD by *too much forced synthesis* — two drafts merged into a 12-line
intersection that runs but does nothing. Both have a single source of diversity at most (the layered
shape's source layer is two nodes, but every downstream node is immediately forced to merge), and both
spend agents on operations — deep refinement, mandatory aggregation — that on an open-ended task destroy
more than they build. The HumanEval lift tells me diversity *is* the useful dial; the SRDD collapse
tells me the *synthesis* dial, pushed hard, is the one that hurts on breadth-hungry tasks. So the move
is to push diversity to its maximum and push forced synthesis to its minimum: give as many agents as
possible the *same* seed to attack independently and in parallel, and make them peers — no agent forced
to refine another's draft, no interior node forced to aggregate — with synthesis happening exactly once,
at the very end, on the full spread rather than on a prematurely-merged intersection. That shape is the
star.

Let me derive the star from the runtime's three dials rather than just naming it as the opposite of what
failed. I want one node to be the seed and every other node to be an independent parallel refinement of
that seed. Concretely: node 0 produces the first artifact from the task, and then node 0 broadcasts to
every other node — edges `0→1, 0→2, …, 0→(node_num-1)`. Check it against the runtime. The input sentinel
feeds node 0 (the single source); node 0's actor produces a draft. Then nodes 1 through `node_num-1`
each receive node 0's *one* artifact on their single incoming edge, each runs its own edge critic and
actor to refine it, and — this is the crucial part — they do this *independently and in parallel*,
because they are all in the same Kahn wavefront and none of them feeds another. Every spoke is a
separate refinement of the same seed, taking it in its own direction. None of these nodes is convergent
(each has exactly one incoming edge, from node 0), so the runtime's aggregation branch does *not* fire
at the spokes — the very operation that hollowed out the layered SRDD result is switched off in the
interior. The only place artifacts come back together is at the output sentinel `-2`, which the runtime
attaches to every sink: all `node_num-1` spokes are sinks, so the runtime drains them all into one final
artifact with a single aggregation at the very end. So the star is `node_num-1` edges — the same edge
count as the chain — but arranged as pure fan-out: depth one (every spoke is one hop from the seed),
width `node_num-1` (the widest possible single-layer divergence), and exactly one shallow synthesis at
the sink. It is the divergent extreme of the DAG family.

Why should this beat the layered shape, derived from the runtime and tied to the measured failures?
Three reasons, each addressing a specific number I just saw. First, **maximum diversity**: the layered
shape's source layer was two parallel drafts; the star's spokes are `node_num-1` parallel refinements of
the seed, every one free to take the artifact in a different direction. On SRDD, where a project has
many independent aspects and the chain's single thread and the layered shape's premature merge both
strangled exploration, the star gives the most parallel takes on the requirement of any topology at this
agent count. Second, **no forced interior synthesis**: the layered shape's 12-line collapse came from
making every downstream node merge two drafts into an intersection. The star has *zero* convergent
interior nodes — no spoke ever aggregates — so the diversity is never compressed into a hollow skeleton
mid-flight; each spoke produces a full, committed refinement, and synthesis happens once at the end over
the *whole* spread rather than pairwise on the way down. Third, **shortest paths under memory control**:
the chain's SRDD rollback was the deep-path-forgetting failure — a late agent breaking a long artifact
it could no longer justify. The star's longest path is one hop. There is no distant upstream agent whose
intent gets pruned, because every spoke is directly adjacent to the seed. Under the runtime's
artifact-only memory, depth one is the safest possible setting: nothing to forget, nothing to roll back.
So the star simultaneously maximizes the dial that helped (diversity) and minimizes the two operations
that hurt (deep refinement and forced interior aggregation).

There is a directional asymmetry worth pulling out, because it is the structural reason the star should
win on an open-ended task and not merely a "less of the bad stuff" argument. The runtime makes two kinds
of nodes: divergent nodes, where one artifact branches outward to several successors, and convergent
nodes, where several artifacts must be merged into one. Divergence is *smooth* — an agent taking one
draft and refining it in its own direction is a well-posed, low-risk act, the same act that worked on
HumanEval. Convergence is *hard and lossy* — fusing several divergent artifacts into one coherent
improved artifact is the demanding operation, and I just watched it produce a 12-line intersection on
SRDD. So a shape that maximizes divergence and minimizes forced convergence should beat one that does
the reverse, and the star sits at the divergent extreme: pure fan-out, with the only convergence a
single shallow synthesis at the sink over solutions that are each already complete. The layered shape,
by contrast, forces convergence at every downstream node — it leans into the hard, lossy operation
precisely on the task where that operation hurts most. Reversing the star into its mirror — a shape
where many sources feed one hub — would turn the smooth fan-out into a forced many-to-one merge and
should *degrade* it, which is the same lesson the layered SRDD number taught from the other side. The
design rule that falls out: maximize divergence, minimize forced convergence — which is exactly the star.

I should be honest about what the star gives up, because the dial it sacrifices is the one the chain and
the layered shape spent: depth. Every spoke refines the seed exactly once and is never refined again.
There is no iterative deepening — no second pass to catch what the first refinement missed. On a task
where the artifact improves monotonically under repeated sequential polishing and where the failure mode
is a subtle bug that takes several passes to surface, the star's depth-one structure is a real
liability, and a deeper shape could beat it. So the star is not a universal winner; it is the right
setting of the dials for *breadth-hungry, open-ended* tasks — many parallel aspects beating one deep
thread — and the HumanEval result will tell me whether the closed-domain function task, which has a
crisp spec and rewards bug-fixing depth, pays a price for the lost depth or whether the breadth of
`node_num-1` independent attempts plus a final synthesis covers the same ground. My read is that even on
HumanEval the star should at least match the layered shape: a single spec'd function does not have much
breadth, but `node_num-1` independent attempts at it, synthesized once, is a strictly richer pool than
the layered shape's two-draft source, and the final aggregation over complete attempts is far less
lossy than the layered shape's mandatory pairwise merges — it is choosing among finished solutions, not
intersecting half-finished ones.

Let me walk the `node_num = 4` case to be sure the construction does the right thing at the agent count
I will be evaluated at. The star generator emits `(0, i)` for `i` in `1, 2, 3` — three edges: `0→1`,
`0→2`, `0→3`. Node 0 is the single source, fed the task by the input sentinel; it produces the seed
draft. Nodes 1, 2, 3 each receive node 0's seed on their single incoming edge, and because none of them
feeds another, they sit in one Kahn wavefront and refine the seed in parallel — three independent takes.
None is convergent, so no spoke aggregates. All three are sinks, so the output sentinel drains all three
into one final artifact with a single synthesis. Depth one, width three, one shallow end-merge. That is
the divergent extreme at four nodes: where the chain refined one draft three times down a thread and the
layered shape merged two drafts at every step into a stub, the star produces three full independent
refinements of one seed and synthesizes them once. Every edge runs low-index to high-index (`0 < i`), so
the index order is a valid topological order and the DAG property is free; the graph is a guaranteed DAG
with one source and three sinks.

So the delta from step 2 is precise and it is a one-line change to the generator: replace the layered
shape's layer-partition-and-full-bipartite-wiring with the fan-out — emit `(0, i)` for `i` in
`1 .. node_num-1` (the full scaffold module is in the answer). Nothing else changes — same runtime, same
prompts, same evaluators, same `node_num`. The edit pushes the diversity dial to its maximum and the
forced-synthesis dial to its minimum, which is exactly what the two prior failures said to do.

Now the falsifiable expectations against the measured numbers. The star maximizes diversity and removes
forced interior synthesis, so it should beat the layered shape where the layered shape's mandatory merge
hurt most — SRDD — and at least match it where diversity already helped — HumanEval. On SRDD I expect
the clearest win of the whole ladder: the chain's 0.05 (150 lines, broken by rollback) and the layered
shape's 0.05 (12 lines, hollowed by premature merge) both come from strangling exploration, and the
star's `node_num-1` independent full project drafts, synthesized once at the end, is the direct antidote
— I expect `srdd_exec_rate` to rise meaningfully above 0.05 for the first time, with a `mean_loc`
between the chain's bloat and the layered shape's stub (real projects, neither 150-line sprawl nor
12-line skeleton), because the final synthesis is choosing among complete attempts rather than
intersecting half-finished ones. On HumanEval I expect the star to match or modestly exceed the layered
shape's 0.697 / 0.697: the breadth of independent attempts plus a less-lossy final merge should cover at
least as much ground as two merged drafts, and if the closed-domain task pays a price for the star's
lost depth it should be small at four nodes. The one way I could be wrong, and the thing a finer study
would chase, is if the final single synthesis at the sink is itself the lossy operation in disguise —
if merging `node_num-1` divergent project drafts at the end is as hard as merging two mid-flight, the
star's SRDD win could be capped. But the structural argument says one end-merge over complete solutions
is strictly easier than repeated mid-flight merges over partial ones, so I expect the star to be the
strongest topology on this task: the divergent extreme is the right setting of the dials for these
breadth-hungry code-generation benchmarks, and the chain's and layered shape's numbers are exactly what
it has to beat.
