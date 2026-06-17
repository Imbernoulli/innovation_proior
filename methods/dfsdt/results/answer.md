# DFSDT, distilled

DFSDT (Depth-First Search-based Decision Tree) is a search/reasoning policy for an LLM
tool-use agent. It turns a forward-only, grounded acting agent (interleaved thoughts and real
API calls) into a depth-first search over a tree of partial action sequences, so the agent can
*retract* a bad decision and branch in a new direction — while spending only one LLM call per
node and never paying for child ranking or value scoring.

## Problem it solves

An LLM plans a multi-step tool-use task: each round it emits a thought, picks an API from a
large pool, fills its parameters; the environment executes the call and returns a real
response; this repeats until "Finish with Final Answer" or "Finish by Giving Up", under a
query budget. The decision space (free-form thought × API choice × parameters) is effectively
infinite, calls can fail, and a single linear trajectory (a) propagates an early wrong action
into a faulty loop and (b) under-explores. The goal is to find *one* valid path on hard
instructions, tolerate wrong actions, and stay within a bounded number of LLM calls.

## Key idea

Search a tree of partial trajectories depth-first. A node is a state reached after some
thoughts/calls; the root is the task; a terminal node is a final answer; a node is pruned on
an API error or a give-up. On a dead end, retract to a nearby ancestor and expand a
*different* child — keeping the good prefix and revising only the bad suffix. Three design
choices make this affordable and effective in an infinite space with no value oracle:

- **Depth-first, not breadth-first.** Only one valid path is needed; DFS drives straight to a
  terminal and only branches sideways on failure, whereas BFS would expand whole frontiers and
  burn excessive calls before reaching any terminal.
- **Pre-order traversal — skip child ranking.** The textbook tree search ranks the `k`
  children at each node (≈ `O(n log n)` LLM-comparison calls, the dominant cost) and expands
  the best first. DFSDT generates *one* child and recurses immediately, expanding a second
  child only if the first subtree dead-ends. The model's first-generated child is usually the
  one ranking would pick anyway, so the ranking is dropped at no real loss. This also degrades
  gracefully: on a simple instruction where the model never retracts, the tree is a single
  chain and the procedure is exactly the flat acting agent at the same cost; after the search
  finishes, the explored node set is almost the same as full ranked DFS, so it still solves the
  hard instructions DFS solves.
- **Diversity prompt on re-expansion.** When a node already has children (we backed up to it),
  prepend a message showing the previous candidate actions and instruct the model to produce a
  *different* one — otherwise the same context regenerates the same failed action. The
  diversity message is marked invalid after use so it does not pollute the child's context or
  the recorded trajectory.

## The backtrack dial

The recursion returns an integer: *how many more levels to keep backing up*. This is the one
real knob, a dial between local retry and global restart:

- ordinary failed API call → return **1** (the caller consumes it by trying the next sibling
  at that same parent);
- "Finish by Giving Up" (status code 4) → return **`prune_back_length = 2`** (escape the local
  dead subtree, branch two levels up);
- a found final answer → return **`final_answer_back_length = 2`** (record it, keep searching
  elsewhere if more answers are requested; otherwise the caller stops the whole search);
- enough valid paths found (`≥ answer_count`) → return a large value to unwind and stop;
- budget hit (`query_count ≥ max_query_count`) → return a large value to abort everything.

A leaf that returns `2` asks its parent to keep backing up: the parent returns `2 - 1 = 1`.
The first ancestor that receives `1` consumes it by staying in its own loop and generating the
next sibling there. That is the exact `result - 1` propagation: values above `1` keep moving
up; value `1` means "branch at this frame." Pushed toward infinity the dial becomes a full
restart (greedy resample / between-episode retry); `back_length = 1` is a pure local retry;
`2` escapes the immediate bad neighborhood while preserving the prefix above it.

## Relation to prior methods

- **ReAct / single chain** = DFSDT when the model never retracts: the tree is one chain, one
  child per node — identical procedure and cost.
- **Reflexion** backs up at the wrong granularity: it restarts the *whole episode* from the
  root on failure. DFSDT backs up *locally* to a nearby ancestor, keeping the good prefix.
- **Tree-of-Thoughts / value-guided DFS** ranks/scores nodes with a state evaluator (extra
  LLM calls) and prunes by value — affordable on small brute-forceable spaces (Game of 24,
  Crosswords). DFSDT drops the evaluator entirely because the space is infinite, there is no
  cheap value oracle, and only one valid path is needed; it spends calls on forward generation
  and diversification, not on evaluation.
- **CoT** (ungrounded chain of thoughts) is excluded as a tool-use backbone because it cannot
  react to real API responses and hallucinates.

## Final algorithm

```
search(root):
    DFS(root)

DFS(node) -> levels_to_back_up:
    final_answer_back_length = 2
    prune_back_length        = 2
    node.expand_num = now_expand_num; now_expand_num += 1

    # depth is raw tree-node depth; one logical round is Thought -> Action -> Action Input
    if depth(node) >= max_step or node.pruned or node.is_terminal:
        if node.is_terminal:                      # final answer
            record terminal node; return 2
        node.pruned = True
        if node.observation_code == 4:            # gave up
            record give-up node; return 2
        return 1                                  # ordinary failed call -> local retry

    # expand depth-first, one child at a time, no ranking
    for _ in range(tree_beam_size):
        added = append_temporary_diversity_message(node)  # only when node already has children
        leaf  = step(node)                        # one LLM parse + optional tool execution
        if query_count >= max_query_count: return 100000        # budget abort after the call
        if added: mark diversity message invalid
        # step handles status 4 -> pruned give-up, status 1 -> hallucinated-API sentinel,
        # status 3 -> terminal final answer + make_finish(2)
        result = DFS(leaf)                        # recurse immediately (pre-order)
        if len(terminal_node) >= answer_count: return 10000     # enough answers -> stop
        if result > 1: return result - 1          # propagate the backtrack up one level
        # result == 1 -> next loop iteration is the sibling retry
    return 1
```

## Working code

Filling the `search()` slot of the tool-use tree harness (per-node bookkeeping: `is_terminal`,
`pruned`, `observation_code`, `expand_num`; primitive `_step`). The re-expansion nudge is built
here by `_diversity_message`, which renders the previous children of a node into a prompt that
asks the model for a different action:

```python
class DFSDT:
    """Depth-First Search-based Decision Tree: pre-order DFS over a tree of partial
    action sequences. One LLM call per node, no child ranking. Diversity nudge on
    re-expansion; backtrack distance encoded as the recursion's return value."""

    def search(self, root_node):
        self._dfs(root_node)

    def _dfs(self, now_node):
        """Recursive DFS. Returns how many more levels the caller should keep backing up."""
        # 1 = local sibling retry (ordinary failed call);
        # 2 = escape the local subtree (give-up or completed answer);
        # toward infinity -> full restart (flat resample). 2 is the middle dial.
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num          # monotone visit counter
        self.now_expand_num += 1

        # get_depth() is raw tree-node depth: Thought, Action, Action Input consume
        # three levels for one logical tool-use round.
        if (now_node.get_depth() >= self.single_chain_max_step
                or now_node.pruned or now_node.is_terminal):
            if now_node.is_terminal:                       # a real final answer
                self.status = 1
                self.terminal_node.append(now_node)        # record this solution path
                return final_answer_back_length            # keep looking elsewhere: back up 2
            now_node.pruned = True
            if now_node.observation_code == 4:             # status 4 == "Finish by Giving Up"
                self.give_up_node.append(now_node)
                return prune_back_length                    # escape this dead subtree: back up 2
            return 1                                        # ordinary failed call: local retry

        # --- expand: up to tree_beam_size children, one at a time, depth-first ---
        for _ in range(self.tree_beam_size):
            added_diversity = False
            if len(now_node.children) > 0:
                # We backed up here; temporarily show previous child actions so the
                # next generated child is different from them.
                now_node.messages.append(self._diversity_message(now_node))
                added_diversity = True

            # One canonical step: content -> Thought; function_call -> Action and
            # Action Input; status 4 marks give-up/pruned, status 1 rewrites a
            # hallucinated API name to a sentinel, status 3 marks terminal and
            # make_finish(final_answer_back_length).
            leaf = self._step(now_node)

            if self.query_count >= self.max_query_count:   # checked after the LLM parse
                return 100000                              # abort: unwind every frame and exit

            if added_diversity:
                now_node.messages[-1]["valid"] = False     # drop from persistent history

            result = self._dfs(leaf)                       # recurse immediately (pre-order, no ranking)

            if len(self.terminal_node) >= self.answer_count:
                return 10000                               # enough valid paths: unwind and stop
            elif result > 1:                               # a descendant wants to keep backing up
                return result - 1                          # propagate the retraction up one level
            # result == 1 -> local sibling retry == this loop's next iteration; fall through

        return 1                                            # no more children here: let parent retry
```

The flat ReAct / single-chain backbone is the special case where every node has exactly one child and the
recursion never returns `> 1`: a single forward chain, no retraction.
