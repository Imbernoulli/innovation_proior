The task is to make a frozen LLM act as a planner for multi-step tool use. Each round it must emit a thought, choose one API from a large documented set, fill in that API's parameters, and let the environment execute the call. The response then becomes part of the history, and the process repeats until the model either produces a final answer or gives up. The action space is effectively infinite because the thought, the API choice, and the arguments are all free-form, and the environment can return real errors, timeouts, or misleading payloads. A single wrong action is dangerous: once it is written into the conversation history, the model conditions on it and tends to compound the mistake, either looping on the same bad call or drifting further away from a valid solution. At the same time, a single linear trajectory can only explore one path through a huge space, so even a strong model often fails on harder multi-tool instructions.

Existing approaches show the right pieces but not the right combination. Chain-of-thought and ReAct interleave reasoning with real tool calls, which grounds the agent in actual observations, but they move only forward: there is no way to retract a bad decision. Reflexion adds a failure loop, yet it restarts the entire episode from the root, throwing away a long correct prefix just to fix one late mistake. Tree-of-thoughts introduces branching and backtracking, but its standard form ranks children with a value heuristic, and in this setting ranking is paid for with O(n log n) extra LLM calls per node. That evaluation tax is exactly what the budget cannot afford, because only one valid path is needed, not the best one. The same problem afflicts textbook depth-first search with child ranking: it solves hard instances but spends most of its budget comparing siblings instead of advancing toward a terminal.

The method I propose is DFSDT, short for Depth-First Search-based Decision Tree. It keeps the grounded acting template—thought, real API call, observation, repeat—but organizes the partial trajectories into a tree that can be searched depth-first. A node is a state reached after some sequence of thoughts and calls; the root is the bare instruction; leaves are either final answers, give-up decisions, or pruned API failures. When a leaf dead-ends, the search retracts to an ancestor and tries a different child, preserving the good prefix and revising only the bad suffix. The crucial difference from ranked DFS is that DFSDT does not rank children at all. It uses pre-order traversal: at each node it generates exactly one child with a single LLM call, recurses into that child immediately, and only generates the next sibling if the entire subtree below the first child fails. Depth-first is the right order because only one valid path is required; breadth-first would expand whole frontiers before reaching any answer. Skipping ranking removes the dominant cost, because depth-first visits the same set of nodes regardless of the order in which siblings are expanded, and the model's first-generated child is usually the one a ranking would place first anyway.

DFSDT also needs a way to avoid repetition on re-expansion. When the search backs up to a node that already has children, the same context would otherwise tend to regenerate the same failed action. To prevent this, before generating the next child the method temporarily appends a diversity prompt that lists the actions already tried at this state and asks the model to produce something different. Once the child is generated, that prompt is marked invalid so it does not contaminate the child's persistent history or any future training data derived from the trajectory.

The backtrack distance is the only real control knob, encoded as an integer returned up the recursion. An ordinary failed API call returns 1, which the parent consumes by simply trying the next sibling right there. A "Finish by Giving Up" status or a completed final answer returns 2, so the parent returns 1 and the grandparent consumes the backup by branching two levels above the dead leaf. That escapes the immediate bad neighborhood without restarting the whole episode. Enough answers found or budget exhausted returns a large sentinel that unwinds the whole stack. On easy instructions where nothing ever goes wrong, the tree collapses to a single chain and DFSDT costs exactly the same as a flat ReAct agent; on hard instructions it retains the exploration power of full DFS while paying only one LLM call per expanded node.

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
