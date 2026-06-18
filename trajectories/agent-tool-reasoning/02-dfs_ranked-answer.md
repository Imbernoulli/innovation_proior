**Problem.** The greedy chain can't retract a committed action: an early wrong call propagates into a
faulty loop and one linear pass explores a single route through an unbounded action space. The fix is
to turn the chain into a *tree of partial trajectories* and search it with backtracking.

**Key idea.** A node is a partial trajectory; a child is one possible next action; returning to a parent
and trying another child is "undo and retry." Search the tree **depth-first** (one valid path suffices,
so don't expand the whole frontier), and at each node:

- **Sample a bounded beam.** Call `_step` up to `tree_beam_size` times to get candidate continuations
  (the action space is unbounded, so it can't be enumerated).
- **Force sibling diversity.** On re-expansion, `_add_diversity_prompt(node)` appends a temporary prompt
  listing prior actions and asking for a different one; mark it invalid after the `_step` call.
- **Rank siblings by LLM pairwise comparison.** `_rank_nodes(candidates)` (via `sum_based_rankn`)
  scores siblings; sort descending and expand the best-judged child first.
- **Backtrack by returned climb length.** Each recursive call returns how many levels the caller should
  keep climbing: `2` for a final answer or a give-up (escape the local subtree), `1` for an ordinary
  failed call (local sibling retry), `10000` once enough answers are found, `100000` when the budget is
  exhausted. The integer is decremented and propagated up the stack.

**Why / cost.** For `n` children, pairwise ranking spends ~`n(n-1)` extra LLM calls per ranked node —
O(n²) queries charged to the same `query_count` — whose only effect is the *order* of expansion. Since
depth-first visits the same node *set* regardless of sibling order, this is a large evaluation tax for
ordering alone.

**What to watch.** `pass_rate` should rise over the chain on the harder backbones (DeepSeek 0.44,
Qwen-7B 0.30) where wrong-tool commitment hurt most; `give_up_rate` low; but `avg_queries` should
explode (vs the chain's 3.7–5.5) from the ranking. If the gain doesn't justify the cost, step 3 keeps
the tree and drops the ranking.

```python
    def search(self, root_node):
        """DFS with LLM ranking: expand best child first, backtrack on failure."""
        self._dfs(root_node)

    def _dfs(self, now_node):
        """Recursive DFS. Returns number of levels to backtrack."""
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1

        # Base cases
        if now_node.get_depth() >= self.single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                return 1

        # Generate beam_size children
        candidates = []
        for i in range(self.tree_beam_size):
            if self.query_count >= self.max_query_count:
                return 100000

            # Add diversity prompt if node already has children
            added_diversity = self._add_diversity_prompt(now_node)

            new_leaves = self._step(now_node)

            # Mark diversity message as invalid
            if added_diversity:
                now_node.messages[-1]["valid"] = False

            if not new_leaves:
                continue
            candidates.append(new_leaves[-1])

        if not candidates:
            return 1

        # Rank candidates using LLM pairwise comparison
        if len(candidates) > 1:
            scores = self._rank_nodes(candidates)
            for score, node in zip(scores, candidates):
                node.prior_score = score
            candidates.sort(key=lambda x: x.prior_score, reverse=True)

        # Expand best candidates in order
        for cand in candidates:
            result = self._dfs(cand)
            if len(self.terminal_node) >= self.answer_count:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1
```
