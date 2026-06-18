**Problem.** The ranked DFS bought retraction (pass rate up) but at an order-of-magnitude query cost
(33–53 vs the chain's 3.7–5.5), because the O(n²) pairwise ranking spends a quadratic LLM-call tax per
branch point on an *ordering* of siblings. Since I need only *one* valid path and depth-first visits the
same node *set* regardless of sibling order, that tax is mostly wasted — worst on the budget-starved
Qwen-7B (52.7 queries, 0.36 pass).

**Key idea.** Keep the tree, the diversity nudge, and the integer-return backtracking; **drop the
ranking**. Pre-order traversal: generate *one* child via `_step`, recurse into it immediately, and only
generate the next sibling if that child's whole subtree dead-ends. No beam pre-generation, no
`_rank_nodes` ever. This is DFSDT (depth-first search with decision tree).

**Why it dominates the ranked rung.** Degrades *downward* to the greedy chain on easy instructions
(one child per node, same cost); degrades *upward* to full DFS's explored node set on hard ones
(depth-first's node set is order-independent and I stop on the first valid path). So it keeps the
hard-instruction reach while shedding the dominant cost. Per node: exactly one LLM call (the diversity
nudge is a prefix on the same call, not an extra one), so total calls ≈ nodes expanded.

**Mechanics.** Diversity nudge on re-expansion (marked invalid after `_step`). Backtrack by returned
climb length: `2` for a final answer or a give-up (escape the local subtree), `1` for an ordinary failed
call (local sibling retry), `10000` once enough answers, `100000` at budget exhaustion; decrement and
propagate up.

**What to watch (vs ranked).** `pass_rate` holds or rises on every backbone (same retraction reach),
rising most on Qwen-7B where ranking starved the search; `avg_queries` collapses toward the chain's
regime; `give_up_rate` stays low; `sopr` tracks pass rate. Strongest rung — full retraction at
single-chain per-node cost.

```python
    def search(self, root_node):
        """DFSDT: generate one child, recurse immediately, backtrack on failure."""
        self._dfsdt(root_node)

    def _dfsdt(self, now_node):
        """Recursive DFSDT. Returns number of levels to backtrack."""
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

        # Try beam_size times (each time generates one child and recurses)
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

            leaf = new_leaves[-1]

            # Immediately recurse (no ranking)
            result = self._dfsdt(leaf)
            if len(self.terminal_node) >= self.answer_count:
                return 10000
            elif result > 1:
                return result - 1

        return 1
```
