The ranked search confirmed the bet was half right and half a waste, and the numbers say which half. Retraction works: `pass_rate` jumped from the chain's 0.44 to 0.72 on DeepSeek and 0.62 to 0.82 on Qwen-72B, with `give_up_rate` falling to 0.02 on both — the tree found valid paths the single chain couldn't by abandoning wrong branches. But the price is the whole story: `avg_queries` exploded from 3.7 / 3.9 / 5.5 to 33.1 / 35.1 / 52.7, roughly an order of magnitude. The worst signal is on the weakest backbone — Qwen-7B spent the *most* queries on the board (52.7) for only 0.36 pass and 0.20 sopr, barely above the chain's 0.30. On the backbone that most needs help, the ranking tax bought almost nothing: it burned the budget reordering siblings and ran out before exploring enough of the tree. The diagnosis is precise — the gain came from *retraction*, the cost came from *ranking*, and the $O(n^2)$ pairwise ranking pays a quadratic LLM-call tax per branch point to produce an *ordering* of siblings I am not sure I need. I do not need the best path, and I do not need to rank anything; I need exactly one path that reaches a final answer within a tight budget (`answer_count` is 1). Every call spent comparing children instead of advancing toward a terminal is, for that goal, wasted.

So the question is sharp: do I even need the ranking? Its only job is to expand the most promising child first so I descend faster. But depth-first visits the *same set* of nodes regardless of the *order* it visits siblings, and I terminate the instant I find a valid path either way — ranking changes only the order, never the eventual set. The one thing it can buy is reaching a terminal a few expansions sooner *on the branch where the best child happens not to be the first-generated one*, and it pays $n(n-1)$ comparison calls per branch point for that chance. Empirically, on these tasks the child the model generates *first* — its own first instinct at a node — is usually the one that ends up ranked highest; the model's strong mode is, most of the time, its best move. So the ranking is, in the common case, paying a quadratic tax to re-derive an ordering that generation order already approximates for free. That is the waste the 33–53 query counts encode.

I propose **DFSDT — depth-first search with a decision tree**: keep the tree, the diversity nudge, and the integer-return backtracking, and *drop the ranking*. Generate one child, recurse into it immediately, and only generate a second child at this node if the first child's whole subtree dead-ends. That is a pre-order traversal — no beam, no sorting, no `_rank_nodes` call anywhere — and it is the right generalization, not a hack, because it brackets exactly between the two rungs I have run. It degrades *downward*: on a simple instruction where the model never has to retract — think, call the right API, finish — the tree is a single chain with one child per node and the procedure is *identical* to the greedy chain, at the same cost; I have not made the easy case more expensive, which the ranked rung could not promise because it ranked even when there was nothing to rank. And it degrades *upward*: after the search finishes, the set of nodes explored is almost exactly the set a full ranked depth-first search would have explored, because the depth-first node set is order-independent and I stop on the first valid path. So I keep the hard-instruction-solving power of real DFS while shedding its dominant cost. That is the precise sense in which DFSDT dominates the ranked rung: same retraction, same exploration reach, minus the quadratic evaluation tax.

Dropping ranking is the *only* change; the parts doing real work stay. The tree structure stays — nodes are partial trajectories, children are alternative actions, retraction is walking up and descending into a different child. The diversity nudge stays and is *more* important now, not less: when a branch dead-ends and I retract to a node that already has a child, the model left alone will regenerate the very action that just failed and loop forever. So before generating the next child at a node that already has children I call `_add_diversity_prompt(node)` to show the model the actions it already tried here and demand a different one, and I mark that temporary message invalid after the `_step` call so it does not contaminate the child's context or the recorded history. Without ranking to diversify by selection, the nudge carries the entire diversification load — it is what turns "retract and re-expand" into genuine exploration. The backtracking machinery stays identical too, because the integer-return control flow was never the expensive part; `_rank_nodes` was. Each recursive call returns how many levels the caller should keep climbing. Base cases: at the depth limit, or pruned, or terminal — if terminal (a real final answer) record it, set `status = 1`, return `final_answer_back_length = 2`; if pruned, mark it, and if `observation_code == 4` it is the explicit give-up, record it in `give_up_node` and return `prune_back_length = 2`, else (an ordinary failed call) return 1. The values encode how bad the dead end was: 1 is a local sibling retry (a lone failed call is local, not a sign the subtree is poisoned), 2 escapes the local subtree (a give-up or a completed answer I want to look past, so I diversify one level higher rather than retry right next to it). Sentinels: 100000 if a `_step` pushes `query_count` to the cap (unwind everything and exit), 10000 once `answer_count` answers are collected.

The one place DFSDT differs structurally from the ranked rung — beyond not ranking — is *when* it recurses. The ranked rung generated the whole beam first, stored the leaves, ranked them, sorted, and only then expanded. DFSDT generates one child and immediately recurses into it; it loops up to `tree_beam_size` times, but each iteration generates exactly one new child and dives into it before considering another. So the loop body is: check budget; if this node already has children, add the diversity nudge; call `_step` once for one leaf; mark the nudge invalid; recurse into the leaf and read its returned climb length. If I now have enough answers, return 10000. If the returned value is greater than 1, a descendant wants me to keep backing up past this level, so decrement and return `result - 1`. If it is exactly 1, the request was consumed at this frame, so I stay in the loop and generate the next sibling. If the loop exhausts `tree_beam_size` attempts without a propagating return, return 1 and let my parent try its next sibling. One subtle difference from the ranked rung's per-node handling: DFSDT does *not* call `make_finish(2)` on the way up the way the ranked branch did — there is no ranked-completion bookkeeping to settle, so the integer alone carries the control flow. Trace it once: deep down a node gives up and returns 2; its parent's loop sees `result = 2 > 1` and returns $2 - 1 = 1$; the grandparent sees `result == 1`, does *not* return, and generates its next child — a sibling two levels above the give-up. That is exactly "back up two levels and branch." Had the give-up returned 1, the parent would try the give-up's sibling, a one-level local retry.

The cost story is the entire point. Per node I spend exactly one LLM call — the single `_step` that generates its one child; no `_rank_nodes`, ever; the diversity nudge is a prefix on that same call, not an extra one. So total LLM calls $\approx$ number of nodes expanded, capped by `max_query_count`. A simple instruction is a single chain at the greedy chain's cost; a hard one is a chain plus however many sibling branches the dead ends forced open, node-for-node, stopping the instant I find one answer or hit the budget. Compare the ranked rung at a branch point with `tree_beam_size` = 3: it spent ~3 generation calls *plus* ~6 comparison calls (the $3 \cdot 2$ ordered pairs); DFSDT spends the generation calls only, and often fewer of those because it generates the next sibling only on failure rather than pre-generating the whole beam. That is where the order-of-magnitude saving lives. So against the ranked numbers I expect `pass_rate` to *hold or rise* on every backbone — same retraction, same exploration reach — rising most on the budget-starved Qwen-7B, where dropping the ranking frees the budget to actually explore the tree instead of burning it on comparisons; `avg_queries` to *collapse* toward the chain's regime relative to the ranked 33–53; `give_up_rate` to stay low; and `sopr` to track `pass_rate`. The sharpest prediction is the Qwen-7B cell: 52.7 queries for 0.36 pass should give way to a better pass at a fraction of the queries. If `pass_rate` instead *dropped* without ranking, the ordering was doing real work after all — but the order-independence of the node set says the opposite, so I expect strictly cheaper, at least as good, and meaningfully better where the budget was binding. This is the strongest rung — full retraction and exploration at single-chain per-node cost — and the honest next place to look is not "rank better" (that path is closed) but a search that spends its calls evaluating value where it is worth it and reusing that value across branches, which is where a Monte-Carlo tree search over this same action space comes in.

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
