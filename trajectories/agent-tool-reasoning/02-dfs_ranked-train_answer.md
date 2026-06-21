The greedy chain told me what was missing, and it told me in numbers: its `avg_queries` was the floor — 3.7 on DeepSeek, 3.9 on Qwen-72B, 5.5 on Qwen-7B — which is the symptom, not a virtue, because the agent makes one guess and, when it dead-ends, has nowhere to go but to stop. Its pass rates were the floor I expected (0.44 / 0.62 / 0.30), tracking raw backbone strength, with a non-trivial `give_up_rate` everywhere (0.04 / 0.04 / 0.06) — the tell that the failures are escapable dead ends, not unsolvable tasks. So the failure is structural: the chain has no operation that means "undo and branch." Once an action is committed the agent moves forward; an early wrong call sits in the context conditioning every later thought, and a single linear pass commits to exactly one route through an unbounded action space. The fix is not a hyperparameter — it is to give the agent a way to take an action back. The cleanest existing idea for using a failure is the verbal-reflection loop (Reflexion, Shinn et al. 2023): on a failed attempt, write down what went wrong and retry the *whole task from the root* with the reflection prepended. But the unit of backup there is the entire episode — if step nine of a long trajectory is the only mistake, restarting from step one throws away eight good steps, and in a setting where every step is a paid LLM call that is enormously wasteful. What I actually want is to keep the good prefix and revise only the bad suffix: back up *a little*, to where it went wrong, not all the way to the start.

The instant I say "keep the prefix, return to a prior state, and choose differently," I am no longer describing a chain — a chain has one successor per state, and this has several, with the ability to return and pick another. That is a decision tree, and the method I propose is **DFS with LLM ranking** over it. A node is a state of the interaction — the partial trajectory of thoughts, calls, and observations reached so far; the root is the bare task; the children of a node are the different actions the agent could take from there; a leaf is a finish, either a real final answer (a good terminal) or a give-up / API-error dead end (a pruned node). Retraction is just "walk up from a dead leaf to an ancestor and descend into a different child" — exactly "keep the prefix up to the ancestor, revise the suffix below it" — and the greedy chain falls out as the degenerate tree with one child per node. The harness makes this natural: `_step(node)` expands a node into a leaf, nodes carry `father`/`children`/`messages`, and the policy I write is the search over that tree.

How I search it is where the real design lives, because the textbook tree-of-thoughts recipe is built for a different problem. That recipe generates $k$ candidate children at each node, *scores* them with a value heuristic — rate each partial state 1 to 10, or vote — and uses the score both to order expansion and to prune branches below a threshold. On a small brute-forceable puzzle that is fine; a tool trajectory is different on two axes. First, the state is a half-finished multi-API interaction with messy real observations, so an absolute "rate this 1 to 10" is noisy, instruction-dependent, and badly calibrated, and pruning on a fixed threshold throws away good branches whose self-score happened to land low while keeping bad ones that scored high. *Relative* judgment survives this far better — the model needn't decide what an absolute 7 means, only whether continuation A or continuation B is more helpful for this same instruction from this same prefix. So I compare candidates *pairwise* rather than assign absolute values, and the harness hands me exactly that primitive: `_rank_nodes` runs `sum_based_rankn` over candidate nodes and returns per-candidate scores. Second, I cannot enumerate children — the next action is the Cartesian product of every free-form thought, every API, and every parameter string — so I sample a fixed number, `tree_beam_size` candidate continuations, by calling `_step` repeatedly from the same node.

Naive repeated sampling has a trap that is the same trap that would otherwise make backtracking useless: call `_step` $k$ times at one node and the model's strong mode returns $k$ near-identical actions, so when I back up to a node that already has children and just regenerate, I get the action that already failed and loop forever. So when a node already has children, before generating the next one I actively push the model off its prior choices with `_add_diversity_prompt(node)`, which appends a temporary prompt listing the actions already tried here and asks for a different one. It returns `True`, and I must then mark that message invalid after the `_step` call — the nudge is scaffolding for *generating* the new child and must not live on in the child's context or the recorded history; it does its job at generation time and disappears. That turns "retract and re-expand" into genuine exploration instead of repetition.

Breadth-first would expand the whole frontier at each depth before going deeper, generating and partially running a whole layer of alternatives before reaching any terminal — but I need only *one* valid path (`answer_count` is 1). Depth-first drives straight down one branch to a terminal as fast as possible and pays for siblings only when the chosen branch fails, so it reaches the stopping condition with far fewer calls. Hence ranked depth-first: at a node, generate `tree_beam_size` children (with the diversity nudge on re-expansion), rank them pairwise with `_rank_nodes`, sort by score descending, expand the best-judged child first, recurse, and on failure come back and take the next-best.

The backtracking machinery has to be exact, because after a leaf fails I may want to jump more than one level up, and a plain true/false return is too weak. So each recursive call returns an integer: how many levels the caller should still climb. The base cases fix the arithmetic. If the node is terminal — a real final answer — record it, set `status = 1`, return `final_answer_back_length = 2`. If it is at the depth limit or already pruned, mark it pruned and check why: `observation_code == 4` is the explicit give-up, recorded in `give_up_node`, returning `prune_back_length = 2`; otherwise an ordinary failed call returns 1. The constants encode *how bad* the dead end was — 1 means "local sibling retry, this single call failed but its neighborhood may be fine," 2 means "escape the local subtree (a give-up, or a completed answer I want to look past), so don't retry right next to it, diversify one level higher." Two sentinels handle global exits: if a generation pushes `query_count` to `max_query_count`, return 100000 so every frame unwinds and the search exits; once `answer_count` answers are collected, return 10000 for the same hard unwind. Trace the climb to see the integer is right: a leaf returns 2; its parent's loop sees `result = 2 > 1`, marks itself finished with `make_finish(2)`, and returns $2 - 1 = 1$; the grandparent sees `result == 1`, does *not* keep unwinding, and instead generates its next sibling — two levels above the failed leaf. That is "back up two levels and branch." Had the leaf returned 1, the parent immediately tries the leaf's *sibling*, a one-level local retry. Decrement-and-propagate converts a single integer into "unwind exactly $N$ frames, then branch."

The cost I am signing up for is the crux. The budget check happens *after* `_step` increments `query_count`, and `_rank_nodes` charges its comparison calls to the same counter, so ranking is not free. For $n$ generated children the pairwise ranking compares every unordered pair, and to beat order bias it asks twice (each presentation order), so it spends on the order of $n(n-1)$ extra LLM calls at each ranked node — $O(n^2)$ queries whose only job is to *reorder* siblings I am going to depth-first-explore anyway. That buys me the *order* of expansion, not the *set* of nodes explored, and depth-first visits the same set regardless of order. So I expect `pass_rate` to rise substantially over the chain on the harder backbones — DeepSeek's 0.44 and especially Qwen-7B's 0.30, where the chain most often committed to a wrong tool, are exactly what retraction targets — `give_up_rate` to drop or stay low, and `avg_queries` to *explode* relative to the chain's 3.7–5.5 as the necessary price of the ranking. The risk I can already feel: if the model's first-generated child is usually its best instinct, I will have spent the bulk of my budget reordering siblings depth-first would have reached anyway, and if `avg_queries` balloons while `pass_rate` lands only at or below a rank-free depth-first search, the diagnosis for the next rung is already written — keep the tree and the backtracking, drop the ranking.

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
