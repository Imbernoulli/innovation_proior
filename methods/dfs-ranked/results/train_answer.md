A language model acting as a tool-use agent must produce a sequence of actions conditioned on everything it has seen so far: the instruction, the actions it already took, and the real responses returned by the APIs. The action space at each step is effectively unbounded, because the model can write any thought, choose any API, and pass any parameter string. A single chain of thought and action works on simple instructions, but on hard multi-tool tasks it collapses as soon as one step goes wrong. A malformed parameter or hallucinated API name gets written into the context and conditions every later thought, so the agent can loop on the same bad call forever. Worse, a single chain explores only one path; if the first tool choice is unhelpful, the agent never learns that a different tool would have worked. What is missing is a way to retract a bad action locally, keep the good prefix, and try a different continuation from an earlier state.

Existing strategies do not provide this. Chain-of-thought keeps reasoning entirely inside the model and never executes anything, so it cannot even ground itself in real API responses. ReAct interleaves thought and action, but it is still one linear trajectory with no undo: every committed action is permanent. Reflexion retries the whole task after a failure with a verbal reflection prepended, which helps between trials but not within a trial; if only step seven of fifteen was wrong, throwing away the first six steps and regenerating them is wasteful and reintroduces risk. Tree search over thoughts works on small puzzles with enumerable next steps and resettable environments, but tool use has unbounded continuations, irreversible real actions, and noisy self-evaluation. A fixed scalar value threshold does not transfer to messy tool trajectories, and breadth-first expansion wastes budget expanding whole frontiers when a single valid path would suffice.

The method I propose is DFS with LLM ranking. It keeps the ReAct substrate, interleaving Thought, Action, Action Input, and Observation, but it treats the interaction history as a decision tree rather than a chain. Each node is a partial trajectory; each child is one possible next action and its real observation. Returning to a parent and descending into a different child is the local undo-and-retry operation. Because only one passing path is needed, the tree is searched depth-first. The four mechanisms that make this practical are a bounded beam, forced sibling diversity, symmetric pairwise ranking, and backtracking by returned climb length.

At each non-terminal node I generate up to tree_beam_size candidate continuations. The action space cannot be enumerated, so sampling is unavoidable. Naive repeated sampling at the same state often yields nearly identical actions, because the model has a strong mode. To avoid this, whenever the node already has children I append a temporary diversity prompt that lists the previous actions generated from this state and asks for a different one. That prompt is marked invalid immediately after the generation call so that descendants do not inherit a sibling-specific instruction.

Once the siblings are generated, I rank them with the LLM itself. Absolute scalar self-scores on half-finished tool trajectories are noisy and instruction-dependent, so I use relative judgment instead. For every unordered pair of candidates, I find their common prefix and present only the divergent suffixes, asking the model which continuation is more helpful. A single prompt ordering can be biased, so I ask twice with the candidates swapped. A candidate wins the pair only if both orderings agree on it; disagreement is counted as a tie. The round-robin scores are accumulated, the candidates are sorted by score descending, and the best-judged child is expanded first. This costs n times n minus one extra model calls per ranked node, because each unordered pair needs two comparisons, so the ranking is quadratic in the beam size and is charged to the same query budget as action generation.

The backtracking logic uses an integer return value that tells the caller how many levels to climb. If a node is a final answer, I record it, set the search status to success, and return two so that the search escapes the immediate neighborhood before trying more siblings. If the node is a give-up, I record it and return two for the same reason. An ordinary dead end returns one, so the parent tries the next sibling right away. If enough answers have been found or the query budget is exhausted, I return large sentinel values that unwind the recursion entirely.

On easy instructions this collapses to the greedy chain: the first generated child at each node is good, so it is followed all the way to the answer with no extra ranking cost. On hard instructions it pays the ranking premium only when the first child fails, exploring alternatives depth-first while preserving the good prefix. Here is a concise implementation:

```python
def search(self, root_node):
    """DFS with LLM ranking: expand best child first, backtrack on failure."""
    return self.DFS(root_node, single_chain_max_step=self.single_chain_max_step,
                    tree_beam_size=self.tree_beam_size,
                    max_query_count=self.max_query_count,
                    answer=self.answer_count, with_filter=True)


def DFS(self, now_node, single_chain_max_step, tree_beam_size,
        max_query_count, answer, with_filter=True):
    """Return how many levels the caller should keep climbing."""
    final_answer_back_length = 2
    prune_back_length = 2

    now_node.expand_num = self.now_expand_num
    self.now_expand_num += 1

    if (now_node.get_depth() >= single_chain_max_step
            or now_node.pruned
            or now_node.is_terminal):
        if now_node.is_terminal:
            self.status = 1
            self.terminal_node.append(now_node)
            return final_answer_back_length

        now_node.pruned = True
        if now_node.observation_code == 4:
            self.give_up_node.append(now_node)
            return prune_back_length
        return 1

    next_tree_split_nodes = []
    for _ in range(tree_beam_size):
        temp_now_node = now_node

        delete_former_diversity_message = False
        if len(temp_now_node.children) > 0:
            diversity_message = self._build_diversity_message(temp_now_node)
            if diversity_message is not None:
                temp_now_node.messages.append(diversity_message)
                delete_former_diversity_message = True

        self.llm.change_messages(temp_now_node.messages)
        new_message, error_code, total_tokens = self.llm.parse(
            self.io_func.functions, process_id=self.process_id)
        self.query_count += 1
        self.total_tokens += total_tokens

        if self.query_count >= max_query_count:
            return 100000

        if delete_former_diversity_message:
            temp_now_node.messages[-1]["valid"] = False

        temp_now_node = self._expand_once(
            temp_now_node, new_message, error_code,
            final_answer_back_length,
        )
        next_tree_split_nodes.append(temp_now_node)

    if len(next_tree_split_nodes) > 1:
        rank_args = {
            "functions": self.io_func.functions,
            "process_id": self.process_id,
            "task_description": self.io_func.task_description,
            "rank_func": rank2_subfix,
        }
        scores, rank_query_count, rank_tokens = sum_based_rankn(
            self.llm, LLM_rank_args=rank_args,
            candidates=next_tree_split_nodes,
        )
        self.query_count += rank_query_count
        self.total_tokens += rank_tokens

        for score, node in zip(scores, next_tree_split_nodes):
            node.prior_score = score
        next_tree_split_nodes = sorted(
            next_tree_split_nodes,
            key=lambda node: node.prior_score,
            reverse=True,
        )

    for child in next_tree_split_nodes:
        result = self.DFS(child, single_chain_max_step, tree_beam_size,
                          max_query_count, answer, with_filter)
        if len(self.terminal_node) >= answer:
            return 10000
        if result > 1:
            now_node.make_finish(2)
            return result - 1

    return 1
```

The helper `_expand_once` parses one assistant message into Thought, Action, and Action Input nodes, executes the tool call, sets the observation code, marks status three as terminal, and appends the assistant and function messages to the conversation. In the ranked branch, children are generated, scored by symmetric pairwise comparison, sorted, and expanded depth-first. This gives the agent local retraction and multi-path exploration while charging every comparison call to the same query budget, degrading to a single ReAct chain when the first choice is already correct.
