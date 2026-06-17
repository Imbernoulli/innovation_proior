# DFS with LLM Ranking, distilled

DFS with LLM ranking is a search strategy for a tool-use agent. The agent still acts in the
ReAct format -- Thought, Action, Action Input, Observation -- but it no longer commits to one
irreversible chain. It keeps a decision tree of interaction states, samples several candidate
continuations at a state, uses the LLM to rank sibling continuations by pairwise comparison, and
then explores them depth-first with explicit backtracking.

## Problem it solves

A tool-use agent fulfills an instruction by emitting actions
`a_t ~ p(a_t | a_1, r_1, ..., a_{t-1}, r_{t-1}, Inst)`, where `a_t` is
(thought, API name, parameters) and `r_t` is the real API response. Single-chain strategies have
no decision retraction: an early malformed call, hallucinated API name, or unhelpful tool choice
stays in the context and conditions every later step. They also explore only one direction in an
effectively unbounded action space.

## Key idea

Decision retraction requires alternative successors from the same interaction state. Each node is
a partial trajectory; each child is one possible next action and observation; returning to a parent
and trying another child is the local "undo and retry" operation. The search then needs four
pieces:

- **Sample a bounded beam.** The action space is unbounded, so generate only `tree_beam_size`
  candidate continuations from a node.
- **Force sibling diversity.** When a node already has children, add a temporary prompt showing
  the previous actions from that node and requiring a different action; mark that prompt invalid
  afterward so descendants do not inherit it.
- **Rank siblings by symmetric pairwise comparison.** For each unordered pair, compare the two
  divergent suffixes after their common prefix twice, once in each presentation order. A candidate
  wins the pair only when the two orderings agree on the same underlying candidate; disagreement
  is split as a tie. Round-robin scores are sorted descending before expansion.
- **Backtrack by returned climb length.** Recursive calls return how many levels the caller should
  keep climbing: `2` for a final answer or give-up leaf, `1` for an ordinary dead end, `10000`
  after enough answers have been found, and `100000` when the model-call budget is exhausted.

## Ranking cost

For `n` generated children, pairwise ranking evaluates every unordered pair in both orders. That
is `2 * (n choose 2) = n(n - 1)` LLM calls, so the ranking cost is `O(n^2)` extra model queries at
each ranked node. The code adds these comparison calls and tokens to the same `query_count` and
`total_tokens` used by action generation. It does not immediately return after ranking if those
comparisons cross the cap; the next generation call is where the `query_count >= max_query_count`
guard returns `100000`.

## Final algorithm

```text
DFS(node):
    if depth(node) >= max_step or node.pruned or node.is_terminal:
        if node.is_terminal:
            record final-answer node
            status <- 1
            return 2
        node.pruned <- True
        if node.observation_code == 4:
            record give-up node
            return 2
        return 1

    children <- []
    repeat tree_beam_size times:
        if node already has children:
            append temporary diversity prompt
        call model for one assistant message
        query_count <- query_count + 1
        if query_count >= max_query_count:
            return 100000
        invalidate temporary diversity prompt
        parse the result, execute any tool call, and append a new leaf
        children.append(leaf)

    if len(children) > 1:
        scores <- symmetric pairwise round-robin(children)
        query_count <- query_count + rank_query_count
        total_tokens <- total_tokens + rank_tokens
        sort children by score descending

    for child in sorted children:
        result <- DFS(child)
        if number of final answers >= answer:
            return 10000
        if result > 1:
            node.make_finish(2)
            return result - 1

    return 1
```

## Working code

Faithful distillation of the `with_filter=True` branch of `DFS.py`. The helper `_expand_once`
stands for the existing ToolBench/StableToolBench parsing block that creates Thought, Action, and
Action Input nodes, sets status codes, and appends assistant/function messages.

```python
def search(self, root_node, single_chain_max_step, tree_beam_size,
           max_query_count, answer=1):
    return self.DFS(root_node, single_chain_max_step, tree_beam_size,
                    max_query_count, answer, with_filter=True)


def DFS(self, now_node, single_chain_max_step, tree_beam_size,
        max_query_count, answer, with_filter=True):
    """Return the number of levels still to climb."""
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
            self.io_func.functions,
            process_id=self.process_id,
        )
        self.query_count += 1
        self.total_tokens += total_tokens

        if self.query_count >= max_query_count:
            return 100000

        if delete_former_diversity_message:
            temp_now_node.messages[-1]["valid"] = False

        temp_now_node = self._expand_once(
            temp_now_node,
            new_message,
            error_code,
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
            self.llm,
            LLM_rank_args=rank_args,
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
                          max_query_count, answer, with_filter=True)
        if len(self.terminal_node) >= answer:
            return 10000
        if result > 1:
            now_node.make_finish(2)
            return result - 1

    return 1
```

`observation_code == 4` is the give-up finish. A status-3 tool result is handled in `_expand_once`
by marking the new Action Input node terminal and calling `make_finish(final_answer_back_length)`.
In the ranked branch, generated children are stored, scored by `sum_based_rankn`, sorted by
`prior_score`, and only then expanded.

## Relation to prior strategies

- **ReAct** is the one-child-per-state special case: no local retraction and no sibling choice.
- **Reflexion** retries the whole task after a failed attempt; this search retries from an
  intermediate state while keeping the good prefix.
- **Tree-of-thoughts DFS** supplies the broad search framing, but tool use needs sampled action
  continuations, real API observations, pairwise sibling judgment instead of absolute
  threshold-pruning, and explicit query-budget accounting.
