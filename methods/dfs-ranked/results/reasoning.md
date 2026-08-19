Let me start from what actually breaks when I point a language model at a pile of real APIs and ask it to fulfill an instruction. The interaction is a loop: at round t the model reads everything that happened -- the instruction plus the alternating actions and real API responses a_1, r_1, ..., a_{t-1}, r_{t-1} -- and emits the next action a_t, which is a thought plus an API name plus the parameters for that call. Then I actually execute the call, the world hands me back r_t, and the loop repeats until the model decides to stop. The way everyone does this today is one straight chain: think, act, observe, think, act, observe, finish. Reasoning interleaved with acting, each observation feeding the next thought. And on simple instructions it is fine. But I keep watching it fail on the hard, multi-tool ones, and the failures have a shape I need to understand before I try to fix anything.

The first failure is that one bad action poisons everything downstream. Say at step 3 the model calls an API with a malformed parameter, or worse, hallucinates an API name that does not even exist. The call errors, but the chain does not know to treat that as a fork in the road: the bad action and its error response are now sitting in the context, and every subsequent thought is conditioned on them. So the model does the obvious thing: it tries the same broken call again, slightly reworded, and again, and gets stuck in a loop calling a nonexistent or misused API. The error does not get corrected; it propagates. The second failure is just as basic: a single chain explores exactly one path through the action space. If the first tool the model reaches for is the wrong tool for this instruction, the chain commits to it and rides it all the way down; it never tries the other tool that would have worked. On complex instructions even a strong model can fail to find any valid solution path this way. So this is not only a tuning problem; the single-chain structure itself is the problem.

What is the structural root? Stare at it. The chain has no way to take back an action. Once a_t is committed, the agent moves forward; after seeing r_t it cannot decide "that was a wrong turn, let me go back to the state I was in before a_t and pick a different action instead." Every action is permanent. So of course an early error is permanent too. There is no operation in the agent's repertoire that means "undo and branch." That is the thing I am missing: retraction.

Let me look at what tools I have for this before inventing anything. ReAct is the chain I just described: interleave Thought, Action, and Observation. Its whole insight is that reasoning and acting help each other: the thought tracks the plan and interprets surprises, the action grounds the reasoning in real feedback so it stops hallucinating. That is genuinely the right substrate for a tool agent, and I want to keep it. But it bakes in the one-trajectory assumption; there is no retraction, by construction. Then Reflexion adds a verbal learning loop: run a trial, and if it fails, have the model write a natural-language reflection on why it failed, stash that in memory, and retry the whole task with the reflection prepended. That attacks error propagation, but the unit of revision is the entire trial. Each retry starts over from the beginning with extra advice. It never branches within a trajectory; it never keeps two half-finished paths alive and chooses between them; it never says "the first six steps were fine, it is step seven that was wrong, back up to step six." If step seven of fifteen was the only mistake, throwing away steps one through six and restarting is wasteful, and re-deriving those six steps re-exposes me to fresh mistakes on each one. So Reflexion corrects between attempts; I need something that corrects within an attempt, at the granularity of a single action.

So I want to keep an action when it is good and revise it when it is bad, locally, mid-trajectory. The instant I phrase it that way -- keep the prefix, revise one step, and be able to come back to a prior state and choose differently -- I am not describing a chain anymore. A chain has one successor per state. What I am describing has multiple possible successors at a state, with the ability to return to that state and pick another. That is a decision tree. Each node is a state of the interaction; the path from the root to a node is a partial trajectory; and the children of a node are the different actions the agent could take from that state. Retraction is just "move back up to the parent and descend into a different child." The chain falls out as the degenerate tree where every node has one child. Good: the object is a decision tree over interaction states, and the question becomes how to build and search it without exploding the cost.

There is prior work that already thinks of LLM problem-solving as tree search. It lays out the knobs I need: a way to generate candidate next steps from a state, a way to evaluate how promising a state is, and a classical search over the resulting tree, breadth-first keeping the best b states per level, or depth-first expanding the most promising state and pruning when a state's value drops below a threshold v_th. That template maps onto my problem, but the toy domains where it works hide assumptions that do not survive contact with real APIs.

Take the state evaluator first. In make-24, the state is a tiny arithmetic expression, so a scalar value like 1 to 10, or a label like "sure" or "impossible," has a chance of being calibrated. Then a fixed v_th can prune a bad subtree. A tool trajectory is different. The state is a half-finished multi-API interaction with real, messy observations. Ask the model "rate this partial trajectory 1 to 10" and the absolute number is noisy, instruction-dependent, and badly calibrated. If I prune on a fixed threshold, I will throw away good branches whose self-score happened to land low and keep bad ones that scored high. What survives this noisy-judge problem better is relative judgment. The model does not need to decide what an absolute 7 means; it only needs to decide whether continuation A or continuation B is more helpful for solving this same instruction from the same prefix. So I should compare candidates pairwise rather than assign absolute values.

Now the generator and the branching factor. In a small puzzle, the next steps can be essentially enumerated. Here the next action space is the Cartesian product of every free-form thought the model could write, every available API, and every possible parameter string. I cannot enumerate children, so I need to sample a finite number of them: tree_beam_size candidate continuations from the current state. But naive repeated sampling has a trap. If I ask the model k times at the same state, I often get k near-identical actions, because the model has a strong mode. k copies of the same action is not exploration. When a state already has children, I need to show the previous actions generated from that state and explicitly ask for a different one. That temporary diversity prompt belongs only to the generation call that uses it; after the call, I mark the prompt invalid so descendants do not inherit an instruction that was meant only to diversify siblings.

Then breadth-first versus depth-first. Breadth-first search would spend calls expanding the whole frontier at each depth before going deeper, but for solution-path annotation the run is useful as soon as it finds one valid answer path. I do not need a globally best path; I need a passing path within a query budget. Depth-first has the right bias: choose the most promising child, drive down to a leaf, and pay for siblings only when the chosen branch fails. That reaches a terminal path quickly on easy instructions and still has a way to explore alternatives on hard ones.

The comparison step is where most of the extra model calls go, so I want to be careful about how many it actually costs. At a node I first generate the whole beam and store the resulting leaf states. I do not recurse immediately in the ranked branch; I wait until I have the sibling set. For every unordered pair of candidates i, j, I find their common prefix and the two divergent suffixes. I ask the model, in one word, whether A or B is more helpful for solving the task. A single prompt can have order bias, so I ask twice: first with candidate i shown as A and candidate j as B, then with the order flipped. If both orderings choose the same underlying candidate, that candidate wins the pair. If the two calls disagree, neither candidate has a clean win and the pair is split. A round-robin over all unordered pairs gives each candidate a score: +1 for a pairwise win, +0.5 for a tie.

Let me count the calls for a small beam before I commit to this, because if it blows up I will burn the whole budget on judging instead of acting. For n = 3 siblings the unordered pairs are (0,1), (0,2), (1,2) -- three pairs -- and at two calls each that is six comparison calls. For n = 4 the pairs are six and the calls are twelve; for n = 5, ten pairs, twenty calls. So in general the count is 2 * (n choose 2) = n(n - 1): for n = 2,3,4,5 that is 2, 6, 12, 20, which matches the per-n counts I just walked. The ranked branch spends n(n - 1) extra comparison calls at one expanded state, which is O(n^2). That is not sorting by a cheap key; it is quadratic LLM-based preference aggregation, and it tells me the beam has to stay small -- a beam of 5 already costs 20 judging calls on top of 5 generation calls at a single node.

Let me also check that the round-robin score actually produces the ordering I want, on a concrete case. Suppose three candidates where the model genuinely prefers C0 over both others but is indifferent between C1 and C2. Pair (0,1): both orders pick 0, so +1 to C0. Pair (0,2): both orders pick 0, so +1 to C0. Pair (1,2): the two orders disagree (the indifference shows up as order bias), so it splits, +0.5 to C1 and +0.5 to C2. Final scores are C0 = 2.0, C1 = 0.5, C2 = 0.5. Sorting descending puts C0 first, then C1, C2. So the clear winner is expanded first and the two indistinguishable candidates fall behind it -- which is exactly the behavior I wanted from relative judgment, without ever asking the model for an absolute number. After those scores are accumulated, I sort the sibling candidates by score descending and expand the best judged child first.

Now I need the backtracking machinery. A plain recursive DFS return value like true or false is too weak, because after a leaf fails I may want to jump more than one level up before trying another branch. So each recursive call returns an integer: how many levels the caller should still climb. The base cases fix the arithmetic. If the current node is terminal, it is a final-answer leaf: record it, set the search status to success, and return final_answer_back_length = 2. If the current node is at the depth limit or already pruned, mark it pruned. If its observation_code is 4, that is the "give up" finish, so record it in give_up_node and return prune_back_length = 2. A non-give-up dead end returns 1. If budget is exhausted during child generation, return 100000 so the stack unwinds all the way out. If enough final-answer nodes have been collected, return 10000 for the same hard-unwind purpose.

I do not trust this arithmetic until I trace it on a concrete tree, because off-by-one errors in a return-value protocol are exactly the kind of thing that silently breaks backtracking. Take a root with two children A and B; A itself has two children A1 (an ordinary dead end) and A2 (a final-answer leaf). Walk it. DFS(root) recurses into A; DFS(A) recurses into A1, which returns 1; A sees result == 1, so it does not finish, it moves to its next sibling A2. DFS(A2) is a final-answer leaf and returns 2. Back in A, result == 2 > 1, so A calls make_finish and returns 2 - 1 = 1 to root. Root sees result == 1: the climb is consumed, root does not keep unwinding, and it continues to its next sibling B. So from the final-answer leaf A2 the search climbed exactly two edges (A2 -> A -> root) and then resumed at root's next child. That is precisely the intent of the constant 2: do not retry right next to the completed leaf -- A is done -- but let the grandparent root diversify by trying B. If A2 had instead been a give-up leaf returning 2, the same two-edge climb happens: A finishes, root resumes at B. And a plain dead end returning 1 is absorbed by its parent immediately, which tries the next sibling right there with no climb at all. The arithmetic holds.

The budget sentinel needs its own look, because I wrote "large enough to leave the tree" but that is not literally true. If a node returns 100000, its parent sees result > 1, finishes, and returns 100000 - 1 = 99999; the grandparent returns 99998, and so on -- the sentinel decrements by one at every ancestor. So it does not escape unconditionally; it escapes only because a real interaction tree is far shallower than 100000 levels (it is capped by single_chain_max_step, which is small). After 100000 decrements it would stop unwinding, but no tool trajectory ever gets near that depth, so in practice it does unwind all the way out. The constants are crude but deliberate: 2 means "do not retry right next to the failed or completed leaf; diversify one level higher," while 10000 and 100000 are sentinel lengths chosen large enough that, after the one-per-level decrement, they still exceed any reachable tree depth and so leave the recursion.

I also have to account for budget exactly where the loop spends it. The budget check happens after the model call is made and query_count is incremented. That means a call that brings query_count to max_query_count triggers the 100000 unwind before the new model message is parsed into child nodes. The comparison calls also count against the same query budget: after the round-robin returns its scores, I add rank_query_count to query_count and add the comparison tokens to total_tokens. There is no separate post-ranking budget gate; if the comparisons push query_count over the cap, the next recursive generation call is where the same guard will return 100000. The policy cannot pretend ranking is free.

Putting this together, the ranked search fills the harness's empty control slot like this. I can factor the existing message-parsing and tool-execution block as `_expand_once`: it creates Thought, Action, and Action Input nodes from one assistant message, sets `observation_code`, sets `pruned` on status 4, marks status 3 as a final answer, appends the assistant and function messages, and returns the new leaf state.

```python
def search(self, root_node, single_chain_max_step, tree_beam_size,
           max_query_count, answer):
    return self.DFS(root_node, single_chain_max_step, tree_beam_size,
                    max_query_count, answer, with_filter=True)


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
                          max_query_count, answer, with_filter)
        if len(self.terminal_node) >= answer:
            return 10000
        if result > 1:
            now_node.make_finish(2)
            return result - 1

    return 1
```

Now the pieces line up. The single ReAct chain fails because a bad action cannot be retracted and because one trajectory explores only one direction through an enormous action space. Retraction turns the interaction history into a decision tree: each state can have multiple action continuations, and the good prefix is kept while only the bad suffix is replaced. Since one passing path is enough, depth-first search is the right budget shape. Since absolute values on messy tool trajectories are noisy, sibling choice is made by symmetric pairwise LLM judgments over shared-prefix continuations. Since child generation is open-ended, I sample a fixed beam and use a temporary diversity prompt to avoid duplicate siblings. Since local failure should sometimes escape its immediate neighborhood, each call returns a climb length, with 2 for final answers and give-up leaves, 1 for ordinary dead ends, 10000 for enough answers, and 100000 for query exhaustion. On easy instructions, where the highest-ranked first child already reaches a final answer, the traced arithmetic shows the search records that answer and unwinds out without ever expanding the lower-ranked siblings -- so the realized trajectory is a single chain, the same shape as ReAct, and the extra beam plus ranking machinery cost is paid only at states where the first choice does not pan out. On hard instructions it keeps the ranked depth-first backtracking machinery that lets the agent abandon a wrong branch and try another without throwing away the whole prefix.
