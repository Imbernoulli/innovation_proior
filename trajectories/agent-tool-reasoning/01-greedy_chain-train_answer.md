The policy is the whole object I am designing, but before I can design any search over it I need the floor: the dumbest control flow that closes the loop of getting a frozen LLM to fulfill a multi-tool instruction at all. That floor is what every later rung has to beat, and its failure mode is what tells me what to build next. The two tricks I already hold do not survive the move to real tools. Few-shot chain-of-thought makes the model reason, and that works for self-contained problems — but the instant a step needs a fact the model doesn't hold, it writes down something plausible, and because every later token conditions on the text already written, the invented fact becomes a premise the rest of the chain reasons from. For tool use this is sharper than for arithmetic: the model will invent an *API that does not exist*, or invent what a real API returned, and plan against the hallucination. Sampling many chains and voting (self-consistency, Wang et al. 2022) doesn't help, because every chain draws from the same internal well; voting cures path noise, not missing grounding, and on a tool task the grounding *is* the task. The mirror trick — make the model act, turning each observation into text and asking for the next action — gives real contact with the world but leaves no place to think between actions, so on a multi-step instruction the model loses the goal, re-calls a tool it already tried, and can't synthesize a final answer from the responses it gathered.

The two failures are mirror images: thought without grounding versus grounding without thought, and the resolution is to interleave them in one trajectory. I propose the **greedy chain** — a single-chain ReAct policy, the CoT@1 floor — built on exactly that interleaving. Frame it cleanly: at round $t$ the agent holds context $c_t = (o_1, a_1, \dots, o_t)$ and emits an action $a_t$ under a policy $\pi(a_t \mid c_t)$. Enlarge the action space to $\hat{A} = A \cup L$, where $L$ is language. An ordinary action in $A$ executes a tool and returns a real observation; a *thought* in $L$ is defined to have no environment effect and to return no observation — it just rewrites the context so the next real action is better reasoned, $c_{t+1} = (c_t, \text{thought})$. Interleaving thought and action is the right per-round shape: think, call an API, read the real response, think again. The thoughts decompose the goal, react to surprises, and synthesize; the actions keep the reasoning grounded so it stops hallucinating. The frozen large model with a few interleaved exemplars is what makes the unbounded $L$ tractable — the strong language prior already knows what a sensible next thought looks like, so there is no training, only demonstration.

What the policy I write actually controls is *not* this round — the harness owns the whole ReAct round inside `_step`. One call to `_step(node)` performs one LLM call, parses the assistant message into a Thought node and, for a tool call, an Action node and an Action Input node, runs the call, sets `observation_code` (0 ok, 1 hallucinated API name, 3 final answer, 4 give up), flips `is_terminal`/`pruned`, increments `query_count`, and returns the new leaf. So `search()` is purely the *control flow* over repeated `_step` calls, and the simplest control flow is a single forward chain: start at the root, call `_step` on the current node, move to the leaf it returns, repeat — think, act, observe — until something stops me. That is the greedy chain, and it is the right floor precisely because it has no notion of taking anything back.

The body of the method is the five stopping conditions, each load-bearing. Before every step I check the budget: if `query_count >= max_query_count`, break — and since `_step` itself returns `[]` once the budget is hit, I treat that empty return as a stop too. Then I check whether I already have enough answers: if `len(terminal_node) >= answer_count`, break (here `answer_count` is 1, so the first valid answer ends the search). I take the step; if it returns no new leaf, break. Then I inspect the leaf the world just produced. If `is_terminal` — the model emitted "Finish with Final Answer" and the server confirmed it with `observation_code` 3 — I set `status = 1`, record the node in `terminal_node`, and stop: I have a solution. If `pruned` — a dead end — I stop, and if `observation_code == 4` it was an explicit give-up, so I record it in `give_up_node` (a give-up is still an honest, recordable outcome: "the tools can't do this"). Finally a depth guard: if the node reached `single_chain_max_step` raw nodes deep, prune it and stop. That is the entire policy — one path, taken greedily, no branching, no ranking, no backtracking. Critically it never calls `_add_diversity_prompt` and never calls `_rank_nodes`, so it spends *zero* queries on anything but advancing the chain — the cheapest possible run.

The structure that makes it cheap is exactly the structure that makes it weak: the chain cannot take back an action. Once $a_t$ is committed the agent moves forward, and after seeing $r_t$ it cannot return to the state before $a_t$ and pick differently. Two failures are baked in. First, error propagation: one wrong call — malformed parameters, or a hallucinated API name — now sits in the context, every later thought conditions on it, and the model retries the same broken call slightly reworded, stuck in a faulty loop. Second, under-exploration: a single chain commits to exactly one route through an effectively unbounded action space (free-form thought $\times$ API name $\times$ arguments), and if the first tool it reaches for is wrong, it rides that tool all the way down and never tries the one that would have worked. So the greedy chain should post by a wide margin the *lowest* `avg_queries` on the ladder — and that low number is the symptom, not a virtue: the agent quits early because it has no machinery to keep trying. It should be the weakest rung on `pass_rate`, the gap widening on harder instructions and weaker backbones where one linear pass most often commits to a wrong tool and never recovers, with a non-trivial `give_up_rate` because an early dead end has nowhere to go but a give-up. The diagnosis is already pointed at the next rung: this is structural, not a tuning problem — the agent needs a way to retract a decision and try another, which means turning the single chain into a tree of partial trajectories I can back up in.

```python
    def search(self, root_node):
        """Greedy chain: follow one path, no backtracking."""
        now_node = root_node
        for step in range(self.single_chain_max_step):
            if self.query_count >= self.max_query_count:
                break
            if len(self.terminal_node) >= self.answer_count:
                break

            new_leaves = self._step(now_node)
            if not new_leaves:
                break

            now_node = new_leaves[-1]

            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                break

            if now_node.pruned:
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                break

            if now_node.get_depth() >= self.single_chain_max_step:
                now_node.pruned = True
                break
```
