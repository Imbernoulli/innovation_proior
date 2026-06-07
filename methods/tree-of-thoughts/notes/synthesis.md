# ToT synthesis

Pain: LLMs decode left-to-right, token by token. For problems where the first few tokens are pivotal (Game of 24: "4+..." already dooms it) or that need exploration/lookahead/backtracking, this "System 1" autoregression fails. CoT adds intermediate thoughts z1..zn but still a single linear chain, no branching, no evaluation of partial progress, no backtrack. CoT-SC ensembles k chains and majority-votes, but no local exploration within a chain and "most frequent" only works for limited output spaces.

Lineage:
- IO prompting: y ~ p_theta(y | prompt_IO(x)).
- CoT (Wei 2022): z_i ~ p_theta^CoT(z_i | x, z_{1..i-1}); y ~ p(y | x, z_{1..n}); decomposition of thoughts left ambiguous (sampled as one continuous sequence).
- CoT-SC (Wang 2022): k iid chains, argmax_y #{i: y^(i)=y}.
- Newell/Simon (1950s): problem solving = search through a combinatorial problem space represented as a tree; nodes = partial solutions, branches = operators; heuristics guide which branch.
- Classical search heuristics: programmed (DeepBlue) or learned (AlphaGo). ToT's novelty: LM self-evaluation as the heuristic.

Method (4 questions to instantiate ToT):
- State s = [x, z_{1..i}] (input + thoughts so far). Tree node.
1. Thought decomposition: a thought should be small enough to sample diverse/promising candidates, big enough to evaluate. Game of 24: one equation step (3 steps). Creative writing: a paragraph plan (1 step). Crosswords: a word (5-10 steps).
2. Thought generator G(p_theta, s, k):
   (a) Sample iid from CoT prompt: z^(j) ~ p^CoT(z_{i+1} | s). Better when thought space rich (paragraphs); iid gives diversity.
   (b) Propose sequentially via propose prompt: [z1..zk] ~ p^propose(... | s). Better when space constrained (a word/line), avoids duplication.
3. State evaluator V(p_theta, S):
   (a) Value each state independently: v ~ p^value(v|s), scalar 1-10 or class sure/likely/impossible mapped to value. Uses lookahead (5 5 14 -> 5+5+14=24) + commonsense (too big/small).
   (b) Vote across states: s* ~ p^vote(s*|S); V(s)=1[s=s*]. For coherency-type tasks. Step-wise self-consistency.
   Can sample multiple times to aggregate.
4. Search:
   BFS (Alg 1): S0={x}; for t=1..T: S'_t = {[s,z] : s in S_{t-1}, z in G(p,s,k)}; V_t=V(p,S'_t); S_t = argmax over size-b subset of S'_t by sum value (keep best b). Return G(p, argmax_{s in S_T} V_T(s), 1). Game24, creative writing (T<=3, b<=5).
   DFS (Alg 2): explore most promising first until t>T (record output) or V <= v_th (prune subtree), then backtrack to parent. Crosswords.

Special cases: IO, CoT, CoT-SC, self-refine = trees of limited depth/breadth. Benefits: generality, modularity, adaptability, convenience (no training).

Grounded code (princeton-nlp/tree-of-thought-llm):
- models.py: gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop) -> ChatCompletion, returns list of n contents.
- bfs.py solve(): ys=[''] candidates; for step in range(task.steps): generate (sample via get_samples standard/cot, or propose via get_proposals), evaluate (vote get_votes / value get_values), select (sample by prob or greedy top n_select_sample). 
  - get_proposals: gpt(propose_prompt, n=1)[0].split('\n'), each appended to y.
  - get_value caches per value_prompt; value_outputs_unwrap maps {'impossible':0.001,'likely':1,'sure':20} counted across samples.
- game24 task: steps=4 (3 equation steps + final answer line), stops=['\n']*4. propose_prompt_wrap: if current_numbers=='24' use cot prompt for final answer, else propose for current numbers. value: if last line lacks 'left:' it's last-step -> value_last_step_prompt. test_output uses sympy.simplify == 24 and number multiset match.
- prompts/game24.py: propose_prompt is 1-shot ("2 8 8 14 -> Possible next steps"), value_prompt few-shot with sure/likely/impossible, get_current_numbers parses "(left: a b c)".

Design whys:
- thought granularity tradeoff (diversity vs evaluability).
- sample vs propose: rich space -> iid sample for diversity; constrained space -> propose in one context to avoid dup.
- value vs vote: directly-scorable progress -> value; hard-to-score (coherency) -> vote (relative comparison).
- BFS vs DFS: shallow with prunable frontier -> BFS keep best b; deep, want backtrack -> DFS with threshold prune.
- value_map sure=20 >> likely=1 >> impossible=0.001: heavily promote sure states, near-eliminate impossible, aggregate over 3 samples => robust ranking. The asymmetry (20 vs 1) makes a single "sure" vote dominate.
- aggregate (n=3 value samples): valuations imperfect, only need approximately helpful; averaging reduces noise.
