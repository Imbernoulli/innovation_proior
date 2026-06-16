Let me look at exactly where chain-of-thought breaks, because the break is so clean it points straight at the fix. I take the last-letter task: given a list of words, output the concatenation of their last letters. "thinking, machine" → "ge". I write CoT exemplars on 2-word lists, with a nice rationale — "last letter of thinking is g, last letter of machine is e, concatenate to ge" — and the model nails 2-word test lists. Then I test on length-12 lists. It falls apart, and the longer the list, the worse it gets, the accuracy sliding down with each added word. Same story on the compositional benchmark where I map commands to action sequences: exemplars on short commands, and the model can't stretch to the long ones in the length split. Same on math: it handles problems with as many steps as the exemplars and degrades as the gold solution needs more steps.

So the failure has a name — the test problem is *harder* than the demonstrations, and CoT can't bridge that. It can imitate the kind of reasoning it was shown; it cannot extend it to a deeper or longer instance. Why not? Look at what a CoT exemplar actually is. It's a single rationale that solves the *entire* problem in one continuous shot. For a 2-word list that rationale is short and the model reproduces its shape. For a 12-word list the model would have to produce a much longer rationale of the same shape, in one pass, holding all twelve steps coherently — and that one-pass extrapolation is exactly what it can't do. The exemplar taught it "here is a complete solution of this size," and a size it hasn't seen is out of reach.

Now contrast with how a person handles the 12-word list. Nobody solves it in one mental shot. You take the first word, get its last letter, then the next word, append, then the next — you've turned one hard 12-step problem into a sequence of trivial 1-step problems, each building on the last. That's the move. The reason a person generalizes from easy to hard is that they *decompose* the hard instance into easy pieces and chain them. CoT skips this; it tries to leap the whole gap at once.

So the idea writes itself: don't ask the model to solve the hard problem in one rationale. First ask it to break the problem into a sequence of simpler subproblems, then solve those subproblems one at a time — and crucially, let the answer to each earlier subproblem feed into the next. Two stages. Decompose, then solve sequentially with carry-over.

Let me be careful about what makes this different from just "decompose," because decomposition isn't new. The usual versions split a question into *independent* subquestions, answer each on its own, and aggregate at the end — and they often need trained models to do the splitting and the aggregating. What I want is different on both counts. The subproblems here are *dependent and ordered*: later ones genuinely need the answers to earlier ones as building blocks (last-letter step k needs the concatenation from step k−1). And both stages are done by the *same frozen LM* via few-shot prompting — no training, no separate decomposition model. Decompose by prompting, solve by prompting.

Stage one, decomposition. I write a prompt whose exemplars demonstrate how to break a problem into an ordered list of subproblems, then I append the actual problem and let the model produce its decomposition. For the last-letter task this is almost embarrassingly simple — the decomposition of a list is its sequence of growing prefixes. The solve trace starts at the first prefix supported by the solving exemplars, the two-word base case, and then extends one word at a time: "think, machine", "think, machine, learning", "think, machine, learning, reasoning". One exemplar teaches the model to expand any list into the chain of sublists. The model reads it and decomposes an arbitrary-length list the same way.

Stage two, sequential solving — and this is where the carry-over has to be built into the prompt itself, so let me think about what the solving exemplars need to demonstrate. I want the model to learn: to answer the current subproblem, *use* the answer to the previous one. So the solving exemplars can't just be independent worked examples; they have to model the dependency. Look at the pair I'd use for last-letter. First exemplar: Q: "think, machine" → A: last letter of "think" is "k", last letter of "machine" is "e", concatenating gives "ke", so "think, machine" outputs "ke". Second exemplar: Q: "think, machine, learning" → A: *"think, machine" outputs "ke"*. The last letter of "learning" is "g". Concatenating "ke", "g" leads to "keg", so the answer is "keg". See what the second exemplar does — it doesn't recompute the last letters of "think" and "machine"; it *starts from the previous answer* "ke" and extends it. The two exemplars together are a base case and a recursive step. They teach the model the recursion: given the answer to the (k−1)-prefix, compute the k-prefix by appending one letter. That's the whole trick — the solving prompt demonstrates building a new answer from a prior answer.

So the procedure at solve time: decompose the input list into its chain of sublists, retaining or appending the original list as the final subproblem. Then walk the chain. For each sublist S, build a solving prompt = [the base-and-recursive exemplars] + [the previously solved sublist/answer pairs so far] + [S], send it to the model, take the answer, and append that (sublist, answer) pair to the running context before moving to the next S. The last answer, produced for the original list, is the final solution. The previously-solved pairs accumulating in the prompt are exactly the "building blocks" the recursive exemplar taught the model to reuse — at each step the model sees its own prior answer sitting right there and extends it, just like the exemplar showed. The hard 12-step problem never gets attempted as one leap; it's a sequence of easy appends.

Compare CoT on the same task to see precisely what changed. The CoT prompt uses the *same* two lists, but its rationale for "think, machine, learning" recomputes everything from scratch — last letter of think, of machine, of learning, concatenate all three — independent of the "think, machine" exemplar above it. CoT's exemplars are independent; least-to-most's are deliberately dependent, the second built on the first. That single design difference — does the later answer reuse the earlier one — is what buys the length generalization.

Now the compositional command→action task, because it shows the method handling a richer decomposition than just prefixes. Here a long command like "look opposite right thrice after walk" composes shorter commands with connectives. So decomposition isn't prefixing; it's parsing the command into its constituent shorter commands in dependency order. The decomposition exemplars demonstrate exactly that: Q: "look opposite right thrice after walk" → A: "look opposite right thrice" can be solved by "look opposite right", "look opposite right thrice"; "walk" can be solved by "walk"; so the prerequisite list is "look opposite right", "look opposite right thrice", "walk". The solve loop then appends the original command as the final mapping question, so the model first translates the pieces and then composes the full command from those translations. The mapping prompt — shared with the CoT baseline — teaches how to map each command to actions, and crucially how a command's output is *composed* from its parts' outputs: "the output of 'jump left' concatenates the output of 'turn left' and the output of 'jump'", and so on. One more practical wrinkle: rather than spell out "LOOK LOOK LOOK", I use a compact Python-ish notation, "LOOK" * 3, so the intermediate representations stay short enough to fit the model's context limit; a tiny postprocessing script expands these expressions at the end. The decomposition prompt and the mapping prompt are separate here (8 decomposition exemplars, 14 mapping exemplars chosen to cover the command semantics), whereas for last-letter the two pieces are smaller.

And for math word problems I notice the two stages don't always have to be separate decomposition and solving prompts. I can fold decomposition and subproblem solving into one generated response: an exemplar first says "Let's break this problem down: 1. <subquestion 1> 2. <subquestion 2>" and then answers them in sequence within the same response, each numbered answer using the earlier ones ("Anna has 2 more than Elsa's 5, so 7; together 5 + 7 = 12"). In the revised one-shot GSM8K setup, the initial prompt ends after "Let's break down this problem:", then the model's reply is appended along with "The answer is:" for one short follow-up request that extracts the final answer. Same least-to-most spirit — decompose, then solve sequentially with carry-over — without needing the full two-prompt loop used for last-letter and SCAN.

Why does this generalize to harder instances when CoT doesn't? Because the difficulty the model actually faces at any single step is bounded by the *subproblem*, not by the whole problem. Each subproblem is no harder than the exemplars — one letter to append, one short command to map, one arithmetic step — regardless of how long or deep the overall instance is. The instance's difficulty is absorbed by the *number* of steps, and the model only ever does one easy step at a time, with the prior answers handed to it. So a 12-word list is just as easy per step as a 2-word list; there are simply more steps. That's the easy-to-hard bridge: hold each step at demonstration-level difficulty and let the chain be as long as the instance demands. It also stays fully interpretable and needs no training — both stages are just the frozen LM following few-shot demonstrations. And it composes with the other prompting tricks: each subproblem's solution can itself be a chain of thought, and I can self-consistency-sample the answers; least-to-most is orthogonal to those.

The code is the two-stage loop made literal: decompose once, then solve the subproblems in order, threading each answer into the next prompt.

```python
def llm(prompt, stop=None):
    ...   # frozen LM completion

# Stage-1 exemplars: how to break a problem into an ordered list of subproblems.
# (last-letter: a list -> growing prefixes, starting at the two-word base case)
DECOMPOSITION_EXEMPLARS = '''Q: "think, machine, learning, reasoning"
A: "think, machine", "think, machine, learning", "think, machine, learning, reasoning"
'''

# Stage-2 exemplars: a base case + a recursive step that BUILDS ON the previous answer.
SOLUTION_EXEMPLARS = '''Q: "think, machine"
A: The last letter of "think" is "k". The last letter of "machine" is "e". \
Concatenating "k", "e" leads to "ke". So, "think, machine" outputs "ke".

Q: "think, machine, learning"
A: "think, machine" outputs "ke". The last letter of "learning" is "g". \
Concatenating "ke", "g" leads to "keg". So, "think, machine, learning" outputs "keg".
'''

def decompose(question):
    out = llm(DECOMPOSITION_EXEMPLARS + f'Q: {question}\nA:')
    return parse_subproblems(out)         # -> ordered list of sub-instances

def ensure_final_subproblem(subproblems, question):
    return subproblems if subproblems and subproblems[-1] == question else [*subproblems, question]

def solve(question):
    subproblems = ensure_final_subproblem(
        decompose(question), question
    )                                     # stage 1: ordered prerequisites + original problem
    history = ""                          # accumulated (subproblem, answer) pairs = the building blocks
    answer = None
    for S in subproblems:                 # stage 2: solve in order, each on top of prior answers
        prompt = SOLUTION_EXEMPLARS + history + f'Q: {S}\nA:'
        answer = llm(prompt).strip()
        history += f'Q: {S}\nA: {answer}\n\n'   # thread this answer into the next subproblem's prompt
    return answer                         # answer to the final subproblem = the solution
```

So the causal chain: chain-of-thought solves the whole problem in one rationale, so it can't extend to instances harder than its exemplars — accuracy slides as test difficulty exceeds demonstration difficulty. Mimic how people generalize easy-to-hard: decompose the hard instance into an ordered sequence of easy prerequisite subproblems, make the original problem the final subproblem, then solve the sequence one at a time with each answer building on the last. Do both stages with the same frozen LM by few-shot prompting — decomposition exemplars that peel the problem apart, and solving exemplars that demonstrate a base case plus a recursive step reusing the previous answer. Then each step the model faces is only as hard as a demonstration, and the instance's difficulty is carried entirely by the *number* of steps, which is what lets a frozen model solve problems far harder than anything in its prompt, with no training.
