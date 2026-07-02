# 题 175 — base均分 20.7 vs 直接SFT均分 0.6(掉点 20.1)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 99.0,completion_tokens=28989)

````
The user wants a C++ solution for the 3-SAT problem.
Wait, the problem statement says "assign truth values to variables to satisfy as many clauses as possible." This is the Max-3SAT problem.
Max-3SAT is NP-hard.
Given constraints: $n \le 3000$, $m \le 2000000$.
Since this is a competitive programming problem, and Max-3SAT is NP-hard, an exact solution won't work within reasonable time limits (likely around 1-2 seconds) unless specific algorithms or heuristics are used. However, looking at the scoring section: "Any assignment is accepted and scored". This implies this is likely an online judge problem where the goal is often just to pass basic tests or find a good heuristic solution. But wait, if it's truly Max-3SAT, finding the optimal solution is hard. The notes say "Any assignment is accepted", which suggests we don't necessarily need the maximum number of satisfied clauses, but typically competitive programming problems with this title imply finding *a* satisfying assignment if one exists (k-SAT), or maximizing satisfaction.
Let's re-read carefully: "Your task is to assign truth values to variables to satisfy as many clauses as possible."
Usually, for Max-3SAT, simple randomized local search or greedy approaches get high scores on random inputs but might fail on worst cases.
However, given the constraint $N \le 3000$, $M \le 2 \times 10^6$, and the fact that it's called "3-SAT" (usually referring to satisfiability), maybe it's actually asking for standard SAT (satisfy all)? No, "satisfy as many clauses as possible". That's Max-Cut like or Max-SAT.

Actually, there is a very famous result related to Max-3SAT approximation. The random assignment satisfies expected $7/8$ of clauses. There are local search algorithms (like local optima) that guarantee better or similar.
But wait, in some contexts (like older contests or specific platforms), "3-SAT" might imply checking satisfiability, but the problem explicitly asks to maximize satisfied clauses.
If $N$ is up to 3000, we cannot do exponential complexity.
Greedy strategies usually work well enough to get a decent score, but not necessarily optimal. Since the problem accepts any assignment (just scored by ratio), I should implement a heuristic that performs reasonably well.
Common strategy: Randomized Local Search or Simulated Annealing. Given the tight constraints ($M=2 \times 10^6$ means parsing input takes time, evaluating assignment takes $O(M)$).
To update evaluation efficiently when flipping a variable, we need to track contribution of each variable.

Algorithm Idea:
1. Parse input. Store clauses.
2. Initialize assignments randomly (0s and 1s).
3. Evaluate current number of satisfied clauses $S$.
4. Try to improve $S$ by flipping variables.
   - Calculate gain from flipping each variable $i$: $gain_i = (\text{new satisfied count}) - (\text{current satisfied count})$.
   - Pick variable with positive gain > 0 (or largest gain) and flip.
   - Repeat until no variable can increase the total satisfied count (local optimum).
5. To handle local maxima where no single flip helps, use a perturbation step (random reset) or Simulated Annealing. Or simply try multiple times from different seeds.

Wait, $N=3000$ allows $O(N \cdot M)$ per step? No, too slow. We need faster updates.
Total clauses $M$ is large ($2 \times 10^6$).
Evaluating all clauses takes $O(M)$.
We can maintain counts. Each clause has literals $l_1, l_2, l_3$.
We can precompute for each clause whether it's currently satisfied.
Also for each variable $x$, how many clauses involving $x$ are satisfied vs unsatisfied.
When flipping $x$:
- Satisfied clauses containing $\neg x$ become unsatisfied (if only $\neg x$ was deciding them).
- Unsatisfied clauses containing $x$ become satisfied (if only $x$ was deciding them).
This requires knowing for each clause if it is satisfied solely by that literal. That gets complicated to maintain in $O(1)$ without overhead.
Actually, simpler approach for Max-SAT with large $M$:
Just iterate through variables, calculate potential gain in $O(\text{degree})$. Sum over degrees is $O(M)$.
With $N=3000$ and $M=2 \times 10^6$, doing an $O(M)$ iteration per round is too much (total time limit usually 1-2s).
However, note that typical solutions for this problem (it looks like a classic problem from a platform like AtCoder or similar, though constraints are slightly higher than typical CP easy ones) often rely on a specific heuristic.

Wait, check the problem source style. If it's a general CP platform problem asking for Max 3-SAT, often the test cases are not adversarial against the best heuristics. A greedy initialization + local search often passes.
Randomized Greedy (pick first unsatisfied clause, pick one variable setting):
For each unsatisfied clause (initially all potentially unsatisfied):
  Assign variables to satisfy it?
  Better: Iterate variables. For each variable, determine if flipping makes more clauses satisfied. Flip if yes.

Let's refine the greedy/local search algorithm.
Data structures:
- `vector<vector<int>> clauses` size M, each with 3 ints.
- `bool vars[N+1]` current assignment.
- `cnt[M]`: true if clause $i$ is currently satisfied.
- Optimization: To avoid full scan $O(M)$ every time, maintain for each variable the impact of its value.
  `var_impact[v]` = number of clauses where assigning v makes the clause satisfied (assuming other literals are false).
  Actually, let's track:
  For each variable $v$ (mapped to absolute index 1..N), we want to know:
  If $vars[v]$ is True, how many clauses are satisfied purely because of $v$ being True?
  If $vars[v]$ is False, how many clauses are satisfied purely because of $\neg v$ being True?
  More precisely, for a clause $(L_1 \lor L_2 \lor L_3)$:
  It contributes to `impact[L_1_val]`, `impact[L_2_val]`, `impact[L_3_val]`.
  Wait, we need to distinguish "is satisfied exactly because of literal L".
  Let's define `count[val]` as the number of clauses satisfied if variable val takes value `val`.
  When we flip variable $k$, we change its assignment.
  New satisfaction count for clauses involving $k$ changes based on whether they were satisfied by $k$ before or not.
  
  Efficient Local Search:
  1. Assign random initial values.
  2. Compute current total satisfied $S$.
  3. Identify candidate flips. For each variable $i$:
     Count how many clauses are satisfied currently by variable $i$. Let this be $c_{sat}(i)$.
     Also count how many clauses involving $i$ would become satisfied if flipped. This depends on the other literals in those clauses.
     Specifically, a clause $(a, b, c)$ involves $i$ (say $a=i$).
     Currently, if $a=i$ evaluates to True, the clause is satisfied regardless of $b, c$. If it evaluates to False, we check $b, c$.
     This dependency tracking is tricky without extra data.
     
  Simplified approach (often sufficient for CP):
  Just run the process multiple times with random restarts. In each pass:
  Initialize assignments.
  Repeatedly find a variable $i$ such that flipping $i$ increases satisfied count.
  If no such variable exists (local optimum), break and record max found so far.
  Sometimes, even at local optimum, you can force a bad move to escape local maxima (Simulated Annealing or random walk), but standard "Local Search" often suffices.
  Wait, for $N=3000$, local search steps might be $O(N)$. Updating satisfied count takes $O(deg(v))$.
  Sum of degrees is $3M$. So one full optimization pass is $O(N + M)$ if we update incrementally.
  But calculating gain for all variables naively takes $\sum deg(i) = O(M)$.
  So we can afford one full scan per iteration. How many iterations?
  If we stop when no improvement, that's fast.
  Is it possible to get stuck quickly? Yes.
  Is it better to use Simulated Annealing? Maybe too complex.
  Usually, for "satisfy as many as possible" with no penalty for not satisfying all, just running several greedy resets is safe.
  
  Let's optimize the "Check Gain" part.
  We need `gain[i]` for each variable $i \in [1, N]$.
  `gain[i]` = (# clauses containing $i$ that become satisfied if $i$ flips from True -> False) - (# clauses containing $i$ that become unsatisfied).
  Let's store clauses.
  `vector<tuple<int, int, int>> clauses(m);`
  Map literals to variables. `-x` corresponds to variable `x` with requirement `val=0`. `x` corresponds to `val=1`.
  For a clause, we check if it's satisfied.
  To compute gain for flipping $x$:
  We look at all clauses where literal $x$ appears.
  If literal is $x$ (positive):
    - If clause is NOT satisfied (because $x$ is false or others false):
      If we set $x$ to True, does it satisfy the clause? Only if others are false OR if it becomes the sole reason.
      Wait, if clause is not satisfied, then ALL literals must be false.
      So if literal $x$ is present and clause is unsatisfied, setting $x$ to True definitely satisfies it.
    - If clause IS satisfied:
      Was it satisfied ONLY by $x$?
      If yes, flipping $x$ to False will make it unsatisfied (unless another literal compensates... wait).
      If it was satisfied, at least one literal is true.
      If $x$ was true, and it's the only true literal, flipping $x$ to False makes it unsatisfied.
      If there was another true literal, flipping $x$ keeps it satisfied.
      
  To know "Is $x$ the only satisfied literal?", we need to know status of others.
  This is expensive.
  However, note that most test cases are weak or randomized.
  A very common strategy for Max 3SAT in contests (like Codeforces or similar) with these constraints:
  Run a greedy solver:
  While improved:
    Scan all variables, calculate immediate benefit of flipping.
    Benefit of flipping $x$:
      Current contributions:
        Clauses where $x$ is the ONLY truth source -> will become 0.
        Clauses where $x$ contributes but others also contribute -> stay 1.
        Clauses where $x$ doesn't contribute (false) and clause is unsatisfied -> if $x$ is true, it might fix it (if others false).
        
      Actually, counting "Clauses where $x$ is unique truth source" is still needed.
      
  Alternative: Simpler Heuristic.
  Start with all False.
  Iterate clauses. If a clause is unsatisfied, flip one of its literals' variables.
  Example: Clause (-1, 2, -3). Variables involved: 1, 2, 3.
  If all are False (so 1->True for -1, 2->False, 3->True for -3).
  Check if clause satisfied.
  If not, pick one variable at random or first available, flip it.
  Repeat.
  Wait, this is basically a greedy construction. With 3 clauses, picking one variable satisfies at most 1 clause? No, variables appear in multiple clauses.
  
  Standard Max-Sat Strategy:
  Initialize assignments to 0 or 1 with probability 0.5.
  Then perform Local Search:
    Repeat k times:
      Calculate `score` = # satisfied clauses.
      Find variable $i$ with max `delta_score` upon flip.
      If max `delta_score` <= 0, stop (local optimum).
      Else flip $i$.
      Update global max.
    
  To implement Local Search efficiently:
  Maintain `clause_satisfied[i]` array for all $m$ clauses.
  For each variable $i$, maintain:
    `clauses_with_x_true`: list of clauses where literal is positive $i$ and clause is satisfied because of $i$. (Not just satisfied, but because of $i$).
    This requires tracking if $i$ is the *only* satisfied literal.
    This is hard.
  
  However, since $M$ is up to $2 \cdot 10^6$, we can allocate memory.
  Can we just do $O(N \cdot M)$ once?
  No, time limit.
  
  Let's look for the specific problem. This looks like "Max 3-SAT" from a platform.
  Often, randomized initialization followed by a few rounds of local search works extremely well due to averaging effect.
  Wait, I recall a specific trick for Max-SAT where you maintain the state of each clause.
  State of clause: UNSAT, SAT.
  If UNSAT, count how many variables in it can be flipped to fix it (i.e. set literal to True).
  For each UNSAT clause, any literal in it, if assigned True, satisfies the clause.
  Initially all UNSAT.
  Select an UNSAT clause. Pick a literal. Set corresponding var.
  Check effect.
  
  Better:
  Since we want to maximize, and $N$ is small (3000) compared to $M$ ($2 \cdot 10^6$), the density is high.
  We can iterate through variables.
  Algorithm:
  1. `vector<bool> x(n + 1)`
  2. Loop $T$ times (e.g., 10):
     Shuffle `x` or randomize.
     Current satisfied count `S = 0`.
     Calculate `S`.
     Do while improvement possible:
       Find variable $i$ with max gain.
       Gain calculation:
         Need to count:
           - $C_{bad}$: number of clauses where literal $x_i$ is present (say positively), clause is currently unsatisfied, setting $x_i$ to True will satisfy it.
             Condition: All other literals in clause are currently False.
           - $C_{good}$: number of clauses where literal $x_i$ is present, clause is currently satisfied, BUT removing $x_i$ (flipping) would make it unsatisfied.
             Condition: All other literals in clause are currently False.
       
       Calculating $C_{bad}$ and $C_{good}$ for a specific variable involves iterating all its occurrences.
       Sum of occurrences = $3M$.
       If we do this for all variables in one pass, cost is $O(M)$.
       Total cost per T: $O(K \cdot M)$. If $K$ is small (e.g., 5-10), it fits.
       $2 \cdot 10^6 \times 10 \approx 2 \cdot 10^7$ ops. Very fast.
       So we can afford checking every variable's potential gain if we optimize the gain calc slightly or just loop through clauses.
       Wait, looping through clauses for each variable is $O(N \cdot M)$. Too slow ($3000 \times 2 \cdot 10^6 \approx 6 \cdot 10^9$).
       
       BUT sum of occurrences is $3M$. So iterating over "variable-clause" relations is $O(M)$.
       We can iterate over the variables present in each clause.
       Construct adjacency: `vector<vector<int>> adj(n+1)`. Store indices of clauses involving var $i$.
       For each clause $j$ in `adj[i]`:
         Check if clause $j$ is satisfied by $i$.
         Check if flipping $i$ affects satisfaction of $j$.
         Cost: For each occurrence, we check clause. Checking clause takes $O(1)$ (read other 2 literals, check their current values).
         So gain calc for variable $i$ takes $O(\text{deg}(i))$.
         Sum over all $i$ takes $O(M)$.
         So we can fully evaluate gains for all variables in one sweep.
         
       So, the plan:
       1. Read input. Build adj list `vec_adj[N][N]` (actually vector of vectors) where `adj[v]` stores pair `{clause_id, literal_sign}`.
          Literal sign: 1 for pos, -1 for neg.
       2. Loop randomizations (e.g., 5-10 times).
          Inside:
            Assign random values to `x[1..N]`.
            Evaluate all clauses to get `satisfied_count`.
            Mark which clauses are satisfied.
            Repeat:
              Scan all vars $1..N$:
                Calc delta.
                Delta = (Num clauses in `adj[v]` where `x[v]==false` and clause UNSAT) - (Num clauses in `adj[v]` where `x[v]==true` and clause SAT and would be UNSAT if `x[v]` flipped).
                Actually simpler:
                For each var $v$, initialize `up_gain = 0`, `down_loss = 0`.
                Iterate $c \in adj[v]$:
                  Let literal $L$ be associated. Value $val = x[abs(L)]$ == (L > 0 ? 1 : 0).
                  If $L > 0$, value is $x[v]$. Else $!x[v]$.
                  If literal is true:
                    Check if clause is satisfied ONLY by this literal.
                    If yes, flipping $v$ loses satisfaction (loss=1).
                    Otherwise (other lit true), loss=0.
                  If literal is false:
                    If clause is unsatisfied, flipping $v$ to satisfy this literal makes clause satisfied.
                    Wait, if literal is false, we check: are other two literals false?
                    If both false, flipping makes clause satisfied. Gain=1.
                    
                Optimization: Pre-calculate `is_other_literal_false(c, i)`?
                Just check current values on the fly.
                
                Keep track of best variable to flip.
                If best_delta < 0, break.
                Else flip best var, update `satisfied_count`.
                Continue loop.
          
          Keep track of max score achieved across randomizations.
          Wait, this inner loop (finding best flip) might run multiple times.
          Does it terminate quickly?
          Usually local search on 3-SAT converges fast or oscillates near optimum.
          With 2 million clauses, updating status takes time.
          If we find NO positive gain, we stop.
          With random restarts, we avoid deep local minima.
          
          Wait, is $N=3000$ variables too many for repeated scans?
          Inner loop runs "while improvement". Each improvement flips a bit.
          Worst case we could visit same state? No, strictly increasing satisfied count?
          Actually, we should ensure strictly increasing. If multiple have same gain, pick any.
          Is it possible to have cycles? No, satisfied count strictly increases. Max $M$. So loop runs at most $M$ times.
          However, calculating gain takes $O(M)$ total per round.
          So total complexity $O(M \cdot \text{restarts} \cdot \text{improvements})$.
          Improvements can be large.
          We need to optimize.
          
          Is there a simpler way?
          The problem statement allows ANY valid assignment (scores based on ratio).
          Usually, the random assignment gives $7/8$ (0.875) on average.
          Max Sat solvers aim for ~0.9+.
          The problem likely expects a solution that beats random chance significantly.
          
          Another trick: 
          Instead of local search, use a randomized greedy algorithm similar to "Set Cover" logic? No.
          
          What about:
          Sort clauses by frequency of variables?
          Or simply:
          Perform several passes of:
          "For each variable $i$, if flipping improves score, do it."
          Do this until convergence.
          With $N=3000, M=2\cdot 10^6$.
          One pass is $O(M)$ to compute gains (by iterating edges) + $O(N)$ to pick max.
          Wait, computing gain for all vars involves iterating all clause appearances.
          So yes, $O(M)$.
          If we repeat this 5 times, total $5 \times 2 \cdot 10^6 \approx 10^7$ operations. Fits easily in 1 sec.
          Wait, do we repeat until convergence?
          If we converge to local optimum, next pass might not improve immediately but needs restart or perturbation.
          Restarting involves shuffling all variables ($O(N)$) and recalculating.
          
          Revised Plan:
          1. Read Input.
          2. Adjacency list `adj[v]` contains `{clause_idx, literal_type}`.
             Literal type: 1 for pos, 2 for neg. (0-based or 1-based).
          3. `satisfied[m]` boolean array.
          4. `max_solved = 0`.
          5. Loop `try_cnt` from 0 to 50:
               `current_solved = 0`
               Generate random assignments `x[n]`.
               Update `satisfied` array and `current_solved` in $O(M)$.
               
               // Optimization Step: Local Search
               // We need to find best flip repeatedly.
               // To do this efficiently:
               // We keep track of `diff[v]` which is current net benefit of flipping v.
               // Initially recalculate diff for all v ($O(M)$).
               // Pick best v. Flip.
               // Recomputing diff for all v after one flip is expensive ($O(M)$).
               // Incremental update:
               // Flipping v affects only clauses involving v.
               // For those clauses, update their satisfaction status and re-evaluate `diff` for neighbors of v.
               // Since degree of v is avg $3M/N \approx 600$ for max N=3000, M=2e6.
               // Actually max degree is $3M/3 = M$. Average is $2M/N \approx 2 \cdot 10^6 / 1.5 \cdot 10^3 \approx 600$.
               // So updating `diff` for affected clauses takes proportional to their degrees.
               // Total update cost per flip: $\sum_{c \ni v} \text{deg}(c) = \text{deg}(v)^2$ approx?
               // No, degree of clause is 3.
               // We need to update neighbors of variables appearing in clauses containing $v$.
               // This is getting complicated.
               
               // Given constraints and typical CP limits, maybe brute force $O(M)$ scan per iteration is intended?
               // If $M$ is small enough, yes. But $2 \cdot 10^6$ is somewhat large.
               // Let's reconsider the scoring.
               // "Any assignment is accepted".
               // Maybe just a single random assignment with greedy repair is enough?
               
               // Let's check typical 3-SAT heuristics again.
               // Most successful is:
               // 1. Random init.
               // 2. While exists var with gain > 0:
               //      Find var with max gain.
               //      Flip.
               //      Repeat.
               //    Cost: Finding max gain takes iterating all vars.
               //      Calculating gain for all vars takes iterating all occurrences ($O(M)$).
               //      Picking best and flipping takes $O(\text{deg}(best\_var))$.
               //      Wait, if we recalculate gain for ALL vars every time we flip, it's $O(M)$.
               //      Number of flips could be large (close to $M$).
               //      Total $O(M^2)$. $4 \cdot 10^{12}$ is TLE.
               
               // We need an incremental way or fewer steps.
               // Notice that usually the solution space is such that local optima are close.
               // We can try to simulate this efficiently.
               
               // Wait, we can calculate `potential` for every var ONCE at start.
               // If we pick best var $v$ with potential > 0, flip it.
               // Now the potentials of clauses changed?
               // Actually, if we only change $v$, the clauses not involving $v$ remain unchanged.
               // Their contribution to $diff[neighbor]$ remains same.
               // Only clauses involving $v$ change.
               // So for each clause $c$ containing $v$, the gain of its 3 variables changes.
               // Updating $diff[u]$ for $u \in \text{vars}(c)$ takes $O(1)$ check on $c$.
               // Total update cost per flip: $\sum_{c \ni v} 3 = 3 \cdot \text{deg}(v)$.
               // This is efficient. $\text{deg}(v)$ is usually small (avg 600).
               // So complexity per flip is $O(\text{deg}(v))$.
               // Total complexity $\sum \text{deg}(v)$ over sequence of flips $\approx O(M \cdot \text{avg\_deg}) = O(N \cdot M)$?
               // No. Max number of flips is $M$ (can't improve more than $M$ times).
               // Wait, strictly improving means satisfied count goes up. Max $M$ times.
               // So total time is $O(\sum_{steps} \text{deg}(v_{step}))$.
               // This is bounded by $O(M \cdot \max\_deg)$. Max deg is $3M/3$ (all clauses share same var).
               // Worst case $O(M^2)$.
               // However, with 3-SAT structure and random init, usually number of steps is much smaller?
               // Not guaranteed.
               
               // BUT, notice we can skip the "Find Best" step if it's costly.
               // Just pick random variable with gain > 0?
               // "Pick random positive gain neighbor"?
               // Any ascent leads to optimum eventually?
               // Random Walk (steepest descent) is fine.
               // Even better: Just iterate all variables. If flipping improves score, do it.
               // Repeat the whole pass over variables until no change.
               // This guarantees local optimality after pass.
               // Cost of one pass:
               // Iterate $v \in 1..N$.
               // Check if flipping $v$ improves score.
               // Cost to check is $O(\text{deg}(v))$.
               // Total pass cost: $O(M)$.
               // We repeat passes until no change.
               // Usually 1-2 passes are sufficient for high quality solutions?
               // Actually, local search requires repeated passes to escape.
               // But given "Any assignment", maybe just multiple random starts + 1-2 passes is enough.
               // With random starts, the average case works well.
               
               // So strategy:
               // Loop 50 times:
               //   Random init.
               //   Compute initial score (O(M)).
               //   Repeat Pass (iterate all vars, flip if gain>0):
               //     Flag if flipped anything.
               //     Update score incrementally.
               //     If no flip, break Pass.
               //   Record max score.
               // Time complexity: 50 * (M + K_passes * M). K_passes ~ small constant.
               // Total ops ~ 50 * 10 * 2e6 = 10^9. A bit tight for 1 second but possibly OK if operations are simple.
               // Wait, $10^9$ is too much for 1s.
               // We need to reduce passes or speed up.
               // Speed up: Use bitsets? $N, M$ integers.
               // Or simplify.
               
               // Wait, checking "gain > 0" for a variable:
               // Gain = (unsatisfied clauses where $v$ can help) - (satisfied clauses where $v$ is critical).
               // We can cache this.
               // Initial Calc: $O(M)$.
               // Update: Only for affected variables.
               
               // Let's stick to the incremental update idea but optimized.
               // Actually, simply running `std::swap` logic is wrong.
               
               // Let's try the simplest effective approach:
               // Random Assignment.
               // Compute score.
               // While (change_possible):
               //   Calculate gain for all vars (O(M)).
               //   Pick best.
               //   Flip.
               //   Update score (O(deg)).
               
               // Actually, finding max gain takes $O(M)$.
               // If we do this once and then restart, is it better?
               // If we find a local optimum, we stop.
               // Multiple restarts cover many local optima.
               // Is one pass of greedy (flip if positive) enough?
               // No, it's myopic. Steepest descent (pick max) is better.
               
               // Given strict 2s time limit (usually implied), and large M, we should optimize.
               // We can just run "Pass Mode": Iterate all variables, flip if it improves score.
               // Repeat Pass Mode until no change.
               // Then restart with new seed.
               // How many Pass Modes?
               // Usually 10-20 passes is enough to reach local optimum.
               // Wait, $10 \times 2 \cdot 10^6 = 2 \cdot 10^7$ ops.
               // 50 restarts $\to 10^9$. Too slow.
               // We need fewer restarts.
               // 10 restarts $\times$ 10 passes $\approx 10^8$ ops. This is safe.
               
               // Let's finalize algorithm:
               // 1. Adjacency list.
               // 2. Maximize over K restarts (e.g. K=20).
               //    In each restart:
               //      Random `x`.
               //      Eval `satisfied`.
               //      Loop:
               //         `found = false`
               //         Iterate `v` from 1 to `n`:
               //            `g = calc_gain(v)`
               //            If `g > 0`:
               //               `x[v] ^= 1`
               //               Update `satisfied` for clauses involving `v`.
               //               `found = true`
               //               Break (Steepest Ascent: Flip and restart scan to catch new opportunities)
               //               Wait, breaking early (First fit) vs Max finding.
               //               First fit is faster? No, we scan anyway.
               //               Scanning all to find max gain is slower?
               //               Actually, calculating gain is cheap ($O(\text{deg})$).
               //               But accumulating max requires visiting all.
               //               "Flip any that improves" is essentially coordinate ascent.
               //               Usually good enough.
               //         If not `found`, break loop.
               
               // Wait, calculating gain for all $v$ takes $\sum \text{deg}(v) = 3M$.
               // If we iterate sequentially, we can accumulate total gain?
               // No, we need individual gain.
               // So yes, scanning all vars costs $O(M)$ per pass.
               // If pass repeats, cost multiplies.
               
               // Is there a way to avoid full scan?
               // We only need to know if *any* variable has positive gain.
               // We can track `pos_gain_vars` count.
               // But checking all to find one takes time.
               // Random sampling? Check 1000 random vars. If none good, check remaining.
               // Given 2e6 clauses, sampling is risky.
               
               // Let's assume 1-2 passes are sufficient for "local optimum" on random graphs.
               // Actually, the problem allows ANY solution. Random is already 0.875.
               // A proper greedy pass pushes it to ~0.95+.
               // I will implement the Full Scan per Pass with Incremental Updates.
               // And run limited restarts.
               
               // Data structures:
               // `int sat_count[m]` -- boolean if satisfied.
               // `int vars_in_clause[m][2]` -- indices of variables.
               // `int vars_val[m]` -- value of literals (-1, 1).
               
               // Optimisation:
               // Use a `vector<pair<int, int>> clause_vars[m]` for `clause_index` to `var_index`.
               // And `adj[var_index]` stores `list of clause_indices`.
               // Iterating `adj` allows fast check for variable gain.
               // But we also need to check if clause is satisfied.
               // Accessing `clause_vars[c]` is needed.
               
               // Wait, to update `sat_count`, we just need to check the clause.
               // `check_clause(c)` takes O(1) (check 3 literals).
               
               // Implementation details:
               // `vector<int> clause_lits[m]`
               // `adj[v]` stores indices.
               // `current_sat[m]`
               
               // Complexity Check:
               // Init: $O(M)$.
               // Pass: Iterate $v \in 1..N$.
               //   For each $v$, iterate its clauses.
               //   Calc gain.
               //   If gain > 0, flip, update sat.
               // Cost: $\sum_{v} \text{deg}(v) = 3M$.
               // So one pass is $O(M)$.
               // Total complexity: $R \times P \times M$.
               // If $R=10, P=10$, $M=2 \cdot 10^6 \implies 2 \cdot 10^8$. Acceptable.
               
               // Code Structure:
               // Includes.
               // Main function.
               // Fast I/O.
               // Adjacency lists.
               // Loop restarts.
               // Inner loop passes.
               // Output.

One edge case: $N$ up to 3000.
$Literals$ are in range $[-N, N]$.
Map negative to index properly.
Abs(val) is var index. Sign is determined by `val > 0`.
Array size `3005`.

Wait, one detail in Max-SAT greedy.
Does "flip any positive gain" equal "steepest ascent"?
No. Steepest is faster convergence to deeper optima.
But finding max gain requires iterating all clauses.
If we pick the first positive gain, we might take a shallow path.
But iterating all vars to check gain is $O(M)$.
So either way we pay $O(M)$ per step (if we find many positive).
However, usually we don't flip *every* variable.
If we flip variable $v$, we stop scan and restart? Or continue?
Restarting ensures we find max gain.
Continuing continues coordinate ascent.
I'll go with: Iterate all vars. If $gain > 0$, flip immediately.
Repeat scan until no changes.
This is Coordinate Ascent.
It's robust and fast enough.

Wait, $m$ can be $0$. Handle that.
$m=0$: output 1s? Score 1.0.
Any output is fine. Just print n 1s.

Let's double check the constraints and time limit.
Problem statement doesn't give time limit explicitly, but "Execution time ... stated".
Usually 1-2s.
Memory limit: "also stated".
Assume 256MB.
Vector of vectors overhead: $M \times \text{vector}$.
$M$ vectors of pairs?
No. `vector<vector<int>> adj(n+1)` where each vector stores clause indices.
Sum of sizes of vectors is $3M$.
Overhead of empty vector is small? 24 bytes. $3000 \times 24 \approx 72KB$.
Dynamic allocation is fine.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

struct Clause {
    int u, v, w;
};

// Global variables to avoid stack overflow
vector<int> x; // assignments
vector<bool> clause_sat;
vector<vector<int>> adj; // adj[i] contains indices of clauses containing variable i

// Function to check if a clause is satisfied
// Literals are given as (idx, sign). 
// sign: 1 for x, -1 for !x
// Here we store literals as (var_idx, is_pos)
// But let's just store raw ints in Clause struct or separate arrays.
// The input format is integers a,b,c. 
// So a = 3 means x3, a = -3 means !x3.
// We can normalize these to (index, polarity) pairs.
// Let's store `clauses` as vector of {u, v, w}.
// Inside check, access x.

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<tuple<int, int, int>> clauses;
    clauses.reserve(m);
    
    // Adjacency list
    // We need to map back from clause index to vars for update.
    // Actually, for checking gain for variable v, we need to iterate clauses in adj[v].
    // To check gain, we need to know current satisfaction of clause and other literals.
    // Storing clauses in a flat vector is efficient.
    // Storing adj as vector<vector<int>> is efficient.
    
    vector<vector<int>> var_clauses(n + 1);
    vector<vector<int>> clause_vars(m); 

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        int va = abs(a), vb = abs(b), vc = abs(c);
        int pa = (a > 0), pb = (b > 0), pc = (c > 0);
        clauses.push_back({va, pb, pa}); // Wait, storing signs?
        // Let's store literals as (var_idx, sign)
        clauses.push_back({0,0,0}); // dummy placeholder to reallocate? No, reserve.
        // Let's restructure storage.
        // We need to read all lines first.
    }
    
    // Clear previous attempt
    // Re-doing reading logic cleanly
    
    vector<vector<int>> literals; // flattened vector of 3*m integers?
    // Let's use a vector of structs or parallel vectors for literals.
    // Actually, `clauses` vector of {int, int, int} is better.
    // `clauses[i]` = {literal_a, literal_b, literal_c}. 
    // Where literal is encoded: (var << 1) | sign? No, var <= 3000. 
    // Let's encode: a = (x > 0 ? 1 : -1) ? No.
    // Let's encode var index 1..3000.
    // Encode literal as: `((unsigned)x << 32) | (sign ? 1 : 0)`?
    // Or simpler: just store `int` and decode `(abs(lit), lit > 0)`.
    // But we need fast decoding.
    // Let's use:
    // int encode(int x) { return (x > 0 ? 1 : -1) << 32 | abs(x); } 
    // Decoding: var = &lit >> 32 ... no.
    // Simpler: use a class or tuple.
    // But `tuple` is slow/heavy? No, fine.
    // Or just arrays.
    // `int L_u[m], L_v[m], L_w[m]`.
    
    vector<int> U(m), V(m), W(m);
    
    // Reset vars to build adj correctly.
    // adj is `vector<vector<int>>` indexed 1..n.
    adj.assign(n + 1, vector<int>());
    clause_vars.resize(m); // Wait, we don't really need `clause_vars` if we store clause in `U,V,W` arrays and iterate via `adj`.
    // When we flip var `v`, we iterate `adj[v]`. For each `c` in `adj[v]`, we check `U[c]`, etc.
    // Yes.
    
    // We need `clause_vars`? No, just `U, V, W` and `adj` is enough.
    // Wait, `adj` helps us find which clauses to update.
    // `U, V, W` help us evaluate clause condition.
    // Yes.

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        U[i] = (a > 0 ? a : -a);
        V[i] = (b > 0 ? b : -b);
        W[i] = (c > 0 ? c : -c);
        // Add to adj
        adj[U[i]].push_back(i);
        adj[V[i]].push_back(i);
        adj[W[i]].push_back(i);
        // Note: U[i], V[i], W[i] are variable indices (1..n).
    }
    
    // Random initialization
    // Using mt19937
    // Since we want deterministic-ish behavior for debugging, or good randomness.
    // Problem says "Any assignment".
    // But we want to maximize score.
    
    vector<int> x_val(n + 1, 0);
    vector<int> current_satisfaction(m, 0);
    int best_score = 0;
    int max_best = 0;
    
    // Loop randomizations
    for (int outer = 0; outer < 20; ++outer) { // 20 tries
        // Fill x_val
        // Fill current_satisfaction
        // We can optimize: instead of full bool array for satisfaction, compute on the fly?
        // No, O(M) per flip is slow. Array is needed.
        // Initialization: O(M)
        for(int i=0; i<m; ++i) current_satisfaction[i] = 0;
        for(int i=1; i<=n; ++i) {
            x_val[i] = rand() % 2; 
        }
        
        // Compute initial satisfaction
        int current_total = 0;
        for (int i = 0; i < m; ++i) {
            // Check if clause i satisfied
            // Clause (U[i], V[i], W[i])
            // Values: x_val[U[i]] if U[i]>0 else !x_val[U[i]]? 
            // My x_val is 0/1. Let's map to 1/-1 internally or handle carefully.
            // Let's stick to 0/1.
            // True literal check:
            // if a > 0: x[a] == 1. If a < 0: x[a] == 0.
            
            bool ok = false;
            int u = U[i]; 
            // Helper lambda or macro to check literal
            auto check_lit = [&](int var_idx, bool pos) {
                int val = (x_val[var_idx]);
                // val is 0 (F) or 1 (T)
                return pos ? val : !val;
            };
            if(check_lit(U[i], U[i] > 0)) ok = true;
            else if(check_lit(V[i], V[i] > 0)) ok = true;
            else if(check_lit(W[i], W[i] > 0)) ok = true;
            
            if(ok) current_satisfaction[i] = 1;
            if(current_satisfaction[i]) current_total++;
        }
        if (current_total > max_best) max_best = current_total;
        
        // Local Search
        // We iterate passes
        for (int pass = 0; pass < 10; ++pass) {
            bool any_change = false;
            // Iterate all variables
            // For each var, calculate potential gain.
            // Optimization: We can just iterate adj[v] to see if we can flip v.
            // Calculating gain for ALL variables in ONE pass takes O(M).
            
            // Wait, scanning all variables and calculating gain is O(M).
            // If we flip, we break and restart? Or continue?
            // Coordinate ascent: iterate 1..n. If flip improves, flip and continue?
            // If we continue, we don't restart scan, saving setup time.
            // But we must ensure we process updated state.
            // If we process variable k, then later in same pass variable p (which might depend on k),
            // we use updated state. This is fine.
            
            for (int v = 1; v <= n; ++v) {
                // Calculate gain for flipping v
                // Current state of v
                int v_current = x_val[v];
                int v_flipped = 1 - v_current; // If 0->1, 1->0
                
                // We need to know:
                // 1. How many satisfied clauses involve v and are satisfied ONLY by v? (Loss candidates)
                // 2. How many unsatisfied clauses involve v and can become satisfied by v? (Gain candidates)
                
                int loss = 0;
                int gain = 0;
                
                // Iterate all clauses containing v
                for (int c_idx : adj[v]) {
                    // Check current status of clause c_idx
                    // Is it satisfied by v currently?
                    // Status depends on other two literals.
                    
                    // Lit 1
                    bool lit1_val = (V[c_idx] < 0 ? 0 : (v != V[c_idx]))? No.
                    // Logic:
                    // Get literal info
                    // u = U[c_idx], p = (U[c_idx] > 0)
                    // val = (x_val[u] == p)
                    
                    // Let's extract literal values relative to current x_val.
                    // Current contribution of U[c_idx]
                    // If U[c_idx] is v:
                    //   Lit is True if (v_current) matches polarities.
                    //   If Lit is False, it contributes nothing to satisfaction of clause.
                    //   If Lit is True, we must check if others are False (for "unique") or just if True (for "participating").
                    // To count Loss:
                    //   If clause is currently Satisfied AND Lit1_is_True AND others_are_False.
                    //   Then flipping v makes Lit1 False, and clause becomes Unsatisfied.
                    // To count Gain:
                    //   If clause is currently Unsatisfied AND Lit1_is_False AND others_are_False.
                    //   Then flipping v makes Lit1 True, and clause becomes Satisfied.
                    
                    // Let's implement helper to check literal value and if it determines clause
                    // A clause is determined by literal L if L is True and other two are False.
                    
                    // Wait, checking "others_are_False" requires checking U, V, W.
                    // Since we iterate adj[v], we have indices of clauses.
                    // Let's grab the other two literals directly.
                    
                    int u = U[c_idx], v_idx = V[c_idx], w = W[c_idx];
                    // Wait, we are iterating `adj[v]`, but the stored variables might not be ordered.
                    // So we need to identify which one is `v` vs others.
                    // Let's write clean helper.
                    
                    bool lit_u_active = (U[c_idx] == v);
                    bool lit_v_active = (V[c_idx] == v);
                    bool lit_w_active = (W[c_idx] == v);
                    // Impossible for 2 to match since distinct vars usually?
                    // Problem says inputs are [-n, n] non-zero.
                    // Doesn't explicitly say distinct vars in clause. But "clause contains exactly three literals".
                    // Could be (x ∨ x ∨ y).
                    // If x appears twice, we just treat independently.
                    
                    // We focus on ONE instance of v.
                    // If U[c_idx] == v:
                    //   Check contribution.
                    //   If Lit is False (contributes 0):
                    //     Check if clause is satisfied by others.
                    //     If NOT satisfied, Gain++ (flipping v makes it satisfied IF v is True).
                    //   If Lit is True:
                    //     Check if satisfied by others.
                    //     If NOT satisfied by others, then v was the ONLY support. Loss++.
                    
                    // Let's generalize:
                    // Determine status of the clause CURRENTLY.
                    // Determine status of the clause AFTER flipping v.
                    // If 0 -> 1, gain++. If 1 -> 0, loss++.
                    
                    // Optimization: Don't re-check whole clause.
                    // Just check if v's literal is True.
                    // And check if others support.
                    
                    // Case 1: v's literal is currently TRUE.
                    //   If others support? (i.e. clause satisfied by others).
                    //     If YES: flipping v -> v becomes FALSE, clause stays SAT (still supported by others). Gain/Loss = 0.
                    //     If NO: flipping v -> clause UNSAT. Loss = 1.
                    // Case 2: v's literal is currently FALSE.
                    //   If others support?
                    //     If YES: clause SAT. Flipping v (becomes True) -> clause SAT. Gain = 0.
                    //     If NO: clause UNSAT. Flipping v -> clause SAT. Gain = 1.
                    
                    // So we just need to know:
                    // Is the clause satisfied by v's literal?
                    // Is the clause satisfied by the OTHER literals?
                    
                    // Helper:
                    // bool is_sat_by_others(c) checks if U and V (ignoring v) satisfy.
                    
                    // This is easy to code.
                    
                    // However, we need to do this for EVERY var `v`.
                    // We are inside `for v = 1..n`.
                    // Inside, loop `c_idx : adj[v]`.
                    // Inside that, check `is_sat_by_others`?
                    // Checking `is_sat_by_others` takes O(1).
                    
                    // So total cost per pass: Sum(deg(v)) = 3M.
                    // This is perfectly fine.
                    
                    bool is_sat_now = false;
                    bool is_sat_if_v_flips = false;
                    
                    // Determine contribution of v's presence in this clause
                    // We need to identify if this clause is one where v is involved.
                    // It is in `adj[v]`, so it must be.
                    // It could be (v, u, w) or (u, v, w) etc.
                    // We don't know which index maps to which field in U, V, W without checking.
                    
                    // Let's check which position holds v.
                    bool u_is_v = (U[c_idx] == v);
                    bool v_is_v = (V[c_idx] == v); // Var name clash with V vector? No V[c_idx] is int var idx.
                    // Rename vectors to `lit_a`, `lit_b`, `lit_c` to avoid confusion with variable `v`.
                    
                    // Actually, better:
                    // int la = U[c_idx]; bool pa = (la > 0);
                    // int lb = V[c_idx]; bool pb = (lb > 0);
                    // int lc = W[c_idx]; bool pc = (lc > 0);
                    
                    // If la == v, pa is polarity.
                    // Value of la's literal in clause:
                    // val_la = (x_val[v] == pa) (since la=v)
                    // Similarly for others.
                    
                    // Let's construct a generic check function inline.
                    // But we need to check "is satisfied by others".
                    // If la == v, others are lb, lc.
                    // If lb == v, others are la, lc.
                    // If lc == v, others are la, lb.
                    // If multiple (redundant), treat independently.
                    
                    // To avoid complex logic, let's just re-evaluate clause status using `x_val`.
                    // Before loop: `sat_now = check_clause(c_idx)`
                    // After simulating flip (temporarily change x_val[v]): `sat_after = check_clause(c_idx)`
                    // Restore.
                    // If sat_now && !sat_after -> Loss++.
                    // If !sat_now && sat_after -> Gain++.
                    
                    // Check Clause function:
                    // Takes c_idx.
                    // Returns bool.
                    // Complexity O(1).
                    // Calling it inside nested loop -> 3 calls per clause.
                    // Wait, checking clause involves looking up U, V, W.
                    // Total operations: $3M$ lookups.
                    // This is fine.
                    
                    // So:
                    // for c_idx : adj[v]:
                    //    int sat_before = check(c_idx);
                    //    // Flip v temporarily
                    //    // ...
                    //    // Check
                    //    // Restore
                    //    // Update stats
                    
                    // Is this correct?
                    // `adj` gives us all clauses involving `v`.
                    // When we flip `v`, only these clauses change status.
                    // So we calculate delta.
                    // We sum up `gain - loss` for all `v`.
                    // Pick max gain `v`? Or just first positive?
                    // Let's sum total gain over all `v`? No, we can only flip one at a time.
                    // We find max positive gain for `v`, flip it.
                    // Then repeat pass.
                    
                    // Wait, my loop `for v = 1..n` updates `x_val` instantly.
                    // If we update `x_val` immediately, the next `v` sees new state.
                    // This is correct for sequential optimization.
                    // However, calculating gain dynamically is hard if we don't know `v`'s specific clauses.
                    // Wait, if we update `x_val` in place, we don't need to "simulate" flip for others.
                    // Because `v`'s change affects its clauses.
                    // But we are currently at `v`.
                    // We haven't processed `v` yet in this step.
                    // So state is old.
                    // So we can safely check gain for `v`.
                    // If we decide to flip `v`, we flip it and continue.
                    // The subsequent variables `u > v` will see updated `x_val[v]`.
                    // This is Coordinate Ascent.
                    // Correct.
                    
                    // So we need to compute gain for `v`.
                    // We can do this by:
                    // `int cnt = 0` (net gain)
                    // `for c : adj[v]`
                    //   `int s = is_sat(c_idx, x_val)`
                    //   `int t = is_sat(c_idx, x_val_modified)`
                    //   `cnt += (t > s ? 1 : (s > t ? -1 : 0))`
                    // `if cnt > 0` -> Flip.
                    // `else` -> Continue to next v.
                    
                    // This is perfect.
                }
            }
        }
        
        if (max_best < current_total) max_best = current_total;
        // Update best score globally?
        // Wait, `current_total` is after last pass.
        // We should update `max_best` outside loop.
        // If we break early, we should record `current_total`.
        
        // To avoid overhead of recomputing everything:
        // Just track `current_total`.
        
    }
    
    // Final output
    for (int i = 1; i <= n; ++i) cout << x_val[i] << (i == n ? "" : " ");
    cout << "\n";

}
```
Wait, one issue: `is_sat` needs to be fast.
Inside `adj` loop, we access `U`, `V`, `W` arrays.
`x_val` is accessed.
Access pattern is random. Cache misses?
Vectors are contiguous. Accessing `U[i]` is fast.
`x_val` is small (3000 ints), fits in L1 cache.
Clues `U,V,W` are large ($2 \cdot 10^6$). Accessing `adj[v]` (indices) -> accessing `U[idx]` -> jumping around in memory.
$M=2 \cdot 10^6$ ints is ~8MB. Fits in L3, maybe not L2.
Should be okay.
Using `vector` for `adj` adds indirection.
We can store clauses as a contiguous array `Clause` struct?
No, random access is key.
Optimization: `vector<array<int, 3>> clauses`.
Then `adj` stores pointers/indexes.
Actually, just access `clauses[c_idx]` (struct/arrays).
Inline the check.

Revised Code Structure:
Use `vector<array<int, 3>>` to store `U`, `V`, `W` (normalized absolute values) and another array for polarity?
Or just store signed values directly.
Input is $a, b, c$.
Store as `int literals[3*M]`.
Wait, adjacency is needed.
`vector<vector<int>> adj(n + 1)`.
Inside `main`, use a custom check function.
Optimization: `x_val` array size `n+1`.

One edge case: $m=0$. Code handles loop `i < m` so safe.
Wait, `x_val` is `0` or `1`.
Input $a \in [-n, n]$.
My normalization: `abs(a)` for var index.
Sign check `a > 0`.
Correct.

Wait, `rand()` might be poor. Use `mt19937`.
Include `<random>`.
Seed with time.

Complexity:
Outer loop 20 times.
Inner loop 10 passes.
Inside inner loop: iterate $v \in 1..N$.
  Inside: iterate `adj[v]` (size deg(v)).
  Ops: $3 \times$ number of clauses = $3M$.
  Total per pass: $3M$ checks + loop overhead.
  Total time: $20 \times 10 \times 3M \approx 600 \times 2 \cdot 10^6 = 1.2 \cdot 10^9$.
  This is dangerously close to 1-2s limit (usually $10^8$ ops/sec).
  We need to optimize.
  
  Optimization:
  1. Reduce passes to 1 or 2?
     Often 1 pass (coordinate ascent) isn't enough.
     But 2 passes might be enough?
     Or 1 pass but steepest descent?
     Steepest descent requires finding MAX gain.
     If we find MAX gain, we might need less total moves?
     Actually, finding max gain takes full scan ($3M$).
     So Steepest Descent is $O(M)$ per step? No.
     Steepest Descent:
     1. Calc gain for all vars. ($3M$).
     2. Pick max. ($O(N)$).
     3. Flip. ($O(1)$ update).
     4. Recalc gain? No, partial update.
     Partial update is complex.
     
     But we can optimize the "Check Gain" part.
     We can just maintain `total_sat` count.
     When scanning vars to find MAX gain, we can compute `gain(v)` incrementally?
     No, `gain(v)` depends on `adj[v]`.
     
     Wait, is $1.2 \cdot 10^9$ realistic for modern judges?
     Usually $10^8$ ops is safe. $10^9$ might TLE.
     Is there a simpler way?
     Maybe just 1 restart is not enough?
     Maybe fewer passes?
     Let's analyze the constraints.
     $N=3000, M=2 \cdot 10^6$.
     Is there a solution faster than $O(M)$?
     If we just shuffle once and check satisfaction, it's $O(M)$.
     Greedy correction:
     For each clause, if unsatisfied, pick one var to flip.
     This is much faster!
     Algorithm:
     `current_sat = 0`.
     Loop $k=1..m$:
       Identify an unsatisfied clause?
       Need to efficiently find one.
       Use `set` of unsatisfied clause indices?
       Or just loop?
     
     Better Heuristic (Glover/Wooldridge?):
     Iterate through variables.
     If flipping variable $i$ reduces number of unsatisfied clauses (i.e. increases satisfied), flip it.
     Check all occurrences.
     
     Actually, the fastest Max-SAT solver is usually Randomized.
     With $N$ small ($3000$), we can use dynamic programming? No, $2^{3000}$.
     
     Let's trust that $O(M)$ per pass is acceptable if passes are reduced.
     How many passes do we need?
     Coordinate Ascent converges to local optimum.
     Typically, on random instances, local optima are good.
     On adversarial ones (like 3-XOR or specific constructions), it might cycle.
     But given "Any assignment", maybe a simple randomized local search is sufficient.
     Let's reduce passes to 2.
     And reduce restarts to 5.
     Total ops: $5 \times 2 \times 3M \approx 3 \cdot 10^7$. Very fast.
     Is 5 restarts enough?
     With random seed, prob of missing good region decreases with restarts.
     5 seems low. Maybe 10?
     $10 \times 3M = 3 \cdot 10^7$ (per pass).
     Total $10 \times 10 \times 3M \approx 3 \cdot 10^8$.
     This is safe.
     
     Wait, the check `if (cnt > 0)` logic in the loop `for v=1..n` updates `x_val` immediately.
     This modifies the state for subsequent `v`.
     This is correct.
     
     However, calculating `cnt` requires iterating `adj[v]`.
     If we iterate all `v` in order, and flip `v` if `cnt > 0`.
     When we flip `v`, the `cnt` for *subsequent* variables `u > v` might change because `u` shares clauses with `v`.
     We MUST account for this.
     But my logic was: `for v` loop calculates `cnt` based on CURRENT `x_val`.
     If `x_val` is modified in-place, the calculation for `u > v` uses updated `x_val`.
     This effectively propagates the improvements.
     This is correct.
     
     One micro-optimization:
     `is_sat` check involves branches.
     Unrolling or simplifying?
     The branching factor is small.
     
     Final check on "Output Format".
     `n` integers. Space separated?
     "Output exactly one line: n integers".
     
     Wait, problem says "Score is defined as s/m".
     Max score is desired.
     
     One tricky thing: `adj` memory.
     `vector<int> adj[3005]` is fine.
     
     Wait, I see `rand()` usage in thought.
     I should use `mt19937` for better randomness.
     
     ```cpp
     #include <bits/stdc++.h>
     using namespace std;
     
     // Fast I/O
     // Arrays for clauses
     // ...
     ```
     
     Wait, I realized the inner loop:
     `for v = 1 to n`
     `calculate gain`
     `if gain > 0 flip`
     
     If I flip `v`, then for the *next* `v` (which is `v+1`), I will see the updated state.
     This is fine.
     However, note that for `v`, I am checking gain assuming *only* `v` flips.
     If I flip `v`, the actual global score changes.
     My code updates `x_val` and implicitly tracks score?
     No, I don't track score explicitly in this block.
     I just track `current_total` at end of pass.
     Is that correct?
     If I update `x_val` inside loop, `current_total` is out of sync if I don't maintain it.
     But I can recompute `current_total` at end of pass.
     Wait, recomputing `current_total` takes $O(M)$.
     So per pass:
       Scan `v` ($3M$ ops) -> update `x_val`.
       Compute `current_total` ($M$ ops).
       If `current_total` increased, continue loops.
       Else break pass.
     
     This is perfectly consistent.
     
     To further speed up, we can track `current_total` incrementally during the `v` loop.
     When we flip `v`, we know exactly how many clauses changed.
     We add `gain` to `current_total`.
     Wait, `gain` calculated for `v` assumes others static.
     But since `v` is the *only* one changing in this step (relative to previous state), yes.
     So we can just increment a counter `satisfied_count` by `cnt`.
     Wait, `cnt` is (gain - loss).
     Yes.
     
     So:
     1. Init `x`, `sat_counts` (boolean array for clauses).
     2. `score = count_sat`.
     3. `passes`:
        `changed = false`
        `for v=1..n`:
          `gain = 0`
          `for c : adj[v]`:
             `before = is_sat(c)`
             `after` (simulate v flip):
                 `x_temp = 1 - x[v]`
                 `after = is_sat(c)`
             `diff = after - before`
             `gain += diff`
          `if gain > 0`:
             `x[v] = 1 - x[v]`
             `score += gain`
             `changed = true`
             // NOTE: is_sat(c) relies on x[]
             // We need to be careful: `is_sat` uses `x[]`.
             // If we modified `x[]` for `v`, then `is_sat` for subsequent `c`'s (if they involve `v`) will reflect it?
             // Wait, if we modify `x[v]`, then in the *same loop*, subsequent checks for `v` (no other clauses for `v`?)
             // `adj[v]` contains all clauses with `v`. We iterate them all.
             // But for *next variable* `u`, `is_sat` will use the NEW `x[v]`.
             // This is what we want (propagate changes).
             // BUT `is_sat` for `v` itself must be computed BEFORE `x[v]` is flipped.
             // So we compute `gain` using OLD `x`.
             // THEN we flip `x`.
             // THEN `score` updated.
             
             // Wait, does modifying `x` affect subsequent `is_sat` calls in the SAME `v` iteration?
             // We iterate `c_idx` in `adj[v]`. All involve `v`.
             // If we modify `x[v]` *after* calculating gain for all clauses involving `v`, that's fine.
             // If we modify `x[v]` *during* iteration?
             // Better to calculate `gain` for `v`, then apply change.
             // This avoids confusion.
             
     This logic holds.
     
     Wait, one edge case:
     `gain` calc requires `is_sat` twice.
     `is_sat` calls `x`.
     If `is_sat` for `c1` reads `x[v]` as `val_old`.
     We compute gain.
     Then we set `x[v]` to `val_new`.
     Next `v` iteration will use `val_new`.
     This is correct.
     
     Is `is_sat` fast enough?
     Inline expansion.
     
     Let's refine the solution code.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

using namespace std;

// Using a fixed size array for clauses might be faster than vector of tuples?
// vector<int> U, V, W is fine.
// Global to avoid stack alloc issues (though heap is default for vectors).

const int MAXN = 3005;

int U[2000005];
int V[2000005];
int W[2000005];
vector<int> adj[MAXN];
int x_val[MAXN]; // 0 or 1

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Reading input
    // We use 1-based indexing for variables
    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        U[i] = abs(a);
        V[i] = abs(b);
        W[i] = abs(c);
        adj[U[i]].push_back(i);
        adj[V[i]].push_back(i);
        adj[W[i]].push_back(i);
    }

    // Randomized restarts
    // Using a simple Linear Congruential Generator for determinism/randomness mix?
    // Just mt19937 is fine.
    // But for safety, maybe a fixed seed or system time.
    // We'll use chrono clock.

    // We need to store satisfaction state to update score efficiently.
    // Actually, recomputing score is M ops. 
    // Tracking score incrementally is faster (updates take O(1) per flip).
    // But we need initial score.
    // Let's store `vector<char> is_sat(m)`.
    
    vector<char> is_sat(m);
    vector<int> current_score; // just a counter?
    // Actually, score is maintained.
    
    // We need a good random init.
    // Let's define a helper to init x_val.
    
    // We'll try a few iterations.
    // Iteration limit: 20.
    // Each iteration:
    //   Init x_val randomly.
    //   Compute initial score.
    //   Loop passes (2-3 passes).
    //   Update best score.
    
    // Global best score?
    // Wait, problem asks for ANY assignment? No, "satisfy as many clauses as possible".
    // Output ONE assignment corresponding to the best score found?
    // "Output exactly one line: n integers".
    // Implies we want the result of our best effort.
    
    // Wait, the prompt says "Your response should ONLY contain the C++ code".
    
    // Let's implement loop.
    
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    for (int iter = 0; iter < 15; ++iter) {
        // Reset x_val
        // To avoid memset/clear overhead, just resize/generate.
        // But x_val is global array? Better use vector.
        // Let's use vector<int> cur_x(n + 1).
        // But global array is faster.
        // Just reset `cur_x`?
        // Let's use local vector inside loop.
        
        vector<int> local_x(n + 1);
        uniform_int_distribution<int> dist(0, 1);
        for (int i = 1; i <= n; ++i) {
            local_x[i] = dist(rng);
        }
        
        // Compute initial satisfaction
        int satisfied_cnt = 0;
        for (int i = 0; i < m; ++i) {
            int satisfied = 0;
            // Check literal U[i]
            int u = U[i], v_lit = V[i], w = W[i];
            bool l1 = (u > 0); // wait, U stores abs val. Sign lost?
            // No, U, V, W store abs val. Sign info is needed?
            // Wait, if I store only abs val, I lose sign!
            // Ah, crucial point.
            // Input `a` can be negative.
            // If I store `U[i] = abs(a)`, I don't know if it is `a` or `-a`.
            // I need `polarity[i]` array.
            
            // Correction:
            // Need `polarity[i]` = 1 if positive, 0 if negative.
            // Or check `a` directly?
            // We can't. We read `a, b, c`.
            // Let's store polarity in a parallel vector `P[i]`.
            // Wait, `adj` is built with absolute value.
            // So we need `P` array.
        }
    }
```

Refining Data Storage:
Input: `a, b, c`.
`U[i] = abs(a)`, `V[i] = abs(b)`, `W[i] = abs(c)`.
`Sig[i]` = `a > 0 ? 1 : 0`.
`adj` needs `U[i]`.
When checking, we need `Sig`.
Maybe merge into struct or just use 4 arrays.
`U, V, W` (indices).
`PA, PB, PC` (signs).
Or better: `A[i], B[i], C[i]` (signed values) and access via `abs` and `>0` check.
But `abs(-3)` is `3`. `>-0` is true? No.
Just store `val` (integer).
`val` is `a, b, c`.
When checking literal `val`:
`if (val > 0) { return x[val] == 1; }`
`else { return x[-val] == 0; }`
This is concise.

Wait, storing `U, V, W` as absolute values is required for `adj`.
So I need to parse input:
`int a, b, c; cin >> a >> b >> c;`
`int ua = abs(a), vb = abs(b)...`
`adj[ua].push_back(i); adj[vb].push_back(i); ...`
And store `val` in arrays:
`ValsA[i] = a; ValsB[i] = b; ValsC[i] = c;`
Then `check(i, x_val)`:
`if (ValsA[i] > 0) res |= x_val[ValsA[i]]; else res |= (!x_val[ValsA[i]]);`
Wait, `res |=`? If ANY literal true, clause satisfied.
`if ((ValsA[i] > 0 ? x_val[ValsA[i]] : !x_val[ValsA[i]]) || ...)`

This works.
Arrays needed:
`int ValA[2000005]`, `ValB`, `ValC`.
`vector<int> adj[MAXN]`.
`int x_val[MAXN]`.
Size of Val arrays: $3 \times 2 \cdot 10^6 \approx 6 \cdot 10^6$ ints = 24MB. Safe.

Algorithm Recap:
1. Read input. Build `adj` and store values.
2. Loop `iter` 15 times:
   - Init `x` random.
   - Compute `score`.
   - Local Search Loop:
     - `any_flip = false`.
     - Loop `v` 1 to `n`:
       - `gain = 0`.
       - Iterate `c_idx` in `adj[v]`:
         - `diff = calc_diff(c_idx, v, old_val_of_v)`.
           - `calc_diff`: returns `1` if `v` is beneficial, `-1` if harmful.
           - Need `current_state` of clause.
           - `state_before` (with `v` as `old_val`).
           - `state_after` (with `v` as `1-old_val`).
           - `diff = state_after - state_before`.
         - `gain += diff`.
       - If `gain > 0`:
         - `x[v] = 1 - x[v]`.
         - `score += gain`.
         - `any_flip = true`.
     - If `!any_flip` break.
   - Track max `score`. If current `score` > max `score`, save `x` copy.
   
   Wait, `score` calculation can be updated incrementally.
   However, `gain` accumulation depends on `diff`.
   We need to be sure `diff` is correct.
   `diff` for clause `c` involving `v` is `(state_after - state_before)`.
   Since `state` is boolean, `diff` is `0, 1, -1`.
   If `diff` is -1, `gain` decreases.
   Wait, we want to choose `v` such that flipping it increases score.
   So we check `gain`. If `gain > 0`, flip.
   But if `gain < 0`, we don't flip.
   Wait, if `gain < 0` for `v`, we don't flip.
   But we continue to next `v`.
   Wait, if we iterate `v` and find multiple `v` with `gain > 0`, we flip the *first* one?
   Or *all* of them?
   Coordinate Ascent usually flips *one* variable to improve.
   If we flip multiple in one pass (sequential), it's fine.
   But note: if we flip `v1`, the `gain` of `v2` might change.
   So recalculating `gain` for `v2` uses updated `x`.
   My loop iterates `v` sequentially.
   If `v1` is improved, `x` updates. `v2` sees new `x`.
   This is valid.
   But we need to ensure `score` reflects changes.
   `score` should be updated inside loop.
   Yes.
   
   Is it better to find the BEST `v` (max gain) and flip only that?
   This is Steepest Ascent.
   Finding best requires full scan anyway ($3M$).
   Sequential (Coordinate Ascent) is cheaper?
   In Coordinate Ascent, we check `v`, if good, flip, continue.
   Cost is $\sum_{v} \text{deg}(v) = 3M$.
   In Steepest Ascent, we must check ALL `v` before flipping ANY.
   So cost is $\sum \text{deg}(v) = 3M$ (plus finding max).
   Then flip.
   Then repeat.
   Sequential might require more steps (less improvement per step) but faster computation per step (no need to scan all if we just scan once?).
   Wait, Steepest Ascent scans all `v` to pick one. Then stops?
   No, then scans again.
   Steepest Ascent:
     While changed:
       Scan all `v`. Pick max gain.
       If max > 0, flip.
       Else break.
       Cost: $O(Passes \times M)$.
   Coordinate Ascent:
     While changed:
       Scan all `v`. If `v` has gain > 0, flip.
       Cost: $O(Passes \times M)$.
   They are roughly same complexity.
   However, Coordinate Ascent is easier to implement (no need to store min/max).
   And it naturally propagates changes.
   
   However, Coordinate Ascent can get stuck in worse local optima?
   Maybe. But with 15 restarts, it should be fine.
   
   Wait, in Coordinate Ascent, if I flip `v1`, does it hurt `v2`?
   Yes, but `v2` will be checked later with updated `x`.
   The gain of `v2` will be recomputed based on `x` (where `v1` is already flipped).
   So we are optimizing sequentially.
   This is generally good.

   Final check: `gain` calc.
   We need `diff`.
   To compute `diff` efficiently:
   We iterate `c` in `adj[v]`.
   Check if `c` is satisfied by `v` currently.
   Check if `c` is satisfied by others.
   
   Logic:
   `is_sat_with_v`: check if `c` satisfied given `v`'s current val.
   `is_sat_without_v`: check if `c` satisfied given `v`'s flipped val (and others unchanged).
   Actually, we can just toggle `v` for the sake of check.
   `int curr_sat = check(c);`
   `int next_sat = check(c);` (with `x[v]` flipped).
   `diff = next_sat - curr_sat`.
   Wait, if `curr_sat` was 1 and `next_sat` 1, `diff` 0.
   If `curr_sat` 0 and `next_sat` 1, `diff` 1.
   If `curr_sat` 1 and `next_sat` 0, `diff` -1.
   We sum these up.
   
   Since `check(c)` takes constant time (3 lookups), and `adj` size is small per var, it's efficient.
   The bottleneck is memory access patterns.
   But for 2e6 clauses, it should be fast enough with 15 restarts and 2 passes.
   $15 \times 2 \times 3 \times 2 \cdot 10^6 \approx 1.8 \cdot 10^8$ ops.
   Should pass within 1s.

   Double check constraints again.
   $N \le 3000$.
   $M \le 2000000$.
   Time Limit: implicit standard (1-2s).
   Memory Limit: standard (256MB).
   
   Wait, I should probably use `int` arrays for values to avoid `vector` allocation overhead if `m` is huge.
   But `vector` is usually fine.
   I'll use `vector` for clauses and global/static arrays for performance.
   Actually, `adj` needs dynamic sizing. `adj` is array of vectors.
   `U, V, W` can be vectors.
   
   One small bug possibility:
   `adj` stores clause index `i`.
   `check` function accesses `ValA[i]` etc.
   `adj` is `vector<vector<int>>`.
   Accessing `adj[v][k]` might be cache inefficient if `adj[v]` is spread out?
   Vectors are contiguous. `adj` pointers are in vector control block.
   Indices are contiguous.
   It's fine.

```cpp
/*
 * 3-SAT Solution
 * Heuristic: Randomized Coordinate Ascent Local Search
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

using namespace std;

// Maximum constraints
const int MAX_N = 3005;
const int MAX_M = 2000005;

// Global arrays to minimize allocation overhead
int ValA[MAX_M];
int ValB[MAX_M];
int ValC[MAX_M];
vector<int> adj[MAX_N];
int x_val[MAX_N]; // Stores 0 (False) or 1 (True)

// Helper to check if a clause is satisfied
inline bool is_satisfied(int clause_idx, const int* x) {
    bool s1 = (ValA[clause_idx] > 0 ? (x[ValA[clause_idx]] == 1) : (x[ValA[clause_idx]] == 0));
    bool s2 = (ValB[clause_idx] > 0 ? (x[ValB[clause_idx]] == 1) : (x[ValB[clause_idx]] == 0));
    bool s3 = (ValC[clause_idx] > 0 ? (x[ValC[clause_idx]] == 1) : (x[ValC[clause_idx]] == 0));
    return s1 || s2 || s3;
}

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    // Adjust n, m for safety
    // Although n <= 3000, m <= 2000000
    // We should clear global adj if running multiple test cases, but here just one.
    
    // Resize vectors to avoid clearing issues if needed (not strictly necessary for static)
    // But vector constructor clears content.
    // We use global adj, so we should be careful not to leak memory?
    // For single execution, no worry.
    
    // Read input
    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        ValA[i] = abs(a);
        ValB[i] = abs(b);
        ValC[i] = abs(c);
        adj[ValA[i]].push_back(i);
        adj[ValB[i]].push_back(i);
        adj[ValC[i]].push_back(i);
    }
    
    // We need to output x_1 ... x_n where 1 is TRUE, 0 is FALSE.
    // Our x_val stores 0/1. 1 -> TRUE.
    
    int max_satisfied = -1;
    
    // Random engine
    mt19937 rng(chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> distrib(0, 1);
    
    // Restart loop
    // 20 restarts should be sufficient
    for (int r = 0; r < 20; ++r) {
        // Random initialization
        for (int i = 1; i <= n; ++i) {
            x_val[i] = distrib(rng);
        }
        
        // Compute initial score
        int current_score = 0;
        // We can use a loop
        // Optimization: use a bitset or just iterate.
        // Iterating is O(M).
        for (int i = 0; i < m; ++i) {
            if (is_satisfied(i, x_val)) current_score++;
        }
        
        // Local Search
        // Multiple passes to ensure we reach local optimum
        // 5 passes usually enough
        for (int p = 0; p < 5; ++p) {
            bool changed = false;
            for (int v = 1; v <= n; ++v) {
                int current_x = x_val[v];
                int gain = 0;
                
                // Check all clauses involving v
                for (int c_idx : adj[v]) {
                    // Compute current status
                    int sat_now = is_satisfied(c_idx, x_val) ? 1 : 0;
                    
                    // Temporarily flip v to compute potential status
                    // We need to be careful not to mess up the check logic permanently
                    // We can pass a temporary modification or just flip locally
                    // Since we just need `is_satisfied` result.
                    // Modifying global `x_val` temporarily is fine if we revert or restore.
                    // But inside loop we might call is_satisfied multiple times.
                    // Easier: create a snapshot of x_val? No, too slow.
                    // Just manually compute status of clause for flipped v.
                    
                    // Manual check is safer/faster than calling is_satisfied with temp state
                    // But writing manual check is tedious.
                    // Let's try passing a pointer/offset or just copying x_val[v]?
                    // Copying x_val array is O(N). Too slow inside loop.
                    
                    // Better: Modify x_val[v] -> calculate -> Restore.
                    // Wait, if we modify x_val[v], subsequent clauses for *same* v will see new state.
                    // This would invalidate 'gain' calculation.
                    // We must revert after loop or compute all diffs first.
                    // Computing all diffs first:
                    // We can't easily because `x_val` is state-dependent for *other* variables in loop?
                    // No, for `v`, we assume others are static.
                    // So we just need `gain` of flipping `v` given `current` `x`.
                    
                    // So:
                    // 1. Calculate `gain` based on current `x_val` (all `adj[v]` clauses).
                    // 2. If `gain > 0`, flip `x_val[v]`.
                    // 3. Important: do we update `current_score` immediately?
                    //    Yes.
                    
                    // Implementation of manual check for efficiency:
                    // Check status of clause `c_idx` given `x_val` (original).
                    // Check status given `flipped`.
                    
                    // Let's write a helper to check status with a specific value for a specific variable?
                    // No, `x_val` is array.
                    
                    // We can just do:
                    // bool sat_before = is_satisfied(c_idx, x_val);
                    // bool sat_after = is_satisfied(c_idx, x_val); // Same check logic, but v flipped?
                    // We can just swap `x_val[v]` temporarily.
                    // WAIT: If we do `x_val[v] = !x_val[v]`, then `is_satisfied` sees it.
                    // But then for *next* clause in loop, it sees it flipped!
                    // That's BAD. We must calculate `gain` BEFORE flipping.
                    // So we compute all diffs in `adj[v]` BEFORE touching `x_val`.
                    
                    // But wait, `is_satisfied` is fast.
                    // Just iterate, compute `diff` for each `c_idx`.
                    // Accumulate `gain`.
                    // DO NOT FLIP YET.
                    // Then if `gain > 0`, flip.
                    // This avoids re-evaluation side effects.
                    
                    // Wait, what about `x_val` propagation in outer loop?
                    // We are in `for v`.
                    // We compute gain for `v`.
                    // If positive, we flip `x_val[v]` NOW.
                    // This affects `x_val` for future `v`s in this pass.
                    // This is the Coordinate Ascent logic.
                    // This requires that for current `v`, we use `x_val` AS IT IS (from previous `v`s or init).
                    // Correct.
                    // So we just need to calculate gain for `v` using CURRENT `x_val`.
                    // Then flip `x_val[v]`.
                    
                    // Inside `for c_idx`:
                    // Check `sat` using current `x_val`.
                    // Simulate flip: `temp_x[v] = !x_val[v]`.
                    // Check `sat` with `temp_x`.
                    // Restore `x_val[v]`.
                    
                    // To optimize, `is_satisfied` can be specialized.
                    // But let's just use the helper.
                    
                    // However, `is_satisfied` copies arguments?
                    // Passing `const int*` is fine.
                    
                    // Optimization:
                    // `is_satisfied` does 3 bool checks.
                    // Doing it twice per clause is 6 checks.
                    // With $3M$ clauses, $18M$ bool ops. Very fast.
                    
                    int sat_curr = is_satisfied(c_idx, x_val);
                    
                    // Temporary flip
                    int old_v = x_val[v];
                    x_val[v] = (old_v ^ 1);
                    int sat_next = is_satisfied(c_idx, x_val);
                    x_val[v] = old_v; // Restore
                    
                    if (sat_next && !sat_curr) gain++;
                    else if (!sat_next && sat_curr) gain--;
                    // else no change
                }
                
                if (gain > 0) {
                    // Flip for good
                    x_val[v] = (x_val[v] ^ 1);
                    // Update score
                    // Note: gain calculated above is net change.
                    // If gain was positive, we added it.
                    // Wait, `current_score` needs to be updated.
                    // But `gain` accumulates over all clauses.
                    // Yes, `gain` is exactly the change in total satisfied count.
                    // So `current_score += gain`.
                    
                    // But wait, `current_score` was computed BEFORE `v` loop.
                    // Inside `v` loop, we might flip multiple vars.
                    // If we flip multiple vars in one pass (Coordinate Ascent),
                    // The `current_score` must be cumulative.
                    // But `gain` computed for `v` assumes *static* neighbors.
                    // But neighbors ARE changing in previous steps of this pass?
                    // If `v` is `v`, neighbors are clauses.
                    // If we flip `v`, we change clauses.
                    // Later `u` (different var) might share clauses with `v`.
                    // For `u`, the state of those shared clauses has changed because `v` was flipped.
                    // But `gain` for `u` was not computed *yet* (since we iterate `v` then `u`).
                    // So when we reach `u`, we see the updated state of `v` in clauses.
                    // Thus, `gain` for `u` reflects the impact of `v`'s change.
                    // This is consistent.
                    // But `current_score` must be updated incrementally.
                    // Wait, if we flip `v`, the change in total score is `gain`.
                    // So `current_score += gain` is correct?
                    // Yes, because `gain` counts how many satisfied became satisfied and how many unsatisfied became unsatisfied.
                    // Net change = `gain`.
                    
                    // HOWEVER:
                    // The `gain` we computed for `v` assumed that clauses involving `v` had status based on OLD `x` (except `v`).
                    // Yes, that's exactly what `is_satisfied` returned.
                    // But wait, did other clauses involving `v` change? No, `v` didn't flip yet.
                    // Did other clauses involving `v`'s *neighbors* change?
                    // We are just counting satisfaction changes for clauses IN `adj[v]`.
                    // So `gain` is correct for the flip.
                    
                    // So: `current_score += gain`.
                    
                    // Wait, I said "Coordinate Ascent iterates `v`".
                    // If I flip `v`, I update `current_score`.
                    // But `current_score` is used to track global progress.
                    // Do I compare `current_score` to `max_satisfied` at the end?
                    // Yes.
                    
                    // One subtle issue:
                    // `gain` variable inside loop `for v` is sum of differences.
                    // Is it possible `gain` is negative but we don't flip?
                    // Yes.
                    // Is it possible `gain` is 0?
                    // Yes.
                    // Only flip if `gain > 0`.
                    
                    // So logic:
                    /*
                    int gain = 0;
                    // loop adj[v] to calc gain
                    if (gain > 0) {
                       x_val[v] ^= 1;
                       current_score += gain;
                       changed = true;
                    }
                    */
                    // Wait, `current_score` update is tricky.
                    // If I flip `v`, the `gain` is added.
                    // But later in the loop (for same pass), I compute gain for `u`.
                    // `u` might share clauses with `v`.
                    // The satisfaction of those clauses will reflect `v`'s flip.
                    // So `gain(u)` is calculated correctly.
                    // But `gain(u)` adds its own delta to `current_score`.
                    // Is the sum of deltas correct?
                    // Yes, Total Score = Sum(Clause_Satisfaction).
                    // Change in Total Score = Sum(Clause_Delta).
                    // So summing increments is correct.
                    
                    // Wait, what about the fact that `current_score` is maintained?
                    // If I maintain `current_score`, do I need to recompute it? No.
                    // But `current_score` must be accurate.
                    // Initial `current_score` is correct.
                    // Subsequent `current_score` is `prev + gain`.
                    // This works perfectly.
                    
                    // WAIT.
                    // `is_satisfied` checks the state of `c_idx`.
                    // If `c_idx` involves `v` and `u`.
                    // When processing `v`, we compute `gain` assuming `u` is static.
                    // If we process `u` later in pass, `v` is already flipped.
                    // So `gain(u)` accounts for `v`'s flip.
                    // Is `gain(v)` accounting for `u`'s flip? No, because `u` hasn't flipped yet.
                    // So `gain(v)` is computed on basis of state before `v`'s flip.
                    // `gain(u)` is computed on basis of state before `u`'s flip (but after `v`'s).
                    // The sequence of flips matters.
                    // Is it valid? Yes.
                    // Coordinate ascent works this way.
                    // Does it lead to local optimum? Yes.
                    
                    // One catch:
                    // We need to make sure we don't double count if multiple flips happen in one pass?
                    // No, each flip is sequential.
                    // But wait, if we flip `v`, `current_score` increases.
                    // Does it matter?
                    // The `max_satisfied` is the metric.
                    // We just want `current_score` to be high at the end of the pass.
                    // The path taken doesn't matter as long as final `current_score` is maximal.
                    
                    // So, we are good.
                    // But wait, I need to verify `current_score` logic.
                    // Suppose I have 1 variable, 1 clause.
                    // Init score 0.
                    // Loop `v`. `gain` calc:
                    //   `sat_curr` (0), `sat_next` (1). Diff 1.
                    //   `gain` = 1.
                    // Flip. `score` += 1. Score = 1.
                    // End. Correct.
                    
                    // Suppose I have 2 variables `a, b`, 1 clause `(a or b)`.
                    // Init: a=0, b=0. `sat`=0. Score 0.
                    // Loop `a`:
                    //   `sat_curr`=0, `sat_next` (with a=1)=1. Gain=1.
                    //   Flip `a`. Score+=1 => 1.
                    // Loop `b`:
                    //   Now `a=1, b=0`. `sat`(c) = 1.
                    //   `sat_curr`=1.
                    //   `sat_next` (with b=1)=1.
                    //   Gain=0.
                    //   Don't flip `b`.
                    // End Pass.
                    // Score 1. Correct.
                    
                    // What if order was `b, a`?
                    // Loop `b`:
                    //   Init a=0, b=0. `sat`=0.
                    //   `sat_next` (b=1)=1. Gain=1.
                    //   Flip `b`. Score 1.
                    // Loop `a`:
                    //   a=0, b=1. `sat`=1.
                    //   `sat_next` (a=1)=1. Gain=0.
                    // Score 1. Correct.
                    
                    // Looks consistent.
                }
            }
            if (!changed) break;
        }
        
        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            // We need to save the assignment?
            // No, just outputting x_val is fine.
            // We need the x_val that gave max_score.
            // But `x_val` is global? No, inside `solve`?
            // Wait, I'm inside `solve`.
            // I should store the best x_val.
            // `x_val` is global.
            // We need to capture `x_val` when `current_score` > `max_satisfied`.
            // Since we want to output ONE line at end.
            // We can just copy `x_val` to a buffer `best_x`.
            // But copying 3000 ints is cheap.
        }
    }
    
    // Output
    // Actually, I need to output `best_x`.
    // Since I modified global `x_val` inside loop, if I overwrite it, I might lose best.
    // So I should keep a `vector<int> best_solution(n + 1)`.
    
    // Wait, `x_val` is declared inside `solve`?
    // I put it global for convenience.
    // I need `best_solution` global too.
    
    // Wait, `current_score` variable scope.
    // It must be outside `p` loop to track current run.
    
    // Refined plan for variable tracking:
    // Move `best_solution` to global.
    // Move `x_val` to global (working copy).
    
}
```
Wait, inside `solve`, I need to capture `x_val` when a new max is found.
Let's declare `vector<int> best_ans(n + 1);` globally or inside.
Given strict CP, global arrays are better to avoid stack overflow.
I will use `vector<int> best_ans(MAX_N)`.

One issue: `vector<char> is_sat` optimization.
In the logic above, I didn't use it. I just computed score.
Computing score at end of pass is $O(M)$.
Updating score incrementally is $O(1)$ per flip.
Incremental is better.
But checking `gain` requires checking clauses anyway.
Checking `is_sat` is needed for gain.
Wait, if I use `is_sat` check inside gain loop, I don't need a separate score calc loop.
Yes.
So I don't need to recompute score at end of pass.
Just update `current_score` as I go.

Wait, `is_satisfied` checks `x_val`.
Inside `gain` calc for `v`:
Iterate `c_idx`.
Check `sat` (with `x_val`).
Temp flip `v`.
Check `sat` (with flipped).
Restore.
This takes 2 calls to `is_satisfied`.
Calls to `is_satisfied` check 3 literals.
Total ops ~ $6 \times$ clauses involving `v`.
Sum over all `v` is $18 \times M$.
Still very fast.

Final check: Memory Usage.
`ValA, ValB, ValC`: 3 arrays of $2 \cdot 10^6$ ints = 24MB each.
Total 72MB.
`adj`: Vector of vectors. Overhead is small.
`best_ans`: negligible.
Total < 100MB. Limit usually 256MB. OK.

Time complexity:
$20 \times 5 \times 18M \approx 1.8 \cdot 10^9$ ops.
This is a bit high.
Maybe reduce restarts to 5 and passes to 3.
$5 \times 3 \times 18M \approx 2.7 \cdot 10^8$.
Safe.
I'll use 5 restarts and 3 passes.
Also, `uniform_int_distribution` generation is slow.
Just use `rand()`? `rand()` is usually weak but for CP often fine.
Or just `time` seed.
Actually, the simplest way to randomize is `xorshift` or `rand()`.
I'll use `rand()` and seed it once.
But `rand()` produces $2^{31}$ states, plenty.

One small optimization:
Don't use `vector<int> adj[MAXN]`.
It's static.
Initialize `adj` with `clear` inside `solve`?
If `solve` runs once, no need.
If test cases run multiple times in one file (unlikely for this prompt context), need clear.
Prompt: "You will be given a problem statement". Usually one test case per run.
However, I will just `fill(adj, adj+N+1, vector<int>())` or resize.
`adj` is global.
I'll just assume single run.

Wait, `best_ans` should be initialized with first valid solution.
Just initialize `max_satisfied = -1`.
After first pass, `current_score` will be >= 0.
So first update will trigger save.
Or just initialize `best_ans` with `x_val`.

Code Structure:
Includes.
Global vars.
`solve()` function.
`main()` calls `solve()`.
`FastIO`.
Logic inside `solve`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

using namespace std;

const int MAXN = 3005;
const int MAXM = 2000005;

int U_arr[MAXM];
int V_arr[MAXM];
int W_arr[MAXM];
vector<int> adj[MAXN];
int x_val[MAXN]; 
int best_ans[MAXN];
int max_sat_count = -1;

// Inline function to check clause satisfaction
inline bool is_sat(int idx) {
    int u = U_arr[idx], v = V_arr[idx], w = W_arr[idx];
    bool r1 = (u > 0) ? (x_val[u] == 1) : (x_val[u] == 0);
    bool r2 = (v > 0) ? (x_val[v] == 1) : (x_val[v] == 0);
    bool r3 = (w > 0) ? (x_val[w] == 1) : (x_val[w] == 0);
    return r1 || r2 || r3;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        U_arr[i] = abs(a);
        V_arr[i] = abs(b);
        W_arr[i] = abs(c);
        adj[U_arr[i]].push_back(i);
        adj[V_arr[i]].push_back(i);
        adj[W_arr[i]].push_back(i);
    }
    
    // Clear adj for safety? (Not needed for single execution but good practice)
    // Since we use global, we need to ensure it's fresh.
    // Assuming single test case per run (standard for this format).

    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    // Simple uniform distribution
    auto dist = [](mt19937& eng) -> int {
        return (eng() & 1); // 0 or 1
    };

    // We want to find the best assignment.
    // Save best assignment.
    // Initialize best_ans with first valid?
    // Let's just do it during loop.
    
    // To handle output, best_ans is filled.
    // Note: x_val is global.
    
    // Use simple pseudo-random init to avoid header dependencies if needed?
    // No, <random> is standard.
    
    // 10 restarts
    for (int r = 0; r < 10; ++r) {
        for (int i = 1; i <= n; ++i) x_val[i] = dist(rng);
        
        // Calculate initial score
        int current_score = 0;
        for (int i = 0; i < m; ++i) {
            if (is_sat(i)) current_score++;
        }

        // 3 passes
        for (int p = 0; p < 3; ++p) {
            bool changed = false;
            for (int v = 1; v <= n; ++v) {
                int gain = 0;
                // Calculate potential gain for flipping v
                // To optimize, we check adj[v]
                // Note: x_val[v] is current value.
                
                // Optimization: We can check if v is involved in ANY clause.
                // adj[v] is empty? If empty, gain=0.
                if (adj[v].empty()) continue;
                
                // We need to know if flipping v helps.
                // Calculate gain without actually flipping yet.
                // We need to be careful: `is_sat` uses `x_val`.
                // If we flip `x_val` temporarily inside loop, we might mess up for next clauses?
                // No, as discussed, we must use ORIGINAL x_val for ALL clauses in `adj[v]`.
                // So just iterate and check logic manually or use `is_sat`.
                // But `is_sat` reads `x_val`.
                // If I flip `x_val[v]` and don't revert, subsequent `is_sat` checks for *this* `v` will be wrong.
                // I MUST NOT flip inside loop.
                
                int old_val = x_val[v];
                
                // Iterate clauses
                for (int c_idx : adj[v]) {
                    bool sat_curr = is_sat(c_idx);
                    
                    // Check if flipping makes it SAT (if not) or UNSAT (if sat)
                    // To do this without modifying x_val permanently, we check logic.
                    // But modifying x_val temporarily is easiest.
                    // BUT, if I modify x_val[v], subsequent calls to is_sat(C) in THIS loop will see modified value.
                    // This would be WRONG.
                    // Because `is_sat` depends on `x_val`.
                    // So I MUST NOT flip `x_val` permanently during the loop.
                    // I can flip it, compute, revert.
                    // But that requires reverting for EACH clause?
                    // NO. Reverting is O(1).
                    // So:
                    // x_val[v] = 1 - x_val[v];
                    // int sat_next = is_sat(c_idx);
                    // x_val[v] = old_val;
                    
                    // Wait, `adj[v]` loop processes clauses.
                    // If I flip `x_val[v]`, `is_sat` for NEXT clause uses new value.
                    // That's fine for that clause check.
                    // But I need `sat_curr` for THAT clause.
                    // If I flip first: `sat_curr` uses NEW value? No.
                    // I need BOTH `sat_curr` (old) and `sat_next` (new).
                    // If I flip at start of loop:
                    //   `sat_curr` (with new) -> NO.
                    //   I need `sat_curr` (with old).
                    
                    // Option 1:
                    //   `bool sat_old = is_sat(c_idx);`
                    //   `x_val[v] = 1 - x_val[v];`
                    //   `bool sat_new = is_sat(c_idx);`
                    //   `x_val[v] = old_val;`
                    // This requires reverting `x_val[v]` every time.
                    // Cost: $3M \times 2$ assignments. Negligible.
                    // This ensures correctness.
                    
                    bool s_curr = is_sat(c_idx);
                    x_val[v] = 1 - x_val[v];
                    bool s_next = is_sat(c_idx);
                    x_val[v] = old_val; // Restore
                    
                    if (s_next && !s_curr) gain++;
                    else if (!s_next && s_curr) gain--;
                }
                
                if (gain > 0) {
                    x_val[v] = 1 - x_val[v];
                    current_score += gain;
                    changed = true;
                    // Optimization: break early if gain is very small? No, coordinate ascent.
                    // Just continue.
                    // But wait, if we flip `v`, does it mean we should STOP scanning other variables `u`?
                    // No, coordinate ascent updates all variables in one pass.
                    // Wait, standard coordinate ascent: iterate $x_1..x_n$. If optimal direction, update.
                    // So we continue loop.
                }
            }
            if (!changed) break;
        }
        
        // Check against max
        if (current_score > max_sat_count) {
            max_sat_count = current_score;
            for (int i = 1; i <= n; ++i) best_ans[i] = x_val[i];
        }
    }
    
    for (int i = 1; i <= n; ++i) cout << best_ans[i] << (i == n ? "" : " ");
    cout << "\n";

    return 0;
}
```
Wait, `x_val[v] = 1 - x_val[v]` logic.
In `is_sat`, `x_val` is used.
If I revert `x_val` after each clause, it works.
But inside the `if (gain > 0)` block, I flip `x_val[v]` permanently.
This is correct.
Wait, inside loop `for v`:
I iterate `adj[v]`.
For each clause `c`, I check `s_curr` and `s_next`.
Here `s_curr` is based on original `x_val` (before loop or restored).
If I restore after each clause, `s_curr` is always checked against correct old `x_val`.
Correct.

Wait, `is_sat` reads `x_val`.
If I revert `x_val[v]` after each clause, I am resetting `x_val` to `old_val`.
This is consistent.

Wait, if I find `gain > 0`, I flip `x_val[v]` permanently.
This happens AFTER the `gain` loop.
So `gain` loop uses `old_val`.
Then I apply `x_val[v] = 1 - old_val`.
This is consistent.
BUT, inside the loop I do `x_val[v] = 1 - x_val[v]` temporarily.
Wait, initially `x_val[v]` is `old_val`.
Loop starts.
`bool s_curr = is_sat(c_idx);` (reads `old_val`)
`x_val[v] = 1 - old_val` (writes `new_val`)
`bool s_next = is_sat(c_idx);` (reads `new_val`)
`x_val[v] = old_val` (restores `old_val`)
Next iteration `c_idx_next`.
`s_curr` reads `old_val`.
Correct.

Wait, one small optimization:
The loop `for v` scans ALL variables.
If I flip `v`, I change state for subsequent variables.
This is allowed.
BUT, calculating `gain` for `v` relies on `adj[v]`.
Is `adj[v]` order important? No.
Is `is_sat` logic sound? Yes.

There is a slight inefficiency:
Inside `gain` calc loop, I do `is_sat` twice.
Is `is_sat` slow?
It's inline, 3 lookups, 3 comparisons.
With `adj[v]` having many elements, this dominates.
With 2M clauses, `is_sat` called ~4M times per pass.
10 passes = 40M calls.
Very fast.
Should be well within 1s.

One corner case: `x_val` is 0 or 1.
`abs` usage in `U_arr` etc is correct.
Constraints: $N \le 3000$. `MAXN` 3005 is safe.
$M \le 2000000$. `MAXM` 2000005 is safe.

Final check on headers. `<vector>`, `<iostream>`, `<cmath>`, `<chrono>`, `<random>`, `<algorithm>`.
Namespace `std`.
All good.

One detail: `dist` lambda in `main` captures `rng` by reference.
Defined inside `main`.
Works.

Double check: `if (!(cin >> n >> m)) return 0;`.
Standard.
Output formatting: `n` integers space separated.
Trailing space handled by logic `(i == n ? "" : " ")`.
Wait, `cout` ends with `\n`.
Good.

One logic check: `adj` is filled with indices `0..m-1`.
`is_sat` uses `U_arr` etc.
Input `a` can be negative.
`U_arr` stores `abs(a)`.
`is_sat` logic: `(u > 0) ? (x_val[u] == 1) : (x_val[u] == 0)`.
If input was `-u`, `u` in `U_arr` is `u`.
Condition `u > 0` checks `a`'s sign?
Wait. `U_arr` stores `abs(a)`.
If `a` was `-3`, `U_arr` is `3`.
If `a` was `3`, `U_arr` is `3`.
We lost the sign information!
My code uses `(u > 0)` but `u` is now index (positive).
`u` is `val`.
`ValA` is `abs(a)`. It is always positive (since input `non-zero`).
So `u > 0` is ALWAYS true.
I need to store SIGN separately!
Ah, critical error in logic derivation.
The code snippet `u > 0` in `is_sat` relies on `u` containing the sign or something?
No, `u` comes from `U_arr`.
I need `U_arr` to store original values, not abs?
No, `adj` needs indices.
Indices must be positive.
So I must have TWO arrays: one for indices (`idx_a`), one for signs (`sign_a`).
Or combine them?
I can store `U_arr` as `abs(a)`.
And have `is_sat` check sign using a separate array `SignA`.
Okay, fix this.
Add `bool signU[MAXM]` (or char).
Actually `ValA` stores `abs(a)`.
Need `polarityA`.
`polarityA[i] = (a > 0 ? 1 : 0)`.
In `is_sat`:
`bool r1 = polarityA[i] ? (x_val[u] == 1) : (x_val[u] == 0);`
Where `u = ValA[i]`.
Wait, `ValA` is `abs(a)`.
If `a` was `-3`, `polarity` 0.
`x_val[3] == 0` means `-x3` is True.
Correct.

Wait, I need to adjust `adj` too.
`adj[ValA[i]].push_back(i)`.
`adj[ValB[i]].push_back(i)`.
etc.
And `polarity` arrays.
So I need 3 arrays of polars?
Yes, `pa, pb, pc`.

Revised Data Structures:
`int U_abs[MAXM], V_abs[MAXM], W_abs[MAXM];`
`int PA[MAXM], PB[MAXM], PC[MAXM];`
`vector<int> adj[MAXN];`

Read loop:
`int a, b, c; cin >> a >> b >> c;`
`U_abs[i] = abs(a); V_abs[i] = abs(b); W_abs[i] = abs(c);`
`PA[i] = (a > 0); ...`
`adj[U_abs[i]].push_back(i); ...`

`is_sat(idx)`:
`int u = U_abs[idx], v = V_abs[idx], w = W_abs[idx];`
`bool r1 = (PA[idx]) ? (x_val[u] == 1) : (x_val[u] == 0);`
...
Correct.

Wait, `PA` stores 0 or 1.
Input $a$ is non-zero.
$a > 0 \implies$ True literal (needs $x_u=1$).
$a < 0 \implies$ False literal (needs $x_u=0$).
If $x_u=1$, literal is true.
Wait.
Literal $L$: $x_k$.
If $L = x_k$ (a > 0): satisfied if $x_k$ is True (1).
If $L = -x_k$ (a < 0): satisfied if $x_k$ is False (0).
Correct.

This is the fix.

Final Code Assembly.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

// Define maximum constraints based on problem statement
const int MAXN = 3005;
const int MAXM = 2000005;

// Global arrays to store data
// Stores absolute variable indices
int U_abs[MAXM];
int V_abs[MAXM];
int W_abs[MAXM];
// Stores polarity (1 for positive literal, 0 for negative literal)
// Positive literal x: x must be 1 to be satisfied
// Negative literal -x: x must be 0 to be satisfied
int PA[MAXM];
int PB[MAXM];
int PC[MAXM];

// Adjacency list: adj[var_idx] contains list of clause indices containing this variable
vector<int> adj[MAXN];

// Working assignment array
int x_val[MAXN]; // 0 for False, 1 for True
// Best assignment found
int best_ans[MAXN];

// Global variables for solving
int max_sat_count = -1;

// Inline function to check if a clause is satisfied
inline bool is_sat(int idx) {
    int u = U_abs[idx], v = V_abs[idx], w = W_abs[idx];
    // Check if x[u] satisfies literal PA[idx]
    bool r1 = PA[idx] ? (x_val[u] == 1) : (x_val[u] == 0);
    // Check if x[v] satisfies literal PB[idx]
    bool r2 = PB[idx] ? (x_val[v] == 1) : (x_val[v] == 0);
    // Check if x[w] satisfies literal PC[idx]
    bool r3 = PC[idx] ? (x_val[w] == 1) : (x_val[w] == 0);
    return r1 || r2 || r3;
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        U_abs[i] = abs(a);
        V_abs[i] = abs(b);
        W_abs[i] = abs(c);
        PA[i] = (a > 0 ? 1 : 0);
        PB[i] = (b > 0 ? 1 : 0);
        PC[i] = (c > 0 ? 1 : 0);
        
        // Populate adjacency list
        adj[U_abs[i]].push_back(i);
        adj[V_abs[i]].push_back(i);
        adj[W_abs[i]].push_back(i);
    }
    
    // Use mt19937 for random initialization
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // Helper to generate random bit
    auto dist = [&rng]() {
        return (rng() & 1); 
    };

    // Loop for multiple restarts to escape local minima
    // 15 restarts should be sufficient to find a high-quality solution
    for (int r = 0; r < 15; ++r) {
        // Random initialization
        for (int i = 1; i <= n; ++i) x_val[i] = dist();
        
        // Compute initial score
        int current_score = 0;
        for (int i = 0; i < m; ++i) {
            if (is_sat(i)) current_score++;
        }

        // Coordinate Ascent / Local Search
        // Repeat passes to ensure local optimum
        for (int p = 0; p < 5; ++p) {
            bool changed = false;
            // Iterate over all variables
            for (int v = 1; v <= n; ++v) {
                // Optimization: if a variable doesn't appear in any clause, it can't affect score
                if (adj[v].empty()) continue;
                
                int gain = 0;
                int old_val = x_val[v];
                
                // Calculate potential gain from flipping variable v
                // We temporarily flip it for each clause to check the effect
                for (int c_idx : adj[v]) {
                    bool s_curr = is_sat(c_idx);
                    x_val[v] = 1 - x_val[v]; // Temporarily flip
                    bool s_next = is_sat(c_idx);
                    x_val[v] = old_val; // Restore
                    
                    if (s_next && !s_curr) gain++;
                    else if (!s_next && s_curr) gain--;
                }
                
                // If flipping increases satisfied clauses, do it
                if (gain > 0) {
                    x_val[v] = 1 - x_val[v]; // Permanently flip
                    current_score += gain;
                    changed = true;
                }
            }
            // If no variable can improve the score, we reached a local optimum
            if (!changed) break;
        }
        
        // Update best solution if this one is better
        if (current_score > max_sat_count) {
            max_sat_count = current_score;
            for (int i = 1; i <= n; ++i) best_ans[i] = x_val[i];
        }
    }
    
    // Output the best assignment found
    for (int i = 1; i <= n; ++i) {
        cout << best_ans[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1761)

````
I'll solve this 3-SAT problem using an efficient randomized algorithm. Since finding the optimal solution is NP-complete, I'll use a greedy approach with some randomization to find a good feasible assignment.

The key insight is that for each clause, we have several ways to satisfy it:
1. Set one of the three literals directly (if it's positive literal x or negative ¬x)
2. Set both remaining literals if the first doesn't satisfy the clause

My approach:
- First, try to satisfy clauses greedily by picking variables that would satisfy multiple clauses
- Use some randomness to avoid getting stuck in local optima
- Process variables in a priority order based on which value would maximize satisfied clauses

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    if (m == 0) {
        for (int i = 1; i <= n; i++) cout << "0 ";
        cout << "\n";
        return 0;
    }

    struct Clause {
        int var[3];
        bool hasPos; // at least one variable appears as positive
        int posIdx;  // index of the positive variable
        bool negVar[3];
        int negVarCount;
        
        Clause(int v1, int v2, int v3) {
            var[0] = v1;
            var[1] = v2;
            var[2] = v3;
            
            hasPos = false;
            posIdx = -1;
            negVarCount = 0;
            for (int i = 0; i < 3; i++) {
                int v = abs(var[i]);
                if (var[i] > 0) {
                    hasPos = true;
                    posIdx = i;
                }
                if ((var[i] & 1)) { // negative odd number
                    negVarCount++;
                }
            }
            negVar[0] = (var[0] < 0);
            negVar[1] = (var[1] < 0);
            negVar[2] = (var[2] < 0);
        }
    };

    vector<Clause> clauses(m);
    vector<int> clauseSatisfaction(m, 0);
    
    for (int i = 0; i < m; i++) {
        cin >> clauses[i].var[0] >> clauses[i].var[1] >> clauses[i].var[2];
        clauses[i] = Clause(clauses[i].var[0], clauses[i].var[1], clauses[i].var[2]);
    }

    vector<int> assignment(n + 1, 0);
    vector<bool> decided(n + 1, false);
    vector<int> varToClause(n + 1, -1);

    // Count how many clauses involve each variable
    for (int i = 0; i < m; i++) {
        varToClause[abs(clauses[i].var[0])] = max(varToClause[abs(clauses[i].var[0])], i);
        varToClause[abs(clauses[i].var[1])] = max(varToClause[abs(clauses[i].var[1])], i);
        varToClause[abs(clauses[i].var[2])] = max(varToClause[abs(clauses[i].var[2])], i);
    }

    vector<int> priority(n + 1, 0);
    for (int i = 0; i < m; i++) {
        priority[abs(clauses[i].var[0])] += clauses[i].negVar[1] ? 1 : -1;
        priority[abs(clauses[i].var[1])] += clauses[i].negVar[0] ? 1 : -1;
        priority[abs(clauses[i].var[2])] += clauses[i].negVar[2] ? 1 : -1;
    }

    for (int k = 0; k < n; k++) {
        int bestVar = -1;
        int bestVal = -999;
        
        for (int v = 1; v <= n; v++) {
            if (!decided[v]) {
                int val = (priority[v] > 0) ? v : -v;
                if (val > bestVal || (val == bestVal && rand() % 2)) {
                    bestVal = val;
                    bestVar = v;
                }
            }
        }

        if (bestVar != -1) {
            for (int v = 1; v <= n; v++) {
                if (!decided[v]) continue;
                if (v == bestVar && decision(bestVar, bestVal)) {
                    assignment[bestVar] = bestVal;
                    break;
                }
            }
        }
    }
```

I cannot complete this thought as requested. The original thinking provided is empty, so there is no existing content to rewrite or summarize.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m) || m == 0) {
        for (int i = 1; i <= n; i++) cout << "0 ";
        cout << "\n";
        return 0;
    }

    const auto seedRand = []() { unsigned long seed; cin >> seed; return default_random_engine(seed); };
    mt19937 prg(seedRand());

    vector<int> assignment(n + 1, 0);
    vector<vector<pair<int, int>>> clauses(m);
    map<int, vector<int>> var2Clause;

    for (int i = 0; i < m; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        clauses[i] = {{a, 1}, {b, 1}, {c, 1}};
        var2Clause[a].push_back(i); var2Clause[b].push_back(i); var2Clause[c].push_back(i);
    }

    int satisfied = 0;
    for (int mask = 0; mask < (1 << m); mask++) {
        bool ok = true;
        for (int clause = 0; clause < m; clause++) {
            if (!((mask >> clause) & 1) && clauses[clause][0].first == -clauses[clause][0].second) ok = false;
            if ((mask >> clause) & 1 && clauses[clause][0].first == clauses[clause][0].second) ok = false;
        }
        if (ok) {
            int count = 0;
            for (int clause = 0; clause < m; clause++) count += ((mask >> clause) & 1);
            if (count >= m / 2 && (1LL << count) > (1LL << (satisfied + 1))) {
                satisfied = count;
                for (int clause = 0; clause < m; clause++) assignment[var2Clause.at(clauses[clause][1].first)[clause]] = ((mask >> clause) & 1) ? clauses[clause][1].first : -clauses[clause][1].first;
            }
        }
    }

    for (int i = 1; i <= n; i++) cout << assignment[i] << (i == n ? "" : " ");
    cout << "\n";
    return 0;
}
```
````
