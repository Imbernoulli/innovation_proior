# 题 176 — base均分 83.8 vs 直接SFT均分 0.0(掉点 83.8)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 97.5,completion_tokens=20904)

````
The user wants a C++ solution for the "3-SAT" problem.
This is not asking to solve the NP-complete SAT problem perfectly (which would be impossible in general polynomial time, and likely TLE given constraints), but rather to satisfy as many clauses as possible.
Actually, looking at the problem description: "Your task is to assign truth values to variables to satisfy as many clauses as possible."
However, usually competitive programming problems like this might have specific constraints or hints about randomized approaches or greedy strategies if exact solving is too hard.
Constraints: $n \le 1000$, $m \le 10000$.
A brute force search is $2^n$, which is $2^{1000}$, definitely impossible.
Since it's 3-SAT (maximizing satisfied clauses), this is MAX-3SAT. This is NP-hard.
For small inputs ($N=1000$), we can't do exponential time. We need an approximation algorithm or a heuristic.
Common heuristics for Max-SAT/Max-CSP include:
1. Greedy: Flip variable that increases satisfaction most.
2. Randomized Local Search: Start random, keep flipping if better.
3. Simple assignment based on majority of clauses involving variable.

Given the scoring system: "Any assignment is accepted and scored", even if score is low. But usually, to get a good score, one needs a decent approximation.
Since $N$ and $M$ are up to 1000 and 10000, $O(N^2)$ or $O(M \cdot N)$ is acceptable.
Let's consider a simple greedy strategy or local search.
Local Search (Gibbs Sampling style or simple hill climbing) is very effective for Max-SAT on these scales.
Algorithm:
1. Assign random truth values to all variables.
2. Count how many clauses are satisfied.
3. If there's a way to increase the count by flipping one variable, do it. Or maybe pick the variable whose flip maximizes the gain and does it.
4. Repeat until convergence or a time limit.

Since we need to maximize satisfied clauses, let's try a randomized local search approach with restarts.
Or simpler: For each variable, calculate its contribution.
But interactions make it complex.
Let's implement a standard local search:
Initialize variables randomly.
Evaluate total satisfied clauses.
Iterate:
   Find a variable $x_i$ such that setting $x_i$ to $\neg x_i$ increases (or decreases loss) satisfied clauses the most.
   Update $x_i$.
   Repeat until no improvement or time expires.

Actually, simply picking a random assignment is sometimes enough for easy instances, but for "competitive programmer" context, we should aim for higher quality than random.
Let's refine the heuristic:
Total clauses = m.
For each variable, check how many clauses contain it positively and negatively.
This is static analysis.
Dynamic adjustment:
Iterate through variables $i = 1$ to $n$.
If flipping $x_i$ improves the number of satisfied clauses among the ones involving $x_i$, then flip it? No, because changing $x_i$ affects other variables' contributions.
Local Search:
Current assignment $A$.
Score $S(A)$.
While changed:
   Best variable to flip?
   Actually, just trying to find *any* improving move is often sufficient.
   Better yet: Simulated Annealing or just pure Hill Climbing.
   Given $N=1000$, calculating the effect of flipping one variable takes $O(M)$. Doing this for all $N$ takes $O(N \cdot M)$. With $10^7$ ops, that's fast enough (limit is usually 1-2 seconds).
   
   So, Loop:
     best_gain = 0
     change_idx = -1
     current_satisfied = calculate()
     
     For i from 1 to n:
        // Calculate gain from flipping variable i
        // Gain = (Clauses where i is false and becomes true) - (Clauses where i is true and becomes false)
        // Note: Clause structure is 3 literals. If clause contains x and other literals are already false, flipping x helps.
        // If clause contains x and other literals are true, x doesn't matter.
        
        // Effect of flipping i:
        // Identify all clauses containing literal +i or -i.
        // For clause c containing +i: if other two are both false, this clause is unsatisfied. Flipping i makes it satisfied. (+1)
        //                     if other two are not both false, it was already satisfied. (-1) -- wait, if satisfied before flip, it stays satisfied. If it becomes unsatisfied? Impossible because i changes True->False, so it loses support.
        // Let's rephrase.
        // Contribution of variable i to a specific clause depends on other literals.
        // Let's precompute for each variable, the list of clauses it participates in.
        
        int gain = 0;
        for each clause k involving variable i:
           Check if clause k is currently satisfied without considering i's value (value of others).
           Wait, easier logic:
           Evaluate clause k status BEFORE flip.
           Evaluate clause k status AFTER flip.
           delta = after - before.
           sum delta over all k involving i.
           
           Wait, evaluating status takes time. We need efficient update.
           But since $O(NM)$ per outer loop is fine for limited iterations.
           
     To speed up, maintain current status of all clauses.
     When flipping i, only update the clauses involving i.
     
     Outer loop runs say 1000 times or until stable.
     Total complexity roughly $T_{iter} \times N \times (\text{avg clauses per var})$.
     Sum of clauses per var is $M$. So one pass over all vars is $O(M)$.
     So we do 1000 passes. Total $1000 \times 10000 = 10^7$. Very safe.
     
     Strategy:
     1. Parse input. Store adjacency list `adj[var] -> {clause_index}`. Need to handle negative indices correctly (abs(x)).
     2. Initialize random bool array `val[1..n]`.
     3. Compute initial satisfied count.
     4. Loop for e.g., 5000 iterations:
          max_delta = 0
          best_v = -1
          
          For v from 1 to n:
             delta = 0
             For each clause_id in adj[v]:
                Determine contribution of v to clause_id.
                // Clause literals L1, L2, L3
                // Status depends on val[L1], val[L2], val[L3].
                // If we flip v, new_val[v] = !val[v].
                // Re-evaluate clause status with old val vs new val.
                
                // Optimization: Pre-store literals involved in clause for fast access.
                // Or even simpler: Since m is small (10000), we can just store the clauses explicitly.
                // `vector<int> clauses[m]`, store indices.
                // `vector<pair<bool, int>>` clauses.
                
                // Let's stick to O(M) scan per iteration. It's clean and fast enough.
             
             If delta > 0: // Only interested in increasing satisfaction
                 max_delta = delta
                 best_v = v
            
            if max_delta > 0:
                 flip(best_v)
                 // Break or continue? 
                 // Greedy: flip one with max delta. Restarting sweep?
                 // Just do one pass finding best, flip it. Repeat sweeps.
                 
     To avoid getting stuck in local minima, we can run multiple restarts or allow occasional "bad" moves (Simulated Annealing).
     Given "Any assignment is accepted", getting a high score isn't strictly required but better is safer.
     Usually for Max-SAT on random instances, hill climbing works well.
     
     Let's refine the inner loop.
     Calculating delta:
       For a variable $v$ and a clause $c$:
       Current satisfaction: $s_c(v)$.
       Next satisfaction: $s_c(\neg v)$.
       $s_c(l_1, l_2, l_3)$ returns 1 if any $l_k$ is true.
       
       Actually, checking delta for all clauses involves iterating clauses containing $v$.
       Let's use vector of vectors `clauses[n+1]`. Each element stores a reference to the clause data.
       Structure: `struct Clause { int id; int l1, l2, l3; };`
       Wait, we iterate m clauses to compute total score initially $O(M)$.
       Then for each variable $v$, we iterate its clauses. Sum of sizes is $M$. So computing deltas for all variables is $O(M)$.
       
       Implementation detail:
       Use 1-based indexing for variables as per problem ($1 \dots n$). Input gives integers in range $[-n, n]$. Map to canonical index $1..n$ and sign.
       Array `bool assigned[n+1]`.
       Vector `vector<int> var_clauses[n+1]` storing clause indices.
       Array `int clause_lits[m][3]` storing literals.
       
       Initial Score:
         int current_satisfied = 0;
         for j in 0..m-1: check(j);
         
       Improvement:
         For iter 1 to 2000:
           best_v = -1;
           best_delta = 0;
           for v in 1..n:
              delta = 0;
              for c_idx in var_clauses[v]:
                 // check contribution
                 // Old status
                 if (check_clause(c_idx)) delta++; // actually wait, checking status is O(1).
                 // New status
                 // Logic: The clause becomes satisfied if flipped. Otherwise it remains as is.
                 // If it was satisfied and stays satisfied, delta += 0.
                 // If it was unsatisfied and becomes satisfied, delta += 1.
                 // If it was satisfied and becomes unsatisfied (loss), delta -= 1.
                 // Actually, we can just calculate `is_satisfied(old_vals)` and `is_satisfied(new_vals)`.
                 // If `new` is true and `old` is false -> +1.
                 // If `new` is false and `old` is true -> -1.
                 
                 // Optimization: The clause is satisfied if any of 3 is true.
                 // If we flip $v$, only literals equal to $\pm v$ change value.
                 // If the clause has exactly one occurrence of $v$ (e.g. +v):
                 //    old: val(+v)=b. If b=1, sat. If b=0, check others.
                 //    new: val(-v)=b. If !b=1 (means b=0), sat. If !b=0 (means b=1), check others.
                 //    Basically, check if (others_are_true) changes.
                 //    Others are true OR $v$ is true => Satisfied.
                 //    Let $R$ be result of other 2 literals.
                 //    If $v$ is +: Sat if R || (!v_sign * val[v]).
                 //    Wait, simpler:
                 //    Literal is true if matches assigned value.
                 //    Before flip: Lit contributes if (lit == val).
                 //    After flip: Lit contributes if (lit == !val).
                 
                 // Let's write a helper `count_false(others)`.
                 // A clause has literals $L_a, L_b, L_c$.
                 // Variable $v$ appears in it.
                 // Let `is_LiteralTrue(lit, val)` be (sign(lit) * val[lit_idx]) > 0.
                 // Current Sat = isTrue(La) | isTrue(Lb) | isTrue(Lc).
                 // If we flip val[v], signs stay same, values invert.
                 // Recalculate.
              
              if delta > best_delta:
                  best_delta = delta
                  best_v = v
           
           if best_delta > 0:
              flip_val(best_v); // Update global state.
              // Update current_satisfied
              // We know delta from calculation, so new_score += best_delta.
              // Wait, delta calculated earlier used current values. Yes.
           else:
               break; // No improvement found.
           
     Wait, what if `delta` < 0 but flipping helps later? Hill climbing gets stuck.
     However, for 3-SAT, usually finding *an* improving neighbor is good.
     Maybe add a temperature (Simulated Annealing)?
     Or simply restart if no improvement for X steps.
     Or just run for fixed large number of steps and flip random variable with some probability?
     Given the problem asks for "as many clauses as possible", we want the maximum.
     For $N=1000$, we can afford maybe 5-10 seconds locally? No, online judge limits usually 1-2s.
     But standard greedy with local search works decently.
     Let's stick to deterministic hill climbing with multiple restarts.
     Run 10 random seeds. Pick best result.
     Within each seed, run hill climb for ~1000 iterations.
     Since each iteration takes $O(N \cdot (\text{avg degree}))$, avg degree is $2M/N$.
     Actually $O(M)$ total per full pass over variables.
     So 1000 iterations takes $1000 \times M \approx 10^7$ ops.
     With 10 restarts, $10^8$ ops. Might be slightly tight for 1 sec but typically $10^8$ simple ops (integers) is OK in C++.
     Can optimize the inner loop.
     
     Optimization:
     Instead of re-checking the whole clause for every variable, precalculate impact.
     For each variable $v$, store `gain[v]`.
     `gain[v]` tracks net change if we flip $v$.
     Initially `gain[v] = 0`.
     We compute actual delta once per pass.
     
     Wait, there is a faster heuristic:
     Assign variables such that they appear most frequently positive/negatively?
     Or random shuffle.
     Let's stick to the randomized local search. It is robust.
     Since I cannot test execution time, I should make the inner loop as efficient as possible.
     Use `std::vector` carefully.
     Since $N, M$ are small enough, we don't need `bitset`.
     
     Data structures:
     `vector<vector<int>> var_clauses(n + 1)`
     Inside `var_clauses[v]`: store `{clause_index, literal_type}`
     `literal_type` indicates whether clause has +v or -v.
     Actually, we need to know the other two literals to evaluate them efficiently?
     Or just store the raw clause content in a flat array and look it up?
     Accessing raw array is cache friendly.
     `int lits[m][3];`
     
     Evaluation function inside loop:
     ```cpp
     int eval(int clause_idx) {
         bool s1 = (lits[clause_idx][0] > 0 && val[lits[clause_idx][0]] > 0);
         // etc
     }
     ```
     Wait, `val` is bool or int (0/1).
     Literals are signed.
     Helper `get_value(int lit, int val[])`: `if (lit > 0) return val[lit]; else return !val[-lit];`
     Then `sat = get(l1) || get(l2) || get(l3)`.
     
     To avoid recalculating for the same clause repeatedly in one pass?
     We iterate $v=1..n$. For each $v$, we touch all clauses in `var_clauses[v]`.
     Total touches = $M$ per pass.
     In each touch, we evaluate the clause twice (current and flipped).
     Evaluating clause is constant time.
     So cost per pass = $2 \cdot M \cdot C$.
     Passes = 2000. Total $4 \cdot 10^7$. Very safe.
     Multi-restart x 10 -> $4 \cdot 10^8$. Might be pushing it for strict 1s limit.
     Reduce passes to 500 per start.
     And reduce starts to 5.
     Actually, one strong start with simulated annealing or longer hill climb might suffice.
     Let's try 2000 passes with one restart first? No, hill climbing often fails early.
     Let's use Multiple Restarts (simply reset and pick best final score).
     Number of restarts = 3. Iterations per restart = 2000.
     Complexity: $3 \times 2000 \times 10000 = 6 \times 10^7$. This is absolutely fine.
     
     Corner cases: $m=0$. Output all zeros (or anything).
     Inputs: $n=1000, m=10000$.
     
     Input format details:
     Line 1: n m
     Lines: a b c
     Integers can be negative.
     
     One more thing: The output format requires $x_1 \dots x_n$.
     $1$ for True, $0$ for False.
     
     Let's write the code.
     
```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Global structures to store problem
struct Problem {
    int n, m;
    vector<tuple<int, int, int>> clauses;
    vector<vector<int>> var_clauses; // var_clauses[i] contains list of clause indices involving variable i
    // Also need to store literal type for quick lookup? 
    // We can just extract from clause tuple.
};

// Returns true if literal 'lit' (signed integer) evaluates to true given assignment 'assign' (bool array 1..n)
bool is_true(int lit, const vector<bool>& assign, int n) {
    int var_idx = abs(lit);
    if (var_idx >= 1 && var_idx <= n) {
        return lit > 0 ? assign[var_idx - 1] : !assign[var_idx - 1];
    }
    // Should not happen based on constraints
    return false; 
}

// Check if clause is satisfied
bool is_satisfied(const Problem& p, int idx, const vector<bool>& assign, int n) {
    const auto& [l1, l2, l3] = p.clauses[idx];
    return is_true(l1, assign, n) || is_true(l2, assign, n) || is_true(l3, assign, n);
}

int calculate_score(const Problem& p, const vector<bool>& assign) {
    int satisfied = 0;
    for (size_t i = 0; i < p.clauses.size(); ++i) {
        if (is_satisfied(p, i, assign, p.n)) {
            satisfied++;
        }
    }
    return satisfied;
}

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    Problem prob;
    prob.n = n;
    prob.m = m;
    prob.clauses.resize(m);
    prob.var_clauses.resize(n + 1);

    for (int i = 0; i < m; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        prob.clauses[i] = {a, b, c};
        // Add to var_clauses for each literal
        if (a != 0) prob.var_clauses[abs(a)].push_back(i);
        if (b != 0) prob.var_clauses[abs(b)].push_back(i);
        if (c != 0) prob.var_clauses[abs(c)].push_back(i);
    }

    // Solver strategy: Randomized Local Search / Hill Climbing with Multiple Restarts
    vector<bool> best_assign(n, false); // Default 0
    
    // Try different seeds to escape local optima
    int best_score = -1;
    
    // Seed random
    srand(time(NULL)); 

    for (int restart = 0; restart < 10; ++restart) {
        vector<bool> current_assign(n, rand() % 2 == 0);
        int current_satisfaction = calculate_score(prob, current_assign);

        // Hill Climbing Phase
        // Iterate a certain number of times
        for (int iter = 0; iter < 2000; ++iter) {
            int max_delta = 0;
            int best_var = -1;
            
            // Analyze flipping each variable
            for (int v = 1; v <= n; ++v) {
                int delta = 0;
                
                // Calculate change in satisfied clauses if v flips
                for (int cid : prob.var_clauses[v]) {
                    // Get literals
                    auto cl = prob.clauses[cid];
                    
                    // Check current satisfaction
                    // Optimize: instead of full function call overhead, inline logic
                    
                    // Check if clause currently satisfied
                    bool cur_sat = false;
                    int cnt_lit_true = 0;
                    // Manually checking literals
                    auto get_true = [&](int lit) {
                        return lit > 0 ? current_assign[abs(lit) - 1] : !current_assign[abs(lit) - 1];
                    };
                    // To prevent lambda creation overhead in loop
                    // Just write out checks for 3 literals
                    
                    // Actually, creating closure inside loop is bad. 
                    // Access directly.
                    
                    // Let's rely on `current_assign` directly.
                    bool l1 = (cl.get<0>() > 0) ? current_assign[abs(cl.get<0>()) - 1] : !current_assign[abs(cl.get<0>()) - 1];
                    bool l2 = (cl.get<1>() > 0) ? current_assign[abs(cl.get<1>()) - 1] : !current_assign[abs(cl.get<1>()) - 1];
                    bool l3 = (cl.get<2>() > 0) ? current_assign[abs(cl.get<2>()) - 1] : !current_assign[abs(cl.get<2>()) - 1];
                    
                    cur_sat = l1 || l2 || l3;
                    
                    // What if we flip v?
                    // New value of v is !current_assign[v-1]
                    // We need to check literals again.
                    
                    // If v corresponds to literal li in clause...
                    // Case 1: literal in clause matches v (e.g. +v is literal, current val is true)
                    //   Then flipping v makes this literal false.
                    //   Case 2: literal is -v (matches v inverted, so val true implies -v false)
                    //     Flipping v makes -v true (since v becomes false).
                    
                    // Efficient check:
                    // Let val_v = current_assign[v-1].
                    // The literal l in clause equals +v or -v.
                    // If literal == +v: current_contrib = val_v. new_contrib = !val_v.
                    // If literal == -v: current_contrib = !val_v. new_contrib = !( !val_v ) = val_v.
                    
                    // The clause status is determined by the 3 literals.
                    // If we flip v, the clause status might change.
                    // It changes if and only if ONLY the literal containing v determines satisfaction?
                    // Or rather, it changes if:
                    //   (Before flip: v-lit was key and kept it sat) vs (After flip: v-lit made it unsat)
                    //   Or (Before flip: v-lit was false and needed fixing)
                    
                    // Simpler:
                    // Check if clause becomes satisfied AFTER flip that wasn't before.
                    // Check if clause becomes unsatisfied AFTER flip that was before.
                    
                    // Let's create temporary assignments.
                    // But modifying global state and backtracking is slow if done blindly.
                    // We just recalculate boolean status.
                    
                    // Let's implement helper to quickly re-eval.
                    int lit_val_l1 = (cl.get<0>() > 0) ? current_assign[abs(cl.get<0>()) - 1] : !current_assign[abs(cl.get<0>()) - 1];
                    int lit_val_l2 = (cl.get<1>() > 0) ? current_assign[abs(cl.get<1>()) - 1] : !current_assign[abs(cl.get<1>()) - 1];
                    int lit_val_l3 = (cl.get<2>() > 0) ? current_assign[abs(cl.get<2>()) - 1] : !current_assign[abs(cl.get<2>()) - 1];
                    
                    bool sat_now = lit_val_l1 || lit_val_l2 || lit_val_l3;
                    
                    int target_var = v;
                    // Find which literal is target_var
                    bool affected_by_flipped = false;
                    // We can determine this easily.
                    // If cl.get<0>() == target_var, then literal 0 changes.
                    // ...
                    // Actually, we can pre-check.
                    
                    // Wait, the cost of checking `cl.get<0>()` is small.
                    // Let's write clean logic.
                    
                    int l1_val = (cl.get<0>() == target_var) ? (1 - current_assign[target_var-1]) : 
                                            ((cl.get<0>() == -target_var) ? (current_assign[target_var-1]) : current_assign[abs(cl.get<0>()) - 1]);
                    // Wait, logic error.
                    // If literal is +v: val is current_assign[v-1].
                    // If literal is -v: val is !current_assign[v-1].
                    // If literal is NOT +/-v: val is ... computed as normal.
                    
                    // Let's just re-evaluate. It's safer and fast enough.
                    // BUT inside the loop of 2000 * 10000, doing this check 3 times is costly?
                    // No, it's just integer ops.
                    
                    // Re-calculation:
                    // Create a modified copy of assignment? No.
                    
                    // Function `evaluate_clause(idx, assign)`
                    // Inside the loop:
                    // We modify `current_assign` temporarily? No, we want delta.
                    
                    // Okay, let's trust the math.
                    // Current saturation for clause c: S_c
                    // New saturation if v flipped: S'_c
                    // Delta = S'_c - S_c
                    
                    // We can write a macro or just repeated lines to avoid function call overhead?
                    // Inline code is fastest.
                    
                    // Optimization:
                    // Only care about clauses containing `v`. `var_clauses[v]` holds these indices.
                    // Inside:
                    // bool cur_l1, cur_l2, cur_l3;
                    // if (cl.l1 == v) cur_l1 = !assign[v-1]; else if (cl.l1 == -v) cur_l1 = assign[v-1]; else cur_l1 = (cl.l1 > 0) ? assign[abs(cl.l1)-1] : !assign[abs(cl.l1)-1];
                    // Wait, `cur_l1` IS the value of literal L1 under CURRENT assignment.
                    // To get NEW value:
                    // if (cl.l1 == v) new_l1 = assign[v-1]; (Wait, if v flips, 1->0. If v was true (assign 1), new is 0. So !)
                    
                    // Let's simplify:
                    // Variable v is at index `v`. Its current value is `val[v-1]`.
                    // If we flip v, new value is `!val[v-1]`.
                    // Literal `lit` evaluates to true if `(lit > 0 && val[lit_idx] == 1) || (lit < 0 && val[lit_idx] == 0)`.
                    // This is equivalent to `lit == val[lit_idx]` (if we treat val as +1/-1 boolean).
                    // Let's use 0/1.
                    // val is 0 or 1.
                    // lit is +k or -k.
                    // Val of lit: `(lit > 0 ? val[k-1] : !val[k-1])`.
                    // If we flip v, `val` changes to `1 - val`.
                    
                    // So:
                    int val_v = current_assign[v-1];
                    int new_val_v = 1 - val_v;
                    
                    bool sat_curr = false;
                    bool sat_new = false;
                    
                    // Clause literals processing
                    // We need to handle three literals.
                    // To minimize branching inside loop (speed), assume literals are accessed.
                    // Since we only iterate clauses involving v, we expect v is present in clause.
                    // But we must still check the others.
                    
                    // Let's unroll for clarity.
                    
                    int l1 = get<0>(cl);
                    bool v_is_in_l1 = (l1 == v || l1 == -v);
                    if (v_is_in_l1) {
                        // Check others first? No.
                        // We just need to know if l1 is true now or will be.
                        bool l1_val = (l1 > 0) ? current_assign[abs(l1) - 1] : !current_assign[abs(l1) - 1];
                        bool l1_new_val = (l1 > 0) ? new_val_v : !new_val_v;
                        
                        bool l2_val = (l2 = get<1>(cl)) > 0 ? current_assign[abs(l2) - 1] : !current_assign[abs(l2) - 1];
                        bool l3_val = (l3 = get<2>(cl)) > 0 ? current_assign[abs(l3) - 1] : !current_assign[abs(l3) - 1];
                        
                        // Check saturation
                        sat_curr = l1_val || l2_val || l3_val;
                        
                        // Wait, above `l1_val` uses `val_v` implicitly? No, if l1!=v/l2/l3, it uses direct `current_assign`.
                        // Correct.
                        
                        bool l1_new = (l1 > 0) ? new_val_v : !new_val_v;
                        bool l2_new = (l2 = get<1>(cl)) > 0 ? new_val_v : !new_val_v; // WAIT, only if l2==v!
                        // My previous logic had `l1` being v.
                        // We need to be careful.
                        
                        // Correct logic:
                        // For each literal in clause:
                        // Check if it involves v. If yes, use new_val_v. Else use old values.
                        // Check old saturation.
                        // Check new saturation.
                        
                        // Re-write cleaner:
                        bool sat_old = false;
                        bool sat_new = false;
                        
                        // Process l1
                        if (l1 == v) { sat_old |= new_val_v; sat_new |= new_val_v; } // Error logic here.
                        // If l1 == v:
                        //   Old val = val_v.
                        //   New val = new_val_v.
                        
                        // Let's write explicit if blocks.
                        
                        // Clause literals stored in `cl`
                        // We need to distinguish whether a literal involves v.
                        // We know `v` is one of them.
                        
                        // Let's assume `get<0>` is +v or -v or something else.
                        // We can identify `is_v_involved(lit)` as `abs(lit) == v`.
                        
                        // If `is_v_involved`:
                        //   val_contribution is either `val_v` (if +) or `!val_v` (if -).
                        //   We calculate `sat` using `val_v` and `new_val_v`.
                        // Else:
                        //   `val_contribution` is constant.
                        
                        // Actually, simplest is to evaluate the clause twice.
                        // Copying val_v is not needed if we handle conditional access.
                    }
                    
                    // Let's go with the `val_v` variable optimization.
                    // We compute `sat_old` and `sat_new` using `current_assign` and modified check.
                    
                    // To optimize: `vector<bool>` access is slightly slow. `vector<char>` is better.
                    
                    // Final simplified logic inside loop:
                    /*
                    int l1 = cl[0]; // Assuming we stored as struct/tuple or separate array
                    // Check old
                    // Check new
                    */
                    
                    // Given time constraint, let's just use `std::tuple` and manual expansion if needed.
                    // But accessing `get<0>` is a bit heavy.
                    // Better to use a struct.
                    }
                }
            }
        }
    }
}
```
Refining the implementation details:
- Use `std::array<int, 3>` for clauses.
- Use `vector<char>` for assignment to speed up (avoid dynamic allocation of vector<bool> bits?). `vector<bool>` is specialized and might be slower/faster depending. `vector<uint8_t>` is usually safest for speed.
- Fast IO (`cin.tie(NULL)`).
- Logic for `delta`:
  Iterate `cid` in `var_clauses[v]`.
  Extract `l1, l2, l3`.
  `old_sat = (lit_val(l1) || lit_val(l2) || lit_val(l3))`
  `new_sat = (lit_val_new(l1) || lit_val_new(l2) || lit_val_new(l3))`
  `delta += (new_sat ? 1 : 0) - (old_sat ? 1 : 0)`
  
  Here `lit_val(l)` uses `assign`. `lit_val_new(l)` uses `modified_assign`.
  We can share code.
  
  `val_v = assign[v-1]` (0 or 1).
  `other_vals` remain same.
  
  To avoid branching inside loop, notice that if a literal is NOT related to `v`, its value doesn't change.
  So:
  `sat_old` calculation:
    `v_in_l1 = (abs(l1) == v)`
    `l1_val = (l1 > 0 ? val_v : !val_v)` if `v_in_l1` else `(l1 > 0 ? assign[abs(l1)-1] : !assign[abs(l1)-1])`
    Repeat for others.
  
  If we optimize:
  Notice that `lit_val_new` differs from `lit_val` ONLY for the literal corresponding to `v`.
  If `l1 == v`: `l1_val_old = val_v`, `l1_val_new = !val_v`.
  If `l1 == -v`: `l1_val_old = !val_v`, `l1_val_new = val_v`.
  If `l1 != +/-v`: `l1_val_old = l1_val_new = fixed_val`.
  
  So:
  `bool l1_val_fixed = (l1 != v && l1 != -v) ? (l1 > 0 ? assign[abs(l1)-1] : !assign[abs(l1)-1]) : 0;`
  `bool l2_val_fixed = ...`
  `bool l3_val_fixed = ...`
  
  `sat_old = (l1_val_fixed || l2_val_fixed || l3_val_fixed) ? (1 if v not involved in satisfying condition else val_v?)` -> Too complicated.
  
  Let's just use two evaluations. It's correct and with `O(M)` per iteration, $M=10000$, iterations=2000, total ops $\approx 2 \cdot 10^7$. Even with overhead, C++ handles $\approx 10^8$ ops/sec. 3-4 restarts x 2000 is safe.
  
  Let's structure the code clearly.
  
  Code plan:
  1. Include headers.
  2. Fast IO.
  3. Read Input.
  4. Build `adj` list.
  5. Store literals in `vector<array<int,3>>`.
  6. Main loop over restarts.
  7. Inside: Init random. Calc score.
  8. While improved: Find best flip.
  9. Update global best.
  
  Wait, "While improved" might loop indefinitely if we don't bound iterations. Bound by time or steps.
  Using a fixed number of steps (e.g. 5000) per restart is safer against infinite loops on non-improving plateaus (though step 8 usually breaks if no improvement found).
  Step 8 logic: `best_flip = -1`. If no var improves, break.
  
  One detail: `val_v` logic.
  If `abs(lit) == v`:
     If `lit == v`, it contributes `val_v`. New contribution `!val_v`.
     If `lit == -v`, it contributes `!val_v`. New contribution `val_v`.
  
  Let's define `get_val(lit, assign, flip)`:
    if `flip`:
      if `abs(lit) == v`:
        return `(lit == v) ? !assign[v-1] : assign[v-1]`
      else:
        return `(lit > 0) ? assign[abs(lit)-1] : !assign[abs(lit)-1]`
    else:
      return `(lit > 0) ? assign[abs(lit)-1] : !assign[abs(lit)-1]`
      
  Inside loop, call `get_val` twice for each clause in `var_clauses[v]`.
  Actually, simpler:
  Calculate `cnt_sat_old` (already computed for all clauses? No, we need delta).
  Just calculate delta on the fly.
  
  Wait, if we do `while(true)` and no improvement, it terminates.
  However, we need to ensure it terminates. The number of states is finite ($2^N$), so it terminates eventually. But we can't iterate all.
  Local search converges quickly.
  
  Scoring:
  "If m > 0: score = s / m".
  We just need to maximize s.
  
  Corner case: $n=0$? $n \ge 1$.
  $m=0$? If $m=0$, any assignment yields score 1.
  
  Memory Limit: Standard CP limit (usually 256MB). Vectors take negligible space.
  Time Limit: 1s usually.
  
  Let's write.

  Variables:
  `int n, m`
  `vector<array<int,3>> clauses`
  `vector<vector<int>> adj` (adj[u] -> list of clause indices where u is a literal)
  `vector<char> vals` (1-based or 0-based mapped)
  
  Implementation:
  Use `clock()` to track time if needed? Or just fixed steps.
  Fixed steps: 2000 iterations per restart. 5 restarts.
  Total 10,000 variable scans.
  $10,000 \times 10,000 \approx 10^8$.
  It might be slightly tight.
  Wait, `adj[u]` size sum is `2*m` (each clause has 3 literals, avg 3. But `m` clauses, sum of degrees is $3m$).
  So sum of work per iteration = $3m$.
  Iterations = 2000.
  Total work = $2000 \times 3m \approx 2000 \times 30000 = 6 \times 10^7$.
  With 5 restarts: $3 \times 10^8$.
  This is close to limit but likely okay in C++ with `-O2` (default CP flags usually optimize well).
  To be safe, reduce restarts to 3 or iterations to 1500. Or combine.
  Also, the loop only checks neighbors that improve the score? No, it finds max improvement.
  Finding max improvement requires checking ALL variables.
  Optimization: If max delta > 0, we proceed. If max delta == 0, we stop (assuming hill climbing).
  Most variables will have delta=0 or small? No.
  But checking all vars costs $3m$.
  Can we optimize?
  Only vars that participate in unsatisfied clauses matter.
  Maintain list of "bad" variables?
  That adds overhead. Given constraints, brute-force neighbor search (all vars) is probably intended.
  
  Let's ensure constants are minimal.
  Use `static_cast` where needed, but compiler optimizes.
  Avoid unnecessary allocations.
  
  One small trick: The score is linear.
  
  Also, note: "Output exactly one line: n integers".
  Format: `x1 x2 ... xn`.
  
  Let's finalize the code.

  Double check constraints: $n \le 1000$, $m \le 10000$.
  Time limit not specified but typical is 1s.
  Complexity $O(K \cdot m)$ per restart.
  $K \approx 2000$. $m = 10000$. Ops $\approx 2 \cdot 10^7$.
  5 restarts $\to 10^8$.
  Operations are simple boolean checks and ints.
  This should pass comfortably in modern judges (usually $10^8$ ops/sec is conservative, often $2-5 \cdot 10^8$ achievable).
  
  One more optimization: `val_v` reuse.
  If `lit == v` and `lit == -v` (impossible), etc.
  
  The `get_val` function will be inlined or written inline to avoid call overhead.
  
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <array>

using namespace std;

struct Clause {
    int l1, l2, l3;
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    // Adjacency list: var -> indices of clauses where var appears
    // Indices are 0 to m-1
    vector<vector<int>> var_clauses(n + 1); 

    for (int i = 0; i < m; ++i) {
        cin >> clauses[i].l1 >> clauses[i].l2 >> clauses[i].l3;
        var_clauses[abs(clauses[i].l1)].push_back(i);
        var_clauses[abs(clauses[i].l2)].push_back(i);
        var_clauses[abs(clauses[i].l3)].push_back(i);
    }

    // We want to maximize satisfied clauses.
    // Local Search with Multiple Restarts.
    
    vector<char> best_assignment(n, 0);
    int max_satisfied = 0;
    
    if (m == 0) {
        cout << 0;
        for(int i=1; i<n; ++i) cout << " " << 0; // Wait, output n integers.
        // Output format: n integers. If n=10, print 10 numbers.
        // Example: 0 0 ...
        for (int i = 0; i < n; ++i) {
            cout << (i > 0 ? " " : "") << 0;
        }
        cout << "\n";
        return 0;
    }

    // Random engine
    mt19937 rng(1337); // Use fixed seed for determinism during testing, or time().
    // Problem doesn't specify deterministic requirement, but consistent behavior is nice.
    // But randomized is required for good performance usually.
    
    for (int r = 0; r < 10; ++r) {
        vector<char> current(n, (char)(rng() % 2));
        int current_score = 0;
        
        // Initial evaluation
        for (int i = 0; i < m; ++i) {
            int val_l1 = clauses[i].l1 > 0 ? current[abs(clauses[i].l1) - 1] : (1 - current[abs(clauses[i].l1) - 1]);
            int val_l2 = clauses[i].l2 > 0 ? current[abs(clauses[i].l2) - 1] : (1 - current[abs(clauses[i].l2) - 1]);
            int val_l3 = clauses[i].l3 > 0 ? current[abs(clauses[i].l3) - 1] : (1 - current[abs(clauses[i].l3) - 1]);
            if (val_l1 || val_l2 || val_l3) current_score++;
        }

        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = current;
        }

        // Hill Climb
        // We perform a few rounds of finding optimal moves.
        // Round: iterate all vars, find best flip, update.
        
        int improvements = 0;
        while (true) {
            bool changed = false;
            int best_v = -1;
            int best_delta = 0; // Increase in satisfied count

            // Calculate potential delta for each variable
            for (int v = 1; v <= n; ++v) {
                int delta = 0;
                // If we flip variable v (currently current[v-1]), how does score change?
                // Check all clauses involving v
                
                int current_v_val = current[v-1];
                int new_v_val = 1 - current_v_val;
                
                // To optimize, we don't want to re-scan all clauses fully.
                // But since we only touch clauses in var_clauses[v], this is proportional to degree.
                
                for (int c_idx : var_clauses[v]) {
                    const Clause& cl = clauses[c_idx];
                    
                    // Helper to check clause sat
                    // Old sat
                    int l1v = (cl.l1 > 0 && abs(cl.l1) != v) ? current[abs(cl.l1) - 1] : 
                               (cl.l1 > 0) ? current_v_val : (1 - current_v_val);
                    int l2v = (cl.l2 > 0 && abs(cl.l2) != v) ? current[abs(cl.l2) - 1] : 
                               (cl.l2 > 0) ? current_v_val : (1 - current_v_val);
                    int l3v = (cl.l3 > 0 && abs(cl.l3) != v) ? current[abs(cl.l3) - 1] : 
                               (cl.l3 > 0) ? current_v_val : (1 - current_v_val);
                               
                    int old_sat = l1v || l2v || l3v;
                    
                    // New sat (with flipped v)
                    // If literal involves v, we swap current_v_val/new_v_val in the condition
                    // Actually, logic:
                    // if (abs(lit) == v) { value = new_v_val if lit>0 else !new_v_val }
                    // else { value = current[abs(lit)-1] if lit>0 else !current[abs(lit)-1] }
                    
                    int nl1v = (abs(cl.l1) == v) ? ((cl.l1 > 0) ? new_v_val : (1 - new_v_val)) 
                                   : ((cl.l1 > 0) ? current[abs(cl.l1) - 1] : (1 - current[abs(cl.l1) - 1]));
                    int nl2v = (abs(cl.l2) == v) ? ((cl.l2 > 0) ? new_v_val : (1 - new_v_val)) 
                                   : ((cl.l2 > 0) ? current[abs(cl.l2) - 1] : (1 - current[abs(cl.l2) - 1]));
                    int nl3v = (abs(cl.l3) == v) ? ((cl.l3 > 0) ? new_v_val : (1 - new_v_val)) 
                                   : ((cl.l3 > 0) ? current[abs(cl.l3) - 1] : (1 - current[abs(cl.l3) - 1]));
                               
                    int new_sat = nl1v || nl2v || nl3v;

                    if (new_sat && !old_sat) delta++;
                    else if (!new_sat && old_sat) delta--;
                    
                    // Optimization: if we found a huge delta, maybe continue?
                    // No, must find max. But delta is bounded by +3 and -3? 
                    // Actually bounded by +1 or -1 per clause? No.
                    // A clause goes Unsatisfied -> Satisfied: +1.
                    // Satisfied -> Unsatisfied: -1.
                    // So delta is integer in [-num_clauses_in_v, num_clauses_in_v].
                    
                    // We can stop processing clauses for this var if current delta exceeds max possible future?
                    // Not really helpful.
                }
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_v = v;
                }
            }

            if (best_delta > 0) {
                // Apply flip
                if (best_v != -1) {
                    // Update best tracking
                    // Wait, best_assignment should track the global max across restarts?
                    // Yes. But inside restart, we want to see if we beat best_satisfied.
                    // But we only need to update if we find a better global state.
                    
                    // Wait, I am inside `while`. I should update `current` here.
                    current[current[v_best]-1] = 1 - current[v_best-1];
                    
                    // Recalculate score or increment/decrement?
                    // Recomputing score of just updated is expensive.
                    // Maintain `current_score`.
                    // But recomputing score is cheap? No, O(M).
                    // We can maintain score.
                    
                    // Actually, `current_score` logic is simpler.
                    // Just update `current` and re-eval score if needed, or track delta.
                    // Since we computed delta, we know `current_score += best_delta`.
                    // However, this assumes `current_score` was tracked.
                    // But my code snippet didn't track `current_score` dynamically, it recomputed initially.
                    // I'll track delta.
                    
                    int idx = best_v - 1;
                    current[idx] ^= 1;
                    // Need to update `current_score` based on `delta` computed for `best_v`.
                    // Wait, I computed `delta` for `best_v` specifically.
                    // So yes, `current_score += best_delta` (Note: if delta was calculated as `new - old`).
                    
                    // But wait, I'm updating `current` in place.
                    // If I calculate delta, I know `current_score += best_delta`.
                    // But this relies on the fact that `current` was used to calculate delta.
                    // Since I haven't flipped `best_v` YET when calculating delta (I used `current_v_val`), it is consistent.
                    // Once I flip, the new state is valid.
                }
            } else {
                break;
            }
        }
    }
    
    // Output best_assignment
    for (int i = 0; i < n; ++i) {
        cout << (i > 0 ? " " : "") << best_assignment[i];
    }
    cout << "\n";

    return 0;
}
```
One logical error:
In the loop `for (int v = 1; v <= n; ++v)`:
If `delta > best_delta`, update `best_v`.
Then if `best_delta > 0`, apply flip.
BUT `current` must be preserved to compute next variables' deltas.
Wait, if I flip `v`, then for the next variable `u`, its contribution to clauses might change if they share clauses with `v`.
So the loop MUST NOT flip immediately. It must simulate all possibilities and pick ONE to flip globally for that round.
Correct. The code block:
```cpp
            for (int v = 1; v <= n; ++v) {
                // ... calculate delta assuming flip of v (keeping other vars constant) ...
                if (delta > best_delta) { best_delta = delta; best_v = v; }
            }
            
            if (best_delta > 0) {
                current[best_v - 1] = 1 - current[best_v - 1];
                // Update score?
                // Recomputing score is safer/easier.
                // Since we are breaking out of this round anyway (unless we keep looping),
                // and we do 2000 rounds.
                // Updating `current_score` manually:
                // We know `best_delta` was the improvement.
                // `current_score += best_delta`.
                // Wait, is `best_delta` calculated assuming other vars don't change?
                // Yes.
                // So `current_score` is consistent.
                
                // Wait, does `current_score` reflect the new state after flipping?
                // Yes.
                // However, I didn't track `current_score` incrementally in the snippet above (it was re-evaluated).
                // Let's add `score` maintenance or just re-evaluate score every time?
                // Re-eval score takes O(M). Done once per round. Rounds = 2000. Total $2000 \times 10000 = 2 \cdot 10^7$. Fine.
                // Tracking incrementally is slightly harder because multiple clauses involve both `v` and `u`.
                // But wait, if we flip `v`, we change status of all clauses in `var_clauses[v]`.
                // So `new_score = old_score + delta`. This is exact.
                
                // So I should track `score` variable.
            }
```
Wait, if I flip `v`, the state changes. The next iteration of `while` starts from THIS new state.
Inside the `for` loop for finding `best_v` (next iteration of `while`), I use the *new* `current` assignment.
So `current` MUST be updated BEFORE entering the next iteration of `while`.
The logic `if (best_delta > 0) { ... flip ...; }` does exactly that.

Wait, inside the `for` loop, I used `current` to calculate deltas.
So before flipping, `current` represents the state *before* any change in this round.
If I find `best_v`, I flip it. `current` updates.
Then I restart the `while` loop.
Correct.

Wait, inside the `for` loop, `current_v_val` refers to `current[v-1]`.
If I update `current` in the middle of the `for` loop (which I shouldn't), it breaks.
My code does NOT update `current` inside `for` loop. It updates only after finding `best_v`.
So `current` is constant throughout the `for` loop.
This allows comparing all candidates simultaneously based on current state.
Then we pick the best candidate and commit to it.
This is standard Hill Climbing.

Optimization:
Check `current_score` periodically to see if we are better than global max?
We save the assignment whenever `current_score` exceeds global max.
Wait, inside the restart loop, `best_assignment` holds the *global* best across all restarts AND progress within restart?
Yes, I should compare `current_score` with `max_satisfied` (variable).
Wait, in the snippet, `max_satisfied` was compared before loop. Inside loop, after updates, check if `current_score > max_satisfied`.
Better.

Wait, `best_assignment` should store the BEST state ever seen across ALL restarts.
Inside a single restart, if I reach a state with score S, and S > global_best, I should save `current` to `best_assignment`.
My snippet does this:
```cpp
        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = current;
        }
```
This happens at start of restart. But not during hill climb.
I should move this check to after `while` loop or after every successful flip.
Actually, after every flip, `current_score` increases.
Wait, hill climbing can also get stuck in local optimum where score is lower than previous high score?
No, we only flip if `delta > 0`. So score strictly increases.
However, `delta` might be 0 for a long time.
Wait, my logic: `if (best_delta > 0) flip`. So score always increases.
So `current_score` keeps growing.
So we should check `if (current_score > max_satisfied) { max_satisfied = current_score; best_assignment = current; }`
Inside the flip block.
Or just check at end of each round (pass). Since score increases monotonicity is guaranteed per pass, checking at end of pass is valid?
Wait, if I flip `v`, score increases. I could update `max` right there.
Checking at end of pass is slightly less frequent, which saves comparisons. It's fine.
Wait, if I don't update `max_satisfied` during hill climb, I might miss the global optimum of that specific path?
But the path ends at a peak. That peak is the highest reached in that restart.
So checking after hill climb terminates (when stuck) is sufficient for that restart.
And since I run multiple restarts, the overall maximum will be found.

Refined loop structure:
Restart loop.
  Init `current`.
  Calc `initial_score`.
  Track `current_score`.
  Update global best if needed.
  While improved:
     Find best move.
     If delta > 0:
         Flip.
         `current_score += delta`.
         Update global best (optional, or just rely on final state).
     Else:
         Break.
         Update global best with final `current_score`.

Wait, I need to check global best.
If I restart from scratch, I get a new peak.
Is it possible that an intermediate state is better than the peak of restart 1, but worse than the peak of restart 10?
Yes. But since I compare peaks against `max_satisfied`, it works.
I'll update global best upon success of a flip (just to be safe/fast).

One edge case: Is `current_score` maintained correctly?
`current_score` initialized at start.
Delta is added.
When restarting, reset.
Looks good.

Complexity check again:
Vars per clause = 3.
Sum of `var_clauses[v].size()` = $3m$.
Loop `for v in 1..n`: iterates `var_clauses[v]`.
Inner body executes exactly $3m$ times per round.
Round iterations = 2000.
Total ops = $2000 \times 3 \times 10000 = 6 \times 10^7$.
With 10 restarts -> $6 \times 10^8$.
This is borderline.
Maybe reduce iterations to 1000? Or reduce restarts to 3?
Usually 1-2 seconds is allowed. $10^8$ simple ops is safe.
But let's optimize the inner check.
Notice we compute `new_sat` and `old_sat` for each clause.
We can optimize `get_val`.
Also `vector` iterator overhead.
Using raw arrays or static buffers for `var_clauses`? `vector` is fine.
Use `mt19937` with fixed seed? Yes.
Use `int` for `val` (0/1).

One micro-optimization:
Inside the inner loop over clauses, we access `current` array.
Accessing `current` by index is fast.
The logic `abs(cl.l1)` is bitwise if handled? No, `abs` is branchy.
Precompute absolute values?
We can replace `cl.l1` with `l1_abs`?
Store `cl_abs` in Clause? No, need original sign.
Sign is important.
`abs(val)` can be done manually: `x < 0 ? -x : x`. Branch free usually? Compiler optimizes.

Let's write the solution.

One subtle thing:
If $m$ is small (e.g. 0), the code should handle it.
Input constraints say $m \ge 0$.
If $m=0$, loop won't run, prints 0s. Correct.

Double check: "Each integer is in [-n, n], non-zero".
So 0 is not a literal. Good.

Final check on logic:
If `best_delta == 0`, we stop.
This means we are at a local optimum.
But maybe we can jump out?
Standard practice for CP: 3 restarts is usually enough for random instances.
Given problem asks to maximize, and random initialization covers many regions.
The solution looks solid.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// Data structures
struct Clause {
    int l1, l2, l3;
};

// Function to quickly evaluate a literal's truth value
inline int get_literal_val(int lit, int v, int val_v, const vector<char>& assign) {
    // If this literal involves variable v, return value based on v's flip or new value
    // But wait, the logic below handles specific contexts.
    // This function is just a generic accessor for literals.
    // It assumes assign is current assignment.
    // But inside the optimization loop we want to handle 'v' specially.
    // So let's just inline the logic inside the loop for speed.
    return 0; // Placeholder
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    vector<vector<int>> var_clauses(n + 1); 

    for (int i = 0; i < m; ++i) {
        cin >> clauses[i].l1 >> clauses[i].l2 >> clauses[i].l3;
        var_clauses[abs(clauses[i].l1)].push_back(i);
        var_clauses[abs(clauses[i].l2)].push_back(i);
        var_clauses[abs(clauses[i].l3)].push_back(i);
    }

    if (m == 0) {
        // Special case m=0
        for (int i = 0; i < n; ++i) {
            cout << (i > 0 ? " " : "") << 0;
        }
        cout << "\n";
        return 0;
    }

    // Global best
    vector<char> best_assignment(n, 0);
    int max_satisfied = -1;

    // Randomness
    // Use time as seed, but maybe just fixed for reproducibility in debug
    // In CP, usually random seed is fine.
    int seed = 1337; // Deterministic for consistency
    
    // To speed up, we can process multiple iterations per restart
    // 1000 iterations per pass seems reasonable. 
    // 5 restarts.

    for (int r = 0; r < 5; ++r) {
        // Reset random seed per restart
        // Or use distinct seeds
        // int s = seed + r; 
        // mt19937 rng(s); 
        
        // Let's use deterministic assignment derived from seed
        // But simple modulo arithmetic is enough for distribution
        // Or just randomize once
        
        vector<char> current(n, 0);
        for(int i=0; i<n; ++i) current[i] = (char)((seed ^ r * 12345 + i) & 1); // Pseudo-random init
        
        int current_score = 0;
        
        // Initial calc
        for (int i = 0; i < m; ++i) {
            int l1 = clauses[i].l1;
            int l2 = clauses[i].l2;
            int l3 = clauses[i].l3;
            bool sat = false;
            
            // Manual eval
            if (l1 > 0) sat |= current[abs(l1) - 1]; else sat |= (!current[abs(l1) - 1]);
            if (l2 > 0) sat |= current[abs(l2) - 1]; else sat |= (!current[abs(l2) - 1]);
            if (l3 > 0) sat |= current[abs(l3) - 1]; else sat |= (!current[abs(l3) - 1]);
            
            if (sat) current_score++;
        }

        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = current;
        }

        // Hill Climbing
        // We run this for a fixed number of iterations to avoid timeout
        // or until no improvement. 
        // 1000 iterations per restart is a safe bet.
        // 5 restarts * 1000 iter * 3 * m = 1.5 * 10^8 ops.
        
        while (true) {
            int best_v = -1;
            int best_delta = 0;
            
            // Scan all variables
            for (int v = 1; v <= n; ++v) {
                int delta = 0;
                
                // Optimization: Retrieve relevant clauses
                // Accessing vector iterators might be slow?
                // Range-based for loop is fine.
                // But to avoid overhead of creating objects in loop, use indices.
                
                int val_v = current[v-1];
                int new_val_v = 1 - val_v;
                
                for (int c_idx : var_clauses[v]) {
                    // Unrolling logic for speed
                    int l1 = clauses[c_idx].l1;
                    int l2 = clauses[c_idx].l2;
                    int l3 = clauses[c_idx].l3;
                    
                    // Check current saturation
                    bool l1_cur = (l1 > 0) ? current[abs(l1) - 1] : !current[abs(l1) - 1];
                    bool l2_cur = (l2 > 0) ? current[abs(l2) - 1] : !current[abs(l2) - 1];
                    bool l3_cur = (l3 > 0) ? current[abs(l3) - 1] : !current[abs(l3) - 1];
                    bool sat_cur = l1_cur || l2_cur || l3_cur;
                    
                    // Check new saturation (flipping v)
                    bool l1_new = (l1 > 0 && abs(l1) == v) ? new_val_v : ((l1 > 0) ? current[abs(l1) - 1] : !current[abs(l1) - 1]);
                    // Note: if l1 == v (positive), val is current[v-1]. If neg, !current.
                    // If new_val_v is set, we must check if abs(l1)==v.
                    // If l1 > 0 and abs(l1) == v -> l1 == v -> value is val_v. New is new_val_v.
                    // If l1 < 0 and abs(l1) == v -> l1 == -v -> value is !val_v. New is !new_val_v.
                    // Else: unchanged.
                    
                    // Let's rewrite cleanly.
                    int get_val_new = [](int lit, int v, int nv, const vector<char>& arr) -> int {
                         if (abs(lit) == v) {
                             return lit > 0 ? nv : (!nv);
                         } else {
                             return lit > 0 ? arr[abs(lit)-1] : !arr[abs(lit)-1];
                         }
                    };
                    // Lambda is slow? Move outside loop.
                    // But capturing `arr` and `nv` is tricky.
                    // Just write code.
                    
                    bool l1_next = (abs(l1) == v) ? ((l1 > 0) ? new_val_v : (1 - new_val_v)) 
                                             : ((l1 > 0) ? current[abs(l1) - 1] : (1 - current[abs(l1) - 1]));
                    bool l2_next = (abs(l2) == v) ? ((l2 > 0) ? new_val_v : (1 - new_val_v)) 
                                             : ((l2 > 0) ? current[abs(l2) - 1] : (1 - current[abs(l2) - 1]));
                    bool l3_next = (abs(l3) == v) ? ((l3 > 0) ? new_val_v : (1 - new_val_v)) 
                                             : ((l3 > 0) ? current[abs(l3) - 1] : (1 - current[abs(l3) - 1]));
                    
                    bool sat_next = l1_next || l2_next || l3_next;
                    
                    if (sat_next && !sat_cur) delta++;
                    else if (!sat_next && sat_cur) delta--;
                }
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_v = v;
                }
            }
            
            if (best_delta > 0) {
                // Apply best flip
                current[best_v - 1] = 1 - current[best_v - 1];
                current_score += best_delta;
                
                if (current_score > max_satisfied) {
                    max_satisfied = current_score;
                    best_assignment = current;
                }
            } else {
                break; // No improvement possible
            }
        }
    }
    
    // Output
    for (int i = 0; i < n; ++i) {
        cout << (i > 0 ? " " : "") << (int)best_assignment[i];
    }
    cout << "\n";

    return 0;
}
```
Wait, `vector<char>` assignment `best_assignment = current` copies elements.
It's fine.
Lambda overhead? `[](int lit...)` is compiled into the code, might be fast. But avoiding it is safer.
Also, the `l1_cur` calculation does not depend on `v` logic, it's direct lookup.
Wait, `l1 > 0` check is fast.
The logic `abs(l1) == v` is fast.
One optimization: `abs` function is in `<cmath>` (std::abs for int is in `<cstdlib>`). `std::abs` is overloaded in C++.
Better to use `std::abs`.
Include `<cstdlib>`.

Also, `mt19937` isn't included, so I removed it and used deterministic init `(seed ^ r * 12345 + i) & 1`. This is pseudo-random but good enough.

One concern: The time limit.
If the judge is very strict, 1000 iterations might be slow.
Can we optimize the loop further?
Notice `best_delta` can be found faster if we ignore vars with 0 impact? No.
Maybe just run fewer restarts and iterations but smarter?
Random shuffle of variables?
Just stick to the provided solution, it is reasonably optimized.
Also `vector<vector<int>>` access `var_clauses[v]` creates an object iterator? No, it's fast.

One tiny bug fix: `abs` is ambiguous? `#include <cmath>` and `#include <cstdlib>`.
Or `#include <algorithm>` (has nothing related).
`abs` for int is in `<cstdlib>`. `std::abs` in `<cmath>` covers floats too, but in C++ headers it's usually available.
To be safe `std::abs`.
Wait, `abs(int)` in `<cstdlib>`. `std::abs` in `<cmath>` adds float overloads.
Using `std::abs` requires `<cmath>` or `<cstdlib>`.
I will use `std::abs` and include `<cmath>`.

Also, in `get_literal_val` placeholder code, I commented it out.

Wait, logic check: `if (sat_next && !sat_cur) delta++; else if (!sat_next && sat_cur) delta--;`.
This correctly computes `delta = new - old`.
If `new=1, old=0` -> `+1`.
If `new=0, old=1` -> `-1`.
If `new=1, old=1` -> `0`.
If `new=0, old=0` -> `0`.
Correct.

Wait, what if `best_delta` is 0?
Then we break loop. This stops the hill climb for this restart.
This assumes we want to climb up.
It might stop prematurely if we have many plateaus.
But usually for 3-SAT, random start hits a high plateau.
I think it's fine.

Final check on includes.
vector, iostream, algorithm, cstdlib, cmath.

One last thing: `best_assignment` should be printed at the end.
And `current_score` update logic.
When we flip, `current` changes. `current_score` updates.
If I update `current_score` based on `best_delta` computed assuming `current` was the old state, and `current` *is* updated, then `current_score` reflects the *new* state.
Correct.

```cpp
/*
Solution for 3-SAT
Greedy Local Search with Multiple Restarts
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

struct Clause {
    int l1, l2, l3;
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    vector<vector<int>> var_clauses(n + 1); 

    for (int i = 0; i < m; ++i) {
        cin >> clauses[i].l1 >> clauses[i].l2 >> clauses[i].l3;
        // Handle variable mapping.
        // Literals are in range [-n, n], non-zero.
        if (clauses[i].l1 != 0) var_clauses[std::abs(clauses[i].l1)].push_back(i);
        if (clauses[i].l2 != 0) var_clauses[std::abs(clauses[i].l2)].push_back(i);
        if (clauses[i].l3 != 0) var_clauses[std::abs(clauses[i].l3)].push_back(i);
    }

    if (m == 0) {
        for (int i = 0; i < n; ++i) {
            cout << (i > 0 ? " " : "") << 0;
        }
        cout << "\n";
        return 0;
    }

    vector<char> best_assignment(n, 0);
    int max_satisfied = -1;

    // Number of restarts
    int restarts = 5; 
    
    // We run multiple passes.
    // To ensure diversity, we shuffle/init differently.
    for (int r = 0; r < restarts; ++r) {
        vector<char> current(n, 0);
        // Deterministic-ish initialization
        int seed = 1337 + r;
        for(int i=0; i<n; ++i) current[i] = (char)((seed ^ (i * 1234567)) & 1);
        
        int current_score = 0;
        
        // Initial evaluation
        for (int i = 0; i < m; ++i) {
            int l1 = clauses[i].l1;
            int l2 = clauses[i].l2;
            int l3 = clauses[i].l3;
            
            bool sat = false;
            if (l1 > 0) sat |= current[std::abs(l1) - 1]; else sat |= (!current[std::abs(l1) - 1]);
            if (l2 > 0) sat |= current[std::abs(l2) - 1]; else sat |= (!current[std::abs(l2) - 1]);
            if (l3 > 0) sat |= current[std::abs(l3) - 1]; else sat |= (!current[std::abs(l3) - 1]);
            
            if (sat) current_score++;
        }

        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = current;
        }

        // Local Search
        while (true) {
            int best_v = -1;
            int best_delta = 0;
            
            for (int v = 1; v <= n; ++v) {
                int delta = 0;
                int val_v = current[v-1];
                int new_val_v = 1 - val_v;
                
                for (int c_idx : var_clauses[v]) {
                    int l1 = clauses[c_idx].l1;
                    int l2 = clauses[c_idx].l2;
                    int l3 = clauses[c_idx].l3;
                    
                    // Compute sat_curr
                    bool l1_cur = (l1 > 0) ? current[std::abs(l1) - 1] : !current[std::abs(l1) - 1];
                    bool l2_cur = (l2 > 0) ? current[std::abs(l2) - 1] : !current[std::abs(l2) - 1];
                    bool l3_cur = (l3 > 0) ? current[std::abs(l3) - 1] : !current[std::abs(l3) - 1];
                    
                    bool sat_cur = l1_cur || l2_cur || l3_cur;
                    
                    // Compute sat_next (assuming v flips)
                    // If literal involves v, use new_val_v
                    bool l1_next = (std::abs(l1) == v) ? ((l1 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l1 > 0) ? current[std::abs(l1) - 1] : !current[std::abs(l1) - 1]);
                    bool l2_next = (std::abs(l2) == v) ? ((l2 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l2 > 0) ? current[std::abs(l2) - 1] : !current[std::abs(l2) - 1]);
                    bool l3_next = (std::abs(l3) == v) ? ((l3 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l3 > 0) ? current[std::abs(l3) - 1] : !current[std::abs(l3) - 1]);
                    
                    bool sat_next = l1_next || l2_next || l3_next;
                    
                    if (sat_next && !sat_cur) delta++;
                    else if (!sat_next && sat_cur) delta--;
                }
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_v = v;
                }
            }
            
            if (best_delta > 0) {
                current[best_v - 1] = 1 - current[best_v - 1];
                current_score += best_delta;
                
                if (current_score > max_satisfied) {
                    max_satisfied = current_score;
                    best_assignment = current;
                }
            } else {
                break;
            }
        }
    }
    
    for (int i = 0; i < n; ++i) {
        cout << (i > 0 ? " " : "") << (int)best_assignment[i];
    }
    cout << "\n";

    return 0;
}
```
Double check `!new_val_v`.
`new_val_v` is 0 or 1.
If `l1 == -v` (negative literal), it means `NOT v`.
If `v` flips to `new_val_v` (where `new` means 0 if `old` was 1), then `NOT v` value changes from `1-old` to `1-new`.
Since `new = 1-old`, `1-new = old`.
So if literal is `-v`, contribution flips from `!val_v` to `!new_val_v` which is `val_v`.
My code: `((l1 > 0) ? new_val_v : !new_val_v)`.
If `l1 == -v`, `l1 > 0` is false, so it uses `!new_val_v`.
Is `!new_val_v` correct for `-v` flipped?
If `val_v` was 0. `new_val_v` becomes 1.
Original `-v` val was `!0 = 1`. New `-v` val should be `!1 = 0`.
`!new_val_v` is `0`. Correct.
If `val_v` was 1. `new_val_v` becomes 0.
Original `-v` val was `!1 = 0`. New `-v` val should be `!0 = 1`.
`!new_val_v` is `1`. Correct.
Logic holds.

The code seems complete.

One minor thing: `std::abs` requires `<cmath>` or `<cstdlib>`. Included.
Also `vector` usage is fine.
Memory limit is usually generous for this size.
Time limit: 1.5s is typical for $10^8$ ops.

I will wrap it in cpp tags.

Wait, input reading `cin >> clauses[i].l1` etc.
Integers can be negative.
`std::abs` works on negatives.
Indices in `current` use `std::abs(l) - 1`.
If `l` is `-5`, `abs` is `5`. Index `4`. Valid (n>=5).
If `n=1000`.
Seems correct.

Final check: "Output exactly one line".
The loop prints space separated integers.
First integer at index 0. No leading space.
Between integers space.
Correct.

One thought: `current` copy to `best_assignment` is $O(N)$.
Done inside `while` loop potentially $M$ times?
No, `while` loop breaks when stuck.
It runs many times.
Copying vector $N$ times?
$N=1000$. `best_assignment` update only happens if `current_score > max_satisfied`.
This is rare (only when moving up).
So copy cost is negligible.
Wait, I said "inside while loop". `if (current_score > max_satisfied)` triggers copy.
Since we maximize, this triggers when we find a NEW global record.
Frequency is low.
However, in my code:
`if (best_delta > 0) { ... update score ... if (score > max) update best }`
Here we update best assignment.
Since we only accept delta > 0, score increases monotonically.
So we might update best assignment every step.
Copying 1000 bytes is very fast (cache locality).
Cost is fine.

Ready.

One tiny optimization: `abs` calls are fast. `vector<bool>` specialization is avoided.
Using `vector<char>` is good.
I will submit this.

Wait, the prompt says "The execution time and memory limit are also stated in the statement".
Since it's not provided in text, I assume standard.
Code is efficient enough.

One corner case: `n` up to 1000, `m` up to 10000.
The `var_clauses` size is $1001$.
Inner loops sum to $3m$ accesses.
Outer loop 1000.
Total accesses $3 \cdot 10^7$.
Very safe.

One check: The input might contain redundant clauses or unit clauses?
Problem says "Each clause contains exactly three literals".
So no unit clauses (unless 3 literals are same, e.g. `x x x`).
If `x x x`, it's equivalent to `(x)`.
My logic handles it correctly (ORing 3 identical booleans).
Redundant literals (like `x x y`) handled correctly.
Zero literals are not allowed (problem says non-zero).
Everything is covered.

```cpp
// ... code ...
```
I will output the code block.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

struct Clause {
    int l1, l2, l3;
};

int main() {
    // Optimize I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<Clause> clauses(m);
    // Adjacency list: var -> indices of clauses where var appears
    vector<vector<int>> var_clauses(n + 1); 

    for (int i = 0; i < m; ++i) {
        cin >> clauses[i].l1 >> clauses[i].l2 >> clauses[i].l3;
        // Store clause index in the adjacency list for variables involved
        if (clauses[i].l1 != 0) var_clauses[std::abs(clauses[i].l1)].push_back(i);
        if (clauses[i].l2 != 0) var_clauses[std::abs(clauses[i].l2)].push_back(i);
        if (clauses[i].l3 != 0) var_clauses[std::abs(clauses[i].l3)].push_back(i);
    }

    // Handle empty clause case
    if (m == 0) {
        for (int i = 0; i < n; ++i) {
            cout << (i > 0 ? " " : "") << 0;
        }
        cout << "\n";
        return 0;
    }

    // Global best assignment
    vector<char> best_assignment(n, 0);
    int max_satisfied = -1;

    // Number of restarts for randomized local search
    int restarts = 5;
    
    for (int r = 0; r < restarts; ++r) {
        // Current assignment
        vector<char> current(n, 0);
        // Pseudo-random initialization to explore different regions
        int seed = 1337 + r;
        for(int i=0; i<n; ++i) current[i] = (char)((seed ^ (i * 1234567)) & 1);
        
        int current_score = 0;
        
        // Initial evaluation
        for (int i = 0; i < m; ++i) {
            int l1 = clauses[i].l1;
            int l2 = clauses[i].l2;
            int l3 = clauses[i].l3;
            
            bool sat = false;
            if (l1 > 0) sat |= current[std::abs(l1) - 1]; else sat |= (!current[std::abs(l1) - 1]);
            if (l2 > 0) sat |= current[std::abs(l2) - 1]; else sat |= (!current[std::abs(l2) - 1]);
            if (l3 > 0) sat |= current[std::abs(l3) - 1]; else sat |= (!current[std::abs(l3) - 1]);
            
            if (sat) current_score++;
        }

        // Check if initial random assignment is the best so far
        if (current_score > max_satisfied) {
            max_satisfied = current_score;
            best_assignment = current;
        }

        // Hill Climbing Phase: find neighbors that improve the score
        while (true) {
            int best_v = -1;
            int best_delta = 0;
            
            // Check flipping each variable to see which gives the best improvement
            for (int v = 1; v <= n; ++v) {
                int delta = 0;
                int val_v = current[v-1];
                int new_val_v = 1 - val_v;
                
                // Iterate over all clauses involving variable v
                for (int c_idx : var_clauses[v]) {
                    int l1 = clauses[c_idx].l1;
                    int l2 = clauses[c_idx].l2;
                    int l3 = clauses[c_idx].l3;
                    
                    // Compute satisfaction with current assignment
                    bool l1_cur = (l1 > 0) ? current[std::abs(l1) - 1] : !current[std::abs(l1) - 1];
                    bool l2_cur = (l2 > 0) ? current[std::abs(l2) - 1] : !current[std::abs(l2) - 1];
                    bool l3_cur = (l3 > 0) ? current[std::abs(l3) - 1] : !current[std::abs(l3) - 1];
                    bool sat_cur = l1_cur || l2_cur || l3_cur;
                    
                    // Compute satisfaction with flipped assignment for v
                    // If literal involves v, use new_val_v; otherwise use current values
                    bool l1_next = (std::abs(l1) == v) ? ((l1 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l1 > 0) ? current[std::abs(l1) - 1] : !current[std::abs(l1) - 1]);
                    bool l2_next = (std::abs(l2) == v) ? ((l2 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l2 > 0) ? current[std::abs(l2) - 1] : !current[std::abs(l2) - 1]);
                    bool l3_next = (std::abs(l3) == v) ? ((l3 > 0) ? new_val_v : !new_val_v) 
                                                        : ((l3 > 0) ? current[std::abs(l3) - 1] : !current[std::abs(l3) - 1]);
                    
                    bool sat_next = l1_next || l2_next || l3_next;
                    
                    // Update delta
                    if (sat_next && !sat_cur) delta++;
                    else if (!sat_next && sat_cur) delta--;
                }
                
                // Keep track of the variable that gives the maximum positive improvement
                if (delta > best_delta) {
                    best_delta = delta;
                    best_v = v;
                }
            }
            
            // If an improvement was found, apply it and update global best
            if (best_delta > 0) {
                current[best_v - 1] = 1 - current[best_v - 1];
                current_score += best_delta;
                
                if (current_score > max_satisfied) {
                    max_satisfied = current_score;
                    best_assignment = current;
                }
            } else {
                // No improvement found, local optimum reached
                break;
            }
        }
    }
    
    // Output the result
    for (int i = 0; i < n; ++i) {
        cout << (i > 0 ? " " : "") << (int)best_assignment[i];
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=909)

````
The user wants a solution to a 3-SAT maximization problem (MAX-3SAT), which is an NP-hard optimization problem.
Constraints: n <= 1000, m <= 10000.

For this size of input, finding the true optimal solution for MAX-3SAT is computationally difficult, but we need *some* assignment. The scoring scheme rewards satisfying as many clauses as possible, but any valid assignment is accepted and gets a score proportional to satisfied clauses. A simple strategy like iterating through all variables at random or using a heuristic search like Greedy / WalkSAT could work well here. Given the constraints and nature of competitive programming "easy" version of Max-3SAT, a randomized local search often suffices to get high satisfaction. Alternatively, just outputting all True/False might be sufficient to pass if there's no specific checker for optimality, but since I should aim for a good solution, a simple iterative approach that tries random assignments or starts with all literals satisfied will suffice. The simplest non-trivial solver would be something greedy: pick a variable that appears most in unsatisfied clauses, set it so more are satisfied, repeat. Or even simpler, satisfy each clause by flipping one literal until stuck or time out. Even simpler still - just assign randomly, repeat, maybe keep best.

Actually, for these contest-style tasks where you need only a reasonable score, a pure random initialization + iteration often beats the naive greedy on average for these small bounds. Let's implement a straightforward randomized approach that settles on a satisfying configuration reasonably fast, and handles the edge case n > 0 with a default fallback.
</think>

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    
    vector<pair<int,int>> clauses(m);
    for (int i = 0; i < m; ++i) {
        int a,b,c;
        cin >> a >> b >> c;
        clauses[i] = {{a,b},{b,c}}; // Store first two pairs for easy unpacking
    }
    
    // Try multiple random seeds, take the assignment that satisfies most clauses
    bool best_found = false;
    int max_satisfied = -1;
    
    srand(0); // Fixed seed for reproducibility
    
    for (int trial = 0; trial < 5000; ++trial) {
        vector<int> x(n+1); // Variables 1..n
        for(int v=1; v<=n; ++v) x[v] = rand()%2;
        
        int s = 0;
        for(int i=0; i<m; ++i) {
            int l1 = clauses[i].first.first;
            int l2 = clauses[i].first.second;
            int l3 = clauses[i].second.second;
            
            int p1 = abs(l1)%2;
            int p2 = abs(l2)%2;
            int p3 = abs(l3)%2;
            
            int f1 = ((l1<0)?1-x[p1]:x[p1]);
            int f2 = ((l2<0)?1-x[p2]:x[p2]);
            int f3 = ((l3<0)?1-x[p3]:x[p3]);
            
            s += (f1||f2||f3);
        }
        
        if (s > max_satisfied) {
            max_satisfied = s;
            best_found = true;
            for(int v=1; v<=n; ++v) {
                cout << (x[v]>0 ? "1" : "0") << (v==n?"":" ");
            }
            cout << endl;
        }
    }
    
    // If n > 0 (at least one variable) and we never found a satisfying assignment, fall back to all-true
    if (!best_found && n > 0) {
        for(int v=1; v<=n; ++v) cout << 1 << (v==n?"":" ");
        cout << endl;
    }
    
    return 0;
}
```
````
