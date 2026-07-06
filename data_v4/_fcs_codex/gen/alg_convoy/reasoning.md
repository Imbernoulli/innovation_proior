The input scale is up to 2*10^5 jobs, so I have an O(n log n) sorting budget and O(n) memory budget. I explicitly reject trying all permutations, dynamic programming over subsets, or pairwise local search without a proven global rule, because those would explode at this n. Even O(n^2) exchange-improvement simulation is too risky at 2*10^5. I need a general ordering rule and a linear pass to evaluate the schedule.

This is the classical two-machine flow-shop makespan problem, solved by Johnson's rule. The rule is not a per-case shortcut: for every job, compare a_i and b_i. Jobs with a_i <= b_i should go in the first block, sorted by increasing a_i. Jobs with a_i > b_i should go in the second block, sorted by decreasing b_i. Concatenating those two blocks gives an optimal order.

I justify the rule by the standard adjacent-swap argument. Consider two neighboring jobs x and y after all earlier jobs have already accumulated some fixed machine-A time. Only the relative order of x and y matters for the local contribution to how much machine B may be delayed. For x before y, the relevant worst delay term is max(a_x + b_x, a_x + a_y) plus the later b_y contribution; for y before x it is the symmetric expression. Simplifying the adjacent comparison gives the known condition: place x before y when min(a_x, b_y) <= min(b_x, a_y). Johnson's construction is exactly a sorting arrangement satisfying this condition for every crossable pair: small A times are pulled early in ascending order, and small B times are pushed late in descending order. Since any inversion violating the condition can be swapped without increasing the makespan, repeatedly removing such inversions reaches Johnson's order, so that order is optimal.

After constructing the order, computing the makespan is direct. I keep tA, the finish time of the current prefix on machine A, and tB, the finish time of the current prefix on machine B. For each job in order, tA += a_i, then B starts it at max(tA, tB), so tB = max(tA, tB) + b_i. The final tB is the answer. Because n can be 2*10^5 and each time can be 10^9, sums can reach about 4*10^14, so I use 64-bit integers.

I verify on a concrete small example:
Jobs:
1: a=3, b=2
2: a=2, b=5
3: a=4, b=1

Partition:
Job 2 has a<=b, so it goes early.
Jobs 1 and 3 have a>b, so they go late sorted by decreasing b: job 1 then job 3.
Order: 2, 1, 3.

Schedule trace:
Start tA=0, tB=0.
Job 2: tA=2, tB=max(0,2)+5=7.
Job 1: tA=5, tB=max(7,5)+2=9.
Job 3: tA=9, tB=max(9,9)+1=10.
Makespan is 10.

Brute-force sanity check over all 6 orders:
1,2,3 gives tB=11.
1,3,2 gives tB=14.
2,1,3 gives tB=10.
2,3,1 gives tB=11.
3,1,2 gives tB=15.
3,2,1 gives tB=14.
The Johnson order reaches the brute-force minimum 10.

This gives a simple correct solution within budget: sort the two groups, concatenate, and simulate once.