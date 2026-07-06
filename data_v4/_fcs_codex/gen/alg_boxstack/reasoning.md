The input scale is n <= 2000, so there are only 3n <= 6000 possible oriented box states. An exponential search over subsets or permutations is impossible: even 2^n is far beyond the budget, and storing per-subset states would MLE immediately. A cubic dynamic program over orientations is also avoidable, while an O((3n)^2) transition scan is about 36 million comparisons, which is acceptable in C++17.

I model each possible rotation as one oriented candidate: base side a, base side b, height c, and the original box id. I sort the two base dimensions so that a <= b; this makes base comparison orientation-independent, because rotating within the horizontal plane does not change whether one rectangle can strictly fit inside another.

Then I sort all oriented candidates by increasing base dimensions. Let dp[i] be the best stack height whose bottom oriented box is candidate i. To place candidate j above candidate i, I need:

a_j < a_i and b_j < b_i,

and the two candidates must not be the same original box. The transition is:

dp[i] = height_i + max(dp[j]) over all valid j above i.

Since base dimensions strictly increase as I go downward, this forms an acyclic order after sorting. I use long long because the maximum height can be as large as 2000 * 10^9.

I reject approaches that try all rotations per box as a recursive assignment first, because 3^2000 is impossible. I also reject sorting by volume or base area alone: strict fitting depends on both base coordinates, and area can increase while one side decreases, so area ordering is not sufficient.

For a concrete check, take two boxes:

Box 1: 1 2 10  
Box 2: 3 4 5

Box 1 can have bases (1,2) height 10, (1,10) height 2, or (2,10) height 1.  
Box 2 can have bases (3,4) height 5, (3,5) height 4, or (4,5) height 3.

The best stack is Box 1 with base (1,2), height 10, on Box 2 with base (3,4), height 5, total 15. The DP sees (1,2) < (3,4), so dp for the Box 2 orientation becomes 5 + 10 = 15. Other rotations either do not fit or produce smaller total height, so the answer is 15. Brute force over the six oriented candidates for this tiny case gives the same maximum.

The general algorithm is therefore: generate all rotations, sort by base, run the O(m^2) longest-stack DP over m <= 6000 oriented candidates, and print the maximum dp value.