The input scale is q <= 2*10^5, so I need roughly O(q log^2 q) or better and O(q) memory. I cannot scan all active points for each query, because that is O(q^2) in the worst case. I also cannot maintain a sorted vector and insert/delete by shifting elements, because updates can also become O(q^2). A direct array over coordinates from -10^9 to 10^9 would be impossible in memory.

The key observation is that for a fixed query `(x, k)`, the predicate

`count of active points p with |p - x| <= d >= k`

is monotone in `d`. If it is true for some distance `d`, it remains true for every larger distance. Therefore, the answer is the smallest `d` for which the active count inside interval `[x - d, x + d]` is at least `k`.

I only need to count active points whose coordinates fall in an interval. Since all update coordinates are known after reading the input, I coordinate-compress every point value that ever appears in an insert/remove operation. A Fenwick tree over these compressed coordinates stores current multiplicities. Insert is `+1`, remove is `-1`, and interval count is two prefix sums after finding compressed index bounds with `lower_bound` and `upper_bound`.

For a query, I binary search the answer distance. A valid upper bound is the maximum distance from `x` to the smallest or largest coordinate that ever appears in an update; this covers every currently active point because active points are a subset of those coordinates. Each binary-search check counts active points in `[x - mid, x + mid]`.

The complexity is:
- Coordinate compression: O(q log q)
- Each update: O(log q)
- Each query: O(log V * (log q + log q)), where V <= 2*10^9, so at most about 31 distance checks
- Memory: O(q)

This is comfortably within the scale. I considered trying to construct the kth nearest point directly by merging left and right neighbors around `x`, but multiplicities and ties make that more delicate. The monotone interval-count solution is simpler, general, and clearly correct within budget, so I land that.

Worked example:

Operations:
```
1 0
1 5
1 -2
1 5
3 1 3
3 5 2
2 5
3 5 2
1 100
3 50 2
```

After inserting `0, 5, -2, 5`, query `(x=1, k=3)` has distances:
`|0-1|=1`, `|5-1|=4`, `|-2-1|=3`, `|5-1|=4`.
Sorted: `1, 3, 4, 4`, so the answer is `4`.

Query `(x=5, k=2)` has distances:
`5, 0, 7, 0`.
Sorted: `0, 0, 5, 7`, so the answer is `0`.

After removing one `5`, active points are `0, -2, 5`. Query `(x=5, k=2)` gives distances:
`5, 7, 0`.
Sorted: `0, 5, 7`, so the answer is `5`.

After inserting `100`, query `(x=50, k=2)` gives distances:
`50, 52, 45, 50`.
Sorted: `45, 50, 50, 52`, so the answer is `50`.

The program produces:
```
4
0
5
50
```

I also compiled the implementation and compared it against a brute-force multiset simulation on many random small operation sequences; the randomized check reported `all ok`.