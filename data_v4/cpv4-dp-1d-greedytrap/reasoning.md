A tightrope walker starts on platform `0` of a row of `n` and must finish on platform `n-1`, hopping
`+1` or `+2` each step and paying toll `t[i] >= 0` for every platform she lands on — start and finish
included. I want the cheapest legal crossing, reading `n` then the tolls from stdin and printing one
integer. What makes this more than a two-liner is the pileup of quiet landmines: it looks exactly like
a place a local greedy should work, the base cases at `dp[1]` and the tiny-`n` corners are easy to get
wrong, and the accumulation can overflow 32 bits.

The scale fixes the arithmetic width before anything else. `n <= 2*10^5` and each `t[i] <= 10^9`, and
a route of pure `+1` hops lands on *every* platform, so the total can reach `n * 10^9 = 2*10^14` —
about a thousand times past the 32-bit ceiling of ~`2.1*10^9`. Every accumulator and the toll array
must be 64-bit `long long`. An `int` here is a silent wrong answer, and because every toll is
non-negative the overflow would not even wrap to an obviously absurd value; it would just be quietly
too small.

The tempting-but-wrong move is greedy: from `i` only `i+1` and `i+2` are reachable, so hop to whichever
toll is smaller. `O(n)`, almost no code. But a hop doesn't only pay a toll, it changes which platforms
are reachable next, and that is exactly the configuration where a one-step horizon fails. Before
trusting it I try to break it on `t = [3, 1, 1, 9, 1, 9, 1]`. Greedy from `0` (pay `3`) sees `1` and
`2` both at toll `1` — a tie; break it toward the nearer platform and I hop to `1` (total `4`), then
`2` (total `5`), then `4` (total `6`), then `6` (total `7`). But hopping `+2` throughout,
`0 -> 2 -> 4 -> 6`, lands on tolls `3 + 1 + 1 + 1 = 6`, strictly cheaper. Greedy loses, and the reason
is structural: landing on `1` is pure waste, because from `1` she still pays to move on, whereas the
route that lands on `2` never pays for platform `1` at all. This is not a tie-break artifact I can
patch either — `t = [2, 1, 100, 1, 1, 1]` gives greedy `6` against optimal `5`, because the only way to
know whether to absorb a small toll now is to know the tolls further ahead. Greedy is out.

So I commit to the DP. Let `dp[i]` be the minimum total toll of a legal route that lands on platform
`i`. A route reaching `i` made its last hop from `i-1` or `i-2` — those are the only two legal hop
lengths, so the two predecessors are exhaustive with no third transition to miss — and then pays
`t[i]`, giving `dp[i] = t[i] + min(dp[i-1], dp[i-2])` for `i >= 2`. Optimal substructure holds by the
usual exchange argument: a cheapest route to `i` must reach its predecessor by a cheapest route, else
swap in the cheaper prefix. The base cases are where this kind of recurrence quietly breaks.
`dp[0] = t[0]`, the start, with nothing earlier. `dp[1] = t[0] + t[1]`, *not* `t[1]`: platform `1` is
reachable only from `0` via a `+1` hop (there is no platform `-1` for a `+2` hop), so the route is
forced and pays both tolls. The answer is `dp[n-1]` — she must finish standing on the last platform,
and unlike a max-weight selection there is no "do nothing" alternative to fold in with a `max(..., 0)`.

Against the sample `t = [3, 1, 1, 9, 1, 9, 1]`, expected `6`: `dp` runs `3, 4, 4, 13, 5, 14, 6`,
threading `dp[2] = 4 -> dp[4] = 5 -> dp[6] = 6`, which is exactly the `0 -> 2 -> 4 -> 6` route I found
by hand.

For `O(1)` memory I keep only the last two values, `prev2 = dp[i-2]` and `prev1 = dp[i-1]`, seeded from
the base cases. That seed is the one place the base case bites: the reflex is `prev1 = t[1]`, but
`dp[1]` is `t[0] + t[1]`, because reaching platform `1` forces having paid for `0` first. The error
hides for `n >= 3` — the `i = 2` step folds `prev2 = t[0]` back in through the `min`, partly masking the
dropped term — and surfaces only at `n = 2`, where the loop (range `2..n-1`) never runs and the seed is
printed verbatim: with `t = [5, 7]` the wrong seed prints `7` for the forced `0 -> 1` crossing whose
true cost is `12`. So `prev1 = t[1] + t[0]`, and the sample carry reproduces the `dp` table above,
ending at `6`.

The carry indexes `t[0]` and `t[1]` unconditionally, so `n = 0` and `n = 1` would read past the end of
the vector. Both need an early guard, after the read and before the carry: `n = 0` prints `0` (no
platforms, nothing paid), `n = 1` prints `t[0]` (a single platform that is both start and finish). At
`n = 2` the loop is empty and the answer is `prev1 = dp[1]`; at `n = 3` it runs once.

One more edge pins the recurrence's flavor: `n = 3`, `t = [0, 10^9, 0]` gives
`dp[2] = 0 + min(10^9, 0) = 0` — she hops straight over the costly platform, the cheapest-route
analogue of the greedy trap. All tolls being non-negative, there is no sentinel or `-infinity` in the
recurrence for `min` to mishandle. Reading with `cin >>` consumes arbitrary whitespace, so `n` and the
tolls may be split across lines however the input pleases. The full module is in the answer.

