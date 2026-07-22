# Downlink Rush: Periodic Ground-Station Phasing

## Problem
A small constellation of `S` imaging satellites shares a common circular orbit of period
`P` ticks, run for `R` consecutive orbits (`T = P * R` ticks total). There are only `K`
ground stations. Because every satellite shares the orbital period, each satellite's line
of sight to each station **recurs identically every orbit**: satellite `i` is visible to
station `k` for a fixed-length window starting at local offset `(phase_i + offset_k)`
within every one of the `R` orbits (this never crosses the period boundary, so each
orbit's window for a pair is one unbroken interval).

Every tick, each satellite's buffer gains `acc_i` units of freshly captured imagery,
capped at `cap_i` (excess is lost that tick, not carried over). While satellite `i`
transmits to station `k` *alone*, its buffer drains by `drain_k` per tick (never more
than it holds). **Co-channel interference**: if `n >= 2` satellites transmit to the same
station in the same tick, the channel collapses to a quarter of nominal capacity
(`drain_k` is always a multiple of 4), split evenly among the `n` senders by integer
division -- a heavily contended tick can floor everyone's rate to `0`. A satellite can
only talk to one station at a time.

Produce a transmission schedule. There is no known easy optimum: piling everyone onto
the fastest station causes interference and overflow, but the exact window/rate values
live only in the input, so you must read and exploit them.

## Input (stdin)
```
S K P R T
acc_1 cap_1 phase_1
...
acc_S cap_S phase_S
drain_1 offset_1 dur_1
...
drain_K offset_K dur_K
```
For satellite `i` and station `k`, the visibility window in orbit `c` (0-indexed,
`0 <= c < R`) is the tick interval `[c*P + phase_i + offset_k, c*P + phase_i + offset_k + dur_k)`.

## Output (stdout)
```
m
sat_1 station_1 start_1 end_1
...
sat_m station_m start_m end_m
```
Print the number of transmission intervals `m`, then `m` lines, each a satellite index
(`0..S-1`), a station index (`0..K-1`), and a half-open tick interval `[start, end)`
during which that satellite transmits to that station.

## Feasibility
An output is valid iff **all** hold:
- every token parses as a finite integer; `0 <= m <= 20000`;
- every `start < end`, `0 <= start`, `end <= T`;
- every interval is fully contained in a genuine visibility window of its
  `(satellite, station)` pair for some orbit `c` (as defined above);
- for any single satellite, its transmission intervals (across all stations) never
  overlap in time.
Any violation scores `Ratio: 0.0`.

## Objective
Simulate all `T` ticks exactly as described above (accrual with cap, then drain at
`drain_k` if alone or `(drain_k // 4) // n` if `n >= 2` share the station, delivering
`min(buffer, rate)`). Maximize `F`, the total data delivered over the whole mission.

## Scoring
The checker also builds its own trivial reference: every satellite uses only the
single highest-drain station's window, every orbit, ignoring every other satellite. Let
`B` be that construction's total delivered data (always positive). With maximization
normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the reference scores `Ratio = 0.1`; a schedule that delivers `10x` more data
caps the score at `1.0`.

## Constraints
- `3 <= S <= 16`, `2 <= K <= 4`, time limit 5s, memory 512m.
- `1 <= acc_i <= 4`, `cap_i = acc_i * (an integer factor)`, `drain_k` is a positive
  multiple of `4`.

## Example
Two satellites, one station, `drain = 16`, `dur = 8`, windows overlapping on 7 of 8
ticks. Transmitting simultaneously: `n = 2` every shared tick, each drains only
`(16 // 4) // 2 = 2`/tick, ~`32` total -- far below one satellite alone (`8 * 16 = 128`).
Splitting the window instead -- satellite 0 on `[0,4)`, satellite 1 on `[4,8)` -- lets
each drain the full uncontended rate `16` for its half: `4*16 = 64` each, `128` total,
four times better than colliding, and well above the single-station baseline `B`.
