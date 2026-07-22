# Town Meeting Logrolling: Referendum Bundle Partition

## Problem

The town clerk has `P` public-works projects up for a vote, each with a
welfare weight `w_i` (its value to the town if it gets built). There are `V`
registered voters. Every voter has an explicit, signed valuation for every
project: she either **supports** project `i` (contributing `+s_i` if the
project is on her ballot), **opposes** it (contributing `-o_i`), or is
**neutral** (contributing `0`). The clerk cannot put every project to its
own separate vote -- state law caps the town meeting at **at most `K`
referendum bundles**. Every project must be assigned to exactly one bundle.

A bundle is put to a single up-or-down vote. For a voter, her total
valuation for a bundle is the SUM of her valuations over every project in
that bundle. A bundle **passes** if and only if a **strict majority** of
the `V` voters have a strictly positive total for it (a voter whose
supports and opposes exactly cancel, or who is neutral on everything in the
bundle, does not count as a yes). If a bundle passes, every project inside
it is built; if it fails, none of them are (no partial credit, whichever
side of the threshold the vote lands on).

Your job: partition the `P` projects into at most `K` bundles to maximize
the total welfare weight of projects that end up in a passing bundle.

The catch: bundling is not the same as popularity. A project that falls
just short of a majority alone may become a clear winner once paired with
the right small companion, whose own supporters are largely voters neutral
on the big project, pushing the combined vote over the threshold. Which
pairings work depends on the exact overlap of supporter/opposer sets, not
on how popular each project looks alone; and a project so reviled that
nearly every voter opposes it drags down anything bundled with it.

## Input (stdin)

```
P V K
```
then, for each project `i = 0..P-1` (in this order), three lines:
```
w_i s_i o_i ns_i no_i
<ns_i supporter voter indices, space-separated, 0-indexed>
<no_i opposer voter indices, space-separated, 0-indexed>
```
Any voter index in `0..V-1` not listed for project `i` is neutral on it.

## Output (stdout)

```
P
b_1 b_2 ... b_P
```
`b_{i+1}` is the bundle id (an integer in `[1,K]`) that project `i`
(0-indexed, in input order) is assigned to. Every project must appear
exactly once.

## Feasibility

- The first output line must echo `P` exactly.
- The second line must contain exactly `P` integers, each a finite integer
  in `[1,K]`.
- Any violation scores `Ratio: 0.0`.

## Scoring

Let `F` be the total welfare weight of projects whose bundle passes.
The checker also builds its own simple reference partition (put project 0
alone in bundle 1, and lump every other project into bundle 2) to get an
internal baseline `B > 0`. The final score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
so reproducing the checker's own reference partition scores about `0.1`,
and a partition roughly `10x` better in total passing weight caps at `1.0`.

## Constraints

- `8 <= P <= 32`, `41 <= V <= 321`, `4 <= K <= 8`.
- Weights, support/oppose magnitudes are positive integers; some projects
  may be effectively unrescuable (opposed by essentially every voter with
  overwhelming intensity) -- no bundle containing one can ever pass.
- Time limit: 5s. Memory: 512MB.

## Example (illustrative shape only, not to scale)

Suppose `P=3, V=5, K=2`. Project A (`w=10`) is supported by voters `{0,1}`
and opposed by `{2,3}` (voter 4 neutral) -- 2 vs 2, no majority alone.
Project B (`w=2`) is supported by voter `{4}` only. Bundling `{A,B}`:
voter 4's total becomes `+s_B > 0`, giving 3 yes votes out of 5 -- a
majority. Project C (`w=1`) is opposed by all 5 voters with a large
magnitude; bundling it with anything sinks that bundle. The best partition
here bundles `{A,B}` together (passes, scores `12`) and puts `C` alone
(fails, scores `0`) -- total `F=12` -- rather than, say, bundling all three
together (`C`'s overwhelming opposition sinks the whole thing, `F=0`).
