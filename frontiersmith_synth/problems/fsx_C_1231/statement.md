# Undocumented Logs: Template Extraction Under an Overgeneralization Penalty

You are handed **N** raw log records from a system nobody documented. Every
record is exactly **W** whitespace-separated fields (a fixed-width format —
you just don't know which fields are literal tags and which are values that
change record to record).

Your job: propose a set of **templates**. Each template is a length-`W`
pattern where every position is either a **literal token** (a claim: "every
record using this template has exactly this string here") or a **wildcard**
`*` (a claim: "this position varies; no promise about its value"). You then
assign every record to exactly one of your templates.

A template's literal claims must actually hold: if template `t` says
position `p` equals `"connect"`, then **every** record assigned to `t` must
literally have `"connect"` at position `p`. Any broken claim makes your
whole submission infeasible.

## Why wildcards aren't free

It is tempting to wildcard every position that ever differs somewhere in
the corpus — then one giant template matches everything. But a template
that explains nothing is worthless: for every record it covers, a wildcard
position still has to be told an actual value. Explaining a corpus costs:

  `cost(t) = W + n_t * v_t`

for a template `t` used by `n_t` records with `v_t` wildcard positions — `W`
to state the template once, plus `n_t * v_t` to state every wildcard's
actual value on every record it covers. Your total score is driven by

  `F = sum over templates actually used of cost(t)`   (**lower is better**)

against the do-nothing baseline `B = N * W` (one private, all-literal
template per record — the cost of explaining nothing at all). A template
that is *too general* makes `F` grow **past** `B`; one that is *too
specific* (one template per record) exactly **equals** `B`. Real structure
— a handful of templates whose wildcards mark only what genuinely varies —
is the only way to push `F` well below `B`.

## Input (stdin)

```
N W
line_1
...
line_N
```

`1 <= N <= 700`, `1 <= W <= 10`. Every `line_i` has exactly `W`
whitespace-separated tokens (no token is ever the literal string `*`).

## Output (stdout)

```
T
tmpl_1
...
tmpl_T
a_1 a_2 ... a_N
```

`1 <= T <= N` templates, each `W` tokens (a literal string, or `*`). Then
`a_1 .. a_N`: the 1-indexed template id assigned to record `i`, in input
order. All `1 + T*W + N` tokens are read positionally; line breaks don't
matter.

## Scoring

The checker validates every literal claim strictly (any violation → `Ratio:
0.0`), computes `F` as above, and reports

  `Ratio = min(1.0, B / (10*F))`

## Illustrative example (form only — small, hand-worked)

`N=4, W=3`: `AUTH login 501` / `AUTH login 777` / `NET ping 12` / `NET ping
88`. `B = 4*3 = 12`.

- One template per line (`T=4`, no wildcards): `F=12=B` → **Ratio 0.1000**.
- Two templates, `["AUTH","login","*"]` and `["NET","ping","*"]`
  (`n=2, v=1` each): `F = (3+2) + (3+2) = 10` → **Ratio 0.1200**.
- One universal template `["*","*","*"]` covering all 4 (matches
  everything): `F = 3 + 4*3 = 15 > B` → **Ratio 0.0800** — *worse* than
  doing nothing, because every record now needs all 3 positions restated.

## Structure you can exploit

Position 0 often looks like a stable tag, but it is not always a reliable
grouping key — some records from genuinely different templates share the
same value there while differing almost everywhere else, and some records
sharing no early field still belong together. Nothing in the input tells
you in advance which position (if any) actually separates the true
templates; the cost formula above is the only honest signal, and it rewards
whichever grouping keeps both the template count and the per-record
wildcard burden low simultaneously — neither one alone.

## Constraints

`N <= 700`, `W <= 10`. Time limit 5 s, memory 512 MB. Scoring is exact
integer arithmetic and fully deterministic.
