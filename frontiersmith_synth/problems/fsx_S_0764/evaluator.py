#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0764 -- "Two Strata of Cold: A Polar Station's Storage
Portfolio" (family: drought-tail-storage-portfolio; format B, quality-metric; theme:
off-grid polar research station power policy).

THEME.  You run the power schedule for an off-grid polar research station over `T`
discrete ticks.  Each tick has a public renewable generation `gen_t` (wind) and a
public station load `load_t`.  There is NO grid: whatever the station consumes must
come from generation or storage THAT tick.

MULTI-TIMESCALE BUFFERING (mechanism 1).  Two stores back up the bus:
  - BATTERY: small capacity, high round-trip efficiency, fast -- the everyday buffer
    for ordinary hour-to-hour noise.
  - FUEL STORE (electrolysis -> synthetic fuel -> fuel cell): huge capacity, LOSSY
    round-trip efficiency -- built to ride out multi-tick generation DROUGHTS that the
    small battery physically cannot hold enough energy to cover.
Charging a store draws bus power (only an `eta_in` fraction is actually stored);
discharging draws stored energy (only an `eta_out` fraction reaches the bus).  Both
are rate-limited per tick and capacity-limited in total.

POLICY-PROGRAM DESIGN (mechanism 2).  A candidate is a program that reads the WHOLE
public trace (`load`, `gen` for all T ticks, plus both stores' parameters) and emits
one schedule covering the entire horizon: four length-T vectors describing what to
draw from / push into each store, every tick, in one shot.

TAIL-EVENT HEDGING (mechanism 3, the innovation hook).  A handful of instances
contain planted multi-tick DROUGHTS (gen collapses far below load for many
consecutive ticks).  The two stores serve different probability strata: the battery
soaks up routine variance, but during a drought only the fuel store's huge capacity
can matter.  Every *efficiency* metric says a store that sits idle 80-90% of the time
is a waste -- but that idleness IS its job: an insurance policy that is "almost
always idle" is not wasted, it is doing exactly what it is for.  The catch is that
the lossy store needs a LONG, EARLY lead-in to accumulate enough stored energy
(its round trip is far worse than the battery's), so hedging only pays off when it is
planned well before the drought hits -- and only for droughts genuinely beyond what
the fast little battery could ever cover alone.

BALANCE RULE (deterministic, evaluated by THIS scorer, not the candidate).  Each tick,
in order: (1) requested discharges are clamped to the rate limit and to what is
actually stored; (2) available power = gen_t + delivered discharge; (3) load is
served FIRST, up to available power (any shortfall is "unserved" -> penalized);
(4) only power left AFTER load is served may charge a store (battery gets first call,
then fuel), clamped to rate and remaining room; anything still left over is curtailed
(lost, but not penalized beyond the missed charging opportunity).

OBJECTIVE (maximize) per instance:
    obj = sum_t served_t  -  BLACKOUT_COEF * sum_t unserved_t
where `served_t + unserved_t == load_t` every tick.  Quoting no storage action at all
(all four vectors zero) still serves whatever generation alone can cover -- that is
the evaluator's own `baseline(inst)` reference, not a score of zero.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int,
             "load": [load_0 ... load_{T-1}], "gen": [gen_0 ... gen_{T-1}],
             "battery": {"cap":f, "rate":f, "eta_in":f, "eta_out":f},
             "fuel":    {"cap":f, "rate":f, "eta_in":f, "eta_out":f},
             "blackout_coef": f}
  stdout: ONE JSON object:
            {"bc": [...T...], "bd": [...T...],   # battery charge / discharge request
             "fc": [...T...], "fd": [...T...]}   # fuel charge / discharge request
  VALID iff each of bc,bd,fc,fd is a list of exactly T finite numbers >= 0 (no
  NaN/inf/bool/negative).  Any violation, crash, timeout, or non-JSON -> 0.0.
  Requests are CLAMPED by the scorer to rate limits and to physically available
  power / stored energy -- an infeasible REQUEST is not rejected, it is honestly
  resolved, same as the reference solutions below.

SCORING (deterministic; no wall-time).  Per instance, let `b = baseline(inst)` be the
objective of submitting all-zero vectors (no storage action at all -- generation
serves load directly whenever it can) and `hi(inst)` a generous, UNREACHABLE upper
bound (load served in full MINUS whatever deficit would remain even with both stores
pre-charged and discharging at maximum simultaneous rate the whole time -- a bound
that ignores provisioning-time limits entirely, so no honest policy attains it):
    r = clamp( 0.1 + 0.9 * (obj - b) / (hi - b), 0, 1 )
Doing nothing scores exactly 0.1.  The final score is the mean of r over 10 fixed
seeded instances: calm (no drought) instances, single-drought traps of varying
length/severity/lead-time, and held-out twin-drought (two droughts, one instance)
generalization instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED
SUBPROCESS via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The
objective and the normalization are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

GAIN = 1.08   # normalization slack for the unreachable upper bound


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- physics / objective --------------------------
def simulate(inst, bc, bd, fc, fd):
    """Replay the deterministic dispatch episode; return the objective."""
    T = inst["T"]
    load = inst["load"]; gen = inst["gen"]
    bat = inst["battery"]; fuel = inst["fuel"]
    bcap = bat["cap"]; brate = bat["rate"]; bin_ = bat["eta_in"]; bout = bat["eta_out"]
    fcap = fuel["cap"]; frate = fuel["rate"]; fin_ = fuel["eta_in"]; fout = fuel["eta_out"]
    blk = inst["blackout_coef"]

    blev = 0.0
    flev = 0.0
    served_total = 0.0
    penalty = 0.0
    for t in range(T):
        bc_t = bc[t] if bc[t] > 0.0 else 0.0
        if bc_t > brate: bc_t = brate
        bd_t = bd[t] if bd[t] > 0.0 else 0.0
        if bd_t > brate: bd_t = brate
        fc_t = fc[t] if fc[t] > 0.0 else 0.0
        if fc_t > frate: fc_t = frate
        fd_t = fd[t] if fd[t] > 0.0 else 0.0
        if fd_t > frate: fd_t = frate
        # clamp discharge requests to available stored energy
        if bd_t > blev: bd_t = blev
        if fd_t > flev: fd_t = flev
        blev -= bd_t
        flev -= fd_t
        delivered = bout * bd_t + fout * fd_t
        avail = gen[t] + delivered
        L = load[t]
        served = avail if avail < L else L
        unserved = (L - served) if L > served else 0.0
        remaining = avail - served
        # remaining power (if any) may charge battery first, then fuel
        b_room = (bcap - blev) / bin_ if bin_ > 0.0 else 0.0
        bc_t2 = bc_t
        if bc_t2 > remaining: bc_t2 = remaining
        if bc_t2 > b_room: bc_t2 = b_room
        if bc_t2 < 0.0: bc_t2 = 0.0
        blev += bin_ * bc_t2
        remaining -= bc_t2
        f_room = (fcap - flev) / fin_ if fin_ > 0.0 else 0.0
        fc_t2 = fc_t
        if fc_t2 > remaining: fc_t2 = remaining
        if fc_t2 > f_room: fc_t2 = f_room
        if fc_t2 < 0.0: fc_t2 = 0.0
        flev += fin_ * fc_t2
        remaining -= fc_t2
        served_total += served
        penalty += blk * unserved
    return served_total - penalty


def baseline(inst):
    T = inst["T"]
    z = [0.0] * T
    return simulate(inst, z, z, z, z)


def _hi(inst):
    fuel = inst["fuel"]; bat = inst["battery"]
    maxpow = fuel["rate"] * fuel["eta_out"] + bat["rate"] * bat["eta_out"]
    unavoid = 0.0
    for t in range(inst["T"]):
        d = inst["load"][t] - inst["gen"][t]
        if d > maxpow:
            unavoid += d - maxpow
    return GAIN * (sum(inst["load"]) - unavoid)


# ----------------------------- instance family ------------------------------
BATTERY = {"cap": 15.0, "rate": 6.0, "eta_in": 0.9, "eta_out": 0.9}
FUEL = {"cap": 300.0, "rate": 10.0, "eta_in": 0.6, "eta_out": 0.6}
BLACKOUT_COEF = 2.0


def _base_series(seed, T, load0, gen0, load_amp, gen_amp):
    ni = _rng(seed)
    load = [load0 + ni(-load_amp, load_amp) / 10.0 for _ in range(T)]
    gen = [gen0 + ni(-gen_amp, gen_amp) / 10.0 for _ in range(T)]
    return load, gen


def _plant_drought(gen, seed, start, dur, level, amp):
    ni = _rng(seed)
    n = len(gen)
    for t in range(start, min(start + dur, n)):
        g = level + ni(-amp, amp) / 10.0
        gen[t] = g if g > 0.0 else 0.0


def _make_instance(name, seed, T, droughts):
    load, gen = _base_series(seed, T, 10.0, 16.0, 10, 20)
    for (start, dur, level, amp, dseed) in droughts:
        _plant_drought(gen, dseed, start, dur, level, amp)
    return {"name": name, "T": T, "load": load, "gen": gen,
            "battery": dict(BATTERY), "fuel": dict(FUEL),
            "blackout_coef": BLACKOUT_COEF}


def _build_instances():
    T = 70
    specs = [
        ("calm1",          501, []),
        ("calm2",          502, []),
        ("drought_short",  511, [(45, 10, 4.0, 5, 611)]),
        ("drought_med1",   512, [(40, 12, 4.0, 5, 612)]),
        ("drought_med2",   513, [(35, 13, 3.5, 4, 613)]),
        ("drought_long1",  514, [(30, 16, 3.5, 4, 614)]),
        ("drought_long2",  515, [(38, 16, 3.0, 4, 615)]),
        ("drought_early",  516, [(5, 10, 3.5, 4, 616)]),
        ("twin1",          521, [(18, 10, 3.5, 4, 621), (46, 10, 3.5, 4, 622)]),
        ("twin2",          522, [(15, 9, 4.0, 4, 623), (42, 12, 3.0, 4, 624)]),
    ]
    return [_make_instance(name, seed, T, dr) for name, seed, dr in specs]


# ----------------------------- validation ------------------------------------
def _vec(answer, key, T):
    v = answer.get(key)
    if not isinstance(v, list) or len(v) != T:
        return None
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")) or x < 0.0:
            return None
        out.append(x)
    return out


def _valid_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    T = inst["T"]
    bc = _vec(answer, "bc", T)
    bd = _vec(answer, "bd", T)
    fc = _vec(answer, "fc", T)
    fd = _vec(answer, "fd", T)
    if bc is None or bd is None or fc is None or fd is None:
        return None
    return bc, bd, fc, fd


def _public(inst):
    return {"name": inst["name"], "T": inst["T"],
            "load": list(inst["load"]), "gen": list(inst["gen"]),
            "battery": dict(inst["battery"]), "fuel": dict(inst["fuel"]),
            "blackout_coef": inst["blackout_coef"]}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = baseline(inst)
        h = _hi(inst)
        denom = h - b
        if denom <= 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            parsed = _valid_answer(inst, ans)
            if parsed is None:
                vec.append(0.0)
                continue
            bc, bd, fc, fd = parsed
            obj = simulate(inst, bc, bd, fc, fd)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj - b) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
