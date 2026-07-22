#!/usr/bin/env python3
"""
gen.py <testId> -- prints one instance of the "Subsystem-Robust Release Schedule"
problem to stdout. Deterministic: all randomness is seeded from testId only.

Instance = a precedence DAG built from disjoint job "tracks" (each track is a
linear chain job_1 -> job_2 -> ... -> job_L), M identical build machines, and K
published "correlated perturbation vectors": each names a subsystem (an explicit
subset of job ids -- a single decoy's PAYLOAD, or a genuinely multi-job group
pairing two DIFFERENT decoy chains' payloads together) and an inflation factor
> 1 that multiplies the duration of every job in that subsystem at once, should
that shock occur.

Two structural flavors are interleaved across the 10-case ladder (both keep every
machine continuously busy from t=0, so the objective genuinely depends on WHICH
job ends up sole occupant of the schedule's tail -- not just on total work):

  TRAP tests  (even testId): M "safe" bulk tracks (never perturbed) plus a few
  short "trigger -> payload" decoy tracks. Nominally each decoy's tail is BELOW
  every bulk track (so a nominal-critical-path scheduler ranks it last and it
  gets queued behind all the "safe" work); its published worst case is ABOVE
  every bulk track (so a worst-case-aware scheduler front-loads it instead).
  This is the family's core mechanism: correlated-perturbation-vectors +
  recomputed-makespan-worst combine to punish nominal-only prioritization.

  SKILL tests (odd testId): M bulk tracks plus a few DEEP chains whose total
  duration exceeds a bulk track's length even though each individual job in the
  chain is tiny. No perturbations at all here (K=0) -- this exercises plain
  precedence-schedule-construction (a scheduler that only looks at a job's own
  duration, ignoring the chain beneath it, queues the chain behind "big-looking"
  bulk work and pays for it).

Jobs are emitted in a global topological order (track order shuffled first), so
every predecessor id is strictly smaller than its successor id.
"""
import sys, random


def emit(tracks, M, subsets):
    """tracks: list of list[int] durations. subsets: list of (num, den, job_ids)."""
    job_durs = []
    job_preds = []
    nxt = 1
    for durs in tracks:
        prev = None
        for d in durs:
            jid = nxt; nxt += 1
            job_durs.append(d)
            job_preds.append([] if prev is None else [prev])
            prev = jid
    N = nxt - 1
    K = len(subsets)
    out = [f"{N} {M} {K}"]
    for i in range(N):
        preds = job_preds[i]
        out.append(f"{job_durs[i]} {len(preds)} " + " ".join(map(str, preds)))
    for num, den, ids in subsets:
        ids_sorted = sorted(set(ids))
        out.append(f"{num} {den} {len(ids_sorted)} " + " ".join(map(str, ids_sorted)))
    return "\n".join(out) + "\n"


def build_trap(rng, M, bulk_len, n_decoy, trig, payload, r_num, r_den,
               pair_num=3, pair_den=2):
    """M safe bulk tracks (1 job each, duration bulk_len) plus n_decoy 2-job
    "trigger -> payload" tracks. Requires (checked by caller): trig+payload <
    bulk_len (nominal tail hides each decoy behind bulk) and r_num/r_den*payload
    > bulk_len (its published worst case is more dangerous than any bulk
    track) -- the trap only bites with both margins in place.

    Two kinds of published shocks are emitted, both real "subsystems":
      - one singleton subset per decoy (its own worst case), and
      - genuinely CORRELATED multi-job subsets that each group two DIFFERENT
        decoys' payloads -- from two different sub-project chains -- under one
        shared (milder) factor, so some shocks hit more than one chain at
        once. A schedule that only reasons about single jobs in isolation, not
        about which chains a shock could jointly own, is exposed by these."""
    tracks = [[bulk_len] for _ in range(M)]
    decoy_start = len(tracks)
    for _ in range(n_decoy):
        tracks.append([trig, payload])

    perm = list(range(len(tracks)))
    rng.shuffle(perm)
    tracks_shuf = [tracks[p] for p in perm]
    # map old index -> new position to know which shuffled track is a decoy
    is_decoy_shuf = [perm[i] >= decoy_start for i in range(len(tracks))]

    # build ids while emitting, to attach subsets to the PAYLOAD job of each decoy
    job_durs = []
    job_preds = []
    track_job_ids = []
    nxt = 1
    for durs in tracks_shuf:
        ids = []
        prev = None
        for d in durs:
            jid = nxt; nxt += 1
            job_durs.append(d)
            job_preds.append([] if prev is None else [prev])
            ids.append(jid)
            prev = jid
        track_job_ids.append(ids)
    N = nxt - 1

    decoy_payloads = [track_job_ids[ti][-1] for ti in range(len(tracks_shuf)) if is_decoy_shuf[ti]]

    subsets = []
    for pid in decoy_payloads:
        subsets.append((r_num, r_den, [pid]))
    if len(decoy_payloads) >= 2:
        pairs = list(decoy_payloads)
        rng.shuffle(pairs)
        for i in range(0, len(pairs) - 1, 2):
            subsets.append((pair_num, pair_den, pairs[i:i + 2]))

    out = [f"{N} {M} {len(subsets)}"]
    for i in range(N):
        preds = job_preds[i]
        out.append(f"{job_durs[i]} {len(preds)} " + " ".join(map(str, preds)))
    for num, den, ids in subsets:
        out.append(f"{num} {den} {len(ids)} " + " ".join(map(str, ids)))
    return "\n".join(out) + "\n"


def build_skill(rng, M, bulk_len, n_chain, chain_len, unit_dur):
    """M bulk tracks (1 job, duration bulk_len) plus n_chain DEEP chains of
    chain_len jobs each of duration unit_dur, with chain_len*unit_dur clearly
    above bulk_len (so the chain is the true bottleneck) but unit_dur itself
    well below bulk_len (so a scheduler that looks only at a job's own size
    misjudges the chain's first job as unimportant). No perturbations."""
    tracks = [[bulk_len] for _ in range(M)]
    for _ in range(n_chain):
        tracks.append([unit_dur] * chain_len)
    rng.shuffle(tracks)
    return emit(tracks, M, [])


def gen(test_id):
    rng = random.Random(1000003 * test_id + 7)

    # ---- SKILL flavor: odd testIds, no perturbations, chain-depth blindness ----
    skill_cfgs = {
        1: dict(M=2, bulk_len=50, n_chain=1, chain_len=17, unit_dur=10),
        3: dict(M=4, bulk_len=80, n_chain=2, chain_len=25, unit_dur=10),
        5: dict(M=6, bulk_len=110, n_chain=3, chain_len=32, unit_dur=10),
        7: dict(M=8, bulk_len=140, n_chain=3, chain_len=40, unit_dur=10),
    }
    if test_id in skill_cfgs:
        return build_skill(rng, **skill_cfgs[test_id])

    # ---- TRAP flavor: even testIds, subsystem-risk blindness ----
    trap_cfgs = {
        2: dict(M=2, bulk_len=50, n_decoy=3, trig=1, payload=42, r_num=2, r_den=1,
                pair_num=3, pair_den=2),
        4: dict(M=3, bulk_len=68, n_decoy=2, trig=1, payload=60, r_num=5, r_den=2,
                pair_num=4, pair_den=3),
        6: dict(M=4, bulk_len=86, n_decoy=3, trig=1, payload=76, r_num=2, r_den=1,
                pair_num=3, pair_den=2),
        8: dict(M=6, bulk_len=115, n_decoy=4, trig=1, payload=102, r_num=5, r_den=2,
                pair_num=5, pair_den=3),
        9: dict(M=8, bulk_len=150, n_decoy=5, trig=1, payload=133, r_num=3, r_den=1,
                pair_num=4, pair_den=3),
        10: dict(M=9, bulk_len=168, n_decoy=6, trig=1, payload=149, r_num=5, r_den=2,
                 pair_num=3, pair_den=2),
    }
    if test_id in trap_cfgs:
        return build_trap(rng, **trap_cfgs[test_id])

    # fallback for any extra testIds
    return build_trap(rng, M=3, bulk_len=70, n_decoy=2, trig=1, payload=62, r_num=2, r_den=1)


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()
