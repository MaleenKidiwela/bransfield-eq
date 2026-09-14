"""Drop S picks that a station S delay would push BEFORE the P pick.

NLLoc applies LOCDELAY as obs_time -= delay. When (tS - dS) - (tP - dP) <= 0 the S arrival
lands before its P companion and NLLoc aborts the WHOLE event with
"ERROR: cannot find companion arrival" (verified on 2019-01-11 14:05:44: BRA13 S-P 0.286 s,
S delay 0.43 s -> fails; 0.20 s -> locates). 4.6% of the sample events were lost that way.
Physically such a pick pair is inconsistent with a constant station S delay (event too close
to the station for the full sediment delay), so the S pick is the one to drop.
Writes <out>.obs and prints how many S picks / events were affected.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def load_delays(p):
    d = {}
    for l in Path(p).read_text().split("\n"):
        f = l.split()
        if len(f) == 5 and f[0] == "LOCDELAY": d[(f[1], f[2])] = float(f[4])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obs"); ap.add_argument("delays"); ap.add_argument("out")
    ap.add_argument("--margin", type=float, default=0.05, help="min corrected S-P to keep (s)")
    a = ap.parse_args()
    dl = load_delays(a.delays)
    blocks, cur = [], []
    for line in Path(a.obs).read_text().split("\n"):
        if line.strip() == "":
            if cur: blocks.append(cur); cur = []
        else:
            cur.append(line)
    if cur: blocks.append(cur)
    n_drop = n_ev = 0; out = []
    for b in blocks:
        rows = []
        for l in b:
            f = l.split()
            if len(f) < 9 or f[0].startswith("#"): rows.append((l, None)); continue
            sta, pha, ymd, hm, sec = f[0], f[4], f[6], f[7], float(f[8])
            t = int(hm[:2]) * 3600 + int(hm[2:]) * 60 + sec + (int(ymd) % 100) * 86400   # same day within an event
            rows.append((l, (sta, pha, t)))
        P = {k[0]: k[2] - dl.get((k[0], "P"), 0.0) for _, k in rows if k and k[1] == "P"}
        keep, dropped = [], 0
        for l, k in rows:
            if k and k[1] == "S" and k[0] in P and (k[2] - dl.get((k[0], "S"), 0.0)) - P[k[0]] <= a.margin:
                dropped += 1; continue
            keep.append(l)
        n_drop += dropped; n_ev += dropped > 0
        out.append("\n".join(keep))
    Path(a.out).write_text("\n\n".join(out) + "\n\n")
    print(f"{a.obs}: {len(blocks):,} events; dropped {n_drop:,} S picks in {n_ev:,} events "
          f"({n_ev/len(blocks)*100:.1f}%) whose corrected S-P <= {a.margin} s -> {a.out}")


if __name__ == "__main__":
    main()
