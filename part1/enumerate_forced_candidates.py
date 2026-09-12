#!/usr/bin/env python3
# Copyright (C) 2026 the authors (anonymized for review)
#
# This file is part of the artifact for "Cryptanalysis of Deep Neural
# Cryptography: Second Round Key Recovery on the Unprotected Implementation
# and a Floating-Point Attack on the Protected Implementation of AES".
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
enumerate_forced_candidates.py

Reproduces the structural numbers of the paper's Appendix A: the forced wrong
candidates of the K1 collision oracle, their score 7*ell, the bound ell <= 2,
and the frequency of the ell = 2 case.

A source byte r of the target column feeds target rows r and r-1 through the
MixColumns coefficients 0x02 and 0x03, and rows r-2 and r-3 through the two
0x01 coefficients. Writing u = MC^{-1}(g) for the four S-box outputs that
MixColumns combines into the column, u_r = 0x00 makes the 0x02 and 0x03 products
vanish, so those two targets receive no perturbation and are absorbed whatever
their value. A probe on source r then collides as soon as the two coefficient-0x01
targets equal 0x52 and |D_j| >= 2 -- a condition on only three of the four bytes
of g, which is why wrong candidates can collide at all.

This script enumerates that condition exactly (no sampling over candidates: for
each r the two target conditions fix two bytes of g and u_r = 0x00 is one linear
equation over GF(2^8) in the remaining two, giving 256 solutions per r). It uses
only the S-box table, so it runs in seconds for the default 20,000 columns (a few
minutes for 200,000) and needs no GPU and no network.

    python enumerate_forced_candidates.py                 # defaults to 20000 columns
    python enumerate_forced_candidates.py --columns 50000 --seed 20260907

    # the census keys of aes256_attack_gpu.py (SEED=2026..2029): predicted forced
    # candidates, ell=2 pairs, exact-real true score; optionally compared with one
    # census's col{c}_hist.txt (HIST=1) and col{c}_hits.txt (RECORD_ALL, COLL_MIN<=7)
    python enumerate_forced_candidates.py --census-seeds 2026,2027,2028,2029
    python enumerate_forced_candidates.py --census-seeds 2028 --hist-dir run_seed2028
"""
import argparse
import random
from collections import Counter

from utils import SBOX, SBOX_INV

S, SI = list(SBOX), list(SBOX_INV)
DZ = 0x52                                   # SBOX^{-1}(0x00)
assert S[DZ] == 0x00

wt = lambda x: bin(x).count("1")


def gmul(a, b):
    """Multiplication in GF(2^8) with the AES polynomial."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def ginv(a):
    for x in range(1, 256):
        if gmul(a, x) == 1:
            return x
    raise ValueError("0 has no inverse")


INV_MC = [[0x0E, 0x0B, 0x0D, 0x09],
          [0x09, 0x0E, 0x0B, 0x0D],
          [0x0D, 0x09, 0x0E, 0x0B],
          [0x0B, 0x0D, 0x09, 0x0E]]

# Precomputed tables: the enumeration below is inner-loop bound, and everything
# it needs depends only on the S-box and on GF(2^8) multiplication by constants.
MULT = {c: [gmul(c, x) for x in range(256)] for c in (0x02, 0x03, 0x09, 0x0B, 0x0D, 0x0E)}
INV_DIAG = [ginv(INV_MC[r][r]) for r in range(4)]
MUL_INV_DIAG = [[gmul(INV_DIAG[r], x) for x in range(256)] for r in range(4)]

# N_GE2[s] = #{j : |D_j| >= 2} for the byte whose S-box output is s, i.e. the
# number of the eight probes on that source that survive the n >= 2 test.
N_GE2 = [sum(1 for j in range(8) if wt(s ^ S[SI[s] ^ (1 << j)]) >= 2) for s in range(256)]
# A source with u_r = 0x00 has its 0x02/0x03 targets absorbed for free, so it
# contributes exactly this many collisions.
ABSORBING_SOURCE_SCORE = N_GE2[0x00]


def inv_mc(g):
    return [MULT[INV_MC[r][0]][g[0]] ^ MULT[INV_MC[r][1]][g[1]] ^
            MULT[INV_MC[r][2]][g[2]] ^ MULT[INV_MC[r][3]][g[3]] for r in range(4)]


def forced_candidates(K1c):
    """Every wrong candidate that collides with no avalanche, with its ell and score."""
    hits = {}
    for r in range(4):
        a, b = (r + 1) % 4, (r + 2) % 4      # the two coefficient-0x01 targets of source r
        g = [0, 0, 0, 0]
        g[a] = K1c[a] ^ DZ
        g[b] = K1c[b] ^ DZ
        rhs = MULT[INV_MC[r][a]][g[a]] ^ MULT[INV_MC[r][b]][g[b]]
        d = (r + 3) % 4
        md, mr = MULT[INV_MC[r][d]], MUL_INV_DIAG[r]
        for gd in range(256):
            g[d] = gd
            g[r] = mr[rhs ^ md[gd]]
            hits.setdefault(tuple(g), []).append(r)
    gstar = tuple(K1c[i] ^ DZ for i in range(4))
    hits.pop(gstar, None)                    # the true candidate is not a wrong one
    # Every source counted here has u_r = 0x00 by construction, so each of them
    # contributes ABSORBING_SOURCE_SCORE collisions.
    return {g: (len(rs), len(rs) * ABSORBING_SOURCE_SCORE) for g, rs in hits.items()}


# T-tables network: the S-box and the 0x02/0x03 multiplications are one fused corner block,
# so a probe enters it with n = 1 and the 0x02/0x03 targets receive eta on the bits of
# 0x02*D_j and 0x03*D_j (never nothing). A probe on a source with S-box output s collides
# at the true candidate iff min(|D_j|, wt(0x02*D_j), wt(0x03*D_j)) >= 2, for every s.
N_TT = [sum(1 for j in range(8)
            if min(wt(d := s ^ S[SI[s] ^ (1 << j)]), wt(MULT[0x02][d]), wt(MULT[0x03][d])) >= 2)
        for s in range(256)]


def true_score(K1c, model="base"):
    """Collision count of the true candidate under the exact-real criterion."""
    u = inv_mc([K1c[i] ^ DZ for i in range(4)])
    if model == "ttables":
        return sum(N_TT[s] for s in u)
    total = 0
    for s in u:
        if s == 0x00:
            total += ABSORBING_SOURCE_SCORE
        elif wt(MULT[0x02][s]) >= 2 and wt(MULT[0x03][s]) >= 2:
            total += N_GE2[s]
    return total


def census_key_columns(seed):
    """The four K1 columns of the census key of aes256_attack_gpu.py for this SEED.
    The key is random.Random(SEED).getrandbits(256); for AES-256, RoundKey0||RoundKey1
    is the master key itself, so K1 is bytes 16..31 and column c is bytes 16+4c..19+4c."""
    kb = random.Random(seed).getrandbits(256).to_bytes(32, "big")
    return [list(kb[16 + 4 * c: 20 + 4 * c]) for c in range(4)]


def cand_int(g):
    """Candidate integer as scanned by aes256_attack_gpu.py (row 0 in the top byte)."""
    return (g[0] << 24) | (g[1] << 16) | (g[2] << 8) | g[3]


def read_hist(path):
    """col{c}_hist.txt written by aes256_attack_gpu.py with HIST=1: 33 counts, score 0..32."""
    with open(path, encoding="utf-8") as fh:
        head = fh.readline(); body = fh.readline()
    scanned = int(head.split("scanned=")[1].split()[0]); limit = int(head.split("limit=")[1].split()[0])
    hist = [int(x) for x in body.split("hist=[")[1].split("]")[0].split(",")]
    return scanned, limit, hist


def read_hits(path):
    """col{c}_hits.txt (RECORD_ALL): one 'c=0x........ ... enc=N ...' line per recorded candidate."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("c=0x"):
                c = int(line.split()[0][2:], 16); en = int(line.split("enc=")[1].split()[0])
                out[c] = en
    return out


def check_census(seeds, hist_dir, hits_dir, tau, model="base"):
    """Predict the forced-candidate structure of the census keys and, when the census
    files are given, compare with what the full 2^32 scan observed."""
    import os
    for seed in seeds:
        cols = census_key_columns(seed)
        print(f"\ncensus seed {seed} ({model} network): K1 = {' '.join(''.join(f'{b:02x}' for b in c) for c in cols)}")
        for c, K1c in enumerate(cols):
            fc = forced_candidates(K1c) if model == "base" else {}   # T-tables: no target is absorbed for free
            ell2 = [g for g, (e, _) in fc.items() if e >= 2]
            at7 = sum(1 for e, _ in fc.values() if e == 1)
            ts = true_score(K1c, model)
            gstar = tuple(k ^ DZ for k in K1c)
            print(f"  col{c}: g*=0x{cand_int(gstar):08x} exact-real true score={ts}  "
                  f"forced wrong candidates={len(fc)} (score 7: {at7}, score 14: {len(ell2)})"
                  + (f"  ell=2 at {[f'0x{cand_int(g):08x}' for g in ell2]}" if ell2 else ""))
            if hist_dir:
                path = os.path.join(hist_dir, f"col{c}_hist.txt")
                if os.path.exists(path):
                    scanned, limit, hist = read_hist(path)
                    cov = "" if scanned >= limit else f" ({100*scanned/limit:.1f}% scanned)"
                    print(f"         census hist{cov}: score7={hist[7]} "
                          f"(predicted {at7}), score14={hist[14]} (predicted {len(ell2)}), "
                          f"scores>=tau={sum(hist[tau:])} ; nonzero scores: "
                          + ", ".join(f"{s}:{hist[s]}" for s in range(33) if hist[s]))
            if hits_dir:
                path = os.path.join(hits_dir, f"col{c}_hits.txt")
                if os.path.exists(path):
                    hits = read_hits(path)
                    pred = {cand_int(g): sc for g, (_, sc) in fc.items()}
                    both = [x for x in hits if x in pred]
                    unexpl = [x for x in hits if x not in pred and x != cand_int(gstar)]
                    print(f"         hits file: {len(hits)} recorded; {len(both)} are predicted forced candidates "
                          f"(score match: {sum(hits[x] == pred[x] for x in both)}/{len(both)}); "
                          f"{len(unexpl)} not explained by the enumeration"
                          + (f": {[f'0x{x:08x}({hits[x]})' for x in unexpl[:8]]}" if unexpl else "")
                          + f"; g* {'recorded' if cand_int(gstar) in hits else 'not recorded'}"
                          + (f" with score {hits[cand_int(gstar)]}" if cand_int(gstar) in hits else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--census-seeds", type=str, default=None,
                    help="comma-separated SEEDs of aes256_attack_gpu.py runs, e.g. 2026,2027,2028,2029: "
                         "predict their forced candidates instead of sampling random columns")
    ap.add_argument("--hist-dir", type=str, default=None,
                    help="directory holding col{c}_hist.txt of one census (HIST=1) to compare against")
    ap.add_argument("--hits-dir", type=str, default=None,
                    help="directory holding col{c}_hits.txt of one census (RECORD_ALL, COLL_MIN<=7) to compare against")
    ap.add_argument("--model", type=str, default="base", choices=["base", "ttables"],
                    help="base network (default) or the T-tables network, whose fused S-box/0x02/0x03 block "
                         "has no forced wrong candidates and a different true-candidate criterion")
    a = ap.parse_args()

    if a.census_seeds:
        check_census([int(x) for x in a.census_seeds.split(",")], a.hist_dir, a.hits_dir, a.tau, a.model)
        return

    rng = random.Random(a.seed)
    distinct, ell_hist, score_hist, true_hist = Counter(), Counter(), Counter(), Counter()
    with_ell2 = 0
    for _ in range(a.columns):
        K1c = [rng.randrange(256) for _ in range(4)]
        fc = forced_candidates(K1c) if a.model == "base" else {}
        distinct[len(fc)] += 1
        ells = [e for e, _ in fc.values()]
        for e, sc in fc.values():
            ell_hist[e] += 1
            score_hist[sc] += 1
        with_ell2 += any(e >= 2 for e in ells)
        true_hist[true_score(K1c, a.model)] += 1

    n = a.columns
    print(f"columns: {n}   seed: {a.seed}   tau: {a.tau}   model: {a.model}\n")
    print("distinct forced wrong candidates per column:")
    for k in sorted(distinct, reverse=True):
        print(f"  {k:5d} : {distinct[k]:7d}  ({100*distinct[k]/n:6.3f}%)")
    print(f"\nper-candidate ell: {dict(sorted(ell_hist.items()))}")
    print(f"per-candidate score: {dict(sorted(score_hist.items()))}")
    print(f"  -> every wrong candidate scores 7*ell, and ell <= 2")

    exact = 4 / 256 - 6 / 256 ** 2 + 4 / 256 ** 3 - 1 / 256 ** 4
    print(f"\ncolumns carrying an ell=2 candidate: {with_ell2}/{n} "
          f"({100*with_ell2/n:.3f}%)   closed form {100*exact:.3f}%")

    below = sum(c for s, c in true_hist.items() if s < a.tau)
    print(f"\ntrue-candidate score (exact-real criterion):")
    for s in sorted(true_hist):
        print(f"  {s:5d} : {true_hist[s]:7d}  ({100*true_hist[s]/n:6.3f}%)")
    print(f"  below tau={a.tau}: {below}/{n} ({100*below/n:.3f}%)")
    print("\nNote: this is the exact-real criterion. The model runs in float16, and")
    print("Appendix A of the paper reports the residual between the two.")


if __name__ == "__main__":
    main()
