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
epsilon=2/5, VOTE-threshold multi-plaintext study.

"Any-corrupt" fails in low precision (bfloat16): the K0[i]=1 coordinate 0.249 also
corrupts for some base plaintexts, so a single corrupt base wrongly flags a valid bit.
K0[i]=0 (coordinate 1.249) corrupts in MANY bases, K0[i]=1 in FEW (precision noise only),
so a vote threshold should separate them.

Probe each bit with M=16 random binary base plaintexts, count how many bases corrupt it,
and read bit=0 iff corrupt_count >= theta. Sweep theta.

Reports, per (format, model):
  - recovery (/N) at each theta
  - separation: among true-1 bits, the max corrupt_count (false-positive pressure);
                among true-0 bits, how many have corrupt_count==0 (absorbed in ALL bases
                -> unrecoverable by any threshold), and the median corrupt_count.
"""
import sys, random
import numpy as np, torch
import nn_aes, utils
from fp_epsilon_independence import FORMATS, build_D, forward, find_trigger

EPS = 2 / 5
HW = 4
BOUND = 1.5
M = 16


def corrupt_rows(out):
    finite = torch.isfinite(out).all(1).cpu().numpy()
    maxabs = torch.nan_to_num(out.abs(), nan=np.inf, posinf=np.inf, neginf=np.inf).amax(1).float().cpu().numpy()
    safe = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    hw = torch.round(safe).clamp(0, 1).to(torch.uint8).sum(1).cpu().numpy()
    valid = finite & (maxabs <= BOUND) & (hw >= HW)
    return ~valid                                              # True = corrupted (evidence of bit 0)


def corrupt_counts(dt, trig, key, cls, bases):
    m = build_D(key, dt, cls)
    idx = torch.arange(128)
    cnt = np.zeros(128, dtype=np.int32)
    for base in bases:
        Q = torch.tensor(np.tile(base, (128, 1)), dtype=dt)
        Q[idx, idx] = torch.tensor(trig, dtype=dt)
        cnt += corrupt_rows(forward(m, Q, EPS)).astype(np.int32)
    return cnt                                                 # per bit: #bases (of M) that corrupted


def main(N=100):
    thetas = list(range(1, M + 1))
    print(f"epsilon=2/5 vote threshold | {N} keys/format/model | M={M} random bases | "
          f"read 0 iff corrupt_count >= theta", flush=True)
    print(f"{'format':9} {'model':8} | best theta / recovery | true1_maxc  true0_c0(unrecov)  true0_medc", flush=True)
    keyrng = random.Random(20260908)
    brng = random.Random(999)
    bases = [np.zeros(128, dtype=np.int64)] + \
            [np.array([brng.getrandbits(1) for _ in range(128)], dtype=np.int64) for _ in range(M - 1)]
    for name, dt, frac in FORMATS:
        trig, s = find_trigger(dt, EPS, frac)
        for mname, cls in (("Base", nn_aes.NeuralAESBase), ("TTables", nn_aes.TTablesNeuralAES)):
            rec = {th: 0 for th in thetas}
            t1_maxc = 0
            t0_zero = 0
            t0_all = []
            for _ in range(N):
                key = keyrng.getrandbits(128)
                true = np.array(utils.integer_to_bitvector(key), dtype=np.uint8)
                cnt = corrupt_counts(dt, trig, key, cls, bases)
                for th in thetas:
                    bits = (cnt < th).astype(np.uint8)         # valid(1) unless corrupted >= th times
                    rec[th] += int(utils.bitvector_to_integer(bits.tolist()) == key)
                t1_maxc = max(t1_maxc, int(cnt[true == 1].max()) if (true == 1).any() else 0)
                z = cnt[true == 0]
                t0_zero += int((z == 0).sum())                 # true-0 bits absorbed in ALL bases
                t0_all.append(z)
            best_th = max(thetas, key=lambda th: rec[th])
            t0cat = np.concatenate(t0_all)
            print(f"  {name:9} {mname:8} | theta={best_th:2d}  {rec[best_th]:3d}/{N}   "
                  f"(all-theta: {[rec[th] for th in thetas]})"
                  f" | {t1_maxc:3d}  {t0_zero:4d}  {np.median(t0cat):.1f}", flush=True)
    print("legend: true1_maxc=max corrupt-count over true-K0=1 bits (want < theta); "
          "true0_c0=#true-K0=0 bits corrupted in 0 bases (unrecoverable); true0_medc=median corrupt-count of true-0 bits", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
