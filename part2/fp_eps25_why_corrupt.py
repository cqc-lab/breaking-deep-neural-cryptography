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
Why do K0[i]=1 rows (coordinate 0.249) get flagged 'corrupt' under some base plaintexts?
Categorize the reason (non-finite / bound-exceed>1.5 / low-HW) and, for bound cases,
how far over 1.5 the value is (marginal => rounding noise; large => real amplification).
Contrast bfloat16 (expected: many, marginal) with float16 (expected: ~none).
"""
import random
import numpy as np, torch
import nn_aes, utils
from fp_epsilon_independence import build_D, forward, find_trigger

EPS = 2 / 5
HW = 4
BOUND = 1.5
M = 16
cls = nn_aes.NeuralAESBase


def analyze(name, dt, frac, N=20):
    trig, s = find_trigger(dt, EPS, frac)
    brng = random.Random(999)
    bases = [np.zeros(128, np.int64)] + [np.array([brng.getrandbits(1) for _ in range(128)], np.int64) for _ in range(M - 1)]
    keyrng = random.Random(20260908)
    reasons = {"nonfinite": 0, "bound": 0, "lowHW": 0}
    bound_over = []        # (max|out| - 1.5) for bound cases
    n1_total = n1_corrupt = 0
    binary_ok = True
    idx = torch.arange(128)
    for _ in range(N):
        key = keyrng.getrandbits(128)
        true = np.array(utils.integer_to_bitvector(key), np.uint8)
        one = (true == 1)
        m = build_D(key, dt, cls)
        for base in bases:
            if not set(np.unique(base)).issubset({0, 1}):
                binary_ok = False
            Q = torch.tensor(np.tile(base, (128, 1)), dtype=dt)
            Q[idx, idx] = torch.tensor(trig, dtype=dt)
            out = forward(m, Q, EPS)
            finite = torch.isfinite(out).all(1).cpu().numpy()
            mx = torch.nan_to_num(out.abs(), nan=np.inf, posinf=np.inf, neginf=np.inf).amax(1).float().cpu().numpy()
            safe = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            hw = torch.round(safe).clamp(0, 1).to(torch.uint8).sum(1).cpu().numpy()
            # restrict to true-K0=1 rows
            f, mxa, h = finite[one], mx[one], hw[one]
            n1_total += int(one.sum())
            r_nonfin = ~f
            r_bound = f & (mxa > BOUND)
            r_lowhw = f & (mxa <= BOUND) & (h < HW)
            reasons["nonfinite"] += int(r_nonfin.sum())
            reasons["bound"] += int(r_bound.sum())
            reasons["lowHW"] += int(r_lowhw.sum())
            n1_corrupt += int((r_nonfin | r_bound | r_lowhw).sum())
            bound_over.extend((mxa[r_bound] - BOUND).tolist())
    bo = np.array(bound_over) if bound_over else np.array([0.0])
    print(f"{name:9} STEP={s:.3f} | K0=1 rows: {n1_total}, corrupt {n1_corrupt} "
          f"({100*n1_corrupt/max(n1_total,1):.1f}%) | reasons {reasons} | "
          f"bound-over 1.5 by: min={bo.min():.3f} med={np.median(bo):.3f} max={bo.max():.3f} | "
          f"offdiag-binary={binary_ok}", flush=True)


def main():
    print(f"epsilon=2/5 | why K0=1 rows corrupt | model=NeuralAESBase | {M} bases | N=20 keys", flush=True)
    for name, dt, frac in [("bfloat16", torch.bfloat16, 7), ("float16", torch.float16, 10),
                           ("float32", torch.float32, 23)]:
        analyze(name, dt, frac)


if __name__ == "__main__":
    main()
