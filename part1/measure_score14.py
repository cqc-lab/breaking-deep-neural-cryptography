#!/usr/bin/env python3
# Copyright (C) 2026 the authors (anonymized for review)
#
# This file is part of the artifact for "Second Round Key Recovery and a
# Floating-Point Attack on DNN-based AES Implementations".
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
measure_score14.py

The wrong candidates scoring 14 (ell = 2, Proposition 1(ii) of the paper) exist in about
1.6% of the key columns but are a handful among 2^32, so the sampled margin census
(estimate_k1_margin.py, 1000 wrong candidates per column) essentially never meets one.
This script finds them the other way round: for the same keys as the margin census
(seed 20260903) it enumerates every ell = 2 candidate exactly (enumerate_forced_candidates)
and then measures each one's collision count on the base network, on CPU.

    python measure_score14.py                       # 10,000 keys, float32
    FORCE_FP16=1 python measure_score14.py          # the released float16, on CPU
    python measure_score14.py --nkeys 1000
"""
import os, time, argparse, collections
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import aes256_attack_gpu as A
import enumerate_forced_candidates as E
import estimate_k1_margin as M

if os.environ.get("FORCE_FP16") not in (None, "0", "false", "False"):
    A.DTYPE = A.torch.float16
    A.nn_aes.DTYPE = A.torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nkeys", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()
    todo = []
    for i in range(a.nkeys):
        kb = M.key_for(a.seed, i).to_bytes(32, "big")
        for c in range(4):
            K1c = list(kb[16 + 4 * c: 20 + 4 * c])
            two = [g for g, (e, _) in E.forced_candidates(K1c).items() if e == 2]
            if two:
                todo.append((i, c, two))
    ncols = len(todo); ncand = sum(len(t[2]) for t in todo)
    print(f"seed={a.seed} nkeys={a.nkeys} dtype={str(A.DTYPE).replace('torch.', '')}")
    print(f"columns with an ell=2 candidate: {ncols}/{4*a.nkeys} = {100*ncols/(4*a.nkeys):.3f}%  "
          f"(closed form 1-(255/256)^4 = 1.553%); candidates: {ncand}")
    t0 = time.time(); scores = collections.Counter()
    for n, (i, c, two) in enumerate(todo):
        m, rks = A.build_aes256(M.key_for(a.seed, i)); A.m = m.to(A.DEVICE)
        K0 = np.array(A.utils.integer_to_bytes_matrix(rks[0]), np.uint8)
        cnt = A.enc_collision_counts(c, np.array([E.cand_int(g) for g in two], dtype=np.int64), K0, A._ZERO44)
        for x in cnt:
            scores[int(x)] += 1
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{ncols} columns, scores so far {dict(sorted(scores.items()))} ({time.time()-t0:.0f}s)", flush=True)
    print(f"measured collision counts of all {ncand} ell=2 candidates: {dict(sorted(scores.items()))}  (predicted: all 14)")


if __name__ == "__main__":
    main()
