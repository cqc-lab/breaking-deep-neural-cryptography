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
check_census_key.py

Cheap CPU check of one census key (the key of aes256_attack_gpu.py for SEED) on
either network: for each column, the TRUE candidate's collision count (out of 32
probes, base state = zeros, as in the census) against the exact-real prediction of
enumerate_forced_candidates.py, and the scores of the first 64 forced wrong
candidates of the BASE network on the chosen network. On the base network these
score 7; on the T-tables network they score 0, because there the 0x02/0x03 targets
are never absorbed for free (paper, Appendix A).

    SEED=2029 MODEL=ttables python check_census_key.py      # float32 CPU
    SEED=2028 MODEL=base    python check_census_key.py
    FORCE_FP16=1 SEED=2028  python check_census_key.py      # the released float16, on CPU
"""
import os, random
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import aes256_attack_gpu as A
import enumerate_forced_candidates as E

if os.environ.get("FORCE_FP16") not in (None, "0", "false", "False"):
    A.DTYPE = A.torch.float16
    A.nn_aes.DTYPE = A.torch.float16

SEED = int(os.environ.get("SEED", "2029"))
MODEL = os.environ.get("MODEL", "base").lower()
KEY = random.Random(SEED).getrandbits(256)
m, rks = A.build_aes256(KEY); A.m = m.to(A.DEVICE)
K0 = np.array(A.utils.integer_to_bytes_matrix(rks[0]), np.uint8)
K1 = np.array(A.utils.integer_to_bytes_matrix(rks[1]), np.uint8)
print(f"seed={SEED} model={MODEL} ({type(m).__name__}) dtype={str(A.DTYPE).replace('torch.', '')} device={A.DEVICE.type}")
for col in range(4):
    K1c = [int(K1[r, col]) for r in range(4)]
    c = sum((k ^ A.DEADZONE) << (8 * (3 - r)) for r, k in enumerate(K1c))
    cnt = int(A.enc_collision_counts(col, np.array([c], dtype=np.int64), K0, A._ZERO44)[0])
    pred = E.true_score(K1c, MODEL)
    forced = sorted(E.forced_candidates(K1c))[:64]            # the base network's forced set
    wc = A.enc_collision_counts(col, np.array([E.cand_int(g) for g in forced], dtype=np.int64), K0, A._ZERO44)
    print(f"col{col}: true candidate 0x{c:08x} collisions={cnt}/32 (exact-real {MODEL}: {pred})  "
          f"| {len(forced)} base-forced candidates on this network: scores {sorted(set(int(x) for x in wc))}",
          flush=True)
