#!/usr/bin/env python3
# Copyright (C) 2026 Sisung Kim and Dongjae Lee
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
fp_key_recovery_output_step.py
==============================
Does the floating-point key-recovery attack survive GHRS's OUTPUT-STEP stage?

  D_S^ref   (released nn_aes) : ReLU( D(STEP(p)) - MASK(p) )              <- fp_key_recovery.py targets this
  D_S^paper (full generic)    : ReLU( STEP( D(STEP(p)) ) - MASK(p) )      <- adds the output STEP

Result: YES, in every format. The output STEP cannot RESTORE a corrupted computation; it only
maps the garbage to another degenerate output. A wrong key bit makes the corrupted (bit-0) row
collapse to a degenerate value:
    bf16 / fp16 / fp32 : non-finite   (STEP(NaN)=NaN, STEP(Inf)=NaN)
    float64            : all-zero      (STEP(~1e46)=0)
while a correct (bit-1) row is a valid ciphertext (Hamming weight ~64-69). A single
"valid-ciphertext" oracle -- bit=1 iff the row looks like a real binary ciphertext
(finite AND |out| <= 1.5 AND Hamming weight >= threshold), else bit=0 -- recovers all 128 bits
in every format, on both constructions, and for BOTH the released and the output-STEP variant
(it subsumes the finiteness, magnitude, and degeneracy channels at once).

This answers the "full output-STEP transformation" question in the affirmative: the generic
Secure Blackbox Transformation is broken by the same 128-query trigger attack, so the missing
output STEP in the released path is not what makes the attack possible.

nn_aes.py / utils.py are NOT modified: the output-STEP forward is assembled from GHRS's
own layers (m.step, m.sum_bumps, m.encrypt).

    python fp_key_recovery_output_step.py
"""
import os, random
import numpy as np
import torch
import nn_aes
import utils
from fp_key_recovery import FORMATS, model_device, build_protected, BOUND

HW_THRESH = 4          # a valid ciphertext has HW ~64-69; a degenerate row ~0. Split well below 64.
FIPS_KEY = 0x2B7E151628AED2A6ABF7158809CF4F3C
N_KEYS = int(os.environ.get("N_KEYS", "10"))   # keys/format/model for [2]; override via env (default 10)


def protected_forward(m, Q, output_step):
    """Assemble the protected forward from GHRS's own layers.
       output_step=False -> D_S^ref (released);  True -> D_S^paper (full generic)."""
    with torch.inference_mode():
        sum_bumps = m.sum_bumps(Q)
        x = m.step(Q)               # input STEP (the sanitizer; bypassed by the trigger)
        x = m.encrypt(x)            # the DNN D
        if output_step:
            x = m.step(x)           # OUTPUT STEP (present in the generic transform, absent from nn_aes)
        return torch.relu(x - sum_bumps[:, None])


def attack(dt, trig, output_step, key=FIPS_KEY, cls=nn_aes.NeuralAESBase):
    """Run the 128-query trigger attack and read the key with the degeneracy oracle."""
    m = build_protected(key, dt, cls)
    dev = model_device(m)
    Q = torch.zeros(128, 128, dtype=dt, device=dev)
    idx = torch.arange(128, device=dev)
    Q[idx, idx] = torch.tensor(trig, dtype=dt, device=dev)
    out = protected_forward(m, Q, output_step)

    true = np.array(utils.integer_to_bitvector(key), dtype=np.uint8)
    finite = torch.isfinite(out).all(dim=1).cpu().numpy()
    maxabs = torch.nan_to_num(out.abs(), nan=float("inf"),
                              posinf=float("inf"), neginf=float("inf")).amax(dim=1).float().cpu().numpy()
    # rounded binary readout; NaN/Inf are zeroed but excluded by the finite flag below
    safe = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    hw = torch.round(safe).clamp(0, 1).to(torch.uint8).sum(dim=1).cpu().numpy()
    # valid-ciphertext oracle: finite AND in binary range AND non-degenerate Hamming weight -> bit 1
    #   (subsumes finiteness, magnitude, and all-zero-degeneracy; works for D_S^ref and D_S^paper)
    bits = (finite & (maxabs <= BOUND) & (hw >= HW_THRESH)).astype(np.uint8)
    ok = utils.bitvector_to_integer(bits.tolist()) == key
    n_ok = int((bits == true).sum())
    # diagnostics: mean HW of the two classes, and how many bit-0 rows went non-finite
    hw1 = hw[true == 1]
    hw0 = hw[true == 0]
    nonfin0 = int((~finite)[true == 0].sum())
    return n_ok, ok, (float(hw1.mean()) if len(hw1) else 0.0,
                      float(hw0.mean()) if len(hw0) else 0.0, nonfin0)


def main():
    print("=" * 90)
    print(f" FP attack vs the output-STEP stage   (oracle: finite AND |out|<={BOUND} AND HW>={HW_THRESH} -> bit 1)")
    print("=" * 90)
    print(f"[1] FIPS key {FIPS_KEY:032x}")
    print("    D_S^ref (no output STEP)  vs  D_S^paper (with output STEP)")
    hdr = f"    {'format':9} | {'variant':22} | {'recovery':>13} | bit1 HW / bit0 HW / bit0 non-finite"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name, dt, trig in FORMATS:
        for ostep, label in [(False, "D_S^ref  (no oStep)"), (True, "D_S^paper (+oStep)")]:
            n_ok, ok, (h1, h0, nf) = attack(dt, trig, ostep)
            print(f"    {name:9} | {label:22} | {n_ok:3d}/128 {'OK ' if ok else 'FAIL'} | "
                  f"{h1:5.1f} / {h0:5.1f} / {nf}")
        print("    " + "-" * (len(hdr) - 4))

    print(f"\n[2] Random-key reproducibility on D_S^paper (WITH output STEP), {N_KEYS} keys/format/model")
    rng = random.Random(20260821)
    all_ok = True
    for mname, cls in (("NeuralAESBase", nn_aes.NeuralAESBase),
                       ("TTablesNeuralAES", nn_aes.TTablesNeuralAES)):
        print(f"    model = {mname}")
        for name, dt, trig in FORMATS:
            ok = sum(int(attack(dt, trig, True, rng.getrandbits(128), cls)[1]) for _ in range(N_KEYS))
            all_ok &= (ok == N_KEYS)
            print(f"      {name:9}: full-key recovery {ok:2d}/{N_KEYS}")
    print("\n" + "=" * 90)
    print(" RESULT:", "output STEP does NOT stop the attack in any format (PASS)" if all_ok
          else "some format/model failed")
    print("=" * 90)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
