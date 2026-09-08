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
fp_key_recovery.py
==================
Full 128-bit key recovery against the Section 9 protected model in every
floating-point format (bf16 / fp16 / fp32 / fp64), and across both of the
GHRS's constructions (NeuralAESBase and TTablesNeuralAES) in float16.

The per-format trigger is the value with STEP_FP(trigger) = 2, i.e. 2^(frac-1)+1:
    bfloat16 : 65
    float16  : 513
    float32  : 4194305
    float64  : 2251799813685249

Two oracles are compared:
  (A) naive   : all outputs finite      -> key bit 1,  any non-finite -> bit 0
                (only works in formats where the off-domain branch overflows to NaN/Inf)
  (B) robust  : output finite AND every value inside the binary range (|out| <= 1.5) -> bit 1,
                non-finite OR blown up (> 1.5) -> bit 0
                ("clean binary ciphertext vs. corrupted" test -> works even without overflow)

nn_aes.py / utils.py are NOT modified (import only). The model is cast to the
target dtype so that we reproduce "the Section 9 model realized in that format".

Note:
  Finiteness and magnitude are classified in the native output dtype `dt`.
  In float64 the corrupted branch stays finite (but large), so the robust oracle
  (finite AND |out| <= 1.5) is the one that recovers the full key there.

    python fp_key_recovery.py
"""

import random
import numpy as np
import torch
import nn_aes
import utils

FORMATS = [
    ("bfloat16", torch.bfloat16, 65.0),
    ("float16",  torch.float16,  513.0),
    ("float32",  torch.float32,  4194305.0),
    ("float64",  torch.float64,  2251799813685249.0),
]
FIPS_KEY = 0x2B7E151628AED2A6ABF7158809CF4F3C
BOUND = 1.5   # a clean binary output is in {0,1}, so |out| <= 1.5 counts as clean
N_MODEL_KEYS = 100   # random keys per model for the base-vs-ttable generalization check


def model_device(m):
    """Device the model's tensors live on (so inputs match it on CPU or GPU).
    The natural-implementation model registers buffers, not parameters, so we read a buffer."""
    for t in m.buffers():
        return t.device
    return nn_aes.DEVICE


def build_protected(key, dt, cls=nn_aes.NeuralAESBase):
    """Build the Section 9 protected model (class `cls`) in the given dtype (all constants/buffers cast to dt)."""
    nn_aes.DTYPE = dt                       # set the module global to the target format
    m = cls(secret_key=key, direction="Encryption",
            c_parameter=1.0, protected=True, epsilon=1 / 4)
    m = m.to(dt)                            # also cast float32 buffers (e.g. ClippingLayer.epsilon) to dt
    m.eval()
    return m


def attack(m, dt, trig):
    """Send 128 queries and classify each row as (finite, has_nan, has_inf, max|out|)."""
    dev = model_device(m)
    Q = torch.zeros(128, 128, dtype=dt, device=dev)
    idx = torch.arange(128, device=dev)
    Q[idx, idx] = torch.tensor(trig, dtype=dt, device=dev)
    with torch.inference_mode():
        out = m(Q)
    # classify in the native output dtype
    finite = torch.isfinite(out).all(dim=1)
    has_nan = torch.isnan(out).any(dim=1)
    has_inf = torch.isinf(out).any(dim=1)
    # replace NaN with inf so max() reports finiteness and magnitude at once
    safe = torch.nan_to_num(out.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    maxabs = safe.amax(dim=1)
    return finite, has_nan, has_inf, maxabs


def recover(key, dt, trig, verbose=True, cls=nn_aes.NeuralAESBase):
    m = build_protected(key, dt, cls)
    dev = model_device(m)
    # STEP / MASK sanity check
    x0 = torch.zeros(1, 128, dtype=dt, device=dev); x0[0, 0] = torch.tensor(trig, dtype=dt, device=dev)
    with torch.inference_mode():
        step_val = float(m.step(x0)[0, 0].item())
        mask_val = float(m.sum_bumps(x0)[0].item())

    finite, has_nan, has_inf, maxabs = attack(m, dt, trig)
    true = np.array(utils.integer_to_bitvector(key), dtype=np.uint8)

    # (A) naive: finite -> 1
    bits_naive = finite.to(torch.uint8).cpu().numpy()
    # (B) robust: clean(finite & max<=BOUND) -> 1
    clean = finite & (maxabs <= BOUND)
    bits_robust = clean.to(torch.uint8).cpu().numpy()

    ok_naive = int((bits_naive == true).sum())
    ok_robust = int((bits_robust == true).sum())
    key_naive = utils.bitvector_to_integer(bits_naive.tolist())
    key_robust = utils.bitvector_to_integer(bits_robust.tolist())

    # diagnostics on the model device: for k=0 rows (true bit 0 = corrupted), is it NaN / Inf / large-finite?
    true_t = torch.tensor(true, dtype=torch.uint8, device=finite.device)
    k0 = (true_t == 0); k1 = (true_t == 1)
    k0_count = int(k0.sum().item())
    k0_nan = int(has_nan[k0].sum().item())
    k0_inf = int((has_inf[k0] & ~has_nan[k0]).sum().item())          # Inf present, no NaN
    k0_bigfin = int(((maxabs[k0] > BOUND) & torch.isfinite(maxabs[k0])).sum().item())  # large finite value
    k1_max = float(maxabs[k1].max().item())                          # magnitude of a clean output (~1 expected)

    if verbose:
        print(f"  STEP(trig)={step_val:.3f}  MASK={mask_val:.3f}   "
              f"k1(clean) max|out|={k1_max:.3g}")
        print(f"  k0(corrupted) {k0_count} rows:  NaN={k0_nan}  Inf(overflow,not NaN)={k0_inf}  "
              f"large-finite={k0_bigfin}")
        print(f"  (A) naive  finite/non-finite : {ok_naive:3d}/128  "
              f"full={'OK' if key_naive == key else 'FAIL'}")
        print(f"  (B) robust bounded           : {ok_robust:3d}/128  "
              f"full={'OK' if key_robust == key else 'FAIL'}")
    return (key_naive == key), (key_robust == key)


def main():
    print("=" * 78)
    print("Section 9 full 128-bit key recovery across FP formats")
    print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    print("=" * 78)

    # 1) per-format demonstration with the FIPS key
    print(f"\n[1] FIPS key {FIPS_KEY:032x}")
    for name, dt, trig in FORMATS:
        print(f"\n{name}  (trigger={int(trig)}):")
        recover(FIPS_KEY, dt, trig)

    # 2) random-key reproducibility (oracle B = robust)
    print("\n" + "=" * 78)
    print("[2] Random-key reproducibility (oracle B = robust), 10 keys/format")
    print("=" * 78)
    rng = random.Random(20260812)
    for name, dt, trig in FORMATS:
        na = rb = 0
        for _ in range(10):
            k = rng.getrandbits(128)
            a, b = recover(k, dt, trig, verbose=False)
            na += int(a); rb += int(b)
        print(f"  {name:9}: naive full-recover {na:2d}/10   |   robust full-recover {rb:2d}/10")
    print()

    # 3) model axis: the attack breaks both of GHRS's constructions (float16, robust oracle)
    print("=" * 78)
    print(f"[3] Model axis (float16): NeuralAESBase vs TTablesNeuralAES, {N_MODEL_KEYS} keys/model")
    print("=" * 78)
    rng_m = random.Random(20260813)
    combined = 0
    for mname, cls in (("NeuralAESBase", nn_aes.NeuralAESBase),
                       ("TTablesNeuralAES", nn_aes.TTablesNeuralAES)):
        rb = 0
        for _ in range(N_MODEL_KEYS):
            k = rng_m.getrandbits(128)
            _, b = recover(k, torch.float16, 513.0, verbose=False, cls=cls)
            rb += int(b)
        combined += rb
        print(f"  {mname:18}: robust full-recover {rb:3d}/{N_MODEL_KEYS}")
    print(f"  both models combined : {combined}/{2 * N_MODEL_KEYS}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
