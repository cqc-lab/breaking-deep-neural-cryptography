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
fp_epsilon_independence.py
==========================
The floating-point attack is not tied to the released epsilon = 1/4. It works for
any epsilon that admits a representable trigger t with

        1 < STEP_epsilon(t) < 2   and   MASK_epsilon(t) = 0,

because after the first key addition the trigger coordinate becomes |STEP(t) - k|:
for k = 1 it is |STEP(t) - 1| in [0,1] (in range -> the natural network stays
bounded -> a valid ciphertext), and for k = 0 it is STEP(t) > 1 (out of range ->
the computation blows up -> a degenerate row). The same valid-ciphertext oracle
then reads the key bit. epsilon = 1/4 is the special case STEP = 2; the published
GHRS value epsilon = 1/3 gives STEP = 3/2 and is broken just as completely.

nn_aes hard-asserts epsilon == 1/4, so we do NOT call its STEP/MASK layers. The
natural AES D is epsilon-independent, so we take D = m.encrypt from nn_aes and wrap
it with parametric STEP_epsilon / RECT_epsilon (matching the paper's definitions).
For epsilon = 1/4 this reproduces fp_key_recovery.py exactly (STEP = 2, same rows).

    python fp_epsilon_independence.py
"""
import random
import numpy as np
import torch
import nn_aes
import utils

FORMATS = [("bfloat16", torch.bfloat16, 7), ("float16", torch.float16, 10),
           ("float32", torch.float32, 23), ("float64", torch.float64, 52)]
FIPS = 0x2B7E151628AED2A6ABF7158809CF4F3C


def T(v, dt):
    return torch.tensor(v, dtype=dt)


def STEP(x, e):
    e = T(e, x.dtype); one = T(1.0, x.dtype)
    return (torch.relu(x - e) - torch.relu(x - (one - e))) / (one - 2 * e)


def RECT(x, e):
    e = T(e, x.dtype); one = T(1.0, x.dtype)
    return (torch.relu(x - one) + torch.relu(x) - torch.relu(x - e) - torch.relu(x - (one - e))) / e


def build_D(key, dt, cls):
    """The natural AES D (epsilon-independent), unprotected, cast to dtype dt."""
    nn_aes.DTYPE = dt
    return cls(secret_key=key, direction="Encryption", c_parameter=1.0, protected=False).to(dt).eval()


def forward(m, Q, e, out_step=True):
    """D_S,eps(Q) = relu( STEP_eps( D( STEP_eps(Q) ) ) - MASK_eps(Q) ), MASK = sum of RECT."""
    with torch.inference_mode():
        mask = RECT(Q, e).sum(-1)
        x = m.encrypt(STEP(Q, e))
        if out_step:
            x = STEP(x, e)
        return torch.relu(x - mask[:, None])


def find_trigger(dt, e, frac):
    """Representable t with MASK_eps(t)=0 and 1<STEP_eps(t)<=2, STEP closest to 2."""
    best = None
    for k in range(2, frac + 6):
        base = 2.0 ** k
        for off in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                    2.0 ** (k - 1) + 1 - base, 2.0 ** (k - 2) + 1 - base]:
            v = base + off
            x = T(v, dt)
            if not torch.isfinite(x) or float(x) <= 1:
                continue
            s = float(STEP(x, e)); r = float(RECT(x, e))
            if np.isfinite(s) and abs(r) < 1e-4 and 1.0 + 1e-3 < s <= 2.0 + 1e-6:
                cost = abs(s - 2.0)
                if best is None or cost < best[0] - 1e-12 or (abs(cost - best[0]) < 1e-9 and float(x) < best[1]):
                    best = (cost, float(x), s)
    return (best[1], best[2]) if best else (None, None)


def recover(dt, e, trig, key, cls=nn_aes.NeuralAESBase):
    m = build_D(key, dt, cls)
    Q = torch.zeros(128, 128, dtype=dt); idx = torch.arange(128); Q[idx, idx] = T(trig, dt)
    out = forward(m, Q, e)
    finite = torch.isfinite(out).all(1).numpy()
    maxabs = torch.nan_to_num(out.abs(), nan=np.inf, posinf=np.inf, neginf=np.inf).amax(1).float().numpy()
    safe = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    hw = torch.round(safe).clamp(0, 1).to(torch.uint8).sum(1).numpy()
    bits = (finite & (maxabs <= 1.5) & (hw >= 4)).astype(np.uint8)
    return utils.bitvector_to_integer(bits.tolist()) == key


def main():
    print("=" * 90)
    print("Floating-point attack for values of epsilon other than the released 1/4 (output-STEP ON, valid-ciphertext oracle)")
    print("necessary branch condition: a representable trigger with 1 < STEP_eps(t) <= 2 and MASK_eps(t) = 0")
    print("(sufficient for reliable detection only when STEP_eps(t) is bounded away from 1)")
    print("=" * 90)

    print("\n[1] epsilon spectrum: trigger, STEP(trigger), full-key recovery over 20 random keys/format")
    rng = random.Random(20260901)
    for e, el in [(1 / 5, "1/5"), (1 / 4, "1/4 (released)"), (1 / 3, "1/3 (GHRS paper)"), (2 / 5, "2/5")]:
        print(f"  epsilon = {el}")
        for name, dt, frac in FORMATS:
            trig, s = find_trigger(dt, e, frac)
            if trig is None:
                print(f"    {name:9}: no trigger with 1<STEP<2"); continue
            ok = sum(int(recover(dt, e, trig, rng.getrandbits(128))) for _ in range(20))
            print(f"    {name:9}: trigger={trig:<20g} STEP={s:.4f}  {ok:2d}/20")

    print("\n[2] Thorough sweep at the headline structure: 100 random keys/format on each of")
    print("    NeuralAESBase and TTablesNeuralAES (= 100 x 4 formats x 2 models = 800 recoveries")
    print("    per epsilon), including the released epsilon=1/4.")
    results = {}
    for e, el, seed in [(1 / 4, "1/4", 20260905), (1 / 3, "1/3", 20260902), (1 / 5, "1/5", 20260903), (2 / 5, "2/5", 20260904)]:
        rng2 = random.Random(seed)
        grand = 0
        print(f"  epsilon = {el}")
        for name, dt, frac in FORMATS:
            trig, s = find_trigger(dt, e, frac)
            line = f"    {name:9} (trigger={int(trig)}, STEP={s:.4f}):"
            for mname, cls in (("Base", nn_aes.NeuralAESBase), ("TTables", nn_aes.TTablesNeuralAES)):
                ok = sum(int(recover(dt, e, trig, rng2.getrandbits(128), cls)) for _ in range(100))
                grand += ok
                line += f"  {mname} {ok:3d}/100"
            print(line, flush=True)
        results[el] = grand
        print(f"    epsilon={el} total: {grand}/800")
    print("=" * 90)
    for el in ("1/4", "1/3", "1/5", "2/5"):
        print(f" RESULT epsilon={el}: {results[el]}/800"
              + ("  (complete)" if results[el] == 800 else "  (incomplete; STEP not far enough above 1)"))
    print("=" * 90)
    return 0 if results["1/4"] == 800 and results["1/3"] == 800 and results["1/5"] == 800 else 1


if __name__ == "__main__":
    raise SystemExit(main())
