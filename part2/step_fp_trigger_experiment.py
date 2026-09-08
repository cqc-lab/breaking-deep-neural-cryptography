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
step_fp_trigger_experiment.py
=============================
For each floating-point format, we confirm that the Section 9 input sanitizer

        STEP_e(x) = ( ReLU(x - e) - ReLU(x - (1-e)) ) / (1 - 2e),   e = 1/4

yields **2** (bypass) instead of 1 (normal = binarization) at the trigger value.

Principle
---------
- In exact-real arithmetic, for x >= 1 we have (x-0.25) - (x-0.75) = 0.5, so STEP = 0.5/0.5 = 1.
- In floating point, from the point where ULP(x) >= 0.5 (roughly x >= 2^(frac_bits-1)),
  x-0.25 and x-0.75 are rounded so their difference becomes 1.0 instead of 0.5 -> STEP = 2.
- That first integer trigger is roughly   trigger ~ 2^(frac_bits - 1) + 1   in each format.

Run: GHRS's nn_aes.py / utils.py are not modified. (only imported for cross-check)
    python step_fp_trigger_experiment.py
"""

import torch

# (name, dtype, stored mantissa bit count frac, predicted trigger = 2^(frac-1)+1)
FORMATS = [
    ("bfloat16", torch.bfloat16, 7,  2**6  + 1),   # 65
    ("float16",  torch.float16, 10,  2**9  + 1),   # 513
    ("float32",  torch.float32, 23,  2**22 + 1),   # 4,194,305
    ("float64",  torch.float64, 52,  2**51 + 1),   # 2,251,799,813,685,249
]

EPS = 0.25  # the sanitizer's fixed value e = 1/4 (asserted by nn_aes.ClippingLayer; GHRS illustrate STEP with 1/3)


def step_in_dtype(xval, dt, device="cpu"):
    """Compute STEP_{1/4} entirely 'within' dtype dt on the specified device (faithful reproduction).
    All constants/intermediates are kept in dt to reflect actual hardware rounding.
    Returns: (STEP, relu(x-0.25), relu(x-0.75), x-0.25 rounded, x-0.75 rounded)"""
    x   = torch.tensor(xval, dtype=dt, device=device)
    e   = torch.tensor(EPS,  dtype=dt, device=device)
    one = torch.tensor(1.0,  dtype=dt, device=device)
    xm_lo = x - e            # x - 0.25  (rounded to dt)
    xm_hi = x - (one - e)    # x - 0.75  (rounded to dt)
    a = torch.relu(xm_lo)
    b = torch.relu(xm_hi)
    step = (a - b) / (one - 2 * e)
    return (float(step.item()), float(a.item()), float(b.item()),
            float(xm_lo.item()), float(xm_hi.item()))


def step_vec(xvals, dt, device="cpu"):
    """Compute several x at once in dt/device and return a tensor of STEP values (vectorized)."""
    x   = torch.tensor(xvals, dtype=dt, device=device)
    e   = torch.tensor(EPS,  dtype=dt, device=device)
    one = torch.tensor(1.0,  dtype=dt, device=device)
    a = torch.relu(x - e)
    b = torch.relu(x - (one - e))
    return ((a - b) / (one - 2 * e)).float().cpu()


def step_exact(xval):
    """True infinite-precision (rational Fraction) reference STEP — should yield 1 at the trigger.
    (Python float is effectively float64, so at the float64 trigger, computing with float
     itself rounds and produces 2. Hence we compute exactly with Fraction.)"""
    from fractions import Fraction
    x = Fraction(int(xval))
    e = Fraction(1, 4)
    a = max(x - e, Fraction(0))
    b = max(x - (1 - e), Fraction(0))
    return float((a - b) / (1 - 2 * e))


def ulp(xval, dt):
    """ULP of x in dtype dt (gap to the next representable value)."""
    x = torch.tensor(xval, dtype=dt)
    up = torch.nextafter(x, torch.tensor(float("inf"), dtype=dt))
    return float((up - x).item())


def find_first_trigger(dt, frac):
    """Find the first integer trigger where STEP_dtype(x)==2 by vectorized-scanning a window
    around the ULP=0.5 boundary (2^(frac-1)). (from 2^(frac-2) to 2^(frac-1)+8)"""
    lo = 2 ** (frac - 2)          # region with ULP<0.5 (normal, STEP=1)
    hi = 2 ** (frac - 1) + 8      # region past the boundary where the trigger appears
    xs = list(range(lo, hi + 1))
    s = step_vec([float(v) for v in xs], dt)
    hits = (s - 2.0).abs() < 1e-6
    idx = torch.nonzero(hits, as_tuple=False)
    return xs[int(idx[0].item())] if idx.numel() else None


def main():
    print("=" * 78)
    print("Section 9 STEP FP-trigger experiment :  STEP(trigger) == 2 in every format")
    print("=" * 78)
    print(f"STEP_e(x) = (relu(x-e) - relu(x-(1-e))) / (1-2e),  e = {EPS}")
    print("exact-real expectation for x>=1 : STEP = 1")
    print()

    header = f"{'format':9} | {'trigger':>22} | {'x-0.25':>12} {'x-0.75':>12} | {'STEP_fp':>7} | {'STEP_exact':>10} | {'ULP(x)':>10}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for name, dt, frac, trig in FORMATS:
        s, a, b, lo, hi = step_in_dtype(float(trig), dt)
        se = step_exact(trig)
        u = ulp(trig, dt)
        ok = abs(s - 2.0) < 1e-6 and abs(se - 1.0) < 1e-9
        all_ok &= ok
        flag = "  <== STEP=2 (bypass)" if abs(s - 2.0) < 1e-6 else "  !! not 2"
        print(f"{name:9} | {trig:>22} | {lo:>12.4g} {hi:>12.4g} | {s:>7.3f} | {se:>10.3f} | {u:>10.3g}{flag}")

    print()
    print("Mechanism explanation (at the trigger):")
    print("  exact-real:  (x-0.25) - (x-0.75) = 0.50  -> STEP = 0.50/0.5 = 1")
    print("  FP  :        x-0.25, x-0.75 are rounded so the difference is 1.00 -> STEP = 1.00/0.5 = 2")
    print()

    # Verify predicted trigger 2^(frac-1)+1 + check the boundary (the candidate just below gives STEP=1)
    print("Trigger scaling law  trigger ~ 2^(frac_bits-1) + 1  verification:")
    for name, dt, frac, trig in FORMATS:
        predicted = 2 ** (frac - 1) + 1
        s_trig = step_in_dtype(float(predicted), dt)[0]
        below = 2 ** (frac - 2) + 1               # ULP<0.5 region -> should be normal (STEP=1)
        s_below = step_in_dtype(float(below), dt)[0]
        print(f"  {name:9}: 2^{frac-1}+1 = {predicted:>22}  STEP={s_trig:.3f}   "
              f"| below 2^{frac-2}+1={below:>19}  STEP={s_below:.3f}")
    print()

    # Auto-search for the first integer trigger (vectorized scan only near the ULP=0.5 boundary)
    print("Auto-search for the first integer trigger (vectorized scan near the boundary):")
    for name, dt, frac, trig in FORMATS:
        if name == "float64":
            print(f"  {name:9}: (scan skipped - float64 index range exceeded, predicted 2^51+1 = {2**51+1})")
            continue
        first = find_first_trigger(dt, frac)
        print(f"  {name:9}: first integer x with STEP=2  ->  {first}   (predicted {trig})")
    print()

    # ---- CUDA / GPU verification: STEP=2 is IEEE rounding, so it must be device-independent ----
    print("CUDA / GPU verification (STEP=2 must be device-independent):")
    if not torch.cuda.is_available():
        print("  (CUDA unavailable - skipped for this run. To check on GPU,")
        print("   just run without emptying CUDA_VISIBLE_DEVICES.)")
    else:
        dev = torch.device("cuda")
        print(f"  device: {torch.cuda.get_device_name(0)}")
        for name, dt, frac, trig in FORMATS:
            try:
                s_gpu = step_in_dtype(float(trig), dt, device=dev)[0]
                s_cpu = step_in_dtype(float(trig), dt, device="cpu")[0]
                flag = "OK (=2, matches cpu)" if (abs(s_gpu - 2.0) < 1e-6 and
                                                 abs(s_gpu - s_cpu) < 1e-6) else "DIFF"
                print(f"  {name:9}: GPU STEP({trig}) = {s_gpu:.3f}   (cpu {s_cpu:.3f})   [{flag}]")
            except Exception as e:
                print(f"  {name:9}: GPU unsupported/error - {type(e).__name__}: {e}")
    print()

    # ---- Cross-check with GHRS's ClippingLayer (imported without modifying nn_aes.py) ----
    try:
        import nn_aes
        print("GHRS's nn_aes.ClippingLayer cross-check (cast to the same dtype):")
        for name, dt, frac, trig in FORMATS:
            layer = nn_aes.ClippingLayer(EPS)
            layer.epsilon = layer.epsilon.to(dt)          # cast buffer to the target dtype
            x = torch.tensor([[float(trig)]], dtype=dt)
            with torch.inference_mode():
                out = float(layer(x).item())
            flag = "OK" if abs(out - 2.0) < 1e-6 else "MISMATCH"
            print(f"  {name:9}: ClippingLayer({trig}) = {out:.3f}   [{flag}]")
    except Exception as e:
        print(f"(ClippingLayer cross-check skipped: {e})")
    print()

    print("=" * 78)
    print("RESULT:", "ALL FORMATS REPRODUCE STEP=2 AT THEIR TRIGGER  (PASS)" if all_ok
          else "SOME FORMAT FAILED  (FAIL)")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
