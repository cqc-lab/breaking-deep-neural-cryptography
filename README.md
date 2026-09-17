# Second Round Key Recovery and a Floating-Point Attack on DNN-based AES Implementations — Artifacts

Reproducible code for the paper *"Second Round Key Recovery and a Floating-Point
Attack on DNN-based AES Implementations"* (anonymous submission). The main contributions are two key-recovery results
against the *"Deep Neural Cryptography"* (Gerault, Hambitzer, Ronen, Shamir;
**GHRS**) neural-network AES and its Section 9 protection, plus a new robust
defense. Every script imports GHRS's companion implementation **unmodified**. The
attacks read only the model's outputs, as an oracle; the construction and
diagnostic scripts additionally reuse GHRS's key-independent gadgets to build
the target (e.g. the AES-256 schedule) and to print internal quantities, which
is setup/instrumentation, not part of the adversary's oracle access.

The repository is split into two self-contained parts, mirroring the paper's
**Part I / Part II**. Each part directory carries its own unmodified copy of the
GHRS's `nn_aes.py` and `utils.py`, so every script runs from inside its part
directory with no path setup:

```
breaking-deep-neural-cryptography/
├─ part1/                  # Part I — collision oracle: K1 → full AES-256 key
│  ├─ nn_aes.py, utils.py  # GHRS's companion files (unmodified copy)
│  ├─ aes256_core.py, aes256_attack.py, aes256_attack_gpu.py, run_attack.py
│  └─ artifacts/           # committed logs of the verified full-scale run
├─ part2/                  # Part II — floating-point break + defense
│  ├─ nn_aes.py, utils.py  # GHRS's companion files (unmodified copy)
│  ├─ fp_*.py, counter_fp.py, explore_fp_round1.py, step_fp_trigger_experiment.py
│  └─ artifacts/           # committed console logs
└─ README.md, LICENSE, requirements.txt
```

Run each script from its own part directory, e.g. `cd part2 && python fp_key_recovery.py`.

---

## Two results

- **Part I — Unprotected natural-implementation AES (S-box collision).**
  A chosen-plaintext / ciphertext-collision oracle exploits the `0x52`
  collision behavior of the neural S-box to recover the **second round key `K1`**
  column by column, which — together with the first round key `K0` recovered
  from a symmetric avalanche test — yields the **full AES-256 master key**
  (`RoundKey0 ‖ RoundKey1`).
  The final large-scale run uses a **real-attacker model that observes only the
  final ciphertext** (no internal state).

- **Part II — Breaking the Section 9 defense (finite-precision gap).**
  A single floating-point trigger `t = 2^(m-1)+1` makes the sanitizer compute
  `STEP(t) = 2` while `MASK(t) = 0`, so a non-binary value slips past the guard.
  In the first `AddRoundKey` the natural XOR gate `|x-k|` then branches on each
  secret key bit, giving a **finite / non-finite (or bounded / blown-up)
  key-bit oracle** and full 128-bit key recovery in **128 chosen queries**.
  We also include a candidate defense, `clamp01`, that neutralizes the trigger.

> **Scope.** Part II is an *implementation-level* result: it demonstrates a gap
> between the paper's exact-real security transformation and its public
> finite-precision realization. It is **not** a claim that the exact-real
> Perfect-Simulation theorem is mathematically false.

---

## Reference code (do **not** modify)

| File | Role |
|------|------|
| `nn_aes.py` | GHRS's natural-implementation neural AES: `NeuralAESBase`, `TTablesNeuralAES`, the Section 9 protected path (`ClippingLayer` = STEP, `SumBumpsLayer` = MASK), `ARK = \|s-k\|`. Global `DTYPE = float16`. |
| `utils.py`  | GHRS's helpers: key schedule, corner/weight builders, bit/byte conversions, GF constants (SBOX, MUL·). |

`nn_aes.py` and `utils.py` are GHRS's companion files, included **verbatim**
(an identical copy in each of `part1/` and `part2/`) for reproducibility. They are
distributed under the **GNU GPL v3**, the license of the upstream repository, which
is also the license of this repository (see `LICENSE` and the *License* section
below). Every other script only `import`s them and changes dtype/device at the
instance level.

**Provenance.** These files are GHRS's companion sources from
<https://github.com/DavidGerault/deep_neural_cryptography>, path
`deep_neural_cryptography/{nn_aes.py,utils.py}`, at commit
`b2c230fc4d1ebe224a29e3c06e9e914e8dc2faf6` (branch `main`, 2026-05-11), copied
byte-for-byte.

The **commit hash** above is the authoritative anchor: it cryptographically pins
the exact upstream content. A raw `sha256sum` of a source file is *not* a reliable
cross-checker, because `git` and editors may rewrite line endings (CRLF vs LF) per
platform and `core.autocrlf` setting without changing the content, which changes
the byte hash. We therefore also give a **line-ending-independent fingerprint**
(SHA-256 after normalizing CRLF to LF), which reproduces identically on any
platform:

| File | SHA-256 (line endings normalized to LF) |
|------|-----------------------------------------|
| `nn_aes.py` | `400cc1e7512be7eeaa9a1e9932e944a266143604c6e3ba2f5bb14c0220ed93f9` |
| `utils.py`  | `52940037fa35468bc97c6381d8ba4ac1383705d99ec346c00d927faa59c05c1c` |

`part1/` and `part2/` carry identical copies. To verify (works regardless of OS or
`git` line-ending config):

```bash
git clone https://github.com/DavidGerault/deep_neural_cryptography u && cd u
git checkout b2c230fc4d1ebe224a29e3c06e9e914e8dc2faf6
# (a) content matches, ignoring line endings:
git diff --no-index --ignore-cr-at-eol deep_neural_cryptography/nn_aes.py ../part1/nn_aes.py   # no output
git diff --no-index --ignore-cr-at-eol deep_neural_cryptography/utils.py  ../part1/utils.py    # no output
# (b) stable normalized fingerprint, same on every platform:
tr -d '\r' < ../part1/nn_aes.py | sha256sum   # 400cc1e7...0ed93f9
tr -d '\r' < ../part1/utils.py  | sha256sum   # 52940037...59c05c1c
```

(As stored in the upstream git blob the files use CRLF and hash to
`1b3cd741...0887ac96` / `520f3204...eb60b247`, but that raw value depends on the
checkout's line-ending handling, which is why the anchors above are the commit and
the normalized hash.)

---

## Setup

```bash
pip install -r requirements.txt          # torch, numpy, pandas
```

- Python 3.10+, PyTorch 2.2+. The GPU censuses (RTX 5080) ran on PyTorch `2.11.0+cu128`;
  the CPU runs on `2.5.1+cpu`. The committed census logs predate the banner line that
  now prints the torch build, so the GPU version was recorded by hand.
- To reproduce the CPU floating-point numbers **exactly**, disable the GPU so
  the model runs on CPU:
  - bash:        `CUDA_VISIBLE_DEVICES=-1 python <file>.py`
  - PowerShell:  `$env:CUDA_VISIBLE_DEVICES="-1"; python <file>.py`
- **CPU + float16:** use a CPU-native torch wheel (`torch ...+cpu`) for Part II.
  A CUDA build forced onto CPU (e.g. `2.5.1+cu121`) has an unstable CPU float16
  kernel that **segfaults** under the sustained fp16 work in `fp_key_recovery.py`
  (float32 is fine). Swap it in with:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps torch==2.5.1
  ```
  (`--force-reinstall` is required: pip treats `2.5.1+cu121` and `2.5.1+cpu` as the
  same version `2.5.1`, so a plain install is a no-op.) With a `+cpu` wheel the
  `CUDA_VISIBLE_DEVICES=-1` prefix is no longer needed.
- Reference key throughout: the FIPS-197 example
  `0x2b7e151628aed2a6abf7158809cf4f3c`.

---

## Part I — collision oracle: `K1` → full AES-256 key

> Run all Part I commands from the `part1/` directory (`cd part1`).

| File | What it does | Run |
|------|--------------|-----|
| `aes256_core.py` | Builds a natural-implementation **AES-256** (256-bit schedule + 14 rounds) by reusing only the key-independent neural gates of `nn_aes`; overwrites `round_keys` / `buff_round_keys` / `number_of_rounds` without touching `nn_aes.py`. | `python aes256_core.py` → checks the NIST AES-256 test vector. |
| `aes256_attack.py` | CPU/float64 **full master-key recovery demo**: `K0` (bit-wise symmetric) + `K1` (collision column, reduced `2^10` candidate pool for the demo). | `CUDA_VISIBLE_DEVICES=-1 python aes256_attack.py` |
| `aes256_attack_gpu.py` | **Real-attacker GPU attack**: recovers the master key observing **only the final ciphertext**. `K0` via ±δ ciphertext differences; `K1` by a blind `[0, 2^32)` streaming search per column, counting ciphertext collisions over 32 probes (≥ threshold). The search never references the key. | Driven per column by `run_attack.py`. |
| `run_attack.py` | **Orchestrator** for the full `2^32`-per-column real attack: dynamic load-balancing across GPUs, per-column logs, automatic `K0+K1` assembly, and verification of the recovered key against an observed `(plaintext, ciphertext)` pair from the oracle (`oracle_ct.txt`). | `python run_attack.py` (see `--help`) |
| `estimate_k1_margin.py` | **Reproducible across-key margin census** of the `K1` collision oracle: for each of many seeded keys and every column, the correct candidate's collision count vs a random sample of wrong candidates, streamed to CSV/JSON artifacts under `--out`. Every draw is derived from `--seed` via a deterministic SHA-256, so runs are byte-identical and machine-independent. `--shard k/N` splits the keys across cores; `--merge "<glob>"` aggregates the shards into a summary (distribution, max wrong, per-key 95% CI). By default it measures in CPU `float32`; setting `FORCE_FP16=1` forces the released `float16` format even on CPU. The committed run is the `float32` one under `artifacts/fp32/` (10,000 keys). `artifacts/fp16/` holds the same census in `float16` (`FORCE_FP16=1`, same keys and wrong sets) on all 10,000 keys, with its `--merge` summary `k1_margin_seed20260903_summary.json`. | `CUDA_VISIBLE_DEVICES=-1 python estimate_k1_margin.py --nkeys 10000 --nwrong 1000 --seed 20260903 --out artifacts/fp32/k1_margin` |
| `enumerate_forced_candidates.py` | **Exact enumeration of the wrong candidates that score 7 or 14** on the `K1` collision test ("forced" candidates in the code; paper, Proposition 1 and Appendix A). When the round-1 S-box output feeding MixColumns is `0x00` (i.e. the byte is `0x52`), the `0x02` and `0x03` branches carry no perturbation, so two of the four target bytes are absorbed for free and a *wrong* candidate collides on 7 of its 32 probes. This script enumerates that condition exactly from the S-box table alone -- no model, no GPU, ~3 min for 200,000 columns -- and reports the count per column, the score law `7*ell`, the bound `ell <= 2`, the frequency of the `ell = 2` case against its closed form `4/256 - 6/256^2 + 4/256^3 - 1/256^4 = 1.553%`, and the exact-real score of the correct candidate. `--census-seeds 2026,2027,2028,2029` instead predicts the structure of the census keys of `aes256_attack_gpu.py` (wrong candidates scoring 7 or 14 per column, `ell = 2` pairs, exact-real score of the correct candidate) and, with `--hist-dir`/`--hits-dir`, compares it with that census's `col{c}_hist.txt` / `col{c}_hits.txt`. `--model ttables` switches to the T-tables network, where the fused S-box/`0x02`/`0x03` block leaves no target absorbed for free: every wrong candidate scores 0, and the exact-real score of the correct candidate lies in `[23, 32]` (`artifacts/forced_candidates_ttables.log`). Committed log: `artifacts/forced_candidates.log`. | `python enumerate_forced_candidates.py --columns 20000` |
| `measure_score14.py` | The score-14 wrong candidates (`ell = 2`, paper Proposition 1(ii)) exist in ≈1.6% of key columns but are a handful among `2^32`, so the sampled margin census never meets one. This script enumerates every `ell = 2` candidate of the same 10,000 keys exactly and measures each on the network (CPU, float32): 642 candidates in 639 of the 40,000 columns, every one scoring exactly 14. Log: `artifacts/measure_score14.log`. | `python measure_score14.py` |
| `check_census_key.py` | For one census key (`SEED`) on either network (`MODEL=base` or `ttables`), the correct candidates' scores on CPU against the exact-real prediction, and the scores on that network of 64 of the 1024 wrong candidates per column that score 7 on the base network (they score 0 on T-tables). Seconds on CPU. Committed logs: `artifacts/check_census_key_seed2029_base.log`, `artifacts/check_census_key_seed2029_ttables.log`. | `SEED=2029 MODEL=ttables python check_census_key.py` |

**Full run (and a fast check):**
```bash
python run_attack.py --record-all              # full 2^32/column real attack (2 GPUs)
CUDA_VISIBLE_DEVICES=-1 python aes256_attack.py # fast CPU correctness check (reduced pool)
```

**Run options.**

- **`aes256_core.py`** — no options; prints the NIST AES-256 test-vector check.
- **`aes256_attack.py`** — reduced-scale demo. Options: `--demo-bits N` (K1 pool = `2^N`
  per column, default 10), `--trials N` (random keys to recover, default 3),
  `--seed N` (default 2026). Prefix `CUDA_VISIBLE_DEVICES=-1` to force CPU/float64.
- **`run_attack.py`** — the orchestrator and recommended entry point (`--help` for all):

  | flag | default | meaning |
  |------|---------|---------|
  | `--gpus 0,1` | `0,1` | GPU ids to distribute the columns over |
  | `--cols 0,1,2,3` | `0,1,2,3` | which state columns to attack |
  | `--record-all` | off | exhaustive **census** scan: no early exit, record every candidate to `col{N}_hits.txt` (guards against a rare `2^32` false positive) |
  | `--seed N` | `2026` | target-key seed; change it to attack a different key |
  | `--outdir DIR` | cwd | write all logs/artifacts under `DIR` (use a per-seed dir to run several blind keys without clobbering) |
  | `--out FILE` | `recovered_key.txt` | assembled-key output path |
  | `--script FILE` | `aes256_attack_gpu.py` | per-column worker script |

- **`aes256_attack_gpu.py`** — the single-process worker (normally launched by `run_attack.py`).
  Every setting has a built-in default (`FULL_COLS`= all 4 columns, `COLL_MIN`=8,
  `RECORD_ALL`=off, `SEED`=2026, full `2^32`/column), so plain `python aes256_attack_gpu.py`
  already runs a standalone full blind attack; the environment variables below just override those defaults:

  | env var | example | meaning |
  |---------|---------|---------|
  | `CUDA_VISIBLE_DEVICES` | `0` | pick a GPU (or `-1` for CPU) |
  | `FULL_COLS` | `0,1` | columns this process handles |
  | `RECORD_ALL` | `1` | census mode (as `--record-all` above) |
  | `COLL_MIN` | `8` | ciphertext-collision acceptance threshold (of 32 probes) |
  | `SEED` | `2026` | target-key seed |

  `FORCE_FP32=True` at the top of the file switches to float32 if float16 misbehaves.

```bash
# manual two-GPU split (equivalent to `run_attack.py --record-all` on 2 GPUs):
CUDA_VISIBLE_DEVICES=0 FULL_COLS=0,1 RECORD_ALL=1 python aes256_attack_gpu.py
CUDA_VISIBLE_DEVICES=1 FULL_COLS=2,3 RECORD_ALL=1 python aes256_attack_gpu.py

# attack several independent keys without clobbering logs:
python run_attack.py --seed 7  --outdir run_seed7  --record-all
python run_attack.py --seed 42 --outdir run_seed42 --record-all
```

**Verified full-scale runs — artifacts in [`part1/artifacts/`](part1/artifacts).**
Each `run_seed<SEED>/` directory is an independent blind census of a fresh random
key (`random.Random(SEED)`), executed with
`python run_attack.py --record-all --seed <SEED> --outdir artifacts/run_seed<SEED>`:
per column a blind scan of the full `[0, 2^32)` space (the window is *not*
positioned using the key) with **`RECORD_ALL=True`**, on 2 × RTX 5080. In `run_seed2026/`,
`run_seed2027/`, `run_hist_2028/`, and `run_hist_2029_ttables/` all four columns were scanned to completion
(`col{N}.log` end at `4,294,950,912/4,294,967,296`); each column produced
**exactly one ciphertext-collision candidate** — the true `K1` column — i.e.
**zero false positives over the full `4 × 2^32 ≈ 1.7×10^10` search space**
(each `col{N}_hits.txt` holds a single line; the `scanned=` value there is the
*position of that hit*, while the `.log` shows the scan continued to `2^32`). The
assembled 256-bit key was checked against an **observed** `(plaintext, ciphertext)`
pair from the oracle (`run_seed<SEED>/oracle_ct.txt`), not against the key itself
(`run_seed<SEED>/recovered_key.txt`: `VERIFIED`); wall-clock **140h 50m (≈ 5.9 days)**
per key, ≈ `3.4×10^4` candidates/s across the two GPUs (each column ≈ `1.7×10^4`
candidates/s, about 70 h; two columns run concurrently, one per GPU). Each per-column
worker prints `256-bit master-key recovery FAIL` at the end of its own log, because it
recovers one column and checks the full key alone; the assembled key is verified in
`recovered_key.txt`.

All Part I scripts build the base network (`NeuralAESBase`) by default; setting
`MODEL=ttables` in the environment builds GHRS's T-tables network
(`TTablesNeuralAES`) instead, for the census (`aes256_attack_gpu.py`,
`run_attack.py`), the margin census (`estimate_k1_margin.py`, which records
`model=` in its CSV header and summary; the committed CSV headers and the `fp32` summary predate this field) and `check_census_key.py`. The base-network
censuses are seeds 2026-2028; `run_hist_2029_ttables/` is a full census of the
seed-2029 key on the T-tables network (`MODEL=ttables HIST=1`, 109h 58m,
`recovered_key.txt` `VERIFIED`): every wrong candidate in all four columns scores
`0` (no `7` at all, as `enumerate_forced_candidates.py --model ttables` predicts)
and the correct candidates score `31` / `28` / `30` / `32` against the exact-real
`31` / `30` / `30` / `32`. `check_census_key.py` scores the same key on CPU
(`artifacts/check_census_key_seed2029_base.log`, `..._ttables.log`).

`run_hist_2028/` is a third census (seed `2028`) run with `HIST=1`, which
additionally writes `col{N}_hist.txt`: the full histogram of the collision score
over every candidate of the column (33 counts, score `0..32`, plus the `scanned=`
position). All four columns are complete and the assembled key is `VERIFIED`
(`recovered_key.txt`, 140h 50m). In every column every wrong candidate scores `0`
or `7`, exactly `1024` score `7` — the wrong candidates enumerated by
`enumerate_forced_candidates.py --census-seeds 2028 --hist-dir artifacts/run_hist_2028
--hits-dir artifacts/run_hist_2028` — and the correct candidates score
`24` / `32` / `28` / `31` against the exact-real prediction `24` / `32` / `30` / `31`
(column 2 loses two collisions to float16 rounding, as discussed in the paper's
Appendix A.4). No wrong candidate scores `1..6` or `8..32`.

> Note: both `run_attack.py` and `aes256_attack_gpu.py` run the **full blind
> `[0, 2^32)` scan** per column — the search never uses the key (`K1_TRUE` is read
> only for the post-run PASS/FAIL report). For a fast correctness check without a
> multi-day GPU run, use the CPU demo `aes256_attack.py` (reduced candidate pool).
> The full-scale numbers above come from the committed logs under `part1/artifacts/`.

---

## Part II — breaking the Section 9 defense (floating point) + defense

> Run all Part II commands from the `part2/` directory (`cd part2`).

> **Released vs. generic (whose omission this is).** GHRS's generic Secure Blackbox
> Transformation is `D_S^{paper}(p) = ReLU( STEP( D(STEP(p)) ) - MASK(p) )`, with an
> **output-`STEP`** stage. **GHRS's released natural-AES code (`nn_aes.py`) omits that
> output-`STEP` stage** — this omission is in *their* published implementation, not something we
> removed. `fp_key_recovery.py` attacks that released version
> `D_S^{ref}(p) = ReLU( D(STEP(p)) - MASK(p) )`; `fp_key_recovery_output_step.py` **re-inserts the output
> `STEP`** to attack the full generic `D_S^{paper}` and shows the recovery still succeeds, so the
> break does not depend on the missing stage.

| File | What it does | Run |
|------|--------------|-----|
| `step_fp_trigger_experiment.py` | Shows `STEP(trigger)=2` in **every** format, the trigger law `t = 2^(m-1)+1`, the ULP-cancellation mechanism, automatic first-trigger search, CPU/GPU agreement, and a cross-check against GHRS's `ClippingLayer`. | `python step_fp_trigger_experiment.py` |
| `fp_key_recovery.py` | Full 128-bit key recovery in **bf16 / fp16 / fp32 / fp64** (compares the *naive* finite/non-finite and *robust* clean-binary/corrupted oracles; in float64 use the robust oracle), plus a **both-models** check on `NeuralAESBase` and `TTablesNeuralAES` in fp16 (`STEP=2`, `MASK=0`, `\|2-k_i\|` branch). | `CUDA_VISIBLE_DEVICES=-1 python fp_key_recovery.py` |
| `fp_key_recovery_output_step.py` | Shows the attack **survives GHRS's output-`STEP` stage** — the full generic transform `D_S^{paper}`, not just the released `D_S^{ref}` that omits it. A wrong key bit collapses the row to a *degenerate* output (non-finite, or all-zero in fp64), still separable from a valid ciphertext; one valid-ciphertext oracle recovers **128/128 in all four formats, both models**. Set `N_KEYS=10000` (env) for the committed large-scale run: **80,000/80,000** full-key recoveries (10000 keys x 4 formats x 2 models, seed `20260821`), logged to `artifacts/fp_output_step_10000keys.log`. | `CUDA_VISIBLE_DEVICES=-1 python fp_key_recovery_output_step.py` |
| `fp_epsilon_independence.py` | Shows the attack is **independent of `ε`**: 100 random keys per format on each of the two networks (800 recoveries per `ε`) at the released `ε = 1/4` (**800/800**, the `ε = 1/4` row of the paper's Table 3), the published `ε = 1/3` (trigger `2^(f-2)+1`, `STEP = 3/2`, **800/800**), `ε = 1/5` (**800/800**), and the low-margin `ε = 2/5` (656/800). | `CUDA_VISIBLE_DEVICES=-1 python fp_epsilon_independence.py` |
| `counter_fp.py` | Validates the **defense** `clamp01(x) = 1 - relu(1 - relu(x))` (the paper's clamp_RR) against GHRS's `ClippedReLU(x) = relu(x) - relu(x-1)` (clamp_CR): over every finite bfloat16/float16 value and all 2^32 float32 bit patterns, `clamp01` never leaves `[0,1]` while `ClippedReLU` does on 64 / 512 / 4,194,304 inputs (always to `2`); float64 is sampled (2*10^7 random finite values plus 2^k, 2^k+-1). Runs the Section 4 attack against the `clamp01`-prepended model in all four formats (every query returns a valid ciphertext, so no key bit is learned) and checks FIPS-197 correctness. About 75 s on CPU. | `CUDA_VISIBLE_DEVICES=-1 python counter_fp.py` |
| `explore_fp_round1.py` | Explores whether the round-1 key `K1` is reachable by the FP oracle (conclusion: blocked under the defense; a single trigger corrupts a whole `M1` column, 30–32/128, rather than a clean per-bit signal). | `CUDA_VISIBLE_DEVICES=-1 python explore_fp_round1.py` |
| `fp_eps25_vote.py` | At `ε = 2/5`, probes each bit with `M` random base plaintexts and a vote threshold `θ`; recovers **every key in fp16/fp32/fp64** (`600/600`), while `bfloat16` is precision-limited (paper, Appendix C). | `CUDA_VISIBLE_DEVICES=-1 python fp_eps25_vote.py` |
| `fp_eps25_why_corrupt.py` | Diagnoses the `bfloat16` exception at `ε = 2/5`: the non-binary `K0[i]=1` coordinate overflows to a **non-finite** value under every non-zero base plaintext. | `CUDA_VISIBLE_DEVICES=-1 python fp_eps25_why_corrupt.py` |

**Verified outcome (CPU, PyTorch FP16):** on both `NeuralAESBase` and
`TTablesNeuralAES`, 128 chosen queries recover the full AES-128 key; across 100
random keys per model this is 200/200 full keys and 25,600/25,600 correct bits.
Re-inserting the generic output-`STEP` stage (`fp_key_recovery_output_step.py`) still yields
128/128 recovery in every format on both models, because the output `STEP` maps a
corrupted computation to a degenerate output rather than restoring it.

`fp_key_recovery.py` classifies finiteness/magnitude in the native
output dtype and is device-safe (inputs are placed on the model's device), so it
runs unchanged on CPU or GPU.

---

## Paper results and their artifacts

The paper's Appendix D contains only the table below (result, script, committed
artifact) and points here for the options of each script, which the Part I and
Part II tables above document. Paths are relative to `part1/artifacts/` or
`part2/artifacts/`.

| Paper result | Script | Artifact |
|---|---|---|
| Full censuses, two keys (Sect. 3.3) | `run_attack.py --record-all` | `run_seed2026/`, `run_seed2027/` |
| Full score histograms, third key, all four columns (Sect. 3.3) | `HIST=1 run_attack.py --record-all` | `run_hist_2028/` |
| Margin census in float32 and float16 (Sect. 3.3, Fig. 2) | `estimate_k1_margin.py` | `fp32/`, `fp16/` |
| Enumeration of the candidates scoring 14 (Sect. 3.3) | `measure_score14.py` | `measure_score14.log` |
| Enumeration over 200,000 columns, both networks (App. A.3) | `enumerate_forced_candidates.py` | `forced_candidates.log`, `forced_candidates_ttables.log` |
| Full census on the T-tables network, fourth key (Sect. 3.3) | `MODEL=ttables HIST=1 run_attack.py --record-all` | `run_hist_2029_ttables/` |
| Base-network colliding candidates on the T-tables key (Sect. 3.3) | `check_census_key.py` | `check_census_key_seed2029_base.log`, `check_census_key_seed2029_ttables.log` |
| Triggers and STEP(t_F) = 2 in the four formats (Table 2) | `step_fp_trigger_experiment.py` | `step_fp_trigger_experiment.log` |
| Recovery on the released code (Sect. 4.4) | `fp_key_recovery.py` | `fp_key_recovery.log` |
| 80,000 keys with the output-side STEP (Sect. 4.4) | `fp_key_recovery_output_step.py` (`N_KEYS=10000`) | `fp_output_step_10000keys.log` |
| Sweep over ε (Table 3) | `fp_epsilon_independence.py` | `fp_epsilon_independence.log` |
| Validation of clamp01 (Sect. 5.2) | `counter_fp.py` | `counter_fp.log` |
| Multi-plaintext voting (Table 4) | `fp_eps25_vote.py` | `fp_eps25_vote.log` |
| bfloat16 diagnosis over 20 keys (App. C) | `fp_eps25_why_corrupt.py` | `fp_eps25_why_corrupt.log` |

`explore_fp_round1.py` and its log are a diagnostic not reported in the paper.

## Reproduce (CPU)

```bash
# Part II (seconds each)
cd part2
CUDA_VISIBLE_DEVICES=-1 python step_fp_trigger_experiment.py
CUDA_VISIBLE_DEVICES=-1 python fp_key_recovery.py
CUDA_VISIBLE_DEVICES=-1 python fp_key_recovery_output_step.py
CUDA_VISIBLE_DEVICES=-1 python fp_epsilon_independence.py
CUDA_VISIBLE_DEVICES=-1 python counter_fp.py
CUDA_VISIBLE_DEVICES=-1 python explore_fp_round1.py
CUDA_VISIBLE_DEVICES=-1 python fp_eps25_vote.py
CUDA_VISIBLE_DEVICES=-1 python fp_eps25_why_corrupt.py

# Part I (verification, seconds to minutes)
cd ../part1
CUDA_VISIBLE_DEVICES=-1 python aes256_attack.py

# Part I (large-scale real attack, GPU)
python run_attack.py --record-all              # full blind 2^32/column (2 GPUs, ~days)
```

---

## Responsible use

This code is released for academic reproducibility and defensive research on the
security of neural-network implementations of cryptographic primitives. It
recovers keys from a *specific research implementation* under a chosen-input
oracle model; it is not an attack on standard AES. Please use it only on systems
you are authorized to test.

---

## AI assistance

The authors used Claude (Anthropic, Claude Fable 5.1, through Claude Code) as an
assistant in preparing this artifact and the paper. It drafted and revised text
under the authors' direction, wrote and debugged parts of the scripts in this
repository, checked the numbers reported in the paper against the committed logs,
and produced independent consistency reviews. The attacks, the analysis, the
defense, and all experiments were designed and carried out by the authors, who
reviewed every generated line and take full responsibility for the correctness of
the code and the results.

## References

1. D. Gerault, A. Hambitzer, E. Ronen, A. Shamir, *"Deep Neural Cryptography,"*
   EUROCRYPT 2026 (Section 9.1–9.5, Algorithm 7–8, Theorem 1).
2. GHRS's companion implementation: `nn_aes.py`, `utils.py` (included here
   unmodified for reproducibility).

## License

This repository is released under the **GNU General Public License v3.0 or later**
(`LICENSE`), Copyright (C) 2026 the authors (anonymized for review).

`nn_aes.py` and `utils.py` are GHRS's companion files, redistributed here verbatim
under the terms of the GPL v3 of their upstream repository
(<https://github.com/DavidGerault/deep_neural_cryptography>), Copyright (C) their
respective authors. They are **not modified**; their provenance and
line-ending-independent digests are given in the *Reference code* section above.

Because every script in this repository imports those GPL-licensed modules and is
executed together with them as a single program, the artifact as a whole is
distributed under the GPL v3. This is a deliberate choice to match the license of
the upstream code we build on.
