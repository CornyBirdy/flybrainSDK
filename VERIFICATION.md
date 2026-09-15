# VERIFICATION — adversarial audit of the `male_cns` work

Status: **IN PROGRESS** — this file is being written as the audit runs.
Sections marked `[PENDING]` are not yet complete. Do not read a `[PENDING]`
section as a result.

Auditor: a session that did not write the code under audit.
Date: 2026-09-15.

---

## 0. READ THIS FIRST — the machine is not the machine in the brief

The task specifies Windows 11, Ryzen 5 3600 (6c/12t), 32 GB DDR4, RTX 5070 Ti.

**This session is not running on that machine.** It is running in a remote
Linux container:

```
Linux 6.18.44-fc-v33 x86_64        (not Windows 11)
Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores   (not Ryzen 5 3600, 6c/12t)
15 GB RAM                          (not 32 GB)
no GPU, no nvidia-smi              (not an RTX 5070 Ti)
```

This is the *same class of machine the prior session used* — `NOTES.md` §12
says "a 4 vCPU / 15 GB Linux container", and §7 says "Intel Xeon @ 2.10 GHz,
4 vCPU, 15 GB RAM". So:

* **Every Windows-specific question in the brief is unanswerable here.** Path
  handling, `FLYBRAIN_MALECNS_DIR` on Windows, `pyarrow`/`scipy` Windows
  wheels — none of it was exercised. See §4.
* **Part F cost numbers are not the numbers the user asked for.** They are
  numbers for a 4-core 2.8 GHz Xeon, i.e. a re-measurement of the prior
  session's own platform, not a port to the target workstation.
* The "confirm peak RAM stays sane on 32 GB" instruction was tested against
  **15 GB**, which is a strictly harder test and it passed.

Everything else in this report is a real re-derivation on real data.

---

## 1. Verdict

`[PENDING — written last]`

---

## 2. Confirmed

`[PENDING]`

---

## 3. Did not reproduce

`[PENDING]`

---

## 4. Could not check

`[PENDING]`

---

## 5. Findings

`[PENDING]`

---

## 6. Environment

| | |
|---|---|
| OS | Linux 6.18.44-fc-v33, x86_64, glibc 2.39 |
| CPU | Intel Xeon @ 2.80 GHz, 4 cores |
| RAM | 15 GB |
| GPU | none |
| Python | 3.11.15 (GCC 13.3.0) |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| flyvis | 1.2.0 |
| BLAS | scipy-openblas 0.3.31 |

Nothing was pre-installed; the venv was built from scratch for this audit.
`NOTES.md` §7 reports "NumPy 2.4 / SciPy 1.17", which matches the major
versions used here.
