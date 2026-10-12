---
name: bench-duts-and-setup
description: "What was measured on the rev B relay bench (DUT identities, references, key results) and how the bench PC is set up (SCPI server, firmware toolchain, DFU driver)"
metadata:
  node_type: memory
  type: project
  originSessionId: 873f79ad-e1f0-432e-b64f-4a0fd8f24ab4
  modified: 2026-10-11T01:07:54.185Z
---

DUT references in scripts/output/ (cached data the user wants kept; label new ones with `--describe`):
- `DUT_20261010`, `DUT_20261010_final` (2026-10-10): toroid T26, Ferroxcube 3C90, 10 + 10 turns (1:1).
  C33 18.75, C13 -8.6, C23 +5.7, C11 11.5, C22 8.3, C12 -4.5 pF; k 0.9985, Lp ~251 uH, ls 0.739 uH (path-corrected). Self-C peaks ~11 pF at 5 MHz then falls
  (windings on MnZn). `_final` is the clean run after the fixture fixes.
- `RM_concentric_20261011` (2026-10-11): RM core, concentric windings, 1:1. k 0.9978, Lp 231.8 uH,
  ls 1.006 uH (path-corrected), C33 21.7, C13 -7.4, C23 +13.1, C11 8.6, C22 18.4, C12 -8.5 pF,
  self-C ~9.9 pF (rising plateau); C13+C23 agrees
  between open and short families (6.2 / 5.7), unlike the toroid (2.4 pF apart).

Bench PC setup:
- Bode 100 R2 (serial LN604P) over USB; SCPI via OMICRON ScpiRunner on 127.0.0.1:5025 (see CLAUDE.md bench notes).
- Relay board on COM7 (ASRL7::INSTR), firmware 1.2.0 since 2026-10-11.
- Firmware toolchain in C:\Users\Alfonso\tools: arm/ (Arm GNU 14.2), libopencm3/ (built f0),
  dfu-util-0.11-binaries/, zadig.exe; make via winget (ezwinports.make, not on Git Bash PATH by default).
  Build: PATH += tools/arm/bin and the winget make dir; `make -C firmware OPENCM3_DIR=C:/Users/Alfonso/tools/libopencm3`.
- Update: `SYST:DFU` (RelayBoardController.enter_dfu) then
  `dfu-util -d 0483:df11 -a 0 -s 0x08000000:leave -D firmware/build/relay_board.bin`.
  WinUSB on 0483:DF11 was installed with Zadig (needs the user to approve UAC; done once).

**Why:** the user asked to keep these results and will keep adding DUTs; the setup took long to discover.
**How to apply:** reuse these references/paths instead of re-measuring or re-installing; compare new DUTs
against these; never flash a firmware image that lacks SYST:DFU (no BOOT0 access on rev B).
