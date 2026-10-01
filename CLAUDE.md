# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An automated magnetics characterization lab. Three coupled parts:

- `scripts/` — Python that drives an **Omicron Bode 100** VNA and a custom **relay switching board** to measure transformers/inductors without manually re-wiring the DUT between measurements. `TransformerModel.py` holds the paper-exact math (Cogitore/Kéradec coupled-inductor model, Blache six-capacitance network, resonance detection, reciprocity/linearity checks) with a real test suite.
- `hardware/relay-board-revB/` — the current KiCad 9 board: 18-relay switching matrix, STM32F072 USB/SCPI controller, and an **integrated B-WIC measurement bridge** (three BNCs straight to the Bode 100 — no external adapter). Generated programmatically by Python, routed by freerouting with locked pre-routes.
- `firmware/` — STM32F072 firmware (libopencm3, USB CDC-ACM, SCPI parser). Crystal-less USB via HSI48 + CRS.

`hardware/relay-board-revB2/` is a work-in-progress respin of revB (USB-C moved to the right edge, D+/D- swapped on the USBLC6, relaxed driver band, extra `cleanup_dangling.py` post-route sweep). `hardware/B-WIC-Adapter/` is the abandoned rev A (kept for reference; its DRC never closed, the relay COM/NC pins were swapped, and its scripts hardcode another user's paths).

## Commands

```bash
pip install -r scripts/requirements.txt   # generate_schematic.py also needs kicad-sch-api (not listed)

# Math/model tests (no instruments needed). Plain script, NOT pytest: check() doesn't
# assert, so pytest would report false passes. Exits 1 on failure.
python scripts/test_model.py

# Synthetic end-to-end pipeline check: builds a known DUT (reference SYNTHETIC), writes its CSVs
python scripts/make_synthetic_dut.py
python scripts/MagneticCharacterizer.py SYNTHETIC --cache

# Relay board self-test: connects, sweeps configs, prints relay states
python scripts/RelayBoardController.py

# Main entry point (argparse): reference [--cache] [--port ASRLn::INSTR] [--bode-ip IP]
#   [--ac-resistance [--gaps ...]] [--no-auto-calibrate]
python scripts/MagneticCharacterizer.py <reference>
```

Hardware scripts that `import pcbnew` **must** run under KiCad's bundled interpreter, not the system Python:

```bash
# Schematic (kicad-sch-api, system Python) — writes relay_board.kicad_sch, runs ERC
python hardware/relay-board-revB/generate_schematic.py

# Netlist truth-table check (graph-simulates all 15 configs + OPEN/SHORT/LOAD)
python hardware/relay-board-revB/verify_board.py

# Board build: placement + all locked pre-routes + zones, then routing
"C:\Program Files\KiCad\9.0\bin\python.exe" hardware/relay-board-revB/generate_pcb.py
"C:\Program Files\KiCad\9.0\bin\python.exe" hardware/relay-board-revB/route_pcb.py

# Post-route audits and outputs
"C:\Program Files\KiCad\9.0\bin\python.exe" hardware/relay-board-revB/parasitic_report.py
python hardware/relay-board-revB/generate_bom.py
python hardware/relay-board-revB/export_fab.py      # DRC-gated Gerber/drill/pos bundle (hardcoded kicad-cli path)
python hardware/relay-board-revB/export_jlcpcb.py   # JLC BOM/CPL xlsx; needs export_fab.py's fab/relay_board_pos.csv
python hardware/relay-board-revB/simulate_calibration.py   # needs ngspice; geometry comes from generate_pcb.py constants

# Firmware (arm-none-eabi + libopencm3 at $OPENCM3_DIR, default ../libopencm3,
# built with: make -C libopencm3 TARGETS=stm32/f0)
make -C firmware
make -C firmware flash   # st-flash
make -C firmware dfu     # dfu-util; hold BOOT0 while plugging USB
```

## Architecture

Three layers, bottom-up:

1. **`Bode100Analyzer.MagneticMeasurer`** — SCPI over TCP to the Bode 100 (`TCPIP::<ip>::5025::SOCKET`; IP is a default argument, override with `--bode-ip`). Measurement primitives (IAD impedance method, long-form pandas DataFrames indexed by `measurement_index` + `frequency`): `take_Rs_Ls_measurement`, `take_Z_phase_measurement`, `take_Cs_measurement`. `calibrate()` requires an explicit calibration group; it loads an existing `.mcalx`, otherwise falls back to an interactive manual OSL. It never touches the relay board.
2. **`RelayBoardController`** — SCPI over serial (115200) to the STM32, auto-detected by scanning `ASRL*` resources for an `*IDN?` starting with `OpenMagnetics,RelayBoard`. Exposes 15 configs (`CONF:MEAS n`), `signal_path()` naming, `calibration_overlay()`, and `set_calibration_mode()` (`CAL:MODE OPEN|SHORT|LOAD|MEAS`). `MATRIX_BITS`/`LOAD_HI_BIT`/`ISOLATE_BITS` mirror `firmware/include/relay_map.h` — **keeping the two in sync is manual**: `verify_board.py` checks `MATRIX_BITS` against the netlist but nothing reads `relay_map.h` (despite comments claiming otherwise).
3. **`MagneticCharacterizer`** — orchestration. Recipes (`measure`, `verify_switching`, `verify_linearity`, `characterize_inductance/resistances/capacitance/ac_resistance`, `characterize_all`): switch config → (re)calibrate if the signal path changed → measure → derive a parameter via `TransformerModel` → plot.

### Calibration is per signal path, at the isolation-relay contact plane

The OSL plane sits at the isolation relay contacts: OPEN = all four iso relays energized, SHORT = additionally close the first HI terminal's LO relay, LOAD = a fifth DUT-less matrix column with an on-board 100 Ω 0.1 % standard (R5, K13/K14). The automatic sequence lives in `MagneticCharacterizer._ensure_calibrated` (drives `set_calibration_mode`, skipped with `--no-auto-calibrate`) and writes per-path `relay_board_{signal_path}.mcalx` under `scripts/calibrations/` (`isi_board.mcalx`/`small_board.mcalx` there are rev A leftovers). Never call `relay_board.set_config()` directly from characterization code — use `MagneticCharacterizer._set_config()`, which handles recalibration on path changes.

### Cache and offline mode

Raw sweeps go to `scripts/output/{reference}_cfg{NN}_{configname}_{RL|Z|Cs}.csv` (plus `{reference}_acr_{label}_RL.csv`, `{reference}_ac_resistance.csv`, `{reference}_results.json`); `allow_use_cache=True` / `--cache` reads an existing CSV instead of re-measuring. Always use the cache when working on math/plotting rather than measurement. If either instrument fails to connect, the constructor silently sets `self.offline = True` and only cached data works.

### Hardware generation flow (rev B)

`generate_schematic.py` → `relay_board.kicad_sch` + ERC. `generate_pcb.py` (pcbnew) is the source of truth for the layout, including **every measurement-critical net as locked pre-routed copper**. 4-layer stackup: F.Cu signals, In1.Cu +5 V, In2.Cu GND, B.Cu coil returns/AGND island/USB. `verify_board.py` checks the exported `relay_board.xml` netlist against a per-relay pin→net truth table and simulates every config — update it alongside any schematic change. `DESIGN_NOTES.md` records the parasitic/calibration reasoning.

`route_pcb.py` automates the freerouting `.dsn`/`.ses` round trip for the non-critical nets only, then repairs freerouting's clearance blindness near locked copper, sweeps dangling fragments, drops fanout vias for plane nets, fills zones, runs DRC, and loops until nothing is unconnected. Do not hand-route in the GUI; change `generate_pcb.py` and re-run.

Exception: revB's `fix_usb_orientation.py`, `fix_usb_cleanup.py`, `fix_usblc6_led.py` are one-shot patches applied directly to the fab `relay_board.kicad_pcb` (run with cwd = the board dir; they write `relay_board_backup_*`). The fab master therefore differs from a fresh `generate_pcb.py` run — don't regenerate revB over it.

`relay_board_BOM.*`, `relay_board_JLC_*.xlsx`, `fab/`, `relay_board_fab.zip`, parasitic output, and `*_erc.txt` / `*_drc.txt` are generated — regenerate, don't hand-edit.

`tools/` (freerouting 1.9.0 jar + JRE, kicad-happy skills) is gitignored and not in the repo — obtain it separately; `route_pcb.py` falls back to `java` on PATH.

### Known KiCad 9 pitfalls (hard-won)

- `ZONE.SetLayer()` silently no-ops — use `LSET` + `SetLayerSet` (see `zone_layer()` in generate_pcb.py); `GetLayerName()` on zones lies, verify in the saved file.
- freerouting ignores clearance to locked ("fix") copper — hence the repair/cull/re-route loop in route_pcb.py. DSN keepouts and net-withdrawal make it worse; "(type power)" inner layers stop inner routing but strand plane pads (hence the deterministic fanout).
- pcbnew `Flip()` needs `Add()` first; scripts here avoid flipping entirely (single-sided assembly).
