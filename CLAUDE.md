# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An automated magnetics characterization lab. Three coupled parts:

- `scripts/` — Python that drives an **Omicron Bode 100** VNA and a custom **relay switching board** to measure transformers/inductors without manually re-wiring the DUT between measurements. `TransformerModel.py` holds the paper-exact math (Cogitore/Kéradec coupled-inductor model, Blache six-capacitance network, resonance detection, reciprocity/linearity checks) with a real test suite.
- `hardware/relay-board-revB/` — the current KiCad 9 board: 18-relay switching matrix, STM32F072 USB/SCPI controller, and an **integrated B-WIC measurement bridge** (three BNCs straight to the Bode 100 — no external adapter). Generated programmatically by Python, routed by freerouting with locked pre-routes.
- `firmware/` — STM32F072 firmware (libopencm3, USB CDC-ACM, SCPI parser). Crystal-less USB via HSI48 + CRS.

`hardware/B-WIC-Adapter/` is the abandoned rev A (kept for reference; its DRC never closed and the relay COM/NC pins were swapped).

## Commands

```bash
pip install -r scripts/requirements.txt

# Math/model tests (no instruments needed)
python scripts/test_model.py

# Synthetic end-to-end pipeline check: builds a known DUT, runs the recipes
python scripts/make_synthetic_dut.py

# Relay board self-test: connects, sweeps configs, prints relay states
python scripts/RelayBoardController.py

# Main entry point — see "Running a characterization" below
python scripts/MagneticCharacterizer.py
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
python hardware/relay-board-revB/export_fab.py     # DRC-gated Gerber/drill/pos bundle
python hardware/relay-board-revB/simulate_calibration.py   # needs ngspice on PATH

# Firmware (arm-none-eabi + libopencm3)
make -C firmware
```

## Architecture

Three layers, bottom-up:

1. **`Bode100Analyzer.MagneticMeasurer`** — SCPI over TCP to the Bode 100 (`TCPIP::<ip>::5025::SOCKET`, IP hardcoded in `__init__`). Owns OSL calibration and the measurement primitives (IAD impedance method, long-form pandas DataFrames indexed by `measurement_index` + `frequency`): `take_Rs_Ls_measurement`, `take_Z_phase_measurement`, `take_Cs_measurement`. `calibrate()` requires an explicit calibration group and drives the board's `CAL:MODE OPEN|SHORT|LOAD` relay overlays.
2. **`RelayBoardController`** — SCPI over serial (115200) to the STM32, auto-detected by scanning `ASRL*` resources for an `*IDN?` starting with `OpenMagnetics,RelayBoard`. `MATRIX_BITS`/`LOAD_HI_BIT`/`ISOLATE_BITS` mirror `firmware/include/relay_map.h` — **the two tables must stay in sync** (verify_board.py cross-checks them). Exposes 15 configs (`CONF:MEAS n`), `signal_path()` naming, and `calibration_overlay()`.
3. **`MagneticCharacterizer`** — orchestration. Each `characterize_*` method is a recipe: switch config → (re)calibrate if the signal path changed → measure → derive a parameter via `TransformerModel` → plot.

### Calibration is per signal path, at the isolation-relay contact plane

The OSL plane sits at the isolation relay contacts: OPEN = all four iso relays energized, SHORT = additionally close the first HI terminal's LO relay, LOAD = a fifth DUT-less matrix column with an on-board 100 Ω 0.1 % standard (R5, K13/K14). Calibration files are per-path `.mcalx` under `scripts/calibrations/`, acquired automatically with the DUT clamped — there is no manual open/short/load fixture step and no `CALIBRATION_GROUPS` table anymore. Never call `relay_board.set_config()` directly from characterization code — use `MagneticCharacterizer._set_config()`, which handles recalibration on path changes.

### Running a characterization

`MagneticCharacterizer.py`'s `__main__` block is the UI: a stack of commented-out constructor calls (one per DUT reference) and method calls. Edit which lines are uncommented rather than adding an argument parser, unless asked to.

Every `characterize_*` method takes `allow_use_cache`. Raw sweeps are written to `scripts/output/{reference}_{recipe}_{quantity}.csv`; with `allow_use_cache=True` an existing CSV is read back instead of re-measuring. This is how analysis/plotting/curve-fitting changes get iterated on with no instruments connected — always pass `True` when working on the math rather than the measurement. `make_synthetic_dut.py` writes a full synthetic CSV set (reference `SYNTHETIC`) with known ground truth for pipeline validation.

Recipes come in `_basic` / `_medium` / `_advanced` tiers: basic reads a value straight off the sweep, medium applies closed-form relations, advanced fits a lumped model with `scipy.optimize` over several configs.

### Hardware generation flow (rev B)

`generate_schematic.py` (kicad-sch-api) → `relay_board.kicad_sch` + ERC. `generate_pcb.py` (pcbnew) is the single source of truth for the layout: board outline with USB tab, face-to-face DUT clamp bay top-center, 4×3 relay crossbar + load column + per-terminal iso relays, the B-WIC bridge band with three BNCs along the bottom edge, and **every measurement-critical net as locked pre-routed copper** (arms, buses, rails, bridge, coil returns, USB differential pair, strap grounds). 4-layer stackup: F.Cu signals, In1.Cu +5 V, In2.Cu GND, B.Cu coil returns/AGND island/USB. `verify_board.py` checks the exported `relay_board.xml` netlist against a per-relay pin→net truth table and simulates every config — update it alongside any schematic change.

`route_pcb.py` automates the freerouting `.dsn`/`.ses` round trip (vendored jar + JRE in `tools/`) for the non-critical nets only, then repairs freerouting's known clearance blindness near locked copper, sweeps dangling fragments, drops fanout vias for plane nets, fills zones, and runs DRC. It loops extra freerouting rounds until nothing is unconnected. Do not hand-route in the GUI; change `generate_pcb.py` and re-run the two scripts.

`relay_board_BOM.md`, `relay_board_netlist.md`, `parasitic` output, and the `*_erc.txt` / `*_drc.txt` reports are generated — regenerate, don't hand-edit. `DESIGN_NOTES.md` records the parasitic/calibration reasoning behind the layout.

Firmware ↔ driver ↔ netlist consistency is enforced three ways: `relay_map.h` and `RelayBoardController.MATRIX_BITS` carry the same bit table, and `verify_board.py` walks both against the netlist.

`tools/` is vendored third-party tooling (freerouting + JRE, kicad-happy review skills) — not project source.

### Known KiCad 9 pitfalls (hard-won)

- `ZONE.SetLayer()` silently no-ops — use `LSET` + `SetLayerSet` (see `zone_layer()` in generate_pcb.py); `GetLayerName()` on zones lies, verify in the saved file.
- freerouting ignores clearance to locked ("fix") copper — hence the repair/cull/re-route loop in route_pcb.py. DSN keepouts and net-withdrawal make it worse; "(type power)" inner layers stop inner routing but strand plane pads (hence the deterministic fanout).
- pcbnew `Flip()` needs `Add()` first; scripts here avoid flipping entirely (single-sided assembly).
