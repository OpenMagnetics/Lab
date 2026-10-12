# devtools -- inspection and debugging for the rev C routing flow

Everything here runs from any checkout (paths are relative to this folder). `K` below is KiCad
10's Python: `"C:\Program Files\KiCad\10.0\bin\python.exe"`.

## Look at the board

| Script | Use |
|---|---|
| `dump_copper.py BOARD OUT.json` (K) | dump tracks, vias, pads, refs, courtyards to JSON |
| `render_copper.py IN.json OUT.png x0 x1 y0 y1 [F.Cu,B.Cu,...]` (system Python, matplotlib) | render a window of the board (mm) to PNG -- the fastest way to see why a net is stuck |
| `pads_of.py BOARD REF [REF...]` (K) | pad numbers, nets, positions, sizes |
| `list_bboxes.py BOARD x0 x1 y0 y1` (K) | footprints inside a window |
| `fp_info.py LIB:NAME ...` (K) | footprint bounding box / courtyard from the library |

Typical: `K dump_copper.py ../relay_board.kicad_pcb c.json && python render_copper.py c.json c.png 168 196 96 124 F.Cu,B.Cu`.

## Debug the router (maze_route.py)

| Script | Use |
|---|---|
| `astar_test.py NET [NET...]` (K) | route nets ALONE on the current board (fine grid, with fan-out escapes and the exact clearance oracle), printing every A* call and its time. If a net routes alone but fails in the pass, it is congestion/ordering, not geometry. Net names without the leading `/` (Git Bash rewrites `/X` into a path). |
| `maze_dbg.py --fine [--board=PATH] NET` (K) | flood-fill each island of a net under the router's own rules (outer layers, vias, A* diagonal rule): how many cells each island reaches and whether islands can meet. A small "reachable" count = a sealed pocket (pad boxed in). |
| `fan_test.py [BOARD]` (K) | run only the fine-pitch fan-out (escape stubs) and save `fan_test.kicad_pcb`; prints which pads had no room |
| `fan_diag.py REF [BOARD]` (K) | per pin of a fine-pitch part: clearance margin of a straight 1 mm escape and the nearest obstacle -- finds what blocks an escape |

## External tools on a new PC

- **KiCad 10** (pcbnew Python, kicad-cli). The flow scripts assume `C:\Program Files\KiCad\10.0`.
- **Firmware**: arm-none-eabi toolchain + libopencm3 built for stm32/f0
  (`make -C libopencm3 TARGETS=stm32/f0`), GNU make (winget `ezwinports.make`), then
  `make -C firmware BOARD=revC OPENCM3_DIR=<libopencm3>`.
- **kicad-happy skills** (EMC pre-compliance, PCB/schematic analysis, JLCPCB, LCSC part search; MIT,
  github.com/aklofas/kicad-happy): `bash setup_skills.sh` clones it and installs the five skills
  used here into `~/.claude/skills`.
- **Faraday** (OpenConverters EMC review, github.com/OpenConverters/Faraday): clone it, then
  `build_faraday.bat <clone>` (MSVC 2022 + its bundled CMake/Ninja; needs `/D_USE_MATH_DEFINES`).
  `../faraday_review.py` runs it on the board.
- Python packages: `pip install -r ../../../scripts/requirements.txt kicad-sch-api matplotlib scipy pymupdf`.
- Not needed any more: freerouting / the JRE (routing is all `maze_route.py`).

## Claude memory

`claude-memory/` holds the assistant's notes from the previous PC. To restore them, copy the
`.md` files into `~/.claude/projects/<project-folder-key>/memory/` (Claude Code creates the folder
the first time it runs in this repo; the key is the repo path with separators replaced by `-`).
