# revB2 TODO

## BOOT0 has no way to be driven high (blank boards cannot be flashed over USB)

**Found on:** first revB board bring-up, 2026-10-01.

**Problem.** BOOT0 (U1 pin 44) is only tied to GND through R4 (10k). A blank
STM32F072 does **not** fall back to the system bootloader when its flash is
empty (no empty-check on this part, confirmed on hardware), so a fresh board
runs nothing and never enumerates on USB. The first board was flashed by
hand-soldering wires to the left pads of R4 (BOOT0) and R3 (+3V3) and touching
them together while plugging USB in. J8 (SWD) is the only other entry point and
needs an ST-Link; it is not populated by JLC.

**Fix (unchanged in revB2 so far):** `generate_schematic.py:436-439`
(R4 pull-down), placement at `generate_pcb.py:139`.

1. Add a tactile switch **SW1 between BOOT0 and +3V3**, next to the USB-C
   connector so it can be held while plugging in. Keep R4 10k to GND. Optionally,
   add a series 1k on the button side so a stuck/shorted switch can't fight a
   future GPIO.
2. Also bring BOOT0 and +3V3 out to two adjacent **test pads** (or a 2-pad
   solder jumper) as a no-BOM fallback.
3. Optional: add a **RESET button** (NRST to GND) so entering DFU doesn't need
   replugging: hold BOOT, tap RESET.
4. Choose a JLC basic/extended SMD tact switch so it's assembled, then update
   `verify_board.py`, the BOM, and re-run generate_schematic → generate_pcb →
   route_pcb → verify_board → export_fab.

**Related firmware item:** add a `SYST:DFU` SCPI command that jumps to the
F072 system bootloader (0x1FFFC800), so already-flashed boards can be
updated over USB with no button at all. The button is then only needed for
first flash / recovery.
