# revC TODO

## HANDOFF 2026-10-12 -- routing in progress (resume here)

Routing is done entirely by our own router now (no freerouting). State when this was committed:
coarse pass routes 64/65 nets; the fine pass (fan-out + rip-up/reroute + exact clearance)
converges everywhere except the relay-drive lanes, which were still being tuned. The committed
relay_board.kicad_pcb is the COARSE-routed board (fine pass not finished) -- regenerate, don't
hand-edit.

Resume (KiCad 10 interpreter, from hardware/relay-board-revC):

    python generate_schematic.py && python verify_board.py && python generate_firmware_map.py
    K="C:/Program Files/KiCad/10.0/bin/python.exe"
    $K generate_pcb.py && $K maze_route.py && $K maze_route.py --fine
    $K route_pcb.py --no-freerouting && $K optimize_tracks.py && $K parasitic_report.py

(coarse ~25 min, fine ~1-2 h). Last open item: the west-column relay lanes (GPIO1-7, 14,
pins PA3..PB2) cross the north-row/east lanes; maze_route.LAYER_COST now puts that group on
B.Cu and the others on F.Cu (run v16 was testing this). If they still thrash, consider the
lane order in generate_schematic.GPIO_PINS or rotating the MCU so its GPIO-rich side faces the
drivers. Run v16 (layer split) also showed GPIO15 (PA0) and GPIO23 (PA1), adjacent north-row lanes to U4 p1 /
U6 p2, ripping each other: check their lane order (GPIO23 -> nearer input should be the westmore
pin) with devtools/astar_test.py before anything else. Not yet run end-to-end: route_pcb --no-freerouting -> optimize_tracks -> parasitic_report
(coupling audit) -> faraday_review / kicad-happy emc skill. Then export_fab/export_jlcpcb.

Done this session (all in source): rev B2 renamed rev C; EU EMC plan (EMC_COMPLIANCE.md);
firmware watchdog + relay-state restore; driver USB-drop reconnect; L1 CM choke netlist bug
fixed (TDK ACM2012 windings 1-4/2-3; D+/D- were swapped); MCU pins reassigned in lane order
(relays N/E/W sides, I2C2 on PB10/11, LEDs + coil switch on the south row, RUN on PA9); north-side
MCU parts re-placed (VDDA stack over pin 9, NRST over pin 7, C8 in the NE notch for VDD48+VBAT);
test pads beside their circuits; LEDs moved SW of the MCU; VBUS pre-routed under J1 on B.Cu;
TMP117 access pocket; measurement nets outer-layers-only with a crosstalk cost; maze router
fine pass with fan-out, inward vias, exact clearance, net-tie transparency, rip-up (path
congestion cost, priority classes, protected generator copper and escapes).

## BOOT0 has no way to be driven high (blank boards cannot be flashed over USB)

**Found on:** first revB board bring-up, 2026-10-01.

**Problem.** BOOT0 (U1 pin 44) is only tied to GND through R4 (10k). A blank
STM32F072 does **not** fall back to the system bootloader when its flash is
empty (no empty-check on this part, confirmed on hardware), so a fresh board
runs nothing and never enumerates on USB. The first board was flashed by
hand-soldering wires to the left pads of R4 (BOOT0) and R3 (+3V3) and touching
them together while plugging USB in. J8 (SWD) is the only other entry point and
needs an ST-Link; it is not populated by JLC.

**Fix (unchanged in revC so far):** `generate_schematic.py:436-439`
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

**Related firmware item (done, firmware 1.1.0):** `SYST:DFU` reboots a flashed
board into the ROM DFU bootloader (`RelayBoardController.enter_dfu()`), so
updates need no strap; verified on hardware 2026-10-01. The button is only
needed for first flash / recovery.

## Status LED is too bright

D1 is fed through R10 = 1k (~1.4 mA). Firmware 1.1.1 dims it with 2 % software
PWM (10 kHz SysTick); raise R10 (e.g. 4.7k–10k) in revC so it isn't distracting without PWM.

## Measurement-path fixes from the rev B bench bring-up (2026-10-10/11)

Measured on the rev B board with two DUTs (T26 3C90 toroid, RM concentric); numbers are
reproducible with `python scripts/bench.py column-map`, `consistency`, `fixture-paths`.
Everything below is outside the per-path OSL plane, so it lands in the DUT result.

### 1. Column paths are asymmetric because both rails are fed near the middle  — IN PROGRESS
**Measured:** HI→column→LO relative to column A: **B −39.3 nH, C −32.7 nH, D +4.1 nH**
(repeatable to 0.04 nH). **Cause:** the HI rail is fed at x = 87 (DESCENT_HI_X) and the LO
rail at x = 111 (DESCENT_LO_X); per column the rail length HI+LO is A 88 mm, B 44 mm,
C 44 mm, D 88 mm (~0.8 nH/mm). This is what put the reciprocity check 7.5 % off and biased
Lsc by ~5 % (now corrected in software from `fixture_paths.json`).
**Fix:** feed the HI rail from the WEST end (west of column A) and the LO rail from the EAST
end (east of column D): HI(t) + LO(t) is then the same for every column. The common
extra length sits inside every path's OSL, so it calibrates out.

### 2. LINK net: ~14.5 pF to ground and 39–94 nH loops through two column buses
**Measured:** the series-aiding identity 4·Y(Lcum) = Y(BC open) leaves −19 pF on both DUTs →
LINK net (rail + two linked columns + their relays) ≈ **14.5 pF to board ground**; it sits
at V/2 in Lcum/Ldif and the bridge (current sensed in LO only) sees −Cg/4.
LINK loops A→t: **B +39 nH, C +53 nH, D +94 nH** (111–198 mΩ).
**Likely cause:** ~12 relays hang on the LINK net when two columns are linked (4 LINK relays
+ 3 matrix relays and an iso relay per linked column); their contact-to-coil capacitance goes
to AC ground through the drivers. The planes are already voided under the signal region
(PLANE_VOID), so it is not plane capacitance.
**Fix (needs a decision — changes firmware/driver tables):** replace the LINK rail and the
four column LINK relays (K3, K6, K9, K12) by **dedicated link relays directly between the
terminal nodes** at the iso plane: A-B, C-D (winding shorts) and B-C, B-D (series aiding /
opposing). Same relay count; links become a few mm long, symmetric, with one contact and no
column bus. All LINK uses today are A+B, C+D, B+C, B+D (configs 2, 4, 5, 6, 18, 19).

### 3. Isolated clamp still couples ~2 pF to ground, ~0.1 pF to HI
**Measured:** guarded-model fit of the open-link family on both DUTs (clamp→GND 1.95/2.26 pF,
clamp→HI 0.13/0.09 pF). A floating terminal is isolated, but its clamp, arm and the iso
relay's open NC contact (plus that relay's coil) remain on the DUT node.
**Fix:** shorten the arm/NC stub, keep copper and the coil-drive traces away from the clamp
bay, and route the iso relay coil feed on the far side of the relay.

### 4. SHORT standard does not follow the DUT path
The OSL SHORT closes the first HI terminal's LO relay, skipping the LO terminal's column and
contact that the DUT path includes. With fix 1 the residual is symmetric; a dedicated
per-pair short at the iso COM side would remove it entirely.

### 5. (unchanged) Kelvin force/sense to the clamps; populate G6K-2F-RF-S (lower open-contact
capacitance than the -Y used on the prototype). See ../relay-board-revB/TODO_revC.md.

## Rev C implementation status (2026-10-11)

Architecture, the review response (N. Rosano) and the reasoning are in `ARCHITECTURE.md`.

### Done in the generators (schematic ERC 0, `verify_board.py` passes)
- [x] BOOT0 button SW1 (+ R17 1k series), RESET button SW2, BOOT0/+3V3 strap pads TP1/TP2, R10 4.7k.
- [x] End-fed rails: HI fed west of column A (SOURCE), LO fed east of column D (shunt + CH2).
- [x] Direct link relays K15..K18 (A-B, C-D, B-C, B-D); LINK rail and K3/K6/K9/K12 removed.
- [x] SHORT: four SHORT relays were designed, simulated and dropped (ARCHITECTURE.md section 3): same-size
      residual, extra Cps error; replaced by clamp-plane fixture compensation (`bench.py fixture-compensation`).
- [x] Standards column: K19..K21 select 100R LOAD, 1R and 10 pF check standards (Kelvin-sensed), verified
      after every OSL.
- [x] Full Kelvin: clamp pad 1 force / pad 2 sense, pole 2 of every signal relay switches sense,
      2-wire tie K22, CH1 select K23, OPA356 front-end (`simulate_frontend.py`: CMRR 60 dB with
      0.1 % mismatch; flat 11 Hz-12.5 MHz into CH1 at 1 Mohm, 240 Hz-12 MHz at 50 R).
- [x] Kelvin LOAD: R5 sensed at its pads through net ties NT1/NT2.
- [x] Unified ground (AGND island + R11 removed), 4-layer JLC04161H-7628 SIG/GND/GND/SIG (stackup in
      the board file), all cable connectors on the bottom edge, VBUS ferrite, USB CM choke, stitching vias
      <= 4 mm, return-path vias at every signal via, decoupler vias at 90 degrees beside the pads.
- [x] MCU per AN4325: 100 nF at every VDD/VDDIO2/VBAT pin, VDDA ferrite + 1 uF + 10 nF, 4.7 uF bulk;
      Cortex 2x5 SWD header next to the MCU; MCU rotated so USB/SWD pins face their connectors.
- [x] Status LEDs: PWR (hardwired), RUN (PC13: 1 Hz blink until the USB host configures the
      board, then on), BUSY (PB6: relays switching), KELVIN (PB7), PROG (PB8: set before the ROM
      DFU jump, lit through a firmware update).
- [x] Firmware 2.0.0: `make BOARD=revC` (generated `relay_map_revc.h`), staggered coil switching,
      16-relay USB budget, unused pins analog, `CONF:KELV`. Driver picks the map from `*IDN?`.

- [x] DUT interface: 8 banana jacks (Kelvin F/S pairs, 19.05 mm) for Grayhill 02-1 clips / Kelvin leads /
      adapter boards; iso relays between the jack rows; arms F.Cu, no vias.
- [x] TMP117 fixture temperature (calibration re-acquired on > 2 C drift), UID serial, relay counters,
      test pads, TPS2051C coil switch (enabled after enumeration), SMF5.0A VBUS TVS, USB shield RC option,
      10 uF per driver, plated GND mounting holes + cable-edge bonding strip.

### JLC rules used from the start (jlcpcb.com capabilities, 2026-10)
0.15 mm min track (0.25 default), 0.18-0.2 mm clearance, vias 0.3/0.6 mm, 0.5 mm copper to edge,
parts >= 2.5 mm from the edge (no rails needed), silk 0.15 mm / 1.0 mm text, 3 fiducials
(1 mm, 2 mm mask, >= 3.35 mm from the edge); no via-in-pad (a paid option on 4 layers).

### Still to verify before ordering
- [ ] Route + DRC clean (route_pcb.py), then `parasitic_report.py` updated for the Kelvin nets.
- [x] CM choke pinout (2026-10-12): TDK ACM2012 windings are 1-4 and 2-3 (TDK equivalent-circuit
      library), not 1-2 / 3-4 -- the first netlist swapped D+/D- at the connector (USB would never
      have enumerated). Netlist fixed, footprint turned 180 deg so the pair reaches D2 uncrossed.
- [ ] LCSC numbers for OPA356, BAV99, BLM21PG221SN1D, ACM2012, 47 uF 1206 (left to JLC's matcher).
- [x] Prototype relays: G6K-2F-Y-TR (C47190, JLC stock) -- decided 2026-10-11; RF-S only if the measured
      open-contact capacitance floor demands it.
- [ ] Bode 100 SCPI for the CH1 1 Mohm input in Kelvin mode; Kelvin recipe + calibration in
      MagneticCharacterizer; one-time 2-wire fixture compensation (open/short at the clamps).
- [x] `simulate_calibration.py` (ngspice DLL) and `kelvin_coupling.py` for the SHORT decision.
- [ ] `export_fab.py` for 4 layers.
- [ ] Shorting-bar accessory for the fixture compensation (a 4-clamp bar or dummy header); declare its
      inductance with `--short-bar-nh`.
- [ ] LCSC for R18 (1R 0.5 % 1206), C30 (10 pF C0G +/-0.1 pF 0603), TMP117AIDRV, TPS2051CDBV,
      SMF5.0A, banana jacks (hand-fit; Cinch/Johnson 108-0740-001 or Keystone 575-4) and Grayhill 02-1 (accessory).
- [x] Decided 2026-10-11: rev C jacks routed as they are (no bleed resistors, no ESD diodes); treat them as
      measurement terminals for ESD and confirm with the lab.
- [x] `RelayBoardController._reconnect`: a USB drop during switching reopens the port (auto-detect if
      the COM number moved), refuses a different board serial, re-drives the word (bench-tested on rev B).
      Firmware 2.0.0: IWDG 1 s + relay state restored after a watchdog reset.
- [ ] MagneticCharacterizer: repeat a Bode sweep interrupted by a relay-board drop.
- [ ] Pre-compliance run per EMC_COMPLIANCE.md section 5 (CM current < 2-3 uA on USB and BNC cables).
- [x] Decided 2026-10-11: 4 Kelvin terminals for rev C (6 terminals for CT / 3-winding parts planned for rev D:
      ~10 more relays, LQFP-64 MCU); no USB isolator for now (Kelvin front-end rejects the ground loop).

## Future (not rev C)
- [ ] DUT-jack ESD: 10 Mohm bleed per jack + 0.2 pF ESD-diode footprints (DNP) if pre-compliance needs them
      (EMC_COMPLIANCE.md section 4).
- [ ] 3-winding / centre-tapped magnetics. Until rev D's 6 terminals: measure as three pairs
      (W1-W2, W1-W3, W2-W3; unused winding shorted on itself) and merge with
      `scripts/combine_windings.py` (3x3 inductance matrix, couplings, both-others-shorted
      inductances; written and self-tested, not yet bench-validated). Capacitances to the third
      winding need the 6-terminal board.
