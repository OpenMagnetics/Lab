# Rev B design notes — parasitics, calibration, and why the board looks like this

## The one idea everything follows from

The OSL calibration plane sits at the **isolation-relay contact**. Everything
on the instrument side of that plane is measured during calibration *in the
same relay state as the measurement*, so its parasitics cancel. Everything on
the DUT side adds directly to the answer. Placement priority is therefore not
"keep everything short" but **"keep the uncalibrated side minimal and the
calibrated side stable"**:

| Priority | Nets | Calibrated? | Treatment |
|---|---|---|---|
| 1 | DUT arms (clamp → iso NC) | **No** | 9.6 mm dead-straight verticals, 22 mm apart, no copper underneath, nothing routed near them |
| 2 | Column buses (iso COM → 3 crossbar COMs) | Per config | Straight F.Cu verticals, identical length in every column |
| 3 | Rails (crossbar NO → bridge → BNCs) | Yes, constant | Straight B.Cu horizontals under the NO-pad rows, descending into the on-board B-WIC bridge |
| 4 | Coils, GPIO, power, USB | DC during a sweep | Freerouted afterwards, crossing signals at 90° on the other layer |

The matrix interconnect graph is K(3,5) — non-planar — so bus/rail crossings
are topologically unavoidable. Giving rails their own layer converts every
crossing into a layer separation instead of a routing puzzle.

## Fixture capacitance budget

Rev A poured the four DUT nets as zones over a solid +5 V plane: **230–340 pF
per node**, in parallel with the tens of pF the instrument exists to resolve,
and varying with relay state within a calibration group. Rev B:

| Element | Estimate | Calibrated? |
|---|---|---|
| DUT arm (9.6 mm, no plane) | ~0.4 pF, ~8 nH | **No — the fixture floor** |
| Iso relay closed contact | ~50 mΩ, ~5 nH | **No** |
| Column bus (44 mm) | ~1.8 pF, ~35 nH | Yes (per config) |
| Open crossbar contacts on a rail (3×) | ~0.75 pF | Yes (per config) |
| Rails + descents + tabs | ~5 pF, ~90 nH | Yes (constant) |
| Column bus + open contacts of a **floating** terminal | measured ~8 pF total (to HI and LO/GND) | **No** -- the OSL isolates every terminal, so a floating terminal's column is never in it. Fixed in the driver/firmware 1.2: floating terminals are isolated in MEAS |
| Isolated clamp of a floating terminal (arm + open iso contact) | measured ~1 pF to HI and ~0.7-1 pF to GND | **No** -- the remaining open-family excess (~4 pF on BC/AD links) |
| Far-winding short path in shorted states (rail + columns) | measured 26 nH (primary pair 9/11) and 59 nH (secondary pair 14/15) differences | **No** -- reciprocity off by 6-8 %; see TODO_revB2 'SHORT follows the DUT path' |

Uncalibrated total per side (terminals on a rail): **≈0.4 pF + 13 nH + 100 mΩ** — three orders of
magnitude below rev A's fixture capacitance, and small against every quantity
the recipes extract. The 100 mΩ is why winding-resistance results carry a
fixture-floor warning; the bridge topology is one-port and has nowhere to
Kelvin-sense, so this floor is honest rather than fixable.

## Integrated B-WIC bridge

Rev B connects to the Bode 100 with three BNCs (SOURCE / CH1 / CH2) and
clones the B-WIC's internal measurement bridge on-board (values from
Omicron's published bridge schematic): RV1 = 1 k and RV2 = 47 R divide the
source drive into CH1 as the voltage reading; RC1 = 2×4R7 in parallel
(2.35 R) shunts the return current, RC2 = 47 R feeds the shunt voltage into
CH2. All four are 0.1 % thin film: the Bode's IAD impedance method stays
valid with the stock B-WIC calibration workflow, but the OSL run through the
relay matrix calibrates the actual resistor values anyway. The bridge and
BNC shells sit on a dedicated AGND island tied to digital GND at a single
0 R link (R11), so relay-coil and USB return currents never share the
measurement return path.

## Calibration standards (Keysight impedance handbook, §4)

The handbook lists "a scanner, multiplexer or matrix switch is used" as a case
where open/short compensation is insufficient and OSL is required, and demands
the load be "measured in the same way as the DUT will be measured." Hence:

- **OPEN** — energize all four isolation relays. The matrix stays in the
  measurement state, so calibration sees the exact fixture it will correct.
- **SHORT** — additionally close the first HI terminal's LO relay: that
  terminal's bus bridges the rails through two matrix contacts. No dedicated
  relay, and the short sits at the same contact plane as the DUT.
- **LOAD** — K13/K14 connect a 100 Ω 0.1 % thin-film resistor through **a
  fifth, DUT-less matrix column**: one contact plus a full-length column bus
  per side, structurally identical to a real measurement.

ngspice validation (`simulate_calibration.py`), full parasitic model, known
2.4 mH ‖ 60 pF ‖ 46 kΩ DUT:

| | raw fixture error | rev B OSL | load-at-rails OSL (rejected) |
|---|---|---|---|
| 1 kHz – 1 MHz | ~200 % | **0.05–0.34 %** | 0.15–0.55 % |
| 10 MHz | 181 % | **0.7 %** | 5.4 % |
| 40 MHz | 74 % | **9.0 %** | 22.8 % |

The 40 MHz residual is the uncalibrated arm/contact inductance — and is why
the capacitance recipes extract from resonance *frequencies* (stable against
series L) rather than absolute |Z| at the top of the band.

Per-config calibration removes rev A's central constraint: `CALIBRATION_GROUPS`
no longer exists. Every configuration owns a calibration file named for its
signal path, acquired automatically with the DUT still clamped.

## Component choices

- **G6K-2F-RF-S**: open-contact isolation 20 dB min at 1 GHz (~0.25 pF)
  against the plain G6K-2F-Y's unspecified coupling; the matrix hangs three
  open contacts on each rail, so this is a measurement spec, not RF vanity.
  Footprint drawn from the datasheet mounting pattern (contacts 0.8×1.8 mm on
  7.0 mm rows; coil 1.6×2.1 mm on 7.5 mm) — the KiCad -Y footprint differs.
- **TBD62003APG** over ULN2003A: DMOS, ~0.15 V drop at 23 mA against 0.8–1.0 V
  bipolar. With 5 V coils (must-operate 3.75 V) and worst-case 4.40 V VBUS,
  the bipolar part leaves 3.4–3.6 V — no margin; the DMOS part leaves ~4.25 V.
- **STM32F072CBT6**, crystal-less: dedicated BOOT0 pin (the F042's shared
  PB8/BOOT0 enabled rev A's strapping bug), 128 K flash, and HSI48 + CRS
  auto-trim from USB SOF -- proven F0 silicon, and deleting the crystal
  freed the congested MCU corner (SCPI over CDC needs no clock accuracy
  beyond USB's own requirement, which CRS satisfies by design).
- **Per-terminal isolation relays**: a shared DPDT would put two DUT terminals
  into one package with ~0.1 pF pole-to-pole coupling — directly, and
  uncalibratably, across the quantity under measurement.
- **AP2112K-3.3** over AMS1117: 250 mV vs ~1.1 V dropout from a 4.4 V floor.

## Deliberate limitations

- Winding-resistance floor ≈ 100–200 mΩ (iso contact + clamp, uncalibrated).
- LOAD standard shares the rails' inductance up to its column; ppm-grade work
  above 10 MHz would need a coaxial fixture, not a switched one.
- One relay set = one signal path at a time; no four-terminal-pair guarding.
- Board is single-sided assembly on purpose; cheaper, and no Flip() bugs.


## As-ordered state (2026-08-30)

The shipped board (`relay_board_fab.zip`) is `relay_board.kicad_pcb` as of
this date: 150 x 90 mm, 4-layer, DRC-clean at warning severity, all
parasitic budgets met (arms 8.5-13.3 mm, no vias, no co-running copper).
Three GPIO links (GPIO12/14/16) were finished interactively in the KiCad
GUI on top of the generated locked pre-routes, so **`generate_pcb.py` no
longer reproduces this exact board** -- treat the saved `.kicad_pcb` as
the fabrication master for rev B. T-bus lengths are asymmetric as-built
(53-92 mm); this is inside the per-config OSL-calibrated region and
cancels in calibration (verified by simulate_calibration.py).

## Prototype BOM substitutions (2026-08-31, JLC assembly order)

To cut the JLC assembly quote (relays alone were $839 of an $866 total),
the JLC BOM (`export_jlcpcb.py`) substitutes two parts for the prototype:

- **K1-K18: G6K-2F-Y DC5 (C1560486, ~$0.6) instead of G6K-2F-RF-S DC5
  (C2750987, ~$9.3).** Land-pattern compatible per Omron datasheets:
  identical terminal rows (3.2/5.4/7.6 mm) and contact pads (1.8x0.8 on
  a 7.0 mm span); the -Y coil terminal is narrower than the RF-S pad it
  lands on. Same 5 V / 237 ohm / 100 mW coil, same DPDT arrangement, so
  firmware, driver, and verify_board are unaffected. The -Y loses the
  50 ohm impedance control and GHz-band isolation of the RF part --
  irrelevant in the Bode 100's <=50 MHz band, and systematic leakage is
  removed by per-path OSL calibration anyway. Expect slightly higher HF
  contact parasitics; if measured floors above ~10 MHz disappoint,
  populate the RF-S on the production build (drop-in).
- **R14,R15 (bridge RC1 shunt halves): 1% thick film 0805W8F470KT5E
  (C17675) instead of TNPW08054R70BEEA (no JLC stock).** R5, the 100R
  0.1% OSL LOAD standard, stays genuine Vishay -- it is the accuracy
  anchor; bridge-resistor error is absorbed by calibration against it.

The reference design (schematic, generate_bom.py BOM) keeps the RF-S
relays and full 0.1% bridge set.

Stock-driven prototype picks (2026-08-31, second pass): relays as
G6K-2F-Y-TR DC5 (C47190, tape packing, deep LCSC stock -- C1560486 had
none); R13/R16 as 1% thick film C17714 (only 1 pc of the 0.1% Vishay
left; same calibration argument as R14/R15); D1 power LED as red
KT-0603R (C2286) -- the green Everlight had 1 pc. R5 and R12 remain
genuine 0.1% Vishay.

## USB-C orientation fix (2026-08-31)

User caught J1 mounted rotated 180 deg -- mating face pointing at the
board interior, solder pins at the tab edge; a plug could never mate.
DRC cannot detect this class of error. Fixed on the fabrication master
by fix_usb_orientation.py + fix_usb_cleanup.py (backups
relay_board_backup_usbfix*.kicad_pcb): J1 now rot 0 at (164, 127.75),
nose flush with the tab edge y=131.4. The connector-side copper was
ripped and re-routed to the mirrored pad positions: CC1 west descent
at x=148.6 + F lane y=126.8; CC2 via B.Cu lane y=127.4 (descent
x=149.3); D+/D- keep their D2 lanes, new drops at x=163.75 (F) and
x=164.25 (via at y=121.55), duplicate-pad ties are southside via pairs
at y=124.95 / 125.75; VBUS B.Cu lane y=123.0 rising into both wide
pads; GND row pads on north stubs to plane vias at y=122.3. J1 silk
nose trimmed inside the tab. DRC 0/0/0 at severity-all; verify_board
ALL PASSED; fab zip and JLC CPL re-exported (J1 position/rotation
changed -- any earlier downloaded CPL is stale).

## Component pinout audit and fixes (2026-08-31)

Prompted by the user after the USB-C orientation catch, a full audit of
every polarized/keyed part against datasheet figures found two more
netlist-level bugs (DRC-invisible, and verify_board's old D2 row encoded
the same wrong assumption it was meant to check):

- **D2 USBLC6-2SC6 miswired.** It is a flow-through ESD array: pins 1&6
  are internally one node (I/O1), pins 3&4 another (I/O2). The old
  wiring (1=DP_CON, 6=DM_CON, 3=DP, 4=DM) shorted D+ to D- on both
  sides -- USB would have been completely dead. Now 1=USB_DP_CON,
  6=USB_DP, 3=USB_DM_CON, 4=USB_DM; D+ enters pad 1 and exits pad 6 to
  the MCU meander, D- enters pad 3 and runs on B.Cu (y=126.45, below
  the J1 shield pad) to the southside tie vias at J1.
- **D1 LED reverse-biased.** LED_A (PC13 drive + R10 pull-up) sat on
  pad 1 = cathode. Swapped: pad 2 (anode) = LED_A, pad 1 = GND.

Audit results for everything else: all 18 relay coils correct (G6K coil
"+" is pin 1 per the datasheet figure = our +5V side; pin numbering of
the RF-S and -Y variants confirmed identical, so the prototype -Y
substitution stands); U1 power/NRST/BOOT0/USB pins correct; TBD62003
IN/OUT mirroring, GND(8), COM(9)=+5V correct; AP2112K pinout correct;
BNCs point off the bottom edge; DUT clamps face-to-face; SWD header
orientation-free. generate_schematic.py, relay_board.xml and
verify_board.py (now with explicit flow-through + LED polarity rows)
all updated; DRC 0/0/0; fab zip re-exported.
