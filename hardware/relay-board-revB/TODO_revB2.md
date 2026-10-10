# Rev B2 TODO

Open items collected from the rev B review (external review by N. Rosano,
2026-10, plus a calibration/AC-resistance code audit). Rev B ships as is;
these are for the B2 respin and the software that drives it.

## Hardware (B2 board)

- [ ] **Kelvin (4-wire) path to the DUT for winding resistance / Rac.**
  Rev B is two-terminal: iso contact + arm + clamp (~100 mΩ + 13 nH per
  side) sit outside the OSL plane, and contact resistance moves on every
  actuation. Rac in the mΩ range is out of scope until separate
  force/sense relays reach the clamps (or a dedicated Kelvin clamp pair
  bypasses the matrix for Rac work). Starting point: each DUT clamp is
  already a 2-position block, but rev B ties both pads to one net
  (schematic label "force + spare/sense" is aspirational). Splitting pad 2
  onto its own sense net gives the clamp-side half of a Kelvin connection
  with no new connector.
- [ ] **Make the SHORT follow the DUT path.** Today SHORT closes the first
  HI terminal's LO relay, so it skips the LO terminal's bus and contact that
  the DUT path includes; that series impedance is counted as DUT. Options:
  short the two used terminal buses together at the iso COM side (one extra
  relay per terminal pair), or model the asymmetry in
  `simulate_calibration.py` and correct it in software. Update the
  simulation either way: it currently models SHORT as contact-only at the
  terminal plane, so its 0.7 % @ 10 MHz figure is optimistic.
- [ ] **Populate G6K-2F-RF-S** (the prototype used G6K-2F-Y to cut cost);
  compare measured floors of both on the same board first.

## Software (applies to rev B already)

### Bench findings 2026-10-10 (DUT_20261010, 1:1 252 uH MnZn transformer)

- [x] **Config 8 (link_BD_open) never joined B-D** (D alone on LINK). Fixed in
  the Python table and `relay_map.h`; `verify_board.py` now checks link intent.
- [x] **Floating terminals loaded the DUT with their column bus** (~8 pF,
  uncalibrated). Driver and firmware source isolate floating terminals in MEAS;
  OSL state unchanged, so stored calibrations stay valid.
- [x] Driver drives relays individually (`RELAY n,s` + readback) instead of
  `CONF:MEAS`, so it works with any flashed table.
- [x] **Firmware 1.2.0 flashed** 2026-10-11 over USB: `SYST:DFU` + dfu-util (WinUSB on 0483:DF11 via
  Zadig, once per PC). Firmware `CONF:MEAS`/`CAL:MODE` words verified equal to the Python table, 15 x 4.
- [x] **Lcum 'missing C33' explained (2026-10-11): the bridge measures a guarded transfer admittance.**
  Current is sensed only in the LO rail; a stray to board GND at relative potential v adds Cg*v*(v-1).
  The LINK net (rail + two columns) has ~14 pF to GND and sits at v = 1/2 in Lcum/Ldif -> -Cg/4.
  Same 14-15 pF on both DUTs. Configs 17-19 therefore measure LO-rail coupling, not ground capacitance.
  B2: keep the LINK rail/columns away from the GND plane or guard them.
- [x] **Reciprocity 7.5 % explained and corrected.** Board-only paths (DUT isolated) differ between A-B
  and C-D: other column -39/+37 nH, LINK loop +39/+75 nH. `characterize_fixture_paths()` measures them
  automatically (calibrations/fixture_paths.json) and subtracts them from the LINK-shorted states:
  reciprocity -> ~2 %, Lsc -5 %. B2: equalise column/LINK path lengths.
- [ ] Clamp arms (~13 nH each) and isolated-clamp coupling (~2 pF to GND) still need the operator
  shorting-bar / empty-clamp step.
- [ ] **Isolated-clamp capacitance** (~1 pF to HI and to GND per floating
  terminal) remains; it is what is left between the open-family links and
  direct C33 (+4 pF on a 1:1 part). Candidate for the clamp-plane residual step.
- [ ] Link-path series inductance measured: 26 nH (9 vs 11) and 59 nH (14 vs 15).
  This is the reciprocity error (6-8 % below 10 MHz) and biases ls by about that.

### Calibration

- [ ] **Clamp-plane residual compensation** *(bench: needs the board + Bode)*. Per signal path, optional
  manual step: shorting bar in the clamps, then empty clamps; store
  Zs_res/Yo_res next to the `.mcalx` and apply
  `Z = (Zm − Zs_res) / (1 − (Zm − Zs_res)·Yo_res)` after OSL. Removes the
  repeatable part of arm + iso contact + clamp (most of the ~13 nH / side);
  does not remove contact repeatability.
- [ ] **Contact repeatability measurement** *(bench)*. Shorting bar in the clamps,
  N ≥ 50 relay actuations per path, record R and L at 1 kHz / 100 kHz.
  Publish the spread as the board's real winding-resistance floor
  (replaces the 100–200 mΩ estimate in DESIGN_NOTES).
- [x] **Relay check before automatic OSL** (`_check_standards`): OPEN,
  SHORT and LOAD must each land in their own decade at 1 kHz, so a stuck
  K13/K14, a dead SHORT relay or a missing R5 aborts instead of producing a
  silently wrong `.mcalx`. After OSL the LOAD is re-actuated and verified,
  and the SHORT residual after re-actuation is logged.
- [x] **`try/finally` around the OSL loop**: the board always returns to
  `CAL:MODE MEAS`.
- [x] **Calibration provenance**: a `.json` sidecar per `.mcalx` records
  date, relay-board IDN, Bode IDN, drive level and the raw standard
  readings; the file is re-acquired when older than
  `MAX_CALIBRATION_AGE_HOURS` (24 h) or when any of those changed.
- [x] **Explicit `:SOUR:POW` before automatic OSL.**
- [ ] **Bench check of the new calibration flow:** confirm the Bode 100 doesn't need a separate
  calibration per drive level (`DRIVE_LOW_BAND_DBM` 0 dBm vs. the 13 dBm the
  OSL runs at): measure the LOAD at both after one OSL. Also tune the
  `STANDARD_CHECK_*` bounds against real raw readings.

### AC resistance (two-state recipe, `characterize_ac_resistance`)

Three or more gaps aren't practical, so the recipe runs with two states and
has no residual to self-check with. Each bias below therefore has to be
removed by construction, not detected afterwards.

- [ ] **Make two states the supported case**: drop the "use three or more"
  warning, and report the expected bias terms instead (below).
- [ ] **De-embed self-capacitance before the fit.** Parallel C inflates
  R by ~1/(1−ω²LC)², which depends on L and so on the state; it is not of
  the form K·L². With the synthetic DUT (2.4 mH, 60 pF) Rw comes out +3 % at
  100 kHz and +13 % at 200 kHz. Use the measured Cp from
  `characterize_capacitance` and remove it from the complex Z of each state
  (Z_s = 1/(1/Z − jωCp)) before calling `separate_winding_resistance`.
  Add the capacitance to the AC-resistance sweeps in `make_synthetic_dut.py`.
- [ ] **Hold flux density, not current, constant across states.** At low
  frequency the Bode 100 drives the DUT almost as a current source, so
  B ∝ L. In the Rayleigh region μ'' grows with B, so K isn't
  state-independent (K ∝ L gives Rw −20 % in simulation). Scale
  `source_power_dbm` per state to keep V/(ωN·Ae) constant.
- [ ] **Fixture resistance:** the constant part lands in Rw (apply the
  clamp-plane compensation); any change between states is amplified by
  L_B²/(L_A²−L_B²). Never unclamp between states.
- [ ] **Fringing bias, two states:**
  Rw_est = Rw_B + (Rw_B − Rw_A)·L_B²/(L_A² − L_B²). Put the varied gap
  away from the winding (outer legs of an E core) and state which Rw is
  reported.
- [ ] **Label `Rc`** as the core loss of the first state only.
- [ ] **Electrical core-loss separation, no gap change:** with a sense
  winding, configs 5/6 already on the board give
  Re(Z12) = Re(Z_aid − Z_opp)/4, the core loss without any copper loss at
  low frequency. At high frequency the aiding and opposing connections have
  different proximity loss, so validate it against a two-gap result before
  trusting it. Can be tried on cached CSVs.
