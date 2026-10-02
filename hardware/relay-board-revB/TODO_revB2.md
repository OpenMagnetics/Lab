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
  bypasses the matrix for Rac work).
- [ ] **Fifth (and sixth) DUT terminal.** Four terminals can't hold a
  center-tapped winding plus a second winding. Evaluate a 3×6 matrix.
- [ ] **SMD/planar DUT adapter** that plugs into the clamp bay with a
  defined, characterised residual.
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

### Calibration

- [ ] **Clamp-plane residual compensation.** Per signal path, optional
  manual step: shorting bar in the clamps, then empty clamps; store
  Zs_res/Yo_res next to the `.mcalx` and apply
  `Z = (Zm − Zs_res) / (1 − (Zm − Zs_res)·Yo_res)` after OSL. Removes the
  repeatable part of arm + iso contact + clamp (most of the ~13 nH / side);
  does not remove contact repeatability.
- [ ] **Contact repeatability measurement.** Shorting bar in the clamps,
  N ≥ 50 relay actuations per path, record R and L at 1 kHz / 100 kHz.
  Publish the spread as the board's real winding-resistance floor
  (replaces the 100–200 mΩ estimate in DESIGN_NOTES).
- [ ] **Verify the LOAD after automatic OSL** in
  `MagneticCharacterizer._ensure_calibrated` (the manual path already calls
  `_verify_load_standard`; the automatic one doesn't, so a stuck K13/K14 or
  bad R5 produces a silently wrong `.mcalx` that is then reused forever).
- [ ] **`try/finally` around the OSL loop** so a failure returns the board
  to `CAL:MODE MEAS` instead of leaving the DUT switched out.
- [ ] **Calibration expiry / provenance.** Store date, temperature (if
  available), firmware IDN and board serial with each `.mcalx`; re-run
  when stale or when any of them changes.
- [ ] **Set `:SOUR:POW` before automatic OSL** and calibrate at the drive
  level(s) actually used (`DRIVE_LOW_BAND_DBM` / `DRIVE_HIGH_BAND_DBM`), or
  confirm on the instrument that receiver ranging doesn't change between
  them.

### AC resistance (multi-gap recipe, `characterize_ac_resistance`)

- [ ] **De-embed self-capacitance before the fit.** Parallel C inflates
  R by ~1/(1−ω²LC)², which depends on L and so on the gap; it is not of the
  form K·L². With the synthetic DUT (2.4 mH, 60 pF) Rw comes out +3 % at
  100 kHz and +13 % at 200 kHz. Use the measured Cp from
  `characterize_capacitance` and remove it from the complex Z of each gap
  (Z_s = 1/(1/Z − jωCp)) before calling `separate_winding_resistance`.
  Then add the capacitance to the AC-resistance sweeps in
  `make_synthetic_dut.py` so the regression covers it.
- [ ] **Hold flux density, not current, constant across gaps.** At low
  frequency the Bode 100 drives the DUT almost as a current source, so
  B ∝ L: the small gap runs at ~4× the B of the 4× smaller-L gap. In the
  Rayleigh region μ'' grows with B, so K isn't gap-independent (+K ∝ L gives
  Rw −20 % in simulation). Scale `source_power_dbm` per gap to keep
  V/(ωN·Ae) constant, or run `verify_linearity` on each gap and refuse
  to fit if R changes with drive.
- [ ] **Tighten the residual check.** The residual is normalised to the
  total R, so it's diluted by the core term: a +10/+20 % fringing-driven Rw
  change (Rw biased +18 %) and a 20 mΩ re-clamp shift (biased +67 %) both
  give ~2.2 % residual and pass the 3 % threshold. Normalise to the fitted
  Rw and compare against the sweep-to-sweep noise from `average_cycles`.
- [ ] **Don't unclamp between gaps,** and say so in the prompt. The fixture
  resistance lands entirely in the Rw intercept (constant part → apply the
  clamp-plane compensation above) and any change between gaps is amplified
  by the noise gain. Read back the relay state and do a short-free check
  (R at the lowest frequency vs. the previous gap's DC-ish value) before
  each sweep.
- [ ] **Define what "Rw" means for a gapped part.** Fringing-flux
  proximity loss is real winding loss at the design gap; the multi-gap
  intercept removes it (and over-shoots: with two gaps the estimate is
  Rw₂ + (Rw₂ − Rw₁)·L₂²/(L₁² − L₂²)). Report Rw(gap-free) and, separately,
  R − K·L² at the design gap, and recommend including the design gap as one
  of the measurements.
- [ ] **Label `Rc`** as the core loss of the first gap only (it is
  `resistances[0] − Rw`), or report it per gap.
