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

### Automated gap change: saturable insert (investigation)

Idea: a thin MnZn ferrite insert in the gap with its own control winding.
Unbiased it is (almost) core; deeply saturated it is (almost) an air gap of
its thickness. Two states, switched electrically: the winding and the clamps
never move, which removes the re-clamping error entirely. Inspired by the
magnetic flux switch of Heidary et al., Sci. Rep. 14:8990 (2024), whose
control loop crosses the main core at a "junction".

- [ ] **Use only the two end states.** The fit assumes the core loss
  coefficient is the same in both states; the insert adds its own loss
  K_i·L². Unbiased: ~K·t/le (negligible). Deeply saturated: ~0. Partially
  saturated: incremental μ'' of a biased ferrite is large and bias-dependent,
  so intermediate bias points are NOT extra gaps. Bias error:
  Rw_est = Rw − (K_iA − K_iB)·L_A²·L_B²/(L_A² − L_B²).
- [ ] **Check deep saturation automatically:** step the control current
  and only accept state B once L is flat (< 0.5 %) over the last steps.
- [ ] **Control MMF budget.** For L_B/L_A ≤ 0.5 the insert's incremental
  μr must drop to ≲ μ_core·t/le (≈ 20 for μ 2000, t 1 mm, le 100 mm).
  Bias across the main flux (MFS geometry) needs H ≈ Bs/(μ0·μr) ≈ 16 kA/m
  for that, since the perpendicular incremental permeability is ~M/H;
  bias along it falls faster. Expect a few hundred ampere-turns: measure
  the heating, because copper changes 0.39 %/K and the control coil sits
  next to the DUT winding. Alternate A/B several times and check for drift.
- [ ] **No coupling to the DUT winding.** Close the bias flux locally in
  the insert (two holes, control winding threaded through them, symmetric
  halves) so the main winding links no DC bias and no first-order AC
  coupling; drive the control winding from a current source with high AC
  impedance (series choke) so it cannot act as a shorted turn.
- [ ] **Remanence:** after saturation the insert sits on a recoil line,
  not the initial curve. That's fine as long as state A is always reached
  the same way (e.g. after a fixed saturate-and-release); no demag needed.
- [ ] **Hardware:** rev B has no current output. Prototype with a SCPI
  bench supply in current mode; consider a switched current sink on B2 only
  if the method works.
- [ ] **Alternatives to compare against:** a motorized gap (linear stage
  moving one core half: no insert loss, no heating); and the electrical
  route with a sense winding, using configs 5/6 already on the board:
  Re(Z12) = Re(Z_aid − Z_opp)/4 gives the core loss without any copper loss
  at low frequency, but at high frequency the aiding and opposing
  connections have different proximity loss, so validate it against the
  two-gap result before trusting it.
