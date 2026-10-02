# Rev D — future ideas

Long-horizon items, deliberately kept out of rev B2. Nothing here is
scheduled.

## Automated gap change: saturable insert

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
  bench supply in current mode; add a switched current sink to the
  board only if the method works.
- [ ] **Compare against a motorized gap:** a linear stage moving one core
  half (no insert loss, no heating, but mechanics).
