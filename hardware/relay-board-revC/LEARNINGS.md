# Rev C -- what we learned routing it ourselves

Hard-won lessons from getting the board routed without an autorouter (2026-10-11/12). Read
before changing the placement, the pin assignment or maze_route.py.

## Floorplan beats router cleverness

Most "routing failures" were placement or pin-assignment mistakes; rip-up only made them thrash.
Before touching the router, render the stuck area (`devtools/`) and ask what put copper there.

- **Pin assignment is free -- use it.** The firmware relay map is generated from the schematic,
  so any MCU pin can drive any relay. Assign in *lane order*: all driver inputs sit on one line
  (x = 181.5, facing east); north-row pins are the innermost lanes and take the nearest inputs,
  east-column pins the next, west-column pins the farthest (`generate_schematic.GPIO_PINS`).
  Min-sum matching (straight-line or geodesic) was *useless* here: every MCU pin is ~7 mm from
  the others, the drivers 20-80 mm away, so all orders score within 0.3 %.
- **Functions go on the side facing their destination.** I2C on the east pins crossed the whole
  relay corridor (moved to I2C2 PB10/PB11, west, facing the TMP117). LEDs east of the USB pins
  had to cross USB (moved to a column SW of the MCU with the coil-switch lines; RUN moved from
  PC13 on the north row to PA9 on the south row).
- **Static nets that must cross a lane fan go under it.** BOOT0 sits between relay pins and must
  run to R4/R17: it escapes inward (via under the package) and runs on B.Cu (`INWARD_FIRST`).
- **Lane groups that must cross each other get different layers.** West-column relay lanes run
  on B.Cu, north/east lanes on F.Cu (`LAYER_COST`).
- **Support parts sit over their own pin, not "near the MCU".** AN4325 "at the pin" means the
  pin it decouples: the VDDA stack over pin 9, NRST R/C over pin 7. One cap in the NE corner notch
  serves VDD48 + VBAT (adjacent +3V3 pins). Anything else in the band between the MCU and the
  drivers blocks lanes.
- **Test pads beside the circuit they probe.** A row of test pads at the board edge dragged
  FE_OUT/VMID/+5VA 70-100 mm across the board: an antenna on the measurement output and a wall.
- **Reserve access around small ICs routed late** (`ACCESS_POCKETS`): the coarse pass otherwise
  walls them in with nets the fine pass may not move.
- **Crossings that are unavoidable: pick the layer deliberately.** The VBUS feed has to cross the
  USB chain -- supply on B.Cu, the USB pair on F.Cu without vias (`LAYER_COST`).

## Datasheet traps

- **TDK ACM2012 CM choke: windings are 1-4 and 2-3**, pads facing each other across the body
  (TDK equivalent-circuit library), not 1-2 / 3-4 as the generic KiCad symbol suggests. The first
  netlist swapped D+/D- at the connector. Check every 4-pad CM choke's winding pairs.
- **STM32 ROM bootloader** drives PA9/PA10 (USART1), PA14/PA15 (USART2), PA11/12 (USB) in DFU:
  a "PROG" LED must not sit on those. Relay pins there are safe only because the coil supply
  switch is held off by its pull-down through reset.
- **F0 I2C2** is clocked from PCLK (48 MHz), I2C1 from HSI by default: the timing argument differs.
- **USB-C (HRO TYPE-C-31-M-12)**: VBUS pads sit second from each end of the 0.5 mm row, between
  GND and CC/SBU, with NPTH pegs 1.45 mm below. Escape VBUS straight down each pad's inner edge
  (0.2 mm, exactly 0.2 to the neighbour, 0.69 mm centre-to-peg) to vias, trunk on B.Cu between
  the shell legs.

## Router internals that bit us (maze_route.py)

- **Grid quantisation cannot resolve 0.5 mm pitch.** Fine-pitch pads get exact-geometry escape
  stubs first (fanned outward, else inward to a via under the package, else a dog-bone), then the
  grid router starts from the stub ends.
- **Pad side from the pad's long axis**, never from its position relative to the package centre
  (corner pins got sideways escapes into their neighbours).
- **Verify every emitted segment and via against exact geometry**; on refusal block that cell and
  re-search. Then the grid can allow diagonal moves freely.
- **Net ties:** the bridge copper belongs to *both* tied nets; modelling it as foreign copper made
  every net-tie pad unreachable (`Obstacles.add(..., net2=)`).
- **A plane on an inner layer blocks tracks, not vias** -- treating the GND planes as obstacles for
  vias made every driver-side net unroutable.
- **Rip-up and reroute:** rip along the path the stuck net *wants* (planned with other nets'
  copper passable at 12x cost -- PathFinder-style congestion), radius rings only as a fallback;
  priority classes (measurement > supplies/coils/USB > digital); never rip the generator's
  pre-routes (`relay_board_prerouted.json`) or the fan-out escapes (a ripped escape is never
  recreated and the pad stays sealed).
- **Weighted A\*** (heuristic x2) on the fine grid: the crosstalk cost (up to 7x) otherwise makes the
  search explode.
- **Footprints added by script need a library ID**, else freerouting writes `(component ` with no
  name and KiCad's SES import fails (relevant only if freerouting is ever used again).
- **Mask openings over copper must be pads with a net** (bonding strip), else every GND via under
  them is a solder_mask_bridge.

## Workflow

- A full run is generate (1 min) -> coarse maze (~25 min) -> fine maze (1-2 h) -> route_pcb
  --no-freerouting -> optimize_tracks -> parasitic_report. Keep a copy of the coarse output and
  re-run only the fine pass while iterating on it.
- When a net fails, first run it alone (`devtools/astar_test.py NET`). Routes alone = ordering or
  congestion; fails alone = geometry (render it).
- Stop a run as soon as the log shows the same nets ripping each other: that is a floorplan
  problem, and every minute after that is wasted.
