# Relay board rev C: EU EMC (CE) compliance plan

Compiled 2026-10-11. The IEC/EN 61326-1 tables are paywalled. The levels below come from test-lab summaries, cross-checked against the 2020/2021 change notes. Items marked **[verify]** must be checked against the purchased standard, or agreed with the lab, before the test plan is frozen.

## 1. Which rules apply

- **Directive:** EMC 2014/30/EU, conformity route Module A (internal production control, Annex II). No notified body is needed. RoHS 2011/65/EU goes on the same Declaration of Conformity. The Low Voltage Directive does not apply: the board is 5 V USB, below 75 V DC.
- **Harmonised standard:** **EN IEC 61326-1:2021 (+A11)**, laboratory and measurement equipment. This product-family standard takes precedence over EN 55032/55035. Emissions are measured to **CISPR 11 / EN 55011, Group 1**: the board does not generate RF on purpose to treat material.
- **Class:** design for **Class B**. A USB instrument plugged into any PC can end up in an office or at home. Class A would need a warning in the manual. Aim for a 6 dB margin.
- **Radiated emission limits (quasi-peak, at 10 m; about +10.5 dB at 3 m):**

  | Class | 30–230 MHz | 230–1000 MHz |
  |---|---|---|
  | B | 30 dBµV/m | 37 dBµV/m |
  | A | 40 dBµV/m | 47 dBµV/m |

- **Conducted emissions** (150 kHz–30 MHz) are measured on the **host PC's mains** through a LISN, with the board as a load on it. **[verify]** whether the lab applies DC-port limits to a USB cable under 3 m. It usually does not.
- **The Bode 100 stimulus:** 61326-1 / CISPR 11 have no explicit exclusion for an intentional measurement signal. The likely test configuration is board + Bode 100 + PC with the VNA sweeping, so the BNC cable shields count as radiators. **[verify]** Agree the operating modes with the lab: idle, relays switching, VNA sweeping.
- **Possible exemption:** Art. 2(2)(e), "custom built evaluation kits destined for professionals to be used solely at R&D facilities". This is narrow, and an instrument sold generally does not qualify. **[verify]** in the Commission EMC Guide before relying on it.

### Immunity (EN 61326-1)

| Test | Basic EM environment | Industrial environment | Criterion | Applies to this board? |
|---|---|---|---|---|
| ESD, IEC 61000-4-2 | ±4 kV contact / ±8 kV air | ±4 kV / ±8 kV **[verify]** | B | **Yes**: USB shell, BNC shells, banana jacks, GND strip, mounting holes |
| Radiated RF, IEC 61000-4-3 | 3 V/m 80 MHz–1 GHz; 3 V/m 1.4–2 GHz; 1 V/m 2–6 GHz **[verify split]** | 10 V/m 80 MHz–1 GHz; 3 V/m 1.4–6 GHz | A | **Yes** |
| EFT/burst, IEC 61000-4-4 | 0.5 kV, signal ports | 1 kV | B | No: only cables over 3 m (USB and BNC are shorter) |
| Surge, IEC 61000-4-5 | - | - | B | No: no AC port, no long or outdoor lines |
| Conducted RF, IEC 61000-4-6 | 3 V | 3 V **[verify]** | A | No: only cables over 3 m |
| Mains-frequency H field, IEC 61000-4-8 | 3 A/m | 30 A/m | A | **Yes, de facto**: an open transformer DUT is a magnetic pickup |

- **Criterion A:** works within specification during the test.
- **Criterion B:** temporary degradation that recovers by itself; no change of operating state, no loss of stored data.
- **Criterion C:** loss of function is allowed if it can be recovered.
- The 2021 edition lets the manufacturer **declare the allowed degradation** in the user documentation. Use this: state the impedance error allowed during the radiated RF test (3 V/m couples into an unshielded open DUT; that is physics, not a board defect), and that a measurement in progress is repeated after an ESD event.
- **A relay changing state, or USB re-enumerating, after ESD counts as a criterion B failure.** See section 3.

## 2. Design checklist, mapped to the board

| # | Failure mechanism | Rule | Rev C implementation | Status |
|---|---|---|---|---|
| 1 | Common-mode current on cables (the dominant radiated failure). About 5 µA on a resonant cable fails Class B 30 MHz–1 GHz (LearnEMC) | No RF voltage between cable shields; filter every conductor leaving the board | Unified GND, no AGND island. USB-C and all three BNCs on **one edge** (bottom): no shield-to-shield dipole across the board. CM choke L1 on D+/D-, 2 A ferrite FB1 on VBUS at the connector | Done |
| 2 | Shield termination (pigtail ≈1 nH/mm, 20–40 dB worse above 100 MHz) | 360° shell bonding, many vias, no isolated BNCs | BNC bodies and USB-C shell soldered to GND pours on all layers. Exposed GND bonding strip on the cable edge for an enclosure lid/gasket. USB shield 0R fitted (1M ‖ 4.7 nF option) | Done; enclosure bulkheads must bond to the panel |
| 3 | ESD into touchable metal | TVS at the connector, shortest via to GND, discharge current kept away from the MCU | USBLC6 D2 at J1, SMF5.0A D9 on VBUS. BNC/USB shells straight onto the planes at the board edge, MCU about 100 mm away. Plated mounting holes on GND | Done for USB/BNC; **banana jacks open, see section 4** |
| 4 | Relay coil di/dt | Local freewheel, small coil loop, current-limited and decoupled supply, staggered switching | TBD62003 clamp diodes (COM to +5V). +5V coil tree on In1 directly under the relays, 0.21 mm from the In2 plane. TPS2051C soft start + limit, 10 µF per driver. Coils energized 2 ms apart, ≤16 at a time | Done |
| 5 | Clock harmonics (48 MHz, USB FS 12 Mbit/s) | No clock outputs, slow edges, short USB pair over a solid plane | No crystal (HSI48 + CRS), MCO unused. Outputs at the low-speed reset default; unused pins analog. USB pair a few mm long, J1 → L1 → D2 → MCU, over In1 GND at 0.21 mm | Done |
| 6 | Plane edges, slots, cavity resonance | Stitching pitch λ/20 at the top frequency (7 mm at 1 GHz); no slot longer than 2× pitch | GND stitching every 4 mm along the board edge and around the measurement void. Return-path via within 1.5 mm of every signal via outside the void. Decoupler GND vias beside the pad at 90° to the pad axis | Done (checked post-route) |
| 7 | Long return-path detours | Every signal layer next to an unbroken reference | SIG/GND/GND/SIG (JLC04161H-7628): every signal layer 0.21 mm from a solid GND plane outside the void. Inside the void the planes are removed on purpose: measurement nodes sit at intermediate potential, and ground capacitance there is a DUT-dependent error | Done; trade-off documented |
| 8 | Fail-safe recovery (criterion B) | Watchdog; relay state survives a reset | IWDG 1 s (firmware 2.0.0). Configuration, calibration mode, Kelvin flag and coil word cross a watchdog reset in .noinit RAM and are re-applied, staggered, once the coil supply is back. Power-up and DFU still start with everything released | Done (firmware); host reconnect open, see section 4 |
| 9 | Magnetic pickup (61000-4-8) | Small loops in the measurement path | Kelvin sense pairs run alongside their force rails (1.5 mm), end-fed equal-length rails, link loops minimised (parasitic_report) | Done |

## 3. What criterion B means for firmware and host

- A hung core is caught by the IWDG; the board resets and comes back with the same relays closed. A USB reset during that time is unavoidable: the MCU's USB peripheral restarts and the host sees an unplug and replug.
- `RelayBoardController` must therefore treat a dropped serial port as recoverable. It should reopen the port, check `*IDN?` and `SYST:SER?`, read the relay word back (it is restored), and repeat the interrupted sweep. **Open item, see TODO.**
- The test-lab script: a SCPI loop that switches configs, reads the word back and logs `*IDN?` uptime. An ESD hit passes if the word never changes without being commanded and the loop resumes without user action.

## 4. Open points

1. **ESD on the banana jacks (decision needed).** The DUT jacks are floating measurement nodes. A gun discharge goes through the jack, across an open relay gap or through closed contacts to the rails, then out of the board via the BNCs into the Bode 100. A TVS on each jack would clamp it, but it would add about 0.3–1 pF of non-linear, temperature-dependent capacitance and leakage to every DUT node: exactly what Rosano's review point 4 objects to. Options:
   - (a) Declare the jacks as measurement terminals, handled with ESD precautions like the Bode 100's own test ports, and apply ESD only to the shells, strip and holes. **[verify]** with the lab that this reading of 61326-1 is accepted. Zero cost to accuracy.
   - (b) Add DNP footprints for a 0.2 pF bidirectional ESD diode per jack, fitted only if pre-compliance fails. Costs a re-route of the jack area.
   - (c) 10 MΩ bleed per jack to GND: stops static build-up from handled DUTs, does not clamp a gun pulse, and adds a constant 10 MΩ inside the fixture compensation.

   **Decided for rev C: (a)** -- jacks routed as they are; (b)/(c) kept for a future revision if pre-compliance needs them. Relay ratings to cite in the risk analysis (G6K-2F-Y datasheet **[verify]**): 1500 VAC coil–contact, 750 VAC across an open contact; the coil side is far from the discharge.
2. **BNC cables while sweeping:** use double-shielded cables (RG223 / RG400) no longer than needed, and measure common-mode current on them at +13 dBm (pre-compliance step 3).
3. **Enclosure:** a metal box with BNC bulkheads bonded to the panel, and the cable-edge GND strip gasketed to it, gives the most margin for both emissions and ESD. Without an enclosure the board edge is touchable; the 0.5 mm copper pull-back and the stitched edge are what protect it.

## 5. Pre-compliance plan (about €2–5k of equipment, or a rented lab day)

1. **Near-field scan** (H/E probes, e.g. Tekbox) of the board: idle, relays cycling through all configs, USB traffic. Look at the USB corner, the coil tree and the drivers.
2. **Common-mode current** with a clamp-on RF current probe on the USB and each BNC cable, 30–300 MHz. Target **< 2–3 µA**: half the 5 µA rule of thumb.
3. Repeat step 2 with the **Bode 100 sweeping at +13 dBm** into an open DUT and a shorted DUT.
4. **ESD** (MinZap-type gun, ramp 0.5 → 4 kV contact, 8 kV air) on the USB shell, BNC shells, GND strip, mounting holes and (option a) the jacks, while the SCPI loop of section 3 runs.
5. **Radiated immunity sanity check:** a hand-held transmitter or a TEM cell at 3 V/m while measuring the 1R and 10 pF check standards. Record the deviation; this sets the degradation to declare.
6. One day of **formal pre-scan at an accredited lab** before the full test.

Spectrum analyser with an EMI option (CISPR RBW, quasi-peak), e.g. Siglent SSA3021X. Optional: a Tekbox TBMA antenna and a 5 µH DC LISN pair.

## 6. CE technical file (keep 10 years)

- **EMC assessment:** each phenomenon in section 1, why it applies or not, and the standard applied.
- Schematics, PCB (Gerbers + stackup), BOM, firmware version.
- Test reports: pre-compliance and accredited.
- User manual: the Class B statement, the declared degradation under RF immunity, ESD handling for the DUT jacks, and the recommended cables.
- EU Declaration of Conformity (EMC + RoHS) and the CE mark on the board or enclosure.

## Sources

- LearnEMC QOTW 2021-06-28 (cable common-mode current)
- InCompliance Magazine p=29394 (CISPR 11 limits)
- C Electronics, "IEC 61326-1 general requirements"
- Element, "EN 61326-1 and EN 61326-2-x 2021"
- RF Essentials, "cable shield termination"
- ALLPCB, "via stitching for EMC"
- Interference Technology, "Your first EMC bench setup"
- Directive 2014/30/EU Art. 2 and Annex II
- IEC 61326-1:2020 preview (iteh.ai)
