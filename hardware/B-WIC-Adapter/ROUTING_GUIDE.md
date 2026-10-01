# Routing Guide for Relay Board

Open `relay_board.kicad_pcb` in KiCad 9. Press **B** to fill all zones first.

## Layer Assignment

| Layer | Use |
|-------|-----|
| F.Cu | Signal zones (A,B,C,D) + Bode100 traces — DON'T route here except Bode100 |
| In1.Cu | +5V plane — use for coil trace crossings (via down, cross, via back up) |
| In2.Cu | GND plane — don't route non-GND signals |
| B.Cu | ALL component routing (GPIO, coil, power, misc) |

## Routing Order (recommended)

### 1. Bode100 traces on F.Cu (2 traces)
- K1 pad 2 via → straight down → J5 (right tab)
- K2 pad 2 via → straight down → J6 (left tab)
- These run through signal zone clearance — the zone filler creates gaps

### 2. GPIO traces on B.Cu (9 traces)
Route from U1 to U2/U3. All short runs.

| Net | From | To |
|-----|------|----|
| GPIO0 | U1.6 (98.3, 130.8) | U2.1 (78.0, 136.4) |
| GPIO1 | U1.7 (98.3, 130.0) | U2.2 (78.0, 135.2) |
| GPIO2 | U1.8 (98.3, 129.2) | U2.3 (78.0, 133.9) |
| GPIO3 | U1.9 (99.7, 127.8) | U2.4 (78.0, 132.6) |
| GPIO4 | U1.10 (100.5, 127.8) | U2.5 (78.0, 131.4) |
| GPIO5 | U1.11 (101.3, 127.8) | U2.6 (78.0, 130.1) |
| GPIO6 | U1.12 (102.1, 127.8) | U2.7 (78.0, 128.8) |
| GPIO7 | U1.13 (102.9, 127.8) | U3.1 (122.0, 136.4) |
| GPIO8 | U1.15 (104.5, 127.8) | U3.2 (122.0, 135.2) |

### 3. Coil traces (9 traces) — B.Cu + In1.Cu for crossings

| Net | From (relay coil-) | To (ULN output) |
|-----|---------------------|------------------|
| K1_COIL | K1.8 (86.0, 121.8) | U2.16 (83.0, 136.4) |
| K2_COIL | K2.8 (112.0, 121.8) | U2.15 (83.0, 135.2) |
| K3_COIL | K3.8 (131.0, 121.8) | U2.14 (83.0, 133.9) |
| K4_COIL | K4.8 (106.0, 79.8) | U2.13 (83.0, 132.6) |
| K5_COIL | K5.8 (106.0, 107.8) | U2.12 (83.0, 131.4) |
| K6_COIL | K6.8 (92.7, 93.5) | U2.11 (83.0, 130.1) |
| K7_COIL | K7.8 (116.7, 93.5) | U2.10 (83.0, 128.8) |
| K8_COIL | K8.8 (80.7, 93.5) | U3.16 (127.0, 136.4) |
| K9_COIL | K9.8 (104.7, 93.5) | U3.15 (127.0, 135.2) |

**Tip:** Route K1_COIL first (shortest). For long ones (K3, K4, K7-K9), drop to In1.Cu to cross other traces.

### 4. Short traces on B.Cu

| Net | From | To |
|-----|------|----|
| NRST | R3.2 (107.0, 124.0) | U1.4 (98.3, 132.4) |
| BOOT0 | R4.1 (110.0, 124.0) | U1.2 (98.3, 134.0) |
| R3_COM | K2.4 (105.0, 114.2) | K3.2 (124.0, 118.6) |

### 5. Power traces on B.Cu
- +3V3: U4.2 → U1.1 (VDD), U1.5 (VDDA), U1.17 (VDDIO2), C1-C3, R3.1, C5
- +5V: U4.3 → U2.9, U3.9, C4, C6 (also handled by In1.Cu plane via vias)

### 6. USB-C (J7) — assign nets in schematic first
Standard USB 2.0 device on USB-C:
- A6+B6 (D+) → USB_DP → U1.22 (PA12)
- A7+B7 (D-) → USB_DM → U1.21 (PA11)
- A1+B1+A12+B12 → GND
- A4+B4+A9+B9 → VBUS (+5V)
- A5 → CC1 → R1.1 (5.1k to GND)
- B5 → CC2 → R2.1 (5.1k to GND)
