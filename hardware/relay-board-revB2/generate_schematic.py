"""
Generate the OpenMagnetics relay board rev B schematic.

    python generate_schematic.py

Produces relay_board.kicad_sch, runs ERC via kicad-cli, and exports the
netlist.  Connectivity is by net label on a wire stub at each pin; genuinely
unused pins get a no-connect flag, which is what keeps ERC clean.

ARCHITECTURE (see scripts/RelayBoardController.py for the software side)

  Four DUT terminals A B C D reach three rails through a full 4x3 crossbar:

      K1..K3    A -> HI / LO / LINK
      K4..K6    B -> HI / LO / LINK
      K7..K9    C -> HI / LO / LINK
      K10..K12  D -> HI / LO / LINK

  Calibration follows the Keysight impedance-handbook rule that a matrix
  fixture needs OPEN/SHORT/LOAD compensation, with the load "measured in the
  same way as the DUT will be measured":

    K13/K14 + R5   a fifth, DUT-less matrix column: 100R 0.1% between two
                   full-length column buses reached through the same relay
                   contact structure as a real terminal.  LOAD standard.
    K15..K18       per-terminal isolation: COM = terminal node, NC = clamp,
                   NO = nothing.  Energized, the DUT is out of circuit while
                   still clamped.  All four energized = OPEN standard.
    SHORT          no dedicated relay: with the DUT isolated, closing one
                   terminal's HI *and* LO crossbar relays bridges the rails
                   through that terminal's bus -- a short at the same plane
                   as the other two standards.

  All three standards therefore sit at the matrix-contact plane, and every
  configuration can be calibrated in its own exact relay state.

RELAY PINOUT -- the defect this board exists to fix.  KiCad's Relay:G6K-2
symbol places the moving armature on pins 3 and 6, confirmed by the footprint
geometry (pin 3 sits physically between 2 and 4):

      coil    1, 8
      pole 1  COM = 3    NC = 2    NO = 4
      pole 2  COM = 6    NC = 7    NO = 5

Rev A used 2=COM/3=NC/4=NO, which left six relays with floating armatures.
"""

import os
import subprocess
import sys

import kicad_sch_api as ksa

KICAD_CLI = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMATIC = os.path.join(HERE, "relay_board.kicad_sch")

# --------------------------------------------------------------------------
# G6K-2 pin names -- single source of truth
# --------------------------------------------------------------------------
COIL_P, COIL_N = "1", "8"
P1_COM, P1_NC, P1_NO = "3", "2", "4"
P2_COM, P2_NC, P2_NO = "6", "7", "5"

RELAY_LIB = "Relay:G6K-2"
RELAY_FP = "OpenMagnetics:Relay_Omron_G6K-2F-RF-S"
RELAY_PART = "G6K-2F-RF-S DC5"

# STM32F072CBTx LQFP-48, parsed from the KiCad symbol.
# BOOT0 is a DEDICATED pin (44) on this part -- on the F042 it shares PB8,
# which is how rev A ended up strapping PF0 instead.
MCU = {
    "VBAT": "1", "PC13": "2", "PC14": "3", "PC15": "4", "PF0": "5", "PF1": "6",
    "NRST": "7", "VSSA": "8", "VDDA": "9",
    "PA0": "10", "PA1": "11", "PA2": "12", "PA3": "13", "PA4": "14", "PA5": "15",
    "PA6": "16", "PA7": "17", "PB0": "18", "PB1": "19", "PB2": "20",
    "PB10": "21", "PB11": "22", "PB12": "25", "PB13": "26", "PB14": "27", "PB15": "28",
    "PA8": "29", "PA9": "30", "PA10": "31", "PA11": "32", "PA12": "33", "PA13": "34",
    "VDDIO2": "36", "PA14": "37", "PA15": "38",
    "PB3": "39", "PB4": "40", "PB5": "41", "PB6": "42", "PB7": "43",
    "BOOT0": "44", "PB8": "45", "PB9": "46", "VSS": "47", "VDD": "48",
}

# ULN2003 / TBD62003APG: inputs 1..7, outputs 16..10, GND 8, COM 9.
ULN_IN = {n: str(n) for n in range(1, 8)}
ULN_OUT = {n: str(17 - n) for n in range(1, 8)}

#: Relay index (1..18) -> GPIO net, in the same order as MATRIX_BITS in
#: scripts/RelayBoardController.py, where relay K(n) drives bit n-1.
GPIO_PINS = ["PA0", "PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7",
             "PB0", "PB1", "PB2", "PB10", "PB11", "PB12", "PB13", "PB14",
             "PB15", "PA8"]

#: (terminal, rail) for the twelve crossbar relays, K1..K12.
MATRIX = [(t, r) for t in ("A", "B", "C", "D") for r in ("HI", "LO", "LINK")]


class Builder:
    def __init__(self):
        self.sch = ksa.create_schematic("relay_board")
        self.sch.set_paper_size("A2")
        self.sch.set_title_block(
            title="OpenMagnetics Automated Relay Board rev B",
            date="2026-08-29",
            company="OpenMagnetics Lab",
            comments={
                1: "18x G6K-2F-RF-S DPDT | 4x3 crossbar + isolation + on-board OSL standards",
                2: "Pin map: coil 1/8, pole1 COM=3 NC=2 NO=4, pole2 COM=6 NC=7 NO=5",
            },
        )
        self.centres = {}
        self.net_uses = {}

    # ---------------------------------------------------------------- parts

    def add(self, lib_id, reference, value, x, y, footprint=None, **properties):
        self.sch.components.add(lib_id, reference, value, position=(x, y),
                                footprint=footprint, **properties)
        self.centres[reference] = (x, y)
        return reference

    def _pin_positions(self, reference):
        if reference not in getattr(self, "_pin_cache", {}):
            self._pin_cache = getattr(self, "_pin_cache", {})
            self._pin_cache[reference] = [
                (number, position) for number, position
                in self.sch.list_component_pins(reference)]
        return self._pin_cache[reference]

    def _stub_end(self, reference, pin, length=3.81):
        point = self.sch.get_component_pin_position(reference, pin)
        if point is None:
            raise KeyError(f"{reference} has no pin {pin}")
        cx, cy = self.centres[reference]
        others = [q for number, q in self._pin_positions(reference)
                  if abs(q.x - point.x) > 0.01 or abs(q.y - point.y) > 0.01]
        shares_column = any(abs(q.x - point.x) < 0.01 for q in others)
        shares_row = any(abs(q.y - point.y) < 0.01 for q in others)

        # A stub along a line that other pins sit on can overlap their stubs
        # (collinear wires merge nets in KiCad), so escape perpendicular to
        # whichever line this pin shares with its neighbours.
        if shares_column and not shares_row:
            direction = "horizontal"
        elif shares_row and not shares_column:
            direction = "vertical"
        elif shares_column and shares_row:
            direction = "horizontal"      # grid: side escape is always safe here
        else:
            direction = "vertical" if abs(point.y - cy) >= abs(point.x - cx) else "horizontal"

        if direction == "vertical":
            end_y = point.y - length if point.y < cy else point.y + length
            return (point.x, point.y), (point.x, end_y), 90
        end_x = point.x - length if point.x < cx else point.x + length
        return (point.x, point.y), (end_x, point.y), 0

    def net(self, reference, pin, net_name, length=3.81):
        """Wire stub from a pin, with a net label at its end."""
        start, end, rotation = self._stub_end(reference, pin, length)
        self.sch.add_wire(start=start, end=end)
        self.sch.add_label(net_name, position=end, rotation=rotation)
        self.net_uses.setdefault(net_name, []).append(f"{reference}.{pin}")

    def unused(self, reference, *pins):
        """No-connect flag. A label on a one-pin net is what ERC calls dangling."""
        for pin in pins:
            point = self.sch.get_component_pin_position(reference, pin)
            self.sch.no_connects.add((point.x, point.y))

    def power_flag(self, net_name, x, y):
        self._flag_count = getattr(self, "_flag_count", 0) + 1
        reference = f"#FLG{self._flag_count:03d}"
        self.add("power:PWR_FLAG", reference, "PWR_FLAG", x, y)
        self.net(reference, "1", net_name)

    # ------------------------------------------------------------- sections

    def build_matrix(self):
        """K1..K12: terminal -> rail, one independently controlled SPST each.

        Only pole 1 carries signal.  Pole 2 is deliberately left open rather
        than paralleled: paralleling would halve contact resistance but double
        the OPEN-contact capacitance, and with twelve relays permanently across
        the measurement that is the wrong trade for a fixture whose job is
        resolving tens of picofarads.
        """
        for index, (terminal, rail) in enumerate(MATRIX, start=1):
            reference = f"K{index}"
            column, row = divmod(index - 1, 3)
            self.add(RELAY_LIB, reference, RELAY_PART,
                     50 + column * 52, 45 + row * 46, RELAY_FP,
                     Description=f"{terminal} -> {rail}")
            self.net(reference, COIL_P, "+5V")
            self.net(reference, COIL_N, f"{reference}_COIL")
            self.net(reference, P1_COM, f"T{terminal}")
            self.net(reference, P1_NO, f"RAIL_{rail}")
            self.unused(reference, P1_NC, P2_COM, P2_NC, P2_NO)

    def build_load_column(self):
        """K13/K14 + R5: the LOAD standard as a fifth matrix column.

        Keysight: "use a load of same size and measure it in the same way as
        the DUT will be measured."  The 100R is reached from the rails through
        one relay contact and a full-length column bus on each side -- the
        same residual structure as a real measurement, so the OSL solution
        transfers to the DUT plane instead of to the rails.
        """
        for reference, cal_net, rail, y in (("K13", "CAL_E", "RAIL_HI", 45),
                                            ("K14", "CAL_F", "RAIL_LO", 91)):
            self.add(RELAY_LIB, reference, RELAY_PART, 240, y, RELAY_FP,
                     Description=f"LOAD standard column: {cal_net} -> {rail}")
            self.net(reference, COIL_P, "+5V")
            self.net(reference, COIL_N, f"{reference}_COIL")
            self.net(reference, P1_COM, cal_net)
            self.net(reference, P1_NO, rail)
            self.unused(reference, P1_NC, P2_COM, P2_NC, P2_NO)

        self.add("Device:R", "R5", "100R 0.1%", 268, 68,
                 "Resistor_SMD:R_0805_2012Metric",
                 Description="LOAD standard: 100R 0.1% 25ppm thin film "
                             "(e.g. Vishay TNPW0805100RBEEA)")
        self.net("R5", "1", "CAL_E")
        self.net("R5", "2", "CAL_F")

    def build_isolation(self):
        """K15..K18: one isolation relay per terminal.

        Per-terminal (rather than a shared DPDT) so no two DUT terminals ever
        meet inside one relay package -- the G6K pole-to-pole coupling
        (~0.1 pF, 30 dB at 1 GHz) would otherwise sit directly and
        uncalibratably across the quantity under measurement.  The NO throw is
        deliberately unconnected: energized, the terminal node ends at an open
        contact, which IS the open standard.
        """
        for offset, terminal in enumerate("ABCD"):
            reference = f"K{15 + offset}"
            self.add(RELAY_LIB, reference, RELAY_PART, 50 + offset * 52, 200,
                     RELAY_FP,
                     Description=f"isolate {terminal}: COM=T{terminal} NC=clamp NO=open(std)")
            self.net(reference, COIL_P, "+5V")
            self.net(reference, COIL_N, f"{reference}_COIL")
            self.net(reference, P1_COM, f"T{terminal}")
            self.net(reference, P1_NC, f"DUT_{terminal}")
            self.unused(reference, P1_NO, P2_COM, P2_NC, P2_NO)

    def build_connectors(self):
        # DUT clamps. Spring-cage terminals so a magnetic can be swapped in
        # seconds -- and, just as importantly, so the four big copper pours of
        # rev A disappear along with the ~300 pF each added to the measurement.
        for reference, terminal, x in (("J2", "A", 330), ("J3", "B", 360),
                                       ("J4", "C", 390), ("J5", "D", 420)):
            self.add("Connector:Conn_01x02_Pin", reference, f"DUT {terminal}", x, 45,
                     "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MPT-0,5-2-2.54_1x02_P2.54mm_Horizontal",
                     Description=f"DUT terminal {terminal} (force + spare/sense)")
            self.net(reference, "1", f"DUT_{terminal}")
            self.net(reference, "2", f"DUT_{terminal}")

        # Bode 100 interface: the B-WIC's impedance bridge, integrated.
        # (Omicron application note "Impedance Measurement Bridge", 2015-09-16:
        # CH1 senses DUT voltage through RV1/RV2, CH2 senses DUT current
        # through the RC1 shunt via RC2. Cloning the B-WIC bridge keeps the
        # instrument's IAD impedance method valid with no adapter attached.)
        for reference, name, x, net_name in (("J6", "SOURCE", 330, "RAIL_HI"),
                                             ("J7", "CH1", 360, "CH1_NODE"),
                                             ("J9", "CH2", 390, "CH2_NODE")):
            self.add("Connector:Conn_Coaxial", reference, f"BNC {name}", x, 90,
                     "Connector_Coaxial:BNC_Amphenol_B6252HB-NPP3G-50_Horizontal",
                     Description=f"BNC to Bode 100 {name}")
            self.net(reference, "1", net_name)
            self.net(reference, "2", "AGND")

        bridge = [
            ("R12", "1k 0.1%", "RAIL_HI", "CH1_NODE",
             "bridge RV1: DUT voltage divider, top"),
            ("R13", "47R 0.1%", "CH1_NODE", "AGND",
             "bridge RV2: DUT voltage divider, bottom"),
            ("R14", "4R7", "RAIL_LO", "AGND",
             "bridge RC1 a: current shunt (2x4R7 parallel = 2.35R)"),
            ("R15", "4R7", "RAIL_LO", "AGND",
             "bridge RC1 b: current shunt pair"),
            ("R16", "47R 0.1%", "RAIL_LO", "CH2_NODE",
             "bridge RC2: current sense series"),
            ("R11", "0R", "AGND", "GND",
             "single-point analog/digital ground tie"),
        ]
        for index, (reference, value, net_a, net_b, description) in enumerate(bridge):
            self.add("Device:R", reference, value, 330 + index * 18, 120,
                     "Resistor_SMD:R_0805_2012Metric", Description=description)
            self.net(reference, "1", net_a)
            self.net(reference, "2", net_b)

        # SWD. Rev A had SWDIO/SWCLK dangling and no USB, so a blank MCU could
        # never be programmed at all.
        self.add("Connector:Conn_01x05_Pin", "J8", "SWD", 460, 90,
                 "Connector_PinHeader_1.27mm:PinHeader_1x05_P1.27mm_Vertical",
                 Description="SWD: 3V3, SWDIO, SWCLK, NRST, GND")
        for pin, net_name in (("1", "+3V3"), ("2", "SWDIO"), ("3", "SWCLK"),
                              ("4", "NRST"), ("5", "GND")):
            self.net("J8", pin, net_name)

    def build_usb(self):
        self.add("Connector:USB_C_Receptacle_USB2.0_16P", "J1", "USB-C", 330, 150,
                 "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                 Description="USB-C receptacle, USB 2.0 device")
        usb = {
            "A1": "GND", "A12": "GND", "B1": "GND", "B12": "GND",
            "A4": "VBUS", "A9": "VBUS", "B4": "VBUS", "B9": "VBUS",
            "A5": "CC1", "B5": "CC2",
            "A6": "USB_DP_CON", "B6": "USB_DP_CON",
            "A7": "USB_DM_CON", "B7": "USB_DM_CON",
            "S1": "GND",
        }
        for pin, net_name in usb.items():
            try:
                self.net("J1", pin, net_name)
            except KeyError:
                pass
        for pin in ("A8", "B8"):        # SBU1/SBU2, unused in USB 2.0
            try:
                self.unused("J1", pin)
            except Exception:
                pass

        for reference, value, net_name, x in (("R1", "5.1k", "CC1", 300), ("R2", "5.1k", "CC2", 315)):
            self.add("Device:R", reference, value, x, 185, "Resistor_SMD:R_0402_1005Metric",
                     Description="USB-C CC pull-down, sink advertisement")
            self.net(reference, "1", net_name)
            self.net(reference, "2", "GND")

        # ESD on the data pair. Rev A had none anywhere, on a bench instrument
        # that gets re-plugged dozens of times a day.
        self.add("Power_Protection:USBLC6-2P6", "D2", "USBLC6-2SC6", 390, 185,
                 "Package_TO_SOT_SMD:SOT-23-6",
                 Description="USB ESD protection, IEC 61000-4-2 level 4")
        self.net("D2", "1", "USB_DM_CON")
        self.net("D2", "2", "GND")
        self.net("D2", "3", "USB_DP_CON")
        self.net("D2", "4", "USB_DP")
        self.net("D2", "5", "VBUS")
        self.net("D2", "6", "USB_DM")

    def build_power(self):
        # AP2112K: 600 mA LDO with 250 mV dropout, versus the AMS1117's ~1.1 V.
        # Worst-case coil load is 8 relays at 23 mA plus the MCU, and the LDO
        # must hold 3.3 V from a 4.40 V bus-powered-hub VBUS.
        self.add("Regulator_Linear:AP2112K-3.3", "U5", "AP2112K-3.3", 300, 240,
                 "Package_TO_SOT_SMD:SOT-23-5",
                 Description="3.3V LDO, 600mA, 250mV dropout")
        self.net("U5", "1", "+5V")
        self.net("U5", "2", "GND")
        self.net("U5", "3", "+5V")     # enable tied high
        self.net("U5", "5", "+3V3")
        self.unused("U5", "4")

        # VBUS is the +5V rail. A dedicated 5V regulator would cost headroom
        # the relay coils cannot spare -- see the pull-in margin analysis.
        self.add("Device:R", "R9", "0R", 300, 210, "Resistor_SMD:R_0805_2012Metric",
                 Description="VBUS to +5V link, 0R jumper for current measurement")
        self.net("R9", "1", "VBUS")
        self.net("R9", "2", "+5V")

        bulk = [("C1", "22uF", "+5V", "Capacitor_SMD:C_0805_2012Metric"),
                ("C2", "22uF", "+3V3", "Capacitor_SMD:C_0805_2012Metric"),
                ("C3", "1uF", "+5V", "Capacitor_SMD:C_0603_1608Metric"),
                ("C4", "1uF", "+3V3", "Capacitor_SMD:C_0603_1608Metric")]
        for index, (reference, value, rail, footprint) in enumerate(bulk):
            self.add("Device:C", reference, value, 250 + index * 14, 300, footprint,
                     Description=f"bulk decoupling, {rail}")
            self.net(reference, "1", rail)
            self.net(reference, "2", "GND")

        for index in range(5, 13):
            reference = f"C{index}"
            # C5-C7 decouple the +5V coil rail at the drivers; C8/C9 are the
            # MCU's +3V3 locals; C10-C12 further +5V distribution.
            rail = "+3V3" if index in (8, 9) else "+5V"
            self.add("Device:C", reference, "100nF", 250 + (index - 5) * 14, 325,
                     "Capacitor_SMD:C_0402_1005Metric",
                     Description=f"decoupling, {rail}")
            self.net(reference, "1", rail)
            self.net(reference, "2", "GND")

        self.add("Device:LED", "D1", "PWR", 460, 240, "LED_SMD:LED_0603_1608Metric",
                 Description="status LED, firmware heartbeat")
        self.net("D1", "1", "GND")
        self.net("D1", "2", "LED_A")
        self.add("Device:R", "R10", "1k", 460, 215, "Resistor_SMD:R_0402_1005Metric")
        self.net("R10", "1", "+3V3")
        self.net("R10", "2", "LED_A")

        # +3V3 needs no flag: U5's VOUT is a power output and drives it.
        for net_name, x in (("VBUS", 240), ("+5V", 260), ("GND", 300)):
            self.power_flag(net_name, x, 355)

    def build_mcu(self):
        self.add("MCU_ST_STM32F0:STM32F072CBTx", "U1", "STM32F072CBT6", 420, 150,
                 "Package_QFP:LQFP-48_7x7mm_P0.5mm",
                 Description="Cortex-M0, 128K flash, native USB with crystal-driven 48 MHz")

        for name in ("VDD", "VDDA", "VDDIO2", "VBAT"):
            self.net("U1", MCU[name], "+3V3")
        self.net("U1", "24", "+3V3")   # second VDD pin; 48 is the other
        for name in ("VSS", "VSSA"):
            self.net("U1", MCU[name], "GND")
        # VSS pins 23 and 35 are drawn stacked on pin 47 in this symbol, so the
        # single GND stub above already connects all three.

        self.net("U1", MCU["NRST"], "NRST")
        self.net("U1", MCU["BOOT0"], "BOOT0")
        self.net("U1", MCU["PA13"], "SWDIO")
        self.net("U1", MCU["PA14"], "SWCLK")
        self.net("U1", MCU["PA11"], "USB_DM")
        self.net("U1", MCU["PA12"], "USB_DP")
        self.net("U1", MCU["PC13"], "LED_A")

        for index, name in enumerate(GPIO_PINS, start=1):
            self.net("U1", MCU[name], f"GPIO{index}")

        used = {"VDD", "VDDA", "VDDIO2", "VBAT", "VSS", "VSSA", "NRST", "BOOT0",
                "PA13", "PA14", "PA11", "PA12", "PC13", *GPIO_PINS}
        self.unused("U1", *[MCU[name] for name in MCU if name not in used])

        # No crystal: the F072 runs USB from HSI48 with CRS auto-trim off
        # the USB SOF -- proven silicon, and it removes a congested routing
        # cluster next to the MCU (PF0/PF1 stay free for future use).

        self.add("Device:R", "R3", "10k", 460, 120, "Resistor_SMD:R_0402_1005Metric",
                 Description="NRST pull-up")
        self.net("R3", "1", "+3V3")
        self.net("R3", "2", "NRST")
        self.add("Device:C", "C15", "100nF", 480, 120, "Capacitor_SMD:C_0402_1005Metric",
                 Description="NRST filter")
        self.net("C15", "1", "NRST")
        self.net("C15", "2", "GND")
        self.add("Device:R", "R4", "10k", 500, 120, "Resistor_SMD:R_0402_1005Metric",
                 Description="BOOT0 pull-down: boot from flash")
        self.net("R4", "1", "BOOT0")
        self.net("R4", "2", "GND")

    def build_drivers(self):
        """Three TBD62003APG on the ULN2003 footprint and pinout.

        DMOS, so about 0.1-0.2 V drop at 23 mA against the ULN2003's 0.8-1.0 V.
        With a 3.6 V must-operate coil and a 4.40 V worst-case bus-powered VBUS,
        the bipolar part leaves 3.40 V -- below spec. This one leaves ~4.2 V.
        """
        assignments = [("U2", range(1, 8)), ("U3", range(8, 15)), ("U4", range(15, 19))]
        for index, (reference, relays) in enumerate(assignments):
            self.add("Transistor_Array:ULN2003", reference, "TBD62003APG",
                     330, 380 + index * 60, "Package_SO:SOIC-16_3.9x9.9mm_P1.27mm",
                     Description="7ch DMOS relay driver, pin-compatible with ULN2003")
            self.net(reference, "8", "GND")
            self.net(reference, "9", "+5V")
            channels = list(relays)
            for channel, relay in enumerate(channels, start=1):
                self.net(reference, ULN_IN[channel], f"GPIO{relay}")
                self.net(reference, ULN_OUT[channel], f"K{relay}_COIL")
            for channel in range(len(channels) + 1, 8):
                self.unused(reference, ULN_IN[channel], ULN_OUT[channel])

    # ----------------------------------------------------------------- run

    def build(self):
        self.build_matrix()
        self.build_load_column()
        self.build_isolation()
        self.build_connectors()
        self.build_usb()
        self.build_power()
        self.build_mcu()
        self.build_drivers()
        return self

    def save(self):
        self.sch.save(SCHEMATIC)
        return SCHEMATIC

    def audit(self):
        """Report nets with a single connection -- always a wiring mistake."""
        singles = {name: uses for name, uses in self.net_uses.items() if len(uses) < 2}
        return singles


def run_erc(path):
    report = path.replace(".kicad_sch", "_erc.txt")
    result = subprocess.run([KICAD_CLI, "sch", "erc", "--format", "report",
                             "--severity-all", "-o", report, path],
                            capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if os.path.exists(report):
        counts = {}
        for line in open(report, encoding="utf-8", errors="replace"):
            if line.startswith("["):
                key = line.split("]")[0][1:]
                counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4d}  {key}")
        return counts
    return {}


def export_netlist(path):
    destination = path.replace(".kicad_sch", ".xml")
    result = subprocess.run([KICAD_CLI, "sch", "export", "netlist", "--format",
                             "kicadxml", "-o", destination, path],
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stderr.strip())
    return destination


if __name__ == "__main__":
    builder = Builder().build()
    path = builder.save()
    print(f"Schematic: {path}")
    statistics = builder.sch.get_statistics()
    print(f"  components {statistics.get('components')}  labels {statistics.get('labels')}  "
          f"wires {statistics.get('wires')}")

    singles = builder.audit()
    if singles:
        print(f"\n  {len(singles)} single-connection nets (each is a wiring bug):")
        for name, uses in sorted(singles.items()):
            print(f"    {name}: {uses}")
    else:
        print("  every net has at least two connections")

    print("\nERC:")
    counts = run_erc(path)
    print(f"\nNetlist: {export_netlist(path)}")
    sys.exit(1 if counts.get("error") else 0)
