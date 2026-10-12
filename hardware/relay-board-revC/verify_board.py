"""
Independent verification of the rev C schematic netlist.

    python verify_board.py            (exit code 0 only if everything passes)

Reads relay_board.xml (exported by kicad-cli from the schematic) and checks it
against knowledge that does NOT come from generate_schematic.py:

  1. The G6K-2 contact map from the Omron datasheet / KiCad library symbol:
         coil 1,8 | pole1 COM=3 NC=2 NO=4 | pole2 COM=6 NC=7 NO=5
  2. The GPIO -> driver -> relay coil chain, all 24 channels.
  3. Every configuration in scripts/RelayBoardController.py, SIMULATED with the
     rev C relay words from scripts/RelayMapRevC.py, in 2-wire and in Kelvin
     mode: relay contacts closed per the word over a connectivity graph built
     from the netlist, then force/sense connectivity compared with the
     configuration's intent.
  4. OPEN / SHORT / LOAD for every configuration, in both modes.
  5. Supervisory nets: MCU, USB, boot, bridge, Kelvin front-end.
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
import RelayBoardController as rbc  # noqa: E402  (configuration table)
import RelayMapRevC as b2              # noqa: E402  (rev C relay words under test)

HERE = os.path.dirname(os.path.abspath(__file__))
NETLIST = os.path.join(HERE, "relay_board.xml")

# Datasheet ground truth -- keep independent of the generator.
COIL = ("1", "8")
POLES = [{"COM": "3", "NC": "2", "NO": "4"},
         {"COM": "6", "NC": "7", "NO": "5"}]

FAILURES = []


def fail(message):
    FAILURES.append(message)
    print(f"  FAIL  {message}")


def ok(message):
    print(f"  ok    {message}")


def load_netlist():
    root = ET.parse(NETLIST).getroot()
    pin_to_net, net_to_pins = {}, {}
    for net in root.find(".//nets"):
        name = net.get("name")
        for node in net.findall("node"):
            key = (node.get("ref"), node.get("pin"))
            pin_to_net[key] = name
            net_to_pins.setdefault(name, []).append(key)
    components = {c.get("ref"): (c.find("value").text if c.find("value") is not None else "")
                  for c in root.find(".//components")}
    return pin_to_net, net_to_pins, components


def check_coils(pin_to_net, components):
    print("\n[1] Relay coil chains: GPIO -> driver input, driver output -> coil")
    before = len(FAILURES)
    drivers = sorted(r for r in components if components[r].startswith(("TBD62003", "ULN")))
    for relay_index in range(1, b2.RELAY_COUNT + 1):
        relay = f"K{relay_index}"
        gpio_net = f"/GPIO{relay_index}"
        coil_net = pin_to_net.get((relay, COIL[1]))
        if coil_net is None:
            fail(f"{relay} coil- (pin {COIL[1]}) is not on any net")
            continue
        found = any(pin_to_net.get((d, str(ch))) == gpio_net
                    and pin_to_net.get((d, str(17 - ch))) == coil_net
                    for d in drivers for ch in range(1, 8))
        if not found:
            fail(f"{relay}: no driver channel joins {gpio_net} to {coil_net}")
        if pin_to_net.get((relay, COIL[0])) != "/+5V":
            fail(f"{relay} coil+ is on {pin_to_net.get((relay, COIL[0]))}, expected /+5V")
    for driver in drivers:
        if pin_to_net.get((driver, "9")) != "/+5V":
            fail(f"{driver} COM (clamp diodes) not on +5V")
    if len(FAILURES) == before:
        ok(f"all {b2.RELAY_COUNT} GPIO -> TBD62003 -> coil chains verified, "
           f"{len(drivers)} drivers with COM on +5V")


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def together(self, a, b):
        return self.find(a) == self.find(b)


def simulate(word, pin_to_net, components, net_ties=True):
    """Connectivity with relays per ``word``; net ties (NT*) always joined."""
    union = Union()
    on = b2.energized(word)
    for relay_index in range(1, b2.RELAY_COUNT + 1):
        relay = f"K{relay_index}"
        for pole in POLES:
            common = pin_to_net.get((relay, pole["COM"]))
            target = pin_to_net.get((relay, pole["NO" if relay in on else "NC"]))
            if common and target:
                union.join(common, target)
    if net_ties:
        for reference in components:
            if reference.startswith("NT"):
                union.join(pin_to_net[(reference, "1")], pin_to_net[(reference, "2")])
    return union


def check_configs(pin_to_net, components):
    print("\n[2] Every measurement configuration, simulated (2-wire and Kelvin)")
    before = len(FAILURES)
    for number, config in sorted(rbc.CONFIGS.items()):
        for kelvin in (False, True):
            mode = "Kelvin" if kelvin else "2-wire"
            try:
                word = b2.measurement_word(config, kelvin=kelvin)
                b2.check_budget(word)
            except ValueError as error:
                fail(f"config {number} {mode}: {error}")
                continue
            union = simulate(word, pin_to_net, components)
            problems = []
            for rail in ("HI", "LO"):
                for terminal in config[rail]:
                    if not union.together(f"/DUT_{terminal}", f"/RAIL_{rail}"):
                        problems.append(f"force DUT_{terminal} misses RAIL_{rail}")
                    if not union.together(f"/DUTS_{terminal}", f"/SNS_{rail}"):
                        problems.append(f"sense DUTS_{terminal} misses SNS_{rail}")
            if union.together("/RAIL_HI", "/RAIL_LO"):
                problems.append("RAIL_HI shorted to RAIL_LO")
            if union.together("/SNS_HI", "/SNS_LO"):
                problems.append("SNS_HI shorted to SNS_LO")
            hi_tied = union.together("/SNS_HI", "/RAIL_HI")
            lo_tied = union.together("/SNS_LO", "/RAIL_LO")
            if kelvin and (hi_tied or lo_tied):
                problems.append("Kelvin mode but sense still tied to force")
            if not kelvin and not (hi_tied and lo_tied):
                problems.append("2-wire mode but sense not tied to force")
            ch1 = "/FE_OUT" if kelvin else "/CH1_NODE"
            if not union.together("/CH1_BNC", ch1):
                problems.append(f"CH1 BNC not on {ch1}")
            assigned = set(config["HI"]) | set(config["LO"]) | set(config["LINK"])
            for terminal in "ABCD":
                if terminal in assigned:
                    continue
                for net in ("/RAIL_HI", "/RAIL_LO", "/SNS_HI", "/SNS_LO"):
                    if union.together(f"/DUT_{terminal}", net) or \
                            union.together(f"/DUTS_{terminal}", net):
                        problems.append(f"floating {terminal} reaches {net}")
                if union.together(f"/DUT_{terminal}", f"/T{terminal}"):
                    problems.append(f"floating {terminal} not isolated from its bus")
            link = list(config["LINK"])
            for first, second in zip(link, link[1:]):
                if not union.together(f"/DUT_{first}", f"/DUT_{second}"):
                    problems.append(f"link {first}-{second} force not joined")
                if not union.together(f"/DUTS_{first}", f"/DUTS_{second}"):
                    problems.append(f"link {first}-{second} sense not joined")
            for terminal in link:
                for net in ("/RAIL_HI", "/RAIL_LO"):
                    if union.together(f"/DUT_{terminal}", net):
                        problems.append(f"linked {terminal} touches {net}")
            for problem in problems:
                fail(f"config {number:2d} ({config['name']}) {mode}: {problem}")
    if len(FAILURES) == before:
        ok(f"{len(rbc.CONFIGS)} configurations x 2 modes: force on rails, sense on "
           "sense rails, tie/CH1 per mode, floating terminals isolated, links joined")


def check_calibration_paths(pin_to_net, components):
    print("\n[3] Calibration standards, simulated (OPEN / SHORT / LOAD, both modes)")
    before = len(FAILURES)
    duts = [f"/DUT{s}_{t}" for t in "ABCD" for s in ("", "S")]
    rails = ("/RAIL_HI", "/RAIL_LO", "/SNS_HI", "/SNS_LO")
    for number, config in sorted(rbc.CONFIGS.items()):
        if not config["HI"] or not config["LO"]:
            continue                      # ground configs: no HI-LO standard
        for kelvin in (False, True):
            tag = f"config {number} {'Kelvin' if kelvin else '2-wire'}"
            for mode in ("OPEN", "SHORT", "LOAD", "CHECK_R1", "CHECK_C10"):
                word = b2.calibration_word(config, mode, kelvin=kelvin)
                try:
                    b2.check_budget(word)
                except ValueError as error:
                    fail(f"{tag} {mode}: {error}")
                union = simulate(word, pin_to_net, components)
                if any(union.together(d, r) for d in duts for r in rails):
                    fail(f"{tag} {mode}: a clamp still reaches a rail")
                bridged = union.together("/RAIL_HI", "/RAIL_LO")
                if mode == "OPEN" and bridged:
                    fail(f"{tag} OPEN: rails bridged")
                if mode == "SHORT":
                    hi = config["HI"][0]
                    if not bridged:
                        fail(f"{tag} SHORT: rails not bridged")
                    if not union.together("/RAIL_HI", f"/T{hi}"):
                        fail(f"{tag} SHORT: not through T{hi}")
                    if kelvin and not (union.together("/SNS_HI", f"/S{hi}")
                                       and union.together("/SNS_LO", f"/S{hi}")):
                        fail(f"{tag} SHORT: sense rails do not meet on S{hi} (no Kelvin zero)")
                if mode in ("LOAD", "CHECK_R1", "CHECK_C10"):
                    stem = {"LOAD": "STD1", "CHECK_R1": "STD2", "CHECK_C10": "STD3"}[mode]
                    if bridged:
                        fail(f"{tag} {mode}: rails shorted (standard bypassed)")
                    if not (union.together("/RAIL_HI", f"/{stem}_F")
                            and union.together("/RAIL_LO", "/CAL_F")):
                        fail(f"{tag} {mode}: {stem} not between the rails")
                    if not (union.together("/SNS_HI", f"/{stem}_S")
                            and union.together("/SNS_LO", "/CAL_FS")):
                        fail(f"{tag} {mode}: {stem} sense not on the sense rails")
                    others = [x for x in ("STD1", "STD2", "STD3") if x != stem]
                    if any(union.together("/CAL_E", f"/{x}_F") for x in others):
                        fail(f"{tag} {mode}: more than one standard selected")
    plain = simulate(0, pin_to_net, components, net_ties=False)
    if plain.together("/CAL_F", "/CAL_FS") or any(
            plain.together(f"/STD{i}_F", f"/STD{i}_S") for i in (1, 2, 3)):
        fail("Kelvin sense nets are joined somewhere other than their net tie")
    if len(FAILURES) == before:
        ok("OPEN/SHORT/LOAD/CHECK verified for every HI-LO configuration in both modes "
           "(clamps isolated, SHORT through the HI bus with an exact Kelvin zero, "
           "one standard at a time, sense at the standard's pads)")


def check_supervisory(pin_to_net, components):
    print("\n[4] MCU, USB, programming, boot, bridge, Kelvin front-end")
    checks = [
        (("U1", "44"), "/BOOT0", "BOOT0 on the F072's dedicated pin 44"),
        (("R4", "2"), "/GND", "BOOT0 pulled down -> boots from flash"),
        (("SW1", "1"), "/+3V3", "BOOT button from +3V3"),
        (("SW1", "2"), "/BOOT_SW", "BOOT button into its series resistor"),
        (("R17", "2"), "/BOOT0", "BOOT button reaches BOOT0 through R17"),
        (("SW2", "1"), "/NRST", "RESET button on NRST"),
        (("SW2", "2"), "/GND", "RESET button pulls NRST to ground"),
        (("TP1", "1"), "/BOOT0", "BOOT0 strap pad"),
        (("TP2", "1"), "/+3V3", "+3V3 strap pad next to it"),
        (("U1", "9"), "/+3V3A", "VDDA on its filtered rail"),
        (("FB3", "1"), "/+3V3", "VDDA ferrite from +3V3"),
        (("U1", "34"), "/SWDIO", "SWDIO reaches"),
        (("J8", "2"), "/SWDIO", "Cortex header pin 2 = SWDIO"),
        (("J8", "4"), "/SWCLK", "Cortex header pin 4 = SWCLK"),
        (("J8", "10"), "/NRST", "Cortex header pin 10 = nRESET"),
        (("J8", "1"), "/+3V3", "Cortex header pin 1 = VTref"),
        (("U1", "33"), "/USB_DP", "PA12 is USB D+"),
        (("U1", "32"), "/USB_DM", "PA11 is USB D-"),
        (("D2", "1"), "/USB_DM_CON", "ESD I/O1 in: connector D-"),
        (("D2", "6"), "/USB_DM_ESD", "ESD I/O1 out"),
        (("D2", "3"), "/USB_DP_CON", "ESD I/O2 in: connector D+"),
        (("D2", "4"), "/USB_DP_ESD", "ESD I/O2 out"),
        # TDK ACM2012: windings 1-4 and 2-3 (TDK equivalent-circuit library)
        (("L1", "1"), "/USB_DP", "CM choke winding 1-4, MCU side: D+"),
        (("L1", "4"), "/USB_DP_ESD", "CM choke winding 1-4, connector side: D+"),
        (("L1", "2"), "/USB_DM", "CM choke winding 2-3, MCU side: D-"),
        (("L1", "3"), "/USB_DM_ESD", "CM choke winding 2-3, connector side: D-"),
        (("J1", "A6"), "/USB_DP_CON", "connector D+ on the protected side"),
        (("J1", "A7"), "/USB_DM_CON", "connector D- on the protected side"),
        (("FB1", "1"), "/VBUS", "VBUS ferrite at the connector"),
        (("FB1", "2"), "/+5V_IN", "VBUS ferrite feeds the cable-side +5V_IN"),
        (("D9", "1"), "/VBUS", "VBUS TVS cathode on VBUS"),
        (("D9", "2"), "/GND", "VBUS TVS anode to ground"),
        (("U5", "1"), "/+5V_IN", "LDO (MCU) powered from the cable side"),
        (("U11", "5"), "/+5V_IN", "coil switch input on +5V_IN"),
        (("U11", "1"), "/+5V", "coil switch output is the coil rail"),
        (("U11", "4"), "/COIL_EN", "coil switch enable from the MCU"),
        (("U1", "25"), "/COIL_EN", "PB12 drives COIL_EN"),
        (("R40", "2"), "/GND", "COIL_EN pulled low: coils off through reset"),
        (("U1", "26"), "/COIL_FLT", "PB13 reads the coil-switch fault flag"),
        (("J1", "SH"), "/USB_SHIELD", "USB shield on its own net (R37 strap / RC option)"),
        (("R37", "2"), "/GND", "R37 bonds the shield to ground"),
        (("U12", "1"), "/I2C_SCL", "TMP117 SCL"),
        (("U12", "6"), "/I2C_SDA", "TMP117 SDA"),
        (("U1", "21"), "/I2C_SCL", "PB10 is I2C2 SCL"),
        (("U1", "22"), "/I2C_SDA", "PB11 is I2C2 SDA"),
        (("R10", "1"), "/LED_RUN", "RUN LED driven from PA9"),
        (("U1", "30"), "/LED_RUN", "PA9 is LED_RUN"),
        (("D1", "2"), "/RUN_A", "RUN LED anode behind its resistor"),
        (("D1", "1"), "/GND", "LED cathode to ground"),
        (("R32", "1"), "/+3V3", "PWR LED hardwired to +3V3"),
        (("U1", "29"), "/LED_BUSY", "PA8 drives BUSY"),
        (("U1", "28"), "/LED_KELVIN", "PB15 drives KELVIN"),
        (("U1", "27"), "/LED_PROG", "PB14 drives PROG"),
        (("D8", "2"), "/PROG_A", "PROG LED anode behind its resistor"),
        (("R1", "1"), "/CC1", "CC1 pulldown present"),
        (("U5", "5"), "/+3V3", "LDO output drives +3V3"),
        # --- integrated B-WIC bridge, per Omicron app note 2015-09-16 ---
        (("J6", "1"), "/RAIL_HI", "SOURCE BNC drives the HI rail"),
        (("J6", "2"), "/GND", "SOURCE shield on the (single) ground"),
        (("J7", "1"), "/CH1_BNC", "CH1 BNC on the K24 common"),
        (("J9", "1"), "/CH2_NODE", "CH2 BNC behind RC2"),
        (("K23", "2"), "/CH1_NODE", "K23 NC: bridge divider (2-wire default)"),
        (("K23", "4"), "/FE_OUT", "K23 NO: Kelvin front-end"),
        (("R5", "1"), "/STD1_F", "100R LOAD standard behind its select relay"),
        (("R18", "1"), "/STD2_F", "1R check standard behind its select relay"),
        (("C30", "1"), "/STD3_F", "10 pF check standard behind its select relay"),
        (("R12", "1"), "/RAIL_HI", "RV1 (1k) from the HI rail"),
        (("R12", "2"), "/CH1_NODE", "RV1 into the CH1 node"),
        (("R13", "1"), "/CH1_NODE", "RV2 (47R) from the CH1 node"),
        (("R13", "2"), "/GND", "RV2 to ground"),
        (("R14", "1"), "/RAIL_LO", "RC1a (4R7) shunt from the LO rail"),
        (("R15", "1"), "/RAIL_LO", "RC1b (4R7) shunt pair"),
        (("R16", "1"), "/RAIL_LO", "RC2 (47R) from the LO rail"),
        (("R16", "2"), "/CH2_NODE", "RC2 into CH2"),
        # --- Kelvin front-end: out = VMID + (SNS_HI - SNS_LO) ---
        (("C20", "1"), "/SNS_HI", "HI sense AC coupling"),
        (("C21", "1"), "/SNS_LO", "LO sense AC coupling"),
        (("U7", "3"), "/FE_P", "HI buffer non-inverting input"),
        (("U7", "1"), "/FE_BP", "HI buffer output"),
        (("U7", "4"), "/FE_BP", "HI buffer is a follower"),
        (("U8", "3"), "/FE_N", "LO buffer non-inverting input"),
        (("U8", "4"), "/FE_BN", "LO buffer is a follower"),
        (("R26", "1"), "/FE_BP", "HI path to the diff stage + input"),
        (("U9", "3"), "/FE_DP", "diff stage + input"),
        (("R24", "1"), "/FE_BN", "LO path to the diff stage - input"),
        (("U9", "4"), "/FE_DM", "diff stage - input"),
        (("R25", "2"), "/FE_AMP", "diff stage feedback from the output"),
        (("R27", "2"), "/VMID", "diff stage reference at VMID"),
        (("U10", "1"), "/VMID", "VMID buffered"),
        (("D3", "1"), "/GND", "clamp anode to ground (BAV99 pin 1)"),
        (("D3", "2"), "/+5VA", "clamp cathode to +5VA (BAV99 pin 2)"),
        (("D3", "3"), "/FE_P", "clamp on the buffer input"),
        (("FB2", "1"), "/+5V", "front-end supply filter"),
    ]
    for key, expected, label in checks:
        actual = pin_to_net.get(key)
        if actual == expected:
            ok(label)
        else:
            fail(f"{label}: {key} is on {actual}, expected {expected}")
    for amp in ("U7", "U8", "U9", "U10"):
        if pin_to_net.get((amp, "5")) != "/+5VA" or pin_to_net.get((amp, "2")) != "/GND":
            fail(f"{amp} supply pins not on +5VA/GND")
    if any(name == "/AGND" for name in pin_to_net.values()):
        fail("an AGND net still exists: rev C uses one unified ground")
    relays = [r for r in components if r.startswith("K")]
    if len(relays) != b2.RELAY_COUNT:
        fail(f"expected {b2.RELAY_COUNT} relays, netlist has {len(relays)}")
    else:
        ok(f"{b2.RELAY_COUNT} relays present, unified ground, op-amp supplies")


if __name__ == "__main__":
    if not os.path.exists(NETLIST):
        print(f"Netlist {NETLIST} not found -- run generate_schematic.py first.")
        sys.exit(2)
    pin_to_net, net_to_pins, components = load_netlist()
    print(f"Netlist: {len(components)} components, {len(net_to_pins)} nets")

    check_coils(pin_to_net, components)
    check_configs(pin_to_net, components)
    check_calibration_paths(pin_to_net, components)
    check_supervisory(pin_to_net, components)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) -- do not fabricate this netlist")
        sys.exit(1)
    print(f"ALL CHECKS PASSED (datasheet pin map, {len(rbc.CONFIGS)} configs x 2 modes, "
          "calibration standards, supervisory nets)")
    sys.exit(0)
