"""
Independent verification of the rev B schematic netlist.

    python verify_board.py            (exit code 0 only if everything passes)

Reads relay_board.xml (exported by kicad-cli from the schematic) and checks it
against knowledge that does NOT come from generate_schematic.py:

  1. The G6K-2 contact map from the Omron datasheet / KiCad library symbol:
         coil 1,8 | pole1 COM=3 NC=2 NO=4 | pole2 COM=6 NC=7 NO=5
  2. The GPIO -> driver -> relay coil chain, all 18 channels.
  3. Every configuration in scripts/RelayBoardController.py, SIMULATED:
     relay contacts are closed per the config's relay word over a connectivity
     graph built from the netlist, then the resulting terminal-to-rail
     connectivity is compared against the config's declared intent.
     This is the check rev A never had -- its verifier validated the netlist
     against the same pin map that generated it, and printed ALL CORRECT over
     a board where energizing a relay disconnected the analyzer.

Rev A's verifier also exited 0 no matter what it found. This one does not.
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
import RelayBoardController as rbc  # noqa: E402  (config table under test)

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


# --------------------------------------------------------------------------


def load_netlist():
    tree = ET.parse(NETLIST)
    root = tree.getroot()
    pin_to_net = {}
    net_to_pins = {}
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
    drivers = sorted(r for r in components if components[r].startswith("TBD62003")
                     or components[r].startswith("ULN"))
    # Build driver channel map from the netlist itself: input pin n couples to
    # output pin 17-n per the ULN2003/TBD62003 datasheet.
    for relay_index in range(1, 19):
        relay = f"K{relay_index}"
        gpio_net = f"/GPIO{relay_index}"
        coil_net = pin_to_net.get((relay, COIL[1]))
        if coil_net is None:
            fail(f"{relay} coil- (pin {COIL[1]}) is not on any net")
            continue
        # find a driver whose input is on the gpio net and output on the coil net
        found = False
        for driver in drivers:
            for channel in range(1, 8):
                input_net = pin_to_net.get((driver, str(channel)))
                output_net = pin_to_net.get((driver, str(17 - channel)))
                if input_net == gpio_net and output_net == coil_net:
                    found = True
                    break
            if found:
                break
        if not found:
            fail(f"{relay}: no driver channel joins {gpio_net} to {coil_net}")
        supply = pin_to_net.get((relay, COIL[0]))
        if supply != "/+5V":
            fail(f"{relay} coil+ (pin {COIL[0]}) is on {supply}, expected /+5V")
    if not FAILURES:
        ok("all 18 GPIO -> TBD62003 -> coil chains verified")


def contact_edges(relay, energized, pin_to_net):
    """Nets bridged by this relay's contacts in the given state."""
    edges = []
    for pole in POLES:
        common = pin_to_net.get((relay, pole["COM"]))
        target = pin_to_net.get((relay, pole["NO" if energized else "NC"]))
        if common and target:
            edges.append((common, target))
    return edges


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


def relay_word_for_config(config):
    """Which relays are energized for a configuration -- the same rule the
    firmware implements: matrix bit for every (terminal, rail) assignment."""
    word = rbc.measurement_word(config)
    return {f"K{bit + 1}" for bit in range(rbc.RELAY_COUNT) if (word >> bit) & 1}


def simulate_config(config, pin_to_net):
    union = Union()
    energized = relay_word_for_config(config)
    for relay_index in range(1, 19):
        relay = f"K{relay_index}"
        for a, b in contact_edges(relay, relay in energized, pin_to_net):
            union.join(a, b)
    return union


def check_configs(pin_to_net):
    print("\n[2] Every measurement configuration, simulated over the netlist")
    all_ok = True
    for number, config in sorted(rbc.CONFIGS.items()):
        union = simulate_config(config, pin_to_net)
        problems = []

        # Terminals assigned to a rail must reach it; DUT side must reach the
        # terminal node (isolation relays de-energized -> COM-NC closed).
        for rail in ("HI", "LO", "LINK"):
            for terminal in config[rail]:
                if not union.together(f"/DUT_{terminal}", f"/RAIL_{rail}"):
                    problems.append(f"DUT_{terminal} does not reach RAIL_{rail}")

        # HI and LO must never be shorted together by the fixture.
        if union.together("/RAIL_HI", "/RAIL_LO"):
            problems.append("RAIL_HI shorted to RAIL_LO")

        # Unassigned terminals must be floating: not on HI, LO or LINK.
        assigned = set(config["HI"]) | set(config["LO"]) | set(config["LINK"])
        for terminal in "ABCD":
            if terminal in assigned:
                continue
            for rail in ("HI", "LO", "LINK"):
                if union.together(f"/DUT_{terminal}", f"/RAIL_{rail}"):
                    problems.append(f"unassigned DUT_{terminal} reaches RAIL_{rail}")

        # LINKed terminals must join each other but not the measurement rails
        # (already implied, but check the pairwise join explicitly).
        link = list(config["LINK"])
        for first, second in zip(link, link[1:]):
            if not union.together(f"/DUT_{first}", f"/DUT_{second}"):
                problems.append(f"LINK terminals {first},{second} not joined")
        if link and (union.together("/RAIL_LINK", "/RAIL_HI")
                     or union.together("/RAIL_LINK", "/RAIL_LO")):
            problems.append("LINK rail touches a measurement rail")

        # Floating terminals must be cut from their column bus too, or the
        # column's uncalibrated capacitance loads the DUT node.
        for terminal in rbc.floating_terminals(config):
            if union.together(f"/DUT_{terminal}", f"/T{terminal}"):
                problems.append(f"floating DUT_{terminal} still on its column bus")

        # Intent, not just connectivity: link_XY_* must join X and Y. Config 8
        # once put D alone on LINK and passed every check above.
        if config["name"].startswith("link_"):
            first, second = config["name"].split("_")[1]
            if not union.together(f"/DUT_{first}", f"/DUT_{second}"):
                problems.append(f"{config['name']} does not join {first}-{second}")

        if problems:
            all_ok = False
            for problem in problems:
                fail(f"config {number:2d} ({config['name']}): {problem}")
        else:
            ok(f"config {number:2d} {config['name']:18s} "
               f"HI={'+'.join(config['HI']) or '-':5s} LO={'+'.join(config['LO']) or '-':5s} "
               f"LINK={'+'.join(config['LINK']) or '-'}")
    return all_ok


def check_calibration_paths(pin_to_net):
    print("\n[3] Calibration standards, simulated (OPEN / SHORT / LOAD per config)")
    iso = {"K15", "K16", "K17", "K18"}

    def union_for(extra, config):
        union = Union()
        energized = relay_word_for_config(config) | set(extra)
        for relay_index in range(1, 19):
            relay = f"K{relay_index}"
            for a, b in contact_edges(relay, relay in energized, pin_to_net):
                union.join(a, b)
        return union

    failures_before = len(FAILURES)
    for number, config in sorted(rbc.CONFIGS.items()):
        # OPEN: isolation only. No DUT terminal may reach a rail, and the
        # rails must not be bridged.
        union = union_for(iso, config)
        leak = [t for t in "ABCD"
                if union.together(f"/DUT_{t}", "/RAIL_HI")
                or union.together(f"/DUT_{t}", "/RAIL_LO")]
        if leak:
            fail(f"config {number} OPEN: DUT terminals {leak} still reach the rails")
        if union.together("/RAIL_HI", "/RAIL_LO"):
            fail(f"config {number} OPEN: rails bridged with everything isolated")

        # SHORT: isolation + the first HI terminal's LO relay -> rails bridge.
        first_hi = config["HI"][0]
        short_relay = f"K{rbc.MATRIX_BITS[(first_hi, 'LO')] + 1}"
        union = union_for(iso | {short_relay}, config)
        if not union.together("/RAIL_HI", "/RAIL_LO"):
            fail(f"config {number} SHORT: closing {short_relay} does not bridge the rails")
        if any(union.together(f"/DUT_{t}", "/RAIL_HI") for t in "ABCD"):
            fail(f"config {number} SHORT: a DUT terminal leaked onto the rails")

        # LOAD: isolation + K13 + K14 -> rails joined only THROUGH R5.
        union = union_for(iso | {"K13", "K14"}, config)
        hi_leg = any(union.together("/RAIL_HI", pin_to_net.get(("R5", pin), "?"))
                     for pin in ("1", "2"))
        lo_leg = any(union.together("/RAIL_LO", pin_to_net.get(("R5", pin), "?"))
                     for pin in ("1", "2"))
        direct = union.together("/RAIL_HI", "/RAIL_LO")
        if not (hi_leg and lo_leg and not direct):
            fail(f"config {number} LOAD: hi_leg={hi_leg} lo_leg={lo_leg} "
                 f"direct_short={direct}")

    if len(FAILURES) == failures_before:
        ok(f"OPEN, SHORT and LOAD verified for all {len(rbc.CONFIGS)} configs "
           "(isolation holds, rails bridge only as intended, load only via R5)")


def check_supervisory(pin_to_net, components):
    print("\n[4] MCU, USB, programming, boot")
    checks = [
        (("U1", "44"), "/BOOT0", "BOOT0 on the F072's dedicated pin 44"),
        (("R4", "2"), "/GND", "BOOT0 pulled down -> boots from flash"),
        (("U1", "34"), "/SWDIO", "SWDIO reaches"),
        (("J8", "2"), "/SWDIO", "SWD header carries SWDIO"),
        (("U1", "33"), "/USB_DP", "PA12 is USB D+"),
        # USBLC6 is flow-through: pins 1&6 = I/O1 (one line), 3&4 = I/O2
        (("D2", "1"), "/USB_DP_CON", "I/O1 in: connector D+"),
        (("D2", "6"), "/USB_DP", "I/O1 out: MCU D+ (internally tied to pin 1)"),
        (("D2", "3"), "/USB_DM_CON", "I/O2 in: connector D-"),
        (("D2", "4"), "/USB_DM", "I/O2 out: MCU D- (internally tied to pin 3)"),
        (("J1", "A6"), "/USB_DP_CON", "connector D+ on the protected side"),
        (("J1", "A7"), "/USB_DM_CON", "connector D- on the protected side"),
        # Device:LED / LED_0603 pin 1 = cathode: drive on pin 2 (anode)
        (("D1", "2"), "/LED_A", "LED anode on the driven net"),
        (("D1", "1"), "/GND", "LED cathode to ground"),
        (("R1", "1"), "/CC1", "CC1 pulldown present"),
        (("U5", "5"), "/+3V3", "LDO output drives +3V3"),
        # --- integrated B-WIC bridge, per Omicron app note 2015-09-16 ---
        (("J6", "1"), "/RAIL_HI", "SOURCE BNC drives the HI rail"),
        (("J6", "2"), "/AGND", "SOURCE shield on analog ground"),
        (("J7", "1"), "/CH1_NODE", "CH1 BNC on the voltage divider"),
        (("J9", "1"), "/CH2_NODE", "CH2 BNC behind RC2"),
        (("R12", "1"), "/RAIL_HI", "RV1 (1k) from the HI rail"),
        (("R12", "2"), "/CH1_NODE", "RV1 into the CH1 node"),
        (("R13", "1"), "/CH1_NODE", "RV2 (47R) from the CH1 node"),
        (("R13", "2"), "/AGND", "RV2 to analog ground"),
        (("R14", "1"), "/RAIL_LO", "RC1a (4R7) shunt from the LO rail"),
        (("R14", "2"), "/AGND", "RC1a to analog ground"),
        (("R15", "1"), "/RAIL_LO", "RC1b (4R7) shunt pair"),
        (("R15", "2"), "/AGND", "RC1b to analog ground"),
        (("R16", "1"), "/RAIL_LO", "RC2 (47R) from the LO rail"),
        (("R16", "2"), "/CH2_NODE", "RC2 into CH2"),
        (("R11", "1"), "/AGND", "ground tie, analog side"),
        (("R11", "2"), "/GND", "ground tie, digital side"),
    ]
    for key, expected, label in checks:
        actual = pin_to_net.get(key)
        if actual == expected:
            ok(f"{label}")
        else:
            fail(f"{label}: {key} is on {actual}, expected {expected}")

    relays = [r for r, v in components.items() if r.startswith("K")]
    if len(relays) != 18:
        fail(f"expected 18 relays, netlist has {len(relays)}")
    else:
        ok("18 relays present")


if __name__ == "__main__":
    if not os.path.exists(NETLIST):
        print(f"Netlist {NETLIST} not found -- run generate_schematic.py first.")
        sys.exit(2)
    pin_to_net, net_to_pins, components = load_netlist()
    print(f"Netlist: {len(components)} components, "
          f"{len(net_to_pins)} nets")

    check_coils(pin_to_net, components)
    check_configs(pin_to_net)
    check_calibration_paths(pin_to_net)
    check_supervisory(pin_to_net, components)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) -- do not fabricate this netlist")
        sys.exit(1)
    print("ALL CHECKS PASSED (datasheet pin map, 15 simulated configs, "
          "calibration paths, supervisory nets)")
    sys.exit(0)
