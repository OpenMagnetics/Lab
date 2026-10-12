"""
Relay map of the rev C board (Kelvin force/sense, direct links, SHORT star).

The measurement configurations themselves (which terminals go to HI, LO and
which are linked) are board-independent and live in RelayBoardController.CONFIGS;
this module turns them into rev C relay words. It must stay in sync with
hardware/relay-board-revC/generate_schematic.py and firmware/include/relay_map.h
(rev C section); hardware/relay-board-revC/verify_board.py simulates every
word produced here over the exported netlist.

Bit n-1 drives relay K n:

    K1..K8    matrix, (terminal, rail) for A/B/C/D x HI/LO -- pole 1 force,
              pole 2 sense
    K9, K10   standards column -> HI, -> LO
    K11..K14  isolation A..D (energized = DUT disconnected)
    K15..K18  direct links A-B, C-D, B-C, B-D
    K19..K21  standard select: 100R LOAD, 1R check, 10 pF check (Kelvin-sensed)
    K22       2-wire tie (energized = Kelvin: sense rails off the force rails)
    K23       CH1 select (energized = Kelvin front-end on CH1)

SHORT is the HI terminal's own LO relay (rev B style). A design with four
SHORT relays on the iso NO throws was evaluated and dropped: it moved the
2-wire residual from +100 mohm/+29 nH to -100 mohm/-31 nH, cost ~90 fF of
Cps accuracy, and both are removed anyway by the clamp-plane fixture
compensation that Keysight and Voltech practice requires
(hardware/relay-board-revC/simulate_calibration.py, kelvin_coupling.py).
On this board it is also an exact Kelvin zero: both sense rails meet on the
HI terminal's current-free sense bus.
"""

TERMINALS = ("A", "B", "C", "D")
RELAY_COUNT = 23

MATRIX_BITS = {(terminal, rail): 2 * index + offset
               for index, terminal in enumerate(TERMINALS)
               for offset, rail in enumerate(("HI", "LO"))}
LOAD_HI_BIT, LOAD_LO_BIT = 8, 9
ISOLATE_BITS = {terminal: 10 + index for index, terminal in enumerate(TERMINALS)}
LINK_BITS = {("A", "B"): 14, ("C", "D"): 15, ("B", "C"): 16, ("B", "D"): 17}
STANDARD_BITS = {"R100": 18, "R1": 19, "C10": 20}
KELVIN_TIE_BIT = 21
CH1_KELVIN_BIT = 22

#: Never energize more relays than this at once: 23 x ~21 mA would exceed the
#: 500 mA a USB 2.0 port grants a bus-powered device.
MAX_ENERGIZED = 16


def _bit(word, bit):
    return word | (1 << bit)


def link_relays(terminals):
    """Link bits that join every terminal in ``terminals`` into one node.

    A spanning tree over the available direct links (A-B, C-D, B-C, B-D).
    Raises ValueError if the set cannot be joined with those links alone.
    """
    terminals = list(dict.fromkeys(terminals))
    if len(terminals) < 2:
        return []
    joined = {terminals[0]}
    bits = []
    while len(joined) < len(terminals):
        for (first, second), bit in LINK_BITS.items():
            if first in terminals and second in terminals and \
                    (first in joined) != (second in joined):
                joined |= {first, second}
                bits.append(bit)
                break
        else:
            raise ValueError(f"no direct link path joins {terminals}")
    return bits


def floating_terminals(config):
    assigned = set(config["HI"]) | set(config["LO"]) | set(config["LINK"])
    return tuple(t for t in TERMINALS if t not in assigned)


def measurement_word(config, kelvin=False):
    """Relay word for a configuration in MEAS mode.

    Same rules as rev B: the matrix relay of every (terminal, rail)
    assignment, the isolation relay of every FLOATING terminal (so its clamp
    and bus do not load the DUT), and the direct links for the LINK set.
    ``kelvin`` opens the 2-wire tie and puts the front-end on CH1.
    """
    word = 0
    for rail in ("HI", "LO"):
        for terminal in config[rail]:
            word = _bit(word, MATRIX_BITS[(terminal, rail)])
    for bit in link_relays(config["LINK"]):
        word = _bit(word, bit)
    for terminal in floating_terminals(config):
        word = _bit(word, ISOLATE_BITS[terminal])
    if kelvin:
        word = _bit(_bit(word, KELVIN_TIE_BIT), CH1_KELVIN_BIT)
    return word


def calibration_word(config, mode, kelvin=False):
    """Relay word for an OSL standard in the exact state of ``config``.

    OPEN   every terminal isolated; the buses end on open NO throws.
    SHORT  isolated + the first HI terminal's LO relay: its bus bridges the
           force rails; in Kelvin mode both sense rails sit on its sense bus.
    LOAD   isolated + the standards column with R5 (100R) selected.
    CHECK_R1 / CHECK_C10: the 1R / 10 pF verification standards instead.
    """
    word = measurement_word(config, kelvin=kelvin)
    for bit in ISOLATE_BITS.values():
        word = _bit(word, bit)
    mode = mode.upper()
    if mode == "SHORT":
        word = _bit(word, MATRIX_BITS[(config["HI"][0], "LO")])
    elif mode in ("LOAD", "CHECK_R1", "CHECK_C10"):
        standard = {"LOAD": "R100", "CHECK_R1": "R1", "CHECK_C10": "C10"}[mode]
        word = _bit(_bit(word, LOAD_HI_BIT), LOAD_LO_BIT)
        word = _bit(word, STANDARD_BITS[standard])
    elif mode != "OPEN":
        raise ValueError(f"unknown calibration mode {mode!r}")
    return word


def energized(word):
    """Relay names (K1..K23) energized by a word."""
    return {f"K{bit + 1}" for bit in range(RELAY_COUNT) if word >> bit & 1}


def check_budget(word):
    count = bin(word).count("1")
    if count > MAX_ENERGIZED:
        raise ValueError(f"{count} relays energized > {MAX_ENERGIZED} (USB budget)")
    return count
