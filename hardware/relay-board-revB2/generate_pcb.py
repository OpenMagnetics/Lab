"""
Generate the rev B PCB with parasitic-driven placement and pre-routed
signal nets.

    "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" generate_pcb.py

PARASITIC HIERARCHY -- what drove every placement decision.

The OSL calibration plane sits at the isolation-relay contact.  Everything on
the instrument side of that plane is measured during calibration in the same
relay state as the measurement, so its parasitics cancel; everything on the
DUT side adds directly to the answer.  Placement priority follows:

  1. DUT arms (clamp -> iso NC): UNCALIBRATED. Dead-straight 9.6 mm verticals,
     one per terminal, 22 mm apart, no plane underneath, nothing routed near
     them.  ~8 nH + ~0.4 pF each is the honest fixture floor (rev A: ~300 pF).
  2. Column buses (iso COM -> three crossbar COMs): calibrated per config.
     Straight F.Cu verticals with 2.5 mm stubs; length identical across
     columns so the load column mimics a terminal column exactly.
  3. Rails (crossbar NO pins -> Bode tabs): constant and calibrated. Routed on
     B.Cu straight under the NO-pad rows -- the matrix graph is K(3,5), which
     is non-planar, so rails and buses MUST cross somewhere; giving rails
     their own layer removes every same-layer crossing by construction.
  4. Coils, GPIO, power, USB: static DC during a sweep. Freerouted afterwards;
     the signal region's F.Cu is already occupied so they naturally fall to
     B.Cu, crossing the F.Cu signal verticals at right angles.

The inner planes (+5V, GND) and the B.Cu ground pour are voided under the
whole signal region: no copper under any measurement net.

LAYOUT (top view).  The DUT bay is a 31 x 24 mm clear area whose four clamp
mouths face each other -- the magnetic sits IN the bay, primary leads west
into A/B, secondary leads east into C/D:

  iso K15  [A >          DUT           < C]  iso K17    | SWD  pwr
  iso K16  [B >          BAY           < D]  iso K18    | U2 drv
     K1  HI    K4  HI    K13 HI    K7  HI    K10 HI     | U3 drv
     K2  LO    K5  LO    K14 LO    K8  LO    K11 LO     | U4 drv
     K3  LNK   K6  LNK   R5=100R   K9  LNK   K12 LNK    | U1 MCU
          [HI tab @87]   [LO tab @111]                  | USB
"""

import os
import xml.etree.ElementTree as ET

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
NETLIST = os.path.join(HERE, "relay_board.xml")
OUTPUT = os.path.join(HERE, "relay_board.kicad_pcb")
FP_SYSTEM = r"C:\Program Files\KiCad\9.0\share\kicad\footprints"
FP_PROJECT = HERE                       # OpenMagnetics.pretty lives here

def mm(value):
    return pcbnew.FromMM(value)

def vec(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))

# ---------------------------------------------------------------- geometry

BOARD_X0, BOARD_Y0, BOARD_X1, BOARD_Y1 = 40.0, 40.0, 190.0, 130.0
#: Integrated B-WIC bridge (Omicron app note 2015-09-16): three BNCs on the
#: bottom edge, body-on-board depth 12.7 mm, collar overhanging the edge.
BNC_Y = BOARD_Y1 - 12.7                 # anchor so the body ends at the edge
BNC_X = {"SOURCE": 64.0, "CH1": 84.0, "CH2": 104.0}
BRIDGE_BAND_Y = 111.5                   # bridge resistor row

#: Terminal columns 22 mm apart; E is the load-standard column.
COLUMNS = {"A": 55.0, "B": 77.0, "E": 99.0, "C": 121.0, "D": 143.0}

ROW_ISO, ROW_HI, ROW_LO, ROW_LINK = 58.0, 72.0, 84.0, 96.0

#: Relay pad geometry (OpenMagnetics:Relay_Omron_G6K-2F-RF-S, rotation 0):
#: pole-1 pads at x = centre-3.5, y = centre + {-0.6 NC, +1.6 COM, +3.8 NO}.
#: Relay centres sit at column+3.5 so pole-1 pads land exactly on the column.
RELAY_DX = 3.5
PAD_NC, PAD_COM, PAD_NO = -0.6, 1.6, 3.8

BUS_OFFSET = 2.5                        # column buses at column - 2.5
RAIL_Y = {"HI": ROW_HI + PAD_NO, "LO": ROW_LO + PAD_NO, "LINK": ROW_LINK + PAD_NO}
DESCENT_HI_X, DESCENT_LO_X = 87.0, 111.0

SIGNAL_TRACK = 0.5                      # mm
PLAIN_TRACK = 0.25

#: No plane copper in the signal region.
PLANE_VOID = (BOARD_X0 + 2.0, BOARD_Y0 + 2.0, 151.5, BOARD_Y1 - 2.0)

#: Face-to-face DUT bay.  Clamp entry is +y in footprint coordinates, so
#: rotation -90 faces east and +90 faces west.  Isolation relays sit rotated
#: 90 outside the bay with their contact row on the clamp line; the arm exits
#: the clamp's OUTER pad and travels one clean horizontal per terminal.
BAY_ROW_A, BAY_ROW_B = 45.0, 53.0        # A/C on the upper row, B/D lower
CLAMP_WEST_X, CLAMP_EAST_X = 79.0, 119.0

PLACEMENT = {
    # DUT clamps around the bay. Footprint entry is +y; KiCad rotation maps
    # +y -> +x at rot 90 and +y -> -x at rot -90 (verified from the placed
    # NPTH positions), so the left pair takes +90, the right pair -90.
    "J2": (CLAMP_WEST_X, BAY_ROW_A, 90),    # A, mouth east into the bay
    "J3": (CLAMP_WEST_X, BAY_ROW_B, 90),    # B
    "J4": (CLAMP_EAST_X, BAY_ROW_A, -90),   # C, mouth west into the bay
    "J5": (CLAMP_EAST_X, BAY_ROW_B, -90),   # D
    # Load standard, east of the bridge so it never sits under the DUT
    # or the BNC bodies.
    "R5": (134.0, 112.0, 90),
    # Isolation relays flank the bay in the SLIM orientation (long axis
    # vertical), contact column facing its clamp: rot 180 puts pole-1 pads on
    # the east side (left flank), rot 0 on the west side (right flank).
    "K15": (70.0, 46.1, 180),               # iso A: NC (73.5, 46.7)
    "K16": (70.0, 57.7, 180),               # iso B: NC (73.5, 58.3)
    "K17": (128.0, 46.1, 0),                # iso C: NC (124.5, 45.5)
    "K18": (128.0, 57.8, 0),                # iso D: NC (124.5, 57.2)
    # Load column relays.  K14 is rotated 180 so its pole-1 pads face the
    # OTHER side (column+7): CAL_E and CAL_F then run as separate verticals
    # instead of colliding on the same column line.
    "K13": (COLUMNS["E"] + RELAY_DX, ROW_HI, 0),
    "K14": (COLUMNS["E"] + RELAY_DX, ROW_LO, 180),
    # Bridge BNCs, mating south off the bottom edge (rot 180).
    "J6": (BNC_X["SOURCE"], BNC_Y, 180),
    "J7": (BNC_X["CH1"], BNC_Y, 180),
    "J9": (BNC_X["CH2"], BNC_Y, 180),
    # Bridge resistors in the band between the rails and the BNC bodies.
    # All bridge nets are inside the OSL calibration loop, so freerouting
    # may route them; only the AGND fanout is locked.
    "R12": (72.0, BRIDGE_BAND_Y, 90),    # RV1 1k
    "R13": (78.0, BRIDGE_BAND_Y, 90),    # RV2 47R
    "R16": (100.0, BRIDGE_BAND_Y, 90),   # RC2 47R
    "R14": (110.0, BRIDGE_BAND_Y, 90),   # RC1 4R7 pair
    "R15": (114.0, BRIDGE_BAND_Y, 90),
    "R11": (147.5, BRIDGE_BAND_Y, 0),    # AGND-GND single-point tie
    # ---- control strip (planes present, x > 151.5) ----
    "J8": (164.0, 45.0, 0),             # SWD
    "U2": (158.0, 62.0, 0), "C5": (166.9, 58.0, 90),
    "U3": (158.0, 78.0, 0), "C6": (168.1, 74.0, 90),
    "U4": (158.0, 92.0, 0), "C7": (168.1, 88.0, 90),
    "U1": (160.0, 112.0, 0),
    "R3": (169.0, 105.0, 0), "C15": (169.0, 108.0, 0), "R4": (169.0, 102.0, 0),
    "C8": (155.9, 118.5, 0), "C9": (167.0, 118.5, 0),
    "C10": (172.5, 109.0, 90), "C11": (168.1, 65.5, 90), "C12": (168.1, 81.5, 90),
    # USB corner on the right edge: connector nose east, ESD array as a
    # true flow-through between it and the MCU (D- on I/O1, D+ on I/O2).
    "J1": (187.75, 112.0, 90),          # USB-C, nose flush with edge tab
    "D2": (178.5, 112.0, 180),          # pads 1/2/3 face J1, 4/5/6 face U1
    "R1": (181.2, 116.0, 270), "R2": (181.2, 107.0, 90),
    "U5": (169.0, 53.0, 0), "R9": (169.0, 47.5, 0),
    "C1": (174.0, 53.0, 90), "C2": (174.0, 59.0, 90),
    "C3": (169.0, 59.0, 90), "C4": (175.0, 65.0, 90),
    "D1": (155.0, 53.0, 0), "R10": (155.0, 49.2, 0),
}

# Matrix relays: column per terminal, row per rail; K index from MATRIX_BITS.
for index, (terminal, rail) in enumerate(
        [(t, r) for t in "ABCD" for r in ("HI", "LO", "LINK")], start=1):
    row = {"HI": ROW_HI, "LO": ROW_LO, "LINK": ROW_LINK}[rail]
    PLACEMENT[f"K{index}"] = (COLUMNS[terminal] + RELAY_DX, row, 0)

MECHANICAL = [
    ("H1", "MountingHole", "MountingHole_3.2mm_M3", 44.5, 44.5),
    ("H2", "MountingHole", "MountingHole_3.2mm_M3", 185.5, 44.5),
    ("H3", "MountingHole", "MountingHole_3.2mm_M3", 44.5, 125.5),
    ("H4", "MountingHole", "MountingHole_3.2mm_M3", 185.5, 125.5),
    ("FID1", "Fiducial", "Fiducial_1mm_Mask2mm", 49.0, 50.0),
    ("FID2", "Fiducial", "Fiducial_1mm_Mask2mm", 169.5, 42.3),
    ("FID3", "Fiducial", "Fiducial_1mm_Mask2mm", 49.0, 121.0),
]



#: Driver-band relaxation: every pre-routed endpoint inside this window
#: moves east with the U2-U5 / decoupler cascade (see relax_driver_band).
BAND_X0, BAND_X1, BAND_Y0, BAND_Y1 = 155.0, 176.0, 41.0, 129.0
BAND_DX = 2.5


def band_shift(point):
    x, y = pcbnew.ToMM(point.x), pcbnew.ToMM(point.y)
    if BAND_X0 <= x <= BAND_X1 and BAND_Y0 <= y <= BAND_Y1:
        return vec(x + BAND_DX, y)
    return point


def zone_layer(zone, layer):
    """ZONE.SetLayer silently fails to move a zone off F.Cu in KiCad 9's
    Python API; SetLayerSet is what actually works."""
    layer_set = pcbnew.LSET()
    layer_set.AddLayer(layer)
    zone.SetLayerSet(layer_set)

# ---------------------------------------------------------------- helpers

def load_netlist():
    root = ET.parse(NETLIST).getroot()
    footprints = {}
    for comp in root.find(".//components"):
        node = comp.find("footprint")
        footprints[comp.get("ref")] = node.text if node is not None else None
    pin_nets, net_names = {}, []
    for net in root.find(".//nets"):
        net_names.append(net.get("name"))
        for node in net.findall("node"):
            pin_nets[(node.get("ref"), node.get("pin"))] = net.get("name")
    return footprints, pin_nets, net_names


def load_footprint(footprint_id):
    library, name = footprint_id.split(":")
    for base in (FP_PROJECT, FP_SYSTEM):
        path = os.path.join(base, library + ".pretty")
        if os.path.isdir(path):
            footprint = pcbnew.FootprintLoad(path, name)
            if footprint:
                return footprint
    raise RuntimeError(f"footprint {footprint_id} not found")


class Board:
    def __init__(self):
        footprints, self.pin_nets, net_names = load_netlist()
        self.footprint_ids = footprints
        self.board = pcbnew.BOARD()
        self.board.SetCopperLayerCount(4)
        self.board.GetDesignSettings().SetBoardThickness(mm(1.6))
        # 0.3 mm copper-to-edge is standard fab capability; the default 0.5
        # rejects legitimate routing in the outline margin.
        self.board.GetDesignSettings().m_CopperEdgeClearance = mm(0.3)
        self.board.GetDesignSettings().m_ViasMinSize = mm(0.4)
        self.board.GetDesignSettings().m_TrackMinWidth = mm(0.12)
        self.board.GetDesignSettings().m_ViasMinAnnularWidth = mm(0.07)
        self.board.GetDesignSettings().m_MinThroughDrill = mm(0.25)
        self.nets = {}
        for index, name in enumerate(sorted(net_names), start=1):
            net = pcbnew.NETINFO_ITEM(self.board, name, index)
            self.board.Add(net)
            self.nets[name] = net

    def net(self, name):
        if name not in self.nets:
            raise KeyError(f"net {name} not in netlist")
        return self.nets[name]

    def pad_position(self, reference, pad_number):
        for footprint in self.board.GetFootprints():
            if footprint.GetReference() == reference:
                for pad in footprint.Pads():
                    if pad.GetNumber() == str(pad_number):
                        position = pad.GetPosition()
                        return pcbnew.ToMM(position.x), pcbnew.ToMM(position.y)
        raise KeyError(f"{reference}.{pad_number} not found")

    def track(self, points, net_name, layer=pcbnew.F_Cu, width=SIGNAL_TRACK,
              locked=True):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
                continue
            track = pcbnew.PCB_TRACK(self.board)
            track.SetStart(vec(x1, y1))
            track.SetEnd(vec(x2, y2))
            track.SetNet(self.net(net_name))
            track.SetLayer(layer)
            track.SetWidth(mm(width))
            track.SetLocked(locked)
            self.board.Add(track)

    def via(self, x, y, net_name, locked=True, size=0.6, drill=0.3):
        via = pcbnew.PCB_VIA(self.board)
        via.SetPosition(vec(x, y))
        via.SetDrill(mm(drill))
        via.SetWidth(mm(size))
        via.SetNet(self.net(net_name))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        via.SetLocked(locked)
        self.board.Add(via)

    # ------------------------------------------------------------ sections

    def outline(self):
        points = [
            (BOARD_X0, BOARD_Y0), (BOARD_X1, BOARD_Y0),
            # USB-C tab on the right edge: the mating face sits flush with
            # the tab face at x = BOARD_X1 + 1.4.
            (BOARD_X1, 106.0), (BOARD_X1 + 1.4, 106.0),
            (BOARD_X1 + 1.4, 118.0), (BOARD_X1, 118.0),
            (BOARD_X1, BOARD_Y1),
            (BOARD_X0, BOARD_Y1),
        ]
        for start, end in zip(points, points[1:] + points[:1]):
            shape = pcbnew.PCB_SHAPE(self.board)
            shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
            shape.SetStart(vec(*start))
            shape.SetEnd(vec(*end))
            shape.SetLayer(pcbnew.Edge_Cuts)
            shape.SetWidth(mm(0.15))
            self.board.Add(shape)

    def place_components(self):
        missing = []
        for reference, footprint_id in sorted(self.footprint_ids.items()):
            if reference.startswith("#") or footprint_id is None:
                continue
            if reference not in PLACEMENT:
                missing.append(reference)
                continue
            footprint = load_footprint(footprint_id)
            footprint.SetReference(reference)
            self.board.Add(footprint)
            x, y, rotation = PLACEMENT[reference]
            footprint.SetPosition(vec(x, y))
            if rotation:
                footprint.SetOrientationDegrees(rotation)
            for pad in footprint.Pads():
                name = self.pin_nets.get((reference, pad.GetNumber()))
                if name:
                    pad.SetNet(self.net(name))
        if missing:
            raise RuntimeError(f"no placement for {missing}")
        for reference, library, name, x, y in MECHANICAL:
            footprint = load_footprint(f"{library}:{name}")
            footprint.SetReference(reference)
            self.board.Add(footprint)
            footprint.SetPosition(vec(x, y))
        # Strip the clamp footprints' silkscreen entirely: their wire-entry
        # arrows land on pads once rotated, and the bay silk says it better.
        for footprint in self.board.GetFootprints():
            if footprint.GetReference() not in ("J2", "J3", "J4", "J5"):
                continue
            for item in list(footprint.GraphicalItems()):
                if item.GetLayer() == pcbnew.F_SilkS:
                    footprint.Remove(item)
        # BNC footprint silk extends over the board edge (the connector
        # bodies do); drop those segments.
        for footprint in self.board.GetFootprints():
            if footprint.GetReference() not in ("J6", "J7", "J9"):
                continue
            for item in list(footprint.GraphicalItems()):
                if item.GetLayer() != pcbnew.F_SilkS:
                    continue
                box = item.GetBoundingBox()
                if pcbnew.ToMM(box.GetBottom()) > BOARD_Y1 - 0.4:
                    footprint.Remove(item)
        # Reference designators go to F.Fab: 56 refs on silk in a dense
        # layout guarantees overlaps, and assembly works from the fab layer.
        for footprint in self.board.GetFootprints():
            reference = footprint.Reference()
            reference.SetLayer(pcbnew.F_Fab)
            reference.SetTextSize(pcbnew.VECTOR2I(mm(0.7), mm(0.7)))
            reference.SetTextThickness(mm(0.11))

    def route_signal_nets(self):
        """Pre-route every measurement net, locked; freerouting never touches
        them.  Straight lines by construction of the placement."""
        # 1. DUT arms: clamp OUTER pad -> horizontal at the outer-pad row ->
        #    short drop onto the iso NC pad.  The outer-pad row clears the iso
        #    relay's coil and contact pads by construction; the run crosses
        #    only the iso body (traces under an SMD relay body are fine).
        for terminal, clamp, iso in (("A", "J2", "K15"), ("B", "J3", "K16"),
                                     ("C", "J4", "K17"), ("D", "J5", "K18")):
            net_name = f"/DUT_{terminal}"
            nc = self.pad_position(iso, 2)
            pad1 = self.pad_position(clamp, 1)
            pad2 = self.pad_position(clamp, 2)
            near = min((pad1, pad2), key=lambda p: abs(p[1] - nc[1]))
            self.track([pad1, pad2], net_name)
            # vertical to the NC row first, then straight across: this order
            # clears the iso relay's NO/COM pads and coil lands by geometry.
            self.track([near, (near[0], nc[1]), nc], net_name)

        # 2. Column buses: iso COM -> down past the head, across to the
        #    crossbar column, then the vertical with COM stubs.  Waypoints
        #    keep 0.65 mm+ from every staggered pad (see placement comments).
        bus_tap_rows = (ROW_HI + PAD_COM, ROW_LO + PAD_COM, ROW_LINK + PAD_COM)

        def column_bus(net_name, com, waypoints, bus_x):
            path = [com] + waypoints + [(bus_x, bus_tap_rows[0])]
            self.track(path, net_name)
            self.track([(bus_x, bus_tap_rows[0]), (bus_x, bus_tap_rows[-1])],
                       net_name)
            for row in bus_tap_rows:
                self.track([(bus_x, row), (bus_x + BUS_OFFSET, row)], net_name)

        com_a = self.pad_position("K15", 3)      # (73.5, 44.5)
        column_bus("/TA", com_a, [(74.9, com_a[1]), (74.9, 40.9),
                                  (52.5, 40.9)], 52.5)
        com_b = self.pad_position("K16", 3)      # (73.5, 56.1)
        column_bus("/TB", com_b, [(72.0, com_b[1]), (72.0, 63.5),
                                  (74.5, 63.5)], 74.5)
        com_c = self.pad_position("K17", 3)      # (124.5, 47.7)
        column_bus("/TC", com_c, [(126.0, com_c[1]), (126.0, 65.0),
                                  (118.5, 65.0)], 118.5)
        # TD must cross TC's westward run: hop under it on B.Cu (TC is F.Cu),
        # clear of the +5V flank horizontals at y 42.3 and 54.
        com_d = self.pad_position("K18", 3)      # (124.5, 59.4)
        self.track([com_d, (123.0, com_d[1]), (123.0, 63.0)], "/TD")
        self.via(123.0, 63.0, "/TD")
        self.track([(123.0, 63.0), (133.0, 63.0)], "/TD", layer=pcbnew.B_Cu)
        self.via(133.0, 63.0, "/TD")
        self.track([(133.0, 63.0), (140.5, 63.0),
                    (140.5, bus_tap_rows[0])], "/TD")
        self.track([(140.5, bus_tap_rows[0]), (140.5, bus_tap_rows[-1])], "/TD")
        for row in bus_tap_rows:
            self.track([(140.5, row), (140.5 + BUS_OFFSET, row)], "/TD")

        # 3. Load column: CAL_E west bus, CAL_F east bus (K14 rotated 180).
        column = COLUMNS["E"]
        r5_1 = self.pad_position("R5", 1)
        r5_2 = self.pad_position("R5", 2)
        k13_com = self.pad_position("K13", 3)
        k14_com = self.pad_position("K14", 3)
        west = column - BUS_OFFSET
        # CAL buses cross the descent verticals on B.Cu (y 103/104.5, north
        # of the AGND pour) to reach the relocated load standard.
        self.track([k13_com, (west, k13_com[1]), (west, 103.0)], "/CAL_E")
        self.via(west, 103.0, "/CAL_E")
        self.track([(west, 103.0), (133.0, 103.0)], "/CAL_E", layer=pcbnew.B_Cu)
        self.via(133.0, 103.0, "/CAL_E")
        self.track([(133.0, 103.0), (r5_1[0] - 1.7, 103.0),
                    (r5_1[0] - 1.7, r5_1[1]), r5_1], "/CAL_E")
        east = column + 10.5
        self.track([k14_com, (east, k14_com[1]), (east, 104.5)], "/CAL_F")
        self.via(east, 104.5, "/CAL_F")
        self.track([(east, 104.5), (135.0, 104.5)], "/CAL_F", layer=pcbnew.B_Cu)
        self.via(135.0, 104.5, "/CAL_F")
        self.track([(135.0, 104.5), (r5_2[0] + 1.0, 104.5),
                    (r5_2[0] + 1.0, r5_2[1]), r5_2], "/CAL_F")

        # 4. Rails on B.Cu, straight under the NO-pad rows; tap vias beside
        #    each NO pad (never in-pad).
        rail_span = (COLUMNS["A"] + 1.8, COLUMNS["D"] + 1.8)
        for rail, columns in (("HI", "ABECD"), ("LO", "ABCD"), ("LINK", "ABCD")):
            net_name = f"/RAIL_{rail}"
            y = RAIL_Y[rail]
            self.track([(rail_span[0], y), (rail_span[1], y)], net_name,
                       layer=pcbnew.B_Cu)
            for terminal in columns:
                if terminal == "E":
                    continue            # K13/K14 tapped separately below
                column = COLUMNS[terminal]
                pad = (column, y)
                via_x = column + 1.8
                self.track([pad, (via_x, y)], net_name)
                self.via(via_x, y, net_name)
        # K13 (HI row, unrotated): NO pad sits on the HI rail line.
        k13_no = self.pad_position("K13", 4)
        via_x = k13_no[0] + 1.8
        self.track([k13_no, (via_x, k13_no[1])], "/RAIL_HI")
        self.via(via_x, k13_no[1], "/RAIL_HI")
        # K14 (LO row, rotated 180): NO pad sits off the rail line. Take a
        # west F.Cu channel down to a via ON the rail -- keeping B.Cu free of
        # verticals the +5V branches would have to cross.
        k14_no = self.pad_position("K14", 4)
        channel_x = k14_no[0] - 1.8
        self.track([k14_no, (channel_x, k14_no[1]),
                    (channel_x, RAIL_Y["LO"])], "/RAIL_LO")
        self.via(channel_x, RAIL_Y["LO"], "/RAIL_LO")

        # 5. Descents drop from the rails to the bridge band on F.Cu
        #    (crossing the B.Cu rails freely); freerouting finishes the last
        #    stretch into the bridge resistors and the SOURCE pin.
        for x, rail in ((DESCENT_HI_X, "HI"), (DESCENT_LO_X, "LO")):
            net_name = f"/RAIL_{rail}"
            self.via(x, RAIL_Y[rail], net_name)
            self.track([(x, RAIL_Y[rail]), (x, 109.0)], net_name, width=1.0)
        # HI runs to the SOURCE BNC pin; LO wraps around R14's AGND pad
        # into the RC1 shunt pad from the east.
        source = self.pad_position("J6", 1)
        self.track([(DESCENT_HI_X, 109.0), (DESCENT_HI_X, 113.9),
                    (source[0], 113.9), source], "/RAIL_HI", width=1.0)
        shunt = self.pad_position("R14", 1)
        self.track([(DESCENT_LO_X, 109.0), (DESCENT_LO_X, 109.6),
                    (111.7, 109.6), (111.7, shunt[1])], "/RAIL_LO", width=0.6)
        self.track([(111.7, shunt[1]), shunt], "/RAIL_LO", width=0.6)
        # CH1 node: R12.2 and R13.1 onto a B.Cu run under the RAIL_HI bar
        # into the CH1 BNC through-hole.
        r12_2 = self.pad_position("R12", 2)
        r13_1 = self.pad_position("R13", 1)
        j7_1 = self.pad_position("J7", 1)
        self.track([r12_2, (70.7, r12_2[1])], "/CH1_NODE", width=PLAIN_TRACK)
        self.via(70.7, r12_2[1], "/CH1_NODE")
        self.track([(70.7, r12_2[1]), (70.7, 115.9), (j7_1[0], 115.9), j7_1],
                   "/CH1_NODE", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.track([r13_1, (79.2, r13_1[1])], "/CH1_NODE", width=PLAIN_TRACK)
        self.via(79.2, r13_1[1], "/CH1_NODE")
        self.track([(79.2, r13_1[1]), (79.2, 115.9)], "/CH1_NODE",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)

    def route_coil_supply(self):
        """+5V to every relay coil+, pre-routed on B.Cu.

        The inner +5V plane is deliberately voided under the signal region, so
        the coil supply must be a routed tree: a west spine, one branch per
        relay row, and a via beside each coil+ pad.  The LO-row branch is
        offset to 81.6 to clear the RAIL_LO tap vias at y=87.8 and K14's tap.
        """
        rows = {"HI": ROW_HI - 3.8, "LO": ROW_LO - 3.8, "LINK": ROW_LINK - 3.8}
        spine_x = 49.0
        self.track([(spine_x, rows["HI"]), (spine_x, rows["LINK"])], "/+5V",
                   layer=pcbnew.B_Cu)
        # HI and LO branches extend east into the plane region: stitch vias
        # plus deterministic feeds for the C11/C12 rail decouplers.
        # Branches run east under the drivers and feed each COM pin (9)
        # directly -- more robust than stitching into the plane fringe, whose
        # fill is at the mercy of clearance slivers.
        for name, driver in (("HI", "U2"), ("LO", "U3"), ("LINK", "U4")):
            com = self.pad_position(driver, 9)   # bottom-most east pad
            branch_y = rows[name]
            self.track([(spine_x, branch_y), (com[0], branch_y)], "/+5V",
                       layer=pcbnew.B_Cu)
            if branch_y > com[1]:
                # branch below pin 9: via right on the branch, straight stub
                self.via(com[0], branch_y, "/+5V")
                self.track([(com[0], branch_y), com], "/+5V", width=PLAIN_TRACK)
            else:
                # branch above pin 9 (a stub up would cross pin 10): come in
                # from below through a via and a B.Cu jog back to the branch
                below = com[1] + 1.6
                self.track([com, (com[0], below)], "/+5V", width=PLAIN_TRACK)
                self.via(com[0], below, "/+5V")
                self.track([(com[0], below), (com[0], branch_y)], "/+5V",
                           layer=pcbnew.B_Cu)

        # Crossbar + load-column coils: via beside each coil+ pad, dropped
        # onto the row branch.
        matrix_coils = {
            "K1": "HI", "K4": "HI", "K13": "HI", "K7": "HI", "K10": "HI",
            "K2": "LO", "K5": "LO", "K8": "LO", "K11": "LO",
            "K3": "LINK", "K6": "LINK", "K9": "LINK", "K12": "LINK",
        }
        for relay, row_name in matrix_coils.items():
            pad = self.pad_position(relay, 1)
            branch_y = rows[row_name]
            via_x = pad[0] + 2.0
            self.track([pad, (via_x, pad[1]), (via_x, branch_y)], "/+5V")
            self.via(via_x, branch_y, "/+5V")
        # K14 (rotated): coil+ sits on the LO rail line; dogleg east and join
        # the LO branch on B.Cu, clear of the CAL_F bus and the rail tap.
        pad = self.pad_position("K14", 1)
        self.track([pad, (pad[0] + 2.0, pad[1]), (pad[0] + 2.0, pad[1] - 1.6)],
                   "/+5V")
        self.via(pad[0] + 2.0, pad[1] - 1.6, "/+5V")
        self.track([(pad[0] + 2.0, pad[1] - 1.6), (pad[0] + 2.0, rows["LO"])],
                   "/+5V", layer=pcbnew.B_Cu)

        # K14's coil return: freerouting has failed this route in every
        # session (E column to U3 across the congested strip), so it is
        # locked: via beside the pad, B.Cu east, entering pad 10 from the
        # east clear of pin 9.
        k14_coil = self.pad_position("K14", 8)
        self.track([k14_coil, (k14_coil[0], 90.2)], "/K14_COIL",
                   width=PLAIN_TRACK)
        self.via(k14_coil[0], 90.2, "/K14_COIL")
        u3_out = self.pad_position("U3", 10)
        self.track([(k14_coil[0], 90.2), (162.0, 90.2), (162.0, u3_out[1])],
                   "/K14_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(162.0, u3_out[1], "/K14_COIL", size=0.4, drill=0.25)
        self.track([(162.0, u3_out[1]), u3_out], "/K14_COIL",
                   width=PLAIN_TRACK)

        # K13 coil return: corridor between the +5V HI and LO branches,
        # entering U3.11 from the east past the end of the LO branch.
        k13_coil = self.pad_position("K13", 8)
        self.track([k13_coil, (k13_coil[0], 69.9)], "/K13_COIL",
                   width=PLAIN_TRACK)
        self.via(k13_coil[0], 69.9, "/K13_COIL")
        u3_11 = self.pad_position("U3", 11)
        self.track([(k13_coil[0], 69.9), (161.5, 69.9), (161.5, u3_11[1])],
                   "/K13_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(161.5, u3_11[1], "/K13_COIL", size=0.4, drill=0.25)
        self.track([(161.5, u3_11[1]), u3_11], "/K13_COIL", width=PLAIN_TRACK)

        # K4/K5/K6 (B column) coil returns -- freerouting fails a different
        # subset every run, so all coil returns are locked. Each has its own
        # corridor y and rise column; the collision matrix against the +5V
        # branches, driver COM jogs and the other coil runs is annotated in
        # the coordinates.
        for relay, stub_y, corridor_y, out_pin, rise_x in (
                ("K4", 69.15, 69.15, 13, 161.9),  # above K13's 69.9 lane
                ("K6", 91.2, 91.2, 11, 162.65)):  # between K14 lane + K16 rise
            coil = self.pad_position(relay, 8)
            net_name = f"/{relay}_COIL"
            self.track([coil, (coil[0], stub_y)], net_name, width=PLAIN_TRACK)
            self.via(coil[0], stub_y, net_name)
            out = self.pad_position("U2", out_pin)
            self.track([(coil[0], stub_y), (coil[0], corridor_y),
                        (rise_x, corridor_y), (rise_x, out[1])],
                       net_name, layer=pcbnew.B_Cu, width=PLAIN_TRACK)
            self.via(rise_x, out[1], net_name, size=0.4, drill=0.25)
            self.track([(rise_x, out[1]), out], net_name, width=PLAIN_TRACK)

        # K5: B.Cu corridor at y 86 (parallel to the branches), F.Cu hop
        # over the K14/K6 rise verticals, B.Cu rise at 163.6.
        k5_coil = self.pad_position("K5", 8)
        self.track([k5_coil, (85.9, k5_coil[1]), (85.9, 86.0)],
                   "/K5_COIL", width=PLAIN_TRACK)
        self.via(85.9, 86.0, "/K5_COIL", size=0.4, drill=0.25)
        self.track([(85.9, 86.0), (107.0, 86.0), (107.0, 86.85),
                    (109.5, 86.85), (109.5, 86.0), (156.2, 86.0)],
                   "/K5_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(156.2, 86.0, "/K5_COIL", size=0.4, drill=0.25)
        self.track([(156.2, 86.0), (159.3, 86.0)], "/K5_COIL",
                   width=0.15)
        self.via(159.3, 86.0, "/K5_COIL", size=0.4, drill=0.25)
        self.track([(159.3, 86.0), (161.4, 86.0)], "/K5_COIL",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(161.4, 86.0, "/K5_COIL", size=0.4, drill=0.25)
        self.track([(161.4, 86.0), (163.3, 86.0)], "/K5_COIL",
                   width=0.15)
        self.via(163.3, 86.0, "/K5_COIL", size=0.4, drill=0.25)
        self.track([(163.3, 86.0), (163.6, 86.0), (163.6, 62.635)],
                   "/K5_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        u2_12 = self.pad_position("U2", 12)
        self.via(163.6, u2_12[1], "/K5_COIL", size=0.4, drill=0.25)
        self.track([(163.6, u2_12[1]), u2_12], "/K5_COIL", width=PLAIN_TRACK)

        # K10/K11: B.Cu corridors between the +5V branches, west-side
        # entry vias under the U3 pad row.
        for relay, stub_y, corridor_y, out_pin in (
                ("K10", 70.6, 70.6, 14),
                ("K11", 79.3, 79.3, 13)):
            coil = self.pad_position(relay, 8)
            out = self.pad_position("U3", out_pin)
            net_name = f"/{relay}_COIL"
            self.track([coil, (coil[0], stub_y)], net_name, width=PLAIN_TRACK)
            self.via(coil[0], stub_y, net_name, size=0.4, drill=0.25)
            self.track([(coil[0], corridor_y), (159.0, corridor_y),
                        (159.0, out[1])], net_name,
                       layer=pcbnew.B_Cu, width=PLAIN_TRACK)
            self.via(159.0, out[1], net_name, size=0.4, drill=0.25)
            self.track([(159.0, out[1]), out], net_name, width=PLAIN_TRACK)

        # K12: B.Cu to x 158.4, then an In1 hop across the LO branch line
        # into U3.12 from under the body.
        k12_coil = self.pad_position("K12", 8)
        u3_12 = self.pad_position("U3", 12)
        self.track([k12_coil, (k12_coil[0], 89.4)], "/K12_COIL",
                   width=PLAIN_TRACK)
        self.via(k12_coil[0], 89.4, "/K12_COIL", size=0.4, drill=0.25)
        self.track([(k12_coil[0], 89.4), (158.45, 89.4), (158.45, 84.6)],
                   "/K12_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(158.45, 84.6, "/K12_COIL", size=0.4, drill=0.25)
        self.track([(158.45, 84.6), (158.45, u3_12[1])], "/K12_COIL",
                   width=PLAIN_TRACK)
        self.via(158.45, u3_12[1], "/K12_COIL", size=0.4, drill=0.25)
        self.track([(158.45, u3_12[1]), u3_12], "/K12_COIL",
                   width=PLAIN_TRACK)

        # K15/K16 coil returns: like K14, freerouting never completes the
        # left-flank-to-U4 runs; locked here. West F.Cu drop at x 63/63.8,
        # B.Cu corridor south of the CAL crossings, east rise into U4.
        for relay, drop_x, corridor_y, out_pin, rise_x in (
                ("K15", 64.6, 107.8, 16, 164.0),
                ("K16", 65.4, 106.9, 15, 163.2)):
            coil = self.pad_position(relay, 8)
            net_name = f"/{relay}_COIL"
            self.track([coil, (drop_x, coil[1]), (drop_x, corridor_y)],
                       net_name, width=PLAIN_TRACK)
            self.via(drop_x, corridor_y, net_name)
            self.track([(drop_x, corridor_y), (rise_x, corridor_y)],
                       net_name, layer=pcbnew.B_Cu, width=PLAIN_TRACK)
            out = self.pad_position("U4", out_pin)
            self.track([(rise_x, corridor_y), (rise_x, out[1])],
                       net_name, layer=pcbnew.B_Cu, width=PLAIN_TRACK)
            self.via(rise_x, out[1], net_name, size=0.4, drill=0.25)
            self.track([(rise_x, out[1]), out], net_name, width=PLAIN_TRACK)

        # Isolation relay coils. Left flank: stubs east onto one B.Cu
        # vertical down to the HI branch. Right flank: stubs west onto B.Cu
        # horizontals running east into the plane region (a vertical there
        # would cross the TD bus hop).
        for relay in ("K15", "K16"):
            pad = self.pad_position(relay, 1)
            self.track([pad, (76.7, pad[1])], "/+5V")
            self.via(76.7, pad[1], "/+5V")
        self.track([(76.7, self.pad_position("K15", 1)[1]),
                    (76.7, rows["HI"])], "/+5V", layer=pcbnew.B_Cu)
        # Right flank: shared B.Cu vertical at x=121.9 (west of the TD bus
        # hop via at x=123) dropping to the HI branch.
        for relay in ("K17", "K18"):
            pad = self.pad_position(relay, 1)
            self.track([pad, (121.9, pad[1])], "/+5V")
            self.via(121.9, pad[1], "/+5V")
        self.track([(121.9, self.pad_position("K17", 1)[1]),
                    (121.9, rows["HI"])], "/+5V", layer=pcbnew.B_Cu)

    PLANE_BOUNDS = (152.3, 41.5, 176.5, 128.5)

    def fanout_plane_pads(self):
        """Locked stub+via for every control-strip SMD pad on GND or +5V.

        Done at GENERATE time, when the only copper is the deterministic
        pre-routes -- so placement is collision-free by construction, and
        freerouting (whose respect for fixed-wire clearance is unreliable)
        never has to touch the plane nets at all.
        """
        x0, y0, x1, y1 = self.PLANE_BOUNDS
        candidates = ((0, 1.4), (0, -1.4), (1.4, 0), (-1.4, 0),
                      (1.1, 1.1), (-1.1, -1.1), (1.1, -1.1), (-1.1, 1.1),
                      (0, 2.2), (0, -2.2), (2.2, 0), (-2.2, 0))
        existing = list(self.board.GetTracks())

        def blocked(x, y, net_code):
            position = vec(x, y)
            for item in existing:
                if item.GetNetCode() != net_code and item.HitTest(position, mm(0.65)):
                    return True
            for footprint in self.board.GetFootprints():
                for pad in footprint.Pads():
                    if pad.GetNetCode() != net_code and pad.HitTest(position, mm(0.55)):
                        return True
            return False

        added, skipped = 0, []
        for footprint in self.board.GetFootprints():
            for pad in footprint.Pads():
                net_name = pad.GetNetname()
                if net_name not in ("/GND", "/+5V"):
                    continue
                if pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                    continue
                position = pad.GetPosition()
                px, py = pcbnew.ToMM(position.x), pcbnew.ToMM(position.y)
                if px < 150.0:
                    continue                    # signal region: pre-routed
                near_via = any(t.GetClass() == "PCB_VIA"
                               and t.GetNetCode() == pad.GetNetCode()
                               and abs(pcbnew.ToMM(t.GetPosition().x) - px) < 2.0
                               and abs(pcbnew.ToMM(t.GetPosition().y) - py) < 2.0
                               for t in existing)
                if near_via:
                    continue
                for dx, dy in candidates:
                    x, y = px + dx, py + dy
                    if not (x0 < x < x1 and y0 < y < y1):
                        continue
                    if blocked(x, y, pad.GetNetCode()):
                        continue
                    self.track([(px, py), (x, py) if dy == 0 else (px, y),
                                (x, y)], net_name, width=PLAIN_TRACK)
                    self.via(x, y, net_name)
                    existing.extend(list(self.board.GetTracks())[-3:])
                    added += 1
                    break
                else:
                    skipped.append(f"{footprint.GetReference()}.{pad.GetNumber()}")
        print(f"  plane fanout: {added} vias" +
              (f", NO ROOM: {skipped}" if skipped else ""))

    def route_strip_grounds(self):
        """Locked stub+via for the GND pads freerouting reliably leaves
        stranded (it believes plane nets are already connected), placed where
        the safety fanout found no room."""
        # driver U4 corner
        u4_gnd = self.pad_position("U4", 8)
        self.track([u4_gnd, (u4_gnd[0], 98.0)], "/GND", width=PLAIN_TRACK)
        self.via(u4_gnd[0], 98.0, "/GND")
        # USB right-edge corner. Row pads sit at x=183.705 in y-order
        # GND VBUS CC2 SBU DP DM DP DM CC1 SBU VBUS GND (0.5 mm pitch).
        # Duplicate pads tie east of the row (vias behind it, inside the
        # usb_field rule area); primaries run west to the ESD array.
        dp_a = self.pad_position("J1", "A6")
        dp_b = self.pad_position("J1", "B6")
        dm_a = self.pad_position("J1", "A7")
        dm_b = self.pad_position("J1", "B7")
        self.track([dp_a, (182.55, dp_a[1]), (182.55, dp_b[1]), dp_b],
                   "/USB_DP_CON", width=0.15)
        self.track([dm_a, (184.75, dm_a[1])], "/USB_DM_CON", width=0.15)
        self.via(184.75, dm_a[1], "/USB_DM_CON", size=0.4, drill=0.25)
        self.track([(184.75, dm_a[1]), (184.75, 113.4), (181.6, 113.4),
                    (181.6, dm_b[1])], "/USB_DM_CON",
                   layer=pcbnew.B_Cu, width=0.15)
        self.via(181.6, dm_b[1], "/USB_DM_CON", size=0.4, drill=0.25)
        # primaries into the ESD array (D2 rot 180: pads 1/3 on the east)
        d2_dm_con = self.pad_position("D2", 1)   # (179.64, 112.95)
        d2_dp_con = self.pad_position("D2", 3)   # (179.64, 111.05)
        self.track([dp_b, (181.0, dp_b[1]), (180.3, d2_dp_con[1]),
                    d2_dp_con], "/USB_DP_CON", width=PLAIN_TRACK)
        self.track([dm_b, (181.0, dm_b[1]), (180.3, d2_dm_con[1]),
                    d2_dm_con], "/USB_DM_CON", width=PLAIN_TRACK)
        # MCU-side pair, hopping to B.Cu across the GPIO14 / NRST F.Cu
        # verticals (x 167.1-167.9), back to F.Cu for the pad entries.
        d2_dp = self.pad_position("D2", 4)       # (177.36, 111.05)
        d2_dm = self.pad_position("D2", 6)       # (177.36, 112.95)
        pa12 = self.pad_position("U1", "33")     # USB D+
        pa11 = self.pad_position("U1", "32")     # USB D-
        self.track([d2_dp, (168.9, d2_dp[1])], "/USB_DP", width=PLAIN_TRACK)
        self.via(168.9, d2_dp[1], "/USB_DP", size=0.4, drill=0.25)
        self.track([(168.9, d2_dp[1]), (166.35, d2_dp[1]),
                    (165.75, pa12[1])], "/USB_DP",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(165.75, pa12[1], "/USB_DP", size=0.4, drill=0.25)
        self.track([(165.75, pa12[1]), pa12], "/USB_DP", width=PLAIN_TRACK)
        self.track([d2_dm, (169.3, d2_dm[1]), (169.0, 112.75)], "/USB_DM",
                   width=PLAIN_TRACK)
        self.via(169.0, 112.75, "/USB_DM", size=0.4, drill=0.25)
        self.track([(169.0, 112.75), (166.4, 112.75), (165.75, 111.55)],
                   "/USB_DM", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(165.75, 111.55, "/USB_DM", size=0.4, drill=0.25)
        self.track([(165.75, 111.55), (165.45, pa11[1]), pa11],
                   "/USB_DM", width=PLAIN_TRACK)
        # VBUS: east stubs behind the row, B.Cu to a vertical at x=180.9,
        # spur to the ESD array, artery north along the east edge to R9.
        a4 = self.pad_position("J1", "A4")
        b4 = self.pad_position("J1", "B4")
        for pad in (a4, b4):
            self.track([pad, (184.4, pad[1])], "/VBUS", width=0.2)
            self.via(184.4, pad[1], "/VBUS", size=0.4, drill=0.25)
            self.track([(184.4, pad[1]), (180.9, pad[1])], "/VBUS",
                       layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.track([(180.9, a4[1]), (180.9, b4[1])], "/VBUS",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        d2_vbus = self.pad_position("D2", 5)
        self.track([(180.9, 110.4), (177.3, 110.4), (177.3, d2_vbus[1])],
                   "/VBUS", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(177.3, d2_vbus[1], "/VBUS", size=0.4, drill=0.25)
        self.track([(177.3, d2_vbus[1]), d2_vbus], "/VBUS",
                   width=PLAIN_TRACK)
        r9_1 = self.pad_position("R9", 1)
        self.track([r9_1, (r9_1[0], 46.95)], "/VBUS", width=PLAIN_TRACK)
        self.via(r9_1[0], 46.95, "/VBUS")
        self.track([(r9_1[0], 46.95), (174.0, 46.95)], "/VBUS",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(174.0, 46.95, "/VBUS")
        self.track([(174.0, 46.95), (180.9, 46.95)], "/VBUS",
                   width=PLAIN_TRACK)
        self.via(180.9, 46.95, "/VBUS")
        self.track([(180.9, 46.95), (180.9, b4[1])],
                   "/VBUS", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        # CC pulldowns just west of the connector
        cc1_j = self.pad_position("J1", "A5")
        cc1_r = self.pad_position("R1", 1)
        self.track([cc1_j, (182.2, cc1_j[1]), (182.2, cc1_r[1]), cc1_r],
                   "/CC1", width=PLAIN_TRACK)
        cc2_j = self.pad_position("J1", "B5")
        cc2_r = self.pad_position("R2", 1)
        self.track([cc2_j, (182.2, cc2_j[1]), (182.2, cc2_r[1]), cc2_r],
                   "/CC2", width=PLAIN_TRACK)

        # MCU ground pins: locked stubs (the area under U1 is the GPIO
        # escape field; the safety fanout finds no room there).
        for pin, dx, dy in (("8", -1.5, 0),
                            ("35", -1.8625, 0), ("47", 0, -1.6)):
            pad = self.pad_position("U1", pin)
            x, y = pad[0] + dx, pad[1] + dy
            self.track([pad, (x, y)], "/GND", width=PLAIN_TRACK)
            self.via(x, y, "/GND", size=0.4, drill=0.25)
        # pin 23: small via clear of pads 22/24 and the GPIO13 entry
        pad23 = self.pad_position("U1", "23")
        self.track([(pad23[0], 116.9), (162.45, 117.5)], "/GND",
                   width=0.15)
        self.via(162.45, 117.5, "/GND", size=0.4, drill=0.25)
        # U3 ground pad: the coil corridors and the GPIO14 In1 transplant
        # box out the safety fanout -- stub south to a via in the gap
        # between the +5V return (y=80.2) and the K5 corridor (y=86.0)
        u3_8 = self.pad_position("U3", 8)
        self.track([u3_8, (u3_8[0], 83.5), (154.5, 84.525), (154.5, 85.3)],
                   "/GND", width=PLAIN_TRACK)
        self.via(154.5, 85.3, "/GND")
        # pin 24 (+3V3): freerouting reaches the area on B.Cu but never
        # lands the pad -- give it a landing via
        pad24 = self.pad_position("U1", "24")
        self.track([(pad24[0], 116.9), (163.35, 117.9)], "/+3V3",
                   width=0.15)
        self.via(163.35, 117.9, "/+3V3", size=0.4, drill=0.25)

        # Driver decouplers: compact vertical via pairs (supply via south
        # of pad 1 in the pad line, ground via in its own column west) plus
        # a B.Cu spine linking the supply vias -- the In1 plane is moated
        # here by the freerouted via field.
        supply_ys = []
        for cap in ("C5", "C11", "C6", "C12", "C7"):
            column_cap = cap != "C5"
            supply = self.pad_position(cap, 1)
            ground = self.pad_position(cap, 2)
            self.track([supply, (supply[0], supply[1] + 0.9)], "/+5V",
                       width=PLAIN_TRACK)
            self.via(supply[0], supply[1] + 0.9, "/+5V",
                     size=0.4, drill=0.25)
            if column_cap:
                supply_ys.append(supply[1] + 0.9)
            if cap == "C11":
                gnd_via_y = ground[1] + 1.93   # south side, under pad 1
            elif cap == "C6":
                gnd_via_y = ground[1] + 1.38   # south side: GPIO8 band above
            else:
                gnd_via_y = ground[1] - 1.32
            if cap in ("C6", "C11"):
                jog_x = ground[0] - 0.65
                self.track([ground, (jog_x, ground[1]),
                            (jog_x, gnd_via_y)], "/GND", width=PLAIN_TRACK)
                self.via(jog_x, gnd_via_y, "/GND", size=0.4, drill=0.25)
            else:
                self.track([ground, (ground[0] + 0.6, gnd_via_y)], "/GND",
                           width=PLAIN_TRACK)
                self.via(ground[0] + 0.6, gnd_via_y, "/GND",
                         size=0.4, drill=0.25)
            if column_cap:
                spine_x = supply[0]
        self.track([(spine_x, min(supply_ys)), (spine_x, max(supply_ys))],
                   "/+5V", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        c9_gnd = self.pad_position("C9", 2)
        self.track([c9_gnd, (168.3, c9_gnd[1]), (168.3, 117.4)], "/GND",
                   width=PLAIN_TRACK)
        self.via(168.3, 117.4, "/GND", size=0.4, drill=0.25)
        c8_gnd = self.pad_position("C8", 2)
        self.track([c8_gnd, (156.6, 119.15)], "/GND", width=PLAIN_TRACK)
        self.via(156.6, 119.15, "/GND", size=0.4, drill=0.25)

        # GPIO2/3/9/10/13 and NRST: freerouting fails a different subset
        # of these every round -- the MCU approach corridors are saturated.
        # Deterministic lanes, planar by construction: F.Cu entries never
        # conflict with B.Cu lanes.

        # GPIO9/10: west stub off U3 (pads are 1.95 long -- stay west of
        # x 154.55 edge), up into the U2/U3 gap band at y 71.1/71.6 (clear
        # of the K4/K13 coil via rings), east on F.Cu, B.Cu downcomers at
        # 169.6/170.1, west along B.Cu lanes 118.5/119.15, F entries.
        for u3_pin, u1_pin, west_x, gap_y, east_x, lane_y in (
                (2, 18, 154.15, 71.7, 169.6, 118.5),
                (3, 19, 153.65, 70.9, 170.35, 119.15)):
            u3_pad = self.pad_position("U3", u3_pin)
            u1_pad = self.pad_position("U1", u1_pin)
            net_name = self.board.FindFootprintByReference("U3")                            .FindPadByNumber(str(u3_pin)).GetNetname()
            if u1_pin == 18:
                # via jogged south-east so the GPIO1 downcomer clears it
                self.track([u3_pad, (west_x, u3_pad[1]), (west_x, gap_y),
                            (169.6, gap_y), (169.8, 72.1), (169.8, 72.5)],
                           net_name, width=PLAIN_TRACK)
                self.via(169.8, 72.5, net_name, size=0.4, drill=0.25)
            else:
                self.track([u3_pad, (west_x, u3_pad[1]), (west_x, gap_y),
                            (east_x, gap_y)], net_name, width=PLAIN_TRACK)
                self.via(east_x, gap_y, net_name, size=0.4, drill=0.25)
            if u1_pin == 18:
                # entry via jogged west so the pad-19 entry line stays clear
                self.track([(169.8, 72.5), (169.8, lane_y),
                            (159.3, lane_y)], net_name,
                           layer=pcbnew.B_Cu, width=PLAIN_TRACK)
                self.via(159.3, lane_y, net_name, size=0.4, drill=0.25)
                self.track([(159.3, lane_y), (159.3, 117.6),
                            (u1_pad[0], 117.2), u1_pad], net_name,
                           width=PLAIN_TRACK)
            else:
                self.track([(east_x, gap_y), (east_x, lane_y),
                            (u1_pad[0], lane_y)], net_name,
                           layer=pcbnew.B_Cu, width=PLAIN_TRACK)
                self.via(u1_pad[0], lane_y, net_name, size=0.4, drill=0.25)
                self.track([(u1_pad[0], lane_y), u1_pad], net_name,
                           width=PLAIN_TRACK)

        # GPIO13 -> U1.22, all F.Cu: down the west strip at x 153.55
        # (R11 moved west to free this), east along y 119.7 between the
        # C8 pads and the CC1 lane, straight up into pad 22.
        u3_6 = self.pad_position("U3", 6)
        pad22 = self.pad_position("U1", 22)
        self.track([u3_6, (153.55, u3_6[1]), (153.55, 119.7),
                    (pad22[0], 119.7), pad22], "/GPIO13",
                   width=PLAIN_TRACK)

        # GPIO2/3 -> U1.11/12 (west row): west stubs off U2, down F.Cu at
        # x 151.3/151.9 (east of the crossbar rails, west of the MCU strip
        # parts), B.Cu hops under the GPIO13 descent, west-row entries.
        u2_2 = self.pad_position("U2", 2)
        pad11 = self.pad_position("U1", 11)
        self.track([u2_2, (151.7, u2_2[1]), (151.7, 113.95)], "/GPIO2",
                   width=PLAIN_TRACK)
        self.via(151.7, 113.95, "/GPIO2", size=0.4, drill=0.25)
        self.track([(151.7, 113.95), (154.6, 113.95)], "/GPIO2",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(154.6, 113.95, "/GPIO2", size=0.4, drill=0.25)
        self.track([(154.6, 113.95), (154.9, 114.25), pad11], "/GPIO2",
                   width=PLAIN_TRACK)
        u2_3 = self.pad_position("U2", 3)
        pad12 = self.pad_position("U1", 12)
        self.track([u2_3, (152.3, u2_3[1]), (152.3, 114.75)], "/GPIO3",
                   width=PLAIN_TRACK)
        self.via(152.3, 114.75, "/GPIO3", size=0.4, drill=0.25)
        self.track([(152.3, 114.75), (154.35, 114.75)], "/GPIO3",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(154.35, 114.75, "/GPIO3", size=0.4, drill=0.25)
        self.track([(154.35, 114.75), pad12], "/GPIO3", width=PLAIN_TRACK)

        # GPIO1 -> U1.10 (west row): out over the top of U2, east
        # downcomer, B.Cu lane at 114.95 under the pad row, entry into
        # pad 10 from the east under the body.
        u2_1 = self.pad_position("U2", 1)
        pad10 = self.pad_position("U1", 10)
        self.track([u2_1, (154.9, u2_1[1]), (154.9, 55.6), (164.45, 55.6)],
                   "/GPIO1", width=PLAIN_TRACK)
        self.via(164.45, 55.6, "/GPIO1", size=0.4, drill=0.25)
        self.track([(164.45, 55.6), (164.55, 56.2), (164.55, 109.15),
                    (157.45, 109.15),
                    (157.45, 113.55), (157.0, 113.75)],
                   "/GPIO1", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(157.0, 113.75, "/GPIO1", size=0.4, drill=0.25)
        self.track([(157.0, 113.75), (156.4, 113.75)], "/GPIO1",
                   width=0.15)

        # GPIO11 -> U1.20: through the U3 pin-4/5 row gap, west strip at
        # 153.25, jog west of the GPIO13 descent, B.Cu lane at 117.9,
        # south entry between the lane-9 and lane-10 vias.
        u3_4 = self.pad_position("U3", 4)
        pad20 = self.pad_position("U1", 20)
        self.track([u3_4, (152.85, u3_4[1]), (152.85, 117.9)], "/GPIO11",
                   width=PLAIN_TRACK)
        self.via(152.85, 117.9, "/GPIO11", size=0.4, drill=0.25)
        self.track([(152.85, 117.9), (160.95, 117.9)], "/GPIO11",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(160.95, 117.9, "/GPIO11", size=0.4, drill=0.25)
        self.track([(160.95, 117.9), (pad20[0], 117.2), pad20], "/GPIO11",
                   width=0.15)

        # SWDIO -> U1.34: B.Cu drop from the J8 through-hole down the
        # freeway (NRST now crosses on F.Cu, so nothing bars the way),
        # via on the freed pin-35 stub line, diagonal into pad 34.
        j8_2 = self.pad_position("J8", 2)
        self.track([j8_2, (165.0, j8_2[1]), (165.0, 109.7),
                    (165.35, 110.05)], "/SWDIO",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(165.35, 110.05, "/SWDIO", size=0.4, drill=0.25)
        self.track([(165.35, 110.05), (164.6, 110.25)], "/SWDIO",
                   width=0.13)

        # GPIO4/5/17/18: far-east descent fan. Ordering rule: the
        # westmost descent carries the shallowest transfer lane, so the
        # eastern descents' transfers pass below their neighbours' ends.
        # F.Cu hops carry each transfer over the GPIO9/10 downcomers and
        # (for GPIO5) over the GPIO17/18 entry verticals.
        u4_4 = self.pad_position("U4", 4)
        pad29 = self.pad_position("U1", 29)
        self.track([u4_4, (154.85, u4_4[1]), (154.85, 92.0),
                    (171.9, 92.0)], "/GPIO18", width=PLAIN_TRACK)
        self.via(171.9, 92.0, "/GPIO18", size=0.4, drill=0.25)
        self.track([(171.9, 92.0), (171.9, 115.9), (171.15, 115.9)],
                   "/GPIO18", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(171.15, 115.9, "/GPIO18", size=0.4, drill=0.25)
        self.track([(171.15, 115.9), (169.25, 115.9)], "/GPIO18",
                   width=0.15)
        self.via(169.25, 115.9, "/GPIO18", size=0.4, drill=0.25)
        self.track([(169.25, 115.9), (162.9, 115.9), (162.9, 112.75)],
                   "/GPIO18", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(162.9, 112.75, "/GPIO18", size=0.4, drill=0.25)
        self.track([(162.9, 112.75), (163.425, 112.75), pad29], "/GPIO18",
                   width=0.15)

        u4_3 = self.pad_position("U4", 3)
        pad28 = self.pad_position("U1", 28)
        self.track([u4_3, (154.85, u4_3[1]), (154.85, 90.73),
                    (172.5, 90.73)], "/GPIO17", width=PLAIN_TRACK)
        self.via(172.5, 90.73, "/GPIO17", size=0.4, drill=0.25)
        self.track([(172.5, 90.73), (172.5, 116.55), (170.9, 116.55)],
                   "/GPIO17", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(170.9, 116.55, "/GPIO17", size=0.4, drill=0.25)
        self.track([(170.9, 116.55), (169.05, 116.55)], "/GPIO17",
                   width=0.15)
        self.via(169.05, 116.55, "/GPIO17", size=0.4, drill=0.25)
        self.track([(169.05, 116.55), (162.3, 116.55), (162.3, 113.25)],
                   "/GPIO17", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(162.3, 113.25, "/GPIO17", size=0.4, drill=0.25)
        self.track([(162.3, 113.25), (163.425, 113.25), pad28], "/GPIO17",
                   width=0.15)

        u2_4 = self.pad_position("U2", 4)
        pad13 = self.pad_position("U1", 13)
        self.track([u2_4, (154.85, u2_4[1]), (154.85, 62.0),
                    (173.7, 62.0)], "/GPIO4", width=PLAIN_TRACK)
        self.via(173.7, 62.0, "/GPIO4", size=0.4, drill=0.25)
        self.track([(173.7, 62.0), (173.7, 120.15), (157.15, 120.15),
                    (157.15, 118.6)], "/GPIO4",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(157.15, 118.6, "/GPIO4", size=0.4, drill=0.25)
        self.track([(157.15, 118.6), (pad13[0], 117.6), (pad13[0], 116.9),
                    pad13], "/GPIO4", width=0.15)

        u2_5 = self.pad_position("U2", 5)
        pad14 = self.pad_position("U1", 14)
        self.track([u2_5, (154.85, u2_5[1]), (154.85, 63.27),
                    (173.1, 63.27)], "/GPIO5", width=PLAIN_TRACK)
        self.via(173.1, 63.27, "/GPIO5", size=0.4, drill=0.25)
        self.track([(173.1, 63.27), (173.1, 119.7), (157.75, 119.7),
                    (157.75, 118.6)], "/GPIO5",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(157.75, 118.6, "/GPIO5", size=0.4, drill=0.25)
        self.track([(157.75, 118.6), (pad14[0], 116.9), pad14], "/GPIO5",
                   width=0.15)

        # LED_A -> U1.2: B.Cu west above the clamp row, F.Cu down the
        # bay channel at x 89.5, B.Cu east under the MCU at y 105.9,
        # F.Cu into the east end of pad 2.
        d1_a = self.pad_position("D1", 2)
        u1_2 = self.pad_position("U1", 2)
        self.track([d1_a, (d1_a[0], 52.2), (151.85, 52.2), (151.15, 53.0)],
                   "/LED_A", width=PLAIN_TRACK)
        self.via(151.15, 53.0, "/LED_A", size=0.4, drill=0.25)
        self.track([(151.15, 53.0), (150.5, 51.1), (122.7, 51.1)],
                   "/LED_A", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(122.7, 51.1, "/LED_A", size=0.4, drill=0.25)
        self.track([(122.7, 51.1), (121.1, 51.1)], "/LED_A",
                   width=0.15)
        self.via(121.1, 51.1, "/LED_A", size=0.4, drill=0.25)
        self.track([(121.1, 51.1), (89.5, 51.1)],
                   "/LED_A", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(89.5, 51.1, "/LED_A", size=0.4, drill=0.25)
        self.track([(89.5, 51.1), (89.5, 105.9)], "/LED_A",
                   width=PLAIN_TRACK)
        self.via(89.5, 105.9, "/LED_A", size=0.4, drill=0.25)
        self.track([(89.5, 105.9), (156.8375, 105.9)], "/LED_A",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(156.8375, 105.9, "/LED_A", size=0.4, drill=0.25)
        self.track([(156.8375, 105.9), (156.8375, u1_2[1]),
                    (156.575, u1_2[1])], "/LED_A", width=0.12)

        # GPIO6/7/8 -> U1.15/16/17: row-gap bands east (with local jogs
        # around the decoupler column and the GPIO9 band via), outermost
        # B.Cu descents east of the VBUS lane, then In1 transfer lanes at
        # y 113.85-114.9 (In1 is empty there) west to F.Cu north entries.
        # A-column coil returns K1/K2/K3: separate stub columns west of
        # the pads, In1 across the void, staggered B.Cu tails behind U2.
        a_specs = (("K1", 62.25, 66.15, 151.0, 158.0),
                   ("K2", 60.5, 65.6, 150.4, 157.45),
                   ("K3", 59.8, 65.05, 151.0, 156.9))
        for relay, stub_x, corr_y, mid_x, vert_x in a_specs:
            coil = self.pad_position(relay, 8)
            net_name = f"/{relay}_COIL"
            target = None
            for ref in ("U2", "U3", "U4"):
                fp = self.board.FindFootprintByReference(ref)
                for pad in fp.Pads():
                    if pad.GetNetname() == net_name:
                        pos = pad.GetPosition()
                        target = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
            self.track([coil, (stub_x, coil[1]), (stub_x, corr_y)],
                       net_name, width=PLAIN_TRACK)
            self.via(stub_x, corr_y, net_name, size=0.4, drill=0.25)
            self.track([(stub_x, corr_y), (mid_x, corr_y)], net_name,
                       layer=pcbnew.In1_Cu, width=PLAIN_TRACK)
            self.via(mid_x, corr_y, net_name, size=0.4, drill=0.25)
            if relay == "K3":
                # east entry: the tail slides over U2 on B.Cu at y 55.75
                self.track([(mid_x, corr_y), (vert_x, corr_y),
                            (vert_x, 55.75), (162.3, 55.75),
                            (162.3, target[1])], net_name,
                           layer=pcbnew.B_Cu, width=PLAIN_TRACK)
                self.via(162.3, target[1], net_name, size=0.4, drill=0.25)
                self.track([(162.3, target[1]), (161.45, target[1])],
                           net_name, width=0.15)
            else:
                self.track([(mid_x, corr_y), (vert_x, corr_y),
                            (vert_x, target[1])], net_name,
                           layer=pcbnew.B_Cu, width=PLAIN_TRACK)
                self.via(vert_x, target[1], net_name, size=0.4, drill=0.25)
                self.track([(vert_x, target[1]), (159.5, target[1])],
                           net_name, width=0.15)

        # C-column coil returns K7/K8/K9: B.Cu corridors in the free
        # bands, F.Cu where a branch or corridor must be crossed.
        k7 = self.pad_position("K7", 8)
        t7 = None
        for ref in ("U2", "U3", "U4"):
            for pad in self.board.FindFootprintByReference(ref).Pads():
                if pad.GetNetname() == "/K7_COIL":
                    pos = pad.GetPosition()
                    t7 = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        self.track([k7, (k7[0], 67.4)], "/K7_COIL", width=PLAIN_TRACK)
        self.via(k7[0], 67.4, "/K7_COIL", size=0.4, drill=0.25)
        self.track([(k7[0], 67.4), (158.6, 67.4)], "/K7_COIL",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(158.6, 67.4, "/K7_COIL", size=0.4, drill=0.25)
        self.track([(158.6, 67.4), (158.6, t7[1]), (159.5, t7[1])],
                   "/K7_COIL", width=0.15)

        k8 = self.pad_position("K8", 8)
        t8 = None
        for pad in self.board.FindFootprintByReference("U3").Pads():
            if pad.GetNetname() == "/K8_COIL":
                pos = pad.GetPosition()
                t8 = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        self.track([k8, (k8[0], 78.45)], "/K8_COIL", width=PLAIN_TRACK)
        self.via(k8[0], 78.45, "/K8_COIL", size=0.4, drill=0.25)
        self.track([(k8[0], 78.45), (148.22, 78.45), (148.22, 76.76),
                    (151.425, t8[1]), (157.8, t8[1])],
                   "/K8_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(157.8, t8[1], "/K8_COIL", size=0.4, drill=0.25)
        self.track([(157.8, t8[1]), (159.5, t8[1])], "/K8_COIL",
                   width=0.15)

        k9 = self.pad_position("K9", 8)
        t9 = None
        for pad in self.board.FindFootprintByReference("U3").Pads():
            if pad.GetNetname() == "/K9_COIL":
                pos = pad.GetPosition()
                t9 = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        self.track([k9, (k9[0], 88.85)], "/K9_COIL", width=PLAIN_TRACK)
        self.via(k9[0], 88.85, "/K9_COIL", size=0.4, drill=0.25)
        self.track([(k9[0], 88.85), (156.95, 88.85), (156.95, 81.1)],
                   "/K9_COIL", layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(156.95, 81.1, "/K9_COIL", size=0.4, drill=0.25)
        self.track([(156.95, 81.1), (156.95, 77.9)], "/K9_COIL",
                   width=0.15)
        self.via(156.95, 77.9, "/K9_COIL", size=0.4, drill=0.25)
        self.track([(156.95, 77.6), (156.95, t9[1])], "/K9_COIL",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(156.95, t9[1], "/K9_COIL", size=0.4, drill=0.25)
        self.track([(156.95, t9[1]), (159.5, t9[1])], "/K9_COIL",
                   width=0.15)

        # Iso C/D coil returns K17/K18: B.Cu along the top edge, down
        # the virgin strip east of the VBUS lane, long F.Cu return stubs
        # riding just south of the GPIO17/18 bands into the U4 east pads.
        k17 = self.pad_position("K17", 8)
        t17 = None
        for pad in self.board.FindFootprintByReference("U4").Pads():
            if pad.GetNetname() == "/K17_COIL":
                pos = pad.GetPosition()
                t17 = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        self.track([k17, (132.6, k17[1])], "/K17_COIL", width=PLAIN_TRACK)
        self.via(132.6, k17[1], "/K17_COIL", size=0.4, drill=0.25)
        self.track([(132.6, k17[1]), (178.1, k17[1]),
                    (178.1, t17[1])], "/K17_COIL",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(178.1, t17[1], "/K17_COIL", size=0.4, drill=0.25)
        self.track([(178.1, t17[1]), (161.45, t17[1])], "/K17_COIL",
                   width=0.15)

        k18 = self.pad_position("K18", 8)
        t18 = None
        for pad in self.board.FindFootprintByReference("U4").Pads():
            if pad.GetNetname() == "/K18_COIL":
                pos = pad.GetPosition()
                t18 = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        self.track([k18, (133.2, k18[1]), (133.2, 41.65)], "/K18_COIL",
                   width=PLAIN_TRACK)
        self.via(133.2, 41.65, "/K18_COIL", size=0.4, drill=0.25)
        self.track([(133.2, 41.65), (178.65, 41.65),
                    (178.65, t18[1])], "/K18_COIL",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(178.65, t18[1], "/K18_COIL", size=0.4, drill=0.25)
        self.track([(178.65, t18[1]), (161.45, t18[1])], "/K18_COIL",
                   width=0.15)

        # GPIO15 -> U1.26 (east row): all F.Cu -- over the top of U4,
        # down the east flank at 165.6 with a jog around the pin-35
        # ground stub, entry from the east at the pad's own row.
        u4_1 = self.pad_position("U4", 1)
        pad26 = self.pad_position("U1", 26)
        self.track([u4_1, (154.85, u4_1[1]), (154.85, 86.9),
                    (165.6, 86.9), (165.6, 89.4)], "/GPIO15",
                   width=PLAIN_TRACK)
        self.via(165.6, 89.4, "/GPIO15", size=0.4, drill=0.25)
        self.track([(165.6, 89.4), (165.6, 106.2)], "/GPIO15",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)
        self.via(165.6, 106.2, "/GPIO15", size=0.4, drill=0.25)
        self.track([(165.6, 106.2), (165.6, 108.95), (166.3, 108.95),
                    (166.3, 113.55), (165.6, pad26[1]),
                    (164.9, pad26[1]), pad26], "/GPIO15",
                   width=PLAIN_TRACK)

        # NRST: all F.Cu -- west of the MCU, north over the pad rows at
        # y 105.3, down into C15.1 from above. Zero vias, and it leaves
        # the B.Cu freeway unobstructed.
        u1_7 = self.pad_position("U1", 7)
        c15_1 = self.pad_position("C15", 1)
        self.track([u1_7, (154.75, u1_7[1]), (154.75, 104.3),
                    (167.85, 104.3), (167.85, 107.6), (168.3, 107.9)],
                   "/NRST", width=PLAIN_TRACK)
        # lane extension into R3.2 (the NRST pull-up)
        self.track([(167.85, 104.3), (169.51, 104.3), (169.51, 104.9)],
                   "/NRST", width=0.15)
        # J8.4 leg: east along the top, down the strip east of VBUS,
        # back west on F.Cu into the R3.2 junction
        j8_4 = self.pad_position("J8", 4)
        self.track([j8_4, (165.55, j8_4[1]), (166.1, 45.5),
                    (181.9, 45.5), (181.9, 103.2), (181.65, 103.6),
                    (170.0, 103.6), (169.51, 104.3)],
                   "/NRST", width=PLAIN_TRACK)

        # LDO input side: the In1 +5V plane is starved right here (via
        # antipad fence), so EN loops to VIN on F.Cu and VIN straps up to
        # R9.2, which reaches the plane; GND keeps a west via.
        u5_vin = self.pad_position("U5", 1)
        u5_gnd = self.pad_position("U5", 2)
        u5_en = self.pad_position("U5", 3)
        r9_2 = self.pad_position("R9", 2)
        self.track([u5_en, (165.6, u5_en[1]), (165.6, u5_vin[1]), u5_vin],
                   "/+5V", width=PLAIN_TRACK)
        self.track([u5_vin, (u5_vin[0], 50.4), (r9_2[0], 50.4), r9_2],
                   "/+5V", width=PLAIN_TRACK)
        self.track([u5_gnd, (166.4, u5_gnd[1])], "/GND", width=PLAIN_TRACK)
        self.via(166.4, u5_gnd[1], "/GND", size=0.4, drill=0.25)

        # bulk cap C1 supply: east stub between the fan and the VBUS lane
        c1_5v = self.pad_position("C1", 1)
        self.track([c1_5v, (174.4, c1_5v[1])], "/+5V", width=PLAIN_TRACK)
        self.via(174.4, c1_5v[1], "/+5V", size=0.4, drill=0.25)
        # LDO input cap C3 ground: short north stub
        c3_gnd = self.pad_position("C3", 2)
        self.track([c3_gnd, (170.15, c3_gnd[1])], "/GND", width=PLAIN_TRACK)
        self.via(170.15, c3_gnd[1], "/GND", size=0.4, drill=0.25)
        # power LED ground: south stub below D1 (pad 1 = cathode)
        d1_gnd = self.pad_position("D1", 1)
        self.track([d1_gnd, (d1_gnd[0], 54.4)], "/GND", width=PLAIN_TRACK)
        self.via(d1_gnd[0], 54.4, "/GND", size=0.4, drill=0.25)

        # LDO output cap ground: explicit east stub clear of the fan.
        c4_gnd = self.pad_position("C4", 2)
        self.track([c4_gnd, (174.3, c4_gnd[1])], "/GND", width=PLAIN_TRACK)
        self.via(174.3, c4_gnd[1], "/GND", size=0.4, drill=0.25)

        # LDO input cap: staggered east stubs for both pads.
        c10_5v = self.pad_position("C10", 1)
        c10_gnd = self.pad_position("C10", 2)
        self.track([c10_5v, (174.35, c10_5v[1])], "/+5V", width=PLAIN_TRACK)
        self.via(174.35, c10_5v[1], "/+5V", size=0.4, drill=0.25)
        self.track([c10_gnd, (174.35, c10_gnd[1] - 1.38)], "/GND",
                   width=PLAIN_TRACK)
        self.via(174.35, c10_gnd[1] - 1.38, "/GND", size=0.4, drill=0.25)

        # U2 ground pad: the coil corridors boxed out its fanout
        u2_8 = self.pad_position("U2", 8)
        self.track([u2_8, (155.7, 66.75)], "/GND", width=PLAIN_TRACK)
        self.via(155.7, 66.75, "/GND", size=0.4, drill=0.25)

        # NRST-filter and BOOT0-pulldown grounds: east stubs into the plane
        for reference in ("C15", "R4"):
            pad = self.pad_position(reference, 2)
            self.track([pad, (171.3, pad[1])], "/GND", width=PLAIN_TRACK)
            self.via(171.3, pad[1], "/GND", size=0.4, drill=0.25)

        # ESD array ground: pad 2 is the MIDDLE pad of the east column,
        # so escape east between the DP_CON/DM_CON diagonals.
        d2_gnd = self.pad_position("D2", 2)
        self.track([d2_gnd, (180.35, d2_gnd[1])], "/GND", width=PLAIN_TRACK)
        self.via(180.35, d2_gnd[1], "/GND", size=0.4, drill=0.25)
        # CC pulldown grounds
        r1_gnd = self.pad_position("R1", 2)
        self.track([r1_gnd, (r1_gnd[0], 117.2)], "/GND", width=PLAIN_TRACK)
        self.via(r1_gnd[0], 117.2, "/GND", size=0.4, drill=0.25)
        r2_gnd = self.pad_position("R2", 2)
        self.track([r2_gnd, (181.5, 105.8)], "/GND", width=PLAIN_TRACK)
        self.via(181.5, 105.8, "/GND", size=0.4, drill=0.25)
        # USB row GND pads: short diagonals to plane vias beside the row
        a1 = self.pad_position("J1", "A1")      # y = 115.25
        a12 = self.pad_position("J1", "A12")    # y = 108.75
        self.track([a1, (182.9, 115.7)], "/GND", width=0.2)
        self.via(182.9, 115.7, "/GND", size=0.4, drill=0.25)
        self.track([a12, (182.9, 108.3)], "/GND", width=0.2)
        self.via(182.9, 108.3, "/GND", size=0.4, drill=0.25)

    def relax_driver_band(self):
        """Rubber-sheet the control-strip pre-routes east to follow the
        U2-U5 / decoupler cascade.

        The drivers sit between two equally dense fields: the west escape
        corridor (relay bodies -> input pads) and the east band (output
        pads -> decouplers). Moving the parts alone would leave every
        coil vertical and GPIO descent behind, so each pre-routed endpoint
        inside the driver band is translated by the same offset. Endpoints
        outside the window (relay-side corridors to the west, U1 entries
        below y=102) stay put, which simply stretches the runs that cross
        the boundary.
        """
        for footprint in self.board.GetFootprints():
            moved = band_shift(footprint.GetPosition())
            if moved != footprint.GetPosition():
                footprint.SetPosition(moved)
        for track in self.board.GetTracks():
            if track.Type() == pcbnew.PCB_VIA_T:
                track.SetPosition(band_shift(track.GetPosition()))
            else:
                track.SetStart(band_shift(track.GetStart()))
                track.SetEnd(band_shift(track.GetEnd()))

    def usb_rule_area(self):
        """Named rule area anchoring the .kicad_dru relaxation to 0.13 mm
        clearance / 0.15 mm track over the USB pad field."""
        area = pcbnew.ZONE(self.board)
        area.SetIsRuleArea(True)
        area.SetDoNotAllowCopperPour(False)
        area.SetDoNotAllowTracks(False)
        area.SetDoNotAllowVias(False)
        area.SetDoNotAllowPads(False)
        area.SetDoNotAllowFootprints(False)
        area.SetZoneName("usb_field")
        layer_set = pcbnew.LSET()
        layer_set.AddLayer(pcbnew.F_Cu)
        layer_set.AddLayer(pcbnew.B_Cu)
        area.SetLayerSet(layer_set)
        shape = area.Outline()
        shape.NewOutline()
        for x, y in ((182.8, 108.1), (185.3, 108.1),
                     (185.3, 115.9), (182.8, 115.9)):
            shape.Append(mm(x), mm(y))
        self.board.Add(area)

    def bridge_copper(self):
        """AGND pour under the bridge band plus locked fanout vias.

        The bridge references the coax shells (AGND); the pour gives the
        RC1 shunt and RV2 a low-inductance return, and R11 ties AGND to the
        digital GND at exactly one point east of the bridge.
        """
        pour = pcbnew.ZONE(self.board)
        pour.SetNet(self.net("/AGND"))
        zone_layer(pour, pcbnew.B_Cu)
        pour.SetAssignedPriority(1)
        pour.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        pour.SetMinThickness(mm(0.25))
        shape = pour.Outline()
        shape.NewOutline()
        for x, y in ((56.0, 106.5), (149.9, 106.5),
                     (149.9, BOARD_Y1 - 0.6), (56.0, BOARD_Y1 - 0.6)):
            shape.Append(mm(x), mm(y))
        self.board.Add(pour)

        # Locked stubs+vias: every SMD AGND pad down to the pour (the pour
        # reads as a plane to freerouting, which would strand them).
        for reference, pin in (("R13", 2), ("R14", 2), ("R15", 2), ("R11", 1)):
            pad = self.pad_position(reference, pin)
            other = self.pad_position(reference, 1 if pin == 2 else 2)
            direction = 1.0 if pad[1] >= other[1] else -1.0
            y = pad[1] + direction * 1.4
            if abs(pad[1] - other[1]) < 0.1:      # horizontal part: go east
                self.track([pad, (pad[0] - 1.4, pad[1])], "/AGND",
                           width=PLAIN_TRACK)
                self.via(pad[0] - 1.4, pad[1], "/AGND")
            else:
                self.track([pad, (pad[0], y)], "/AGND", width=PLAIN_TRACK)
                self.via(pad[0], y, "/AGND")
        # R11 digital side: stub east into the GND pour region.
        r11_gnd = self.pad_position("R11", 2)
        self.track([r11_gnd, (149.6, r11_gnd[1])], "/GND", width=PLAIN_TRACK)
        self.via(149.6, r11_gnd[1], "/GND")
        self.track([(149.6, r11_gnd[1]), (154.3375, 112.75)], "/GND",
                   layer=pcbnew.B_Cu, width=PLAIN_TRACK)

    def tab_zones(self):
        # GND patch over the USB tab so the shell vias have copper to land in
        # (the main pours stop at y=129; the patch overlaps them to merge).
        patch = pcbnew.ZONE(self.board)
        patch.SetNet(self.net("/GND"))
        zone_layer(patch, pcbnew.B_Cu)
        patch.SetAssignedPriority(1)
        patch.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        patch.SetMinThickness(mm(0.25))
        outline_shape = patch.Outline()
        outline_shape.NewOutline()
        for x, y in ((157.1, 127.5), (170.9, 127.5),
                     (170.9, BOARD_Y1 + 1.1), (157.1, BOARD_Y1 + 1.1)):
            outline_shape.Append(mm(x), mm(y))
        self.board.Add(patch)

    def planes(self):
        void_x0, void_y0, void_x1, void_y1 = PLANE_VOID
        # The B.Cu keepout stops north of the bridge band so the AGND pour
        # has room; the inner-plane voids span the full height.
        keepout_bottoms = {pcbnew.In1_Cu: void_y1, pcbnew.In2_Cu: void_y1,
                           pcbnew.B_Cu: 106.0}
        for net_name, layer in (("/+5V", pcbnew.In1_Cu), ("/GND", pcbnew.In2_Cu),
                                ("/GND", pcbnew.B_Cu)):
            zone = pcbnew.ZONE(self.board)
            zone.SetNet(self.net(net_name))
            zone_layer(zone, layer)
            zone.SetAssignedPriority(0)
            zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL if net_name == "/GND"
                                  else pcbnew.ZONE_CONNECTION_THERMAL)
            zone.SetMinThickness(mm(0.25))
            zone.SetLocalClearance(mm(0.25))
            zone.SetThermalReliefGap(mm(0.4))
            zone.SetThermalReliefSpokeWidth(mm(0.4))
            outline_shape = zone.Outline()
            outline_shape.NewOutline()
            for x, y in ((BOARD_X0 + 1.0, BOARD_Y0 + 1.0),
                         (BOARD_X1 - 1.0, BOARD_Y0 + 1.0),
                         (BOARD_X1 - 1.0, BOARD_Y1 - 1.0),
                         (BOARD_X0 + 1.0, BOARD_Y1 - 1.0)):
                outline_shape.Append(mm(x), mm(y))
            self.board.Add(zone)

            keepout = pcbnew.ZONE(self.board)
            keepout.SetIsRuleArea(True)
            keepout.SetDoNotAllowCopperPour(True)
            keepout.SetDoNotAllowTracks(False)
            keepout.SetDoNotAllowVias(False)
            keepout.SetDoNotAllowPads(False)
            keepout.SetDoNotAllowFootprints(False)
            zone_layer(keepout, layer)
            outline_shape = keepout.Outline()
            outline_shape.NewOutline()
            bottom = keepout_bottoms[layer]
            for x, y in ((void_x0, void_y0), (void_x1, void_y0),
                         (void_x1, bottom), (void_x0, bottom)):
                outline_shape.Append(mm(x), mm(y))
            self.board.Add(keepout)

    def silkscreen(self):
        def text(content, x, y, size=1.2, bold=False):
            item = pcbnew.PCB_TEXT(self.board)
            item.SetText(content)
            item.SetPosition(vec(x, y))
            item.SetLayer(pcbnew.F_SilkS)
            item.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size)))
            item.SetTextThickness(mm(size * (0.2 if bold else 0.15)))
            self.board.Add(item)

        text("OpenMagnetics Relay Board rev B2  2026-08-31", 99.0, 108.0, 1.4, bold=True)
        text("A", 63.0, 44.5, 2.5, bold=True)
        text("B", 63.0, 56.0, 2.5, bold=True)
        text("C", 135.0, 44.5, 2.5, bold=True)
        text("D", 135.0, 56.0, 2.5, bold=True)
        text("DUT", 99.0, 49.0, 2.0, bold=True)
        # bay outline: where the magnetic sits
        for x1, y1, x2, y2 in ((84.5, 41.0, 113.5, 41.0),
                               (84.5, 62.0, 113.5, 62.0),
                               (84.5, 41.0, 84.5, 62.0),
                               (113.5, 41.0, 113.5, 62.0)):
            shape = pcbnew.PCB_SHAPE(self.board)
            shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
            shape.SetStart(vec(x1, y1))
            shape.SetEnd(vec(x2, y2))
            shape.SetLayer(pcbnew.F_SilkS)
            shape.SetWidth(mm(0.25))
            self.board.Add(shape)
        text("OUT", BNC_X["SOURCE"], 113.6, 1.3, bold=True)
        text("CH1", BNC_X["CH1"], 113.6, 1.3, bold=True)
        text("CH2", BNC_X["CH2"], 113.6, 1.3, bold=True)
        text("CAL", 137.3, 112.0, 1.0)
        text("SWD", 168.5, 45.0, 1.0)

    def save(self):
        self.board.Save(OUTPUT)
        return self.board


def main():
    builder = Board()
    builder.outline()
    builder.place_components()
    builder.route_signal_nets()
    for fp in builder.board.GetFootprints():
        if fp.GetReference() != "J1":
            continue
        limit = mm(191.15)
        for item in list(fp.GraphicalItems()):
            if item.GetClass() != "PCB_SHAPE":
                continue
            if item.GetLayer() != pcbnew.F_SilkS:
                continue
            start, end = item.GetStart(), item.GetEnd()
            if start.x > limit and end.x > limit:
                fp.Remove(item)
            elif end.x > limit:
                item.SetEnd(pcbnew.VECTOR2I(limit, end.y))
            elif start.x > limit:
                item.SetStart(pcbnew.VECTOR2I(limit, start.y))
    builder.route_coil_supply()
    builder.route_strip_grounds()
    builder.relax_driver_band()
    builder.usb_rule_area()
    builder.bridge_copper()
    builder.planes()
    builder.silkscreen()
    board = builder.save()
    tracks = [t for t in board.GetTracks()]
    print(f"PCB saved: {OUTPUT}")
    print(f"  {BOARD_X1 - BOARD_X0:.0f} x {BOARD_Y1 - BOARD_Y0:.0f} mm + Bode tabs")
    print(f"  footprints {len(board.GetFootprints())}, "
          f"pre-routed tracks+vias {len(tracks)} (locked), "
          f"zones {len(list(board.Zones()))}")
    destination = OUTPUT.replace(".kicad_pcb", ".dsn")
    if pcbnew.ExportSpecctraDSN(board, destination):
        # In1/In2 become "power" layers: freerouting then never routes on the
        # planes. (Its plane-pad fanout is not needed -- every GND/+5V pad is
        # deterministically pre-wired.)  Experiments log for the rest: "type
        # protect" -> worse; DSN keepouts -> much worse; net withdrawal ->
        # netless pads routed through. Plain "(type fix)" + the repair pass
        # in route_pcb.py handles the residue.
        with open(destination, encoding="utf-8") as handle:
            dsn_lines = handle.readlines()
        count = 0
        pending = None
        for index, line in enumerate(dsn_lines):
            stripped = line.strip()
            is_inner = (stripped.startswith("(layer In1.Cu")
                        or stripped.startswith("(layer In2.Cu"))
            if is_inner:
                pending = index
            elif pending is not None and "(type signal)" in line:
                dsn_lines[index] = line.replace("(type signal)",
                                                "(type power)")
                count += 1
                pending = None
            elif stripped.startswith("(layer "):
                pending = None
        with open(destination, "w", encoding="utf-8") as handle:
            handle.writelines(dsn_lines)
        if count != 2:
            raise SystemExit(f"DSN power-layer patch matched {count}/2!")
        print(f"  DSN exported, inner layers = power: {destination}")
    return board


if __name__ == "__main__":
    main()
