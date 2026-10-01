"""
Generate 4-layer relay board PCB.
- F.Cu: Signal zones (A,B,C,D), relays, DUT connectors, Bode100 edge pads
- In1.Cu: +5V power plane, coil routing
- In2.Cu: GND plane
- B.Cu: MCU, ULN2003A, USB-C, regulator, passives, MCU routing

Run with: "C:\Program Files\KiCad\9.0\bin\python.exe" generate_pcb.py
"""

import pcbnew
import os

BX, BY = 60, 55
BW, BH = 85, 85
CX = BX + BW / 2   # 102.5
SY = BY + 35        # 90 — zone split Y
M = 2
G = 1.5

TAB_W = 20
TAB_H = 8
TAB_GAP = 4

FP_BASE = r'C:\Program Files\KiCad\9.0\share\kicad\footprints'

NET_NAMES = [
    "", "GND", "+5V", "+3V3",
    "Bode100_1", "Bode100_2",
    "A", "B", "C", "D",
    "USB_DP", "USB_DM", "VBUS", "R3_COM",
    "K1_COIL", "K2_COIL", "K3_COIL", "K4_COIL", "K5_COIL",
    "K6_COIL", "K7_COIL", "K8_COIL", "K9_COIL",
    "GPIO0", "GPIO1", "GPIO2", "GPIO3", "GPIO4",
    "GPIO5", "GPIO6", "GPIO7", "GPIO8",
    "NRST", "BOOT0", "SWDIO", "SWCLK", "CC1", "CC2",
]

def mm(v): return pcbnew.FromMM(v)
def vec(x, y): return pcbnew.VECTOR2I(mm(x), mm(y))
def load_fp(lib, name): return pcbnew.FootprintLoad(os.path.join(FP_BASE, lib), name)


def place(board, lib, fp_name, ref, value, x, y, angle, pin_nets, get_net, layer="F.Cu"):
    fp = load_fp(lib, fp_name)
    fp.SetReference(ref)
    fp.SetValue(value)
    fp.SetPosition(vec(x, y))
    if angle:
        fp.SetOrientationDegrees(angle)
    # Must add to board BEFORE flipping (Flip needs parent board context)
    board.Add(fp)
    if layer == "B.Cu":
        fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
    for pad in fp.Pads():
        try:
            pnum = int(pad.GetNumber())
        except (ValueError, TypeError):
            continue
        if pnum in pin_nets:
            pad.SetNet(get_net(pin_nets[pnum]))
    return fp


def add_zone(board, net_name, layer, coords, get_net, priority=0):
    zone = pcbnew.ZONE(board)
    zone.SetNet(get_net(net_name))
    zone.SetLayer(layer)
    zone.SetAssignedPriority(priority)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    zone.SetMinThickness(mm(0.25))
    zone.SetThermalReliefGap(mm(0.5))
    zone.SetThermalReliefSpokeWidth(mm(0.5))
    zone.SetIsFilled(False)
    outline = zone.Outline()
    outline.NewOutline()
    for x, y in coords:
        outline.Append(mm(x), mm(y))
    board.Add(zone)


def add_via(board, x, y, net_name, get_net):
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(vec(x, y))
    via.SetDrill(mm(0.3))
    via.SetWidth(mm(0.6))
    via.SetNet(get_net(net_name))
    via.SetViaType(pcbnew.VIATYPE_THROUGH)
    board.Add(via)


def add_track(board, x1, y1, x2, y2, net_name, layer, get_net, width=0.3):
    net = get_net(net_name)
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(vec(x1, y1))
    t.SetEnd(vec(x2, y2))
    t.SetNet(net)
    t.SetLayer(layer)
    t.SetWidth(mm(width))
    board.Add(t)


def get_pad_pos(board, ref, pad_num):
    for fp in board.GetFootprints():
        if fp.GetReference() == ref:
            for pad in fp.Pads():
                if pad.GetNumber() == str(pad_num):
                    pos = pad.GetPosition()
                    return pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
    return None


def generate_pcb():
    board = pcbnew.BOARD()
    board.GetDesignSettings().SetBoardThickness(mm(1.6))

    # Enable 4 layers
    board.SetCopperLayerCount(4)

    # Nets
    net_objects = {}
    for i, name in enumerate(NET_NAMES):
        net = pcbnew.NETINFO_ITEM(board, name, i)
        board.Add(net)
        net_objects[name] = net
    def get_net(name):
        return net_objects.get(name, net_objects[""])

    # ================================================================
    # BOARD OUTLINE
    # ================================================================
    tab_l1 = CX - TAB_W - TAB_GAP/2
    tab_l2 = CX - TAB_GAP/2
    tab_r1 = CX + TAB_GAP/2
    tab_r2 = CX + TAB_W + TAB_GAP/2

    outline = [
        (BX, BY), (BX+BW, BY), (BX+BW, BY+BH),
        (tab_r2, BY+BH), (tab_r2, BY+BH+TAB_H), (tab_r1, BY+BH+TAB_H), (tab_r1, BY+BH),
        (tab_l2, BY+BH), (tab_l2, BY+BH+TAB_H), (tab_l1, BY+BH+TAB_H), (tab_l1, BY+BH),
        (BX, BY+BH),
    ]
    for i in range(len(outline)):
        x1, y1 = outline[i]
        x2, y2 = outline[(i+1) % len(outline)]
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(vec(x1, y1))
        seg.SetEnd(vec(x2, y2))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(mm(0.15))
        board.Add(seg)

    # ================================================================
    # ALL COMPONENTS ON B.Cu (bottom side)
    # F.Cu is reserved for clean copper pour planes only
    # ================================================================
    relay_lib = "Relay_SMD.pretty"
    relay_fp = "Relay_DPDT_Omron_G6K-2F-Y"

    # Port routing relays — on B.Cu
    place(board, relay_lib, relay_fp, "K1", "G6K-2F-RF-S", CX-20, SY+28, 0,
        {1:"+5V", 8:"K1_COIL", 2:"Bode100_1", 3:"A", 4:"C"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K2", "G6K-2F-RF-S", CX+6, SY+28, 0,
        {1:"+5V", 8:"K2_COIL", 2:"Bode100_2", 3:"B", 4:"R3_COM"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K3", "G6K-2F-RF-S", CX+25, SY+28, 0,
        {1:"+5V", 8:"K3_COIL", 2:"R3_COM", 3:"D", 4:"C"}, get_net, layer="B.Cu")

    # C-D and A-B boundary relays — on B.Cu
    place(board, relay_lib, relay_fp, "K4", "G6K-2F-RF-S", CX, SY-14, 0,
        {1:"+5V", 8:"K4_COIL", 2:"C", 4:"D", 5:"D", 7:"C"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K5", "G6K-2F-RF-S", CX, SY+14, 0,
        {1:"+5V", 8:"K5_COIL", 2:"A", 4:"B", 5:"B", 7:"A"}, get_net, layer="B.Cu")

    # Cross-connect relays (rotated 90°, 12mm spacing) — on B.Cu
    place(board, relay_lib, relay_fp, "K8", "G6K-2F-RF-S", CX-18, SY, 90,
        {1:"+5V", 8:"K8_COIL", 2:"A", 4:"C"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K6", "G6K-2F-RF-S", CX-6, SY, 90,
        {1:"+5V", 8:"K6_COIL", 2:"B", 4:"C"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K9", "G6K-2F-RF-S", CX+6, SY, 90,
        {1:"+5V", 8:"K9_COIL", 2:"A", 4:"D"}, get_net, layer="B.Cu")
    place(board, relay_lib, relay_fp, "K7", "G6K-2F-RF-S", CX+18, SY, 90,
        {1:"+5V", 8:"K7_COIL", 2:"B", 4:"D"}, get_net, layer="B.Cu")

    # DUT connectors — through-hole, so they work on both sides
    for ref, val, x, net in [("J1","A",BX+10,"A"), ("J3","C",BX+28,"C"),
                              ("J4","D",CX+15,"D"), ("J2","B",BX+BW-12,"B")]:
        place(board, "Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical",
            ref, val, x, BY+6, 0, {1: net}, get_net)

    # Bode100 tab connectors — through-hole
    place(board, "Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical",
        "J5", "Bode100_1", (tab_r1+tab_r2)/2, BY+BH+TAB_H/2, 0, {1:"Bode100_1"}, get_net)
    place(board, "Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical",
        "J6", "Bode100_2", (tab_l1+tab_l2)/2, BY+BH+TAB_H/2, 0, {1:"Bode100_2"}, get_net)

    # ================================================================
    # B.Cu: ALL ICs AND PASSIVES (bottom side)
    # ================================================================
    # B.Cu MCU area: below the relays (Y > 126) to avoid courtyard overlap
    # K1/K2/K3 bottom edge is at ~Y=124 (center Y=118 + half height 5.3)
    # Need MCU components at Y >= 130
    mcu_x, mcu_y = CX, BY + BH - 8  # Y = 132

    # STM32F042K6Tx
    place(board, "Package_QFP.pretty", "LQFP-32_7x7mm_P0.8mm",
        "U1", "STM32F042K6Tx", mcu_x, mcu_y, 0,
        {1:"+3V3", 2:"BOOT0", 4:"NRST", 5:"+3V3",
         6:"GPIO0", 7:"GPIO1", 8:"GPIO2", 9:"GPIO3",
         10:"GPIO4", 11:"GPIO5", 12:"GPIO6", 13:"GPIO7",
         15:"GPIO8", 16:"GND", 17:"+3V3",
         21:"USB_DM", 22:"USB_DP", 23:"SWDIO", 24:"SWCLK", 32:"GND"},
        get_net, layer="B.Cu")

    # ULN2003A x2 — offset left and right of MCU, same Y
    place(board, "Package_SO.pretty", "SOIC-16_3.9x9.9mm_P1.27mm",
        "U2", "ULN2003A", CX-22, mcu_y, 0,
        {1:"GPIO0", 2:"GPIO1", 3:"GPIO2", 4:"GPIO3",
         5:"GPIO4", 6:"GPIO5", 7:"GPIO6", 8:"GND", 9:"+5V",
         16:"K1_COIL", 15:"K2_COIL", 14:"K3_COIL", 13:"K4_COIL",
         12:"K5_COIL", 11:"K6_COIL", 10:"K7_COIL"},
        get_net, layer="B.Cu")

    place(board, "Package_SO.pretty", "SOIC-16_3.9x9.9mm_P1.27mm",
        "U3", "ULN2003A", CX+22, mcu_y, 0,
        {1:"GPIO7", 2:"GPIO8", 3:"GND", 4:"GND", 5:"GND",
         6:"GND", 7:"GND", 8:"GND", 9:"+5V",
         16:"K8_COIL", 15:"K9_COIL"},
        get_net, layer="B.Cu")

    # USB-C connector — right edge, above tab area
    usb_x = BX + BW - 8
    usb_y = mcu_y
    place(board, "Connector_USB.pretty", "USB_C_Receptacle_G-Switch_GT-USB-7051x",
        "J7", "USB-C", usb_x, usb_y, 270,
        {}, get_net, layer="B.Cu")  # Nets assigned in KiCad (complex USB-C pin mapping)

    # Voltage regulator — between relay row and MCU, rotated to fit
    place(board, "Package_TO_SOT_SMD.pretty", "SOT-223-3_TabPin2",
        "U4", "AMS1117-3.3", CX+32, mcu_y, 90,
        {1:"GND", 2:"+3V3", 3:"+5V"}, get_net, layer="B.Cu")

    # Resistors (B.Cu)
    for i, (ref, val, x_off, pnets) in enumerate([
        ("R1", "5.1k", -8, {1:"CC1", 2:"GND"}),
        ("R2", "5.1k", -4, {1:"CC2", 2:"GND"}),
        ("R3", "10k",  4,  {1:"+3V3", 2:"NRST"}),
        ("R4", "10k",  8,  {1:"BOOT0", 2:"GND"}),
    ]):
        place(board, "Resistor_SMD.pretty", "R_0402_1005Metric",
            ref, val, mcu_x + x_off, mcu_y - 8, 0, pnets, get_net, layer="B.Cu")

    # Capacitors (B.Cu)
    for ref, val, fp_name, x, y, pnets in [
        ("C1", "100nF", "C_0402_1005Metric", mcu_x-6, mcu_y+5, {1:"+3V3", 2:"GND"}),
        ("C2", "100nF", "C_0402_1005Metric", mcu_x,   mcu_y+5, {1:"+3V3", 2:"GND"}),
        ("C3", "100nF", "C_0402_1005Metric", mcu_x+6, mcu_y+5, {1:"+3V3", 2:"GND"}),
        ("C4", "10uF",  "C_0805_2012Metric", CX-28,   mcu_y,   {1:"+5V", 2:"GND"}),
        ("C5", "10uF",  "C_0805_2012Metric", CX-16,   mcu_y+5, {1:"+3V3", 2:"GND"}),
        ("C6", "1uF",   "C_0402_1005Metric", mcu_x+12, mcu_y,  {1:"+5V", 2:"GND"}),
    ]:
        place(board, "Capacitor_SMD.pretty", fp_name,
            ref, val, x, y, 0, pnets, get_net, layer="B.Cu")

    # ================================================================
    # COPPER ZONES
    # ================================================================
    # F.Cu: Full-width signal zones A, B, C, D
    add_zone(board, "A", pcbnew.F_Cu, [
        (BX+M, SY+G/2), (CX-G/2, SY+G/2), (CX-G/2, BY+BH-M), (BX+M, BY+BH-M)], get_net)
    add_zone(board, "B", pcbnew.F_Cu, [
        (CX+G/2, SY+G/2), (BX+BW-M, SY+G/2), (BX+BW-M, BY+BH-M), (CX+G/2, BY+BH-M)], get_net)
    add_zone(board, "C", pcbnew.F_Cu, [
        (BX+M, BY+M), (CX-G/2, BY+M), (CX-G/2, SY-G/2), (BX+M, SY-G/2)], get_net)
    add_zone(board, "D", pcbnew.F_Cu, [
        (CX+G/2, BY+M), (BX+BW-M, BY+M), (BX+BW-M, SY-G/2), (CX+G/2, SY-G/2)], get_net)

    # In1.Cu: +5V power plane (main board area, not tabs)
    add_zone(board, "+5V", pcbnew.In1_Cu, [
        (BX+M, BY+M), (BX+BW-M, BY+M), (BX+BW-M, BY+BH-M), (BX+M, BY+BH-M)], get_net)

    # In2.Cu: GND plane (main board area, not extending into tab areas)
    add_zone(board, "GND", pcbnew.In2_Cu, [
        (BX+M, BY+M), (BX+BW-M, BY+M), (BX+BW-M, BY+BH-M), (BX+M, BY+BH-M)], get_net)
    # Also In1.Cu +5V plane — same boundary


    # B.Cu: GND zone (main board area only, NOT extending into tabs)
    add_zone(board, "GND", pcbnew.B_Cu, [
        (BX+M, BY+M), (BX+BW-M, BY+M), (BX+BW-M, BY+BH-M), (BX+M, BY+BH-M)], get_net)

    # Bode100 tab zones (F.Cu + B.Cu)
    for net, x1, x2 in [("Bode100_2", tab_l1, tab_l2), ("Bode100_1", tab_r1, tab_r2)]:
        add_zone(board, net, pcbnew.F_Cu, [(x1,BY+BH),(x2,BY+BH),(x2,BY+BH+TAB_H),(x1,BY+BH+TAB_H)], get_net)
        add_zone(board, net, pcbnew.B_Cu, [(x1,BY+BH),(x2,BY+BH),(x2,BY+BH+TAB_H),(x1,BY+BH+TAB_H)], get_net, priority=1)

    # ================================================================
    # VIA STITCHING for signal zones (F.Cu ↔ through-hole to In1/In2)
    # ================================================================
    relay_centers = [
        # F.Cu relays
        (CX-20,SY+28),(CX+6,SY+28),(CX+25,SY+28),
        (CX,SY-14),(CX,SY+14),
        (CX-18,SY),(CX-6,SY),(CX+6,SY),(CX+18,SY),
        # B.Cu components (must exclude from via stitching)
        (mcu_x, mcu_y),          # U1
        (CX-22, mcu_y),          # U2
        (CX+22, mcu_y),          # U3
        (CX+18, mcu_y+12),       # U4
        (usb_x, usb_y),          # J7 USB-C
        (mcu_x-8, mcu_y-10), (mcu_x-4, mcu_y-10),   # R1, R2
        (mcu_x+4, mcu_y-10), (mcu_x+8, mcu_y-10),   # R3, R4
        (mcu_x-6, mcu_y+8), (mcu_x, mcu_y+8), (mcu_x+6, mcu_y+8),  # C1-C3
        (CX-22, mcu_y+10), (CX+22, mcu_y+10),        # C4, C5
        (mcu_x+12, mcu_y),                             # C6
        ((tab_r1+tab_r2)/2, BY+BH+TAB_H/2),           # J5 Bode100_1
        ((tab_l1+tab_l2)/2, BY+BH+TAB_H/2),           # J6 Bode100_2
    ]

    for zone_name, net_id, xmin, xmax, ymin, ymax in [
        ("A", 6, BX+5, CX-5, SY+5, BY+BH-5),
        ("B", 7, CX+5, BX+BW-5, SY+5, BY+BH-5),
        ("C", 8, BX+5, CX-5, BY+5, SY-5),
        ("D", 9, CX+5, BX+BW-5, BY+5, SY-5),
    ]:
        x = xmin
        while x <= xmax:
            y = ymin
            while y <= ymax:
                if not any(abs(x-cx)<8 and abs(y-cy)<8 for cx,cy in relay_centers):
                    add_via(board, x, y, zone_name, get_net)
                y += 6
            x += 6

    # Bode100 tab vias
    for bx in range(int(tab_l1)+3, int(tab_l2)-2, 4):
        add_via(board, bx, BY+BH+TAB_H/2, "Bode100_2", get_net)
    for bx in range(int(tab_r1)+3, int(tab_r2)-2, 4):
        add_via(board, bx, BY+BH+TAB_H/2, "Bode100_1", get_net)

    # ================================================================
    # ROUTING: Relay coil- pins (F.Cu) -> ULN outputs (B.Cu) via In1.Cu
    # Each relay coil+ is on +5V (In1.Cu plane handles this)
    # Each relay coil- needs a via down to B.Cu ULN output
    # ================================================================
    pad = lambda ref, num: get_pad_pos(board, ref, num)

    coil_routes = [
        ("K1_COIL", "K1", 8, "U2", 16), ("K2_COIL", "K2", 8, "U2", 15),
        ("K3_COIL", "K3", 8, "U2", 14), ("K4_COIL", "K4", 8, "U2", 13),
        ("K5_COIL", "K5", 8, "U2", 12), ("K6_COIL", "K6", 8, "U2", 11),
        ("K7_COIL", "K7", 8, "U2", 10), ("K8_COIL", "K8", 8, "U3", 16),
        ("K9_COIL", "K9", 8, "U3", 15),
    ]
    # Coil traces (relay coil- to ULN output) — both on B.Cu
    # Route in KiCad interactive router (L-routes cross each other)

    # ================================================================
    # VIAS: Connect relay signal pads (B.Cu) to F.Cu copper pour zones
    # Each relay contact pad that carries A/B/C/D/Bode100 needs a via
    # to reach the F.Cu copper pour above it
    # ================================================================
    relay_signal_pads = {
        # ref: [(pad_num, net_name), ...]  — only signal pads, not coil/+5V
        'K1': [(2,'Bode100_1'), (3,'A'), (4,'C')],
        'K2': [(2,'Bode100_2'), (3,'B'), (4,'R3_COM')],
        'K3': [(2,'R3_COM'), (3,'D'), (4,'C')],
        'K4': [(2,'C'), (4,'D'), (5,'D'), (7,'C')],
        'K5': [(2,'A'), (4,'B'), (5,'B'), (7,'A')],
        'K6': [(2,'B'), (4,'C')],
        'K7': [(2,'B'), (4,'D')],
        'K8': [(2,'A'), (4,'C')],
        'K9': [(2,'A'), (4,'D')],
    }
    for ref, pad_nets in relay_signal_pads.items():
        for pad_num, net_name in pad_nets:
            px, py = pad(ref, pad_num)
            # Place via right at the pad — it goes from B.Cu through to F.Cu
            add_via(board, px, py, net_name, get_net)

    # Relay coil+ (pin 1) vias to +5V plane (In1.Cu)
    for ref in ["K1","K2","K3","K4","K5","K6","K7","K8","K9"]:
        px, py = pad(ref, 1)
        add_via(board, px, py, "+5V", get_net)

    # Bode100 traces (K1.2->J5, K2.2->J6) — route in KiCad
    # These cross signal zone boundaries on F.Cu and need zone cutouts

    # R3_COM (K2.4 -> K3.2) — route in KiCad (short trace, easy to do manually)

    # GPIO routing (U1 -> U2/U3) must be done in KiCad interactive router
    # because L-routes cross each other on B.Cu and SetLayer doesn't mirror pads

    # ================================================================
    # GND vias for bottom-side components
    # ================================================================
    for ref, pnum in [("U1",16),("U1",32),("U2",8),("U3",8),("U4",1),
                       ("C1",2),("C2",2),("C3",2),("C4",2),("C5",2),("C6",2),
                       ("R1",2),("R2",2),("R4",2)]:
        px, py = pad(ref, pnum)
        add_via(board, px, py+0.5, "GND", get_net)

    # +3V3 vias for bottom components to In2 GND return
    # (Caps connect to +3V3 on B.Cu, GND vias above handle ground)

    # ================================================================
    # SILKSCREEN
    # ================================================================
    for text, x, y, size in [
        ("C", BX+12, SY-15, 5), ("D", CX+12, SY-15, 5),
        ("A", BX+12, SY+15, 5), ("B", CX+12, SY+15, 5),
        ("Bode100_2", (tab_l1+tab_l2)/2, BY+BH+TAB_H/2+2, 1.0),
        ("Bode100_1", (tab_r1+tab_r2)/2, BY+BH+TAB_H/2+2, 1.0),
    ]:
        txt = pcbnew.PCB_TEXT(board)
        txt.SetText(text)
        txt.SetPosition(vec(x, y))
        txt.SetLayer(pcbnew.F_SilkS)
        txt.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size)))
        txt.SetTextThickness(mm(size*0.15))
        board.Add(txt)

    # ================================================================
    # SAVE
    # ================================================================
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay_board.kicad_pcb")
    board.Save(out)
    print(f"PCB saved: {out}")
    print(f"  Layers: 4 (F.Cu, In1.Cu, In2.Cu, B.Cu)")
    print(f"  Footprints: {len(board.GetFootprints())}")
    print(f"  Zones: {len(list(board.Zones()))}")
    print(f"  Tracks+Vias: {len(board.GetTracks())}")
    return out


if __name__ == "__main__":
    out = generate_pcb()
