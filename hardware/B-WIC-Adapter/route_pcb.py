"""
Route traces using channel-based approach on dedicated layers.
- B.Cu: GPIO only (short, fan-out from U1 to U2/U3)
- In1.Cu: Coil traces only (relay coil- to ULN outputs)
- F.Cu: Bode100 traces only
- B.Cu: NRST, BOOT0, R3_COM (short isolated traces)

Key insight: GPIO3-6 share Y=127.8 on U1, so they need vertical fan-out FIRST,
then horizontal runs at staggered Y values.

Run with: "C:\Program Files\KiCad\9.0\bin\python.exe" route_pcb.py
"""

import pcbnew, os

def mm(v): return pcbnew.FromMM(v)
def vec(x, y): return pcbnew.VECTOR2I(mm(x), mm(y))

def track(board, x1, y1, x2, y2, net, layer, width=0.25):
    n = board.FindNet(net)
    if not n: return
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(vec(x1, y1)); t.SetEnd(vec(x2, y2))
    t.SetNet(n); t.SetLayer(layer); t.SetWidth(mm(width))
    board.Add(t)

def via(board, x, y, net):
    n = board.FindNet(net)
    if not n: return
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(vec(x, y))
    v.SetDrill(mm(0.3)); v.SetWidth(mm(0.6))
    v.SetNet(n); v.SetViaType(pcbnew.VIATYPE_THROUGH)
    board.Add(v)

def pad_pos(board, ref, num):
    for fp in board.GetFootprints():
        if fp.GetReference() == ref:
            for pad in fp.Pads():
                if pad.GetNumber() == str(num):
                    p = pad.GetPosition()
                    return pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
    return None

def route_board():
    pcb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay_board.kicad_pcb")
    board = pcbnew.LoadBoard(pcb_path)
    p = lambda ref, num: pad_pos(board, ref, num)

    BCu = pcbnew.B_Cu
    In1 = pcbnew.In1_Cu
    FCu = pcbnew.F_Cu

    # =================================================================
    # GPIO0-6 on B.Cu: U1 left/bottom pins -> U2 left pins
    # Fan out vertically from U1, then horizontal to U2 X, then vertical to U2 pin
    # Each trace gets its own horizontal channel Y
    # =================================================================
    gpio_data = [
        # (net, U1_pin, U2_pin, channel_y)
        ("GPIO0", 6,  1,  138.0),  # Lowest channel (U1 pin at Y=130.8, U2 at Y=136.4)
        ("GPIO1", 7,  2,  137.0),
        ("GPIO2", 8,  3,  136.0),
        ("GPIO3", 9,  4,  135.0),
        ("GPIO4", 10, 5,  134.0),
        ("GPIO5", 11, 6,  133.0),
        ("GPIO6", 12, 7,  132.0),  # Highest channel
    ]

    for net, u1_pin, u2_pin, ch_y in gpio_data:
        sx, sy = p("U1", u1_pin)
        ex, ey = p("U2", u2_pin)
        # Fan out X: offset left of U1 to create unique vertical paths
        fan_x = sx - 2  # All GPIO go left from U1
        # Step 1: U1 pin -> fan_x at same Y
        track(board, sx, sy, fan_x, sy, net, BCu, 0.2)
        # Step 2: fan_x down to channel Y
        track(board, fan_x, sy, fan_x, ch_y, net, BCu, 0.2)
        # Step 3: horizontal at channel Y to U2 X
        track(board, fan_x, ch_y, ex, ch_y, net, BCu, 0.2)
        # Step 4: vertical from channel to U2 pin Y
        track(board, ex, ch_y, ex, ey, net, BCu, 0.2)

    # GPIO7-8: U1 bottom-right -> U3 left (go right then down)
    for net, u1_pin, u3_pin, ch_y in [("GPIO7", 13, 1, 138.0), ("GPIO8", 15, 2, 137.5)]:
        sx, sy = p("U1", u1_pin)
        ex, ey = p("U3", u3_pin)
        fan_x = sx + 2
        track(board, sx, sy, fan_x, sy, net, BCu, 0.2)
        track(board, fan_x, sy, fan_x, ch_y, net, BCu, 0.2)
        track(board, fan_x, ch_y, ex, ch_y, net, BCu, 0.2)
        track(board, ex, ch_y, ex, ey, net, BCu, 0.2)

    # =================================================================
    # COIL TRACES on In1.Cu
    # ULN output -> via -> In1.Cu horizontal channel -> via -> relay coil-
    # Each trace gets a unique Y channel between 70 and 90 (above relay area)
    # =================================================================
    coil_data = [
        # (net, k_ref, k_pin, u_ref, u_pin, channel_y)
        ("K1_COIL", "K1", 8, "U2", 16, 70.0),
        ("K2_COIL", "K2", 8, "U2", 15, 71.5),
        ("K3_COIL", "K3", 8, "U2", 14, 73.0),
        ("K4_COIL", "K4", 8, "U2", 13, 74.5),
        ("K5_COIL", "K5", 8, "U2", 12, 76.0),
        ("K6_COIL", "K6", 8, "U2", 11, 77.5),
        ("K7_COIL", "K7", 8, "U2", 10, 79.0),
        ("K8_COIL", "K8", 8, "U3", 16, 80.5),
        ("K9_COIL", "K9", 8, "U3", 15, 82.0),
    ]

    for net, k_ref, k_pin, u_ref, u_pin, ch_y in coil_data:
        kx, ky = p(k_ref, k_pin)
        ux, uy = p(u_ref, u_pin)

        # Via near ULN output
        vu_x, vu_y = ux + 2.5, uy
        track(board, ux, uy, vu_x, vu_y, net, BCu, 0.3)
        via(board, vu_x, vu_y, net)

        # Via near relay
        vk_x, vk_y = kx + 2.5, ky
        track(board, kx, ky, vk_x, vk_y, net, BCu, 0.3)
        via(board, vk_x, vk_y, net)

        # In1.Cu: vertical from ULN via to channel, horizontal, vertical to relay via
        track(board, vu_x, vu_y, vu_x, ch_y, net, In1, 0.3)
        track(board, vu_x, ch_y, vk_x, ch_y, net, In1, 0.3)
        track(board, vk_x, ch_y, vk_x, vk_y, net, In1, 0.3)

    # =================================================================
    # NRST on B.Cu: R3.2 (107,124) -> U1.4 (98.3,132.4)
    # =================================================================
    r3 = p("R3", 2); u14 = p("U1", 4)
    track(board, r3[0], r3[1], r3[0], u14[1], "NRST", BCu, 0.25)
    track(board, r3[0], u14[1], u14[0], u14[1], "NRST", BCu, 0.25)

    # =================================================================
    # BOOT0 on B.Cu: R4.1 (110,124) -> U1.2 (98.3,134)
    # Route right then down to avoid crossing NRST
    # =================================================================
    r4 = p("R4", 1); u12 = p("U1", 2)
    track(board, r4[0], r4[1], r4[0], u12[1], "BOOT0", BCu, 0.25)
    track(board, r4[0], u12[1], u12[0], u12[1], "BOOT0", BCu, 0.25)

    # =================================================================
    # R3_COM on B.Cu: K2.4 (105,114.2) -> K3.2 (124,118.6)
    # =================================================================
    k24 = p("K2", 4); k32 = p("K3", 2)
    track(board, k24[0], k24[1], k32[0], k24[1], "R3_COM", BCu, 0.4)
    track(board, k32[0], k24[1], k32[0], k32[1], "R3_COM", BCu, 0.4)

    # =================================================================
    # Bode100 on F.Cu: relay pad vias -> edge tab connectors
    # These run through signal zones which will create clearance
    # =================================================================
    k1_2 = p("K1", 2); j5 = p("J5", 1)
    track(board, k1_2[0], k1_2[1], k1_2[0], j5[1], "Bode100_1", FCu, 0.5)
    track(board, k1_2[0], j5[1], j5[0], j5[1], "Bode100_1", FCu, 0.5)

    k2_2 = p("K2", 2); j6 = p("J6", 1)
    track(board, k2_2[0], k2_2[1], k2_2[0], j6[1], "Bode100_2", FCu, 0.5)
    track(board, k2_2[0], j6[1], j6[0], j6[1], "Bode100_2", FCu, 0.5)

    # =================================================================
    # +3V3 bus on B.Cu: U4.2 -> U1 power pins, caps, R3
    # Use a vertical bus line
    # =================================================================
    u4_2 = p("U4", 2)
    bus_x = 96.0  # +3V3 bus X

    # U4 output to bus
    track(board, u4_2[0], u4_2[1], bus_x, u4_2[1], "+3V3", BCu, 0.4)
    track(board, bus_x, u4_2[1], bus_x, 124.0, "+3V3", BCu, 0.4)

    # Taps from bus
    u1_1 = p("U1", 1)  # VDD
    track(board, bus_x, u1_1[1], u1_1[0], u1_1[1], "+3V3", BCu, 0.3)
    u1_5 = p("U1", 5)  # VDDA
    track(board, bus_x, u1_5[1], u1_5[0], u1_5[1], "+3V3", BCu, 0.3)

    # =================================================================
    # USB-C connector net assignments (J7)
    # Will need to be done in KiCad GUI — complex pin mapping
    # =================================================================

    # =================================================================
    # SAVE
    # =================================================================
    board.Save(pcb_path)
    tracks = board.GetTracks()
    via_count = sum(1 for t in tracks if isinstance(t, pcbnew.PCB_VIA))
    print(f"Routed PCB saved: {pcb_path}")
    print(f"  Track segments: {len(tracks) - via_count}")
    print(f"  Vias: {via_count}")

if __name__ == "__main__":
    route_board()
