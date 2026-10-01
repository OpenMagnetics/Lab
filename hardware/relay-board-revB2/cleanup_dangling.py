"""Post-route crumb sweep: parse DRC's dangling-item findings and delete
them, iterating until the report is clean or no progress is made. Runs
after route_pcb.py:

    "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" cleanup_dangling.py

Deletes unlocked dangling tracks and any dangling via (a locked landing
via that freerouting ignored is redundant copper on the finished board).
Zones are refilled after each pass so plane connectivity is re-judged.
"""

import os
import re
import subprocess

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(HERE, "relay_board.kicad_pcb")
KICAD_CLI = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
REPORT = os.path.join(HERE, "logs", "cleanup_drc.rpt")
os.makedirs(os.path.dirname(REPORT), exist_ok=True)

ENTRY = re.compile(
    r"\[(track_dangling|via_dangling)\][^@]*@\((\d+\.\d+) mm, (\d+\.\d+) mm\):"
    r" (Track|Via) \[([^\]]+)\]", re.S)


def drc_dangling():
    subprocess.run([KICAD_CLI, "pcb", "drc", "--severity-all",
                    "-o", REPORT, BOARD], capture_output=True)
    text = open(REPORT, encoding="utf-8", errors="replace").read()
    return [(kind, float(x), float(y), item, net)
            for kind, x, y, item, net in ENTRY.findall(text)]


for round_number in range(6):
    findings = drc_dangling()
    if not findings:
        print("no dangling items")
        break
    board = pcbnew.LoadBoard(BOARD)
    graveyard = []
    removed = 0
    for kind, x, y, item, net in findings:
        target = pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
        for track in list(board.GetTracks()):
            if track.GetNetname() != net:
                continue
            is_via = track.Type() == pcbnew.PCB_VIA_T
            if is_via != (item == "Via"):
                continue
            if is_via:
                hit = track.GetPosition() == target or (
                    abs(track.GetPosition().x - target.x) < 5000
                    and abs(track.GetPosition().y - target.y) < 5000)
            else:
                if track.IsLocked():
                    continue
                hit = track.HitTest(target, pcbnew.FromMM(0.02))
            if hit:
                board.Remove(track)
                graveyard.append(track)
                removed += 1
                break
    print(f"round {round_number}: {len(findings)} flagged, "
          f"{removed} removed")
    if not removed:
        break
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(BOARD, board)
else:
    print("gave up after 6 rounds")
