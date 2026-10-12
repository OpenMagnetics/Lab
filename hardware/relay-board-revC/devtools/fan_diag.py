import os
import sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy, pcbnew, optimize_tracks as ex
board = pcbnew.LoadBoard(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "relay_board.kicad_pcb"))
obs = ex.build_obstacles(board)
data, nets, refs = obs.arrays[pcbnew.F_Cu]
names = {c: n.GetNetname() for c, n in board.GetNetsByNetcode().items()}
fp = board.FindFootprintByReference(sys.argv[1])
xy = [(pcbnew.ToMM(p.GetPosition().x), pcbnew.ToMM(p.GetPosition().y)) for p in fp.Pads()]
cx = sum(x for x, _ in xy) / len(xy); cy = sum(y for _, y in xy) / len(xy)
for pad in fp.Pads():
    if pad.GetNetname() in ("", "/GND"): continue
    px, py = pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y)
    box = pad.GetBoundingBox(); hw, hh = pcbnew.ToMM(box.GetWidth())/2, pcbnew.ToMM(box.GetHeight())/2
    if hw > hh: n = (1.0 if px > cx else -1.0, 0.0); toe = (px+n[0]*hw, py)
    else: n = (0.0, 1.0 if py > cy else -1.0); toe = (px, py+n[1]*hh)
    end = (toe[0]+n[0]*1.0, toe[1]+n[1]*1.0)
    net = pad.GetNetCode()
    m = nets != net
    d = ex.seg_seg_distance(toe, end, data[m, 0:2], data[m, 2:4]) - data[m, 4] - data[m, 5] - 0.075
    k = numpy.argmin(d)
    row = data[m][k]
    print(f"{pad.GetNumber():>3} {pad.GetNetname():12s} margin {d[k]:+.3f} vs net {names.get(int(nets[m][k]), nets[m][k])} at ({row[0]:.2f},{row[1]:.2f})")
