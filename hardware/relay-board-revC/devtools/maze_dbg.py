import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy, pcbnew
import maze_route as m
board = pcbnew.LoadBoard(next((a[8:] for a in sys.argv if a.startswith('--board=')), m.BOARD))
grid = m.Grid(board)
for item in board.GetTracks():
    if item.GetClass() == "PCB_VIA":
        p = item.GetPosition()
        grid.via_cells[(int(round((pcbnew.ToMM(p.y) - grid.y0) / m.GRID)), int(round((pcbnew.ToMM(p.x) - grid.x0) / m.GRID)))] = item.GetNetCode()
for name in ["/" + a for a in sys.argv[1:] if not a.startswith("--")]:
    net = {n.GetNetname(): c for c, n in board.GetNetsByNetcode().items()}[name]
    ok = grid.free_for(net, 0.1)
    via_ok = grid.via_ok(ok, net)
    for L in (1, 2):
        ok[L] &= grid.copper[L] == net
    for isl in grid.components(net):
        # flood fill on ok, within layer + via cells
        seen = set(isl); stack = list(isl); vias = 0
        while stack and len(seen) < 30000000:
            l, j, i = stack.pop()
            nbs = [(l, j + a, i + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (not (a and b) or ok[l, j + a, i] or ok[l, j, i + b])]
            if via_ok[j, i]:
                vias += 1
                nbs += [(k, j, i) for k in range(4) if k != l]
            for nb in nbs:
                L, J, I = nb
                if 0 <= J < grid.ny and 0 <= I < grid.nx and nb not in seen and ok[L, J, I]:
                    seen.add(nb); stack.append(nb)
        others = [o for o in grid.components(net) if o is not isl]
        joined = [any(cell in seen for cell in o[:5]) for o in others]
        c = isl[0]
        print("   reaches other islands:", joined)
        print(name, "island at", grid.to_mm(c[1], c[2]), "layer", c[0], "cells", len(isl), "-> reachable", len(seen), "via cells", vias)
