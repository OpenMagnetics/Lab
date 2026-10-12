import os
import sys, time
sys.argv.append("--fine")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pcbnew, maze_route as m
board = pcbnew.LoadBoard(m.BOARD)
grid = m.Grid(board)
for item in board.GetTracks():
    if item.GetClass() == "PCB_VIA":
        p = item.GetPosition()
        grid.via_cells[(int(round((pcbnew.ToMM(p.y) - grid.y0) / m.GRID)), int(round((pcbnew.ToMM(p.x) - grid.x0) / m.GRID)))] = item.GetNetCode()
nb = {n.GetNetname(): n for n in board.GetNetsByNetcode().values()}
grid.set_classes(nb)
grid.exact = m.escape_fine_pitch(board, grid, nb)
orig = grid.astar
def astar(*a, **k):
    t = time.time(); r = orig(*a, **k); print("   astar", "found" if r else "NONE", round(time.time() - t, 1), "s"); return r
grid.astar = astar
for name in ["/" + a for a in sys.argv[1:] if not a.startswith("--")]:
    print(name, m.route_net(board, grid, name, 0.4, nb))
