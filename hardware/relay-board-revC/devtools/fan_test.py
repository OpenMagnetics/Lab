import os
import sys, shutil
sys.argv.append("--fine")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pcbnew, maze_route as m
src = next((a for a in sys.argv[1:] if a.endswith(".kicad_pcb")), os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "relay_board.kicad_pcb"))
board = pcbnew.LoadBoard(src)
grid = m.Grid(board)
nb = {n.GetNetname(): n for n in board.GetNetsByNetcode().values()}
m.escape_fine_pitch(board, grid, nb)
board.Save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fan_test.kicad_pcb"))
