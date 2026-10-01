"""
Export fabrication data: Gerbers, drill, pick-and-place, IPC-356 netlist.

    python export_fab.py          (uses kicad-cli; runs DRC first and refuses
                                   to export over a failing board)
"""

import os
import subprocess
import sys
import zipfile

KICAD_CLI = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(HERE, "relay_board.kicad_pcb")
FAB = os.path.join(HERE, "fab")

LAYERS = "F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,B.Paste,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts"


def run(arguments, allow_fail=False):
    result = subprocess.run([KICAD_CLI] + arguments, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode and not allow_fail:
        print(output)
        raise SystemExit(f"kicad-cli failed: {' '.join(arguments[:3])}")
    return output


def drc_gate():
    report = os.path.join(HERE, "relay_board_drc.txt")
    run(["pcb", "drc", "--format", "report", "--severity-error",
         "-o", report, BOARD], allow_fail=True)
    errors = 0
    for line in open(report, encoding="utf-8", errors="replace"):
        if line.startswith("** Found"):
            print("  " + line.strip())
            if "violations" in line or "unconnected" in line:
                errors += int("".join(filter(str.isdigit, line.split("Found")[1].split()[0])) or 0)
    if errors:
        raise SystemExit(f"DRC reports {errors} error(s) -- not exporting fab data.")
    print("  DRC clean at error severity.")


def main():
    os.makedirs(FAB, exist_ok=True)
    print("DRC gate:")
    drc_gate()

    print("Gerbers:")
    run(["pcb", "export", "gerbers", "--layers", LAYERS,
         "--subtract-soldermask", "-o", FAB + os.sep, BOARD])
    print("  11 layers")

    print("Drill:")
    run(["pcb", "export", "drill", "--format", "excellon",
         "--excellon-separate-th", "--generate-map", "--map-format", "gerberx2",
         "-o", FAB + os.sep, BOARD])

    print("Pick and place:")
    run(["pcb", "export", "pos", "--format", "csv", "--units", "mm",
         "--use-drill-file-origin",
         "-o", os.path.join(FAB, "relay_board_pos.csv"), BOARD])

    print("IPC-356 netlist:")
    run(["pcb", "export", "ipc356", "-o",
         os.path.join(FAB, "relay_board.d356"), BOARD], allow_fail=True)

    print("BOM:")
    import shutil
    for name in ("relay_board_BOM.csv", "relay_board_BOM.md"):
        source = os.path.join(HERE, name)
        if not os.path.exists(source):
            raise SystemExit(f"missing {name} -- run generate_bom.py first")
        shutil.copy(source, os.path.join(FAB, name))
        print(f"  {name}")

    archive = os.path.join(HERE, "relay_board_fab.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(os.listdir(FAB)):
            bundle.write(os.path.join(FAB, name), name)
    print(f"Fab bundle: {archive} ({os.path.getsize(archive)//1024} KB)")


if __name__ == "__main__":
    main()
