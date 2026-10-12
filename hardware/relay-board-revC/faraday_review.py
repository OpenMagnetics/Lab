"""
EMC design review of the rev C layout with OpenConverters Faraday.

    python faraday_review.py [board.kicad_pcb] [--faraday PATH] [-o report.json]

Faraday (github.com/OpenConverters/Faraday, faraday.openconverters.com) screens a
layout for crosstalk, return-path breaks, ESD clamp paths, connector ground
spread, plane cavity modes, PDN anti-resonance and more, and ranks the
findings. It reads KiCad 5-9 files; KiCad 10 dropped numeric net codes
(`(net "/X")` instead of `(net 7 "/X")` plus a top-level net table), so the
board is first written to a KiCad-9-shaped copy: every net reference gets a
code and the net table is restored. Geometry is untouched.

The stackup is the real JLC04161H-7628 build, read from the board file
(generate_pcb.py writes it), never Faraday's generic default.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FARADAY = r"C:\Users\Alfonso\tools\ext\Faraday\build\faraday_cli.exe"


def to_kicad9_nets(text):
    """Number every `(net "NAME")` and add the top-level net table."""
    names = []
    for name in re.findall(r'\(net "((?:[^"\\]|\\.)*)"\)', text):
        if name not in names:
            names.append(name)
    codes = {"": 0}
    for name in names:
        if name not in codes:
            codes[name] = len(codes)
    text = re.sub(r'\(net "((?:[^"\\]|\\.)*)"\)',
                  lambda m: f'(net {codes[m.group(1)]} "{m.group(1)}")', text)
    table = "".join(f'\t(net {code} "{name}")\n' for name, code in codes.items())
    anchor = text.index("\n\t(footprint")
    return text[:anchor + 1] + table + text[anchor + 1:], len(codes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("board", nargs="?", default=os.path.join(HERE, "relay_board.kicad_pcb"))
    parser.add_argument("--faraday", default=DEFAULT_FARADAY)
    parser.add_argument("-o", "--output", default=os.path.join(HERE, "relay_board_faraday.json"))
    arguments = parser.parse_args()

    text = open(arguments.board, encoding="utf-8").read()
    converted, count = to_kicad9_nets(text)
    shadow = os.path.join(HERE, "relay_board_faraday_input.kicad_pcb")
    with open(shadow, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(converted)
    print(f"KiCad-9-shaped copy with {count} net codes: {os.path.basename(shadow)}")
    result = subprocess.run([arguments.faraday, shadow, "-o", arguments.output],
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    report = json.load(open(arguments.output, encoding="utf-8"))
    findings = report.get("findings", [])
    names = {net["id"]: net["name"] for net in report.get("board", {}).get("nets", [])}
    counts, waived = {}, []
    for finding in findings:
        reason = waiver(finding, names)
        if reason:
            waived.append((finding, reason))
            continue
        label = finding.get("severityLabel", "?")
        counts[label] = counts.get(label, 0) + 1
    print(f"{len(findings)} findings; open by severity: {counts}; waived by design: {len(waived)}")
    for finding, reason in waived:
        print(f"  waived {finding['id']} {finding['title']}: {reason}")
    print(f"report: {arguments.output}")
    return 0


#: Nets that are deliberately over no plane (ARCHITECTURE.md / generate_pcb.py
#: VOID): DUT arms and clamps, terminal buses, links. A linked terminal floats
#: at an intermediate potential, so ground capacitance there is an
#: uncalibrated Cps error -- the measurement-accuracy side of the trade-off.
VOID_BY_DESIGN = ("/DUT_", "/DUTS_", "/CAL_", "/STD")
BUSES = {f"/{kind}{terminal}" for kind in "TS" for terminal in "ABCD"}


def waiver(finding, names):
    if finding.get("rule") not in ("plane-crossing", "sparse-reference"):
        return None
    nets = [names.get(n, "") for n in (finding.get("netA"), finding.get("netB"))
            if isinstance(n, int) and n > 0]
    if nets and all(n.startswith(VOID_BY_DESIGN) or n in BUSES for n in nets):
        return "measurement node kept off planes (Cps accuracy)"
    return None


if __name__ == "__main__":
    sys.exit(main())
