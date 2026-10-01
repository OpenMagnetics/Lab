"""
Generate JLCPCB SMT-assembly BOM and CPL spreadsheets.

    python export_jlcpcb.py

Follows JLC's sample templates exactly:
  BOM: Comment | Designator | Footprint | JLCPCB Part #(optional)
  CPL: Designator | Mid X | Mid Y | Layer | Rotation

BOM data derives from relay_board.xml (same source generate_bom.py
verifies against); CPL positions from fab/relay_board_pos.csv, so run
export_fab.py first. Through-hole connectors (BNC jacks, Phoenix DUT
clamps, SWD header) are excluded -- hand-solder those; the USB-C
receptacle is SMT and stays in.

The JLCPCB Part # column is left blank on purpose: pick the parts in
JLC's BOM-matching UI (it matches on Comment/MPN) rather than trusting
hardcoded LCSC numbers that may have gone end-of-life.
"""

import csv
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import openpyxl

from generate_bom import PART_NUMBERS

HERE = os.path.dirname(os.path.abspath(__file__))

#: Pin generic passives to JLC *Basic* library parts: every Extended line
#: item adds a ~$3 loading fee, Basic adds none. C45783/C15849/C25905/
#: C17477 were confirmed by JLC's own matching report for this BOM; the
#: rest are the canonical evergreen Basic picks -- the matching UI shows
#: what each number resolves to, confirm there before ordering.
JLC_BASIC = {
    ("22uF", "C_0805_2012Metric"): ("CL21A226MAQNNNE", "C45783"),
    ("1uF", "C_0603_1608Metric"): ("CL10A105KB8NNNC", "C15849"),
    ("100nF", "C_0402_1005Metric"): ("CL05B104KO5NNNC", "C1525"),
    ("5.1k", "R_0402_1005Metric"): ("0402WGF5101TCE", "C25905"),
    ("10k", "R_0402_1005Metric"): ("0402WGF1002TCE", "C25744"),
    ("1k", "R_0402_1005Metric"): ("0402WGF1001TCE", "C11702"),
    ("0R", "R_0805_2012Metric"): ("0805W8F0000T5E", "C17477"),
    ("PWR", "LED_0603_1608Metric"): ("KT-0603R", "C2286"),
    # Confirmed by JLC's second matching report (2026-08-31) -- these are
    # the parts its own matcher resolved for this BOM. D2 and U5 are the
    # TECH PUBLIC clones JLC auto-picked (cheapest); swap to the ST/Diodes
    # originals in the UI if preferred. Not pinned: TNPW08054R70BEEA
    # (R14,R15 -- no JLC stock, deselect and hand-solder the Vishay part).
    # PROTOTYPE SUBSTITUTION: 1% thick film for the RC1 shunt halves;
    # OSL calibration against R5 (kept genuine 0.1%) absorbs the error.
    # Production board: TNPW08054R70BEEA.
    ("4R7", "R_0805_2012Metric"): ("0805W8F470KT5E", "C17675"),
    ("USBLC6-2SC6", "SOT-23-6"): ("USBLC6-2SC6", "C2827654"),
    ("USB-C", "USB_C_Receptacle_HRO_TYPE-C-31-M-12"):
        ("TYPE-C-31-M-12", "C165948"),
    # PROTOTYPE SUBSTITUTION: standard G6K-2F-Y instead of the $9.3 RF
    # version. Same land pattern per Omron datasheets (rows 3.2/5.4/7.6,
    # contact pads 1.8x0.8 on 7.0 mm span; the -Y coil terminal is
    # narrower than the RF-S pad, which is fine), same 5 V/237R coil,
    # same DPDT arrangement. Loses 50R impedance control -- irrelevant
    # below 50 MHz. Production board: G6K-2F-RF-S DC5 (C2750987).
    ("G6K-2F-RF-S DC5", "Relay_Omron_G6K-2F-RF-S"):
        ("G6K-2F-Y-TR DC5", "C47190"),
    ("100R 0.1%", "R_0805_2012Metric"): ("TNPW0805100RBEEA", "C2073390"),
    ("1k 0.1%", "R_0805_2012Metric"): ("TNPW08051K00BEEA", "C2073519"),
    ("47R 0.1%", "R_0805_2012Metric"): ("0805W8F470JT5E", "C17714"),
    ("STM32F072CBT6", "LQFP-48_7x7mm_P0.5mm"): ("STM32F072CBT6", "C81720"),
    ("TBD62003APG", "SOIC-16_3.9x9.9mm_P1.27mm"):
        ("TBD62003AFG(Z,EL)", "C163227"),
    ("AP2112K-3.3", "SOT-23-5"): ("AP2112K-3.3TRG1", "C23380830"),
}


def jlc_footprint(tail):
    """JLC's matcher parses sizes out of the footprint name and reads the
    KiCad metric suffix (C_0402_1005Metric) as an 01005 -- feed it plain
    imperial names like its own sample (R0603) instead."""
    m = re.match(r"^(C|R|L|LED)_(\d{4})_\d{4}Metric$", tail)
    if m:
        return m.group(1) + m.group(2)
    for prefix in ("SOIC-16", "LQFP-48"):
        if tail.startswith(prefix):
            return prefix
    return tail


def jlc_comment(value, part):
    """Bare MPN only: JLC strips spaces from comments, so a manufacturer
    prefix ('ST USBLC6-2SC6' -> 'STUSBLC6-2SC6') kills the part search."""
    if value == "PWR":
        return "LTST-C190GKT"      # generic green 0603 LED
    if part:
        return part.split(" ", 1)[1] if " " in part else part
    return value


#: Hand-soldered through-hole parts, not for the SMT line.
HAND_SOLDER = {"J2", "J3", "J4", "J5",   # Phoenix DUT clamps
               "J6", "J7", "J9",         # BNC jacks
               "J8"}                     # SWD pin header


def load_components():
    root = ET.parse(os.path.join(HERE, "relay_board.xml")).getroot()
    components = {}
    for comp in root.find(".//components"):
        reference = comp.get("ref")
        if reference.startswith("#") or reference in HAND_SOLDER:
            continue
        value = comp.find("value").text if comp.find("value") is not None else ""
        footprint_node = comp.find("footprint")
        footprint = footprint_node.text if footprint_node is not None else ""
        components[reference] = (value, footprint.split(":")[-1])
    return components


def sort_key(reference):
    return (reference.rstrip("0123456789"),
            int("".join(filter(str.isdigit, reference)) or 0))


def write_bom(components):
    groups = defaultdict(list)
    for reference, key in components.items():
        groups[key].append(reference)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Comment", "Designator", "Footprint",
               "JLCPCB Part #（optional）"])
    for (value, tail), references in sorted(groups.items(),
                                            key=lambda kv: sorted(
                                                kv[1], key=sort_key)[0]):
        references.sort(key=sort_key)
        part = PART_NUMBERS.get((value, tail), ("", ""))[0]
        comment, lcsc = JLC_BASIC.get(
            (value, tail), (jlc_comment(value, part), ""))
        ws.append([comment, ",".join(references),
                   jlc_footprint(tail), lcsc])

    destination = os.path.join(HERE, "relay_board_JLC_BOM.xlsx")
    wb.save(destination)
    print(f"JLC BOM: {destination} ({len(groups)} line items, "
          f"{len(components)} placements)")


def write_cpl(components):
    positions = os.path.join(HERE, "fab", "relay_board_pos.csv")
    if not os.path.exists(positions):
        raise SystemExit("fab/relay_board_pos.csv missing -- run export_fab.py")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
    count = 0
    with open(positions, newline="") as handle:
        for row in csv.DictReader(handle):
            reference = row["Ref"]
            if reference not in components:
                continue
            ws.append([reference,
                       f"{float(row['PosX']):.4f}mm",
                       f"{float(row['PosY']):.4f}mm",
                       "Top" if row["Side"] == "top" else "Bottom",
                       float(row["Rot"])])
            count += 1

    missing = set(components) - {
        row[0].value for row in ws.iter_rows(min_row=2)}
    if missing:
        raise SystemExit(f"components missing from pos file: {sorted(missing)}")

    destination = os.path.join(HERE, "relay_board_JLC_CPL.xlsx")
    wb.save(destination)
    print(f"JLC CPL: {destination} ({count} placements, all Top)")


def main():
    components = load_components()
    write_bom(components)
    write_cpl(components)


if __name__ == "__main__":
    main()
