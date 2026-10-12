# Faraday (OpenConverters) -- user feedback log

Feedback from using [Faraday](https://github.com/OpenConverters/Faraday) (commit cloned
2026-10-11, `faraday_cli`) on the OpenMagnetics relay board: rev B (KiCad 9, 4 layers,
routed fab board) and rev C (KiCad 10, 6 layers, JLC06161H-3313). Entry point here:
`python faraday_review.py [board]`. New items are appended at the end with a date.

## What works well

- Findings are actionable: mechanism, a number, a confidence statement, a remediation.
  The calibration caveat ("closed form ~6.5 dB optimistic, ranking reliable") is honest
  and exactly what a reviewer needs.
- It found real issues on rev C that our own audits missed: 96 mm open stubs on the
  sense rails (400 MHz resonators), decoupling caps reaching ground through 13-15 mm,
  the USB ESD clamp 5 mm from the pin (~122 V overshoot), and connector ground spread.
- On rev B it independently confirmed the problems rev C was redesigned for (no
  reference plane, USB clamp 12 mm away, USB pair skew 17 mm, decoupling 13 mm away).
- Strict stackup validation and no silent default stackup.
- Fast: a full 6-layer, 117-part board in seconds.
- Bundle reporting (24 GPIO lines as one group with summed NEXT) is better than
  276 pairwise findings.

## Problems and suggestions

1. **Does not build with MSVC.** `M_PI` is used in GerberImporter, KicadImporter,
   Tline, Screener and Fixes; MSVC only defines it with `_USE_MATH_DEFINES`. Built here
   with `-DCMAKE_CXX_FLAGS="/D_USE_MATH_DEFINES /EHsc /utf-8"`. Suggest
   `std::numbers::pi` (the code is already C++20) and a Windows CI job.
2. **KiCad 10 boards fail with an opaque error** (`faraday: SExpr: not a number:
   '/+3V3'`). KiCad 10 (file version 20260206) writes `(net "/NAME")` with no numeric
   code and no top-level net table. Suggest keying nets by name (works for v5-v10), or
   at least detecting the version and saying "KiCad 10 is not supported yet". Workaround:
   `faraday_review.py` writes a KiCad-9-shaped copy (numbers every net, restores the table).
3. **`faraday_cli --help` is read as a board path** ("cannot open --help"). A usage
   message for `--help`/`-h`/no arguments would help.
4. **BNC cable entries missed by `connector-ground-spread` on rev B.** It paired USB-C
   with the SWD header but ignored the three BNCs, whose shells sit on a net named
   `AGND`. Coax shell pads (and AGND/PGND/CHASSIS/SHIELD nets) should count as cable
   grounds; on rev B the AGND-to-GND split *is* the cable-CM problem the rule exists for.
5. **Only the nearest plane counts as a reference.** On rev C the In2 sense rails are
   referenced by a GND strip on In4 (0.67 mm away) while In1 (0.55 mm, voided there) is
   nearer; the earlier run flagged "In1 does not reach there". Considering every plane
   within a few dielectrics, weighted by distance, would be more faithful.
6. **No way to declare intentional voids from the CLI.** Measurement fixtures void the
   planes under high-impedance nodes on purpose (here: DUT arms, terminal buses, links,
   for Cps accuracy). They rank as HIGH return-path breaks. The GUI has per-finding
   dismissal, but CI needs a waiver file (net patterns or named rule areas + a reason).
   Workaround: `faraday_review.py` applies such a list and prints the reasons.
7. **Findings carry net codes, not names** (`netA`/`netB` are integers), so every script
   has to join `board.nets`. Including the names in each finding would help.
8. **`severity` is a float score** while `severityLabel` is the category; the README
   calls them severities. Worth documenting the score's meaning and range.

## Domain notes (relay matrix / impedance fixture use)

- The return-path rule is right for the force rails: their loop is the radiating
  structure, and a GND strip under them is calibrated out because they always sit at
  the HI or LO potential. A fixture-aware mode could distinguish "nodes at a fixed
  potential" (planes fine) from "floating / intermediate-potential nodes" (planes cost
  accuracy), which is the real trade-off on this class of board.
