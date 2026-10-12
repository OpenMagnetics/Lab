---
name: revc-design-decisions
description: "Rev C relay board decisions made with the user on 2026-10-11 (Kelvin, links, no SHORT relays, 4 layers, EMC, routing rules) and why"
metadata:
  node_type: memory
  type: project
  originSessionId: 873f79ad-e1f0-432e-b64f-4a0fd8f24ab4
  modified: 2026-10-11T14:55:37.815Z
---

Rev C (hardware/relay-board-revC, KiCad 10; called "rev B2" until the user renamed it rev C on
2026-10-12, so the future 6-terminal board is now rev D) is a full redesign answering N. Rosano's review
(2-wire contact R, per-path nH, open-relay pF, cal plane at the DUT joint). User decisions 2026-10-11:

- Full Kelvin via relays (pole 1 force, pole 2 sense of every DPDT), tie relay for 2-wire mode,
  OPA356 difference front-end switched onto CH1. Direct link relays A-B, C-D, B-C, B-D.
- Four SHORT relays were built then DROPPED after simulation + online check (user said: review deeply,
  remove if useless): same-size residual, extra Cps error; clamp-plane fixture compensation
  (Keysight/Voltech practice) replaces them. Their slots became a standards column (100R/1R/10pF).
- User wants "all calibration necessary": per-path OSL + post-OSL check standards + one-time
  clamp-plane open/short compensation (FixtureCompensation.py, bench.py fixture-compensation).
- 4 layers JLC04161H-7628 SIG/GND/GND/SIG (user chose it over 6 after the trade-off: 6 bought no
  room inside the measurement void); must pass EU EMC (EN 61326-1); JLC rules and stackup in the
  board file; unified ground; all cables on one edge.

**Why:** the user wants a board whose integrity survives an expert review; they prefer evidence
(simulation + sources) before removing features they approved.
**How to apply:** don't re-add SHORT relays or an AGND split without new evidence; keep
RelayMapRevC.py, verify_board.py and generate_firmware_map.py in sync with the schematic.
Related: [[bench-duts-and-setup]]

Later the same day: DUT interface = 8 banana jacks (Kelvin F/S pairs, Grayhill 02-1 clips);
4 terminals for rev C (6 for rev D); G6K-2F-Y relays for the prototype; no USB isolator; added
TMP117, UID serial, relay counters, test pads, TPS2051C coil switch, VBUS TVS, USB shield RC
option, status LEDs, plated GND holes + cable-edge bonding strip. Faraday (OpenConverters) is
used for EMC review via hardware/relay-board-revC/faraday_review.py; the user wants every
Faraday observation logged in FARADAY_FEEDBACK.md.

Routing: the user does not trust the autorouter -- measurement nets, supplies and the 23 coil
returns are routed by our own A* maze router (maze_route.py) and locked; freerouting only
finishes digital nets; optimize_tracks.py then pulls tracks taut (user asked "optimize the
tracks"). 2026-10-12 the user asked for signals on top/bottom only plus coupling best practice:
measurement nets are outer-layer-only; inner layers inside the void carry only the +5V coil feed
and coil returns; a crosstalk cost (K = 1/(1+(d/H)^2), same layer + broadside across the 0.21 mm
prepreg) steers measurement away from coil/supply/digital copper; parasitic_report audits it.
EU EMC target: EN 61326-1 / CISPR 11 Class B (EMC_COMPLIANCE.md); DUT-jack ESD parts deferred
to the future by the user.

2026-10-12 handoff: all work committed and pushed to claude/exciting-allen-152izs (645a5f7) before
the user moved to another PC; routing (own router, no freerouting) was mid fine pass -- resume from
hardware/relay-board-revC/TODO.md "HANDOFF" section. User wants no autorouter dependence.
