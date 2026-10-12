#!/usr/bin/env bash
# Install the kicad-happy Claude Code skills used for the rev C review
# (kicad, emc, jlcpcb, lcsc, bom). MIT licensed, github.com/aklofas/kicad-happy.
set -euo pipefail
dest="${HOME}/.claude/skills"
work="$(mktemp -d)"
git clone --depth 1 https://github.com/aklofas/kicad-happy.git "${work}/kicad-happy"
mkdir -p "${dest}"
for skill in kicad emc jlcpcb lcsc bom; do
    rm -rf "${dest:?}/${skill}"
    cp -r "${work}/kicad-happy/skills/${skill}" "${dest}/${skill}"
done
rm -rf "${work}"
echo "installed into ${dest}: kicad emc jlcpcb lcsc bom"
