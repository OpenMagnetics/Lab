"""
Update the relay board firmware over USB, no BOOT0 strap needed.

    python scripts/update_firmware.py [firmware.bin]      (or: make -C firmware update)

Needs firmware >= 1.1.0 already on the board (for SYST:DFU). A blank or
pre-1.1.0 board must be strapped into the bootloader once (BOOT0 high at
power-up: bridge the left pads of R4 and R3), then this script still flashes
it -- it skips SYST:DFU when the bootloader is already present.

Flashing uses STM32CubeProgrammer's CLI, found on PATH, in a standalone
STM32CubeProgrammer install, or bundled inside STM32CubeIDE. On Windows the
bootloader (0483:DF11) needs the WinUSB driver once per PC (Zadig).
"""

import glob
import os
import shutil
import subprocess
import sys
import time

from RelayBoardController import RelayBoardController

FIRMWARE = os.path.join(os.path.dirname(__file__), "..", "firmware", "build", "relay_board.bin")
FLASH_BASE = "0x08000000"


def find_programmer():
    found = shutil.which("STM32_Programmer_CLI")
    if found:
        return found
    patterns = [
        r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
        r"C:\ST\STM32CubeIDE_*\STM32CubeIDE\plugins\*cubeprogrammer*\tools\bin\STM32_Programmer_CLI.exe",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]          # newest bundled version
    raise RuntimeError("STM32_Programmer_CLI not found; install STM32CubeProgrammer "
                       "or put it on PATH.")


def bootloader_present(programmer):
    listing = subprocess.run([programmer, "-l", "usb"], capture_output=True, text=True)
    return "Device Index" in listing.stdout


def wait_for(condition, timeout_s, what):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out after {timeout_s} s waiting for {what}.")


def board_identity():
    try:
        board = RelayBoardController()
    except Exception:
        return None
    identity = board.visa_session.query("*IDN?").strip()
    board.visa_session.close()
    return identity


def main():
    firmware = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else FIRMWARE)
    if not os.path.isfile(firmware):
        raise SystemExit(f"{firmware} not found -- run `make -C firmware` first.")
    programmer = find_programmer()

    if bootloader_present(programmer):
        print("Bootloader already present (strapped board), skipping SYST:DFU")
    else:
        board = RelayBoardController()
        print("Rebooting into the DFU bootloader")
        board.enter_dfu()
        wait_for(lambda: bootloader_present(programmer), 15, "the DFU bootloader")

    print(f"Flashing {firmware}")
    result = subprocess.run([programmer, "-c", "port=usb1", "-w", firmware, FLASH_BASE,
                             "-v", "-s", FLASH_BASE], capture_output=True, text=True)
    if "Download verified successfully" not in result.stdout:
        print(result.stdout[-2000:])
        raise SystemExit("Flashing failed.")
    print("Flash written and verified")

    identity = []
    wait_for(lambda: identity.append(board_identity()) or identity[-1], 30,
             "the board to re-enumerate")
    print(f"Running: {identity[-1]}")


if __name__ == "__main__":
    main()
