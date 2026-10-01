"""
Generate the automated relay switching board for OpenMagnetics Lab.
Uses kicad-sch-api to create a proper KiCad 9 schematic.

Components:
- U1: STM32F042K6T6 (LQFP-32, crystal-less USB)
- U2, U3: ULN2003A (SOIC-16, Darlington relay drivers)
- K1-K9: G6K-2F-Y DC4.5 (DPDT signal relays)
- J1-J4: Conn_01x01 (DUT terminals A, B, C, D)
- J5-J6: Conn_01x01 (Bode100 connectors)
- R1-R4: Pull-up/pull-down resistors
- C1-C6: Decoupling capacitors
- U4: AMS1117-3.3 voltage regulator
"""

import os
from kicad_sch_api import create_schematic


def generate_schematic():
    sch = create_schematic("relay_board")
    sch.set_paper_size("A3")
    sch.set_title_block(
        title="OpenMagnetics Automated Relay Board",
        date="2026-04-01",
        company="OpenMagnetics Lab",
    )

    # =========================================================================
    # COMPONENT PLACEMENT
    # =========================================================================

    # --- MCU ---
    u1 = sch.components.add('MCU_ST_STM32F0:STM32F042K6Tx', 'U1', 'STM32F042K6Tx')
    u1.footprint = 'Package_QFP:LQFP-32_7x7mm_P0.8mm'
    u1.move(80, 80)

    # --- ULN2003A #1 (drives K1-K7) ---
    u2 = sch.components.add('Transistor_Array:ULN2003', 'U2', 'ULN2003A')
    u2.footprint = 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm'
    u2.move(140, 80)

    # --- ULN2003A #2 (drives K8-K9) ---
    u3 = sch.components.add('Transistor_Array:ULN2003', 'U3', 'ULN2003A')
    u3.footprint = 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm'
    u3.move(140, 130)

    # --- Voltage regulator ---
    u4 = sch.components.add('Regulator_Linear:AMS1117-3.3', 'U4', 'AMS1117-3.3')
    u4.footprint = 'Package_TO_SOT_SMD:SOT-223-3_TabPin2'
    u4.move(40, 170)

    # --- Relays K1-K9 ---
    # G6K-2 pins: 1=Coil+, 8=Coil-, 2=COM1, 3=NC1, 4=NO1, 5=NO2, 6=NC2, 7=COM2
    relay_x = 220
    relay_spacing = 30
    relays = {}
    for i in range(9):
        ref = f'K{i+1}'
        k = sch.components.add('Relay:G6K-2', ref, 'G6K-2F-RF-S')
        k.footprint = 'Relay_SMD:Relay_DPDT_Omron_G6K-2F-Y'
        k.move(relay_x, 40 + i * relay_spacing)
        relays[ref] = k

    # --- Resistors ---
    r1 = sch.components.add('Device:R', 'R1', '5.1k')
    r1.footprint = 'Resistor_SMD:R_0402_1005Metric'
    r1.move(30, 120)

    r2 = sch.components.add('Device:R', 'R2', '5.1k')
    r2.footprint = 'Resistor_SMD:R_0402_1005Metric'
    r2.move(40, 120)

    r3 = sch.components.add('Device:R', 'R3', '10k')
    r3.footprint = 'Resistor_SMD:R_0402_1005Metric'
    r3.move(50, 120)

    r4 = sch.components.add('Device:R', 'R4', '10k')
    r4.footprint = 'Resistor_SMD:R_0402_1005Metric'
    r4.move(60, 120)

    # --- Capacitors ---
    cap_x = 30
    for i in range(6):
        c = sch.components.add('Device:C', f'C{i+1}', '100nF')
        c.footprint = 'Capacitor_SMD:C_0402_1005Metric'
        c.move(cap_x + i * 12, 200)

    # --- DUT connectors ---
    for i, (ref, label) in enumerate([('J1', 'DUT_A'), ('J2', 'DUT_B'), ('J3', 'DUT_C'), ('J4', 'DUT_D')]):
        j = sch.components.add('Connector_Generic:Conn_01x01', ref, label)
        j.footprint = 'Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical'
        j.move(220 + i * 20, 320)

    # --- Bode100 connectors ---
    j5 = sch.components.add('Connector_Generic:Conn_01x01', 'J5', 'Bode100_1')
    j5.footprint = 'Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical'
    j5.move(220, 340)

    j6 = sch.components.add('Connector_Generic:Conn_01x01', 'J6', 'Bode100_2')
    j6.footprint = 'Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical'
    j6.move(250, 340)

    # =========================================================================
    # NET CONNECTIONS VIA LABELS
    # =========================================================================
    # Strategy: attach labels directly to pins using pin=(ref, pin_number)

    # --- MCU GPIO connections ---
    gpio_map = {
        '6': 'GPIO0',   # PA0
        '7': 'GPIO1',   # PA1
        '8': 'GPIO2',   # PA2
        '9': 'GPIO3',   # PA3
        '10': 'GPIO4',  # PA4
        '11': 'GPIO5',  # PA5
        '12': 'GPIO6',  # PA6
        '13': 'GPIO7',  # PA7
        '15': 'GPIO8',  # PB1
    }
    for pin_num, gpio_name in gpio_map.items():
        sch.add_label(gpio_name, pin=('U1', pin_num))

    # MCU USB pins
    sch.add_label('USB_DM', pin=('U1', '21'))  # PA11
    sch.add_label('USB_DP', pin=('U1', '22'))  # PA12

    # MCU debug pins
    sch.add_label('SWDIO', pin=('U1', '23'))   # PA13
    sch.add_label('SWCLK', pin=('U1', '24'))   # PA14

    # MCU NRST and BOOT0
    sch.add_label('NRST', pin=('U1', '4'))
    sch.add_label('BOOT0', pin=('U1', '2'))     # PF0 = BOOT0

    # MCU power
    sch.add_label('+3V3', pin=('U1', '1'))      # VDD
    sch.add_label('+3V3', pin=('U1', '5'))      # VDDA
    sch.add_label('+3V3', pin=('U1', '17'))     # VDDIO2
    sch.add_label('GND', pin=('U1', '16'))      # VSS
    sch.add_label('GND', pin=('U1', '32'))      # VSSA

    # Unused MCU pins - add labels to avoid ERC errors
    # PF1(3), PB0(14), PA8(18), PA9(19), PA10(20), PA15(25)
    # PB3-PB8(26-31)
    unused_mcu = ['3', '14', '18', '19', '20', '25', '26', '27', '28', '29', '30', '31']
    for pin_num in unused_mcu:
        sch.add_label(f'NC_U1_{pin_num}', pin=('U1', pin_num))

    # --- ULN2003A #1 connections (GPIO0-6 -> K1-K7) ---
    for i in range(7):
        sch.add_label(f'GPIO{i}', pin=('U2', str(i + 1)))       # Input
        sch.add_label(f'K{i+1}_COIL', pin=('U2', str(16 - i)))  # Output

    sch.add_label('GND', pin=('U2', '8'))
    sch.add_label('+5V', pin=('U2', '9'))

    # --- ULN2003A #2 connections (GPIO7-8 -> K8-K9) ---
    sch.add_label('GPIO7', pin=('U3', '1'))
    sch.add_label('GPIO8', pin=('U3', '2'))
    sch.add_label('K8_COIL', pin=('U3', '16'))
    sch.add_label('K9_COIL', pin=('U3', '15'))
    sch.add_label('GND', pin=('U3', '8'))
    sch.add_label('+5V', pin=('U3', '9'))

    # Unused ULN2003A #2 inputs - tie to GND, leave outputs unconnected
    for pin_num in ['3', '4', '5', '6', '7']:
        sch.add_label('GND', pin=('U3', pin_num))
    for pin_num in ['14', '13', '12', '11', '10']:
        sch.add_label(f'NC_U3_{pin_num}', pin=('U3', pin_num))

    # --- Relay connections ---
    # G6K-2 pins: 1=Coil+, 8=Coil-, 2=COM1, 3=NC1, 4=NO1, 5=NO2, 6=NC2, 7=COM2
    relay_nets = {
        'K1': {'1': '+5V', '8': 'K1_COIL', '2': 'Bode100_1', '3': 'A', '4': 'C'},
        'K2': {'1': '+5V', '8': 'K2_COIL', '2': 'Bode100_2', '3': 'B', '4': 'R3_COM'},
        'K3': {'1': '+5V', '8': 'K3_COIL', '2': 'R3_COM', '3': 'D', '4': 'C'},
        'K4': {'1': '+5V', '8': 'K4_COIL', '2': 'C', '4': 'D', '5': 'D', '7': 'C'},  # parallel poles
        'K5': {'1': '+5V', '8': 'K5_COIL', '2': 'A', '4': 'B', '5': 'B', '7': 'A'},  # parallel poles
        'K6': {'1': '+5V', '8': 'K6_COIL', '2': 'B', '4': 'C'},
        'K7': {'1': '+5V', '8': 'K7_COIL', '2': 'B', '4': 'D'},
        'K8': {'1': '+5V', '8': 'K8_COIL', '2': 'A', '4': 'C'},
        'K9': {'1': '+5V', '8': 'K9_COIL', '2': 'A', '4': 'D'},
    }

    for ref, pin_nets in relay_nets.items():
        for pin_num, net_name in pin_nets.items():
            sch.add_label(net_name, pin=(ref, pin_num))

        # Label unused relay pins
        all_pins = {'1', '2', '3', '4', '5', '6', '7', '8'}
        used_pins = set(pin_nets.keys())
        unused_pins = all_pins - used_pins
        for pin_num in unused_pins:
            sch.add_label(f'NC_{ref}_{pin_num}', pin=(ref, pin_num))

    # --- DUT terminal connections ---
    sch.add_label('A', pin=('J1', '1'))
    sch.add_label('B', pin=('J2', '1'))
    sch.add_label('C', pin=('J3', '1'))
    sch.add_label('D', pin=('J4', '1'))

    # --- Bode100 connections ---
    sch.add_label('Bode100_1', pin=('J5', '1'))
    sch.add_label('Bode100_2', pin=('J6', '1'))

    # --- Voltage regulator ---
    sch.add_label('VBUS', pin=('U4', '3'))    # Vin
    sch.add_label('+3V3', pin=('U4', '2'))    # Vout
    sch.add_label('GND', pin=('U4', '1'))     # GND

    # --- Pull-down resistors for USB CC ---
    sch.add_label('CC1', pin=('R1', '1'))
    sch.add_label('GND', pin=('R1', '2'))
    sch.add_label('CC2', pin=('R2', '1'))
    sch.add_label('GND', pin=('R2', '2'))

    # --- NRST pull-up ---
    sch.add_label('+3V3', pin=('R3', '1'))
    sch.add_label('NRST', pin=('R3', '2'))

    # --- BOOT0 pull-down ---
    sch.add_label('BOOT0', pin=('R4', '1'))
    sch.add_label('GND', pin=('R4', '2'))

    # --- Decoupling caps ---
    for i in range(3):
        sch.add_label('+3V3', pin=(f'C{i+1}', '1'))
        sch.add_label('GND', pin=(f'C{i+1}', '2'))
    sch.add_label('+5V', pin=('C4', '1'))
    sch.add_label('GND', pin=('C4', '2'))
    sch.add_label('+5V', pin=('C5', '1'))
    sch.add_label('GND', pin=('C5', '2'))
    sch.add_label('VBUS', pin=('C6', '1'))
    sch.add_label('GND', pin=('C6', '2'))

    # --- VBUS = +5V connection (direct for simplicity) ---
    sch.add_label('VBUS', position=(30, 240))
    sch.add_label('+5V', position=(40, 240))
    sch.add_wire((30, 240), (40, 240))

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = os.path.dirname(os.path.abspath(__file__))
    sch_path = os.path.join(output_dir, 'relay_board.kicad_sch')
    sch.save_as(sch_path)
    print(f"Schematic saved: {sch_path}")

    # Print statistics
    stats = sch.get_statistics()
    print(f"  Components: {stats.get('components', '?')}")
    print(f"  Labels: {stats.get('labels', '?')}")
    print(f"  Wires: {stats.get('wires', '?')}")
    print(f"  No-connects: {stats.get('no_connects', '?')}")

    return sch_path


if __name__ == "__main__":
    sch_path = generate_schematic()

    # Test with KiCad
    import subprocess
    result = subprocess.run(
        [r'C:\Program Files\KiCad\9.0\bin\kicad-cli.exe', 'sch', 'erc',
         '--format', 'report', '--severity-all',
         '-o', sch_path.replace('.kicad_sch', '_erc.txt'), sch_path],
        capture_output=True, text=True
    )
    print(f"\nKiCad ERC: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"ERC stderr: {result.stderr.strip()}")

    # Print ERC report
    erc_path = sch_path.replace('.kicad_sch', '_erc.txt')
    if os.path.exists(erc_path):
        with open(erc_path) as f:
            lines = f.readlines()
        # Print summary
        for line in lines:
            if 'Found' in line or 'ERC messages' in line:
                print(f"  {line.strip()}")
