/* Board identity, relay wear counters, and (rev C) coil supply + fixture
 * temperature. */

#ifndef BOARD_EXTRAS_H
#define BOARD_EXTRAS_H

#include <stdint.h>

/* 96-bit factory UID as 24 hex characters (NUL-terminated, 25 bytes). */
void board_serial(char out[25]);

/* Relay actuation counters (energize events), kept in RAM and persisted to
 * the last flash page: every COUNTER_SAVE_EVERY events, before DFU, and on
 * RELAY:COUNT:SAVE. */
void counters_load(void);
void counters_bump(uint32_t energized_mask);
void counters_poll(void);              /* saves when due; call when idle */
void counters_save(void);
uint32_t counters_get(unsigned relay);

#ifdef BOARD_REVC
/* Coil supply (TPS2051C enable on PB12, fault on PB13). Enabled once the USB host
 * has configured the board: inrush stays inside the pre-enumeration limit. */
void coil_supply_init(void);
void coil_supply_poll(int usb_configured);
int coil_supply_state(void);           /* 1 on, 0 off, -1 fault */

/* TMP117 on I2C2 (PB10 SCL / PB11 SDA, address 0x48): milli-degrees C, or
 * INT32_MIN if the sensor does not answer. */
void temperature_init(void);
int32_t temperature_millicelsius(void);
#endif

#endif
