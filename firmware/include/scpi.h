/* Minimal SCPI-ish command parser for the relay board. */

#ifndef SCPI_H
#define SCPI_H

#include <stdint.h>
#include "relay_map.h"

/* SCPI standard error codes (negative per the spec). */
#define SCPI_ERR_NONE               0
#define SCPI_ERR_UNDEFINED_HEADER (-113)
#define SCPI_ERR_DATA_OUT_OF_RANGE (-222)
#define SCPI_ERR_SETTINGS_CONFLICT (-221)
#define SCPI_ERR_SELF_TEST         (-330)

typedef void (*scpi_write_fn)(const char *data, uint32_t length);

void scpi_init(scpi_write_fn write);
void scpi_feed_byte(uint8_t byte);
void scpi_poll(void);

/* Implemented by main.c -- the parser calls these. */
int scpi_action_reset(void);
int scpi_action_set_config(uint32_t number);
uint32_t scpi_query_config(void);
int scpi_action_cal_mode(cal_mode_t mode);
cal_mode_t scpi_query_cal_mode(void);
int scpi_action_set_relay(uint32_t index, uint32_t state);
uint32_t scpi_query_relay_word(void);
int scpi_action_self_test(void);

#endif
