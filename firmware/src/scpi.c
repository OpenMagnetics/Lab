/*
 * SCPI command parser for the relay board.
 *
 * Grammar (newline-terminated, case-insensitive, ; separates commands):
 *
 *   *IDN?                 -> OpenMagnetics,RelayBoard,revB,<version>
 *   *RST                  release everything, DUT connected
 *   *OPC?                 -> 1        (relays settle synchronously, so
 *                                      completion is immediate by the time
 *                                      the reply is sent)
 *   *TST?                 -> 0 on pass, error code on fail
 *   CONF:MEAS <1..15>     select measurement configuration
 *   CONF:MEAS?            -> current configuration (0 = none)
 *   CAL:MODE MEAS|OPEN|SHORT|LOAD
 *   CAL:MODE?             -> current mode
 *   RELAY <0..17>,<0|1>   manual override of one relay (leaves config = 0)
 *   RELAY:ALL?            -> 18 comma-separated coil states, K1 first
 *   SYST:ERR?             -> oldest queued error, "0,\"No error\"" when empty
 *   SYST:DFU              release all relays, detach USB and reboot into the
 *                         ROM DFU bootloader (no reply; firmware update)
 *
 * Matches the driver in scripts/RelayBoardController.py.
 */

#include <stddef.h>
#include <string.h>

#include "scpi.h"

#define VERSION "1.1.1"
#define LINE_MAX 96
#define ERROR_QUEUE 8

static scpi_write_fn write_out;
static char line[LINE_MAX];
static volatile uint32_t line_length;
static volatile uint8_t line_ready;

static int16_t error_queue[ERROR_QUEUE];
static uint8_t error_head, error_count;

static void push_error(int code)
{
    if (code == 0) {
        return;
    }
    if (error_count < ERROR_QUEUE) {
        error_queue[(error_head + error_count) % ERROR_QUEUE] = (int16_t)code;
        error_count++;
    }
}

static void reply(const char *text)
{
    write_out(text, (uint32_t)strlen(text));
    write_out("\n", 1);
}

static void reply_number(int32_t value)
{
    char buffer[12];
    char *cursor = buffer + sizeof buffer;
    uint32_t magnitude = value < 0 ? (uint32_t)(-value) : (uint32_t)value;
    *--cursor = '\0';
    do {
        *--cursor = (char)('0' + magnitude % 10u);
        magnitude /= 10u;
    } while (magnitude);
    if (value < 0) {
        *--cursor = '-';
    }
    reply(cursor);
}

/* ------------------------------------------------------------------ lexing */

static int starts(const char *text, const char *keyword)
{
    while (*keyword) {
        char a = *text, b = *keyword;
        if (a >= 'a' && a <= 'z') a = (char)(a - 32);
        if (b >= 'a' && b <= 'z') b = (char)(b - 32);
        if (a != b) {
            return 0;
        }
        text++;
        keyword++;
    }
    return 1;
}

static const char *skip_spaces(const char *text)
{
    while (*text == ' ' || *text == '\t') {
        text++;
    }
    return text;
}

static int parse_number(const char **cursor, uint32_t *value)
{
    const char *text = skip_spaces(*cursor);
    if (*text < '0' || *text > '9') {
        return 0;
    }
    uint32_t result = 0;
    while (*text >= '0' && *text <= '9') {
        result = result * 10u + (uint32_t)(*text - '0');
        text++;
    }
    *cursor = text;
    *value = result;
    return 1;
}

/* --------------------------------------------------------------- dispatch */

static void execute(const char *command)
{
    command = skip_spaces(command);
    if (*command == '\0') {
        return;
    }

    if (starts(command, "*IDN?")) {
        reply("OpenMagnetics,RelayBoard,revB," VERSION);
    } else if (starts(command, "*RST")) {
        push_error(scpi_action_reset());
    } else if (starts(command, "*OPC?")) {
        reply("1");
    } else if (starts(command, "*TST?")) {
        int result = scpi_action_self_test();
        reply_number(result);
        push_error(result);
    } else if (starts(command, "CONF:MEAS?")) {
        reply_number((int32_t)scpi_query_config());
    } else if (starts(command, "CONF:MEAS")) {
        const char *cursor = command + 9;
        uint32_t number;
        if (parse_number(&cursor, &number)) {
            push_error(scpi_action_set_config(number));
        } else {
            push_error(SCPI_ERR_DATA_OUT_OF_RANGE);
        }
    } else if (starts(command, "CAL:MODE?")) {
        static const char *names[] = {"MEAS", "OPEN", "SHORT", "LOAD"};
        reply(names[scpi_query_cal_mode()]);
    } else if (starts(command, "CAL:MODE")) {
        const char *argument = skip_spaces(command + 8);
        if (starts(argument, "MEAS")) {
            push_error(scpi_action_cal_mode(CAL_MODE_MEAS));
        } else if (starts(argument, "OPEN")) {
            push_error(scpi_action_cal_mode(CAL_MODE_OPEN));
        } else if (starts(argument, "SHORT")) {
            push_error(scpi_action_cal_mode(CAL_MODE_SHORT));
        } else if (starts(argument, "LOAD")) {
            push_error(scpi_action_cal_mode(CAL_MODE_LOAD));
        } else {
            push_error(SCPI_ERR_DATA_OUT_OF_RANGE);
        }
    } else if (starts(command, "RELAY:ALL?")) {
        char buffer[RELAY_COUNT * 2];
        uint32_t word = scpi_query_relay_word();
        for (uint32_t index = 0; index < RELAY_COUNT; index++) {
            buffer[index * 2] = (word & (1u << index)) ? '1' : '0';
            buffer[index * 2 + 1] = (index + 1 < RELAY_COUNT) ? ',' : '\0';
        }
        reply(buffer);
    } else if (starts(command, "RELAY")) {
        const char *cursor = command + 5;
        uint32_t index, state;
        if (parse_number(&cursor, &index)) {
            cursor = skip_spaces(cursor);
            if (*cursor == ',') {
                cursor++;
            }
            if (parse_number(&cursor, &state)) {
                push_error(scpi_action_set_relay(index, state));
                return;
            }
        }
        push_error(SCPI_ERR_DATA_OUT_OF_RANGE);
    } else if (starts(command, "SYST:DFU")) {
        push_error(scpi_action_enter_dfu());
    } else if (starts(command, "SYST:ERR?")) {
        if (error_count == 0) {
            reply("0,\"No error\"");
        } else {
            int16_t code = error_queue[error_head];
            error_head = (uint8_t)((error_head + 1) % ERROR_QUEUE);
            error_count--;
            reply_number(code);
        }
    } else {
        push_error(SCPI_ERR_UNDEFINED_HEADER);
    }
}

/* ---------------------------------------------------------------- public */

void scpi_init(scpi_write_fn write)
{
    write_out = write;
    line_length = 0;
    line_ready = 0;
    error_head = error_count = 0;
}

void scpi_feed_byte(uint8_t byte)
{
    if (line_ready) {
        return;                        /* previous line not yet processed */
    }
    if (byte == '\n' || byte == '\r') {
        if (line_length > 0) {
            line[line_length] = '\0';
            line_ready = 1;
        }
        return;
    }
    if (line_length < LINE_MAX - 1) {
        line[line_length++] = (char)byte;
    }
}

void scpi_poll(void)
{
    if (!line_ready) {
        return;
    }
    /* Split on ';' for compound commands. */
    char *cursor = line;
    while (cursor) {
        char *next = strchr(cursor, ';');
        if (next) {
            *next = '\0';
            next++;
        }
        execute(cursor);
        cursor = next;
    }
    line_length = 0;
    line_ready = 0;
}
