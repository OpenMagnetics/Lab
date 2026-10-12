/*
 * OpenMagnetics relay board rev B -- STM32F072CBT6 firmware.
 *
 * USB CDC-ACM device exposing a SCPI command set (see scpi.c).  Built on
 * libopencm3; see the Makefile for toolchain and flashing instructions.
 *
 * Clocking: HSI48 + CRS auto-trim from USB SOF (crystal-less USB, native
 * on the F072).
 *
 * Relay drive: 18 (rev B) / 24 (rev C) GPIOs -> TBD62003A -> G6K coils.  Relay state
 * changes settle for RELAY_SETTLE_MS before *OPC? reports complete, so the
 * host can trust that a measurement started after OPC sees settled contacts.
 */

#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/stm32/crs.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/iwdg.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/st_usbfs.h>
#include <libopencm3/stm32/syscfg.h>

#include "board_extras.h"
#include "relay_map.h"
#include "scpi.h"
#include "usb_cdc.h"

#define RELAY_SETTLE_MS 10u   /* G6K operate/release 3 ms + margin + bounce */
/* Coils are energized one at a time, this far apart: limits the di/dt and
 * the VBUS droop the USB cable sees when a whole configuration switches
 * (EMC pre-compliance; releasing is not staggered). */
#define RELAY_STAGGER_MS 2u

static const uint8_t relay_gpio[RELAY_COUNT] = RELAY_GPIO_TABLE;
static const uint32_t config_table[CONFIG_COUNT] = CONFIG_TABLE;
#ifdef BOARD_REVC
static const uint32_t cal_open_table[CONFIG_COUNT] = CAL_OPEN_TABLE;
static const uint32_t cal_short_table[CONFIG_COUNT] = CAL_SHORT_TABLE;
static const uint32_t cal_load_table[CONFIG_COUNT] = CAL_LOAD_TABLE;
static uint8_t kelvin_mode;          /* 1: sense rails to the front-end */
#endif

static volatile uint32_t milliseconds;
static uint32_t relay_word;          /* current coil states, bit n = K(n+1) */
static uint8_t current_config;      /* 1..CONFIG_COUNT, 0 = none           */
static cal_mode_t cal_mode = CAL_MODE_MEAS;

/* SysTick runs at TICK_HZ; `milliseconds` advances every TICK_HZ/1000 ticks.
 *
 * RUN LED (high = lit). Until the USB host has configured the board it
 * blinks at 1 Hz (firmware alive, no host); afterwards it is on. Rev B's LED
 * hangs on PC13 through a 1k pull-up, so it is additionally PWM-dimmed to
 * 2 % there; rev C has a 4.7k series resistor and needs no dimming. */
#define TICK_HZ          10000u
#define LED_PERIOD_TICKS 100u
#ifdef BOARD_REVC
#define LED_ON_TICKS     LED_PERIOD_TICKS
#else
#define LED_ON_TICKS     2u
#endif

/* RUN LED: PA9 on rev C (south row, beside the other status LEDs), PC13 on rev B. */
#ifdef BOARD_REVC
#define RUN_PORT    GPIOA
#define RUN_PIN     GPIO9
#else
#define RUN_PORT    GPIOC
#define RUN_PIN     GPIO13
#endif

void sys_tick_handler(void)
{
    static uint8_t sub_ms;
    static uint8_t led_phase;

    if (++sub_ms >= TICK_HZ / 1000u) {
        sub_ms = 0u;
        milliseconds++;
    }
    if (++led_phase >= LED_PERIOD_TICKS) {
        led_phase = 0u;
    }
    int lit = led_phase < LED_ON_TICKS;
    if (!usb_cdc_configured() && (milliseconds % 1000u) >= 500u) {
        lit = 0;                         /* no host yet: 1 Hz blink */
    }
    if (lit) {
        gpio_set(RUN_PORT, RUN_PIN);
    } else {
        gpio_clear(RUN_PORT, RUN_PIN);
    }
}

/* Relay pin codes are 16 * port + bit (generate_firmware_map.py):
 * port A=0, B=1, C=2, F=3. */
static uint32_t relay_port(uint8_t code)
{
    switch (code >> 4) {
    case 0u: return GPIOA;
    case 1u: return GPIOB;
    case 2u: return GPIOC;
    default: return GPIOF;
    }
}

/* Rev C status LEDs on the south row: PA8 BUSY, PB15 KELVIN, PB14 PROG (PWR
 * is hardwired). PB12 is the coil-supply enable, PB13 its fault; PB10/PB11
 * are I2C2 (TMP117). Every relay-drive pin faces the drivers (relay_map_revc.h). */
#ifdef BOARD_REVC
#define LED_BUSY   GPIOA, GPIO8
#define LED_KELVIN GPIOB, GPIO15
#define LED_PROG_PORT GPIOB
#define LED_PROG_PIN  GPIO14
#define FIXED_A     (GPIO8 | GPIO9)                      /* BUSY, RUN */
#define FIXED_C     0u
#define FIXED_B     (GPIO10 | GPIO11 | GPIO12 | GPIO13 | GPIO14 | GPIO15)
static void led_set(uint32_t port, uint16_t pin, int on)
{
    if (on) {
        gpio_set(port, pin);
    } else {
        gpio_clear(port, pin);
    }
}
#define led(which, on) led_set(which, on)
#else
#define FIXED_A     0u
#define FIXED_B     0u
#define FIXED_C     GPIO13
#define led(which, on) ((void)0)
#endif

static void delay_ms(uint32_t amount)
{
    uint32_t start = milliseconds;
    while ((milliseconds - start) < amount) {
        __asm__("wfi");
    }
}

/* ------------------------------------------------------- DFU bootloader */

/* SYST:DFU re-enters the ROM USB DFU bootloader without the BOOT0 strap, so
 * a flashed board can be updated over USB alone.  The jump is made right
 * after a reset, with the core in its reset state (HSI 8 MHz, no peripherals,
 * no interrupts) -- what the ROM expects.  The request crosses the reset in
 * .noinit RAM, which the startup code neither loads nor zeroes.
 *
 * Power-up SRAM is random (and can retain old contents through a short
 * unplug), so the request is honoured only after a pure software reset, and
 * only if both the magic and its complement are present. */
#define DFU_MAGIC     0xDF00B007u
#define SYSTEM_MEMORY 0x1FFFC800u      /* F072 ROM bootloader (AN2606) */

static uint32_t dfu_request[2] __attribute__((section(".noinit")));
static volatile uint8_t dfu_pending;
static uint32_t reset_cause;            /* RCC_CSR as found at boot */

/* ------------------------------------------------- watchdog recovery
 *
 * EMC immunity (EN 61326-1, ESD and radiated RF at criterion B) requires
 * the board to recover by itself, without a relay changing state. The
 * independent watchdog (LSI, cannot be stopped once started) resets a hung
 * core; the relay state crosses that reset in .noinit RAM, guarded by a
 * magic and a complement, and is re-applied (staggered) as soon as the coil
 * supply is back. Only a watchdog reset restores: power-up and SYST:DFU
 * start released, as before. */
#define WATCHDOG_PERIOD_MS 1000u        /* longest blocking step: ~60 ms */
#define RESTORE_MAGIC      0x5EA7B0A7u

struct restore_state {
    uint32_t magic;
    uint32_t word;
    uint32_t settings;                  /* config | cal_mode << 8 | kelvin << 16 */
    uint32_t check;                     /* ~(magic ^ word ^ settings) */
};
static struct restore_state restore __attribute__((section(".noinit")));
static uint32_t pending_restore;
static uint8_t restore_due;

static void relay_apply(uint32_t word);

static void enter_bootloader_if_requested(void)
{
    reset_cause = RCC_CSR;
    RCC_CSR |= RCC_CSR_RMVF;            /* next boot sees only its own cause */
    int requested = (reset_cause & RCC_CSR_SFTRSTF)
                    && !(reset_cause & RCC_CSR_PORRSTF)
                    && dfu_request[0] == DFU_MAGIC
                    && dfu_request[1] == ~DFU_MAGIC;
    dfu_request[0] = 0u;
    dfu_request[1] = 0u;
    if (!requested) {
        return;
    }
#ifdef BOARD_REVC
    /* PROG LED on before the jump: the ROM bootloader (AN2606: USART1
     * PA9/PA10, USART2 PA14/PA15, USB PA11/PA12) never touches PB14, so it
     * stays lit for the whole firmware update. */
    rcc_periph_clock_enable(RCC_GPIOB);
    gpio_mode_setup(LED_PROG_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED_PROG_PIN);
    gpio_set(LED_PROG_PORT, LED_PROG_PIN);
#endif
    rcc_periph_clock_enable(RCC_SYSCFG_COMP);
    SYSCFG_CFGR1 = (SYSCFG_CFGR1 & ~(uint32_t)SYSCFG_CFGR1_MEM_MODE)
                   | SYSCFG_CFGR1_MEM_MODE_SYSTEM;
    const volatile uint32_t *vectors = (const volatile uint32_t *)SYSTEM_MEMORY;
    uint32_t stack = vectors[0];
    uint32_t entry = vectors[1];
    __asm__ volatile("msr msp, %0\n\tbx %1" : : "r"(stack), "r"(entry));
    for (;;) {
    }
}

int scpi_action_enter_dfu(void)
{
    dfu_pending = 1u;                   /* acted on from the main loop */
    return SCPI_ERR_NONE;
}

static void reboot_into_bootloader(void)
{
    relay_apply(0u);                    /* leave every relay released */
    counters_save();                    /* wear counters survive the update */
    *USB_BCDR_REG &= ~(uint32_t)USB_BCDR_DPPU;   /* drop the D+ pull-up: host sees */
    delay_ms(100u);                     /* an unplug before DFU appears   */
    dfu_request[0] = DFU_MAGIC;
    dfu_request[1] = ~DFU_MAGIC;
    scb_reset_system();
}

/* ------------------------------------------------------------------ clock */

static void clock_setup(void)
{
    /* HSI48 with CRS auto-trim from USB SOF: the F072's crystal-less USB.
     * (rev B dropped the crystal: the F072 CRS is proven silicon, and the
     * crystal cluster crowded the MCU corner of the board.) */
    rcc_clock_setup_in_hsi48_out_48mhz();
    rcc_periph_clock_enable(RCC_CRS);
    crs_autotrim_usb_enable();
    rcc_set_usbclk_source(RCC_HSI48);

    rcc_periph_clock_enable(RCC_GPIOA);
    rcc_periph_clock_enable(RCC_GPIOB);
    rcc_periph_clock_enable(RCC_GPIOC);
    rcc_periph_clock_enable(RCC_GPIOF);

    systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
    systick_set_reload(48000000 / TICK_HZ - 1);
    systick_interrupt_enable();
    systick_counter_enable();
}

/* ----------------------------------------------------------------- relays */

static void relay_gpio_setup(void)
{
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        uint32_t port = relay_port(code);
        uint16_t pin = 1u << (code & 15u);
        gpio_mode_setup(port, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, pin);
        gpio_clear(port, pin);
    }
    /* RUN LED (sys_tick_handler) and rev C status LEDs on the south row. */
    gpio_mode_setup(RUN_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, RUN_PIN);
#ifdef BOARD_REVC
    gpio_mode_setup(GPIOA, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO8);
    gpio_mode_setup(GPIOB, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO14 | GPIO15);
    gpio_clear(GPIOA, GPIO8);
    gpio_clear(GPIOB, GPIO14 | GPIO15);
#endif

    /* Every other pin to analog mode: no floating digital inputs (lowest
     * current and no stray toggling for EMC). USB (PA11/12) and SWD
     * (PA13/14) are left alone; outputs keep the reset-default low slew.
     * Ports A, B, C, F (index 0..3, as in the relay pin codes). */
    uint32_t used[4] = {(1u << 11) | (1u << 12) | (1u << 13) | (1u << 14) | FIXED_A,
                        FIXED_B, FIXED_C, 0u};
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        used[(code >> 4) & 3u] |= 1u << (code & 15u);
    }
    static const uint32_t ports[4] = {GPIOA, GPIOB, GPIOC, GPIOF};
    for (unsigned port = 0; port < 4u; port++) {
        uint16_t spare = (uint16_t)(~used[port] & 0xFFFFu);
        if (port == 2u) {
            spare &= GPIO13 | GPIO14 | GPIO15;           /* the F072CB has PC13..15 only */
        } else if (port == 3u) {
            spare &= GPIO0 | GPIO1;                      /* and PF0/PF1 */
        }
        if (spare) {
            gpio_mode_setup(ports[port], GPIO_MODE_ANALOG, GPIO_PUPD_NONE, spare);
        }
    }
}

static void restore_save(void)
{
    if (restore_due) {
        return;                         /* a restore is still owed: keep it */
    }
    uint32_t settings = (uint32_t)current_config | ((uint32_t)cal_mode << 8);
#ifdef BOARD_REVC
    settings |= (uint32_t)kelvin_mode << 16;
#endif
    restore.magic = RESTORE_MAGIC;
    restore.word = relay_word;
    restore.settings = settings;
    restore.check = ~(RESTORE_MAGIC ^ relay_word ^ settings);
}

/* After a watchdog reset: take back config, calibration mode, Kelvin flag
 * and the coil word; the word itself is applied from the main loop. */
static void restore_load(void)
{
    int valid = (reset_cause & RCC_CSR_IWDGRSTF)
                && !(reset_cause & RCC_CSR_PORRSTF)
                && restore.magic == RESTORE_MAGIC
                && restore.check == ~(RESTORE_MAGIC ^ restore.word ^ restore.settings);
    restore.magic = 0u;
    if (!valid) {
        return;
    }
    current_config = (uint8_t)(restore.settings & 0xFFu);
    cal_mode = (cal_mode_t)((restore.settings >> 8) & 0xFFu);
#ifdef BOARD_REVC
    kelvin_mode = (uint8_t)((restore.settings >> 16) & 1u);
    led(LED_KELVIN, kelvin_mode);
#endif
    pending_restore = restore.word;
    restore_due = pending_restore != 0u;
}

static void relay_apply(uint32_t word)
{
    led(LED_BUSY, 1);
    /* Release first (break before make), then energize one coil at a time. */
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        uint32_t port = relay_port(code);
        uint16_t pin = 1u << (code & 15u);
        if (!(word & (1u << index))) {
            gpio_clear(port, pin);
        }
    }
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        uint32_t port = relay_port(code);
        uint16_t pin = 1u << (code & 15u);
        if ((word & (1u << index)) && !(relay_word & (1u << index))) {
            gpio_set(port, pin);
            delay_ms(RELAY_STAGGER_MS);
        }
    }
    counters_bump(word & ~relay_word);
    relay_word = word;
    restore_save();
    delay_ms(RELAY_SETTLE_MS);
    led(LED_BUSY, 0);
}

static unsigned energized_count(uint32_t word)
{
    unsigned count = 0u;
    while (word) {
        count += word & 1u;
        word >>= 1;
    }
    return count;
}

/* Compose the full relay word: measurement matrix + calibration overlay. */
#ifdef BOARD_REVC
static uint32_t compose_word(void)
{
    if (current_config < 1u || current_config > CONFIG_COUNT) {
        return 0u;
    }
    uint32_t index = current_config - 1u;
    uint32_t word = config_table[index];   /* floating terminals isolated */
    if (cal_mode == CAL_MODE_OPEN) {
        word = cal_open_table[index];
    } else if (cal_mode == CAL_MODE_SHORT) {
        word = cal_short_table[index];
    } else if (cal_mode == CAL_MODE_LOAD) {
        word = cal_load_table[index];
    }
    return kelvin_mode ? (word | MASK_KELVIN) : word;
}

int scpi_action_set_kelvin(uint32_t state)
{
    kelvin_mode = state ? 1u : 0u;
    led(LED_KELVIN, kelvin_mode);
    if (current_config) {
        relay_apply(compose_word());
    }
    return 0;
}

uint32_t scpi_query_kelvin(void)
{
    return kelvin_mode;
}
#else
static const uint8_t config_first_hi[CONFIG_COUNT] = CONFIG_FIRST_HI_TABLE;

static uint32_t compose_word(void)
{
    uint32_t word = (current_config >= 1u && current_config <= CONFIG_COUNT)
                        ? config_table[current_config - 1u] : 0u;

    if (cal_mode == CAL_MODE_MEAS) {
        if (word == 0u) {
            return word;                    /* no config: everything released */
        }
        /* A terminal on no rail must float: isolate it so its column bus
         * (open matrix contacts + copper, a few pF to HI and LO) does not
         * load the DUT node. Calibration isolates every terminal, so this
         * capacitance is never in the OSL and would land in the result. */
        for (uint32_t terminal = 0u; terminal < 4u; terminal++) {
            uint32_t column = 7u << MATRIX_BIT(terminal, 0u);
            if ((word & column) == 0u) {
                word |= 1u << (BIT_ISOLATE_A + terminal);
            }
        }
    } else {
        word |= MASK_ISOLATION;             /* DUT out; open contact = OPEN */
        if (cal_mode == CAL_MODE_SHORT) {
            /* Bridge the rails through the first HI terminal's bus by also
             * closing that terminal's LO relay -- a short presented at the
             * same contact plane as the DUT. */
            uint8_t terminal = config_first_hi[current_config - 1u];
            word |= 1u << MATRIX_BIT(terminal, 1u /* LO */);
        } else if (cal_mode == CAL_MODE_LOAD) {
            word |= MASK_LOAD;              /* 100R column across the rails */
        }
    }
    return word;
}
#endif

/* ---------------------------------------------------- SCPI action callbacks
 * (called by the parser in scpi.c; return 0 on success, SCPI error code
 * otherwise -- the parser owns the error queue) */

int scpi_action_reset(void)
{
    current_config = 0u;
    cal_mode = CAL_MODE_MEAS;
    relay_apply(0u);
    return 0;
}

int scpi_action_set_config(uint32_t number)
{
    if (number < 1u || number > CONFIG_COUNT) {
        return SCPI_ERR_DATA_OUT_OF_RANGE;
    }
    current_config = (uint8_t)number;
    cal_mode = CAL_MODE_MEAS;
    relay_apply(compose_word());
    return 0;
}

uint32_t scpi_query_config(void)
{
    return current_config;
}

int scpi_action_cal_mode(cal_mode_t mode)
{
    if (current_config == 0u && mode != CAL_MODE_MEAS) {
        return SCPI_ERR_SETTINGS_CONFLICT;   /* need a configuration first */
    }
    cal_mode = mode;
    relay_apply(compose_word());
    return 0;
}

cal_mode_t scpi_query_cal_mode(void)
{
    return cal_mode;
}

int scpi_action_set_relay(uint32_t index, uint32_t state)
{
    if (index >= RELAY_COUNT) {
        return SCPI_ERR_DATA_OUT_OF_RANGE;
    }
    uint32_t word = relay_word;
    if (state) {
        word |= (1u << index);
    } else {
        word &= ~(1u << index);
    }
    if (energized_count(word) > MAX_ENERGIZED) {
        return SCPI_ERR_SETTINGS_CONFLICT;   /* USB 500 mA coil budget */
    }
    current_config = 0u;                     /* manual override: no config */
    relay_apply(word);
    return 0;
}

uint32_t scpi_query_relay_word(void)
{
    return relay_word;
}

int scpi_action_self_test(void)
{
    /* Walk every coil output high then low, reading the ODR back.  This
     * verifies the GPIO path; contact verification is done from the host via
     * the Z0*Zsc' = Z0'*Zsc reciprocity identity, which sees the contacts. */
    uint32_t saved = relay_word;
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        uint32_t port = relay_port(code);
        uint16_t pin = 1u << (code & 15u);
        gpio_set(port, pin);
        if (!(gpio_port_read(port) & pin)) {
            relay_apply(saved);
            return SCPI_ERR_SELF_TEST;
        }
        gpio_clear(port, pin);
    }
    relay_apply(saved);
    return 0;
}

/* ------------------------------------------------------------------- main */

int main(void)
{
    enter_bootloader_if_requested();         /* before any clock/peripheral */
    clock_setup();
    relay_gpio_setup();
    counters_load();
#ifdef BOARD_REVC
    coil_supply_init();                      /* off until the host configures us */
    temperature_init();
#endif
    restore_load();                          /* before relay_apply saves */
    relay_apply(0u);                         /* safe state: all released */
    iwdg_set_period_ms(WATCHDOG_PERIOD_MS);
    iwdg_start();

    usb_cdc_init(scpi_feed_byte);            /* RX bytes go to the parser */
    scpi_init(usb_cdc_write);                /* parser replies over CDC   */

    for (;;) {
        iwdg_reset();
        usb_cdc_poll();
        scpi_poll();
        counters_poll();
#ifdef BOARD_REVC
        coil_supply_poll(usb_cdc_configured());
        int coils_powered = coil_supply_state() == 1;
#else
        int coils_powered = 1;                /* rev B coils hang on VBUS */
#endif
        if (restore_due && coils_powered) {
            restore_due = 0u;
            relay_apply(pending_restore);     /* staggered, like any change */
        }
        if (dfu_pending) {
            reboot_into_bootloader();
        }
    }
}
