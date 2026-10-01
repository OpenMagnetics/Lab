/*
 * OpenMagnetics relay board rev B -- STM32F072CBT6 firmware.
 *
 * USB CDC-ACM device exposing a SCPI command set (see scpi.c).  Built on
 * libopencm3; see the Makefile for toolchain and flashing instructions.
 *
 * Clocking: HSI48 + CRS auto-trim from USB SOF (crystal-less USB, native
 * on the F072).
 *
 * Relay drive: 18 GPIOs -> 3x TBD62003APG -> G6K-2F-RF-S coils.  Relay state
 * changes settle for RELAY_SETTLE_MS before *OPC? reports complete, so the
 * host can trust that a measurement started after OPC sees settled contacts.
 */

#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/stm32/crs.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/st_usbfs.h>
#include <libopencm3/stm32/syscfg.h>

#include "relay_map.h"
#include "scpi.h"
#include "usb_cdc.h"

#define RELAY_SETTLE_MS 10u   /* G6K operate/release 3 ms + margin + bounce */

static const uint8_t relay_gpio[RELAY_COUNT] = RELAY_GPIO_TABLE;
static const uint32_t config_table[CONFIG_COUNT] = CONFIG_TABLE;

static volatile uint32_t milliseconds;
static uint32_t relay_word;          /* current coil states, bit n = K(n+1) */
static uint8_t current_config;      /* 1..CONFIG_COUNT, 0 = none           */
static cal_mode_t cal_mode = CAL_MODE_MEAS;

/* SysTick runs at TICK_HZ; `milliseconds` advances every TICK_HZ/1000 ticks.
 *
 * Status LED (PC13 high = lit): steady and dim rather than a full-brightness
 * blink.  PC13 has no timer channel, so PWM it from the tick: lit for
 * LED_ON_TICKS out of every LED_PERIOD_TICKS (100 Hz, too fast to flicker;
 * 2 % duty). */
#define TICK_HZ          10000u
#define LED_PERIOD_TICKS 100u
#define LED_ON_TICKS     2u

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
    if (led_phase < LED_ON_TICKS) {
        gpio_set(GPIOC, GPIO13);
    } else {
        gpio_clear(GPIOC, GPIO13);
    }
}

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
 * .noinit RAM, which the startup code neither loads nor zeroes. */
#define DFU_MAGIC     0xDF00B007u
#define SYSTEM_MEMORY 0x1FFFC800u      /* F072 ROM bootloader (AN2606) */

static uint32_t dfu_request __attribute__((section(".noinit")));
static volatile uint8_t dfu_pending;

static void relay_apply(uint32_t word);

static void enter_bootloader_if_requested(void)
{
    if (dfu_request != DFU_MAGIC) {
        return;
    }
    dfu_request = 0u;
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
    *USB_BCDR_REG &= ~(uint32_t)USB_BCDR_DPPU;   /* drop the D+ pull-up: host sees */
    delay_ms(100u);                     /* an unplug before DFU appears   */
    dfu_request = DFU_MAGIC;
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
        uint32_t port = (code >= 16u) ? GPIOB : GPIOA;
        uint16_t pin = 1u << (code & 15u);
        gpio_mode_setup(port, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, pin);
        gpio_clear(port, pin);
    }
    /* Status LED on PC13, dimmed by sys_tick_handler. */
    gpio_mode_setup(GPIOC, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO13);
}

static void relay_apply(uint32_t word)
{
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        uint8_t code = relay_gpio[index];
        uint32_t port = (code >= 16u) ? GPIOB : GPIOA;
        uint16_t pin = 1u << (code & 15u);
        if (word & (1u << index)) {
            gpio_set(port, pin);
        } else {
            gpio_clear(port, pin);
        }
    }
    relay_word = word;
    delay_ms(RELAY_SETTLE_MS);
}

/* Compose the full relay word: measurement matrix + calibration overlay. */
static const uint8_t config_first_hi[CONFIG_COUNT] = CONFIG_FIRST_HI_TABLE;

static uint32_t compose_word(void)
{
    uint32_t word = (current_config >= 1u && current_config <= CONFIG_COUNT)
                        ? config_table[current_config - 1u] : 0u;

    if (cal_mode != CAL_MODE_MEAS) {
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
        uint32_t port = (code >= 16u) ? GPIOB : GPIOA;
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
    relay_apply(0u);                         /* safe state: all released */

    usb_cdc_init(scpi_feed_byte);            /* RX bytes go to the parser */
    scpi_init(usb_cdc_write);                /* parser replies over CDC   */

    for (;;) {
        usb_cdc_poll();
        scpi_poll();
        if (dfu_pending) {
            reboot_into_bootloader();
        }
    }
}
