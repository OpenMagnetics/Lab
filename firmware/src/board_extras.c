/*
 * Board identity, relay wear counters, coil supply and fixture temperature.
 */

#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/i2c.h>
#include <libopencm3/stm32/rcc.h>

#include "board_extras.h"
#include "relay_map.h"

/* ------------------------------------------------------------- identity */

#define UID_BASE 0x1FFFF7ACu             /* STM32F0 96-bit unique ID */

void board_serial(char out[25])
{
    static const char hex[] = "0123456789ABCDEF";
    const volatile uint8_t *uid = (const volatile uint8_t *)UID_BASE;
    for (unsigned index = 0; index < 12u; index++) {
        uint8_t byte = uid[11u - index];  /* most significant word first */
        out[2u * index] = hex[byte >> 4];
        out[2u * index + 1u] = hex[byte & 15u];
    }
    out[24] = '\0';
}

/* ------------------------------------------------------- wear counters */

/* Last 2 KB page of the 128 KB part; the linker script stops at 126 KB. */
#define COUNTER_PAGE       0x0801F800u
#define COUNTER_MAGIC      0x52434E54u    /* "RCNT" */
#define COUNTER_SAVE_EVERY 256u

static uint32_t counts[RELAY_COUNT];
static uint32_t unsaved;

static uint32_t checksum(void)
{
    uint32_t sum = COUNTER_MAGIC;
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        sum = (sum << 5 | sum >> 27) ^ counts[index];
    }
    return sum;
}

void counters_load(void)
{
    const volatile uint32_t *page = (const volatile uint32_t *)COUNTER_PAGE;
    if (page[0] != COUNTER_MAGIC || page[1] != RELAY_COUNT) {
        return;                           /* blank or another board layout */
    }
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        counts[index] = page[2u + index];
    }
    if (page[2u + RELAY_COUNT] != checksum()) {
        for (unsigned index = 0; index < RELAY_COUNT; index++) {
            counts[index] = 0u;           /* torn write: start over */
        }
    }
}

void counters_bump(uint32_t energized_mask)
{
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        if (energized_mask & (1u << index)) {
            counts[index]++;
            unsaved++;
        }
    }
}

void counters_save(void)
{
    if (unsaved == 0u) {
        return;
    }
    flash_unlock();
    flash_erase_page(COUNTER_PAGE);
    flash_program_word(COUNTER_PAGE, COUNTER_MAGIC);
    flash_program_word(COUNTER_PAGE + 4u, RELAY_COUNT);
    for (unsigned index = 0; index < RELAY_COUNT; index++) {
        flash_program_word(COUNTER_PAGE + 8u + 4u * index, counts[index]);
    }
    flash_program_word(COUNTER_PAGE + 8u + 4u * RELAY_COUNT, checksum());
    flash_lock();
    unsaved = 0u;
}

void counters_poll(void)
{
    if (unsaved >= COUNTER_SAVE_EVERY) {
        counters_save();
    }
}

uint32_t counters_get(unsigned relay)
{
    return relay < RELAY_COUNT ? counts[relay] : 0u;
}

#ifdef BOARD_REVC
/* --------------------------------------------------------- coil supply */

static int coil_on;

/* PB12 enables the TPS2051C, PB13 reads its open-drain /FLT. */
void coil_supply_init(void)
{
    gpio_mode_setup(GPIOB, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO12);
    gpio_clear(GPIOB, GPIO12);                       /* off (R40 also holds it) */
    gpio_mode_setup(GPIOB, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, GPIO13);
}

void coil_supply_poll(int usb_configured)
{
    if (usb_configured && !coil_on) {
        gpio_set(GPIOB, GPIO12);                     /* soft start in the switch */
        coil_on = 1;
    }
}

int coil_supply_state(void)
{
    if (coil_on && !(gpio_get(GPIOB, GPIO13))) {
        return -1;                                   /* /FLT low: overcurrent */
    }
    return coil_on;
}

/* --------------------------------------------------------- temperature */

#define TMP117_ADDRESS 0x48u
#define TMP_I2C        I2C2

void temperature_init(void)
{
    /* I2C2 on PB10 (SCL) / PB11 (SDA), AF1: the west pins, facing the
     * TMP117. On the F072 I2C2 runs from PCLK (48 MHz), not HSI. */
    rcc_periph_clock_enable(RCC_I2C2);
    gpio_mode_setup(GPIOB, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO10 | GPIO11);
    gpio_set_output_options(GPIOB, GPIO_OTYPE_OD, GPIO_OSPEED_2MHZ, GPIO10 | GPIO11);
    gpio_set_af(GPIOB, GPIO_AF1, GPIO10 | GPIO11);
    i2c_peripheral_disable(TMP_I2C);
    i2c_enable_analog_filter(TMP_I2C);
    i2c_set_digital_filter(TMP_I2C, 0);
    i2c_set_speed(TMP_I2C, i2c_speed_sm_100k, rcc_apb1_frequency / 1000000u);
    i2c_set_7bit_addr_mode(TMP_I2C);
    i2c_peripheral_enable(TMP_I2C);
}

/* Wait for any of `flags` in I2C_ISR, bounded: libopencm3's i2c_transfer7
 * spins forever on a NACK, which an unfitted sensor would turn into a hung
 * board. Returns 0 on success, -1 on NACK or timeout. */
static int wait_isr(uint32_t flags)
{
    for (uint32_t spin = 0; spin < 200000u; spin++) {
        uint32_t isr = I2C_ISR(TMP_I2C);
        if (isr & I2C_ISR_NACKF) {
            I2C_ICR(TMP_I2C) = I2C_ICR_NACKCF | I2C_ICR_STOPCF;
            return -1;
        }
        if (isr & flags) {
            return 0;
        }
    }
    return -1;
}

int32_t temperature_millicelsius(void)
{
    uint8_t data[2];
    /* write the register pointer (0x00 = temperature) */
    i2c_set_7bit_address(TMP_I2C, TMP117_ADDRESS);
    i2c_set_write_transfer_dir(TMP_I2C);
    i2c_set_bytes_to_transfer(TMP_I2C, 1);
    i2c_disable_autoend(TMP_I2C);
    i2c_send_start(TMP_I2C);
    if (wait_isr(I2C_ISR_TXIS)) {
        return INT32_MIN;
    }
    i2c_send_data(TMP_I2C, 0x00u);
    if (wait_isr(I2C_ISR_TC)) {
        return INT32_MIN;
    }
    /* repeated start, read two bytes */
    i2c_set_read_transfer_dir(TMP_I2C);
    i2c_set_bytes_to_transfer(TMP_I2C, 2);
    i2c_enable_autoend(TMP_I2C);
    i2c_send_start(TMP_I2C);
    for (unsigned index = 0; index < 2u; index++) {
        if (wait_isr(I2C_ISR_RXNE)) {
            return INT32_MIN;
        }
        data[index] = i2c_get_data(TMP_I2C);
    }
    int16_t raw = (int16_t)((data[0] << 8) | data[1]);
    return ((int32_t)raw * 125) / 16;                /* 7.8125 m°C per LSB */
}
#endif
