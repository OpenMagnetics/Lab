/* STM32F072CBT6: 128K flash, 16K RAM. The last 2K page holds the relay
 * wear counters (board_extras.c), so the image stops at 126K. */
MEMORY
{
    rom (rx)  : ORIGIN = 0x08000000, LENGTH = 126K
    ram (rwx) : ORIGIN = 0x20000000, LENGTH = 16K
}

INCLUDE cortex-m-generic.ld
