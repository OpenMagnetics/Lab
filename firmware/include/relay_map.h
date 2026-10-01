/*
 * Relay and configuration map for the OpenMagnetics relay board rev B.
 *
 * SINGLE SOURCE OF TRUTH shared with the host software: the bit order here
 * must match MATRIX_BITS in scripts/RelayBoardController.py and the GPIO
 * assignment in hardware/relay-board-revB/generate_schematic.py.  All three
 * are checked against each other by hardware/relay-board-revB/verify_board.py,
 * which simulates every configuration over the exported netlist.
 *
 *   bit 0..11   crossbar  K1..K12   (A,B,C,D) x (HI, LO, LINK)
 *   bit 12,13   load std  K13, K14  (100R column: CAL_E->RAIL_HI, CAL_F->RAIL_LO)
 *   bit 14..17  isolation K15..K18  (per terminal A..D; energized = DUT out,
 *                                    terminal ends at an open contact = OPEN std)
 *
 * SHORT standard: no dedicated relay.  With the DUT isolated, closing one
 * terminal's HI and LO crossbar relays bridges the rails through that
 * terminal's bus -- a short at the same contact plane as the other standards.
 */

#ifndef RELAY_MAP_H
#define RELAY_MAP_H

#include <stdint.h>

#define RELAY_COUNT 18u

/* Crossbar bit for terminal t (0=A..3=D) and rail r (0=HI,1=LO,2=LINK). */
#define MATRIX_BIT(t, r)   ((uint32_t)((t) * 3u + (r)))

#define BIT_LOAD_HI        12u   /* K13: 100R column to RAIL_HI */
#define BIT_LOAD_LO        13u   /* K14: 100R column to RAIL_LO */
#define BIT_ISOLATE_A      14u   /* K15..K18 */
#define BIT_ISOLATE_B      15u
#define BIT_ISOLATE_C      16u
#define BIT_ISOLATE_D      17u

#define MASK_ISOLATION     ((1u << BIT_ISOLATE_A) | (1u << BIT_ISOLATE_B) | \
                            (1u << BIT_ISOLATE_C) | (1u << BIT_ISOLATE_D))
#define MASK_LOAD          ((1u << BIT_LOAD_HI) | (1u << BIT_LOAD_LO))

/* GPIO pins driving each relay, in relay order K1..K18.
 * Port A bit numbers are 0..15, port B encoded as 16+bit.
 * Matches generate_schematic.py GPIO_PINS:
 *   PA0-PA7, PB0, PB1, PB2, PB10, PB11, PB12, PB13, PB14, PB15, PA8      */
#define RELAY_GPIO_TABLE                                                   \
    { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,          /* K1..K8  = PA0..PA7  */   \
      16u + 0u, 16u + 1u, 16u + 2u, 16u + 10u, /* K9..K12 = PB0,1,2,10 */  \
      16u + 11u, 16u + 12u, 16u + 13u,         /* K13..K15 = PB11,12,13 */ \
      16u + 14u, 16u + 15u, 8u }               /* K16,K17 = PB14,15; K18 = PA8 */

/* Measurement configurations: relay words for CONF:MEAS 1..15.
 * Derived from CONFIGS in RelayBoardController.py: energize MATRIX_BIT(t, r)
 * for every terminal t assigned to rail r.  Terminals 0=A 1=B 2=C 3=D.     */
#define CFG(aH, aL, aK, bH, bL, bK, cH, cL, cK, dH, dL, dK)                \
    (((aH) << MATRIX_BIT(0, 0)) | ((aL) << MATRIX_BIT(0, 1)) |             \
     ((aK) << MATRIX_BIT(0, 2)) |                                          \
     ((bH) << MATRIX_BIT(1, 0)) | ((bL) << MATRIX_BIT(1, 1)) |             \
     ((bK) << MATRIX_BIT(1, 2)) |                                          \
     ((cH) << MATRIX_BIT(2, 0)) | ((cL) << MATRIX_BIT(2, 1)) |             \
     ((cK) << MATRIX_BIT(2, 2)) |                                          \
     ((dH) << MATRIX_BIT(3, 0)) | ((dL) << MATRIX_BIT(3, 1)) |             \
     ((dK) << MATRIX_BIT(3, 2)))

#define CONFIG_COUNT 15u

/*                          A         B         C         D
 *                        H  L  K   H  L  K   H  L  K   H  L  K            */
#define CONFIG_TABLE {                                                     \
    CFG(1, 0, 0,  0, 1, 0,  0, 0, 0,  0, 0, 0), /*  1 Z0    A-B          */\
    CFG(1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 1), /*  2 Zsc   A-B, CD link */\
    CFG(0, 0, 0,  0, 0, 0,  1, 0, 0,  0, 1, 0), /*  3 Z0p   C-D          */\
    CFG(0, 0, 1,  0, 0, 1,  1, 0, 0,  0, 1, 0), /*  4 Zscp  C-D, AB link */\
    CFG(1, 0, 0,  0, 0, 1,  0, 0, 1,  0, 1, 0), /*  5 Lcum  A-D, BC link */\
    CFG(1, 0, 0,  0, 0, 1,  0, 1, 0,  0, 0, 1), /*  6 Ldif  A-C, BD link */\
    CFG(1, 0, 0,  1, 0, 0,  0, 1, 0,  0, 1, 0), /*  7 C33   AB-CD        */\
    CFG(1, 0, 0,  0, 1, 0,  0, 0, 0,  0, 0, 1), /*  8 BD open            */\
    CFG(1, 0, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0), /*  9 BD short           */\
    CFG(1, 0, 0,  0, 1, 0,  1, 0, 0,  0, 0, 0), /* 10 AC open            */\
    CFG(1, 0, 0,  0, 1, 0,  1, 0, 0,  1, 0, 0), /* 11 AC short           */\
    CFG(1, 0, 0,  0, 1, 0,  0, 1, 0,  0, 0, 0), /* 12 BC open            */\
    CFG(1, 0, 0,  0, 1, 0,  0, 0, 0,  1, 0, 0), /* 13 AD open            */\
    CFG(0, 1, 0,  0, 1, 0,  1, 0, 0,  0, 1, 0), /* 14 BD short, sec side */\
    CFG(1, 0, 0,  1, 0, 0,  1, 0, 0,  0, 1, 0), /* 15 AC short, sec side */\
}

/* Calibration overlays (CAL:MODE), applied on top of the measurement word.
 * SHORT needs the first-HI terminal per configuration, listed here in the
 * same 1-based order as CONFIG_TABLE (0=A 1=B 2=C 3=D).                    */
#define CONFIG_FIRST_HI_TABLE \
    { 0u, 0u, 2u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 2u, 0u }

typedef enum {
    CAL_MODE_MEAS = 0,   /* isolation off, load off: DUT in circuit          */
    CAL_MODE_OPEN,       /* + MASK_ISOLATION                                  */
    CAL_MODE_SHORT,      /* + MASK_ISOLATION + firstHI terminal's LO relay    */
    CAL_MODE_LOAD,       /* + MASK_ISOLATION + MASK_LOAD                      */
} cal_mode_t;

#endif /* RELAY_MAP_H */
