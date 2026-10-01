# OpenMagnetics relay board rev B firmware -- STM32F072CBT6.
#
# Requires:
#   arm-none-eabi-gcc          https://developer.arm.com/downloads/-/gnu-rm
#   libopencm3                 git clone https://github.com/libopencm3/libopencm3
#                              && make -C libopencm3 TARGETS=stm32/f0
#
# Point OPENCM3_DIR at the built tree, then:
#   make                       build/relay_board.elf/.bin
#   make flash                 via SWD (st-flash from stlink-tools)
#   make dfu                   via the built-in DFU bootloader (hold BOOT0)

OPENCM3_DIR ?= ../libopencm3
PREFIX      ?= arm-none-eabi
BUILD       := build
TARGET      := $(BUILD)/relay_board

CC      := $(PREFIX)-gcc
OBJCOPY := $(PREFIX)-objcopy
SIZE    := $(PREFIX)-size

SOURCES := src/main.c src/scpi.c src/usb_cdc.c
OBJECTS := $(SOURCES:src/%.c=$(BUILD)/%.o)

DEFS    := -DSTM32F0
ARCH    := -mcpu=cortex-m0 -mthumb -msoft-float
CFLAGS  := $(ARCH) $(DEFS) -Os -g -std=c11 \
           -Wall -Wextra -Wshadow -Wconversion \
           -ffunction-sections -fdata-sections -fno-common \
           -Iinclude -I$(OPENCM3_DIR)/include
LDFLAGS := $(ARCH) -nostartfiles -specs=nano.specs \
           -Wl,--gc-sections -Wl,-Map=$(TARGET).map \
           -L$(OPENCM3_DIR)/lib -Tstm32f072cb.ld
LDLIBS  := -lopencm3_stm32f0 -lc -lgcc -lnosys

all: $(TARGET).bin
	$(SIZE) $(TARGET).elf

$(BUILD)/%.o: src/%.c include/relay_map.h include/scpi.h include/usb_cdc.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET).elf: $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(BUILD):
	mkdir -p $(BUILD)

flash: $(TARGET).bin
	st-flash --reset write $(TARGET).bin 0x8000000

dfu: $(TARGET).bin
	dfu-util -a 0 -s 0x08000000:leave -D $(TARGET).bin

clean:
	rm -rf $(BUILD)

.PHONY: all flash dfu clean
