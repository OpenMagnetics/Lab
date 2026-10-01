/* USB CDC-ACM device: the board appears as a virtual COM port. */

#ifndef USB_CDC_H
#define USB_CDC_H

#include <stdint.h>

typedef void (*usb_cdc_rx_fn)(uint8_t byte);

void usb_cdc_init(usb_cdc_rx_fn on_byte);
void usb_cdc_poll(void);
void usb_cdc_write(const char *data, uint32_t length);

#endif
