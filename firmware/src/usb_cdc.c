/*
 * USB CDC-ACM on the STM32F072 USB device peripheral, via libopencm3.
 *
 * Enumerates as a virtual COM port; RX bytes are handed to the SCPI parser,
 * TX is buffered and drained from the main loop.  The F072 has true USB
 * hardware (unlike bit-banged solutions) and, with the 8 MHz crystal feeding
 * the PLL, its 48 MHz clock is crystal-accurate -- no HSI48/CRS trimming.
 */

#include <stdlib.h>
#include <string.h>

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/crs.h>
#include <libopencm3/usb/cdc.h>
#include <libopencm3/usb/usbd.h>

#include "usb_cdc.h"

#define TX_BUFFER 512

static usbd_device *usb_device;
static usb_cdc_rx_fn rx_callback;
static uint8_t usbd_control_buffer[128];

static uint8_t tx_buffer[TX_BUFFER];
static volatile uint32_t tx_head, tx_tail;
static volatile uint8_t configured;

/* ------------------------------------------------------------- descriptors */

static const struct usb_device_descriptor device_descriptor = {
    .bLength = USB_DT_DEVICE_SIZE,
    .bDescriptorType = USB_DT_DEVICE,
    .bcdUSB = 0x0200,
    .bDeviceClass = USB_CLASS_CDC,
    .bDeviceSubClass = 0,
    .bDeviceProtocol = 0,
    .bMaxPacketSize0 = 64,
    .idVendor = 0x1209,            /* pid.codes open-source VID           */
    .idProduct = 0x0001,           /* TEST PID -- register a real one at  */
    .bcdDevice = 0x0200,           /* pid.codes before distributing hw    */
    .iManufacturer = 1,
    .iProduct = 2,
    .iSerialNumber = 3,
    .bNumConfigurations = 1,
};

static const struct usb_endpoint_descriptor comm_endpoints[] = {{
    .bLength = USB_DT_ENDPOINT_SIZE,
    .bDescriptorType = USB_DT_ENDPOINT,
    .bEndpointAddress = 0x83,
    .bmAttributes = USB_ENDPOINT_ATTR_INTERRUPT,
    .wMaxPacketSize = 16,
    .bInterval = 255,
}};

static const struct usb_endpoint_descriptor data_endpoints[] = {{
    .bLength = USB_DT_ENDPOINT_SIZE,
    .bDescriptorType = USB_DT_ENDPOINT,
    .bEndpointAddress = 0x01,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK,
    .wMaxPacketSize = 64,
    .bInterval = 1,
}, {
    .bLength = USB_DT_ENDPOINT_SIZE,
    .bDescriptorType = USB_DT_ENDPOINT,
    .bEndpointAddress = 0x82,
    .bmAttributes = USB_ENDPOINT_ATTR_BULK,
    .wMaxPacketSize = 64,
    .bInterval = 1,
}};

static const struct {
    struct usb_cdc_header_descriptor header;
    struct usb_cdc_call_management_descriptor call_management;
    struct usb_cdc_acm_descriptor acm;
    struct usb_cdc_union_descriptor cdc_union;
} __attribute__((packed)) cdc_functional_descriptors = {
    .header = {
        .bFunctionLength = sizeof(struct usb_cdc_header_descriptor),
        .bDescriptorType = CS_INTERFACE,
        .bDescriptorSubtype = USB_CDC_TYPE_HEADER,
        .bcdCDC = 0x0110,
    },
    .call_management = {
        .bFunctionLength = sizeof(struct usb_cdc_call_management_descriptor),
        .bDescriptorType = CS_INTERFACE,
        .bDescriptorSubtype = USB_CDC_TYPE_CALL_MANAGEMENT,
        .bmCapabilities = 0,
        .bDataInterface = 1,
    },
    .acm = {
        .bFunctionLength = sizeof(struct usb_cdc_acm_descriptor),
        .bDescriptorType = CS_INTERFACE,
        .bDescriptorSubtype = USB_CDC_TYPE_ACM,
        .bmCapabilities = 0x02,        /* line coding supported */
    },
    .cdc_union = {
        .bFunctionLength = sizeof(struct usb_cdc_union_descriptor),
        .bDescriptorType = CS_INTERFACE,
        .bDescriptorSubtype = USB_CDC_TYPE_UNION,
        .bControlInterface = 0,
        .bSubordinateInterface0 = 1,
    },
};

static const struct usb_interface_descriptor comm_interface = {
    .bLength = USB_DT_INTERFACE_SIZE,
    .bDescriptorType = USB_DT_INTERFACE,
    .bInterfaceNumber = 0,
    .bAlternateSetting = 0,
    .bNumEndpoints = 1,
    .bInterfaceClass = USB_CLASS_CDC,
    .bInterfaceSubClass = USB_CDC_SUBCLASS_ACM,
    .bInterfaceProtocol = USB_CDC_PROTOCOL_AT,
    .iInterface = 0,
    .endpoint = comm_endpoints,
    .extra = &cdc_functional_descriptors,
    .extralen = sizeof(cdc_functional_descriptors),
};

static const struct usb_interface_descriptor data_interface = {
    .bLength = USB_DT_INTERFACE_SIZE,
    .bDescriptorType = USB_DT_INTERFACE,
    .bInterfaceNumber = 1,
    .bAlternateSetting = 0,
    .bNumEndpoints = 2,
    .bInterfaceClass = USB_CLASS_DATA,
    .endpoint = data_endpoints,
};

static const struct usb_interface interfaces[] = {
    {.num_altsetting = 1, .altsetting = &comm_interface},
    {.num_altsetting = 1, .altsetting = &data_interface},
};

static const struct usb_config_descriptor config_descriptor = {
    .bLength = USB_DT_CONFIGURATION_SIZE,
    .bDescriptorType = USB_DT_CONFIGURATION,
    .wTotalLength = 0,
    .bNumInterfaces = 2,
    .bConfigurationValue = 1,
    .iConfiguration = 0,
    .bmAttributes = 0x80,
    .bMaxPower = 250,                  /* 500 mA: up to 9 relay coils     */
    .interface = interfaces,
};

static const char *usb_strings[] = {
    "OpenMagnetics",
    "RelayBoard revB",
    "OM-RB-0001",
};

/* --------------------------------------------------------------- handlers */

static enum usbd_request_return_codes control_request(
    usbd_device *device, struct usb_setup_data *request, uint8_t **buffer,
    uint16_t *length,
    void (**complete)(usbd_device *, struct usb_setup_data *))
{
    (void)device; (void)buffer; (void)complete;
    switch (request->bRequest) {
    case USB_CDC_REQ_SET_CONTROL_LINE_STATE:
        return USBD_REQ_HANDLED;
    case USB_CDC_REQ_SET_LINE_CODING:
        if (*length < sizeof(struct usb_cdc_line_coding)) {
            return USBD_REQ_NOTSUPP;
        }
        return USBD_REQ_HANDLED;
    }
    return USBD_REQ_NOTSUPP;
}

static void data_rx(usbd_device *device, uint8_t endpoint)
{
    (void)endpoint;
    uint8_t packet[64];
    int length = usbd_ep_read_packet(device, 0x01, packet, sizeof packet);
    for (int index = 0; index < length; index++) {
        if (rx_callback) {
            rx_callback(packet[index]);
        }
    }
}

static void set_configuration(usbd_device *device, uint16_t value)
{
    (void)value;
    usbd_ep_setup(device, 0x01, USB_ENDPOINT_ATTR_BULK, 64, data_rx);
    usbd_ep_setup(device, 0x82, USB_ENDPOINT_ATTR_BULK, 64, NULL);
    usbd_ep_setup(device, 0x83, USB_ENDPOINT_ATTR_INTERRUPT, 16, NULL);
    usbd_register_control_callback(
        device, USB_REQ_TYPE_CLASS | USB_REQ_TYPE_INTERFACE,
        USB_REQ_TYPE_TYPE | USB_REQ_TYPE_RECIPIENT, control_request);
    configured = 1;
}

/* ----------------------------------------------------------------- public */

void usb_cdc_init(usb_cdc_rx_fn on_byte)
{
    rx_callback = on_byte;
    rcc_periph_clock_enable(RCC_USB);
    usb_device = usbd_init(&st_usbfs_v2_usb_driver, &device_descriptor,
                           &config_descriptor, usb_strings, 3,
                           usbd_control_buffer, sizeof usbd_control_buffer);
    usbd_register_set_config_callback(usb_device, set_configuration);
}

void usb_cdc_poll(void)
{
    usbd_poll(usb_device);
    /* Drain the TX ring, one packet per poll. */
    if (configured && tx_head != tx_tail) {
        uint8_t packet[64];
        uint32_t count = 0;
        while (tx_tail != tx_head && count < 63) {   /* <64 avoids ZLP need */
            packet[count++] = tx_buffer[tx_tail];
            tx_tail = (tx_tail + 1) % TX_BUFFER;
        }
        if (usbd_ep_write_packet(usb_device, 0x82, packet, (uint16_t)count) == 0) {
            tx_tail = (tx_tail + TX_BUFFER - count) % TX_BUFFER;   /* retry */
        }
    }
}

void usb_cdc_write(const char *data, uint32_t length)
{
    for (uint32_t index = 0; index < length; index++) {
        uint32_t next = (tx_head + 1) % TX_BUFFER;
        if (next == tx_tail) {
            break;                     /* full: drop rather than block */
        }
        tx_buffer[tx_head] = (uint8_t)data[index];
        tx_head = next;
    }
}
