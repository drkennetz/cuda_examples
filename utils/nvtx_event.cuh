#include <nvToolsExt.h>

// Some colors to set for the profiler.
enum NvtxColors : uint32_t
{
    Grey       = 0x00'00'00,
    Red        = 0xaa'00'00,
    BrightBlue = 0x00'00'dd,
    Blue       = 0x00'00'aa,
    Green      = 0x2c'a0'2c,
};

// Starts an nvtx event with a name and color attribute.
void startNamedEventWithStream(const char* name, const uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.message.ascii = name;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtxRangePushEx(&eventAttrib);
}

// Pops an event.
void endEvent() {
    nvtxRangePop();
}
