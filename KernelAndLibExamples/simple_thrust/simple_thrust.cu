#include <cstdint>
#include <cstdio>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

using uint8 = uint8_t;
using uint32 = uint32_t;

int main() {
    // ---*** Setup host ***--- //
    const uint32 sizeCheck = 32000000U;
    // Initialize host vector with 32M size.
    thrust::host_vector<uint8> hData(sizeCheck, 0);

    printf("hDataSize: %zu\n", hData.size());
    // Set host vector to value.
    for (uint32 i = 0; i < sizeCheck; ++i) {
        // Mask higher order bits.
        hData[i] = uint8(i & 0xff);
    }
    printf("Host data pre GPU for_each:\n");
    for (uint32 i = 0; i < 10; ++i) { printf("%u ", hData[i]); }
    printf("\n");

    // ---*** simple thrust ***--- //
    thrust::device_vector<uint8> dData = hData;
    thrust::for_each(dData.begin(), dData.end(), [] __device__ (uint8& x)
    {
        if (255 == x) { x = 0; }
        else { x = 1; }
    });

    // Copy data back.
    hData = dData;
    printf("Host data post GPU for_each:\n");
    for (uint32 i = 0; i < 10; ++i) { printf("%u ", hData[i]); }
    printf("\n");
    return 0;
}

