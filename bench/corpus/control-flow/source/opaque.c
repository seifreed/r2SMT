#include <stdint.h>

__attribute__((noinline)) uint32_t opaque_even(uint32_t value) {
    uint32_t product = value * (value + 1U);
    return (product & 1U) == 0U ? 1U : 0U;
}

__attribute__((noinline)) uint32_t mba_equal(uint32_t left, uint32_t right) {
    uint32_t mba = (left ^ right) + ((left & right) << 1U);
    return mba == left + right ? 1U : 0U;
}
