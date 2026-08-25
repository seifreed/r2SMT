#include <stdint.h>

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

static volatile uint32_t matrix_state;

NOINLINE uint32_t portable_branch(uint32_t value) {
    uint32_t mix = (value ^ 0x5a5a5a5aU) + ((value & 0x0fU) << 4U);
    if ((mix & 1U) == 0U) {
        matrix_state += mix;
    } else {
        matrix_state ^= mix;
    }
    return matrix_state;
}

int main(int argc, char **argv) {
    (void)argv;
    return (int)portable_branch((uint32_t)argc);
}
