#include <stdint.h>

static volatile uint32_t state;

__attribute__((noinline)) uint32_t update_state(uint32_t value) {
    if ((value & 1U) != 0U) {
        state += value;
    } else {
        state ^= value;
    }
    return state;
}

int main(int argc, char **argv) {
    (void)argv;
    return (int)update_state((uint32_t)argc);
}
