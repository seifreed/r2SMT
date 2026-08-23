#include <stdint.h>

volatile uint32_t r2smt_sink;

uint32_t opaque_even(uint32_t value);
uint32_t mba_equal(uint32_t left, uint32_t right);

__attribute__((noinline)) static int signed_branch(int value) {
    return value < 0 ? -value : value;
}

__attribute__((noinline)) static uint32_t unsigned_branch(uint32_t value) {
    return value < 16U ? value + 1U : value - 1U;
}

__attribute__((noinline)) static uint32_t loop_and_memory(uint32_t value) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < value; ++i) {
        total += i;
    }
    r2smt_sink = total;
    return r2smt_sink;
}

int main(int argc, char **argv) {
    uint32_t value = (uint32_t)argc;
    uint32_t result = unsigned_branch(value);
    result += loop_and_memory(value);
    result += opaque_even(value);
    result += mba_equal(value, (uint32_t)(uintptr_t)argv);
    return signed_branch((int)result);
}
