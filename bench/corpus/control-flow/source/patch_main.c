#include <stdint.h>

uint32_t patchable(uint32_t value);

int main(int argc, char **argv) {
    return (int)patchable((uint32_t)argc + (uint32_t)(uintptr_t)argv);
}
