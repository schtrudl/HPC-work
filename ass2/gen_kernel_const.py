#!/usr/bin/env python3
"""Generate sparse kernel as C constant array for CUDA."""

import numpy as np

KERNEL_SIZE = 26


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_kernel(size):
    mu, sigma = 0.5, 0.15
    r = size // 2
    K = np.zeros((size, size), dtype=np.float64)  # Use double precision

    for y in range(-r, r):
        for x in range(-r, r):
            distance = np.sqrt((1.0 + x) ** 2 + (1.0 + y) ** 2) / r
            val = gauss(distance, mu, sigma) if distance <= 1.0 else 0.0
            K[y + r, x + r] = val

    K /= K.sum()
    return K


def main():
    K = generate_kernel(KERNEL_SIZE)
    half = KERNEL_SIZE // 2

    entries = []
    for ki in range(KERNEL_SIZE):
        for kj in range(KERNEL_SIZE):
            weight = K[ki, kj]
            if weight != 0.0:
                di = half - 1 - ki
                dj = half - 1 - kj
                entries.append((di, dj, weight))

    print(f"// Auto-generated sparse kernel ({len(entries)} entries)")
    print(
        f"// KERNEL_SIZE={KERNEL_SIZE}, {100 * (1 - len(entries) / (KERNEL_SIZE**2)):.1f}% sparsity"
    )
    print()
    print("#ifndef SPARSE_KERNEL_H")
    print("#define SPARSE_KERNEL_H")
    print()
    print(f"#define NUM_KERNEL_ENTRIES {len(entries)}")
    print()
    print("struct kernel_entry {")
    print("    int di, dj;")
    print("    float weight;")
    print("};")
    print()
    print("#ifdef __CUDACC__")
    print("__constant__ struct kernel_entry d_sparse_k[NUM_KERNEL_ENTRIES];")
    print("#define SPARSE_K d_sparse_k")
    print("#else")
    print("#define SPARSE_K sparse_kernel")
    print("#endif")
    print()
    print("static const struct kernel_entry sparse_kernel[NUM_KERNEL_ENTRIES] = {")
    for i, (di, dj, w) in enumerate(entries):
        comma = "," if i < len(entries) - 1 else ""
        print(f"    {{{di:3d}, {dj:3d}, {w}f}}{comma}")
    print("};")
    print()
    print("#ifdef __CUDACC__")
    print("inline void init_kernel_const() {")
    print("    cudaMemcpyToSymbol(d_sparse_k, sparse_kernel, sizeof(sparse_kernel));")
    print("}")
    print("#endif")
    print()
    print("#endif // SPARSE_KERNEL_H")


if __name__ == "__main__":
    main()
