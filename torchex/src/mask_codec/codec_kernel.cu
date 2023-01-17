#include "../utils/error.cuh"
#include <cstdio>
#include <vector>

#define MAX_THREADS 1024
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

__global__ void encoder(const bool *mask, unsigned long long *code, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_size = sizeof(unsigned long long) * 8;
    if (idx * bit_size >= total){
        return;
    }
    int res = min(bit_size, total - idx * bit_size);
    unsigned long long temp = 0;
    for (int i = 0; i < res; i++){
        temp |= mask[idx * bit_size + i] ? (1ULL << i) : 0ULL;
    }
    code[idx] = temp;
}

__global__ void decoder(const unsigned long long *code, bool *mask, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_size = sizeof(unsigned long long) * 8;
    if (idx * bit_size >= total){
        return;
    }
    int res = min(bit_size, total - idx * bit_size);
    unsigned long long temp = code[idx];
    for (int i = 0; i < res; i++){
        mask[idx * bit_size + i] = ((temp >> i) & 1ULL) ? true : false;
    }
}

void encoder_launcher(const bool *mask, unsigned long long *code, int total) {
    encoder<<<DIVUP(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(mask, code, total);
}

void decoder_launcher(const unsigned long long *code, bool *mask, int total) {
    decoder<<<DIVUP(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(code, mask, total);
}

void decoder_cpu_launcher(const unsigned long long *code, bool *mask, int total) {
    int bit_size = sizeof(unsigned long long) * 8;
    for (int idx = 0; idx * bit_size < total; idx++){
        int res = min(bit_size, total - idx * bit_size);
        unsigned long long temp = code[idx];
        for (int i = 0; i < res; i++){
            mask[idx * bit_size + i] = ((temp >> i) & 1ULL) ? true : false;
        }
    }
}