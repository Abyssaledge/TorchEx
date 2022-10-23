#pragma once
#include <cstdio>

struct GPUTimer {
    cudaEvent_t beg, end;
    GPUTimer() {
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
    }
    ~GPUTimer() {
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }
    void start() {
        cudaEventSynchronize(beg);
        cudaEventRecord(beg, 0);
    }
    double stop() {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms;
        cudaEventElapsedTime(&ms, beg, end);
        printf("time between cudaEventRecord begin and end is %.4f ms\n", ms);
        return ms;
    }
};