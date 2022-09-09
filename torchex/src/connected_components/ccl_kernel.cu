// Thanks for https://dl.acm.org/doi/10.1145/3208040.3208041
// Modified from https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/
#include "../utils/error.cuh"
#include <cmath>
#include <cstdio>
#include <queue>
#include <time.h>
#include <unordered_map>

static const int blockSize = 256;
static const int adj_blockSize = 32;

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
        return ms;
    }
};

static __device__ int topL, posL;

__global__ void calc_adj_3(const float *points, const int *const __restrict__ labels, const float thresh_dist_squre, int *const __restrict__ adj_matrix, int *const __restrict__ adj_len, const int N, const int MAXNeighbor) {
    int tidx = threadIdx.x;
    int active_idx = tidx + blockIdx.y * blockDim.x;
    if ((blockIdx.y * blockDim.x <= blockIdx.x * blockDim.x) && (active_idx < N)) {
        extern __shared__ float s_points[];
        int passive_idx = tidx + blockIdx.x * blockDim.x;
        if (passive_idx < N) {
            s_points[tidx * 3] = points[passive_idx * 3];
            s_points[tidx * 3 + 1] = points[passive_idx * 3 + 1];
            s_points[tidx * 3 + 2] = points[passive_idx * 3 + 2];
        }
        __syncthreads();
        const float *currentPoint = points + 3 * active_idx;
        for (int i = 0; i < blockDim.x; i++) {
            passive_idx = i + blockIdx.x * blockDim.x;
            if (passive_idx > active_idx && passive_idx < N) {
                if (labels && (labels[active_idx] != labels[passive_idx]))
                    continue;
                float delta_x = currentPoint[0] - s_points[i * 3], delta_y = currentPoint[1] - s_points[i * 3 + 1], delta_z = currentPoint[2] - s_points[i * 3 + 2];
                float dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                if (dist <= thresh_dist_squre) {
                    if (adj_len[active_idx] >= MAXNeighbor)
                        printf("Warning! The point %d is surrounded with more than %d points! Please enlarge the MAXNeighbor!\n", active_idx, MAXNeighbor);
                    else
                        adj_matrix[active_idx * MAXNeighbor + atomicAdd(&adj_len[active_idx], 1)] = passive_idx;
                    if (adj_len[passive_idx] >= MAXNeighbor)
                        printf("Warning! The point %d is surrounded with more than %d points! Please enlarge the MAXNeighbor!\n", passive_idx, MAXNeighbor);
                    else
                        adj_matrix[passive_idx * MAXNeighbor + atomicAdd(&adj_len[passive_idx], 1)] = active_idx;
                }
            }
        }
    }
}

__global__ void calc_adj_2(const float *points, const int *const __restrict__ labels, const float thresh_dist_squre, int *const __restrict__ adj_matrix, int *const __restrict__ adj_len, const int N, const int MAXNeighbor) {
    int tidx = threadIdx.x;
    int active_idx = tidx + blockIdx.y * blockDim.x;
    if ((blockIdx.y * blockDim.x <= blockIdx.x * blockDim.x) && (active_idx < N)) {
        extern __shared__ float s_points[];
        int passive_idx = tidx + blockIdx.x * blockDim.x;
        if (passive_idx < N) {
            s_points[tidx * 2] = points[passive_idx * 3];
            s_points[tidx * 2 + 1] = points[passive_idx * 3 + 1];
        }
        __syncthreads();
        const float *currentPoint = points + 3 * active_idx;
        for (int i = 0; i < blockDim.x; i++) {
            passive_idx = i + blockIdx.x * blockDim.x;
            if (passive_idx > active_idx && passive_idx < N) {
                if (labels && (labels[active_idx] != labels[passive_idx]))
                    continue;
                float delta_x = currentPoint[0] - s_points[i * 2], delta_y = currentPoint[1] - s_points[i * 2 + 1];
                float dist = delta_x * delta_x + delta_y * delta_y;
                if (dist <= thresh_dist_squre) {
                    if (adj_len[active_idx] >= MAXNeighbor)
                        printf("Warning! The point %d is surrounded with more than %d points! Please consider enlarging the MAXNeighbor!\n", active_idx, MAXNeighbor);
                    else
                        adj_matrix[active_idx * MAXNeighbor + atomicAdd(&adj_len[active_idx], 1)] = passive_idx;
                    if (adj_len[passive_idx] >= MAXNeighbor)
                        printf("Warning! The point %d is surrounded with more than %d points! Please consider enlarging the MAXNeighbor!\n", passive_idx, MAXNeighbor);
                    else
                        adj_matrix[passive_idx * MAXNeighbor + atomicAdd(&adj_len[passive_idx], 1)] = active_idx;
                }
            }
        }
    }
}

__device__ int Find(const int idx, int *const __restrict__ parent) {
    int current = parent[idx];
    if (idx != current) {
        int prev = idx, next;
        while (current > (next = parent[current])) {
            parent[prev] = next;
            prev = current;
            current = next;
        }
    }
    return current;
}

__device__ void Union(int root_vertex, int root_adj, int *const __restrict__ parent) {
    bool repeat;
    do {
        repeat = false;
        if (root_vertex != root_adj) {
            int temp;
            if (root_vertex < root_adj) {
                temp = atomicCAS(&parent[root_adj], root_adj, root_vertex);
                if (temp != root_adj) {
                    repeat = true;
                    root_adj = temp;
                }
            } else {
                temp = atomicCAS(&parent[root_vertex], root_vertex, root_adj);
                if (temp != root_vertex) {
                    repeat = true;
                    root_vertex = temp;
                }
            }
        }
    } while (repeat);
}

__global__ void init(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, int *const __restrict__ parent, const const int N, const int MAXNeighbor) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < N) {
        int adj_idx = 0;
        int degree = adj_len[vertex];
        for (; (adj_idx < degree) && (adj_matrix[vertex * MAXNeighbor + adj_idx] >= vertex); adj_idx++) {
        }
        if (adj_idx == degree)
            parent[vertex] = vertex;
        else
            parent[vertex] = adj_matrix[vertex * MAXNeighbor + adj_idx];
    }
    if (vertex == 0) {
        topL = 0;
        posL = 0;
    }
}

__global__ void compute1(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, int *const __restrict__ parent, const int N, int *const __restrict__ wl, const int MAXNeighbor) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < N) {
        if (vertex != parent[vertex]) {
            int degree = adj_len[vertex];
            if (degree > 16) {
                int temp_idx = atomicAdd(&topL, 1);
                wl[temp_idx] = vertex;
            } else {
                int root_vertex = Find(vertex, parent);
                for (int adj_idx = 0; adj_idx < adj_len[vertex]; adj_idx++) {
                    int adj_vertex = adj_matrix[vertex * MAXNeighbor + adj_idx];
                    if (vertex > adj_vertex) {
                        int root_adj = Find(adj_vertex, parent);
                        Union(root_vertex, root_adj, parent);
                    }
                }
            }
        }
    }
}

__global__ void compute2(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, int *const __restrict__ parent, const int N, int *const __restrict__ wl, const int MAXNeighbor) {
    int lane = threadIdx.x % warpSize;
    int vertex_idx;
    if (lane == 0) {
        vertex_idx = atomicAdd(&posL, 1);
    }
    vertex_idx = __shfl_sync(0xffffffff, vertex_idx, 0);
    while (vertex_idx < topL) {
        int vertex = wl[vertex_idx];
        int root_vertex = Find(vertex, parent);
        for (int adj_idx = lane; adj_idx < adj_len[vertex]; adj_idx += warpSize) {
            int adj_vertex = adj_matrix[vertex * MAXNeighbor + adj_idx];
            if (vertex > adj_vertex) {
                int root_adj = Find(adj_vertex, parent);
                Union(root_vertex, root_adj, parent);
            }
        }
        if (lane == 0) {
            vertex_idx = atomicAdd(&posL, 1);
        }
        vertex_idx = __shfl_sync(0xffffffff, vertex_idx, 0);
    }
}

__global__ void flatten(int *const __restrict__ parent, const int N) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < N) {
        int current = parent[vertex], next;
        while (current > (next = parent[current])) {
            current = next;
        }
        if (parent[vertex] != current)
            parent[vertex] = current;
    }
}

__global__ void contiguous(const int *input_inds, int *out_inds, int *buffer, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    buffer[tid] = -1; // reset buffer for counting
    int this_inds = input_inds[tid];
    atomicCAS(&buffer[this_inds], -1, 1);
    if (tid == 0) out_inds[0] = 0;
    __syncthreads();
    if (buffer[tid] == 1){
        int cnt = atomicAdd(&out_inds[0], 1);
        buffer[tid] = cnt;
    }
    __syncthreads();
    out_inds[tid] = buffer[this_inds];
}

void verify(const float *const __restrict__ points, const int *const __restrict__ labels, const int N, const int MAXNeighbor, float thresh, int mode) {
    clock_t start, end;
    start = clock();
    int *adj_matrix = new int[N * MAXNeighbor]();
    int *adj_len = new int[N]();
    for (int i = 0; i < N; i++) {
        const float *current_pts = points + 3 * i;
        for (int j = i + 1; j < N; j++) {
            if (labels && (labels[i] != labels[j]))
                continue;
            const float *temp_pts = points + 3 * j;
            float dist;
            if (mode == 3) {
                float delta_x = current_pts[0] - temp_pts[0];
                float delta_y = current_pts[1] - temp_pts[1];
                float delta_z = current_pts[2] - temp_pts[2];
                dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
            } else {
                float delta_x = current_pts[0] - temp_pts[0];
                float delta_y = current_pts[1] - temp_pts[1];
                dist = delta_x * delta_x + delta_y * delta_y;
            }
            if (dist < thresh) {
                if (adj_len[i] < MAXNeighbor)
                    adj_matrix[i * MAXNeighbor + adj_len[i]++] = j;
                if (adj_len[j] < MAXNeighbor)
                    adj_matrix[j * MAXNeighbor + adj_len[j]++] = i;
            }
        }
    }
    int *visited = new int[N]();
    int len = 0;
    for (int vertex = 0; vertex < N; vertex++) {
        if (visited[vertex] == 0) {
            std::queue<int> myQueue;
            visited[vertex] = 1;
            myQueue.push(vertex);
            while (!myQueue.empty()) {
                int temp = myQueue.front();
                myQueue.pop();
                for (int adj_idx = 0; adj_idx < adj_len[temp]; adj_idx++) {
                    int adj_vertex = adj_matrix[temp * MAXNeighbor + adj_idx];
                    if (visited[adj_vertex] == 0) {
                        myQueue.push(adj_vertex);
                        visited[adj_vertex] = 1;
                    }
                }
            }
            len++;
        }
    }
    end = clock(); 
    printf("time cost of cpu ccl: %f ms\n", 1000 * double(end - start) / CLOCKS_PER_SEC);
    printf("number of cc in cpu: %3d\n", len);
    delete[] visited;
    delete[] adj_matrix;
    delete[] adj_len;
}

void get_CCL(const int N, const float *const d_points, const int *const d_labels, const float thresh_dist_squre, int *const components, const int MAXNeighbor, int mode, bool check) {
    // !!!d_points,d_labels为gpu端指针，components为cpu端指针!!!

    // GPUTimer timer_cuda_malloc;
    // timer_cuda_malloc.start();

    // int *parent = new int[N];
    // int *components_cpu = new int[N];
    int *d_adj_matrix;
    int *d_adj_len;
    int *d_parent;
    int *d_wl;

    int adj_mem = N * MAXNeighbor * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_adj_matrix, adj_mem));
    int len_mem = N * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_adj_len, len_mem));
    CHECK_CALL(cudaMemset(d_adj_len, 0, len_mem));
    CHECK_CALL(cudaMalloc(&d_parent, len_mem));
    CHECK_CALL(cudaMalloc(&d_wl, len_mem));
    // printf("time of cudaMalloc and Memset %.4f ms\n", timer_cuda_malloc.stop());


    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 adj_gridSize((N + adj_blockSize - 1) / adj_blockSize, (N + adj_blockSize - 1) / adj_blockSize);

    // GPUTimer timer_adj;
    // timer_adj.start();
    if (mode == 2)
        calc_adj_2<<<adj_gridSize, adj_blockSize, sizeof(float) * 2 * adj_blockSize>>>(d_points, d_labels, thresh_dist_squre, d_adj_matrix, d_adj_len, N, MAXNeighbor);
    else
        calc_adj_3<<<adj_gridSize, adj_blockSize, sizeof(float) * 3 * adj_blockSize>>>(d_points, d_labels, thresh_dist_squre, d_adj_matrix, d_adj_len, N, MAXNeighbor);
    // printf("time of adj %.4f ms\n", timer_adj.stop());

    // GPUTimer timer_ccl;
    // timer_ccl.start();
    init<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, MAXNeighbor);
    compute1<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, d_wl, MAXNeighbor);
    compute2<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, d_wl, MAXNeighbor);
    flatten<<<gridSize, blockSize>>>(d_parent, N);
    contiguous<<<gridSize, blockSize>>>(d_parent, components, d_wl, N); //resue d_wl as buffer
    // printf("time of ccl %.4f ms\n", timer_ccl.stop());

    // GPUTimer timer_cpu;
    // timer_cpu.start();
    // CHECK_CALL(cudaMemcpy(parent, d_parent, len_mem, cudaMemcpyDeviceToHost));

    // std::unordered_map<int, int> myMap;
    // for (int i = 0; i < N; i++) {
    //     if (!myMap.count(parent[i])) {
    //         myMap[parent[i]] = myMap.size();
    //     }
    //     components_cpu[i] = myMap[parent[i]];
    // }
    // CHECK_CALL(cudaMemcpy(components, components_cpu, len_mem, cudaMemcpyHostToDevice));
    // printf("time of cpu %.4f ms\n", timer_cpu.stop());
    // printf("Number of cc in cuda: %3d\n", (int)myMap.size());

    if (check) {
        float *h_points = new float[3 * N];
        int *h_labels = nullptr;
        CHECK_CALL(cudaMemcpy(h_points, d_points, 3 * N * sizeof(float), cudaMemcpyDeviceToHost));
        if (d_labels) {
            h_labels = new int[N];
            CHECK_CALL(cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost));
        }
        verify(h_points, h_labels, N, MAXNeighbor, thresh_dist_squre, mode);
        delete[] h_points;
    }
    // GPUTimer timer_free;
    // timer_free.start();
    CHECK_CALL(cudaFree(d_adj_matrix));
    CHECK_CALL(cudaFree(d_adj_len));
    CHECK_CALL(cudaFree(d_parent));
    CHECK_CALL(cudaFree(d_wl));
    // delete[] parent;
    // delete[] components_cpu;
    // printf("time of cudaFree %.4f ms\n", timer_free.stop());
}

// int main() {
//     // input
//     std::vector<float> points;
//     std::ifstream infile("/home/yangyuxue/CudaPractice/connect/test_data.txt");
//     std::string line, word;
//     if (!infile) {
//         printf("Cannot open test_data.txt");
//         exit(1);
//     }
//     int N = 0;
//     while (std::getline(infile, line)) {
//         std::istringstream words(line);
//         if (line.length() == 0) {
//             continue;
//         }
//         N++;
//         for (int i = 0; i < 3; i++) {
//             if (words >> word) {
//                 points.push_back(std::stod(word));
//             } else {
//                 printf("Error for reading test_data.txt\n");
//                 exit(1);
//             }
//         }
//     }
//     infile.close();
//     float *d_points;
//     int points_mem = N * 3 * sizeof(float);
//     cudaMalloc(&d_points, points_mem);
//     cudaMemcpy(d_points, points.data(), points_mem, cudaMemcpyHostToDevice);

//     int *components = new int[N];
//     float thresh_dist_squre = 0.5;
//     int MAXNeighbor = 100;
//     get_CCL(N, d_points, thresh_dist_squre, components, MAXNeighbor, true);
//     printf("over");
//     delete[] components;
//     cudaFree(d_points);
//     return 0;
// }