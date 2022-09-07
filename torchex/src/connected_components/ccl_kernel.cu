// Thanks for https://dl.acm.org/doi/10.1145/3208040.3208041
// Modified from https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/
#include "../utils/error.cuh"
#include <assert.h>
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

// 工作流，用于存{度>16的点}
static __device__ int topL, posL;

__global__ void calc_adj(const float *points, const float thresh_dist, int *const __restrict__ adj_matrix, int *const __restrict__ adj_len, const int N, const int MAXNeighbor) {
    // y方向表示主导节点，x方向表示被比较的节点，对于每一个x遍历y
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
                float delta_x = currentPoint[0] - s_points[i * 3], delta_y = currentPoint[1] - s_points[i * 3 + 1], delta_z = currentPoint[2] - s_points[i * 3 + 2];
                float dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                if (dist <= thresh_dist) {
                    adj_matrix[active_idx * MAXNeighbor + atomicAdd(&adj_len[active_idx], 1)] = passive_idx;
                    adj_matrix[passive_idx * MAXNeighbor + atomicAdd(&adj_len[passive_idx], 1)] = active_idx;
                    if ((adj_len[active_idx] >= MAXNeighbor) || (adj_len[passive_idx] >= MAXNeighbor)) {
                        printf("Error! There is a point surrounded with more than %d points! Please enlarge the MAXNeighbor!\n", MAXNeighbor);
                        assert(false);
                    }
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

// 初始化parent数组，每个点指向邻域内第一个比它小的点，没有就指向自己
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

void verify(const float *const __restrict__ points, const int N, const int MAXNeighbor, float thresh) {
    clock_t start, end;
    start = clock();
    int *adj_matrix = new int[N * MAXNeighbor]();
    int *adj_len = new int[N]();
    for (int i = 0; i < N; i++) {
        const float *current_pts = points + 3 * i;
        for (int j = i + 1; j < N; j++) {
            const float *temp_pts = points + 3 * j;
            float delta_x = current_pts[0] - temp_pts[0];
            float delta_y = current_pts[1] - temp_pts[1];
            float delta_z = current_pts[2] - temp_pts[2];
            float dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
            if (dist < thresh) {
                adj_matrix[i * MAXNeighbor + adj_len[i]++] = j;
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
    end = clock(); //结束时间
    printf("cpu计算连通域为%3d\n", len);
    printf("串行BFS用时: %f ms\n", 1000 * double(end - start) / CLOCKS_PER_SEC);
    delete[] visited;
    delete[] adj_matrix;
    delete[] adj_len;
}

void get_CCL(const int N, const float *const d_points, const float thresh_dist, int *const components, const int MAXNeighbor, bool check) {
    // d_points为gpu端指针，components为cpu端指针
    // !!!传入的d_points是gpu端的!!!

    // GPUTimer timer_cuda_malloc;
    // timer_cuda_malloc.start();

    int *parent = new int[N];
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

    // 计算邻接表

    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 adj_gridSize((N + adj_blockSize - 1) / adj_blockSize, (N + adj_blockSize - 1) / adj_blockSize);

    // GPUTimer timer_adj;
    // timer_adj.start();
    calc_adj<<<adj_gridSize, adj_blockSize, sizeof(float) * 3 * adj_blockSize>>>(d_points, thresh_dist, d_adj_matrix, d_adj_len, N, MAXNeighbor);
    // printf("time of adj %.4f ms\n", timer_adj.stop());

    // GPUTimer timer_ccl;
    // timer_ccl.start();
    init<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, MAXNeighbor);
    compute1<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, d_wl, MAXNeighbor);
    compute2<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, d_parent, N, d_wl, MAXNeighbor);
    flatten<<<gridSize, blockSize>>>(d_parent, N);
    // printf("time of ccl %.4f ms\n", timer_ccl.stop());

    // GPUTimer timer_cpu;
    // timer_cpu.start();
    CHECK_CALL(cudaMemcpy(parent, d_parent, len_mem, cudaMemcpyDeviceToHost));

    std::unordered_map<int, int> myMap;
    for (int i = 0; i < N; i++) {
        if (!myMap.count(parent[i])) {
            myMap[parent[i]] = myMap.size();
        }
        components[i] = myMap[parent[i]];
    }
    // printf("time of cpu %.4f ms\n", timer_cpu.stop());
    // printf("CUDA计算连通域数量为%3d\n", (int)myMap.size());

    if (check) {
        float *h_points = new float[3 * N];
        CHECK_CALL(cudaMemcpy(h_points, d_points, 3 * N * sizeof(float), cudaMemcpyDeviceToHost));
        verify(h_points, N, MAXNeighbor, thresh_dist);
        delete[] h_points;
    }
    // GPUTimer timer_free;
    // timer_free.start();
    CHECK_CALL(cudaFree(d_adj_matrix));
    CHECK_CALL(cudaFree(d_adj_len));
    CHECK_CALL(cudaFree(d_parent));
    CHECK_CALL(cudaFree(d_wl));
    delete[] parent;
    // printf("time of cudaFree %.4f ms\n", timer_free.stop());
}

// int main() {
//     // 输入样例
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

//     // 计算连通分量
//     int *components = new int[N];
//     float thresh_dist = 0.5;
//     int MAXNeighbor = 100;
//     get_CCL(N, d_points, thresh_dist, components, MAXNeighbor, true);
//     printf("over");
//     delete[] components;
//     cudaFree(d_points);
//     return 0;
// }