// Thanks for https://dl.acm.org/doi/10.1145/3208040.3208041
// Modified from https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/
#include "../utils/error.cuh"
#include <cmath>
#include <cstdio>
#include <queue>
#include <time.h>
#include <unordered_map>

static const int blockSize = 256;

// 构建双向工作流，L用于存{16<度≤352的点}，R用于存{度>352的点}，待优化
static __device__ int topL, posL, topR, posR;

__global__ void calc_adj(const float *points, const float thresh_dist, int *const __restrict__ adj_matrix, int *const __restrict__ adj_len, const int N, const int MAXNeighbor) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        adj_len[n] = 0;
        const float *currentPoint = points + 3 * n;
        for (int i = n + 1; i < N; i++) {
            const float *tempPoint = points + 3 * i;
            float delta_x = currentPoint[0] - tempPoint[0], delta_y = currentPoint[1] - tempPoint[1], delta_z = currentPoint[2] - tempPoint[2];
            float dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
            if (dist <= thresh_dist) {
                adj_matrix[n * MAXNeighbor + atomicAdd(&adj_len[n], 1)] = i;
                adj_matrix[i * MAXNeighbor + atomicAdd(&adj_len[i], 1)] = n;
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
__global__ void init(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, const int N, int *const __restrict__ parent, const int MAXNeighbor) {
    int uniq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int increament = blockDim.x * gridDim.x;

    for (int vertex = uniq_idx; vertex < N; vertex += increament) {
        int adj_idx = 0;
        int degree = adj_len[vertex];
        for (; (adj_idx < degree) && (adj_matrix[vertex * MAXNeighbor + adj_idx] >= vertex); adj_idx++) {
        }
        if (adj_idx == degree)
            parent[vertex] = vertex;
        else
            parent[vertex] = adj_matrix[vertex * MAXNeighbor + adj_idx];
    }

    if (uniq_idx == 0) {
        topL = 0;
        posL = 0;
        topR = N - 1;
        posR = N - 1;
    }
}

__global__ void compute1(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, const int N, int *const __restrict__ parent, int *const __restrict__ wl, const int MAXNeighbor) {
    int uniq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int increament = blockDim.x * gridDim.x;

    for (int vertex = uniq_idx; vertex < N; vertex += increament) {
        if (vertex != parent[vertex]) {
            int degree = adj_len[vertex];
            if (degree > 16) {
                int temp_idx;
                if (degree <= 352) {
                    temp_idx = atomicAdd(&topL, 1);
                } else {
                    temp_idx = atomicAdd(&topR, -1);
                }
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

__global__ void compute2(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, const int N, int *const __restrict__ parent, int *const __restrict__ wl, const int MAXNeighbor) {
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

__global__ void compute3(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, const int N, int *const __restrict__ parent, int *const __restrict__ wl, const int MAXNeighbor) {
    __shared__ int vertex_idx;
    if (threadIdx.x == 0) {
        vertex_idx = atomicAdd(&posR, -1);
    }
    __syncthreads();
    while (vertex_idx > topR) {
        int vertex = wl[vertex_idx];
        int root_vertex = Find(vertex, parent);
        for (int adj_idx = threadIdx.x; adj_idx < adj_len[vertex]; adj_idx += blockSize) {
            int adj_vertex = adj_matrix[vertex * MAXNeighbor + adj_idx];
            if (vertex > adj_vertex) {
                int root_adj = Find(adj_vertex, parent);
                Union(root_vertex, root_adj, parent);
            }
        }
        if (threadIdx.x == 0) {
            vertex_idx = atomicAdd(&posR, -1);
        }
        __syncthreads();
    }
}

__global__ void flatten(const int N, int *const __restrict__ parent) {
    int uniq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int increament = blockDim.x * gridDim.x;

    for (int vertex = uniq_idx; vertex < N; vertex += increament) {
        int current = parent[vertex], next;
        while (current > (next = parent[current])) {
            current = next;
        }
        if (parent[vertex] != current)
            parent[vertex] = current;
    }
}

int verify(const int *const __restrict__ adj_matrix, const int *const __restrict__ adj_len, const int N, const int MAXNeighbor) {
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
    delete[] visited;
    return len;
}

void get_CCL(const int N, const float *const d_points, const float thresh_dist, int *const components, const int MAXNeighbor, bool check) {
    // d_points为gpu端指针，components为cpu端指针
    // !!!传入的d_points是gpu端的!!!

    int *parent;
    CHECK_CALL(cudaHostAlloc(&parent, N * sizeof(int), cudaHostAllocDefault));

    // 构建邻接表
    // 初始化邻接表和每个节点的度
    int *adj_matrix = new int[N * MAXNeighbor];
    int *adj_len = new int[N];
    for (int i = 0; i < N * MAXNeighbor; i++)
        adj_matrix[i] = -1;
    // float *d_points;
    int *d_adj_matrix;
    int *d_adj_len;
    int *d_parent;
    int *d_wl;
    // int points_mem = N * 3 * sizeof(float);
    // cudaMalloc(&d_points, points_mem);
    // cudaMemcpy(d_points, points, points_mem, cudaMemcpyHostToDevice);
    int adj_mem = N * MAXNeighbor * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_adj_matrix, adj_mem));
    CHECK_CALL(cudaMemcpy(d_adj_matrix, adj_matrix, adj_mem, cudaMemcpyHostToDevice));
    int len_mem = N * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_adj_len, len_mem));
    CHECK_CALL(cudaMalloc(&d_parent, len_mem));
    CHECK_CALL(cudaMalloc(&d_wl, len_mem));
    // 计算邻接表
    int gridSize = (N + blockSize - 1) / blockSize;

    calc_adj<<<gridSize, blockSize>>>(d_points, thresh_dist, d_adj_matrix, d_adj_len, N, MAXNeighbor);
    init<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, N, d_parent, MAXNeighbor);
    compute1<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, N, d_parent, d_wl, MAXNeighbor);
    compute2<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, N, d_parent, d_wl, MAXNeighbor);
    compute3<<<gridSize, blockSize>>>(d_adj_matrix, d_adj_len, N, d_parent, d_wl, MAXNeighbor);
    flatten<<<gridSize, blockSize>>>(N, d_parent);

    CHECK_CALL(cudaMemcpy(adj_matrix, d_adj_matrix, adj_mem, cudaMemcpyDeviceToHost));
    CHECK_CALL(cudaMemcpy(adj_len, d_adj_len, len_mem, cudaMemcpyDeviceToHost));
    CHECK_CALL(cudaMemcpy(parent, d_parent, len_mem, cudaMemcpyDeviceToHost));

    std::unordered_map<int, int> myMap;
    for (int i = 0; i < N; i++) {
        if (!myMap.count(parent[i])) {
            myMap[parent[i]] = myMap.size();
        }
        components[i] = myMap[parent[i]];
        // printf("第%2d个点所在连通域为%2d\n", i, components[i]);
    }
    printf("CUDA计算连通域数量为%3d\n", (int)myMap.size());
    if (check) {
        clock_t start, end;
        start = clock();
        int num_comp = verify(adj_matrix, adj_len, N, MAXNeighbor);

        end = clock(); //结束时间
        printf("串行BFS合并邻接表用时: %f ms\n", 1000 * double(end - start) / CLOCKS_PER_SEC);
        if (myMap.size() == num_comp)
            printf("all right!\n");
        else {
            printf("连通域数量不匹配，cpu计算为%3d\n", num_comp);
        }
    }
    CHECK_CALL(cudaFree(d_adj_matrix));
    CHECK_CALL(cudaFree(d_adj_len));
    CHECK_CALL(cudaFree(d_parent));
    CHECK_CALL(cudaFree(d_wl));
    delete []adj_matrix;
    delete []adj_len;
    cudaFreeHost(parent);
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