#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <unistd.h>
#include <iomanip>
#include <nv/target>


#include "galeshapley.h"

__global__ void stable_matching(int n, int *men_pref, int *women_pref, int *male_match, int *woman_match, 
                                int *propose_next, int *is_stable, int *is_stable_global, int *women_lock) {

    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx >= n) {
        return; // prevent out of bounds
    }

    while (true) {
        __syncthreads();
        // a stable matching has been found
        if (*is_stable_global == 0) {
            break;
        }

        if (male_match[m_idx] == -1) { // no match
            *is_stable = 0; // not stable

            int w_idx = men_pref[m_idx * n + propose_next[m_idx]];
        
            bool getLock = false;
            do {
                if(getLock = atomicCAS(&women_lock[w_idx], 0, 1) == 0) {
                    if(woman_match[w_idx] == -1) {
                        woman_match[w_idx] = m_idx;
                        male_match[m_idx] = w_idx;
                    }
                    else if(women_pref[w_idx * n + woman_match[w_idx]] > women_pref[w_idx * n + m_idx]) {
                        male_match[woman_match[w_idx]] = -1;
                        male_match[m_idx] = w_idx;
                        woman_match[w_idx] = m_idx;
                    }
                    propose_next[m_idx]++;
                }
                if(getLock) {
                    atomicCAS(&women_lock[w_idx], 1, 0);
                }
            } while(!getLock);    
        }

        __syncthreads();
        if (threadIdx.x == 0 && *is_stable == 1) {
            // stable from prev iteration so globally stable
            *is_stable_global = 0;
        } else if (threadIdx.x == 0 && *is_stable == 0) {
            // if any changes in next iteration set back to 0
            *is_stable = 1;
        }
    }
    __syncthreads();
}

bool is_stable_func(const std::vector<int>& men_data, const std::vector<int>& women_data, const std::vector<int>& men_match, int n) {
    int cnt = 0;

    for (int m = 0; m < n; ++m) {
        int w = men_match[m];
        if (w == -1) {
            printf("Instability: woman %d has no partner\n", m);
            return false;
        }

        for (int i = 0; i < n; ++i) {
            cnt++;
            int preferred_w = men_data[m * n + i];
            if (preferred_w == w) {
                break;
            }

            // find women's partner
            int her_current = -1;
            for (int other_m = 0; other_m < n; ++other_m) {
                if (men_match[other_m] == preferred_w) {
                    her_current = other_m;
                    break;
                }
            }
            if (her_current == -1) {
                printf("Instability: woman %d has no partner\n", her_current);
                return false;
            }

            int m_rank = women_data[preferred_w * n + m];
            int other_rank = women_data[preferred_w * n + her_current];
            if (m_rank < other_rank) {
                printf("Instability: man %d prefers woman %d, and she prefers him over her current partner %d\n",
                        m, preferred_w, her_current);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    int opt;
    int n;
    while ((opt = getopt(argc, argv, "m:n:")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }

    const auto init_start = std::chrono::steady_clock::now();
    
    std::vector<int> men_data(n * n);
    std::vector<int> women_data(n * n);
    // generate random preferences
    for (int i = 0; i < 2 * n; i++) {
        std::vector<int> prefs(n);
        for (int j = 0; j < n; ++j) {
            prefs[j] = j;
        }
        std::mt19937 rng(i * 1000 + 42);
        std::shuffle(prefs.begin(), prefs.end(), rng);

        if (i < n) {
            // man i, prefs[j] is j-th preference
            for (int j = 0; j < n; ++j) {
                men_data[i * n + j] = prefs[j];
            }
        } else {
            // woman i-n -> inverse ranking
            int w = i - n;
            for (int rank = 0; rank < n; ++rank) {
                int j = prefs[rank];
                women_data[w * n + j] = rank;
            }
        }
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    int *men_pref, *women_pref, *male_match, *woman_match, *propose_next, *is_stable, *is_stable_global, *women_lock;

    size_t mat_size = n * n;
    cudaMalloc(&men_pref, mat_size * sizeof(int));
    cudaMalloc(&women_pref, mat_size * sizeof(int));
    cudaMalloc(&propose_next, (n) * sizeof(int));
    cudaMalloc(&male_match, (n) * sizeof(int));
    cudaMalloc(&woman_match, (n) * sizeof(int));
    cudaMalloc(&women_lock, (n) * sizeof(int));
    cudaMalloc(&is_stable, sizeof(int));
    cudaMalloc(&is_stable_global, sizeof(int));

    cudaMemcpy(men_pref, men_data.data(), mat_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(women_pref, women_data.data(), mat_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(propose_next, 0, (n) * sizeof(int));
    cudaMemset(male_match, -1, (n) * sizeof(int));
    cudaMemset(woman_match, -1, (n) * sizeof(int));
    cudaMemset(women_lock, 0, (n) * sizeof(int));
    int one = 1;
    cudaMemcpy(is_stable, &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(is_stable_global, &one, sizeof(int), cudaMemcpyHostToDevice);

    // kernel
    int threads_per_block = 256; // TODO
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    stable_matching<<<1, n>>>(n, men_pref, women_pref, male_match, woman_match, propose_next, is_stable, is_stable_global, women_lock);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }
    
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    // copy back data
    std::vector<Participant> participants(n * 2);
    std::vector<int> h_men_match(n);
    cudaMemcpy(h_men_match.data(), male_match, n * sizeof(int), cudaMemcpyDeviceToHost);
    bool stable = is_stable_func(men_data, women_data, h_men_match, n);

    // bool stable = is_stable_matching(participants, n);
    std::cout << "Stable? cuda " << (stable ? "yes" : "no") << std::endl;

    cudaFree(men_pref);
    cudaFree(women_pref);
    cudaFree(propose_next);
    cudaFree(male_match);
    cudaFree(woman_match);
    cudaFree(women_lock);
    cudaFree(is_stable);
    cudaFree(is_stable_global);

    return 0;
}

