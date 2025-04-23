#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_set>
#include <tuple>
#include <unistd.h>
#include <random>
#include <vector>
#include <queue>
#include <omp.h>

#include "galeshapley.h"

bool is_stable_matching(const std::vector<int>&participants, int n) {
    for (int m = 0; m < n; m++) {
        int w = participants[m*(n+1)];
        for (int i = 0; i < n; i++) {
            int preferred_w = participants[m*(n+1) + 1 + i];
            if (preferred_w == w) {
                break; // found the current partner, no need to check further
            }
            int her_current = participants[preferred_w*(n+1)];
            int m_rank = -1;
            int her_current_rank = -1;
            for (int j = 0; j < n; j++) {
                int val = participants[preferred_w*(n+1) + 1 + j];
                if (val == m) {
                    m_rank = j;
                }
                if (val == her_current) {
                    her_current_rank = j;
                }
            }
            if (m_rank < her_current_rank) {
                // m prefers preferred_w over w, and preferred_w prefers m over her current partner
                return false; // not stable
            }
        }
    }
    return true;
}

void find_stable_pairs_parallel(std::vector<int>& participants, int n, int numPreferences, int num_threads){
    std::vector<int> propose_next(n, 0);
    std::vector<int> free_males;
    for (int i = 0; i < n; i++) {
        free_males.push_back(i); // all men are free at first
    }

    int iterations = 0;
    auto compute_start = std::chrono::steady_clock::now();

    while (!free_males.empty()) {
        iterations += 1;
        if (iterations % 1000 == 0 || iterations == 1) {
            compute_start = std::chrono::steady_clock::now();
        }
        // proposal containers for each participant
        std::vector<std::vector<int>> proposals(2*n);
        // each man makes their next proposal
        // std::vector<std::vector<std::vector<int>>> thread_proposals(num_threads, std::vector<std::vector<int>>(2*n));
        #pragma omp parallel for num_threads(num_threads)
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int f = participants[m*(n+1) + 1 + propose_next[m]];
                #pragma omp critical
                proposals[f].push_back(m);
                propose_next[m]++;
            }
        }

        std::vector<bool> is_free(n, false);
        std::vector<int> new_free_men;
        // for each woman, choose the best candidate on her list
        #pragma omp parallel for num_threads(num_threads)
        for (int w = n; w < 2*n; w++) {
            int current_partner_id = participants[w*(n+1)];
            int best_candidate = current_partner_id;
            for (int m : proposals[w]) {
                // if she doesn't have a match yet, choose m
                if (best_candidate == -1) {
                best_candidate = m;
                } else {
                    int rank_new = -1; 
                    int rank_best = -1;
                    for (int i = 0; i < n; i++) {
                        int val = participants[w*(n+1) + 1 + i];
                        if (val == m) {
                            rank_new = i;
                        }
                        if (val == best_candidate) {
                            rank_best = i;
                        }
                    }
                    if (rank_new < rank_best) {
                        best_candidate = m;
                    }
                }
            }
            // if changes partner
            if (best_candidate != current_partner_id) {
                // if previously had a match, free the former match
                if (current_partner_id != -1) {
                    participants[current_partner_id*(n+1)] = -1;
                    if (!is_free[current_partner_id]) {
                        #pragma omp critical
                        new_free_men.push_back(current_partner_id);
                        is_free[current_partner_id] = true;
                    }
                }
                participants[w*(n+1)] = best_candidate;
                participants[best_candidate*(n+1)] = w;
            }
        }

        // check if a man is unmatched and still has someone left to propose to
        // add him to the free men list
        for (int m = 0; m < n; m++) {
            if (participants[m*(n+1)] == -1 && 
                propose_next[m] < numPreferences) {
                    if (!is_free[m]) {
                        new_free_men.push_back(m);
                        is_free[m] = true;
                    }
            }
        }
        free_males = new_free_men;
        if (iterations % 1000 == 0 || iterations == 1) {
            const double init_time = (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count());
            // printf("time for iteration %d is %f\n", iterations, init_time);
        }

    }
    printf("total iterations %d\n", iterations);
}

int main (int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    std::string input_filename, mode;
    int num_threads = 1;
    int opt;
    int num;
    int seed = 42;
    while ((opt = getopt(argc, argv, "m:n:t:s:")) != -1) {
        switch (opt) {
            case 'n':
                num = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }
    
    std::vector<int> participants(num*2 * (num + 1));
    for (int i = 0; i < num * 2; i++) {
        // Set current partner to -1
        participants[i * (num + 1)] = -1;

        // Create a preference list
        std::vector<int> prefs(num);
        for (int j = 0; j < num; j++) {
            prefs[j] = j;
        }

        std::mt19937 rng(i * 1000 + seed);  // stable deterministic shuffle
        std::shuffle(prefs.begin(), prefs.end(), rng);

        // Write to the participants array
        for (int j = 0; j < num; j++) {
            if (i < num) {
                // Man: offset woman IDs by +num
                participants[i * (num + 1) + 1 + j] = prefs[j] + num;
            } else {
                // Woman: use man IDs directly
                participants[i * (num + 1) + 1 + j] = prefs[j];
            }
        }
    }
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    
    std::cout << "Running shared address code \n";
    find_stable_pairs_parallel(participants, num, num, num_threads);
    
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
        input_filename.resize(std::size(input_filename) - 4);
    }
    const std::string output_filename = std::to_string(num) + "_output.txt";

    std::ofstream output(output_filename, std::fstream::out);
    if (!output) {
        std::cerr << "Unable to open file: " << output_filename << '\n';
        exit(EXIT_FAILURE);
    }
    
    // validation
    bool valid;
    valid = is_stable_matching(participants, num);
    if (!valid) {
        std::cerr << "Warning: the matching is not stable.\n";
    } else {
        std::cout << "The matching is stable.\n";
    }
}