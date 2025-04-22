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


void find_stable_pairs(std::vector<Participant>& participants, int n, int numPreferences) {
    std::vector<int> propose_next(n, 0);
    std::vector<int> free_males;
    for (int i = 0; i < n; i++) {
        free_males.push_back(i); // all men are free at first
    }

    while (!free_males.empty()) {
        // proposal containers for each participant
        std::vector<std::vector<int>> proposals(2*n);
        // each man makes their next proposal
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int f = participants[m].preferences[propose_next[m]];
                proposals[f].push_back(m);
                propose_next[m]++;
            }
        }

        std::vector<bool> is_free(n, false);
        std::vector<int> new_free_men;
        // for each woman, choose the best candidate on her list
        for (int w = n; w < 2*n; w++) {
            int current_partner_id = participants[w].current_partner_id;
            int best_candidate = current_partner_id;
            for (int m : proposals[w]) {
                // if she doesn't have a match yet, choose m
                if (best_candidate == -1) {
                best_candidate = m;
                } else {
                    std::vector<int> prefs = participants[w].preferences;
                    int rank_new = std::find(prefs.begin(), prefs.end(), m) - prefs.begin();
                    int rank_best = std::find(prefs.begin(), prefs.end(), best_candidate) - prefs.begin();
                    if (rank_new < rank_best) {
                        best_candidate = m;
                    }
                }
            }
            // if changes partner
            if (best_candidate != current_partner_id) {
                // if previously had a match, free the former match
                if (current_partner_id != -1) {
                    participants[current_partner_id].current_partner_id = -1;
                    if (!is_free[current_partner_id]) {
                        new_free_men.push_back(current_partner_id);
                        is_free[current_partner_id] = true;
                    }
                }
                participants[w].current_partner_id = best_candidate;
                participants[best_candidate].current_partner_id = w;
            }
        }

        // check if a man is unmatched and still has someone left to propose to
        // add him to the free men list
        for (int m = 0; m < n; m++) {
            if (participants[m].current_partner_id == -1 && 
                propose_next[m] < numPreferences) {
                    if (!is_free[m]) {
                        new_free_men.push_back(m);
                        is_free[m] = true;
                    }
            }
        }
        free_males = new_free_men;

    }
}

bool is_stable_matching(const std::vector<Participant>&participants, int n) {
    for (int m = 0; m < n; m++) {
        int w = participants[m].current_partner_id;
        const auto& m_prefs = participants[m].preferences;
        const auto& w_prefs = participants[w].preferences;
        for (int preferred_w : m_prefs) {
            if (preferred_w == w) {
                break; // found the current partner, no need to check further
            }
            int her_current = participants[preferred_w].current_partner_id;
            const auto& her_prefs = participants[preferred_w].preferences;
            int m_rank = -1;
            int her_current_rank = -1;
            for (int i = 0; i < her_prefs.size(); i++) {
                if (her_prefs[i] == m) {
                    m_rank = i;
                }
                if (her_prefs[i] == her_current) {
                    her_current_rank = i;
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

void find_stable_pairs_parallel(std::vector<Participant>& participants, int n, int numPreferences, int num_threads){
    std::vector<int> propose_next(n, 0);
    std::vector<int> free_males;
    for (int i = 0; i < n; i++) {
        free_males.push_back(i); // all men are free at first
    }

    while (!free_males.empty()) {
        // proposal containers for each participant
        std::vector<std::vector<int>> proposals(2*n);
        // each man makes their next proposal
        // std::vector<std::vector<std::vector<int>>> thread_proposals(num_threads, std::vector<std::vector<int>>(2*n));
        #pragma omp parallel for num_threads(num_threads)
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int f = participants[m].preferences[propose_next[m]];
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
            int current_partner_id = participants[w].current_partner_id;
            int best_candidate = current_partner_id;
            for (int m : proposals[w]) {
                // if she doesn't have a match yet, choose m
                if (best_candidate == -1) {
                best_candidate = m;
                } else {
                    std::vector<int> prefs = participants[w].preferences;
                    int rank_new = std::find(prefs.begin(), prefs.end(), m) - prefs.begin();
                    int rank_best = std::find(prefs.begin(), prefs.end(), best_candidate) - prefs.begin();
                    if (rank_new < rank_best) {
                        best_candidate = m;
                    }
                }
            }
            // if changes partner
            if (best_candidate != current_partner_id) {
                // if previously had a match, free the former match
                if (current_partner_id != -1) {
                    participants[current_partner_id].current_partner_id = -1;
                    if (!is_free[current_partner_id]) {
                        #pragma omp critical
                        new_free_men.push_back(current_partner_id);
                        is_free[current_partner_id] = true;
                    }
                }
                participants[w].current_partner_id = best_candidate;
                participants[best_candidate].current_partner_id = w;
            }
        }

        // check if a man is unmatched and still has someone left to propose to
        // add him to the free men list
        for (int m = 0; m < n; m++) {
            if (participants[m].current_partner_id == -1 && 
                propose_next[m] < numPreferences) {
                    if (!is_free[m]) {
                        new_free_men.push_back(m);
                        is_free[m] = true;
                    }
            }
        }
        free_males = new_free_men;

    }
}

void find_stable_pairs_pii(std::vector<Participant>& participants, int n, int numPreferences, int num_threads){
    int iter = 0;
    // randomized initialization
    std::vector<int> woman_ids;
    for (int i = n; i < 2*n; i++) {
        woman_ids.push_back(i);
    }
    std::mt19937 rng(std::time(nullptr));
    std::shuffle(woman_ids.begin(), woman_ids.end(), rng);
    for (int i = 0; i < n; i++) {
        participants[i].current_partner_id = woman_ids[i];
        participants[woman_ids[i]].current_partner_id = i;
    }

    // finding unstable pairs
    std::vector<std::tuple<int, int, int, int>> unstable_pairs;
    bool stable = false;
    while (!stable) {
        iter++;
        if (iter >= 10000) {
            std::cerr << "Warning: too many iterations, exiting...\n";
            return;
        }
        stable = true;
        unstable_pairs.clear();
        for (int m = 0; m < n; m++) {
            for (int w : participants[m].preferences) {
                if (participants[m].current_partner_id == w) {
                    break; 
                }
                int m_current = participants[m].current_partner_id;
                int w_current = participants[w].current_partner_id;
                const auto& m_prefs = participants[m].preferences;
                int w_rank = std::find(m_prefs.begin(), m_prefs.end(), w) - m_prefs.begin();
                const auto& w_prefs = participants[w].preferences;
                int m_rank = std::find(w_prefs.begin(), w_prefs.end(), m) - w_prefs.begin();
                // bool m_prefers_w = m_current == -1 || participants[m].preferences[w] < participants[m].preferences[m_current];
                // bool w_prefers_m = w_current == -1 || participants[w].preferences[m] < participants[w].preferences[w_current];
                bool m_prefers_w = m_current == -1 || w_rank < std::find(m_prefs.begin(), m_prefs.end(), m_current) - m_prefs.begin();
                bool w_prefers_m = w_current == -1 || m_rank < std::find(w_prefs.begin(), w_prefs.end(), w_current) - w_prefs.begin();
                if (m_prefers_w && w_prefers_m) {
                    // m and w prefer each other over their current partners
                    stable = false;
                    unstable_pairs.emplace_back(m, w, participants[m].preferences[w], participants[w].preferences[m]);
                }
            }
        }
        if (stable) {
            break;
        }

        // iteration phase
        // select the higest-ranked unstable pairs
        std::sort(unstable_pairs.begin(), unstable_pairs.end(), [](const auto& a, const auto& b) {
            int man_pref_a = std::get<2>(a);
            int man_pref_b = std::get<2>(b);
            if (man_pref_a != man_pref_b) {
                return man_pref_a < man_pref_b;
            }
            int woman_pref_a = std::get<3>(a);
            int woman_pref_b = std::get<3>(b);
            return woman_pref_a < woman_pref_b;
        });

        // update matching

        std::unordered_set<int> used_men;
        std::unordered_set<int> used_women;
        for (const auto& [m, w, man_rank, woman_rank] : unstable_pairs) {
            if (used_men.count(m) || used_women.count(w)) continue;
            int m_current = participants[m].current_partner_id;
            int w_current = participants[w].current_partner_id;

            if (m_current != -1) {
                participants[m_current].current_partner_id = -1;
            }
            if (w_current != -1) {
                participants[w_current].current_partner_id = -1;
            }
            participants[m].current_partner_id = w;
            participants[w].current_partner_id = m;
            used_men.insert(m);
            used_women.insert(w);
        }

        // greedy fill
        std::vector<int> unmatched_men;
        std::vector<int> unmatched_women;
        for (int i = 0; i < n; i++) {
            if (participants[i].current_partner_id == -1) {
                unmatched_men.push_back(i);
            }
        }
        for (int i = n; i < 2*n; i++) {
            if (participants[i].current_partner_id == -1) {
                unmatched_women.push_back(i);
            }
        }
        for (size_t i = 0; i < unmatched_men.size(); i++) {
            participants[unmatched_men[i]].current_partner_id = unmatched_women[i];
            participants[unmatched_women[i]].current_partner_id = unmatched_men[i];
        }
    }
}


int main (int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    std::string input_filename, mode;
    int num_threads = 1;
    int opt;
    int num;
    while ((opt = getopt(argc, argv, "m:n:t:")) != -1) {
        switch (opt) {
            case 'n':
                num = atoi(optarg);
                break;
            case 'm':
                mode = optarg;
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }
    
    std::vector<Participant> participants(num*2);
    
    for (int i = 0; i < num * 2; i++) {
        std::vector<int> prefs(num);
        for (int j = 0; j < num; j++) {
            prefs[j] = j;
        }
        std::mt19937 rng(i * 1000 + 42); // check this
        std::shuffle(prefs.begin(), prefs.end(), rng);
        Participant p;
        p.id = i;
        if (i < num) {
            for (int j = 0; j < num; j++) {
                p.preferences.push_back(prefs[j] + num);
            }
        } else {
            for (int j = 0; j < num; j++) {
                p.preferences.push_back(prefs[j]);
            }
        }
        participants[i] = p;
    }
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    if (mode == "s") {
        std::cout << "Running serial code \n";
        find_stable_pairs(participants, num, num);
    } else if (mode == "p") {
        std::cout << "Running shared address code \n";
        find_stable_pairs_parallel(participants, num, num, num_threads);
    } else if (mode == "p1") {
        std::cout << "Running pii code \n";
        find_stable_pairs_pii(participants, num, num, num_threads);
    } else {
        std::cerr << "Invalid mode: " << mode << '\n';
        exit(EXIT_FAILURE);
    }
    
    
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