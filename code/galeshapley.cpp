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

#include "galeshapley.h"


void find_stable_pairs(std::vector<Participant>& participants, int n, int numPreferences) {
    std::vector<int> propose_next(n, 0); // which woman each man will propose to next
    std::queue<int> free_males;
    for (int i = 0; i < n; i++) {
        free_males.push(i); // all men are free at first
    }

    while (!free_males.empty()) {
        int m_id = free_males.front();
        free_males.pop();

        if (propose_next[m_id] >= numPreferences){
            // this man has no one else to propose to
            printf("something wrong here - no match found for man %d\n", m_id);
            continue;
        } 

        Participant& man = participants[m_id];
        int f_id = man.preferences[propose_next[m_id]];
        propose_next[m_id]++;
        Participant& woman = participants[f_id];

        // check if man is on her preference list
        if (std::find(woman.preferences.begin(), woman.preferences.end(), m_id) == woman.preferences.end()) {
            // reject proposal
            free_males.push(m_id);
            continue;
        }
        
        // first proposal, accept this one
        if (woman.current_partner_id == -1) {
            man.current_partner_id = f_id;
            woman.current_partner_id = m_id;
        }
        else {
            if (std::find(woman.preferences.begin(), woman.preferences.end(), m_id) < std::find(woman.preferences.begin(), woman.preferences.end(), woman.current_partner_id)) {
            // if (woman.preferenceRank[m_id] < woman.preferenceRank[woman.current_partner_id]) {
                // woman prefers the new man
                participants[woman.current_partner_id].current_partner_id = -1; // old partner no longer has match
                free_males.push(woman.current_partner_id);

                woman.current_partner_id = m_id;
                man.current_partner_id = f_id;
            } else {
                // she rejects new proposer
                free_males.push(m_id);
            }
        }
    }

}

void find_stable_pairs_parallel(std::vector<Participant>& participants, int n, int numPreferences, int num_threads){
    // intialize everyone as free
    for (int i = 0; i < 2*n; i++) {
        participants[i].current_partner_id = -1;
    }

    std::vector<int> propose_next(n, 0);
    std::vector<int> free_males;
    for (int i = 0; i < n; i++) {
        free_males.push_back(i); // all men are free at first
    }

    while (!free_males.empty()) {
        // proposal containers for each participant
        std::vector<std::vector<int>> proposals(2*n);
        // each man makes their next proposal
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


int main (int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    std::string input_filename, mode;
    int num_threads = 0;
    int opt;
    while ((opt = getopt(argc, argv, "f:m:n:")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'm':
                mode = optarg;
                break;
            case 'n':
                num_threads = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename) || empty(mode) || num_threads <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads -m parallel_mode\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Input file: " << input_filename << '\n';
    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }
    int num, preferenceNum;

    // read the information about the data set
    fin >> num;
    std::vector<Participant> participants(num*2);
    for (int i = 0; i < num*2; i++) {
        Participant p;
        p.id = i;
        fin >> preferenceNum;
        
        
        if (i >= num) { // is a woman
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
                p.preferences.push_back(preference-1);
            }
            // for (int idx = preferenceNum-1; idx >= 0; idx--) {
            //     int man_id = p.preferences[idx];
            //     p.preferenceRank[man_id] = idx;
            // }
            
        } else {
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
                p.preferences.push_back(preference+num-1);
            }
        }
        participants[i] = p;
        // std::cout << "id: " << p.id << ", preferences: ";
        // for (int pref : p.preferences) {
        //     std::cout << pref << " ";
        // }
        // std::cout << '\n';
    }
    //print out participants
    // std::cout << "Participants:\n";
    // for (auto& p : participants) {
    //     std::cout << "id: " << p.id << ", preferences: ";
    //     for (int pref : p.preferences) {
    //         std::cout << pref << " ";
    //     }
    //     std::cout << '\n';
    // }
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    if (mode == "s") {
        std::cout << "Running serial code \n";
        find_stable_pairs(participants, num, preferenceNum);
    } else if (mode == "p1") {
        std::cout << "Running pii code \n";
        find_stable_pairs_parallel(participants, num, preferenceNum, num_threads);
    } else if (mode == "p2") {
        std::cout << "Running pii-sc code \n";
        // find_stable_pairs_parallel_sc(participants, num, preferenceNum);
    } else {
        std::cerr << "Invalid mode: " << mode << '\n';
        exit(EXIT_FAILURE);
    }
    
    
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
        input_filename.resize(std::size(input_filename) - 4);
    }
    const std::string output_filename = input_filename + "_output.txt";

    std::ofstream output(output_filename, std::fstream::out);
    if (!output) {
        std::cerr << "Unable to open file: " << output_filename << '\n';
        exit(EXIT_FAILURE);
    }
    
    // for (int i = 0; i < num; ++i) {
    //     output << participants[i].id << " " << participants[i].current_partner_id << "\n";
    // }
    for (int i = 0; i < num; ++i) {
        if (participants[i].current_partner_id != -1) {
            output << participants[i].id << " " << participants[i].current_partner_id << "\n";
        } else {
            std::cerr << "Warning: participant " << i << " was not matched.\n";
        }
    }
    
    output.close();
    std::cout << "Matches written to " << output_filename << "\n";

}