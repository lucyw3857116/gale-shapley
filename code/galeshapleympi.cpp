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
#include <mpi.h>



void find_stable_pairs_parallel(std::vector<Participant>& men, std::vector<Participant>& women, int n, int numPreferences, int nproc, int pid){
    int typePerProc = (n + nproc - 1) / nproc;

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, n);

    std::vector<int> propose_next(typePerProc, 0);
    std::vector<int> free_males;
    for (int i = 0; i < typePerProc; i++) {
        free_males.push_back(i); // all men are free at first
    }

    bool stable = false;
    while (!stable) {
        std::vector<std::pair<int, int>> proposals; // (woman_id, man_id)

        // create list of proposals per processor so each processor only gets proposals for its women
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int w = men[m].preferences[propose_next[m]];
                proposals.push_back(std::make_pair(w, m));
                // proposals[f].push_back(m);
                propose_next[m]++;
            }
        }

        std::vector<std::vector<int>> send_buffers(nproc); // flattened proposals per proc

        for (const auto& p : proposals) {
            int w = p.first;
            int m = p.second;
            int target_rank = (w - n) / typePerProc;
            send_buffers[target_rank].push_back(w);
            send_buffers[target_rank].push_back(m);
        }

        // every processor sends its proposals relevant processors

        std::vector<int> send_counts(nproc), send_displs(nproc);
        int total_send = 0;
        for (int i = 0; i < nproc; ++i) {
            send_counts[i] = send_buffers[i].size();
            send_displs[i] = total_send;
            total_send += send_counts[i];
        }
        
        std::vector<int> send_data(total_send);
        for (int i = 0; i < nproc; ++i) {
            std::copy(send_buffers[i].begin(), send_buffers[i].end(), send_data.begin() + send_displs[i]);
        }
        
        std::vector<int> recv_counts(nproc), recv_displs(nproc);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Compute total recv size and displacements
        int total_recv = 0;
        for (int i = 0; i < nproc; ++i) {
            recv_displs[i] = total_recv;
            total_recv += recv_counts[i];
        }
        
        std::vector<int> recv_data(total_recv); // flat [w1, m1, w2, m2, ...
        
        MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_INT, recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);
        
        std::vector<std::pair<int, int>> local_received_proposals;
        for (int i = 0; i < total_recv; i += 2) {
            int w = recv_data[i];
            int m = recv_data[i+1];
            local_received_proposals.emplace_back(w, m);
        }
        
        // each processor goes through all the proposals and all women either reject or accept a proposal
        std::unordered_map<int, std::vector<int>> woman_proposals;
        for (auto& [w, m] : local_received_proposals) {
            woman_proposals[w].push_back(m);
        }

        std::vector<std::tuple<int, int, bool>> responses; // (man_id, woman_id, accepted)
        for (auto& [w_id, suitors] : woman_proposals) {
            Participant& woman = women[w_id - n]; // 
        
            int best_candidate = woman.current_partner_id;
            for (int m : suitors) {
                if (best_candidate == -1) {
                    best_candidate = m;
                } else {
                    std::vector<int> prefs = women[w_id-n].preferences;
                    int rank_new = std::find(prefs.begin(), prefs.end(), m) - prefs.begin();
                    int rank_best = std::find(prefs.begin(), prefs.end(), best_candidate) - prefs.begin();
                    if (rank_new < rank_best) {
                        best_candidate = m;
                    }
                }
            }
        
            for (int m : suitors) {
                bool accepted = (m == best_candidate);
                responses.emplace_back(m, w_id, accepted);
            }

            if (woman.current_partner_id != -1 && woman.current_partner_id != best_candidate) {
                // have to reject old partner
                responses.emplace_back(woman.current_partner_id, w_id, false);
            }
        
            woman.current_partner_id = best_candidate;
        }
        
        // send rejections and accepts to everyone        
        std::vector<std::vector<int>> send_response_buffers(nproc); // flat (m, w, accepted)
        for (auto& [m, w, accepted] : responses) {
            int dest = m / typePerProc;
            send_response_buffers[dest].push_back(m);
            send_response_buffers[dest].push_back(w);
            send_response_buffers[dest].push_back(accepted ? 1 : 0);
        }

        std::vector<int> send_counts_resp(nproc), send_displs_resp(nproc);
        int total_send_resp = 0;
        for (int i = 0; i < nproc; ++i) {
            send_counts_resp[i] = send_response_buffers[i].size();
            send_displs_resp[i] = total_send_resp;
            total_send_resp += send_counts_resp[i];
        }

        std::vector<int> send_response_data(total_send_resp);
        for (int i = 0; i < nproc; ++i) {
            std::copy(send_response_buffers[i].begin(), send_response_buffers[i].end(), send_response_data.begin() + send_displs_resp[i]);
        }

        std::vector<int> recv_counts_resp(nproc), recv_displs_resp(nproc);
        MPI_Alltoall(send_counts_resp.data(), 1, MPI_INT, recv_counts_resp.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        int total_recv_resp = 0;
        for (int i = 0; i < nproc; ++i) {
            recv_displs_resp[i] = total_recv_resp;
            total_recv_resp += recv_counts_resp[i];
        }
        
        std::vector<int> recv_response_data(total_recv_resp);
        MPI_Alltoallv(send_response_data.data(), send_counts_resp.data(), send_displs_resp.data(), MPI_INT, recv_response_data.data(), recv_counts_resp.data(), recv_displs_resp.data(), MPI_INT, MPI_COMM_WORLD);

        // if no more rejections then stable = true we are done and have found a stable pairing
        std::vector<int> new_free_males;

        for (int i = 0; i < recv_response_data.size(); i += 3) {
            int m = recv_response_data[i];
            int w = recv_response_data[i+1];
            bool accepted = recv_response_data[i+2];
        
            int local_id = m % typePerProc;
            if (accepted) {
                men[local_id].current_partner_id = w;
            } else {
                men[local_id].current_partner_id = -1;
                new_free_males.push_back(m);
                // man remains unmatched; will propose to next woman next round
            }
        }
        free_males = new_free_males;
        
        int local_active = 0;
        if (!free_males.empty()) {
            stable = false;
            local_active = 1;
        }

        int global_active = 0;        
        MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (global_active == 0) {
            stable = true;
            break;
        }
    }
}

int main (int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    int pid;
    int nproc;
  
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes  
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    
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
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename) || empty(mode) || num_threads <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads -m parallel_mode\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    std::cout << "Input file: " << input_filename << '\n';
    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    int num, preferenceNum;

    int typePerProc = (num + nproc - 1) / nproc;

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, num);
    std::vector<Participant> men(typePerProc);
    std::vector<Participant> women(typePerProc);

    fin >> num;

    for (int i = 0; i < num; i++) {
        if (i >= start_idx && i < end_idx) {
            // save it for yourself
            Participant p;
            p.id = i;
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
                p.preferences.push_back(preference-1);
            }
            men[i] = p;    
        } else {
            // ignore the values
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
            }
        }
    }

    for (int i = 0; i < num; i++) {
        if (i >= start_idx && i < end_idx) {
            // save it for yourself
            Participant p;
            p.id = i+num;
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
                p.preferences.push_back(preference-1);
            }
            women[i] = p;    
        } else {
            // ignore the values
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
            }
        }
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    if (mode == "s") {
        std::cout << "Running serial code \n";
    } else if (mode == "p1") {
        std::cout << "Running pii code \n";
        find_stable_pairs_parallel(men, women, num, preferenceNum, nproc, pid);
    } else if (mode == "p2") {
        std::cout << "Running pii-sc code \n";
        // find_stable_pairs_parallel_sc(participants, num, preferenceNum);
    } else {
        std::cerr << "Invalid mode: " << mode << '\n';
        MPI_Finalize();
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
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    
    // for (int i = 0; i < num; ++i) {
    //     output << participants[i].id << " " << participants[i].current_partner_id << "\n";
    // }

    //TODO ring reduce to get all the final pairings
    for (int i = 0; i < num; ++i) {
        if (participants[i].current_partner_id != -1) {
            output << participants[i].id << " " << participants[i].current_partner_id << "\n";
        } else {
            std::cerr << "Warning: participant " << i << " was not matched.\n";
        }
    }
    
    output.close();
    std::cout << "Matches written to " << output_filename << "\n";
    MPI_Finalize();

}