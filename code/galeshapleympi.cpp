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
    int typePerProc = (n + nproc - 1) / nproc; // nproc is multiple of n

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, n);
    int local_size = end_idx - start_idx;
    std::vector<int> propose_next(local_size, 0);
    std::vector<int> free_males;
    for (int i = 0; i < local_size; i++) {
        free_males.push_back(i); // all men are free at first
    }
    bool stable = false;
    int count = 0;
    while (!stable) {
        count += 1;
        std::vector<std::pair<int, int>> proposals; // (woman_id, man_id)

        // create list of proposals per processor so each processor only gets proposals for its women
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int w = men[m].preferences[propose_next[m]];
                // proposals.push_back(std::make_pair(w, pid*local_size+m));
                proposals.push_back(std::make_pair(w, start_idx + m));
                // proposals[f].push_back(m);
                propose_next[m]++;
            }
        }

        std::vector<std::vector<int>> send_buffers(nproc); // flattened proposals per proc

        for (const auto& p : proposals) {
            int w = p.first;
            int m = p.second;
            // int target_rank = (w - n) / local_size;
            int woman_global_idx = w - n;
            int target_rank = woman_global_idx / typePerProc;

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
            int local_woman_idx = w_id - n - start_idx;
            if (local_woman_idx < 0 || local_woman_idx >= (int)women.size()) {
                // skip women not local to this processor
                continue;
            }
            Participant& woman = women[w_id - n - start_idx]; // 
        
            int best_candidate = woman.current_partner_id;
            for (int m : suitors) {
                if (best_candidate == -1) {
                    best_candidate = m;
                } else {
                    std::vector<int> prefs = woman.preferences;
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
        
            int local_id = m - start_idx;
            if (accepted) {
                men[local_id].current_partner_id = w;
            } else {
                men[local_id].current_partner_id = -1;
                new_free_males.push_back(m);
                // man remains unmatched; will propose to next woman next round
            }
        }
        // free_males = new_free_males;
        free_males.clear();
        for (int i = 0; i < local_size; i++) {
            if (men[i].current_partner_id == -1 && propose_next[i] < numPreferences) {
                free_males.push_back(i);
            }
        }
        
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
    std::cout << count << " iterations\n";
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
    
    std::string mode;
    int num_threads = 0;
    int num = 0;
    int opt;
    int seed;
    while ((opt = getopt(argc, argv, "m:n:t:s:")) != -1) {
        switch (opt) {
            case 's':
                seed = atoi(optarg);
                break;
            case 'm':
                mode = optarg;
                break;
            case 'n':
                num = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (num <= 0 || empty(mode) || num_threads <= 0) {
        std::cerr << "Usage: " << argv[0] << " -n num -t num_threads -m parallel_mode\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    int typePerProc = (num + nproc - 1) / nproc;

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, num);
    std::vector<Participant> men(end_idx - start_idx);
    std::vector<Participant> women(end_idx - start_idx);
    std::vector<Participant> participants(num*2);
    std::vector<int> serialized(num*2*(1+num));
    if (pid == 0) {
        // std::random_device rd;
        // std::mt19937 rng(rd());
        for (int i = 0; i < num * 2; i++) {
            std::vector<int> prefs(num);
            for (int j = 0; j < num; j++) {
                prefs[j] = j;
            }
            std::mt19937 rng(i * 1000 + seed); // check this
            // std::mt19937 rng(std::time(nullptr));
            std:shuffle(prefs.begin(), prefs.end(), rng);
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
            serialized[i*(1+num)] = i;
            for (int j = 0; j < num; j++) {
                serialized[i*(1+num)+j+1] = p.preferences[j];
            }
        }

    }
    // for (const auto& p : participants) {
    //     std::cout << "Participant " << p.id
    //               << " is matched with " << p.current_partner_id
    //               << "\nPreferences: ";
    //     for (int pref : p.preferences) {
    //         std::cout << pref << " ";
    //     }
    //     std::cout << "\n\n";
    // }
    
    MPI_Bcast(serialized.data(), num*2*(1+num), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // if (pid != 0) {
    for (int i = 0; i < num*2; i++) {
        if (i < num) {
            if (i < start_idx || i >= end_idx) {
                continue;
            }
        } else {
            if (i < start_idx + num || i >= end_idx + num) {
                continue;
            }
        }
        Participant p;
        p.id = serialized[i*(1+num)];
        for (int j = 0; j < num; j++) {
            p.preferences.push_back(serialized[i*(1+num)+j+1]);
        }
        participants[i] = p;
        if (i < num) {
            men[i - start_idx] = p;
        } else {
            women[i - num - start_idx] = p;
        }
    }
    // }
    


    
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    if (mode == "s") {
        std::cout << "Running serial code \n";
    } else if (mode == "p") {
        std::cout << "Running mpi code \n";
        find_stable_pairs_parallel(men, women, num, num, nproc, pid);
        printf("after running pii code %d\n", pid);
    } else if (mode == "p1") {
        std::cout << "Running pii code \n";
    } else {
        std::cerr << "Invalid mode: " << mode << '\n';
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    
    
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    
    MPI_Barrier(MPI_COMM_WORLD);
    const auto finalizing_start = std::chrono::steady_clock::now();

    //TODO ring reduce to get all the final pairings
    std::vector<int> local_results;
    for (const auto& man : men) {
        if (man.current_partner_id != -1) {
            local_results.push_back(man.id);
            local_results.push_back(man.current_partner_id);
        }
    }
    if (pid != 0) {
        int send_size = local_results.size();
        MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_results.data(), send_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
        
    } else {
        // Include PID 0's own results
        for (size_t i = 0; i < local_results.size(); i += 2) {
            int m = local_results[i];
            int w = local_results[i + 1];
            participants[m].current_partner_id = w;
            participants[w].current_partner_id = m;
        }

        for (int src = 1; src < nproc; src++) {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> recv_data(recv_size);
            MPI_Recv(recv_data.data(), recv_size, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < recv_size; i+=2) {
                int m = recv_data[i];
                int w = recv_data[i+1];
                participants[m].current_partner_id = w;
                participants[w].current_partner_id = m;
            }

        }
    }

    

    const double finalizing_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - finalizing_start).count();
    std::cout << "Finalizing time (sec): " << finalizing_time << '\n';
    // // validation
    // if (pid == 0){
    //     bool valid;
    //     valid = is_stable_matching(participants, num);
    //     if (!valid) {
    //         std::cerr << "Warning: the matching is not stable.\n";
    //     } else {
    //         std::cout << "The matching is stable.\n";
    //     }
    // }
    
    
    MPI_Finalize();

}