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



bool is_stable_matching(const std::vector<int>&participants, int n) {
    for (int m = 0; m < n; m++) {
        int w = participants[m*(n+1)];
        for (int i = 0; i < n; i++) {
            int preferred_w = participants[m*(n+1) + 1 + i];
            if (preferred_w == w) {
                break; // found the current partner, no need to check further
            }
            int her_current = participants[preferred_w*(n+1)];
            int m_rank = participants[preferred_w*(n+1) + 1 + m];
            int her_current_rank = participants[preferred_w*(n+1) + 1 + her_current];
            if (m_rank < her_current_rank) {
                // m prefers preferred_w over w, and preferred_w prefers m over her current partner
                return false; // not stable
            }
        }
    }
    return true;
}


void find_stable_pairs_parallel(std::vector<int>& men, std::vector<int>& propose_next, std::vector<int>& women, int n, int numPreferences, int nproc, int pid, int threshold){
    int typePerProc = (n + nproc - 1) / nproc; // nproc is multiple of n

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, n);
    int local_size = end_idx - start_idx;
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
                int w = men[m*(n+1) + 1 + propose_next[m]];
                proposals.push_back(std::make_pair(w, start_idx + m));
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
            int local_w_idx = w_id - n - start_idx;
            int w_base = local_w_idx * (n + 1);
            int best_candidate = women[w_base];
            std::vector<int> prefs(women.begin() + w_base + 1, women.begin() + w_base + 1 + n);
            int prev_partner = women[w_base];
            for (int m : suitors) {
                if (best_candidate == -1) {
                    best_candidate = m;
                } else {
                    int rank_new = women[w_base + 1 + m];
                    int rank_best = women[w_base + 1 + best_candidate];
                    if (rank_new < rank_best) {
                        best_candidate = m;
                    }
                }
            }
            women[w_base] = best_candidate;
            
            for (int m : suitors) {
                bool accepted = (m == best_candidate);
                responses.emplace_back(m, w_id, accepted);
            }

            if (prev_partner != -1 && prev_partner != best_candidate) {
                // have to reject old partner
                responses.emplace_back(prev_partner, w_id, false);
            }
        
            
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
        for (int i = 0; i < recv_response_data.size(); i += 3) {
            int m = recv_response_data[i];
            int w = recv_response_data[i+1];
            bool accepted = recv_response_data[i+2];
        
            int local_id = m - start_idx;
            if (accepted) {
                men[local_id * (n + 1)] = w;
            } else {
                men[local_id * (n + 1)] = -1;
                // man remains unmatched; will propose to next woman next round
            }
        }
        free_males.clear();
        for (int i = 0; i < local_size; i++) {
            if (men[i * (n + 1)] == -1 && propose_next[i] < numPreferences) {
                free_males.push_back(i);
            }
        }
        
        int local_active = free_males.size();
        int global_active = 0;        
        MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_active <= threshold) {
            stable = true;
            break;
        }
    }
    std::cout << count << " iterations\n";
}


void find_stable_pairs_sequential(std::vector<int>&participants, int n, std::vector<int>&propose_next, std::queue<int>&free_males) {    
    int iterations = 0;
    while (!free_males.empty()) {
        iterations++;
        int m_id = free_males.front();
        free_males.pop();
        int base_m = m_id * (n+1);

        if (propose_next[m_id] >= n){
            // this man has no one else to propose to
            if (participants[base_m] == -1) {
                printf("something wrong here - no match found for man %d\n", m_id);
            }
            continue;
        } 

        int f_id = participants[base_m + 1 + propose_next[m_id]];
        propose_next[m_id]++;

        int base_w = f_id*(n+1);
        int curr_partner = participants[base_w];

        int rank_new = participants[base_w + 1 + m_id];
        int rank_old = n+1;
        if (curr_partner != -1) {
            rank_old = participants[base_w + 1 + curr_partner];
        }

        if (curr_partner == -1 || rank_new < rank_old) {
            participants[base_w] = m_id;
            participants[base_m] = f_id;

            // if she was with someone else, free him
            if (curr_partner != -1) {
                participants[curr_partner*(n+1)] = -1;
                free_males.push(curr_partner);
            }
        } else {
            // she rejects m
            free_males.push(m_id);
        }
    }
    printf("iteratios serial %d\n", iterations);
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
    int num = 0;
    int opt;
    int seed = 42;
    int threshold = 0;
    while ((opt = getopt(argc, argv, "n:s:h:")) != -1) {
        switch (opt) {
            case 's':
                seed = atoi(optarg);
                break;
            case 'n':
                num = atoi(optarg);
                break;
            case 'h':
                threshold = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                MPI_Finalize();
                exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (num <= 0 || seed <= 0) {
        std::cerr << "Usage: " << argv[0] << " -n num -t num_threads -m parallel_mode\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    int typePerProc = (num + nproc - 1) / nproc;

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, num);
    int local_size = end_idx - start_idx;
    std::vector<int> propose_next(local_size, 0);
    std::vector<int> men((end_idx - start_idx) * (num + 1));
    std::vector<int> women((end_idx - start_idx) * (num + 1));
    std::vector<int> participants;


    if (pid == 0) {
        participants.resize(num*2 * (num + 1));
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
            
            if (i < num) {
                // Man: offset woman IDs by +num
                for (int j = 0; j < num; j++) {
                    participants[i * (num + 1) + 1 + j] = prefs[j] + num;
                }
            } else {
                // use inverse ranking
                // int w = i - num;
                for (int rank = 0; rank < num; rank ++) {
                    int man_id = prefs[rank];
                    participants[i * (num + 1) + 1 + man_id] = rank;
                }
            }
        }

    }
    MPI_Scatter(participants.data(), ((end_idx - start_idx) * (num + 1)), MPI_INT, men.data(), ((end_idx - start_idx) * (num + 1)), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(participants.data() + num * (num + 1), ((end_idx - start_idx) * (num + 1)), MPI_INT, women.data(), ((end_idx - start_idx) * (num + 1)), MPI_INT, 0, MPI_COMM_WORLD);


    
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(15) << init_time << '\n';
    const auto compute_start = std::chrono::steady_clock::now();

    
    std::cout << "Running mpi code \n";
    find_stable_pairs_parallel(men, propose_next, women, num, num, nproc, pid, threshold);
    std::cout << "after running mpi code \n";
    
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Collect local man-to-woman matchings
    std::vector<int> local_men_results;
    for (int i = 0; i < men.size() / (num + 1); i++) {
        int partner = men[i * (num + 1)];
        int man_global_id = start_idx + i;
        local_men_results.push_back(man_global_id);
        local_men_results.push_back(partner);
    }

    // Collect local woman-to-man matchings
    std::vector<int> local_women_results;
    for (int i = 0; i < women.size() / (num + 1); i++) {
        int partner = women[i * (num + 1)];
        int woman_global_id = num + start_idx + i;
        local_women_results.push_back(woman_global_id);
        local_women_results.push_back(partner);
    }

    // std::vector<int> free_men_list_serial;
    std::queue<int> free_men_list_serial;
    // free_men_list_serial.reserve(threshold);

    if (pid != 0) {
        // Send man matchings
        int size_men = local_men_results.size();
        MPI_Send(&size_men, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_men_results.data(), size_men, MPI_INT, 0, 1, MPI_COMM_WORLD);

        // Send woman matchings
        int size_women = local_women_results.size();
        MPI_Send(&size_women, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(local_women_results.data(), size_women, MPI_INT, 0, 3, MPI_COMM_WORLD);

    } else {
        // PID 0 includes its own results
        for (size_t i = 0; i < local_men_results.size(); i += 2) {
            int m = local_men_results[i];
            int w = local_men_results[i + 1];
            participants[m * (num + 1)] = w;
            if (w == -1) {
                free_men_list_serial.push(m);
            }
        }
        for (size_t i = 0; i < local_women_results.size(); i += 2) {
            int w = local_women_results[i];
            int m = local_women_results[i + 1];
            participants[w * (num + 1)] = m;
        }

        // Receive from all other processes
        for (int src = 1; src < nproc; src++) {
            // Receive man matchings
            int recv_size_men;
            MPI_Recv(&recv_size_men, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> recv_men_data(recv_size_men);
            MPI_Recv(recv_men_data.data(), recv_size_men, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < recv_size_men; i += 2) {
                int m = recv_men_data[i];
                int w = recv_men_data[i + 1];
                participants[m * (num + 1)] = w;
                if (w == -1) {
                    free_men_list_serial.push(m);
                }
            }

            // Receive woman matchings
            int recv_size_women;
            MPI_Recv(&recv_size_women, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> recv_women_data(recv_size_women);
            MPI_Recv(recv_women_data.data(), recv_size_women, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < recv_size_women; i += 2) {
                int w = recv_women_data[i];
                int m = recv_women_data[i + 1];
                participants[w * (num + 1)] = m;
            }
        }
    }
    
    std::vector<int> counts(nproc), displs(nproc);
    for (int r = 0; r < nproc; ++r) {
      int s = std::min((r+1)*typePerProc, num) - r*typePerProc;
      counts[r] = s;
      displs[r] = r*typePerProc;
    }
    std::vector<int> next_proposal;
    if (pid == 0) {
        next_proposal.resize(num);
    }

    MPI_Gatherv(propose_next.data(), local_size, MPI_INT, next_proposal.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    find_stable_pairs_sequential(participants, num, next_proposal, free_men_list_serial);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const auto finalizing_start = std::chrono::steady_clock::now();

    // validation
    if (pid == 0){
        bool valid;
        valid = is_stable_matching(participants, num);
        if (!valid) {
            std::cerr << "Warning: the matching is not stable.\n";
        } else {
            std::cout << "The matching is stable.\n";
        }
    }
    
    
    MPI_Finalize();

    const double finalizing_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - finalizing_start).count();
    std::cout << "Finalizing time (sec): " << finalizing_time << '\n';

}
