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
    // printf("%d typePerProc %d\n", pid, typePerProc);

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, n);
    // printf("%d start_idx %d end_idx %d\n", pid, start_idx, end_idx);
    int local_size = end_idx - start_idx;
    // printf("%d local_size %d\n", pid, local_size);
    std::vector<int> propose_next(local_size, 0);
    std::vector<int> free_males;
    for (int i = 0; i < local_size; i++) {
        free_males.push_back(i); // all men are free at first
    }
    // printf("%d before while loop\n", pid);
    bool stable = false;
    int count = 0;
    while (!stable) {
        // printf("%d while loop %d\n", pid, count);
        count += 1;
        std::vector<std::pair<int, int>> proposals; // (woman_id, man_id)

        // printf("%d before create list \n", pid);
        // create list of proposals per processor so each processor only gets proposals for its women
        for (unsigned int i = 0; i < free_males.size(); i++) {
            int m = free_males[i];
            if (propose_next[m] < numPreferences) {
                int w = men[m].preferences[propose_next[m]];
                // proposals.push_back(std::make_pair(w, pid*local_size+m));
                proposals.push_back(std::make_pair(w, start_idx + m));
                // printf("%d m %d w %d\n", pid, m, w);
                // proposals[f].push_back(m);
                propose_next[m]++;
            }
        }

        std::vector<std::vector<int>> send_buffers(nproc); // flattened proposals per proc
        // printf("%d before flattening \n", pid);

        for (const auto& p : proposals) {
            int w = p.first;
            int m = p.second;
            // int target_rank = (w - n) / local_size;
            int woman_global_idx = w - n;
            int target_rank = woman_global_idx / typePerProc;

            // printf("%d w %d m %d\n", pid, w, m);
            // printf("%d target rank %d\n", pid, target_rank);
            send_buffers[target_rank].push_back(w);
            send_buffers[target_rank].push_back(m);
        }

        // every processor sends its proposals relevant processors
        // printf("%d before send counts \n", pid);

        std::vector<int> send_counts(nproc), send_displs(nproc);
        int total_send = 0;
        for (int i = 0; i < nproc; ++i) {
            send_counts[i] = send_buffers[i].size();
            send_displs[i] = total_send;
            total_send += send_counts[i];
        }

        // printf("%d before send data \n", pid);
        
        std::vector<int> send_data(total_send);
        for (int i = 0; i < nproc; ++i) {
            std::copy(send_buffers[i].begin(), send_buffers[i].end(), send_data.begin() + send_displs[i]);
        }
        // printf("%d total_send: %d \n", pid, total_send);
        // printf("%d before alltoallv \n", pid);
        std::vector<int> recv_counts(nproc), recv_displs(nproc);
        // printf("send count size %d\n",send_counts.size());
        // printf("nproc %d\n", nproc);
        // for (int i = 0; i < nproc; ++i) {
        //     printf("%d ", send_counts[i]);
        // }
        // printf("\n");
        // printf("send counts size %d\n",send_counts.size());
        // printf("recv counts size %d\n",recv_counts.size());
        // printf("send_counts.data() = %p\n", (void*)send_counts.data());
        // printf("recv_counts.data() = %p\n", (void*)recv_counts.data());

        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Compute total recv size and displacements
        // printf("%d before recv counts \n", pid);
        int total_recv = 0;
        for (int i = 0; i < nproc; ++i) {
            recv_displs[i] = total_recv;
            // printf("%d recv count %d\n", pid, recv_counts[i]);
            total_recv += recv_counts[i];
        }
        // printf("%d before receive data \n", pid);
        
        std::vector<int> recv_data(total_recv); // flat [w1, m1, w2, m2, ...
        
        MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_INT, recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);
        
        std::vector<std::pair<int, int>> local_received_proposals;
        for (int i = 0; i < total_recv; i += 2) {
            int w = recv_data[i];
            int m = recv_data[i+1];
            local_received_proposals.emplace_back(w, m);
        }
        // printf("%d after alltoallv \n", pid);
        // each processor goes through all the proposals and all women either reject or accept a proposal
        std::unordered_map<int, std::vector<int>> woman_proposals;
        for (auto& [w, m] : local_received_proposals) {
            woman_proposals[w].push_back(m);
        }

        // printf("%d, responses\n", pid);

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
        // printf("%d after responses\n", pid);
        
        // send rejections and accepts to everyone        
        std::vector<std::vector<int>> send_response_buffers(nproc); // flat (m, w, accepted)
        for (auto& [m, w, accepted] : responses) {
            // printf("%d local_size: %d\n", pid, local_size);
            int dest = m / typePerProc;
            // printf("%d dest %d, m: %d\n", pid, dest, m);
            send_response_buffers[dest].push_back(m);
            send_response_buffers[dest].push_back(w);
            send_response_buffers[dest].push_back(accepted ? 1 : 0);
        }
        // printf("%d before flattening responses\n", pid);
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
        // printf("%d before processing responses\n", pid);
        for (int i = 0; i < recv_response_data.size(); i += 3) {
            int m = recv_response_data[i];
            int w = recv_response_data[i+1];
            bool accepted = recv_response_data[i+2];
        
            int local_id = m - start_idx;
            // printf("local id: %d, m: %d, pid: %d\n", local_id, m, pid);
            if (accepted) {
                men[local_id].current_partner_id = w;
            } else {
                men[local_id].current_partner_id = -1;
                new_free_males.push_back(m);
                // man remains unmatched; will propose to next woman next round
            }
        }
        // free_males = new_free_males;
        // printf("%d after processing responses\n", pid);
        free_males.clear();
        for (int i = 0; i < local_size; i++) {
            if (men[i].current_partner_id == -1 && propose_next[i] < numPreferences) {
                free_males.push_back(i);
            }
        }
        // printf("%d after processing free males\n", pid);

        
        int local_active = 0;
        if (!free_males.empty()) {
            stable = false;
            local_active = 1;
        }

        int global_active = 0;        
        MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        // printf("%d global_active %d\n", pid, global_active);
        if (global_active == 0) {
            stable = true;
            break;
        }
        // printf("%d after allreduce\n", pid);
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
    // printf("nproc %d, pid: %d\n", nproc, pid);
    
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
    fin >> num;
    int typePerProc = (num + nproc - 1) / nproc;

    int start_idx = pid * typePerProc;
    int end_idx = std::min((pid + 1) * typePerProc, num);
    std::vector<Participant> men(end_idx - start_idx);
    std::vector<Participant> women(end_idx - start_idx);
    // printf("start: %d, end: %d, pid: %d \n", start_idx, end_idx, pid);

    // printf("start loading nproc %d, pid: %d\n", nproc, pid);
    for (int i = 0; i < num; i++) {
        // printf("i: %d, pid: %d\n", i, pid);
        if (i >= start_idx && i < end_idx) {
            // save it for yourself
            Participant p;
            p.id = i;
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
                p.preferences.push_back(preference-1 + num);
            }
            men[i-pid*typePerProc] = p;    
        } else {
            // ignore the values
            fin >> preferenceNum;
            for (int j = 0; j < preferenceNum; j++) {
                int preference;
                fin >> preference;
            }
        }
    }
    // printf("after reading men %d\n", pid);
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
            women[i-pid*typePerProc] = p;    
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
        // printf("nproc %d, pid: %d\n", nproc, pid);
        find_stable_pairs_parallel(men, women, num, preferenceNum, nproc, pid);
        printf("after running pii code %d\n", pid);
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
    MPI_Barrier(MPI_COMM_WORLD);
    const auto finalizing_start = std::chrono::steady_clock::now();

    //TODO ring reduce to get all the final pairings
    std::vector<int> local_results;
    for (const auto& man : men) {
        if (man.current_partner_id != -1) {
            local_results.push_back(man.id);
            local_results.push_back(man.current_partner_id);
        }
        for (int i = 0; i < local_results.size(); i += 2) {
            // printf("%d local_results: %d %d\n", pid, local_results[i], local_results[i+1]);
        }
    }
    if (pid != 0) {
        int send_size = local_results.size();
        MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_results.data(), send_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
        
    } else {
        for (int i = 0; i < local_results.size(); i += 2) {
            output << local_results[i] << " " << local_results[i+1] << "\n";
            // printf("%d local: %d %d\n", pid, local_results[i], local_results[i+1]);
        }
        for (int src = 1; src < nproc; src++) {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> recv_data(recv_size);
            MPI_Recv(recv_data.data(), recv_size, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < recv_data.size(); i += 2) {
                output << recv_data[i] << " " << recv_data[i+1] << "\n";
                // printf("%d recv: %d %d\n", pid, recv_data[i], recv_data[i+1]);
            }
        }
    }

    const double finalizing_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - finalizing_start).count();
    std::cout << "Finalizing time (sec): " << finalizing_time << '\n';

    // for (int i = 0; i < num; ++i) {
    //     if (participants[i].current_partner_id != -1) {
    //         output << participants[i].id << " " << participants[i].current_partner_id << "\n";
    //     } else {
    //         std::cerr << "Warning: participant " << i << " was not matched.\n";
    //     }
    // }
    
    output.close();
    std::cout << "Matches written to " << output_filename << "\n";
    MPI_Finalize();

}