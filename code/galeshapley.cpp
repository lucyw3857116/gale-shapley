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

        Participant& man = participants[m_id];
        if (propose_next[m_id] >= numPreferences){
            // this man has no one else to propose to
            if (man.current_partner_id == -1) {
                printf("something wrong here - no match found for man %d\n", m_id);
            }
            continue;
        } 

        int f_id = man.preferences[propose_next[m_id]];
        propose_next[m_id]++;

        Participant& woman = participants[f_id];

        // first proposal, accept this one
        if (woman.current_partner_id == -1) {
            man.current_partner_id = f_id;
            woman.current_partner_id = m_id;
        }
        else {
            if (woman.preferenceRank[m_id] < woman.preferenceRank[woman.current_partner_id]) {
                // woman prefers the new man
                participants[woman.current_partner_id].current_partner_id = -1;
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

int main (int argc, char *argv[]) {
    std::string input_filename;
    int opt;
    while ((opt = getopt(argc, argv, "f:")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename)) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Input file: " << input_filename << '\n';
    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }
    int num, groups, popNum, preferenceNum;

    // read the information about the data set
    fin >> num >> groups >> popNum >> preferenceNum;
    std::cout << "num: " << num << ", groups: " << groups << ", popNum: " << popNum << ", preferenceNum: " << preferenceNum << '\n';
    std::vector<Participant> participants(num);
    for (int i = 0; i < num; i++) {
        Participant p;
        p.id = i;
        
        for (int j = 0; j < preferenceNum; j++) {
            int preference;
            fin >> preference;
            p.preferences.push_back(preference);
        }
        if (i >= num/2) { // is a woman
            for (int idx = 0; idx < preferenceNum; ++idx) {
                int man_id = p.preferences[idx];
                p.preferenceRank[man_id] = idx;
            }
        }
        participants[i] = p;
    }

    

    find_stable_pairs(participants, num/2, preferenceNum);

    // go thorugh first half of participants (all males)
    // create txt file that has maleId matchId on each row

    if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
        input_filename.resize(std::size(input_filename) - 4);
    }
    const std::string output_filename = input_filename + "_output.txt";

    std::ofstream output(output_filename, std::fstream::out);
    if (!output) {
        std::cerr << "Unable to open file: " << output_filename << '\n';
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < num/2; ++i) {
        output << participants[i].id << " " << participants[i].current_partner_id << "\n";
    }
    output.close();
    std::cout << "Matches written to " << output_filename << "\n";


}