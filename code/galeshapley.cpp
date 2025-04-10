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

#include <galeshapley.h>

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
        
    }
    
}