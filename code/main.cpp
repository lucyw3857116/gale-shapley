#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <iterator>

int main (int argc, char *argv[]) {
    std::string input_filename;
    int opt;
    while ((opt = getopt(argc, argv, "f:"))) {
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
}