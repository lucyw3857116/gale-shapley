#include <vector>


struct Participant {
    int id; // first half of ids male, second half of ids female
    std::vector<int> preferences;
    // std::unordered_map<int, int> preferenceRank; // for females (rank maleid - index)
    int current_partner_id = -1;
};

struct Match {
    int id;
    int partner_id;
};
