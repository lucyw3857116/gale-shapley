

#include <vector>


struct Participant {
    int id; // first half of ids male, second half of ids female
    std::vector<int> preferences;
    int current_partner_id; // only for male
    std::vector<int> proposals; // only for female
};

struct Match {
    int id;
    int partner_id;
};
