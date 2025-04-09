

#include <vector>

struct Participant {
    int id;
    std::vector<int> preferences;
};

struct Match {
    int id;
    int partner_id;
};
