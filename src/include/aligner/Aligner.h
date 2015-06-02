#pragma once

#include <vector>

namespace Aligner {

class Aligner {

    Aligner() = delete;
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

public:
    
    static std::vector<std::vector<double>> calculateIbmModel1(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        int src_num_vocab,
        int trg_num_vocab,
        int num_iteration,
        int src_null_id);

}; // class Aligner

} // namespace Aligner

