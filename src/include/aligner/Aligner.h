#pragma once

#include <vector>

namespace Aligner {

class Aligner {

    Aligner() = delete;
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

public:
    
    static std::vector<std::vector<double>> trainIbmModel1(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        int src_num_vocab,
        int trg_num_vocab,
        int src_null_id,
        int num_iteration);

    static void trainHmmModel(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const std::vector<std::vector<double>> & prior_translation_prob,
        int src_num_vocab,
        int trg_num_vocab,
        int src_null_id,
        int num_iteration,
        int distance_limit);

}; // class Aligner

} // namespace Aligner

