#pragma once

#include <vector>
#include <tuple>
#include <utility>

namespace TreeAligner {

struct HmmModel {
    std::vector<std::vector<double>> generation_prob;
    std::vector<double> jumping_factor;
    double null_jumping_factor;
}; // struct HmmModel

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

    static HmmModel trainHmmModel(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const std::vector<std::vector<double>> & prior_translation_prob,
        int src_num_vocab,
        int trg_num_vocab,
        int src_null_id,
        int num_iteration,
        int distance_limit);

    static std::vector<std::pair<int, int>> generateIbmModel1ViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const std::vector<std::vector<double>> & translation_prob,
        int src_num_vocab,
        int src_null_id);

    static std::vector<std::pair<int, int>> generateHmmViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const HmmModel & hmm_model,
        int src_num_vocab,
        int src_null_id);

private:
    
    static std::tuple<std::vector<int>, std::vector<int>> calculateHmmJumpingRange(
        int src_len,
        int distance_limit);

    static std::tuple<std::vector<std::vector<double>>, std::vector<double>> calculateHmmJumpingProbability(
        const std::vector<double> & jumping_factor,
        double null_jumping_factor,
        int src_len,
        int distance_limit,
        const std::vector<int> min_jumping_range,
        const std::vector<int> max_jumping_range);

}; // class Aligner

} // namespace TreeAligner

