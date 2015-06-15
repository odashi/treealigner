#pragma once

#include <treealigner/Tree.h>

#include <vector>
#include <tuple>
#include <utility>

namespace TreeAligner {

struct HmmModel {
    std::vector<std::vector<double>> generation_prob;
    std::vector<double> jumping_factor;
    double null_jumping_factor;
    int distance_limit;
}; // struct HmmModel

struct TreeHmmModel {
}; // struct TreeHmmModel

class Aligner {

    Aligner() = delete;
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

public:
    
    static std::vector<std::vector<double>> trainIbmModel1(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration);

    static HmmModel trainHmmModel(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const std::vector<std::vector<double>> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration,
        const int distance_limit);

    static TreeHmmModel trainTreeHmmModel(
        const std::vector<Tree<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const std::vector<std::vector<double>> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_num_tags,
        const int src_null_id,
        const int num_iteration,
        const int distance_limit);

    static std::vector<std::pair<int, int>> generateIbmModel1ViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const std::vector<std::vector<double>> & translation_prob,
        const int src_num_vocab,
        const int src_null_id);

    static std::vector<std::pair<int, int>> generateHmmViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const HmmModel & hmm_model,
        const int src_num_vocab,
        const int src_null_id);

private:
    
    static std::tuple<std::vector<int>, std::vector<int>> calculateHmmJumpingRange(
        const int src_len,
        const int distance_limit);

    static std::tuple<std::vector<std::vector<double>>, std::vector<double>> calculateHmmJumpingProbability(
        const std::vector<double> & jumping_factor,
        const double null_jumping_factor,
        const int src_len,
        const int distance_limit,
        const std::vector<int> & min_jumping_range,
        const std::vector<int> & max_jumping_range);

}; // class Aligner

} // namespace TreeAligner

