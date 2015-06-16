#pragma once

#include <treealigner/Tensor.h>
#include <treealigner/Tree.h>

#include <vector>
#include <tuple>
#include <utility>

namespace TreeAligner {

struct HmmJumpingRange {
    std::vector<int> min;
    std::vector<int> max;
}; // struct HmmJumpingRange

struct HmmModel {
    Tensor2<double> generation_prob;
    std::vector<double> jumping_factor;
    double null_jumping_factor;
    int distance_limit;
}; // struct HmmModel

struct TreeHmmModel {
    Tensor2<double> generation_prob;
    std::vector<double> pop_prob;
    Tensor2<double> move_prob;
    Tensor2<double> push_prob;
    double null_prob;
    int distance_limit;
}; // struct TreeHmmModel

struct TopDownPath {
    int label;
    int next;
}; // struct TopDownPath

class Aligner {

    Aligner() = delete;
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

public:
    
    static Tensor2<double> trainIbmModel1(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration);

    static HmmModel trainHmmModel(
        const std::vector<std::vector<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const Tensor2<double> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration,
        const int distance_limit);

    static TreeHmmModel trainTreeHmmModel(
        const std::vector<Tree<int>> & src_corpus,
        const std::vector<std::vector<int>> & trg_corpus,
        const Tensor2<double> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_num_tags,
        const int src_null_id,
        const int num_iteration,
        const int distance_limit);

    static std::vector<std::pair<int, int>> generateIbmModel1ViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const Tensor2<double> & translation_prob,
        const int src_num_vocab,
        const int src_null_id);

    static std::vector<std::pair<int, int>> generateHmmViterbiAlignment(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const HmmModel & hmm_model,
        const int src_num_vocab,
        const int src_null_id);

private:
    
    static HmmJumpingRange calculateHmmJumpingRange(
        const int src_len,
        const int distance_limit);

    static std::tuple<Tensor2<double>, std::vector<double>> calculateHmmJumpingProbability(
        const std::vector<double> & jumping_factor,
        const double null_jumping_factor,
        const int src_len,
        const int distance_limit,
        const HmmJumpingRange & range);

    static std::tuple<Tensor2<double>, std::vector<double>> performHmmForwardStep(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const Tensor2<double> & translation_prob,
        const Tensor2<double> & jumping_prob,
        const std::vector<double> & null_jumping_prob,
        const int src_null_id,
        const HmmJumpingRange & range);

    static Tensor2<double> performHmmBackwardStep(
        const std::vector<int> & src_sentence,
        const std::vector<int> & trg_sentence,
        const Tensor2<double> & translation_prob,
        const Tensor2<double> & jumping_prob,
        const std::vector<double> & null_jumping_prob,
        const int src_null_id,
        const HmmJumpingRange & range,
        const std::vector<double> & scaling_factor);

    static std::vector<std::vector<TopDownPath>> calculateTopDownPaths(const Tree<int> & tree);

}; // class Aligner

} // namespace TreeAligner

