#pragma once

#include <treealigner/types.h>
#include <treealigner/Tree.h>

#include <vector>
#include <tuple>
#include <utility>

namespace TreeAligner {



struct HmmModel {
    Tensor2<double> generation_prob;
    RangedTensor1<double> jumping_factor;
    double null_jumping_factor;
    int distance_limit;
}; // struct HmmModel

struct TreeHmmModel {
    Tensor2<double> generation_prob; // pr[t][s]
    Tensor1<double> pop_factor; // c[tag]
    Tensor1<double> stop_factor; // c[tag]
    Tensor1<RangedTensor1<double>> move_factor; // c[tag][pos]
    Tensor1<RangedTensor1<double>> push_factor; // c[tag][pos]
    double leave_factor;
    double stay_factor;
    double null_factor;
    int move_limit;
    int push_limit;
}; // struct TreeHmmModel

struct TreeTraversalProbability {
    Tensor1<double> pop_prob; // pr[tag]
    Tensor1<double> stop_prob; // pr[tag]
    Tensor1<RangedTensor3<double>> move_prob; // pr[tag][min][max][pos]
    Tensor1<RangedTensor3<double>> push_prob; // pr[tag][min][max][pos]
    double leave_prob;
    double stay_prob;
    double null_prob;
}; // struct TreeHmmJumpingProbabilityTable

struct TopDownPath {
    int label;
    int next;
    int degree;
}; // struct TopDownPath

inline bool operator==(const TopDownPath & a, const TopDownPath & b) {
    return
        a.label == b.label &&
        a.next == b.next &&
        a.degree == b.degree;
}

struct TreeHmmPath {
    enum Operation { POP, STOP, MOVE, PUSH };
    Operation op;
    bool skip;
    int label;
    int distance; // used for MOVE and PUSH
    int range_min; // used for MOVE and PUSH
    int range_max; // used for MOVE and PUSH
}; // struct TreeHmmPath

inline bool operator==(const TreeHmmPath & a, const TreeHmmPath & b) {
    return
        a.op == b.op &&
        a.label == b.label &&
        a.distance == b.distance &&
        a.range_min == b.range_min &&
        a.range_max == b.range_max;
}

class Aligner {

    Aligner() = delete;
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

public:
    
    static Tensor2<double> trainIbmModel1(
        const std::vector<Sentence<int>> & src_corpus,
        const std::vector<Sentence<int>> & trg_corpus,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration);

    static HmmModel trainHmmModel(
        const std::vector<Sentence<int>> & src_corpus,
        const std::vector<Sentence<int>> & trg_corpus,
        const Tensor2<double> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_null_id,
        const int num_iteration,
        const int distance_limit);

    static TreeHmmModel trainTreeHmmModel(
        const std::vector<Tree<int>> & src_corpus,
        const std::vector<Sentence<int>> & trg_corpus,
        const Tensor2<double> & prior_translation_prob,
        const int src_num_vocab,
        const int trg_num_vocab,
        const int src_num_tags,
        const int src_null_id,
        const int num_iteration,
        const int move_limit,
        const int push_limit);

    static std::vector<Alignment> generateIbmModel1ViterbiAlignment(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const Tensor2<double> & translation_prob,
        const int src_num_vocab,
        const int src_null_id);

    static std::vector<Alignment> generateHmmViterbiAlignment(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const HmmModel & model,
        const int src_num_vocab,
        const int src_null_id);

private:

    static TreeTraversalProbability calculateTreeTraversalProbability(
        const TreeHmmModel & model);
    
    static std::vector<std::vector<TopDownPath>> calculateTopDownPaths(
        const Tree<int> & tree);

    static Tensor2<std::vector<TreeHmmPath>> calculateTreeHmmPaths(
        const std::vector<std::vector<TopDownPath>> & topdown_paths,
        const int move_limit,
        const int push_limit);

    static std::tuple<Tensor2<double>, Tensor1<double>> calculateTreeHmmJumpingProbability(
        const TreeHmmModel & model,
        const TreeTraversalProbability & traversal_prob,
        const Tensor2<std::vector<TreeHmmPath>> & paths);

}; // class Aligner

} // namespace TreeAligner

