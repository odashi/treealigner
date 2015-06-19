#pragma once

#include <treealigner/types.h>

#include <tuple>

namespace TreeAligner {

struct HmmJumpingRange {
    Tensor1<int> min;
    Tensor1<int> max;
}; // struct HmmJumpingRange

class Hmm {

    Hmm() = delete;
    Hmm(const Hmm &) = delete;
    Hmm & operator=(const Hmm &) = delete;

public:

    static HmmJumpingRange getFlatJumpingRange(
        const int src_len);
    
    static HmmJumpingRange getLimitedJumpingRange(
        const int src_len,
        const int distance_limit);

    static std::tuple<Tensor2<double>, Tensor1<double>> getJumpingProbability(
        const RangedTensor1<double> & jumping_factor,
        const double null_jumping_factor,
        const int src_len,
        const HmmJumpingRange & range);

    static std::tuple<Tensor2<double>, Tensor1<double>> forwardStep(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const Tensor2<double> & translation_prob,
        const Tensor2<double> & jumping_prob,
        const Tensor1<double> & null_jumping_prob,
        const int src_null_id,
        const HmmJumpingRange & range);

    static Tensor2<double> backwardStep(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const Tensor2<double> & translation_prob,
        const Tensor2<double> & jumping_prob,
        const Tensor1<double> & null_jumping_prob,
        const int src_null_id,
        const HmmJumpingRange & range,
        const Tensor1<double> & scaling_factor);

    static Tensor3<double> getEdgeProbability(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const Tensor2<double> & translation_prob,
        const Tensor2<double> & jumping_prob,
        const Tensor1<double> & null_jumping_prob,
        const int src_null_id,
        const HmmJumpingRange & range,
        const Tensor2<double> & forward_prob,
        const Tensor2<double> & backward_prob);

    static Tensor2<double> getNodeProbability(
        const Sentence<int> & src_sent,
        const Sentence<int> & trg_sent,
        const Tensor2<double> & forward_prob,
        const Tensor2<double> & backward_prob,
        const Tensor1<double> & scaling_factor);

}; // class Hmm

} // namespace TreeAligner

