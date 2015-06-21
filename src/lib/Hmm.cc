#include <treealigner/Hmm.h>

#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/irange.hpp>


using namespace std;
using boost::adaptors::reversed;
using boost::irange;

namespace TreeAligner {

HmmJumpingRange Hmm::getFlatJumpingRange(
    const int src_len) {

    return HmmJumpingRange {
        make_tensor1<int>(src_len, 0),
        make_tensor1<int>(src_len, src_len)
    };
}

HmmJumpingRange Hmm::getLimitedJumpingRange(
    const int src_len,
    const int distance_limit) {

    auto rmin = make_tensor1<int>(src_len);
    auto rmax = make_tensor1<int>(src_len);

    for (int is : irange(0, src_len)) {
        rmin[is] = is > distance_limit ? is - distance_limit : 0;
        rmax[is] = is < src_len - distance_limit ? is + distance_limit + 1 : src_len;
    }

    return HmmJumpingRange { std::move(rmin), std::move(rmax) };
}

tuple<Tensor2<double>, Tensor1<double>> Hmm::getJumpingProbability(
    const RangedTensor1<double> & jumping_factor,
    const double null_jumping_factor,
    const int src_len,
    const HmmJumpingRange & range) {

    auto pj = make_tensor2<double>(src_len, src_len, 0.0);
    auto pj_null = make_tensor1<double>(src_len, 0.0);

    // aliases
    const auto & fj = jumping_factor;
    const double fj_null = null_jumping_factor;

    for (int is2 : irange(0, src_len)) {
        double sum = 0.0;

        for (int is : irange(range.min[is2], range.max[is2])) {
            sum += fj[is - is2];
        }
        sum += fj_null;

        for (int is : irange(range.min[is2], range.max[is2])) {
            pj[is][is2] = fj[is - is2] / sum;
        }
        pj_null[is2] = fj_null / sum;
    }

    return make_tuple(std::move(pj), std::move(pj_null));
}

tuple<Tensor2<double>, Tensor1<double>> Hmm::forwardStep(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range) {

    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & pt = translation_prob;
    const auto & pj = jumping_prob;
    const auto & pj_null = null_jumping_prob;

    auto a = make_tensor2<double>(trg_len, 2 * src_len, 0.0);
    auto scale = make_tensor1<double>(trg_len);

    // initial
    {
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt[trg_sent[0]][src_sent[is]];
                a[0][is] = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt[trg_sent[0]][src_null_id];
                a[0][is + src_len] = delta;
                sum += delta;
            }
        }
        const double scale_0 = 1.0 / sum;
        scale[0] = scale_0;
        for (int is : irange(0, src_len)) {
            a[0][is] *= scale_0;
            a[0][is + src_len] *= scale_0;
        }
    }

    // remaining
    for (int it : irange(1, trg_len)) {
        double sum = 0.0;
        for (int is : irange(0, src_len)) {
            const double pt_it_is = pt[trg_sent[it]][src_sent[is]];
            {
                double delta = 0.0;
                for (int is2 : irange(range.min[is], range.max[is])) {
                    delta += (a[it - 1][is2] + a[it - 1][is2 + src_len]) * pj[is][is2] * pt_it_is;
                }
                a[it][is] = delta;
                sum += delta;
            }
            {
                const double delta = (a[it - 1][is] + a[it - 1][is + src_len]) * pj_null[is] * pt[trg_sent[it]][src_null_id];
                a[it][is + src_len] = delta;
                sum += delta;
            }
        }
        const double scale_it = 1.0 / sum;
        scale[it] = scale_it;
        for (int is : irange(0, src_len)) {
            a[it][is] *= scale_it;
            a[it][is + src_len] *= scale_it;
        }
    }

    return make_tuple(std::move(a), std::move(scale));
}

Tensor2<double> Hmm::backwardStep(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range,
    const Tensor1<double> & scaling_factor) {

    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & pt = translation_prob;
    const auto & pj = jumping_prob;
    const auto & pj_null = null_jumping_prob;
    const auto & scale = scaling_factor;

    auto b = make_tensor2<double>(trg_len, 2 * src_len, 0.0);

    // final
    {
        for (int is : irange(0, 2 * src_len)) {
            b[trg_len - 1][is] = scale[trg_len - 1];
        }
    }
    
    // remaining
    for (int it : irange(0, trg_len - 1) | reversed) {
        for (int is : irange(0, src_len)) {
            for (int is2 : irange(range.min[is], range.max[is])) {
                b[it][is] += b[it + 1][is2] * pj[is2][is] * pt[trg_sent[it + 1]][src_sent[is2]];
            }
            b[it][is] += b[it + 1][is + src_len] * pj_null[is] * pt[trg_sent[it + 1]][src_null_id];
            b[it][is] *= scale[it];
            for (int is2 : irange(0, src_len)) {
                b[it][is + src_len] += b[it + 1][is2] * pj[is2][is] * pt[trg_sent[it + 1]][src_sent[is2]];
            }
            b[it][is + src_len] += b[it + 1][is + src_len] * pj_null[is] * pt[trg_sent[it + 1]][src_null_id];
            b[it][is + src_len] *= scale[it];
        }
    }

    return b;
}

tuple<Tensor2<double>, Tensor1<double>, Tensor2<int>> Hmm::viterbiForwardStep(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range) {

    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & pt = translation_prob;
    const auto & pj = jumping_prob;
    const auto & pj_null = null_jumping_prob;

    auto viterbi = make_tensor2<double>(trg_len, 2 * src_len);
    auto scale = make_tensor1<double>(trg_len);
    auto prev = make_tensor2<int>(trg_len, 2 * src_len);

    {
        // initial
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt[trg_sent[0]][src_sent[is]];
                viterbi[0][is] = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt[trg_sent[0]][src_null_id];
                viterbi[0][is + src_len] = delta;
                sum += delta;
            }
        }
        const double scale_0 = 1.0 / sum;
        scale[0] = scale_0;
        for (int is : irange(0, src_len)) {
            viterbi[0][is] *= scale_0;
            viterbi[0][is + src_len] *= scale_0;
        }
    }
    for (int it : irange(1, trg_len)) {
        // remaining
        double sum = 0.0;
        for (int is : irange(0, src_len)) {
            {
                double pt_it_is = pt[trg_sent[it]][src_sent[is]];
                for (int is2 : irange(range.min[is], range.max[is])) {
                    const double pj_and_pt = pj[is][is2] * pt_it_is;
                    {
                        const double score = viterbi[it - 1][is2] * pj_and_pt;
                        if (score > viterbi[it][is]) {
                            viterbi[it][is] = score;
                            prev[it][is] = is2;
                        }
                    }
                    {
                        const double score = viterbi[it - 1][is2 + src_len] * pj_and_pt;
                        if (score > viterbi[it][is]) {
                            viterbi[it][is] = score;
                            prev[it][is] = is2 + src_len;
                        }
                    }
                }
                sum += viterbi[it][is];
            }
            {
                const double pj_and_pt = pj_null[is] * pt[trg_sent[it]][src_null_id];
                if (viterbi[it - 1][is] > viterbi[it - 1][is + src_len]) {
                    viterbi[it][is + src_len] = viterbi[it - 1][is] * pj_and_pt;
                    prev[it][is + src_len] = is;
                } else {
                    viterbi[it][is + src_len] = viterbi[it - 1][is + src_len] * pj_and_pt;
                    prev[it][is + src_len] = is + src_len;
                }
                sum += viterbi[it][is + src_len];
            }
        }
        const double scale_it = 1.0 / sum;
        scale[it] = scale_it;
        for (int is : irange(0, src_len)) {
            viterbi[it][is] *= scale_it;
            viterbi[it][is + src_len] *= scale_it;
        }
    }

    return make_tuple(std::move(viterbi), std::move(scale), std::move(prev));
}


Tensor3<double> Hmm::getEdgeProbability(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range,
    const Tensor2<double> & forward_prob,
    const Tensor2<double> & backward_prob) {
    
    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & pt = translation_prob;
    const auto & pj = jumping_prob;
    const auto & pj_null = null_jumping_prob;
    const auto & a = forward_prob;
    const auto & b = backward_prob;

    auto xi = make_tensor3<double>(trg_len, 2 * src_len, 2 * src_len);

    for (int it : irange(1, trg_len)) {
        auto & xi_it = xi[it];
        const auto & pt_trg = pt[trg_sent[it]];
        const auto & a_it = a[it - 1];
        const auto & b_it = b[it];
        
        for (int is : irange(0, src_len)) {
            auto & xi_it_is = xi_it[is];
            const auto & pj_is = pj[is];
            const double pt_and_b = pt_trg[src_sent[is]] * b_it[is];

            for (int is2 : irange(range.min[is], range.max[is])) {
                const double pj_and_pt_and_b = pj_is[is2] * pt_and_b;
                xi_it_is[is2] = a_it[is2] * pj_and_pt_and_b;
                xi_it_is[is2 + src_len] = a_it[is2 + src_len] * pj_and_pt_and_b;
            }

            const double pj_and_pt_and_b = pj_null[is] * pt_trg[src_null_id] * b_it[is + src_len];
            xi_it[is + src_len][is] = a_it[is] * pj_and_pt_and_b;
            xi_it[is + src_len][is + src_len] = a_it[is + src_len] * pj_and_pt_and_b;
        }
    }
    
    return xi;
}

Tensor2<double> Hmm::getNodeProbability(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & forward_prob,
    const Tensor2<double> & backward_prob,
    const Tensor1<double> & scaling_factor) {
    
    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & a = forward_prob;
    const auto & b = backward_prob;

    auto gamma = make_tensor2<double>(trg_len, 2 * src_len);

    for (int it : irange(0, trg_len)) {
        auto & gamma_it = gamma[it];
        const auto & a_it = a[it];
        const auto & b_it = b[it];
        const double factor = 1.0 / scaling_factor[it];

        for (int is : irange(0, src_len)) {
            gamma_it[is] = a_it[is] * b_it[is] * factor;
            gamma_it[is + src_len] = a_it[is + src_len] * b_it[is + src_len] * factor;
        }
    }

    return gamma;
}

} // namespace TreeAligner

