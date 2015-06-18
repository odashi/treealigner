#include <treealigner/Aligner.h>
#include <treealigner/Tracer.h>
#include <treealigner/assertion.h>

#include <boost/format.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/irange.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using boost::adaptors::reversed;
using boost::format;
using boost::irange;

namespace TreeAligner {

Tensor2<double> Aligner::trainIbmModel1(
    const vector<Sentence<int>> & src_corpus,
    const vector<Sentence<int>> & trg_corpus,
    const int src_num_vocab,
    const int trg_num_vocab,
    const int src_null_id,
    const int num_iteration) {

    Tracer::println(0, "Training IBM model 1 ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_null_id < src_num_vocab);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, num_iteration >= 0);

    const int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    auto pt = make_tensor2<double>(trg_num_vocab, src_num_vocab, 1.0 / (trg_num_vocab - 1));
    
    for (int iteration : irange(0, num_iteration)) {

        Tracer::println(1, format("Iteration %d") % (iteration + 1));
        
        // probabilistic counts
        // c[t][s]
        auto c = make_tensor2<double>(trg_num_vocab, src_num_vocab, 0.0);
        // sumc[s] = sum_t c[t][s]
        auto sumc = make_tensor1<double>(src_num_vocab, 0.0);

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            const auto & src_sentence = src_corpus[k];
            const auto & trg_sentence = trg_corpus[k];
            
            // sum of prob. for each target word
            // sumpt[t] = sum_s Pt(t|s)
            auto sumpt = make_tensor1<double>(trg_num_vocab, 0.0);

            double likelihood = 0.0;

            //calculate sumpt[t] and ppl
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    const double delta = pt[t][s];
                    sumpt[t] += delta;
                    likelihood += delta;
                }
                // null word
                {
                    const double delta = pt[t][src_null_id];
                    sumpt[t] += delta;
                    likelihood += delta;
                }
            }

            log_likelihood += log(likelihood) - trg_sentence.size() * log(src_sentence.size() + 1);

            // calculate c[t][s] and sumc[s]
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    const double delta = pt[t][s] / sumpt[t];
                    c[t][s] += delta;
                    sumc[s] += delta;
                }
                // null word
                {
                    const double delta = pt[t][src_null_id] / sumpt[t];
                    c[t][src_null_id] += delta;
                    sumc[src_null_id] += delta;
                }
            }
        }

        // calculate pt[t][s]
        for (int t : irange(0, trg_num_vocab)) {
            for (int s : irange(0, src_num_vocab)) {
                pt[t][s] = (sumc[s] > 0.0) ? c[t][s] / sumc[s] : 0.0;
            }
        }

        Tracer::println(2, format("LL = %.10e") % log_likelihood);
    }

    return pt;
}

HmmModel Aligner::trainHmmModel(
    const vector<Sentence<int>> & src_corpus,
    const vector<Sentence<int>> & trg_corpus,
    const Tensor2<double> & prior_translation_prob,
    const int src_num_vocab,
    const int trg_num_vocab,
    const int src_null_id,
    const int num_iteration,
    const int distance_limit) {

    Tracer::println(0, "Training HMM model ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id < src_num_vocab);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, num_iteration >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, distance_limit >= 1);

    const int num_sentences = src_corpus.size();

    HmmModel model {
        prior_translation_prob,
        make_ranged_tensor1<double>(-distance_limit, distance_limit, 1.0),
        1.0,
        distance_limit
    };

    // aliases
    auto & pt = model.generation_prob;
    auto & fj = model.jumping_factor;
    auto & fj_null = model.null_jumping_factor;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));

        // probabilistic counts
        // ct[t][s]
        auto ct = make_tensor2<double>(trg_num_vocab, src_num_vocab, 0.0);
        // sumct[s] = sum_t count(t|s)
        auto sumct = make_tensor1<double>(src_num_vocab, 0.0);
        // cj[d + distance_limit] = count(d)
        auto cj = make_ranged_tensor1<double>(-distance_limit, distance_limit, 0.0);
        double cj_null = 0.0;

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            const auto & src_sentence = src_corpus[k];
            const auto & trg_sentence = trg_corpus[k];
            const int src_len = src_sentence.size();
            const int trg_len = trg_sentence.size();

            auto range = calculateHmmJumpingRange(src_len, distance_limit);

            // calculate jumping prob: pj[is][is'] = Pj(is' -> is)
            Tensor2<double> pj;
            Tensor1<double> pj_null;
            tie(pj, pj_null) = calculateHmmJumpingProbability(model, src_len, range);

            // alpha (forward) scaled prob: a[it][is]
            Tensor2<double> a;
            // scaling factor: scale[it]
            Tensor1<double> scale;
            tie(a, scale) = performHmmForwardStep(src_sentence, trg_sentence, pt, pj, pj_null, src_null_id, range);

            // calculate log likelihood
            for (int it : irange(0, trg_len)) {
                log_likelihood -= log(scale[it]);
            }
            
            // beta (backward) scaled prob: b[it][is]
            auto b = performHmmBackwardStep(src_sentence, trg_sentence, pt, pj, pj_null, src_null_id, range, scale);

            // calculate jumping counts
            // xi[is][is2] = Pr( (it-1, is2) -> (it, is) ) * sum
            auto xi = make_tensor2<double>(2 * src_len, 2 * src_len, 0.0);

            for (int it : irange(1, trg_len)) {
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    {
                        const double pt_and_b = pt[trg_sentence[it]][src_sentence[is]] * b[it][is];
                        for (int is2 : irange(range.min[is], range.max[is])) {
                            const double pj_and_pt_and_b = pj[is][is2] * pt_and_b;
                            {
                                const double delta = a[it - 1][is2] * pj_and_pt_and_b;
                                xi[is][is2] = delta;
                                sum += delta;
                            }
                            {
                                const double delta = a[it - 1][is2 + src_len] * pj_and_pt_and_b;
                                xi[is][is2 + src_len] = delta;
                                sum += delta;
                            }
                        }
                    }
                    {
                        const double pj_and_pt_and_b = pj_null[is] * pt[trg_sentence[it]][src_null_id] * b[it][is + src_len];
                        {
                            const double delta = a[it - 1][is] * pj_and_pt_and_b;
                            xi[is + src_len][is] = delta;
                            sum += delta;
                        }
                        {
                            const double delta = a[it - 1][is + src_len] * pj_and_pt_and_b;
                            xi[is + src_len][is + src_len] = delta;
                            sum += delta;
                        }
                    }
                }
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(range.min[is], range.max[is])) {
                        cj[is - is2] += xi[is][is2] / sum;
                        cj[is - is2] += xi[is][is2 + src_len] / sum;
                    }
                    cj_null += xi[is + src_len][is] / sum;
                    cj_null += xi[is + src_len][is + src_len] / sum;
                }
            }

            // calculate translation counts
            // gamma[is] = Pr( (it, is) ) * sum
            auto gamma = make_tensor1<double>(2 * src_len, 0.0);

            for (int it : irange(0, trg_len)) {
                double sum = 0;
                for (int is : irange(0, src_len)) {
                    {
                        const double delta = a[it][is] * b[it][is];
                        gamma[is] = delta;
                        sum += delta;
                    }
                    {
                        const double delta = a[it][is + src_len] * b[it][is + src_len];
                        gamma[is + src_len] = delta;
                        sum += delta;
                    }
                }
                for (int is : irange(0, src_len)) {
                    {
                        const double delta = gamma[is] / sum;
                        ct[trg_sentence[it]][src_sentence[is]] += delta;
                        sumct[src_sentence[is]] += delta;
                    }
                    {
                        const double delta = gamma[is + src_len] / sum;
                        ct[trg_sentence[it]][src_null_id] += delta;
                        sumct[src_null_id] += delta;
                    }
                }
            }
        }

        // set new jumping factors
        fj = cj;
        fj_null = cj_null;

        // calculate pt[t][s]
        for (int t : irange(0, trg_num_vocab)) {
            for (int s : irange(0, src_num_vocab)) {
                pt[t][s] = (sumct[s] > 0.0) ? ct[t][s] / sumct[s] : 0.0;
            }
        }

        Tracer::println(2, format("LL = %.10e") % log_likelihood);
    }

    return model;
}

TreeHmmModel Aligner::trainTreeHmmModel(
    const vector<Tree<int>> & src_corpus,
    const vector<Sentence<int>> & trg_corpus,
    const Tensor2<double> & prior_translation_prob,
    const int src_num_vocab,
    const int trg_num_vocab,
    const int src_num_tags,
    const int src_null_id,
    const int num_iteration,
    const int move_limit,
    const int push_limit) {

    Tracer::println(0, "Training Tree HMM model ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_tags > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id < src_num_vocab);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, num_iteration >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, move_limit >= 1);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, push_limit >= 1);

    const int num_sentences = src_corpus.size();

    TreeHmmModel model {
        prior_translation_prob,
        make_tensor1<double>(src_num_tags, 1.0),
        make_tensor1<double>(src_num_tags, 1.0),
        make_tensor1(src_num_tags, make_ranged_tensor1<double>(-move_limit, move_limit, 1.0)),
        make_tensor1(src_num_tags, make_ranged_tensor1<double>(-push_limit, push_limit, 1.0)),
        1.0,
        0.1,
        0.1,
        move_limit,
        push_limit
    };

    // aliases
    auto & pt = model.generation_prob;
    auto & fj_pop = model.pop_factor;
    auto & fj_stop = model.pop_factor;
    auto & fj_move = model.move_factor;
    auto & fj_push = model.push_factor;
    auto & fj_leave = model.leave_factor;
    auto & fj_stay = model.stay_factor;
    auto & fj_null = model.null_factor;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));
        
        const auto pj_table = calculateTreeHmmJumpingProbabilityTable(model);
        
        for (int k : irange(0, num_sentences)) {
            const auto topdown_paths = calculateTopDownPaths(src_corpus[k]);
            const auto treehmm_paths = calculateTreeHmmPaths(topdown_paths, move_limit, push_limit);
            
            // TODO
        }

    }

    return TreeHmmModel {};
}

vector<Alignment> Aligner::generateIbmModel1ViterbiAlignment(
    const Sentence<int> & src_sentence,
    const Sentence<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const int src_num_vocab,
    const int src_null_id) {

    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    vector<Alignment> align;
    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    for (int it : irange(0, trg_len)) {
        const int t = trg_sentence[it];
        int max_is = -1;
        double max_prob = -1.0;
        for (int is : irange(0, src_len)) {
            const double prob = translation_prob[t][src_sentence[is]];
            if (prob > max_prob) {
                max_is = is;
                max_prob = prob;
            }
        }
        if (max_prob > translation_prob[t][src_null_id]) {
            align.push_back(Alignment { max_is, it });
        }
    }

    return align;
}

vector<Alignment> Aligner::generateHmmViterbiAlignment(
    const Sentence<int> & src_sentence,
    const Sentence<int> & trg_sentence,
    const HmmModel & model,
    const int src_num_vocab,
    const int src_null_id) {
    
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    // aliases
    const auto & pt = model.generation_prob;

    const auto range = calculateHmmJumpingRange(src_len, model.distance_limit);

    // calculate jumping prob.
    // pj[is][is'] = Pj(is' -> is) = Fj(is - is') / sum[ Fj(j - is') for j = [0, src_len) ]
    Tensor2<double> pj;
    Tensor1<double> pj_null;
    tie(pj, pj_null) = calculateHmmJumpingProbability(model, src_len, range);

    // scaling factor
    // scale[it]
    auto scale = make_tensor1<double>(trg_len);

    // scaled Viterbi score
    // viterbi[it][is]
    auto viterbi = make_tensor2<double>(trg_len, 2 * src_len, -1.0);
    
    // previous position
    // prev[it][is]
    auto prev = make_tensor2<double>(trg_len, 2 * src_len);

    // forward step
    {
        // initial
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt[trg_sentence[0]][src_sentence[is]];
                viterbi[0][is] = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt[trg_sentence[0]][src_null_id];
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
                double pt_it_is = pt[trg_sentence[it]][src_sentence[is]];
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
                const double pj_and_pt = pj_null[is] * pt[trg_sentence[it]][src_null_id];
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
    
    // backward step
    vector<Alignment> align;
    double max_score = -1.0;
    int pos = -1;
    for (int is : irange(0, src_len)) {
        if (viterbi[trg_len - 1][is] > max_score) {
            max_score = viterbi[trg_len - 1][is];
            pos = is;
        }
    }
    for (int it : irange(0, trg_len) | reversed) {
        if (pos < src_len) {
            align.push_back(Alignment { pos, it });
        }
        pos = prev[it][pos];
    }
    
    return align;
}

HmmJumpingRange Aligner::calculateHmmJumpingRange(
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

tuple<Tensor2<double>, Tensor1<double>> Aligner::calculateHmmJumpingProbability(
    const HmmModel & model,
    const int src_len,
    const HmmJumpingRange & range) {

    auto pj = make_tensor2(src_len, src_len, 0.0);
    auto pj_null = make_tensor1(src_len, 0.0);

    // aliases
    const auto & fj = model.jumping_factor;
    const double fj_null = model.null_jumping_factor;

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

tuple<Tensor2<double>, Tensor1<double>> Aligner::performHmmForwardStep(
    const Sentence<int> & src_sentence,
    const Sentence<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range) {

    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    // aliases
    const auto & pt = translation_prob;
    const auto & pj = jumping_prob;
    const auto & pj_null = null_jumping_prob;

    auto a = make_tensor2(trg_len, 2 * src_len, 0.0);
    auto scale = make_tensor1<double>(trg_len);

    // initial
    {
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt[trg_sentence[0]][src_sentence[is]];
                a[0][is] = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt[trg_sentence[0]][src_null_id];
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
            const double pt_it_is = pt[trg_sentence[it]][src_sentence[is]];
            {
                double delta = 0.0;
                for (int is2 : irange(range.min[is], range.max[is])) {
                    delta += (a[it - 1][is2] + a[it - 1][is2 + src_len]) * pj[is][is2] * pt_it_is;
                }
                a[it][is] = delta;
                sum += delta;
            }
            {
                const double delta = (a[it - 1][is] + a[it - 1][is + src_len]) * pj_null[is] * pt[trg_sentence[it]][src_null_id];
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

Tensor2<double> Aligner::performHmmBackwardStep(
    const Sentence<int> & src_sentence,
    const Sentence<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const Tensor1<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range,
    const Tensor1<double> & scaling_factor) {

    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

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
                b[it][is] += b[it + 1][is2] * pj[is2][is] * pt[trg_sentence[it + 1]][src_sentence[is2]];
            }
            b[it][is] += b[it + 1][is + src_len] * pj_null[is] * pt[trg_sentence[it + 1]][src_null_id];
            b[it][is] *= scale[it];
            for (int is2 : irange(0, src_len)) {
                b[it][is + src_len] += b[it + 1][is2] * pj[is2][is] * pt[trg_sentence[it + 1]][src_sentence[is2]];
            }
            b[it][is + src_len] += b[it + 1][is + src_len] * pj_null[is] * pt[trg_sentence[it + 1]][src_null_id];
            b[it][is + src_len] *= scale[it];
        }
    }

    return b;
}

TreeHmmJumpingProbabilityTable Aligner::calculateTreeHmmJumpingProbabilityTable(
    const TreeHmmModel & model) {

    // aliases
    const int ml = model.move_limit;
    const int pl = model.push_limit;
    const int num_tags = model.pop_factor.size();

    TreeHmmJumpingProbabilityTable pj_table {
        make_tensor1<double>(model.pop_factor.size(), -1.0),
        make_tensor1<double>(model.stop_factor.size(), -1.0),
        make_tensor1(num_tags, make_ranged_tensor3<double>(-ml, ml, -ml, ml, -ml, ml, -1.0)),
        make_tensor1(num_tags, make_ranged_tensor3<double>(-pl, pl, -pl, pl, -pl, pl, -1.0)),
        -1.0,
        -1.0,
        -1.0
    };

    // leave/stay/null prob.
    {
        double sum = model.leave_factor + model.stay_factor + model.null_factor;
        pj_table.leave_prob = model.leave_factor / sum;
        pj_table.stay_prob = model.stay_factor / sum;
        pj_table.null_prob = model.null_factor / sum;
    }

    // pop/stop prob.
    for (int tag : irange(0, num_tags)) {
        double sum = model.pop_factor[tag] + model.stop_factor[tag];
        pj_table.pop_prob[tag] = model.pop_factor[tag] / sum;
        pj_table.stop_prob[tag] = model.stop_factor[tag] / sum;
    }

    // move prob.
    for (int tag : irange(0, num_tags)) {
        for (int min : irange(-ml, 1)) {
            for (int max : irange(0, ml + 1)) {
                double sum = 0.0;
                // TODO
            }
        }
    }

    return pj_table;
}

vector<vector<TopDownPath>> Aligner::calculateTopDownPaths(
    const Tree<int> & tree) {
    
    vector<vector<TopDownPath>> paths;
    vector<TopDownPath> cur_path;

    function<void(const Tree<int> &)> recursive
        = [&recursive, &paths, &cur_path](const Tree<int> & node) {
        
        int d = node.size();
        
        if (d == 0) {
            // add result
            paths.push_back(cur_path);
        } else {
            // search child
            for (int i : irange(0, d)) {
                cur_path.push_back(TopDownPath { node.label(), i, static_cast<int>(node.size()) });
                recursive(node[i]);
                cur_path.pop_back();
            }
        }
    };

    recursive(tree);

    /*
    for (size_t i : irange(0UL, paths.size())) {
        cout << i << ' ';
        for (auto node : paths[i]) {
            cout << format("[%d; %d/%d] ") % node.label % node.next % (node.degree - 1);
        }
        cout << endl;
    }
    */

    return paths;
}

Tensor2<vector<TreeHmmPath>> Aligner::calculateTreeHmmPaths(
    const vector<vector<TopDownPath>> & topdown_paths,
    const int move_limit,
    const int push_limit) {

    int n = topdown_paths.size();
    auto paths = make_tensor2<vector<TreeHmmPath>>(n, n);

    for (int dst : irange(0, n)) {
        for (int org : irange(0, n)) {
            if (dst == org) continue;

            const auto & dst_path = topdown_paths[dst];
            const auto & org_path = topdown_paths[org];

            size_t top = 0;
            while (dst_path[top] == org_path[top]) ++top;

            auto & path = paths[dst][org];

            // POP/THROUGH step
            for (size_t k : irange(top + 1, org_path.size()) | reversed) {
                auto op = org_path[k].degree == 1 ? TreeHmmPath::THROUGH : TreeHmmPath::POP;
                path.push_back(TreeHmmPath { op, org_path[k].label, 0, 0, 0 });
            }

            // STOP step
            path.push_back(TreeHmmPath { TreeHmmPath::STOP, org_path[top].label, 0, 0, 0 });

            // MOVE step
            {
                int move_distance = dst_path[top].next - org_path[top].next;
                if (move_distance > move_limit || move_distance < -move_limit) {
                    path.clear();
                    continue;
                }

                int move_range_min = -org_path[top].next;
                if (move_range_min < -move_limit) move_range_min = -move_limit;
                int move_range_max = org_path[top].degree - org_path[top].next - 1;
                if (move_range_max > move_limit) move_range_max = move_limit;
                path.push_back(TreeHmmPath { TreeHmmPath::MOVE, dst_path[top].label, move_distance, move_range_min, move_range_max + 1 });
            }

            // PUSH step
            for (size_t k : irange(top + 1, dst_path.size())) {
                int push_pos = dst_path[k].next < dst_path[k].degree - dst_path[k].next ? dst_path[k].next : dst_path[k].next - dst_path[k].degree;
                if (push_pos >= push_limit || push_pos < -push_limit) {
                    path.clear();
                    break;
                }

                int push_range_min = -(dst_path[k].degree / 2);
                if (push_range_min < -push_limit) push_range_min = -push_limit;
                int push_range_max = (dst_path[k].degree - 1) / 2;
                if (push_range_max >= push_limit) push_range_max = push_limit - 1;
                
                if (push_range_min == push_range_max) {
                    path.push_back(TreeHmmPath { TreeHmmPath::THROUGH, dst_path[k].label, 0, 0, 0 });
                } else {
                    path.push_back(TreeHmmPath { TreeHmmPath::PUSH, dst_path[k].label, push_pos, push_range_min, push_range_max + 1 });
                }
            }
        }
    }

    /*
    vector<string> op_str { "POP", "STOP", "MOVE", "PUSH", "THROUGH" };
    for (int org : irange(0, n)) {
        for (int dst : irange(0, n)) {
            cout << format("%d -> %d: ") % org % dst;
            for (auto node : paths.at(dst, org)) {
                if (node.op == TreeHmmPath::POP || node.op == TreeHmmPath::STOP || node.op == TreeHmmPath::THROUGH) {
                    cout << format("[%s; %d] ") % op_str[node.op] % node.label;
                } else {
                    cout << format("[%s; %d, %d/%d:%d] ") % op_str[node.op] % node.label % node.distance % node.range_min % (node.range_max - 1);
                }
            }
            cout << endl;
        }
    }
    */
    
    return paths;
}

} // namespace TreeAligner

