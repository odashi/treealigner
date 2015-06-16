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
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
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
    Tensor2<double> pt(trg_num_vocab, src_num_vocab, 1.0 / (trg_num_vocab - 1));
    
    for (int iteration : irange(0, num_iteration)) {

        Tracer::println(1, format("Iteration %d") % (iteration + 1));
        
        // probabilistic counts
        // c[t][s] = count(t|s)
        Tensor2<double> c(trg_num_vocab, src_num_vocab, 0.0);
        // sumc[s] = sum_t count(t|s)
        vector<double> sumc(src_num_vocab, 0.0);

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            const auto & src_sentence = src_corpus[k];
            const auto & trg_sentence = trg_corpus[k];
            
            // sum of prob. for each target word
            // sumpt[t] = sum_s Pt(t|s)
            vector<double> sumpt(trg_num_vocab, 0.0);

            double likelihood = 0.0;

            //calculate sumpt[t] and ppl
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    const double delta = pt.at(t, s);
                    sumpt[t] += delta;
                    likelihood += delta;
                }
                // null word
                {
                    const double delta = pt.at(t, src_null_id);
                    sumpt[t] += delta;
                    likelihood += delta;
                }
            }

            log_likelihood += log(likelihood) - trg_sentence.size() * log(src_sentence.size() + 1);

            // calculate c[t][s] and sumc[s]
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    const double delta = pt.at(t, s) / sumpt[t];
                    c.at(t, s) += delta;
                    sumc[s] += delta;
                }
                // null word
                {
                    const double delta = pt.at(t, src_null_id) / sumpt[t];
                    c.at(t, src_null_id) += delta;
                    sumc[src_null_id] += delta;
                }
            }
        }

        // calculate pt[t][s]
        for (int t : irange(0, trg_num_vocab)) {
            for (int s : irange(0, src_num_vocab)) {
                pt.at(t, s) = (sumc[s] > 0.0) ? c.at(t, s) / sumc[s] : 0.0;
            }
        }

        Tracer::println(2, format("LL = %.10e") % log_likelihood);
    }

    return pt;
}

HmmModel Aligner::trainHmmModel(
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
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
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, distance_limit >= 0);

    const int num_sentences = src_corpus.size();

    // lexical translation prob: pt[t][s]
    Tensor2<double> pt = prior_translation_prob;
    
    // jumping (transition) factor: fj[d + distance_limit]
    vector<double> fj(2 * distance_limit + 1, 1.0);
    double fj_null = 1.0;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));

        // probabilistic counts
        // ct[t][s]
        Tensor2<double> ct(trg_num_vocab, src_num_vocab, 0.0);
        // sumct[s] = sum_t count(t|s)
        vector<double> sumct(src_num_vocab, 0.0);
        // cj[d + distance_limit] = count(d)
        vector<double> cj(2 * distance_limit + 1, 0.0);
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
            vector<double> pj_null;
            tie(pj, pj_null) = calculateHmmJumpingProbability(fj, fj_null, src_len, distance_limit, range);

            // alpha (forward) scaled prob: a[it][is]
            Tensor2<double> a;
            // scaling factor: scale[it]
            vector<double> scale;
            tie(a, scale) = performHmmForwardStep(src_sentence, trg_sentence, pt, pj, pj_null, src_null_id, range);

            // calculate log likelihood
            for (int it : irange(0, trg_len)) {
                log_likelihood -= log(scale[it]);
            }
            
            // beta (backward) scaled prob: b[it][is]
            Tensor2<double> b = performHmmBackwardStep(src_sentence, trg_sentence, pt, pj, pj_null, src_null_id, range, scale);

            // calculate jumping counts
            // xi[is][is2] = Pr( (it-1, is2) -> (it, is) ) * sum
            Tensor2<double> xi(2 * src_len, 2 * src_len, 0.0);

            for (int it : irange(1, trg_len)) {
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    {
                        const double pt_and_b = pt.at(trg_sentence[it], src_sentence[is]) * b.at(it, is);
                        for (int is2 : irange(range.min[is], range.max[is])) {
                            const double pj_and_pt_and_b = pj.at(is, is2) * pt_and_b;
                            {
                                const double delta = a.at(it - 1, is2) * pj_and_pt_and_b;
                                xi.at(is, is2) = delta;
                                sum += delta;
                            }
                            {
                                const double delta = a.at(it - 1, is2 + src_len) * pj_and_pt_and_b;
                                xi.at(is, is2 + src_len) = delta;
                                sum += delta;
                            }
                        }
                    }
                    {
                        const double pj_and_pt_and_b = pj_null[is] * pt.at(trg_sentence[it], src_null_id) * b.at(it, is + src_len);
                        {
                            const double delta = a.at(it - 1, is) * pj_and_pt_and_b;
                            xi.at(is + src_len, is) = delta;
                            sum += delta;
                        }
                        {
                            const double delta = a.at(it - 1, is + src_len) * pj_and_pt_and_b;
                            xi.at(is + src_len, is + src_len) = delta;
                            sum += delta;
                        }
                    }
                }
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(range.min[is], range.max[is])) {
                        cj[is - is2 + distance_limit] += xi.at(is, is2) / sum;
                        cj[is - is2 + distance_limit] += xi.at(is, is2 + src_len) / sum;
                    }
                    cj_null += xi.at(is + src_len, is) / sum;
                    cj_null += xi.at(is + src_len, is + src_len) / sum;
                }
            }

            // calculate translation counts
            // gamma[is] = Pr( (it, is) ) * sum
            vector<double> gamma(2 * src_len, 0.0);

            for (int it : irange(0, trg_len)) {
                double sum = 0;
                for (int is : irange(0, src_len)) {
                    {
                        const double delta = a.at(it, is) * b.at(it, is);
                        gamma[is] = delta;
                        sum += delta;
                    }
                    {
                        const double delta = a.at(it, is + src_len) * b.at(it, is + src_len);
                        gamma[is + src_len] = delta;
                        sum += delta;
                    }
                }
                for (int is : irange(0, src_len)) {
                    {
                        const double delta = gamma[is] / sum;
                        ct.at(trg_sentence[it], src_sentence[is]) += delta;
                        sumct[src_sentence[is]] += delta;
                    }
                    {
                        const double delta = gamma[is + src_len] / sum;
                        ct.at(trg_sentence[it], src_null_id) += delta;
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
                pt.at(t, s) = (sumct[s] > 0.0) ? ct.at(t, s) / sumct[s] : 0.0;
            }
        }

        Tracer::println(2, format("LL = %.10e") % log_likelihood);
    }

    return HmmModel { std::move(pt), std::move(fj), fj_null, distance_limit };
}

TreeHmmModel Aligner::trainTreeHmmModel(
    const vector<Tree<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
    const Tensor2<double> & prior_translation_prob,
    const int src_num_vocab,
    const int trg_num_vocab,
    const int src_num_tags,
    const int src_null_id,
    const int num_iteration,
    const int distance_limit) {

    Tracer::println(0, "Training Tree HMM model ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_tags > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id < src_num_vocab);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, num_iteration >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, distance_limit >= 0);

    const int num_sentences = src_corpus.size();

    // lexical translation prob: pt[t][s]
    Tensor2<double> pt = prior_translation_prob;
    
    // tree traversal probs:
    // fj_pop[tag]
    vector<double> pj_pop(src_num_tags, 1.0);
    // fj_move[tag][d + distance_limit]
    Tensor2<double> pj_move(src_num_tags, 2 * distance_limit + 1, 1.0);
    // fj_push[tag][d + distance_limit]
    Tensor2<double> pj_push(src_num_tags, 2 * distance_limit + 1, 1.0);
    // fj_null
    double pj_null = 1.0;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));
        
        for (int k : irange(0, num_sentences)) {
            auto topdown_paths = calculateTopDownPaths(src_corpus[k]);
            
            // TODO
        }

    }

    return TreeHmmModel {};
}

vector<pair<int, int>> Aligner::generateIbmModel1ViterbiAlignment(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const int src_num_vocab,
    const int src_null_id) {

    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    vector<pair<int, int>> align;
    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    for (int it : irange(0, trg_len)) {
        const int t = trg_sentence[it];
        int max_is = -1;
        double max_prob = -1.0;
        for (int is : irange(0, src_len)) {
            const double prob = translation_prob.at(t, src_sentence[is]);
            if (prob > max_prob) {
                max_is = is;
                max_prob = prob;
            }
        }
        if (max_prob > translation_prob.at(t, src_null_id)) {
            align.push_back(pair<int, int>(max_is, it));
        }
    }

    return align;
}

vector<pair<int, int>> Aligner::generateHmmViterbiAlignment(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const HmmModel & hmm_model,
    const int src_num_vocab,
    const int src_null_id) {
    
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    // aliases
    const Tensor2<double> & pt = hmm_model.generation_prob;
    const vector<double> & fj = hmm_model.jumping_factor;
    const double fj_null = hmm_model.null_jumping_factor;
    const double dl = hmm_model.distance_limit;

    auto range = calculateHmmJumpingRange(src_len, dl);

    // calculate jumping prob.
    // pj[is][is'] = Pj(is' -> is) = Fj(is - is') / sum[ Fj(j - is') for j = [0, src_len) ]
    Tensor2<double> pj;
    vector<double> pj_null;
    tie(pj, pj_null) = calculateHmmJumpingProbability(fj, fj_null, src_len, dl, range);

    // scaling factor
    // scale[it]
    vector<double> scale(trg_len);

    // scaled Viterbi score
    // viterbi[it][is]
    Tensor2<double> viterbi(trg_len, 2 * src_len, -1.0);
    
    // previous position
    // prev[it][is]
    Tensor2<int> prev(trg_len, 2 * src_len);

    // forward step
    {
        // initial
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt.at(trg_sentence[0], src_sentence[is]);
                viterbi.at(0, is) = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt.at(trg_sentence[0], src_null_id);
                viterbi.at(0, is + src_len) = delta;
                sum += delta;
            }
        }
        const double scale_0 = 1.0 / sum;
        scale[0] = scale_0;
        for (int is : irange(0, src_len)) {
            viterbi.at(0, is) *= scale_0;
            viterbi.at(0, is + src_len) *= scale_0;
        }
    }
    for (int it : irange(1, trg_len)) {
        // remaining
        double sum = 0.0;
        for (int is : irange(0, src_len)) {
            {
                double pt_it_is = pt.at(trg_sentence[it], src_sentence[is]);
                for (int is2 : irange(range.min[is], range.max[is])) {
                    const double pj_and_pt = pj.at(is, is2) * pt_it_is;
                    {
                        const double score = viterbi.at(it - 1, is2) * pj_and_pt;
                        if (score > viterbi.at(it, is)) {
                            viterbi.at(it, is) = score;
                            prev.at(it, is) = is2;
                        }
                    }
                    {
                        const double score = viterbi.at(it - 1, is2 + src_len) * pj_and_pt;
                        if (score > viterbi.at(it, is)) {
                            viterbi.at(it, is) = score;
                            prev.at(it, is) = is2 + src_len;
                        }
                    }
                }
                sum += viterbi.at(it, is);
            }
            {
                const double pj_and_pt = pj_null[is] * pt.at(trg_sentence[it], src_null_id);
                if (viterbi.at(it - 1, is) > viterbi.at(it - 1, is + src_len)) {
                    viterbi.at(it, is + src_len) = viterbi.at(it - 1, is) * pj_and_pt;
                    prev.at(it, is + src_len) = is;
                } else {
                    viterbi.at(it, is + src_len) = viterbi.at(it - 1, is + src_len) * pj_and_pt;
                    prev.at(it, is + src_len) = is + src_len;
                }
                sum += viterbi.at(it, is + src_len);
            }
        }
        const double scale_it = 1.0 / sum;
        scale[it] = scale_it;
        for (int is : irange(0, src_len)) {
            viterbi.at(it, is) *= scale_it;
            viterbi.at(it, is + src_len) *= scale_it;
        }
    }
    
    // backward step
    vector<pair<int, int>> align;
    double max_score = -1.0;
    int pos = -1;
    for (int is : irange(0, src_len)) {
        if (viterbi.at(trg_len - 1, is) > max_score) {
            max_score = viterbi.at(trg_len - 1, is);
            pos = is;
        }
    }
    for (int it : irange(0, trg_len) | reversed) {
        if (pos < src_len) {
            align.push_back(pair<int, int>(pos, it));
        }
        pos = prev.at(it, pos);
    }
    
    return align;
}

HmmJumpingRange Aligner::calculateHmmJumpingRange(
    const int src_len,
    const int distance_limit) {

    vector<int> rmin(src_len);
    vector<int> rmax(src_len);

    for (int is : irange(0, src_len)) {
        rmin[is] = is > distance_limit ? is - distance_limit : 0;
        rmax[is] = is < src_len - distance_limit ? is + distance_limit + 1 : src_len;
    }

    return HmmJumpingRange { std::move(rmin), std::move(rmax) };
}

tuple<Tensor2<double>, vector<double>> Aligner::calculateHmmJumpingProbability(
    const std::vector<double> & jumping_factor,
    const double null_jumping_factor,
    const int src_len,
    const int distance_limit,
    const HmmJumpingRange & range) {

    Tensor2<double> pj(src_len, src_len, 0.0);
    vector<double> pj_null(src_len, 0.0);

    for (int is2 : irange(0, src_len)) {
        double sum = 0.0;

        for (int is : irange(range.min[is2], range.max[is2])) {
            sum += jumping_factor[is - is2 + distance_limit];
        }
        sum += null_jumping_factor;

        for (int is : irange(range.min[is2], range.max[is2])) {
            pj.at(is, is2) = jumping_factor[is - is2 + distance_limit] / sum;
        }
        pj_null[is2] = null_jumping_factor / sum;
    }

    return make_tuple(std::move(pj), std::move(pj_null));
}

tuple<Tensor2<double>, vector<double>> Aligner::performHmmForwardStep(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const vector<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range) {

    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    // aliases
    const Tensor2<double> & pt = translation_prob;
    const Tensor2<double> & pj = jumping_prob;
    const vector<double> & pj_null = null_jumping_prob;

    Tensor2<double> a(trg_len, 2 * src_len, 0.0);
    vector<double> scale(trg_len);

    // initial
    {
        double sum = 0.0;
        const double initial_prob = 1.0 / (2.0 * src_len);
        for (int is : irange(0, src_len)) {
            {
                const double delta = initial_prob * pt.at(trg_sentence[0], src_sentence[is]);
                a.at(0, is) = delta;
                sum += delta;
            }
            {
                const double delta = initial_prob * pt.at(trg_sentence[0], src_null_id);
                a.at(0, is + src_len) = delta;
                sum += delta;
            }
        }
        const double scale_0 = 1.0 / sum;
        scale[0] = scale_0;
        for (int is : irange(0, src_len)) {
            a.at(0, is) *= scale_0;
            a.at(0, is + src_len) *= scale_0;
        }
    }

    // remaining
    for (int it : irange(1, trg_len)) {
        double sum = 0.0;
        for (int is : irange(0, src_len)) {
            const double pt_it_is = pt.at(trg_sentence[it], src_sentence[is]);
            {
                double delta = 0.0;
                for (int is2 : irange(range.min[is], range.max[is])) {
                    delta += (a.at(it - 1, is2) + a.at(it - 1, is2 + src_len)) * pj.at(is, is2) * pt_it_is;
                }
                a.at(it, is) = delta;
                sum += delta;
            }
            {
                const double delta = (a.at(it - 1, is) + a.at(it - 1, is + src_len)) * pj_null[is] * pt.at(trg_sentence[it], src_null_id);
                a.at(it, is + src_len) = delta;
                sum += delta;
            }
        }
        const double scale_it = 1.0 / sum;
        scale[it] = scale_it;
        for (int is : irange(0, src_len)) {
            a.at(it, is) *= scale_it;
            a.at(it, is + src_len) *= scale_it;
        }
    }

    return make_tuple(std::move(a), std::move(scale));
}

Tensor2<double> Aligner::performHmmBackwardStep(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const Tensor2<double> & translation_prob,
    const Tensor2<double> & jumping_prob,
    const vector<double> & null_jumping_prob,
    const int src_null_id,
    const HmmJumpingRange & range,
    const vector<double> & scaling_factor) {

    const int src_len = src_sentence.size();
    const int trg_len = trg_sentence.size();

    // aliases
    const Tensor2<double> & pt = translation_prob;
    const Tensor2<double> & pj = jumping_prob;
    const vector<double> & pj_null = null_jumping_prob;
    const vector<double> & scale = scaling_factor;

    Tensor2<double> b(trg_len, 2 * src_len, 0.0);

    // final
    {
        for (int is : irange(0, 2 * src_len)) {
            b.at(trg_len - 1, is) = scale[trg_len - 1];
        }
    }
    
    // remaining
    for (int it : irange(0, trg_len - 1) | reversed) {
        for (int is : irange(0, src_len)) {
            for (int is2 : irange(range.min[is], range.max[is])) {
                b.at(it, is) += b.at(it + 1, is2) * pj.at(is2, is) * pt.at(trg_sentence[it + 1], src_sentence[is2]);
            }
            b.at(it, is) += b.at(it + 1, is + src_len) * pj_null[is] * pt.at(trg_sentence[it + 1], src_null_id);
            b.at(it, is) *= scale[it];
            for (int is2 : irange(0, src_len)) {
                b.at(it, is + src_len) += b.at(it + 1, is2) * pj.at(is2, is) * pt.at(trg_sentence[it + 1], src_sentence[is2]);
            }
            b.at(it, is + src_len) += b.at(it + 1, is + src_len) * pj_null[is] * pt.at(trg_sentence[it + 1], src_null_id);
            b.at(it, is + src_len) *= scale[it];
        }
    }

    return b;
}

vector<vector<TopDownPath>> Aligner::calculateTopDownPaths(const Tree<int> & tree) {
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
                cur_path.push_back(TopDownPath { node.label(), i });
                recursive(node[i]);
                cur_path.pop_back();
            }
        }
    };

    recursive(tree);

    return paths;
}

} // namespace TreeAligner

