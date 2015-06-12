#include <treealigner/Aligner.h>
#include <treealigner/Tracer.h>
#include <treealigner/assertion.h>

#include <boost/format.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/irange.hpp>

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using boost::adaptors::reversed;
using boost::format;
using boost::irange;

namespace TreeAligner {

vector<vector<double>> Aligner::trainIbmModel1(
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
    int src_num_vocab,
    int trg_num_vocab,
    int src_null_id,
    int num_iteration) {

    Tracer::println(0, "Training IBM model 1 ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, num_iteration >= 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateIbmModel1, src_null_id < src_num_vocab);

    int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    vector<vector<double>> pt(trg_num_vocab, vector<double>(src_num_vocab, 1.0 / (trg_num_vocab - 1)));
    
    for (int iteration : irange(0, num_iteration)) {

        Tracer::println(1, format("Iteration %d") % (iteration + 1));
        
        // probabilistic counts
        // c[t][s] = count(t|s)
        vector<vector<double>> c(trg_num_vocab, vector<double>(src_num_vocab, 0.0));
        // sumc[s] = sum_t count(t|s)
        vector<double> sumc(src_num_vocab, 0.0);

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            auto & src_sentence = src_corpus[k];
            auto & trg_sentence = trg_corpus[k];
            
            // sum of prob. for each target word
            // sumpt[t] = sum_s Pt(t|s)
            vector<double> sumpt(trg_num_vocab, 0.0);

            double likelihood = 0.0;

            //calculate sumpt[t] and ppl
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    double delta = pt[t][s];
                    sumpt[t] += delta;
                    likelihood += delta;
                }
                // null word
                double delta = pt[t][src_null_id];
                sumpt[t] += delta;
                likelihood += delta;
            }

            log_likelihood += log(likelihood) - trg_sentence.size() * log(src_sentence.size() + 1);

            // calculate c[t][s] and sumc[s]
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    double delta = pt[t][s] / sumpt[t];
                    c[t][s] += delta;
                    sumc[s] += delta;
                }
                // null word
                double delta = pt[t][src_null_id] / sumpt[t];
                c[t][src_null_id] += delta;
                sumc[src_null_id] += delta;
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
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
    const vector<vector<double>> & prior_translation_prob,
    int src_num_vocab,
    int trg_num_vocab,
    int src_null_id,
    int num_iteration,
    int distance_limit) {

    Tracer::println(0, "Training HMM model ...");

    // check constraints
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_corpus.size() == trg_corpus.size());
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, trg_num_vocab > 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, num_iteration >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, src_null_id < src_num_vocab);
    MYASSERT(TreeAligner::Aligner::calculateHmmModel, distance_limit >= 0);

    int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    vector<vector<double>> pt = prior_translation_prob;
    
    // jumping (transition) factor
    // fj[d + distance_limit] = Fj(d) for x = [-distance_limit, distance_limit]
    vector<double> fj(2 * distance_limit + 1, 1.0);
    double fj_null = 1.0;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));

        // probabilistic counts
        // ct[t][s] = count(t|s)
        vector<vector<double>> ct(trg_num_vocab, vector<double>(src_num_vocab, 0.0));
        // sumct[s] = sum_t count(t|s)
        vector<double> sumct(src_num_vocab, 0.0);
        // cj[d + distance_limit] = count(d)
        vector<double> cj(2 * distance_limit + 1, 0.0);
        double cj_null = 0.0;

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            auto & src_sentence = src_corpus.at(k);
            auto & trg_sentence = trg_corpus.at(k);
            int src_len = src_sentence.size();
            int trg_len = trg_sentence.size();

            // ranges of possible path connections
            vector<int> is_min;
            vector<int> is_max;
            tie(is_min, is_max) = calculateHmmJumpingRange(src_len, distance_limit);

            // calculate jumping prob.
            // pj[is][is'] = Pj(is' -> is) = Fj(is - is') / sum[ Fj(j - is') for j = [0, src_len) ]
            vector<vector<double>> pj;
            vector<double> pj_null;
            tie(pj, pj_null) = calculateHmmJumpingProbability(fj, fj_null, src_len, distance_limit, is_min, is_max);

            // scaling factor
            // scale[it]
            vector<double> scale(trg_len);

            // alpha (forward) scaled prob.
            // a[it][is]
            vector<vector<double>> a(trg_len, vector<double>(2 * src_len, 0.0));
            
            // calculate alpha
            {
                // initial
                double sum = 0.0;
                double initial_prob = 1.0 / (2.0 * src_len);
                for (int is : irange(0, src_len)) {
                    double delta = initial_prob * pt[trg_sentence[0]][src_sentence[is]];
                    a[0][is] = delta;
                    sum += delta;
                    delta = initial_prob * pt[trg_sentence[0]][src_null_id];
                    a[0][is + src_len] = delta;
                    sum += delta;
                }
                scale.at(0) = 1.0 / sum;
                for (int is : irange(0, src_len)) {
                    a[0][is] *= scale[0];
                    a[0][is + src_len] *= scale[0];
                }
            }
            for (int it : irange(1, trg_len)) {
                // remaining
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    double pt_it_is = pt[trg_sentence[it]][src_sentence[is]];
                    double delta = 0.0;
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        delta += (a[it - 1][is2] + a[it - 1][is2 + src_len]) * pj[is][is2] * pt_it_is;
                    }
                    a[it][is] = delta;
                    sum += delta;
                    delta = (a[it - 1][is] + a[it - 1][is + src_len]) * pj_null[is] * pt[trg_sentence[it]][src_null_id];
                    a[it][is + src_len] = delta;
                    sum += delta;
                }
                double scale_it = 1.0 / sum;
                scale[it] = scale_it;
                for (int is : irange(0, src_len)) {
                    a[it][is] *= scale_it;
                    a[it][is + src_len] *= scale_it;
                }
            }

            // calculate log likelihood
            for (int it : irange(0, trg_len)) {
                log_likelihood -= log(scale[it]);
            }
            
            // beta (backward) scaled prob.
            // b[it][is]
            vector<vector<double>> b(trg_len, vector<double>(2 * src_len, 0.0));

            // calculate beta (backward)
            {
                // final
                for (int is : irange(0, 2 * src_len)) {
                    b[trg_len - 1][is] = scale[trg_len - 1];
                }
            }
            for (int it : irange(0, trg_len - 1) | reversed) {
                // remaining
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
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

            // calculate jumping counts
            // xi[is][is2] = Pr( (it-1, is2) -> (it, is) ) * sum
            vector<vector<double>> xi(2 * src_len, vector<double>(2 * src_len, 0.0));

            for (int it : irange(1, trg_len)) {
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    {
                        double pt_and_b = pt[trg_sentence[it]][src_sentence[is]] * b[it][is];
                        for (int is2 : irange(is_min[is], is_max[is])) {
                            double pj_and_pt_and_b = pj[is][is2] * pt_and_b;
                            double delta = a[it - 1][is2] * pj_and_pt_and_b;
                            xi[is][is2] = delta;
                            sum += delta;
                            delta = a[it - 1][is2 + src_len] * pj_and_pt_and_b;
                            xi[is][is2 + src_len] = delta;
                            sum += delta;
                        }
                    }
                    {
                        double pj_and_pt_and_b = pj_null[is] * pt[trg_sentence[it]][src_null_id] * b[it][is + src_len];
                        double delta = a[it - 1][is] * pj_and_pt_and_b;
                        xi[is + src_len][is] = delta;
                        sum += delta;
                        delta = a[it - 1][is + src_len] * pj_and_pt_and_b;
                        xi[is + src_len][is + src_len] = delta;
                        sum += delta;
                    }
                }
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        cj[is - is2 + distance_limit] += xi[is][is2] / sum;
                        cj[is - is2 + distance_limit] += xi[is][is2 + src_len] / sum;
                    }
                    cj_null += xi[is + src_len][is] / sum;
                    cj_null += xi[is + src_len][is + src_len] / sum;
                }
            }

            // calculate translation counts
            // gamma[is] = Pr( (it, is) ) * sum
            vector<double> gamma(2 * src_len, 0.0);

            for (int it : irange(0, trg_len)) {
                double sum = 0;
                for (int is : irange(0, src_len)) {
                    double delta = a[it][is] * b[it][is];
                    gamma[is] = delta;
                    sum += delta;
                    delta = a[it][is + src_len] * b[it][is + src_len];
                    gamma[is + src_len] = delta;
                    sum += delta;
                }
                for (int is : irange(0, src_len)) {
                    double delta = gamma[is] / sum;
                    ct[trg_sentence[it]][src_sentence[is]] += delta;
                    sumct[src_sentence[is]] += delta;
                    delta = gamma[is + src_len] / sum;
                    ct[trg_sentence[it]][src_null_id] += delta;
                    sumct[src_null_id] += delta;
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

    return HmmModel { std::move(pt), std::move(fj), fj_null, distance_limit };
}

vector<pair<int, int>> Aligner::generateIbmModel1ViterbiAlignment(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const vector<vector<double>> & translation_prob,
    int src_num_vocab,
    int src_null_id) {

    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    vector<pair<int, int>> align;
    int src_len = src_sentence.size();
    int trg_len = trg_sentence.size();

    for (int it : irange(0, trg_len)) {
        int t = trg_sentence[it];
        int max_is = -1;
        double max_prob = -1.0;
        for (int is : irange(0, src_len)) {
            double prob = translation_prob[t][src_sentence[is]];
            if (prob > max_prob) {
                max_is = is;
                max_prob = prob;
            }
        }
        if (max_prob > translation_prob[t][src_null_id]) {
            align.push_back(pair<int, int>(max_is, it));
        }
    }

    return align;
}

vector<pair<int, int>> Aligner::generateHmmViterbiAlignment(
    const vector<int> & src_sentence,
    const vector<int> & trg_sentence,
    const HmmModel & hmm_model,
    int src_num_vocab,
    int src_null_id) {
    
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    int src_len = src_sentence.size();
    int trg_len = trg_sentence.size();

    vector<pair<int, int>> align;

    // TODO
    
    return align;
}

tuple<vector<int>, vector<int>> Aligner::calculateHmmJumpingRange(
    int src_len,
    int distance_limit) {

    vector<int> is_min(src_len);
    vector<int> is_max(src_len);

    for (int is : irange(0, src_len)) {
        is_min[is] = is > distance_limit ? is - distance_limit : 0;
        is_max[is] = is < src_len - distance_limit ? is + distance_limit + 1 : src_len;
    }

    return make_tuple(std::move(is_min), std::move(is_max));
}

tuple<vector<vector<double>>, vector<double>> Aligner::calculateHmmJumpingProbability(
    const std::vector<double> & jumping_factor,
    double null_jumping_factor,
    int src_len,
    int distance_limit,
    const std::vector<int> min_jumping_range,
    const std::vector<int> max_jumping_range) {

    vector<vector<double>> pj(src_len, vector<double>(src_len, 0.0));
    vector<double> pj_null(src_len, 0.0);

    for (int is2 : irange(0, src_len)) {
        double sum = 0.0;

        for (int is : irange(min_jumping_range[is2], max_jumping_range[is2])) {
            sum += jumping_factor[is - is2 + distance_limit];
        }
        sum += null_jumping_factor;

        for (int is : irange(min_jumping_range[is2], max_jumping_range[is2])) {
            pj[is][is2] = jumping_factor[is - is2 + distance_limit] / sum;
        }
        pj_null[is2] = null_jumping_factor / sum;
    }

    return make_tuple(std::move(pj), std::move(pj_null));
}

} // namespace TreeAligner

