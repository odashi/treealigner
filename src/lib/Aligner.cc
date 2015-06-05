#include <aligner/Aligner.h>
#include <aligner/assertion.h>

#include <boost/format.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/irange.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using boost::adaptors::reversed;
using boost::format;
using boost::irange;

namespace Aligner {

vector<vector<double>> Aligner::calculateIbmModel1(
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
    int src_num_vocab,
    int trg_num_vocab,
    int num_iteration,
    int src_null_id) {

    cerr << "Calculating IBM Model 1 ..." << endl;

    // check constraints
    MYASSERT(Aligner::calculateIbmModel1, src_corpus.size() == trg_corpus.size());
    MYASSERT(Aligner::calculateIbmModel1, src_num_vocab > 0);
    MYASSERT(Aligner::calculateIbmModel1, trg_num_vocab > 0);
    MYASSERT(Aligner::calculateIbmModel1, num_iteration >= 0);
    MYASSERT(Aligner::calculateIbmModel1, src_null_id >= 0);
    MYASSERT(Aligner::calculateIbmModel1, src_null_id < src_num_vocab);

    int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    vector<vector<double>> pt(trg_num_vocab, vector<double>(src_num_vocab, 1.0 / (trg_num_vocab - 1)));
    
    for (int iteration : irange(0, num_iteration)) {

        cerr << "  Iteration " << (iteration + 1) << endl;
        
        // probabilistic counts
        // c[t][s] = count(t|s)
        vector<vector<double>> c(trg_num_vocab, vector<double>(src_num_vocab, 0.0));
        // sumc[s] = sum_t count(t|s)
        vector<double> sumc(src_num_vocab, 0.0);

        // previous entropy
        double entropy = 0.0;

        for (int k : irange(0, num_sentences)) {
            auto & src_sentence = src_corpus[k];
            auto & trg_sentence = trg_corpus[k];
            
            // sum of prob. for each target word
            // sumpt[t] = sum_s Pt(t|s)
            vector<double> sumpt(trg_num_vocab, 0.0);

            // previous perplexity of the text
            double ppl = 0.0;

            //calculate sumpt[t] and ppl
            for (int t : trg_sentence) {
                // inner words
                for (int s : src_sentence) {
                    double delta = pt[t][s];
                    sumpt[t] += delta;
                    ppl += delta;
                }
                // null word
                double delta = pt[t][src_null_id];
                sumpt[t] += delta;
                ppl += delta;
            }

            entropy -= log(ppl) - trg_sentence.size() * log(src_sentence.size());

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
                pt[t][s] = c[t][s] / sumc[s];
            }
        }

        cerr << (format("    H = %.10e") % entropy) << endl;
    }

    return pt;
}

void Aligner::calculateHmmModel(
    const vector<vector<int>> & src_corpus,
    const vector<vector<int>> & trg_corpus,
    const vector<vector<double>> & prior_translation_prob,
    int src_num_vocab,
    int trg_num_vocab,
    int num_iteration,
    int src_null_id,
    int distance_limit) {

    cerr << "Calculating HMM model ..." << endl;

    // check constraints
    MYASSERT(Aligner::calculateHmmModel, src_corpus.size() == trg_corpus.size());
    MYASSERT(Aligner::calculateHmmModel, src_num_vocab > 0);
    MYASSERT(Aligner::calculateHmmModel, trg_num_vocab > 0);
    MYASSERT(Aligner::calculateHmmModel, num_iteration >= 0);
    MYASSERT(Aligner::calculateHmmModel, src_null_id >= 0);
    MYASSERT(Aligner::calculateHmmModel, src_null_id < src_num_vocab);
    MYASSERT(Aligner::calculateHmmModel, distance_limit >= 0);

    int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    vector<vector<double>> pt = prior_translation_prob;
    
    // jumping (transition) factor
    // fj[d + distance_limit] = Fj(d) for x = [-distance_limit, distance_limit]
    vector<double> fj(2 * distance_limit + 1, 1.0);

    for (int iteration : irange(0, num_iteration)) {
        
        cerr << "  Iteration " << (iteration + 1) << endl;

        // probabilistic counts
        // ct[t][s] = count(t|s)
        vector<vector<double>> ct(trg_num_vocab, vector<double>(src_num_vocab, 0.0));
        // sumct[s] = sum_t count(t|s)
        vector<double> sumct(src_num_vocab, 0.0);
        // cj[d + distance_limit] = count(d)
        vector<double> cj(2 * distance_limit + 1, 0.0);

        // entropy
        double entropy = 0.0;

        for (int k : irange(0, num_sentences)) {
            auto & src_sentence = src_corpus.at(k);
            auto & trg_sentence = trg_corpus.at(k);
            int src_len = src_sentence.size();
            int trg_len = trg_sentence.size();

            // ranges of possible path connections
            vector<int> is_min(src_len);
            vector<int> is_max(src_len);
            for (int is : irange(0, src_len)) {
                is_min[is] = is > distance_limit ? is - distance_limit : 0;
                is_max[is] = is < src_len - distance_limit ? is + distance_limit + 1 : src_len;
            }

            // calculate jumping prob.
            // pj[is][is'] = Pj(is' -> is) = Fj(is - is') / sum[ Fj(j - is') for j = [0, src_len) ]
            vector<vector<double>> pj(src_len, vector<double>(src_len, 0.0));
            for (int is2 : irange(0, src_len)) {
                double sum = 0.0;
                for (int is : irange(is_min[is2], is_max[is2])) {
                    sum += fj[is - is2 + distance_limit];
                }
                for (int is : irange(is_min[is2], is_max[is2])) {
                    pj[is][is2] = fj[is - is2 + distance_limit] / sum;
                }
            }

            // scaling factor
            // scale[it]
            vector<double> scale(trg_len);

            // alpha (forward) scaled prob.
            // a[it][is]
            vector<vector<double>> a(trg_len, vector<double>(src_len, 0.0));
            
            // calculate alpha
            {
                // initial
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    double delta = (1.0 / src_len) * pt[trg_sentence[0]][src_sentence[is]];
                    a[0][is] = delta;
                    sum += delta;
                }
                scale.at(0) = 1.0 / sum;
                for (int is : irange(0, src_len)) {
                    a[0][is] *= scale[0];
                }
            }
            for (int it : irange(1, trg_len)) {
                // remaining
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        a[it][is] += a[it - 1][is2] * pj[is][is2] * pt[trg_sentence[it]][src_sentence[is]];
                    }
                    sum += a.at(it).at(is);
                }
                scale[it] = 1.0 / sum;
                for (int is : irange(0, src_len)) {
                    a[it][is] *= scale[it];
                }
            }

            // calculate entropy
            for (int it : irange(0, trg_len)) {
                entropy += log(scale[it]);
            }
            
            // beta (backward) scaled prob.
            // b[it][is]
            vector<vector<double>> b(trg_len, vector<double>(src_len, 0.0));

            // calculate beta (backward)
            {
                // final
                for (int is : irange(0, src_len)) {
                    b[trg_len - 1][is] = scale[trg_len - 1];
                }
            }
            for (int it : irange(0, trg_len - 1) | reversed) {
                // remaining
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        b[it][is] += b[it + 1][is2] * pj[is2][is] * pt[trg_sentence[it + 1]][src_sentence[is2]];
                    }
                    b.at(it).at(is) *= scale.at(it);
                }
            }

            // calculate jumping counts
            // xi[is][is2] = Pr( (it-1, is2) -> (it, is) ) * sum
            vector<vector<double>> xi(src_len, vector<double>(src_len, 0.0));

            for (int it : irange(1, trg_len)) {
                double sum = 0.0;
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        double delta = a[it - 1][is2] * pj[is][is2] * pt[trg_sentence[it]][src_sentence[is]] * b[it][is];
                        xi[is][is2] = delta;
                        sum += delta;
                    }
                }
                
                for (int is : irange(0, src_len)) {
                    for (int is2 : irange(is_min[is], is_max[is])) {
                        cj[is - is2 + distance_limit] += xi[is][is2] / sum;
                        //cerr << (format("%2d--%2d: %2d--%2d: %.4f") % (it-1) % it % is2 % is % xi[it][is][is2]) << endl;
                    }
                }
            }

            // calculate translation counts
            // gamma[is] = Pr( (it, is) ) * sum
            vector<double> gamma(src_len, 0.0);

            for (int it : irange(0, trg_len)) {
                double sum = 0;
                for (int is : irange(0, src_len)) {
                    double delta = a[it][is] * b[it][is];
                    gamma[is] = delta;
                    sum += delta;
                }
                for (int is : irange(0, src_len)) {
                    double delta = gamma[is] / sum;
                    ct[trg_sentence[it]][src_sentence[is]] += delta;
                    sumct[src_sentence[is]] += delta;
                    //cerr << (format("%2d,%2d: %.4f") % it % is % gamma[it][is]) << endl;
                }
            }
        }

        /*
        for (int i : irange(-distance_limit, distance_limit + 1)) {
            cerr << (format("%d: %f") % i % cj[i + distance_limit]) << endl;
        }
        */

        // set new jumping factors
        fj = cj;

        // calculate pt[t][s]
        for (int t : irange(0, trg_num_vocab)) {
            for (int s : irange(0, src_num_vocab)) {
                pt[t][s] = ct[t][s] / sumct[s];
            }
        }

        cerr << (format("    H = %.20e") % entropy) << endl;
    }
}

} // namespace Aligner

