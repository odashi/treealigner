#include <aligner/Aligner.h>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;
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
    if (src_corpus.size() != trg_corpus.size()) throw runtime_error("Aligner::calculateIbmModel1: sizes of src/trg are different.");
    if (src_num_vocab <= 0) throw runtime_error("Aligner::calculateIbmModel1: src_num_vocab must be greater than 0.");
    if (trg_num_vocab <= 0) throw runtime_error("Aligner::calculateIbmModel1: trg_num_vocab must be greater than 0.");
    if (num_iteration < 0) throw runtime_error("Aligner::calculateIbmModel1: num_iteration must be greater than -1.");

    int num_sentences = src_corpus.size();

    // lexical translation prob.
    // pt[t][s] = Pt(t|s)
    vector<vector<double>> pt(trg_num_vocab, vector<double>(src_num_vocab, 1.0 / (trg_num_vocab - 1)));
    
    for (int iteration : irange(0, num_iteration)) {

        cerr << "Iteration " << (iteration + 1) << endl;
        
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

        cerr << (format("  H = %.10e") % entropy) << endl;
    }

    return pt;
}

} // namespace Aligner

