#include <treealigner/Aligner.h>
#include <treealigner/Hmm.h>
#include <treealigner/Tracer.h>
#include <treealigner/Utility.h>
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
            const auto & src_sent = src_corpus[k];
            const auto & trg_sent = trg_corpus[k];
            
            // sum of prob. for each target word
            // sumpt[t] = sum_s Pt(t|s)
            auto sumpt = make_tensor1<double>(trg_num_vocab, 0.0);

            double likelihood = 0.0;

            //calculate sumpt[t] and ppl
            for (int t : trg_sent) {
                // inner words
                for (int s : src_sent) {
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

            log_likelihood += log(likelihood) - trg_sent.size() * log(src_sent.size() + 1);

            // calculate c[t][s] and sumc[s]
            for (int t : trg_sent) {
                // inner words
                for (int s : src_sent) {
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
        auto ct = make_tensor2<double>(trg_num_vocab, src_num_vocab, 0.0);
        auto sumct = make_tensor1<double>(src_num_vocab, 0.0);
        auto cj = make_ranged_tensor1<double>(-distance_limit, distance_limit, 0.0);
        double cj_null = 0.0;

        double log_likelihood = 0.0;

        for (int k : irange(0, num_sentences)) {
            const auto & src_sent = src_corpus[k];
            const auto & trg_sent = trg_corpus[k];
            const int src_len = src_sent.size();
            const int trg_len = trg_sent.size();

            const auto range = Hmm::getLimitedJumpingRange(src_len, distance_limit);

            Tensor2<double> pj;
            Tensor1<double> pj_null;
            tie(pj, pj_null) = Hmm::getJumpingProbability(fj, fj_null, src_len, range);

            Tensor2<double> a;
            Tensor1<double> scale;
            tie(a, scale) = Hmm::forwardStep(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range);

            // calculate likelihood
            for (int it : irange(0, trg_len)) {
                log_likelihood -= log(scale[it]);
            }
            
            auto b = Hmm::backwardStep(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range, scale);
            auto xi = Hmm::getEdgeProbability(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range, a, b);
            auto gamma = Hmm::getNodeProbability(src_sent, trg_sent, a, b, scale);

            // update jumping counts
            for (int it : irange(1, trg_len)) {
                const auto & xi_it = xi[it];
                for (int is_dst : irange(0, src_len)) {
                    for (int is_org : irange(range.min[is_dst], range.max[is_dst])) {
                        cj[is_dst - is_org] += xi_it[is_dst][is_org];
                        cj[is_dst - is_org] += xi_it[is_dst][is_org + src_len];
                    }
                    cj_null += xi_it[is_dst + src_len][is_dst];
                    cj_null += xi_it[is_dst + src_len][is_dst + src_len];
                }
            }

            // update generation counts
            for (int it : irange(0, trg_len)) {
                const auto & gamma_it = gamma[it];
                auto & ct_trg = ct[trg_sent[it]];
                for (int is : irange(0, src_len)) {
                    ct_trg[src_sent[is]] += gamma_it[is];
                    sumct[src_sent[is]] += gamma_it[is];
                    ct_trg[src_null_id] += gamma_it[is + src_len];
                    sumct[src_null_id] += gamma_it[is + src_len];
                }
            }
        }

        // set new jumping factors
        fj = std::move(cj);
        fj_null = std::move(cj_null);

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
    auto & fj_stop = model.stop_factor;
    auto & fj_move = model.move_factor;
    auto & fj_push = model.push_factor;
    auto & fj_leave = model.leave_factor;
    auto & fj_stay = model.stay_factor;
    auto & fj_null = model.null_factor;

    for (int iteration : irange(0, num_iteration)) {
        
        Tracer::println(1, format("Iteration %d") % (iteration + 1));

        // probabilistic counts
        auto ct = make_tensor2<double>(trg_num_vocab, src_num_vocab, 0.0);
        auto sumct = make_tensor1<double>(src_num_vocab, 0.0);
        auto cj_pop = make_tensor1<double>(src_num_tags, 0.0);
        auto cj_stop = make_tensor1<double>(src_num_tags, 0.0);
        auto cj_move = make_tensor1(src_num_tags, make_ranged_tensor1<double>(-move_limit, move_limit, 0.0));
        auto cj_push = make_tensor1(src_num_tags, make_ranged_tensor1<double>(-move_limit, move_limit, 0.0));
        double cj_leave = 0.0;
        double cj_stay = 0.0;
        double cj_null = 0.0;
        
        auto pj_table = calculateTreeTraversalProbability(model);

        double log_likelihood = 0.0;
        
        for (int k : irange(0, num_sentences)) {
            const auto src_sent = Utility::extractWords(src_corpus[k]);
            const auto & trg_sent = trg_corpus[k];
            const int src_len = src_sent.size();
            const int trg_len = trg_sent.size();
            
            const auto topdown_paths = calculateTopDownPaths(src_corpus[k]);
            const auto treehmm_paths = calculateTreeHmmPaths(topdown_paths, move_limit, push_limit);

            const auto range = Hmm::getFlatJumpingRange(src_len);
            
            Tensor2<double> pj;
            Tensor1<double> pj_null;
            tie(pj, pj_null) = calculateTreeHmmJumpingProbability(model, pj_table, treehmm_paths);

            Tensor2<double> a;
            Tensor1<double> scale;
            tie(a, scale) = Hmm::forwardStep(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range);

            // calculate likelihood
            for (int it : irange(0, trg_len)) {
                log_likelihood -= log(scale[it]);
            }
            
            auto b = Hmm::backwardStep(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range, scale);
            auto xi = Hmm::getEdgeProbability(src_sent, trg_sent, pt, pj, pj_null, src_null_id, range, a, b);
            auto gamma = Hmm::getNodeProbability(src_sent, trg_sent, a, b, scale);

            // update jumping counts
            for (int it : irange(1, trg_len)) {
                const auto & xi_it = xi[it];
                for (int is_dst : irange(0, src_len)) {
                    for (int is_org : irange(range.min[is_dst], range.max[is_dst])) {
                        if (is_dst == is_org) {
                            // stay factor
                            cj_stay += xi_it[is_dst][is_org];
                            cj_stay += xi_it[is_dst][is_org + src_len];
                        } else {
                            // leave factor
                            double delta = xi_it[is_dst][is_org] + xi_it[is_dst][is_org + src_len];
                            cj_leave += delta;
                            // tree traversal factors
                            for (const auto & node : treehmm_paths[is_dst][is_org]) {
                                if (node.skip) continue;
                                switch (node.op) {
                                case TreeHmmPath::POP:
                                    cj_pop[node.label] += delta;
                                    break;
                                case TreeHmmPath::STOP:
                                    cj_stop[node.label] += delta;
                                    break;
                                case TreeHmmPath::MOVE:
                                    cj_move[node.label][node.distance] += delta;
                                    break;
                                case TreeHmmPath::PUSH:
                                    cj_push[node.label][node.distance] += delta;
                                    break;
                                }
                            }
                        }
                    }
                    // null factor
                    cj_null += xi_it[is_dst + src_len][is_dst];
                    cj_null += xi_it[is_dst + src_len][is_dst + src_len];
                }
            }

            // update generation counts
            for (int it : irange(0, trg_len)) {
                const auto & gamma_it = gamma[it];
                auto & ct_trg = ct[trg_sent[it]];
                for (int is : irange(0, src_len)) {
                    ct_trg[src_sent[is]] += gamma_it[is];
                    sumct[src_sent[is]] += gamma_it[is];
                    ct_trg[src_null_id] += gamma_it[is + src_len];
                    sumct[src_null_id] += gamma_it[is + src_len];
                }
            }
        }

        // set new jumping factors
        fj_pop = std::move(cj_pop);
        fj_stop = std::move(cj_stop);
        fj_move = std::move(cj_move);
        fj_push = std::move(cj_push);
        fj_leave = std::move(cj_leave);
        fj_stay = std::move(cj_stay);
        fj_null = std::move(cj_null);

        // calculate pt[t][s]
        for (int t : irange(0, trg_num_vocab)) {
            for (int s : irange(0, src_num_vocab)) {
                pt[t][s] = (sumct[s] > 0.0) ? ct[t][s] / sumct[s] : 0.0;
            }
        }

        Tracer::println(2, format("LL = %.10e") % log_likelihood);
    }

    return TreeHmmModel {};
}

vector<Alignment> Aligner::generateIbmModel1ViterbiAlignment(
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const Tensor2<double> & translation_prob,
    const int src_num_vocab,
    const int src_null_id) {

    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    vector<Alignment> align;
    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    for (int it : irange(0, trg_len)) {
        const int t = trg_sent[it];
        int max_is = -1;
        double max_prob = -1.0;
        for (int is : irange(0, src_len)) {
            const double prob = translation_prob[t][src_sent[is]];
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
    const Sentence<int> & src_sent,
    const Sentence<int> & trg_sent,
    const HmmModel & model,
    const int src_num_vocab,
    const int src_null_id) {
    
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id >= 0);
    MYASSERT(TreeAligner::Aligner::generateIbmModel1ViterbiAlignment, src_null_id < src_num_vocab);
    
    const int src_len = src_sent.size();
    const int trg_len = trg_sent.size();

    // aliases
    const auto & pt = model.generation_prob;
    const auto & fj = model.jumping_factor;
    const double fj_null = model.null_jumping_factor;

    const auto range = Hmm::getLimitedJumpingRange(src_len, model.distance_limit);

    // calculate jumping prob.
    // pj[is][is'] = Pj(is' -> is) = Fj(is - is') / sum[ Fj(j - is') for j = [0, src_len) ]
    Tensor2<double> pj;
    Tensor1<double> pj_null;
    tie(pj, pj_null) = Hmm::getJumpingProbability(fj, fj_null, src_len, range);

    // scaling factor
    // scale[it]
    auto scale = make_tensor1<double>(trg_len);

    // scaled Viterbi score
    // viterbi[it][is]
    auto viterbi = make_tensor2<double>(trg_len, 2 * src_len, -1.0);
    
    // previous position
    // prev[it][is]
    auto prev = make_tensor2<int>(trg_len, 2 * src_len);

    // forward step
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

TreeTraversalProbability Aligner::calculateTreeTraversalProbability(
    const TreeHmmModel & model) {

    // aliases
    const int ml = model.move_limit;
    const int pl = model.push_limit;
    const int num_tags = model.pop_factor.size();

    TreeTraversalProbability pj_table {
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
                if (max - min < 2) continue;
                double sum = 0.0;
                for (int pos : irange(min, max + 1)) {
                    if (pos == 0) continue;
                    sum += model.move_factor[tag][pos];
                }
                for (int pos : irange(min, max + 1)) {
                    if (pos == 0) continue;
                    pj_table.move_prob[tag][min][max][pos] = model.move_factor[tag][pos] / sum;
                }
            }
        }
    }

    // push prob.
    for (int tag : irange(0, num_tags)) {
        for (int degree : irange(2, 2 * pl + 1)) {
            int min = -(degree / 2);
            int max = (degree - 1) / 2;
            double sum = 0;
            for (int pos : irange(min, max + 1)) {
                sum += model.push_factor[tag][pos];
            }
            for (int pos : irange(min, max + 1)) {
                pj_table.push_prob[tag][min][max][pos] = model.push_factor[tag][pos] / sum;
            }
        }
    }

    /*
    cout << format("leave: %.4f") % pj_table.leave_prob << endl;
    cout << format("stay : %.4f") % pj_table.stay_prob << endl;
    cout << format("null : %.4f") % pj_table.null_prob << endl;

    cout << "pop, stop:" << endl;
    for (int tag : irange(0, num_tags)) {
        cout << format("  %2d: %.4f, %.4f") % tag % pj_table.pop_prob[tag] % pj_table.stop_prob[tag] << endl;
    }

    cout << "move:" << endl;
    for (int tag : irange(0, num_tags)) {
        cout << format("  tag=%2d:") % tag << endl;
        for (int min : irange(-ml, ml + 1)) {
            for (int max : irange(-ml, ml + 1)) {
                cout << format("    min=%2d, max=%2d:") % min % max << endl;
                for (int pos : irange(-ml, ml + 1)) {
                    if (pj_table.move_prob[tag][min][max][pos] >= 0.0) {
                        cout << format("      pos=%2d: %.4f") % pos % pj_table.move_prob[tag][min][max][pos] << endl;
                    } else {
                        cout << format("      pos=%2d: NA") % pos << endl;
                    }
                }
            }
        }
    }

    cout << "push:" << endl;
    for (int tag : irange(0, num_tags)) {
        cout << format("  tag=%2d:") % tag << endl;
        for (int min : irange(-pl, pl + 1)) {
            for (int max : irange(-pl, pl + 1)) {
                cout << format("    min=%2d, max=%2d:") % min % max << endl;
                for (int pos : irange(-pl, pl + 1)) {
                    if (pj_table.push_prob[tag][min][max][pos] >= 0.0) {
                        cout << format("      pos=%2d: %.4f") % pos % pj_table.push_prob[tag][min][max][pos] << endl;
                    } else {
                        cout << format("      pos=%2d: NA") % pos << endl;
                    }
                }
            }
        }
    }
    */

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

    // remove topmost unary chain
    const Tree<int> * root = &tree;
    while (root->size() == 1) root = &(*root)[0];

    recursive(*root);

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

            // POP step
            for (size_t k : irange(top + 1, org_path.size()) | reversed) {
                path.push_back(TreeHmmPath { TreeHmmPath::POP, org_path[k].degree == 1, org_path[k].label, 0, 0, 0 });
            }

            // STOP step
            path.push_back(TreeHmmPath { TreeHmmPath::STOP, top == 0, org_path[top].label, 0, 0, 0 });

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

                path.push_back(TreeHmmPath { TreeHmmPath::MOVE, move_range_max - move_range_min == 1, dst_path[top].label, move_distance, move_range_min, move_range_max });
            }

            // PUSH step
            for (size_t k : irange(top + 1, dst_path.size())) {
                const auto & cur = dst_path[k];
                int push_pos = cur.next < cur.degree - cur.next ? cur.next : cur.next - cur.degree;
                if (push_pos >= push_limit || push_pos < -push_limit) {
                    path.clear();
                    break;
                }

                int push_range_min = -(cur.degree / 2);
                if (push_range_min < -push_limit) push_range_min = -push_limit;
                int push_range_max = (cur.degree - 1) / 2;
                if (push_range_max >= push_limit) push_range_max = push_limit - 1;
                
                path.push_back(TreeHmmPath { TreeHmmPath::PUSH, push_range_min == push_range_max, cur.label, push_pos, push_range_min, push_range_max });
            }
        }
    }

    /*
    vector<string> op_str { "POP", "STOP", "MOVE", "PUSH" };
    for (int org : irange(0, n)) {
        for (int dst : irange(0, n)) {
            cout << format("%d -> %d: ") % org % dst;
            for (auto node : paths[dst][org]) {
                if (node.op == TreeHmmPath::POP || node.op == TreeHmmPath::STOP) {
                    cout << format(node.skip ? "(%s; %d) " : "[%s; %d] ") % op_str[node.op] % node.label;
                } else {
                    cout << format(node.skip ? "(%s; %d, %d/%d:%d) " : "[%s; %d, %d/%d:%d] ") % op_str[node.op] % node.label % node.distance % node.range_min % node.range_max;
                }
            }
            cout << endl;
        }
    }
    */
    
    return paths;
}

tuple<Tensor2<double>, Tensor1<double>> Aligner::calculateTreeHmmJumpingProbability(
    const TreeHmmModel & model,
    const TreeTraversalProbability & traversal_prob,
    const Tensor2<vector<TreeHmmPath>> & paths) {

    const int src_len = paths.size();

    auto pj = make_tensor2<double>(src_len, src_len, 0.0);
    auto pj_null = make_tensor1(src_len, 0.0);

    for (int org : irange(0, src_len)) {
        for (int dst : irange(0, src_len)) {
            if (dst == org) {
                // stay current position
                pj[dst][org] = traversal_prob.stay_prob;
            } else if (paths[dst][org].size() == 0) {
                // unavailable path
                pj[dst][org] = 0.0;
            } else {
                // go to other node
                // calculate the product of traversal prob.
                double & cur_pj = pj[dst][org];
                cur_pj = traversal_prob.leave_prob;
                for (const auto & node : paths[dst][org]) {
                    if (node.skip) continue;
                    switch (node.op) {
                    case TreeHmmPath::POP:
                        cur_pj *= traversal_prob.pop_prob[node.label];
                        break;
                    case TreeHmmPath::STOP:
                        cur_pj *= traversal_prob.stop_prob[node.label];
                        break;
                    case TreeHmmPath::MOVE:
                        cur_pj *= traversal_prob.move_prob[node.label][node.range_min][node.range_max][node.distance];
                        break;
                    case TreeHmmPath::PUSH:
                        cur_pj *= traversal_prob.push_prob[node.label][node.range_min][node.range_max][node.distance];
                        break;
                    }
                }
            }
        }

        // null transition
        pj_null[org] = traversal_prob.null_prob;
    }

    /*
    for (int org : irange(0, src_len)) {
        double sum = 0.0;
        for (int dst : irange(0, src_len)) {
            cout << format("%3d -> %3d: %.4f") % org % dst % pj[dst][org] << endl;
            sum += pj[dst][org];
        }
        cout << format("%3d -> NUL: %.4f") % org % pj_null[org] << endl;
        sum += pj_null[org];
        cout << format("%3d -> ALL: %.4f") % org % sum << endl;
    }
    */

    return make_tuple(std::move(pj), std::move(pj_null));
}

} // namespace TreeAligner

