#include <treealigner/assertion.h>
#include <treealigner/Aligner.h>
#include <treealigner/Dictionary.h>
#include <treealigner/Tracer.h>
#include <treealigner/Utility.h>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace TreeAligner;
using boost::format;
using boost::irange;

namespace PO = boost::program_options;

PO::variables_map parseOptions(int argc, char * argv[]) {
    string description = "TreeAligner by Yusuke ODA";
    string binname = "aligner";

    // generic options
    PO::options_description opt_generic("Generic Options");
    opt_generic.add_options()
        ("help", "print this manual and exit")
        ("trace-level", PO::value<int>()->default_value(0), "tracing detail")
        ;
    // input/output
    PO::options_description opt_io("I/O Options");
    opt_io.add_options()
        //("src-tok", PO::value<string>(), "path to tokens of source language")
        //("trg-tok", PO::value<string>(), "path to tokens of target language")
        //("src-class", PO::value<string>(), "path to word classes of source language")
        //("trg-class", PO::value<string>(), "path to word classess of target language")
        ("src-tree", PO::value<string>(), "path to parses of source language")
        ("trg-tree", PO::value<string>(), "path to parses of target language")
        //("output", PO::value<string>(), "path to output file")
        ;
    // configuration
    PO::options_description opt_config("Configurations");
    opt_config.add_options()
        ("method", PO::value<string>(), "alignment strategy\ncandidates: model1, hmm")
        ("unknown-threshold", PO::value<int>()->default_value(5), "maximum frequency to assume the word is unknown")
        ("model1-iteration", PO::value<int>()->default_value(10), "number of iterations for IBM model 1 training")
        ("hmm-iteration", PO::value<int>()->default_value(10), "number of iterations for HMM model training")
        ("hmm-distance-limit", PO::value<int>()->default_value(10), "maximum distance to connect HMM nodes")
        ;
    
    PO::options_description opt;
    opt.add(opt_generic).add(opt_io).add(opt_config);

    // parse
    PO::variables_map args;
    PO::store(PO::parse_command_line(argc, argv, opt), args);
    PO::notify(args);

    // process usage
    if (args.count("help")) {
        cerr << description << endl;
        cerr << "Usgae: " << binname << " --method <str> --src-tree <path> --trg-tree <path> [options]" << endl;
        cerr << opt << endl;
        exit(1);
    }

    // check required options
    if (!args.count("src-tree") || !args.count("trg-tree") || !args.count("method")) {
        cerr << "ERROR: insufficient required options" << endl;
        cerr << "(--help to show usage)" << endl;
        exit(1);
    }

    return move(args);
}

// initialize tracer settings
void initializeTracer(const PO::variables_map & args) {
    int level = args["trace-level"].as<int>();
    MYASSERT(::initializeTracer, level >= 0);
    Tracer::setTraceLevel(level);
}

// generate IBM model 1 Viterbi alignment
void processModel1(
    const vector<vector<int>> & src_sentence_list,
    const vector<vector<int>> & trg_sentence_list,
    int src_num_words,
    int trg_num_words,
    int src_null_id,
    const PO::variables_map & args) {

    auto model1_translation_prob = Aligner::trainIbmModel1(
        src_sentence_list,
        trg_sentence_list,
        src_num_words,
        trg_num_words,
        src_null_id,
        args["model1-iteration"].as<int>());

    Tracer::println(0, "Generating IBM model 1 Viterbi alignment ...");

    for (size_t k : irange(0UL, src_sentence_list.size())) {
        auto align = Aligner::generateIbmModel1ViterbiAlignment(
            src_sentence_list[k],
            trg_sentence_list[k],
            model1_translation_prob,
            src_num_words,
            src_null_id);

        for (size_t ia : irange(0UL, align.size())) {
            if (ia > 0) cout << ' ';
            cout << align[ia].first << '-' << align[ia].second;
        }
        cout << endl;
    }
}

// generate HMM model Viterbi alignment
void processHmm(
    const vector<vector<int>> & src_sentence_list,
    const vector<vector<int>> & trg_sentence_list,
    int src_num_words,
    int trg_num_words,
    int src_null_id,
    const PO::variables_map & args) {

    auto model1_translation_prob = Aligner::trainIbmModel1(
        src_sentence_list,
        trg_sentence_list,
        src_num_words,
        trg_num_words,
        src_null_id,
        args["model1-iteration"].as<int>());

    auto hmm_model = Aligner::trainHmmModel(
        src_sentence_list,
        trg_sentence_list,
        model1_translation_prob,
        src_num_words,
        trg_num_words,
        src_null_id,
        args["hmm-iteration"].as<int>(),
        args["hmm-distance-limit"].as<int>());

    Tracer::println(0, "Generating HMM Viterbi alignment ...");

    for (size_t k : irange(0UL, src_sentence_list.size())) {
        auto align = Aligner::generateHmmViterbiAlignment(
            src_sentence_list[k],
            trg_sentence_list[k],
            hmm_model,
            src_num_words,
            src_null_id);

        for (size_t ia : irange(0UL, align.size())) {
            if (ia > 0) cout << ' ';
            cout << align[ia].first << '-' << align[ia].second;
        }
        cout << endl;
    }

}

int main(int argc, char * argv[]) {
    auto args = parseOptions(argc, argv);

    initializeTracer(args);

    const string NULL_WORD = "(NULL)";
    const string UNKNOWN_WORD = "(UNKNOWN)";

    const int unknown_threshold = args["unknown-threshold"].as<int>();

    unique_ptr<ifstream> ifs_src_tree = Utility::openInputStream(args["src-tree"].as<string>());
    unique_ptr<ifstream> ifs_trg_tree = Utility::openInputStream(args["trg-tree"].as<string>());

    Dictionary src_tag_dict, trg_tag_dict;
    Dictionary src_word_dict, trg_word_dict;
    vector<Tree<int>> src_tree_list, trg_tree_list;
    vector<vector<int>> src_sentence_list, trg_sentence_list;

    // add nul/unknown identifier
    src_word_dict.getId(NULL_WORD);
    trg_word_dict.getId(NULL_WORD);
    src_word_dict.getId(UNKNOWN_WORD);
    trg_word_dict.getId(UNKNOWN_WORD);
    const int NULL_ID = 0;
    const int UNKNOWN_ID = 1;

    // load trees and extract words
    Tracer::println(0, "Loading data ..");
    
    string src, trg;
    int num_data = 0;

    while (getline(*ifs_src_tree, src) && getline(*ifs_trg_tree, trg)) {
        auto src_tree = Utility::parseTree(src, src_tag_dict, src_word_dict);
        auto trg_tree = Utility::parseTree(trg, trg_tag_dict, trg_word_dict);
        auto src_sentence = Utility::extractWords(src_tree);
        auto trg_sentence = Utility::extractWords(trg_tree);

        src_tree_list.push_back(std::move(src_tree));
        trg_tree_list.push_back(std::move(trg_tree));
        src_sentence_list.push_back(std::move(src_sentence));
        trg_sentence_list.push_back(std::move(trg_sentence));

        ++num_data;
        if (num_data % 100000 == 0) {
            Tracer::println(1, format("%d sentences loaded") % num_data);
        }
    }
    
    Tracer::println(1, format("%d sentences loaded") % num_data);
    Tracer::println(1, format("#src grammar tags: %d") % src_tag_dict.size());
    Tracer::println(1, format("#trg grammar tags: %d") % trg_tag_dict.size());
    Tracer::println(1, format("#src vocaburaly: %d") % src_word_dict.size());
    Tracer::println(1, format("#trg vocaburaly: %d") % trg_word_dict.size());

    // count words and replace rare words with unknown word
    Tracer::println(0, "Reducing vocabularies ...");

    vector<int> src_word_freq(src_word_dict.size(), 0);
    vector<int> trg_word_freq(trg_word_dict.size(), 0);

    for (auto & src_sentence : src_sentence_list) {
        for (int w : src_sentence) {
            ++src_word_freq[w];
        }
    }
    for (auto & trg_sentence : trg_sentence_list) {
        for (int w : trg_sentence) {
            ++trg_word_freq[w];
        }
    }

    vector<int> src_word_map(src_word_dict.size(), UNKNOWN_ID);
    vector<int> trg_word_map(trg_word_dict.size(), UNKNOWN_ID);
    src_word_map[NULL_ID] = NULL_ID;
    trg_word_map[NULL_ID] = NULL_ID;

    int src_num_reduced_words = 2;
    for (size_t i : irange(2UL, src_word_dict.size())) {
        if (src_word_freq[i] > unknown_threshold) src_word_map[i] = src_num_reduced_words++;
    }

    int trg_num_reduced_words = 2;
    for (size_t i : irange(2UL, trg_word_dict.size())) {
        if (trg_word_freq[i] > unknown_threshold) trg_word_map[i] = trg_num_reduced_words++;
    }

    for (auto & sent : src_sentence_list) {
        for (size_t i : irange(0UL, sent.size())) {
            sent[i] = src_word_map[sent[i]];
        }
    }
    for (auto & sent : trg_sentence_list) {
        for (size_t i : irange(0UL, sent.size())) {
            sent[i] = trg_word_map[sent[i]];
        }
    }

    function<void(Tree<int> &, const vector<int> &)> replaceLeaves
        = [&replaceLeaves](Tree<int> & node, const vector<int> & mapping) -> void {
        if (!node.size()) {
            node.setLabel(mapping[node.label()]);
        } else {
            for (auto & ch : node) {
                replaceLeaves(ch, mapping);
            }
        }
    };

    for (auto & tree : src_tree_list) {
        replaceLeaves(tree, src_word_map);
    }
    for (auto & tree : trg_tree_list) {
        replaceLeaves(tree, trg_word_map);
    }
    
    Tracer::println(1, format("#src reduced vocaburaly: %d") % src_num_reduced_words);
    Tracer::println(1, format("#trg reduced vocaburaly: %d") % trg_num_reduced_words);

    const string method = args["method"].as<string>();
    if (method == "model1") {
        ::processModel1(
            src_sentence_list,
            trg_sentence_list,
            src_num_reduced_words,
            trg_num_reduced_words,
            NULL_ID,
            args);
    } else if (method == "hmm") {
        ::processHmm(
            src_sentence_list,
            trg_sentence_list,
            src_num_reduced_words,
            trg_num_reduced_words,
            NULL_ID,
            args);
    } else {
        throw runtime_error("main: unknown alignment strategy: " + method);
    }

    return 0;
}

