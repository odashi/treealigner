#include <aligner/utils.h>
#include <aligner/Aligner.h>
#include <aligner/Dictionary.h>

#include <boost/range/irange.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <iostream>

using namespace std;
using namespace Aligner;

namespace PO = boost::program_options;

PO::variables_map parseOptions(int argc, char * argv[]) {
    string description = "AHC Aligner";
    string binname = "aligner";

    // generic options
    PO::options_description opt_generic("Generic Options");
    opt_generic.add_options()
        ("help", "print this manual and exit")
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
        ("out-prob", PO::value<string>(), "path to output lexical probabilities")
        ;
    // configuration
    PO::options_description opt_config("Configurations");
    opt_config.add_options()
        ("unknown-threshold", PO::value<int>()->default_value(5), "maximum frequency to assume the word is unknown")
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
        cerr << "Usgae: " << binname << " [options]" << endl;
        cerr << opt << endl;
        exit(1);
    }

    // check required options
    if (!args.count("src-tree") || !args.count("trg-tree")) {
        cerr << "ERROR: insufficient required options" << endl;
        cerr << "(--help to show usage)" << endl;
        exit(1);
    }

    return move(args);
}

int main(int argc, char * argv[]) {
    auto args = parseOptions(argc, argv);
    
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
    cerr << "loading data ..." << endl;
    
    string src, trg;
    int num_data = 0;

    while (getline(*ifs_src_tree, src) && getline(*ifs_trg_tree, trg)) {
        const auto src_tree = Utility::parseTree(src, src_tag_dict, src_word_dict);
        const auto trg_tree = Utility::parseTree(trg, trg_tag_dict, trg_word_dict);
        const auto src_sentence = Utility::extractWords(src_tree);
        const auto trg_sentence = Utility::extractWords(trg_tree);

        src_tree_list.push_back(std::move(src_tree));
        trg_tree_list.push_back(std::move(trg_tree));
        src_sentence_list.push_back(std::move(src_sentence));
        trg_sentence_list.push_back(std::move(trg_sentence));

        ++num_data;
        if (num_data % 100000 == 0) cerr << num_data << endl;
        else if (num_data % 10000 == 0) cerr << '.';
    }
    
    cerr << endl;
    cerr << "loaded " << num_data << " pairs" << endl;

    cerr << "recognized " << src_tag_dict.size() << " types of src grammar tags" << endl;
    cerr << "recognized " << trg_tag_dict.size() << " types of trg grammar tags" << endl;
    cerr << "recognized " << src_word_dict.size() << " types of src words" << endl;
    cerr << "recognized " << trg_word_dict.size() << " types of trg words" << endl;

    // count words and replace rare words with unknown word
    cerr << "reduce vocabularies ..." << endl;

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
    for (size_t i : boost::irange(2UL, src_word_dict.size())) {
        if (src_word_freq[i] > unknown_threshold) src_word_map[i] = src_num_reduced_words++;
    }

    int trg_num_reduced_words = 2;
    for (size_t i : boost::irange(2UL, trg_word_dict.size())) {
        if (trg_word_freq[i] > unknown_threshold) trg_word_map[i] = trg_num_reduced_words++;
    }

    cerr << "the size of src vocabulary is reduced to " << src_num_reduced_words << endl;
    cerr << "the size of trg vocabulary is reduced to " << trg_num_reduced_words << endl;

    for (auto & sent : src_sentence_list) {
        for (size_t i : boost::irange(0UL, sent.size())) {
            sent[i] = src_word_map[sent[i]];
        }
    }
    for (auto & sent : trg_sentence_list) {
        for (size_t i : boost::irange(0UL, sent.size())) {
            sent[i] = trg_word_map[sent[i]];
        }
    }
    
    for (int i : boost::irange(0, 10)) {
        for (auto w : src_sentence_list[i]) cout << w << ' ';
        cout << endl;
    }

    Aligner::Aligner::calculateIbmModel1(
        src_sentence_list, trg_sentence_list,
        src_num_reduced_words, trg_num_reduced_words,
        100, NULL_ID);

    return 0;
}

