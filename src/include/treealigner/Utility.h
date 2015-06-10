#pragma once

#include <treealigner/Dictionary.h>
#include <treealigner/Tree.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <tuple>

namespace TreeAligner {

class Utility {

    Utility() = delete;
    Utility(const Utility &) = delete;
    Utility & operator=(const Utility &) = delete;

public:
    // IO utils
    static std::unique_ptr<std::ifstream> openInputStream(const std::string & filename);

    // make tree from S-expression
    static Tree<int> parseTree(const std::string & text, Dictionary & tag_dict, Dictionary & word_dict);

    // extract word list from tree
    static std::vector<int> extractWords(const Tree<int> & tree);

}; // class Utility

} // namespace TreeAligner

