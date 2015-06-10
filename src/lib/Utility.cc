#include <treealigner/Utility.h>

#include <boost/algorithm/string.hpp>

#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace std;

namespace TreeAligner {

namespace {

Tree<int> convertTree(const Tree<string> & tree, Dictionary & tag_dict, Dictionary & word_dict) {
    function<Tree<int> (const Tree<string> &)> convert_node
        = [&convert_node, &tag_dict, &word_dict](const Tree<string> & node) -> Tree<int> {
        if (!node.size()) {
            return Tree<int>(word_dict.getId(node.label()));
        } else {
            Tree<int> ret(tag_dict.getId(node.label()));
            for (const Tree<string> & ch : node) ret.add(convert_node(ch));
            return ret;
        }
    };

    return convert_node(tree);
}

} // namespace

unique_ptr<ifstream> Utility::openInputStream(const string & filename) {
    unique_ptr<ifstream> ifs(new ifstream(filename));
    if (!ifs->is_open()) {
        throw runtime_error("could not open \"" + filename + "\"");
    }
    return ifs;
}

Tree<int> Utility::parseTree(const string & text, Dictionary & tag_dict, Dictionary & word_dict) {
    auto skip = [](const string & text, size_t & pos) -> void {
        while (pos < text.size() && text[pos] == ' ') ++pos;
        if (pos == text.size()) throw runtime_error("invalid S-expression format");
    };

    auto parse_label = [](const string & text, size_t & pos) -> string {
        size_t begin = pos;
        while (pos < text.size() && text[pos] != '(' && text[pos] != ')' && text[pos] != ' ') ++pos;
        return text.substr(begin, pos-begin);
    };

    function<Tree<string>(const string &, size_t &)> parse_tree
        = [&skip, &parse_label, &parse_tree](const string & text, size_t & pos) -> Tree<string> {
        if (text[pos] == '(') {
            // branch
            ++pos;
            skip(text, pos);
            Tree<string> node(parse_label(text, pos));
            skip(text, pos);
            while (text[pos] != ')') {
                node.add(parse_tree(text, pos));
                skip(text, pos);
            }
            ++pos;
            return node;
        } else {
            // leaf
            return Tree<string>(parse_label(text, pos));
        }
    };

    size_t pos = 0;
    string tmp = text;
    boost::trim(tmp);
    Tree<int> root = convertTree(parse_tree(tmp, pos), tag_dict, word_dict);
    if (pos != tmp.size()) {
        throw runtime_error("invalid S-expression format");
    }

    return root;
}

vector<int> Utility::extractWords(const Tree<int> & tree) {
    function<void(const Tree<int> &, vector<int> &)> search_node
        = [&search_node](const Tree<int> & node, vector<int> & result) -> void {
        if (!node.size()) {
            result.push_back(node.label());
        } else {
            for (const Tree<int> & ch : node) search_node(ch, result);
        }
    };

    vector<int> result;
    search_node(tree, result);
    return result;
}

} // namespace TreeAligner

