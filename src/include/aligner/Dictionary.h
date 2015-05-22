#pragma once

#include <map>
#include <string>
#include <vector>

namespace Aligner {

class Dictionary {

    Dictionary(const Dictionary &) = delete;
    Dictionary & operator=(const Dictionary &) = delete;

public:
    Dictionary();
    ~Dictionary();

    int getId(const std::string & word);
    int getId(const std::string & word) const;
    
    std::string getWord(int id) const;

    inline size_t size() const { return ids_.size(); }

    inline const std::vector<std::string> & getWordList() const { return rev_; }

private:
    std::map<std::string, int> ids_;
    std::vector<std::string> rev_;

}; // class Dictionary

} // namespace Aligner

