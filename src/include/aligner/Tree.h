#pragma once

#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>

namespace Aligner {

template <class T>
class Tree {

    static const int NUM_RESERVED_CHILDREN = 0;

public:
    inline Tree()
        : label_(), children_() { children_.reserve(NUM_RESERVED_CHILDREN); }
    inline explicit Tree(const T & label)
        : label_(label), children_() { children_.reserve(NUM_RESERVED_CHILDREN); }
    inline Tree(const Tree<T> & src) 
        : label_(src.label_), children_(src.children_) {}
    inline Tree(Tree<T> && src)
        : label_(std::move(src.label_)), children_(std::move(src.children_)) {}

    inline ~Tree() {}

    inline Tree & operator=(const Tree<T> & src) {
        label_ = src.label_;
        children_ = src.children_;
    }
    inline Tree & operator=(Tree<T> && src) {
        label_ = std::move(src.label_);
        children_ = std::move(src.children_);
    }

    inline const Tree & operator[](int index) const { return children_[index]; }

    inline T label() const { return label_; }
    inline size_t size() const { return children_.size(); }

    inline typename std::vector<Tree<T>>::const_iterator begin() const { return children_.begin(); }
    inline typename std::vector<Tree<T>>::const_iterator end() const { return children_.end(); }

    inline void add(const Tree<T> & child) { children_.push_back(child); }
    inline void add(Tree<T> && child) { children_.push_back(std::move(child)); }

private:
    T label_;
    std::vector<Tree<T>> children_;

}; // class Tree

template <class T>
std::ostream & operator<<(std::ostream & os, const Tree<T> & tree) {
    if (!tree.size()) {
        os << tree.label();
    } else {
        os << '(' << tree.label();
        for (const Tree<T> & ch : tree) os << ' ' << ch;
        os << ')';
    }
    return os;
}

} // namespace Aligner

