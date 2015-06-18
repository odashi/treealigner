#pragma once

#include <utility>
#include <vector>

namespace TreeAligner {

// Array for range [min, max]
template <class T>
class RangedVector {
    
public:
    inline RangedVector()
        : mem_(), min_(0), max_(0) {}
    inline RangedVector(int min, int max)
        : mem_(max - min + 1), min_(min), max_(max) {}
    inline RangedVector(int min, int max, const T & default_value)
        : mem_(max - min + 1, default_value), min_(min), max_(max) {}
    inline RangedVector(const RangedVector<T> & src)
        : mem_(src.mem_), min_(src.min_), max_(src.max_) {}
    inline RangedVector(RangedVector<T> && src)
        : mem_(std::move(src.mem_)), min_(src.min_), max_(src.max_) {}

    inline ~RangedVector() {}

    inline RangedVector & operator=(const RangedVector<T> & src) {
        mem_ = src.mem_;
        min_ = src.min_;
        max_ = src.max_;
        return *this;
    }
    inline RangedVector & operator=(RangedVector<T> && src) {
        mem_ = std::move(src.mem_);
        min_ = src.min_;
        max_ = src.max_;
        return *this;
    }

    inline const T & operator[](int index) const { return mem_[index - min_]; }
    inline T & operator[](int index) { return mem_[index - min_]; }

    inline const T & at(int index) const { return mem_.at(index - min_); }
    inline T & at(int index) { return mem_.at(index - min_); }

    inline typename std::vector<T>::iterator begin() { return mem_.begin(); }
    inline typename std::vector<T>::const_iterator begin() const { return mem_.begin(); }
    inline typename std::vector<T>::iterator end() { return mem_.end(); }
    inline typename std::vector<T>::const_iterator end() const { return mem_.end(); }

    inline int min() const { return min_; }
    inline int max() const { return max_; }

private:
    std::vector<T> mem_;
    int min_;
    int max_;

}; // class RangedVector

} // namespace TreeAligner

