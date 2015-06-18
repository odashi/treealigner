#pragma once

#include <utility>
#include <vector>

namespace TreeAligner {

// Array for range [min, max]
template <class T>
class ranged_vector {
    
public:
    inline ranged_vector()
        : mem_(), min_(0), max_(0) {}
    inline ranged_vector(int min, int max)
        : mem_(max - min + 1), min_(min), max_(max) {}
    inline ranged_vector(int min, int max, const T & default_value)
        : mem_(max - min + 1, default_value), min_(min), max_(max) {}
    inline ranged_vector(const ranged_vector<T> & src)
        : mem_(src.mem_), min_(src.min_), max_(src.max_) {}
    inline ranged_vector(ranged_vector<T> && src)
        : mem_(std::move(src.mem_)), min_(src.min_), max_(src.max_) {}

    inline ~ranged_vector() {}

    inline ranged_vector & operator=(const ranged_vector<T> & src) {
        mem_ = src.mem_;
        min_ = src.min_;
        max_ = src.max_;
        return *this;
    }
    inline ranged_vector & operator=(ranged_vector<T> && src) {
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

}; // class ranged_vector

struct Alignment {
    int src;
    int trg;
}; // struct Alignment

template <class T> using Sentence = std::vector<T>;

template <class T> using Tensor1 = std::vector<T>;
template <class T> using Tensor2 = std::vector<Tensor1<T>>;
template <class T> using Tensor3 = std::vector<Tensor2<T>>;

template <class T> using RangedTensor1 = ranged_vector<T>;
template <class T> using RangedTensor2 = ranged_vector<RangedTensor1<T>>;
template <class T> using RangedTensor3 = ranged_vector<RangedTensor2<T>>;

template <class T> inline Tensor1<T> make_tensor1(int n1) { return Tensor1<T>(n1); }
template <class T> inline Tensor1<T> make_tensor1(int n1, const T & val) { return Tensor1<T>(n1, val); }
template <class T> inline Tensor2<T> make_tensor2(int n1, int n2) { return Tensor2<T>(n1, Tensor1<T>(n2)); }
template <class T> inline Tensor2<T> make_tensor2(int n1, int n2, const T & val) { return Tensor2<T>(n1, Tensor1<T>(n2, val)); }
template <class T> inline Tensor3<T> make_tensor3(int n1, int n2, int n3) { return Tensor3<T>(n1, Tensor2<T>(n2, Tensor1<T>(n3))); }
template <class T> inline Tensor3<T> make_tensor3(int n1, int n2, int n3, const T & val) { return Tensor3<T>(n1, Tensor2<T>(n2, Tensor1<T>(n3, val))); }

template <class T> inline RangedTensor1<T> make_ranged_tensor1(int l1, int h1) { return RangedTensor1<T>(l1, h1); }
template <class T> inline RangedTensor1<T> make_ranged_tensor1(int l1, int h1, const T & val) { return RangedTensor1<T>(l1, h1, val); }
template <class T> inline RangedTensor2<T> make_ranged_tensor2(int l1, int h1, int l2, int h2) { return RangedTensor2<T>(l1, h1, RangedTensor1<T>(l2, h2)); }
template <class T> inline RangedTensor2<T> make_ranged_tensor2(int l1, int h1, int l2, int h2, const T & val) { return RangedTensor2<T>(l1, h1, RangedTensor1<T>(l2, h2, val)); }
template <class T> inline RangedTensor3<T> make_ranged_tensor3(int l1, int h1, int l2, int h2, int l3, int h3) { return RangedTensor3<T>(l1, h1, RangedTensor2<T>(l2, h2, RangedTensor1<T>(l3, h3))); }
template <class T> inline RangedTensor3<T> make_ranged_tensor3(int l1, int h1, int l2, int h2, int l3, int h3, const T & val) { return RangedTensor3<T>(l1, h1, RangedTensor2<T>(l2, h2, RangedTensor1<T>(l3, h3, val))); }

} // namespace TreeAligner

