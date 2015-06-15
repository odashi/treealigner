#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

namespace TreeAligner {

template <class T>
class Tensor2 {

public:
    inline Tensor2()
        : mem_(), rows_(0), cols_(0) {}
    inline Tensor2(size_t rows, size_t cols)
        : mem_(rows, std::vector<T>(cols)), rows_(rows), cols_(cols) {}
    inline Tensor2(size_t rows, size_t cols, const T & default_value)
        : mem_(rows, std::vector<T>(cols, default_value)), rows_(rows), cols_(cols) {}
    inline Tensor2(const Tensor2<T> & src)
        : mem_(src.mem_), rows_(src.rows_), cols_(src.cols_) {}
    inline Tensor2(Tensor2<T> && src)
        : mem_(std::move(src.mem_)), rows_(src.rows_), cols_(src.cols_) {}

    inline ~Tensor2() {}

    inline Tensor2 & operator=(const Tensor2<T> & src) {
        mem_ = src.mem_;
        rows_ = src.rows_;
        cols_ = src.cols_;
        return *this;
    }
    inline Tensor2 & operator=(Tensor2<T> && src) {
        mem_ = std::move(src.mem_);
        rows_ = src.rows_;
        cols_ = src.cols_;
        return *this;
    }

    inline const T & at(size_t row, size_t col) const {
        return mem_[row][col];
    }

    inline T & at(size_t row, size_t col) {
        return const_cast<T &>(static_cast<const Tensor2 &>(*this).at(row, col));
    }

    inline size_t rows() const { return rows_; }
    inline size_t cols() const { return cols_; }

private:
    std::vector<std::vector<T>> mem_;
    size_t rows_;
    size_t cols_;

}; // class Tensor2

} // namespace TreeAligner

