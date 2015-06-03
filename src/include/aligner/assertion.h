#pragma once

#include <stdexcept>

#define MYASSERT(func, cond) { if (!(cond)) throw std::runtime_error("Assertion failed: " #func ": " #cond "."); }

