#pragma once

#include <boost/format.hpp>

#include <string>

namespace TreeAligner {

class Tracer {

    Tracer() = delete;
    Tracer(const Tracer &) = delete;
    Tracer & operator=(const Tracer &) = delete;

public:
    static void println(unsigned int level, const std::string & text);
    static void println(unsigned int level, const boost::format & format);
    static unsigned int getTraceLevel();
    static void setTraceLevel(unsigned int value);

private:
    static unsigned int trace_level_;

}; // class Tracer

} // namespace TreeAligner

