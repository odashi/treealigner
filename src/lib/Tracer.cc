#include <aligner/Tracer.h>

#include <boost/range/irange.hpp>

#include <iostream>

using namespace std;
using boost::irange;

namespace Aligner {

unsigned int Tracer::trace_level_ = 0;

void Tracer::println(unsigned int level, const string & text) {
    if (level > trace_level_) return;
    cerr << string(2 * level, ' ') << text << endl;
}

void Tracer::println(unsigned int level, const boost::format & format) {
    if (level > trace_level_) return;
    cerr << string(2 * level, ' ') << format << endl;
}

unsigned int Tracer::getTraceLevel() {
    return trace_level_;
}

void Tracer::setTraceLevel(unsigned int value) {
    trace_level_ = value;
}

} // namespace Aligner

