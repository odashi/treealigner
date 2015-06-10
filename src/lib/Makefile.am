lib_LTLIBRARIES = libtreealigner.la

AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS) -DPKGDATADIR='"$(pkgdatadir)"'

libtreealigner_la_SOURCES = \
    Aligner.cc \
    Dictionary.cc \
    Tracer.cc \
    Utility.cc

libtreealigner_la_LDFLAGS = -version-info 0:1:0 $(BOOST_LDFLAGS) $(BOOST_IOSTREAMS_LIB)

