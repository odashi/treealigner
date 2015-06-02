lib_LTLIBRARIES = libaligner.la

AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS) -DPKGDATADIR='"$(pkgdatadir)"'

libaligner_la_SOURCES = \
    Aligner.cc \
    Dictionary.cc \
    utils.cc

libaligner_la_LDFLAGS = -version-info 0:1:0 $(BOOST_LDFLAGS) $(BOOST_IOSTREAMS_LIB)

