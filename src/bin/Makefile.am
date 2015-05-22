AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = ../lib/libaligner.la $(BOOST_LDFLAGS) $(BOOST_IOSTREAMS_LIBS) $(BOOST_PROGRAM_OPTIONS_LIB)

bin_PROGRAMS = aligner

aligner_SOURCES = main.cc
aligner_LDADD = $(LDADD)
