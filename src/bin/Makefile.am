AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = ../lib/libtreealigner.la $(BOOST_LDFLAGS) $(BOOST_IOSTREAMS_LIBS) $(BOOST_PROGRAM_OPTIONS_LIB)

bin_PROGRAMS = treealigner

treealigner_SOURCES = main.cc
treealigner_LDADD = $(LDADD)
