#ACLOCAL_AMFLAGS = -I config/m4
ga_client_PROGRAMS = ga_client
ga_client_SOURCES =	\
ga_client.cpp		\
kar2words.cpp

#ga_client_DATA = 	\
#ga.ini		\
#kmeans.ini		\
#nn.ini		\
#nn_addon.ini		\
#nn_addon_ga.ini		\
#nn_addon_new.ini

ga_client_LDADD = -L@top_builddir@/utils/src -L@top_builddir@/alg/src -lha_utils -lha_alg
ga_client_CPPFLAGS = -I@top_srcdir@/utils/src -I@top_srcdir@/alg/src
ga_client_CXXFLAGS = 
