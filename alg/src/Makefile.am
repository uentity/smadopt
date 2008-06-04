#ACLOCAL_AMFLAGS = -I config/m4
alg_LTLIBRARIES = libha_alg.la
libha_alg_la_SOURCES = 			\
alg.cpp			\
alg_api.cpp			\
determ_annealing.cpp			\
ga.cpp			\
ga_opt.cpp			\
ga_stat.cpp			\
kmeans.cpp			\
km_opt.cpp			\
mnet.cpp			\
nna_opt.cpp			\
nn_addon.cpp			\
nn_except.cpp			\
nn_opt.cpp			\
objnet.cpp

alginc_HEADERS = 				\
alg_api.h			\
ccn.h			\
determ_annealing.h			\
ga.h			\
ga_common.h			\
kmeans.h			\
kmeans_common.h			\
layer.h			\
mlp.h			\
mnet.h			\
neuron.h			\
nn_addon.h			\
nn_addon_common.h			\
nn_common.h			\
objnet.h			\
objnet_impl.h			\
pca.h			\
rbn.h

libha_alg_la_LIBADD = -L@top_srcdir@/exe/release -lha_utils
libha_alg_la_CPPFLAGS = -W -Wall -I@top_srcdir@/utils/src
libha_alg_la_CXXFLAGS = 
