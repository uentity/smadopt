AC_INIT([hybrid_adapt], 0.1, [GagarinAV@ufanipi.ru])
AC_PREREQ([2.60])
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([-Wall 0.1 foreign dist-bzip2])
AC_DISABLE_STATIC

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL

data_suffix=$PACKAGE_NAME
#AC_DEFINE_UNQUOTED([DATA_SUFFIX], ["$plugins"],[plugins dir])
AC_SUBST([utilsdir], [$datadir/$data_suffix])
AC_SUBST([utilsincdir], [$includedir/$data_suffix/utils])
AC_SUBST([algdir],[$datadir/$data_suffix])
AC_SUBST([algincdir],[$includedir/$data_suffix/alg])
AC_SUBST([jamaincdir],[$includedir/$data_suffix/alg/jama])
AC_SUBST([ga_clientdir],[$datadir/$data_suffix])
AC_SUBST([ga_client_inidir],[$datadir/$data_suffix])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile ./utils/src/Makefile ./alg/src/Makefile \
                 ./ga_client/src/Makefile])
AC_OUTPUT

