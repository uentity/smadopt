#ifndef _M_ALGORITHM_H
#define _M_ALGORITHM_H

#include "matrix.h"

#define USE_JAMA

_LIBAPI void lu_gepp(Matrix& m);
_LIBAPI void svd(const Matrix& A, Matrix& U, Matrix& E, Matrix& V);
_LIBAPI void pseudo_inv(const Matrix& A, Matrix& piA);

#endif
