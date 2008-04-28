#ifndef _M_ALGORITHM_H
#define _M_ALGORITHM_H

#include "matrix.h"
#include <iostream>
#include <fstream>

#define USE_JAMA

_LIBAPI void lu_gepp(Matrix& m);
_LIBAPI void svd(const Matrix& A, Matrix& U, Matrix& E, Matrix& V);
_LIBAPI void pseudo_inv(const Matrix& A, Matrix& piA);
_LIBAPI void eig(const Matrix& A, Matrix& E);

//prints matrix to file or to cout
template< class T, template <class> class buf_traits_type >
void DumpMatrix(const TMatrix<T, buf_traits_type>& m, const char* pFname = NULL, int num_width = 0)
{
	if(pFname) {
		std::ofstream fd(pFname, std::ios::out | std::ios::trunc);
		if(num_width > 0) fd.precision(num_width);
		m.Print(fd, true, num_width);
	}
	else m.Print(std::cout, true, num_width);
}

#endif
