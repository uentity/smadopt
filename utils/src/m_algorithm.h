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

//template< norm_types nt = l2 >
struct _CLASS_DECLSPEC norm_tools {
	enum norm_types {
		l2 = 1
	};

	struct dist_stat {
		//minimum point-to-point distance
		double min_;
		//mean distance between points
		double mean_;
		//mean square error of distances
		double mse_;
		//mean distance between nearest neighbours
		double mean_nn_;
		//mean square error of distances between nearest neighbours
		double mse_nn_;

		dist_stat() : min_(0), mean_(0), mse_(0), mean_nn_(0), mse_nn_(0) {}
	};

	typedef Matrix (*vm_norm_fn_t)(const Matrix&, const Matrix&);
	typedef double (*vv_norm_fn_t)(const Matrix&, const Matrix&);
	typedef Matrix (*deriv_fn_t)(const Matrix&, const Matrix&);
	typedef dist_stat (*cdm_fn_t)(const Matrix&, Matrix&);

	//
	template< norm_types nt >
	static Matrix vm_norm(const Matrix& v_from, const Matrix& m_to);

	template< norm_types nt >
	static Matrix vm_norm2(const Matrix& v_from, const Matrix& m_to);

	template< norm_types nt >
	static double vv_norm(const Matrix& v_from, const Matrix& v_to);

	template< norm_types nt >
	static double vv_norm2(const Matrix& v_from, const Matrix& v_to);

	template< norm_types nt >
	static Matrix deriv(const Matrix& v1, const Matrix& v2);

	template< norm_types nt >
	static dist_stat calc_dist_matrix(const Matrix& data, Matrix& dist);

	template< norm_types nt >
	static Matrix::indMatrix closest_pairs(Matrix& dist);
};

#endif
