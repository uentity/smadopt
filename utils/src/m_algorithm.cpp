#include "m_algorithm.h"
#include "prg.h"
#include "jama/jama_svd.h"

#define TOL 1.0e-15

using namespace std;
using namespace JAMA;

typedef auto_ptr<Matrix> dMPtr;
typedef Matrix::r_iterator r_iterator;
typedef Matrix::col_iterator col_iterator;

void lu_gepp(Matrix& m)
{
	if(m.row_num() != m.col_num()) return;
	r_iterator p_row, pos;
	col_iterator p_col(m);
	ulong n = m.row_num();
	for(ulong i = 0; i < n - 1; ++i) {
		p_row = m.begin() + (i*n + i);
		p_col = p_row;
		//
		//find row with maximum column value
		ulong ind_max = i;
		double max_col = *p_row;
		++p_col;
		for(ulong j = i + 1; i < n; ++i) {
			if(*p_col > max_col) {
				max_col = *p_col;
				ind_max = j;
			}
			++p_col;
		}
		//exchange i & ind_max
		if(ind_max != i) m.permutate_rows(i, ind_max);

		// L
		p_col = p_row;
		transform(++p_col, p_col + (n - i), p_col, bind2nd(divides<double>(), max_col));

		//      U
		pos = ++p_row + n;
		for(ulong k = i + 1; k < n; ++k) {
			for(ulong j = i + 1; j < n; ++j) {
				*pos -= (*pos)*(*p_row);
				++pos; ++p_row;
			}
			pos += i;
			p_row -= n - i - 1;
		}
	}
}

double det(const Matrix& m)
{
	if(m.row_num() != m.col_num()) return 0;
	dMPtr r(new Matrix);
	*r = m;
	lu_gepp(*r);
	double ret = 1;
	for(r_iterator pos = r->begin(); pos != r->end(); pos += r->row_num() + 1)
		ret *= *pos;
	return ret;
}

void assign_mat_ar2d(Matrix& m, const Array2D<double>& a)
{
	m.NewMatrix(a.dim1(), a.dim2());
	Matrix::r_iterator p_m(m.begin());
	for(long i = 0; i < a.dim1(); ++i)
		for(long j = 0; j < a.dim2(); ++j) {
			*p_m = a[i][j];
			++p_m;
		}
}

void svd(const Matrix& A, Matrix& U, Matrix& E, Matrix& V)
{
	Array2D<double> AA(A.row_num(), A.col_num(), const_cast< double* >(A.GetBuffer()));
	SVD<double> kernel(AA);
	Array2D<double> AU, AE, AV;
	kernel.getU(AU); kernel.getV(AV); kernel.getS(AE);
	assign_mat_ar2d(U, AU);
	assign_mat_ar2d(V, AV);
	assign_mat_ar2d(E, AE);
}

void pseudo_inv(const Matrix& A, Matrix& piA)
{
	Matrix U, E, V;
	//calc SVD
	svd(A, U, E, V);
	//calc E^-1 - only diagonal elements are non-zero
	Matrix::r_iterator p_e(E.begin());
	for(ulong i = 0; i < E.row_num(); ++i) {
		if(abs(*p_e) > TOL) *p_e = 1 / *p_e;
		else *p_e = 0;
		p_e += E.col_num() + 1;
	}
	//calc pseudo inverse
	piA = V * E * (!U);
}
