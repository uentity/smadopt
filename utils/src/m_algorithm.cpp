#include "m_algorithm.h"
#include "prg.h"
#include "jama/jama_svd.h"
#include "jama/jama_eig.h"

#include <cmath>

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

void assign_mat_ar1d(Matrix& m, const Array1D< double >& a)
{
	m.NewMatrix(1, a.dim());
	Matrix::r_iterator p_m(m.begin());
	for(long i = 0; i < a.dim(); ++i) {
		*p_m = a[i];
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

void eig(const Matrix& A, Matrix& E) {
	Array2D< double > AA(A.row_num(), A.col_num(), const_cast< double* >(A.GetBuffer()));
	Eigenvalue< double > eig_core(AA);
	Array1D< double > res_eigs;
	eig_core.getRealEigenvalues(res_eigs);
	assign_mat_ar1d(E, res_eigs);
}

//==================================== norm_tools implementation =======================================================
//typedef Matrix (*vm_norm2_fn_t)(const Matrix&, const Matrix&);
//typedef double (*vv_norm2_fn_t)(const Matrix&, const Matrix&);

//template< int cdm_method >
norm_tools::dist_stat calc_dist_matrix_impl(const Matrix& data, Matrix& dist, norm_tools::vm_norm_fn_t norm2_fn)
{
	//some constant values used
	const ulong points_num = data.row_num();
	const double mult = 2.0/(points_num * (points_num - 1));
	const double mult1 = 1.0/data.row_num();

	//resize distance matrix
	dist(points_num, points_num);
	//statistics
	norm_tools::dist_stat stat;
	//make a copy of input data
	Matrix data_cpy(data.row_num(), data.col_num(), data.GetBuffer());
	//locally used matrices
	Matrix dist_row(1, data.row_num()), dv, norm;
	//mean distance^2
	double meand2 = 0, meand2_nn = 0;
	//current min distance (between nearest neighbours)
	double cur_mind;

	for(ulong i = 0; i < points_num - 1; ++i) {
		dv <<= data_cpy.GetRows(0);
		data_cpy.DelRows(0);
		//calc norms^2
		norm <<= norm2_fn(dv, data_cpy);
		//update mean distance^2
		meand2 += mult * norm.Sum();
		//update mean nn distance^2
		meand2_nn += mult1 * cur_mind * cur_mind;

		//calc norm
		transform(norm, ptr_fun< double, double >(std::sqrt));

		dist_row = 0;
		dist_row.SetColumns(norm, i + 1, norm.size());
		dist.SetRows(dist_row, i);
		//calc mean
		stat.mean_ += mult * norm.Sum();
		//calc distance to nearest neighbour
		cur_mind = norm.Min();
		if(i == 0 || cur_mind < stat.min_) stat.min_ = cur_mind;
		//update mean distance to nearest neighbour
		stat.mean_nn_ += cur_mind * mult1;
	}
	//fill lower triangle of distance matrix
	dist <<= dist + !dist;

	//calc mse
	stat.mse_ = sqrt(stat.mean_*(meand2/stat.mean_ - stat.mean_));
	stat.mse_nn_ = sqrt(stat.mean_nn_*(meand2_nn/stat.mean_nn_ - stat.mean_nn_));
	return stat;
}

norm_tools::dist_stat calc_dist_matrix_impl(const Matrix& data, Matrix& dist, norm_tools::vv_norm_fn_t norm2_fn)
{
	//some constant values used
	const ulong points_num = data.row_num();
	const double mult = 2.0/(points_num * (points_num - 1));
	const double mult1 = 1.0/(points_num - 1);

	//localy used matrices
	Matrix dv, norm; //, dist_row(1, points_num);
	//statistics
	norm_tools::dist_stat stat;
	//meand distance^2 & meand distance^2 between nearest neighbours
	double meand2 = 0, meand2_nn = 0;
	//current distance & minimum distances
	double cur_dist, cur_mind = 0, cur_mind2 = 0;

	//resize distance matrix
	dist.Resize(points_num, points_num);
	//zero distance matrix
	dist = 0;
	//start distances calculation
	for(ulong i = 0; i < points_num - 1; ++i) {
		dv <<= data.GetRows(i);
		//calc dist^2 to all other rows
		for(ulong j = i + 1; j < points_num; ++j) {
			//calc dist^2
			//norm <<= dv - data.GetRows(j);
			//transform(norm, norm, multiplies<double>());
			//cur_dist = distance^2
			//cur_dist = norm.Sum();
			cur_dist = norm2_fn(dv, data.GetRows(j));

			//update mean of distance^2
			meand2 += mult * cur_dist;
			//update current min distance^2
			if(j == i + 1 || cur_dist < cur_mind2) cur_mind2 = cur_dist;
			//cur_dist = pure l2 distance
			cur_dist = sqrt(cur_dist);
			//update current minimum distance (to nearest neighbour)
			if(j == i + 1 || cur_dist < cur_mind) cur_mind = cur_dist;
			//save it into matrix elements (i,j) and (j, i)
			dist(i, j) = cur_dist; dist(j, i) = cur_dist;
			//update global mean distance
			stat.mean_ += cur_dist * mult;
		}
		//update global min distance
		if(i == 0 || cur_mind < stat.min_) stat.min_ = cur_mind;
		//update mean nearest neighbour distance
		stat.mean_nn_ += cur_mind * mult1;
		//update mean nearest neighbour distance^2
		meand2_nn += cur_mind2 * mult1;
	}

	//calc mse
	stat.mse_ = sqrt(stat.mean_*(meand2/stat.mean_ - stat.mean_));
	stat.mse_nn_ = sqrt(stat.mean_nn_*(meand2_nn/stat.mean_nn_ - stat.mean_nn_));
	return stat;
}

template< >
Matrix norm_tools::vm_norm2< norm_tools::l2 >(const Matrix& v_from, const Matrix& m_to) {
	Matrix norm(1, m_to.row_num());
	Matrix diff;
	for(ulong i = 0; i < norm.size(); ++i) {
		diff <<= v_from - m_to.GetRows(i);
		norm[i] = diff.Mul(diff).Sum();
	}
	return norm;
}
//instantiate
template Matrix norm_tools::vm_norm2< norm_tools::l2 >(const Matrix&, const Matrix&);

template< >
Matrix norm_tools::vm_norm< norm_tools::l2 >(const Matrix& v_from, const Matrix& m_to) {
	Matrix norm(1, m_to.row_num());
	for(ulong i = 0; i < norm.size(); ++i)
		norm[i] = (v_from - m_to.GetRows(i)).norm2();
	return norm;

//		Matrix res = vm_norm2(v_from, m_to);
//		transform(res, ptr_fun< double, double >(std::sqrt));
//		return res;
}
//instantiate
template Matrix norm_tools::vm_norm< norm_tools::l2 >(const Matrix&, const Matrix&);

template< >
double norm_tools::vv_norm2< norm_tools::l2 >(const Matrix& v_from, const Matrix& v_to) {
	Matrix diff = v_from - v_to;
	return diff.Mul(diff).Sum();
}
//instantiate
template double norm_tools::vv_norm2< norm_tools::l2 >(const Matrix&, const Matrix&);

template< typename norm_tools::norm_types nt >
double norm_tools::vv_norm(const Matrix& v_from, const Matrix& v_to) {
	return sqrt(vv_norm2< nt >(v_from, v_to));
}
//instantiate
template double norm_tools::vv_norm< norm_tools::l2 >(const Matrix&, const Matrix&);

template< >
Matrix norm_tools::deriv< norm_tools::l2 >(const Matrix& v1, const Matrix& v2) {
	return v1 - v2;
}
//instantiate
template Matrix norm_tools::deriv< norm_tools::l2 >(const Matrix&, const Matrix&);

template< typename norm_tools::norm_types nt >
norm_tools::dist_stat norm_tools::calc_dist_matrix(const Matrix& data, Matrix& dist) {
	return calc_dist_matrix_impl(data, dist, &norm_tools::vv_norm2< nt >);
}
//instantiate
template norm_tools::dist_stat norm_tools::calc_dist_matrix< norm_tools::l2 >(const Matrix&, Matrix&);

template< typename norm_tools::norm_types nt >
Matrix::indMatrix norm_tools::closest_pairs(Matrix& dist) {
	//make copy
	//Matrix dist_cpy; dist_cpy = dist;
	//sort distances by ascending
	Matrix::indMatrix asc_di = dist.RawSort();
	//remove indexes of first zero elements
	asc_di.DelColumns(0, dist.row_num());
	//remove indexes of every second duplicating element
	for(ulong i = dist.size() - 1; i < dist.size(); i-=2)
		asc_di.DelColumns(i);
	//now we build ordered by ascending distance set of points
	return asc_di;
}
//instantiate
template Matrix::indMatrix norm_tools::closest_pairs< norm_tools::l2 >(Matrix&);
