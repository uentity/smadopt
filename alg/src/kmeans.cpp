#include "kmeans.h"
#include "prg.h"
#include "ga.h"
#include "m_algorithm.h"

#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <fstream>
//DEBUG!
#include <iostream>
#include <assert.h>

using namespace std;
using namespace GA;
using namespace KM;

#define KM_CALC_DIST_MAT_METHOD 1

/*
void DumpM(ulMatrix* m, char* pFname = NULL)
{
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		m->Print(fd);
	}
	else m->Print(cout);
}
*/

//---------------------------- kmeans_impl  declaration ----------------------------------------------
namespace KM {

struct km_data {
	//matrix of input data, cluster centers and distances from points to corresponding centers
	Matrix data_, c_, norms_;
	//winners matrix - point->center relation
	ulMatrix w_;
	//points affiliation - center->points collection
	kmeans::vvul aff_;
	//sum of distances from points to corresponding cluster centers
	double e_;

	km_data() : e_(0) {}
};

class kmeans::kmeans_impl : public km_data
{
	friend class kmeans;
	friend class pat_sel;

private:
	//misc structures
//	struct dist_stat {
//		//minimum point-to-point distance
//		double min_;
//		//mean distance between points
//		double mean_;
//		//mean square error of distances
//		double mse_;
//		//mean distance between nearest neighbours
//		double mean_nn_;
//		//mean square error of distances between nearest neighbours
//		double mse_nn_;
//
//		dist_stat() : min_(0), mean_(0), mse_(0), mean_nn_(0), mse_nn_(0) {}
//	};

	//---------------------------- pat_sel implementation ----------------------------------------------
	class pat_sel
	{
		typedef Matrix::indMatrix indMatrix;

		//kmeans_impl& km_;

		indMatrix _selection(const Matrix& f)
		{
			Matrix expect;
			indMatrix sel_ind;

			expect <<= ga_.ScalingCall(f);
			sel_ind <<= ga_.SelectionCall(expect, expect.size());

			return sel_ind;
		}

		//void set_def_opt()
		//{
		//	//ga_.iniFname_ = iniFname_;
		//	ga_.opt_.scalingT = Rank;
		//	ga_.opt_.selectionT = StochasticUniform;
		//	ga_.opt_.minimizing = true;
		//	ga_.opt_.ffscParam = 2;
		//	ga_.opt_.nTournSize = 3;
		//}

	public:
		ga ga_;

		pat_sel(kmeans_impl& km) //: km_(km)
		{
			km.opt_.add_embopt(ga_.opt_);
			//set_def_opt();
			ga_.prepare2run(false);
		}
		~pat_sel() {};

		ulMatrix select_by_f(const km_data& kmd, const Matrix& f_all) {
			//assume that w_ filled with patterns affiliations
			ulMatrix res(kmd.data_.row_num(), 1), p_ind(kmd.data_.row_num(), 1);
			indMatrix sel_ind;
			Matrix expect, f(kmd.data_.row_num(), 1);
			ulong cnt, res_ind = 0;
			for(ulong i = 0; i < kmd.c_.row_num(); ++i) {
				//collect points that belongs to current center
				//f.Resize(data.row_num(), 1); p_ind.Resize(data.row_num(), 1);
				cnt = 0;
				for(ulong j = 0; j < kmd.data_.row_num(); ++j) {
					if(kmd.w_[j] == i) {
						f[cnt] = f_all[j];
						p_ind[cnt++] = j;
					}
				}
				if(cnt == 0) continue;
				//f.Resize(cnt); p_ind.Resize(cnt);

				//do selection and assign indexes
				expect <<= ga_.ScalingCall(f.GetRows(0, cnt));
				sel_ind <<= ga_.SelectionCall(expect, expect.size());

				for(ulong j = 0; j < sel_ind.size() && res_ind < res.row_num(); ++j)
					res[res_ind++] = p_ind[sel_ind[j]];
			}

			return res;
		}

		//just performes selection of given values and return the result
		ulMatrix selection(const Matrix& f, ulong how_many = 0) {
			Matrix expect = ga_.ScalingCall(f);
			if(how_many == 0) how_many = expect.size();
			return ga_.SelectionCall(expect, how_many);
		}

		Matrix selection_prob(const Matrix& f, ulong maxcycles = 100) {
			const ulong how_many = f.size(); //min(max(f.size(), (ulong)1000), (ulong)1000);
			const double prob_quant = 1.0/how_many;
			Matrix prob(1, f.size(), 0);
			Matrix new_prob(1, f.size(), 0);
			Matrix diff;
			ulMatrix sel_ind;
			double dist;
			bool stop_patience = false;
			for(ulong i = 0; i < maxcycles && !stop_patience; ++i) {
				//first of all invoke selection
				sel_ind <<= selection(f, how_many);
				//parse selection results
				new_prob = 0;
				for(ulMatrix::r_iterator pos = sel_ind.begin(), end = sel_ind.end(); pos != end; ++pos)
					new_prob[*pos] += prob_quant;
				//new_prob = (new_prob + prob)/2
				new_prob += prob;
				new_prob /= 2;
				//now we have updated probability distribution - check patience
				//diff = distance between prob & new_prob
				diff <<= new_prob - prob;
				diff *= diff;
				dist = sqrt(diff.Sum());
				if(dist < sqrt(prob.Mul(prob).Sum()) * 0.05)
					stop_patience = true;
				prob = new_prob;
			}
			return prob;
		}
	};

	pat_sel& get_ps() {
		static pat_sel ps(*this);
		return ps;
	};

	//drops description
	struct _drop_params {
		double r_;			//radius
		ulong qcnt_;		//number of quants in a drop
		//initialization value for radius
		static double init_r_;

		_drop_params() : r_(init_r_), qcnt_(0) {}
		_drop_params(double r, ulong qcnt) : r_(r), qcnt_(qcnt) {}
	};
	typedef map< ulong, _drop_params > drops_map;
	typedef drops_map::iterator drops_iterator;
	typedef pair< ulong, _drop_params > drop_pair;

	//data members
	Matrix f_, v_, oldc_;
	double preve_;
	ulong cycle_, ni_cycles_;

	Matrix (*_pNormFcn)(const Matrix&, const Matrix&);
	Matrix (*_pDerivFcn)(const Matrix&, const Matrix&);
	void (kmeans_impl::*_pBUFcn)();
	void (kmeans_impl::*_pOrderFcn)(ul_vec&);

//	void l2_norm(const Matrix& dv, const Matrix& points, Matrix& norm) const;
//	Matrix l2_deriv(ulong dv_ind, const Matrix& center) const;
//	Matrix l2f_deriv(ulong dv_ind, const Matrix& center) const;

	void calc_winners(km_data& kmd) const;
	void l2_batch_update();
	void selection_batch_update();
	void selection_prob_batch_update();
	void empty_bu() {};

	//patterns order processing
	void simple_shuffle(ul_vec& porder);
	void selection_based_order(ul_vec& porder);

	void seed(const Matrix& data, ulong clust_num, const Matrix* pCent = NULL, bool use_prev_cent = false);
	void proc_empty_cent(ulMatrix& empty_idx);
	void batch_phase(ulong maxiter);
	void online_phase(ulong maxiter);
	void online_phase_simple(ulong maxiter);
	bool join_phase(ulong maxiter);

	bool patience_check();
	//Xie-Beni validity check for current partitioning
	double xie_beni_validity(const km_data& kmd, double sep = 0) const;

	//helper function to access dist matrix
	template<class dist_buf>
	double _dist_ij(const dist_buf& db, ulong i, ulong j);
	void gen_uniform_cent(const Matrix& bound, const Matrix& dim, Matrix& dv, drops_map& drops, const double quant);

//	template< int method >
//	dist_stat calc_dist_matrix(const Matrix& data, Matrix& dist) const;

public:

	km_opt& opt_;

	kmeans_impl(km_opt& opt) : opt_(opt) {}

	void find_clusters(const Matrix& data, ulong clust_num, ulong maxiter = 1000,
		bool skip_online = false, const Matrix* pCent = NULL, bool use_prev_cent = false);
	void find_clusters_f(const Matrix& data, const Matrix& f, ulong clust_num,
		ulong maxiter = 1000, const Matrix* pCent = NULL, bool use_prev_cent = false);
	void restart(const Matrix& data, ulong clust_num = 0, ulong maxiter = 100,
		bool skip_online = false, const Matrix* pCent = NULL, bool use_prev_cent = false);

	Matrix drops_homo(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100,
		double quant_mult = 0.3);
	Matrix drops_hetero(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100,
		double quant_mult = 0.3);
	Matrix drops_hetero_map(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100,
		double quant_mult = 0.3);
	Matrix drops_hetero_simple(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100,
		double quant_mult = 0.3);
};


//----------------------------kmeans_impl implementation--------------------------------------------------------------
double kmeans::kmeans_impl::_drop_params::init_r_ = 0;

//void kmeans::kmeans_impl::l2_norm(const Matrix& dv, const Matrix& points, Matrix& norm) const {
//	norm.Resize(1, points.row_num());
//	//Matrix diff;
//	for(ulong i = 0; i < points.row_num(); ++i) {
//		norm[i] = (dv - points.GetRows(i)).norm2();
//	}
//}
//
//Matrix kmeans::kmeans_impl::l2_deriv(ulong dv_ind, const Matrix& center) const {
//	return (data_.GetRows(dv_ind) - center);
//}

//Matrix kmeans::kmeans_impl::l2f_deriv(ulong dv_ind, const Matrix& center) const {
//	double mult = 1;
//	//find minimum in current cluster
//	//double fmin = f_[dv_ind];
//	//ulong dv_min = dv_ind;
//	//for(ulong i = 0; i < data_.row_num(); ++i)
//	//	if(w_[i] == w_[dv_ind] && f_[i] < fmin) {
//	//		fmin = f_[i];
//	//		dv_min = i;
//	//	}
//	//do a step also in direction to minimum
//	//return (data_.GetRows(dv_min) - center);
//	//return (data_.GetRows(dv_ind) - center + data_.GetRows(dv_min) - center)/2;
//
//	//mult = 1 + norms_[dv_ind];
//	//if(mult != 0)
//	//mult = f_[dv_ind] / (mult*mult);
//
//	//double sigma = 0;
//	//for(ulong i = 0; i < data_.row_num(); ++i) {
//	//	if(w_[i] == w_[dv_ind]) sigma += norms_[i];
//	//}
//	//sigma /= data_.row_num();
//	//mult = exp(-norms_[dv_ind]/(sigma*sigma)) * f_[dv_ind];
//
//	double min_f = f_[dv_ind], sigma = 0;
//	ulong cnt = 0;
//	for(ulong i = 1; i < data_.row_num(); ++i) {
//		if(w_[i] == w_[dv_ind]) {
//			 if(f_[i] < min_f) min_f = f_[i];
//			sigma += f_[i];
//			++cnt;
//		}
//	}
//	sigma -= cnt*min_f;
//	mult = exp(-(f_[dv_ind] - min_f)/sigma);
//
//	return (data_.GetRows(dv_ind) - center) * mult;
//
//	//approximate gradient
//	//Matrix dv(data_.GetRows(dv_ind)), g(1, data_.col_num(), 0);
//	//for(ulong i = 0; i < data_.row_num(); ++i) {
//	//	if(w_[i] == w_[dv_ind] && norms_[i] > 0)
//	//		g += (data_.GetRows(i) - center) / sqrt(norms_[i]);
//	//}
//	//g /= data_.row_num();
//	//return (dv - center + g)/2;
//}

void kmeans::kmeans_impl::simple_shuffle(ul_vec& porder)
{
	random_shuffle(porder.begin(), porder.end(), prg::randIntUB);
}

void kmeans::kmeans_impl::selection_based_order(ul_vec& porder)
{
	//first determine points affiliation
	calc_winners(*this);

	//now do a selection
	ulMatrix sel_ind = get_ps().select_by_f(*this, f_);

	//porder = sel_ind
	//for(ulong i = 0; i < porder.size(); ++i)
	//	porder[i] = sel_ind[i];
	copy(sel_ind.begin(), sel_ind.begin() + porder.size(), porder.begin());

	//shuffle order obtained
	random_shuffle(porder.begin(), porder.end(), prg::randIntUB);
}

bool kmeans::kmeans_impl::patience_check()
{
	if(cycle_ == 0) {
		ni_cycles_ = 0;
		preve_ = e_;
	}
	else if(preve_ - e_ > opt_.patience*preve_) {
		ni_cycles_ = 0;
		preve_ = e_;
	}
	else if(++ni_cycles_ >= opt_.patience_cycles)
		return true;

	return false;
}

void kmeans::kmeans_impl::seed(const Matrix& data, ulong clust_num, const Matrix* pCent, bool use_prev_sent)
{
	data_ = data;
	Matrix prior_c;
	vector<ulong> idx;
	ulong cnt;
	if(pCent && data.col_num() == pCent->col_num()) {
		//take centers randomly
		idx.resize(pCent->row_num());
		for(ulong i = 0; i < idx.size(); ++i)
			idx[i] = i;
		random_shuffle(idx.begin(), idx.end(), prg::randIntUB);
		cnt = min(clust_num, pCent->row_num());
		for(ulong i = 0; i < cnt; ++i)
			prior_c &= pCent->GetRows(idx[i]);
	}
	if(use_prev_sent && c_.col_num() == data.col_num() && prior_c.row_num() < clust_num) {
		//take centers randomly
		idx.resize(c_.row_num());
		for(ulong i = 0; i < idx.size(); ++i)
			idx[i] = i;
		random_shuffle(idx.begin(), idx.end(), prg::randIntUB);
		cnt = min(clust_num - prior_c.row_num(), c_.row_num());
		for(ulong i = 0; i < cnt; ++i)
			prior_c &= c_.GetRows(idx[i]);

		//while(c_.row_num() > 0 && prior_c.row_num() < clust_num) {
		//	ind = prg::randIntUB(c_.row_num());
		//	prior_c &= c_.GetRows(ind);
		//	c_.DelRows(ind);
		//}
	}
	if(prior_c.row_num() < clust_num) {
		c_.NewMatrix(clust_num - prior_c.row_num(), data.col_num());
		switch(opt_.seed_t) {
			default:
			case sample:
				idx.resize(data.row_num());
				for(ulong i = 0; i < idx.size(); ++i)
					idx[i] = i;
				random_shuffle(idx.begin(), idx.end(), prg::randIntUB);
				for(ulong i = 0; i < c_.row_num() && i < idx.size(); ++i)
					c_.SetRows(data.GetRows(idx[i]), i);
					//c_ &= data.GetRows(idx[i]);
				break;

			case uniform:
				generate(c_.begin(), c_.end(), prg::rand01);
				Matrix mm = data.minmax();
				Matrix::r_iterator p_c = c_.begin();
				for(ulong i = 0; i < c_.row_num(); ++i) {
					for(ulong j = 0; j < c_.col_num(); ++j) {
						*p_c = *p_c * (mm(1, j) - mm(0, j)) + mm(0, j);
						++p_c;
					}
				}
				/*
				Matrix::col_iterator p_c(c_);
				for(ulong i = 0; i < c_.col_num(); ++i) {
					transform(p_c, p_c + c_.row_num(), p_c, bind2nd(multiplies<double>(), mm(1, i) - mm(0, i)));
					p_c = transform(p_c, p_c + c_.row_num(), p_c, bind2nd(plus<double>(), mm(0, i)));
					//p_c += c_.row_num();
				}
				*/
				break;
		}
		c_ &= prior_c;
	}
	else
		c_ <<= prior_c;
	//if(pCent && c_.row_num() < clust_num) c_ &= pCent->GetRows(0, clust_num - c_.row_num());

	switch(opt_.norm_t) {
		default:
		case eucl_l2:
			_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
			_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
			_pBUFcn = &kmeans_impl::l2_batch_update;
			break;
	}

	//other initialization
	aff_.resize(c_.row_num());
	//mem reserve
	for(ulong i = 0; i < c_.row_num(); ++i)
		aff_[i].reserve(data.row_num());

	w_.NewMatrix(1, data.row_num());
	norms_.NewMatrix(1, data.row_num());
	cycle_ = 0;
}

void kmeans::kmeans_impl::calc_winners(km_data& kmd) const
{
	Matrix dv, norm;

	//clear affiliation list
	kmd.aff_.clear();
	kmd.aff_.resize(kmd.data_.row_num());
	//for(ulong i = 0; i < aff_.size(); ++i)
	//	aff_[i].clear();

	kmd.e_ = 0;
	//calc centers-winners for each data point
	for(ulong i = 0; i < kmd.data_.row_num(); ++i) {
		dv <<= kmd.data_.GetRows(i);
		norm <<= (*_pNormFcn)(dv, kmd.c_);
		//save winner
		kmd.w_[i] = norm.min_ind();
		//save affiliation
		kmd.aff_[kmd.w_[i]].push_back(i);

		kmd.norms_[i] = norm[kmd.w_[i]];
		kmd.e_ += kmd.norms_[i];
	}
	kmd.e_ /= kmd.data_.row_num();
}

void kmeans::kmeans_impl::proc_empty_cent(ulMatrix& empty_idx)
{
	if(empty_idx.size() == 0) return;
	Matrix new_c, norm;
	ulong p;
	switch(opt_.emptyc_pol) {
		default:
		case do_nothing:
			break;

		case drop:
			empty_idx.RawSort();
			for(ulong i = empty_idx.size() - 1; i < empty_idx.size(); --i) {
				c_.DelRows(empty_idx[i]);
				aff_.erase(aff_.begin() + empty_idx[i]);
			}
			break;

		case singleton:
			//double max_norm;

			//update norms
			for(ulong i = 0; i < data_.row_num(); ++i) {
				norm <<= (*_pNormFcn)(data_.GetRows(i), c_.GetRows(w_[i]));
				norms_[i] = norm[0];
			}

			for(ulMatrix::r_iterator p_idx = empty_idx.begin(); p_idx != empty_idx.end(); ++p_idx) {
				//find farthest point
				p = norms_.max_ind();
				//replace empty center with found point
				new_c <<= data_.GetRows(p);
				c_.SetRows(new_c, *p_idx);
				w_[p] = *p_idx;
				//set norm to zero to prevent selecting this point again
				norms_[p] = 0;
			}
			//update altered centers
			//(this->*_pBUFcn)();
			break;
	}
}

void kmeans::kmeans_impl::l2_batch_update()
{
	Matrix new_c(1, c_.col_num()), norm;
	ulong pat_cnt;
	ulMatrix empty_c;
	for(ulong i = 0; i < c_.row_num(); ++i) {
		new_c = 0; pat_cnt = 0;
		//for(ulong j = 0; j < data_.row_num(); ++j)
		//	if(w_[j] == i) {
		//		new_c += data_.GetRows(j);
		//		++pat_cnt;
		//	}

		//affilation-based algorithm
		ul_vec& cent_aff = aff_[i];
		for(ulong j = 0; j < cent_aff.size(); ++j) {
			new_c += data_.GetRows(cent_aff[j]);
			++pat_cnt;
		}
		if(pat_cnt > 0) {
			new_c /= pat_cnt;
			c_.SetRows(new_c, i);
		}
		else empty_c.push_back(i);
	}

	if(empty_c.size() > 0)
		proc_empty_cent(empty_c);
}

void kmeans::kmeans_impl::selection_prob_batch_update()
{
	Matrix new_c(1, c_.col_num()), norm;
	ulong pat_cnt;
	ulMatrix empty_c;
	Matrix f, prob;
	for(ulong i = 0; i < c_.row_num(); ++i) {
		new_c = 0; pat_cnt = 0;
		//affilation-based algorithm
		ul_vec& cent_aff = aff_[i];
		f.clear();
		for(ulong j = 0; j < cent_aff.size(); ++j)
			f.push_back(f_[cent_aff[j]], false);
		if(f.size()) {
			//calc probabilities of selection
			prob <<= get_ps().selection_prob(f);
			//calculate new center position based on selection prob
			for(ulong j = 0; j < cent_aff.size(); ++j)
				new_c += data_.GetRows(cent_aff[j]) * prob[j];
		}
		else empty_c.push_back(i);
	}

	if(empty_c.size() > 0)
		proc_empty_cent(empty_c);
}

void kmeans::kmeans_impl::selection_batch_update()
{
	Matrix new_c(1, c_.col_num()), norm;
	ulong pat_cnt;
	ulMatrix empty_c, sel_ind;
	Matrix f;
	for(ulong i = 0; i < c_.row_num(); ++i) {
		new_c = 0; pat_cnt = 0;
		//affilation-based algorithm
		//collect function values of given cluster
		ul_vec& cent_aff = aff_[i];
		f.clear();
		for(ulong j = 0; j < cent_aff.size(); ++j)
			f.push_back(f_[cent_aff[j]], false);
		//calculate new center
		if(f.size()) {
			//calc probabilities of selection
			sel_ind <<= get_ps().selection(f);
			//calculate new center position based on selection indices
			for(ulong j = 0; j < sel_ind.size(); ++j)
				new_c += data_.GetRows(cent_aff[sel_ind[j]]);
			new_c /= sel_ind.size();
		}
		else empty_c.push_back(i);
	}

	if(empty_c.size() > 0)
		proc_empty_cent(empty_c);
}

void kmeans::kmeans_impl::batch_phase(ulong maxiter)
{
	//Matrix dv, norm;
	//ulong winner;

	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		//points affilations
		calc_winners(*this);

		//check patience
		if(patience_check()) break;

		//update centers
		(this->*_pBUFcn)();
	}
	//ensure correct winners & norms are calculated
	calc_winners(*this);
}

void kmeans::kmeans_impl::online_phase(ulong maxiter)
{
	//init distributions v
	v_.NewMatrix(1, c_.row_num());
	v_ = opt_.init_v;

	//patterns present order
	vector<ulong> porder(data_.row_num());
	for(ulong i = 0; i < porder.size(); ++i)
		porder[i] = i;

	//w_.Resize(1, data_.col_num());
	Matrix dv, norm, vnorm, w_c;
	ulong winner;
	double h, ln_k = log((double)v_.size());
	//ni_cycles_ = 0; ulong cyc_start = cycle_;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		//call order processing function
		(this->*_pOrderFcn)(porder);

		e_ = 0;
		//present patterns
		for(ul_vec::iterator p_order = porder.begin(); p_order != porder.end(); ++p_order) {
			dv <<= data_.GetRows(*p_order);

			//calc norm to find closest center
			norm <<= (*_pNormFcn)(dv, c_);
			//norms_.SetRows(norm, i);
			vnorm <<= norm.Mul(v_);
			//find winner
			winner = vnorm.min_ind();
			w_[*p_order] = winner;
			//save norm
			norms_[*p_order] = norm[winner];
			//calc error function
			e_ += norm[winner];

			//calc adaptive speed
			//if(ln_k != 0.0) {
				vnorm = v_ / v_.Sum();
				h = 0;
				for(Matrix::r_iterator p_v(vnorm.begin()); p_v != vnorm.end(); ++p_v)
					h -= (*p_v)*log(*p_v);
				opt_.nu = (ln_k - h)/ln_k;
			//}
			//else opt_.nu = 1;

			//update corresponding center
			w_c <<= c_.GetRows(winner);
			w_c += (*_pDerivFcn)(dv, w_c) * opt_.nu;
			c_.SetRows(w_c, winner);

			//update distributions v
			v_ *= opt_.alfa;
			v_[winner] += (1 - opt_.alfa)*norm[winner];
		}

		e_ /= data_.row_num();
		//if(cycle_ == cyc_start) preve_ = e_;

		if(patience_check()) break;
	}
	//ensure correct winners & norms are calculated
	calc_winners(*this);
}

void kmeans::kmeans_impl::online_phase_simple(ulong maxiter)
{
	//patterns present order
	vector<ulong> porder(data_.row_num());
	for(ulong i = 0; i < porder.size(); ++i)
		porder[i] = i;
	//matrix with points counters
	ulMatrix pc(1, c_.row_num());

	//w_.Resize(1, data_.col_num());
	Matrix dv, norm, w_c;
	ulong winner;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		//call order processing function
		(this->*_pOrderFcn)(porder);

		e_ = 0;
		//present patterns
		for(ul_vec::iterator p_order = porder.begin(); p_order != porder.end(); ++p_order) {
			dv <<= data_.GetRows(*p_order);

			//calc norm to find closest center
			norm <<= (*_pNormFcn)(dv, c_);
			//find winner
			winner = norm.min_ind();
			w_[*p_order] = winner;
			//save norm
			norms_[*p_order] = norm[winner];
			++pc[winner];
			//calc error function
			e_ += norm[winner];

			//adaptive nu calculation
			opt_.nu = 1/(double)pc[winner];
			//update corresponding center
			w_c <<= c_.GetRows(winner);
			w_c += (*_pDerivFcn)(dv, w_c) * opt_.nu;
			c_.SetRows(w_c, winner);
		}

		e_ /= data_.row_num();
		//if(cycle_ == cyc_start) preve_ = e_;

		if(patience_check()) break;
	}
	//ensure correct winners & norms are calculated
	calc_winners(*this);
}

template< class T >
struct less_except_null {
	bool operator()(const T& lhs, const T& rhs) const {
		if(lhs == 0) return false;
		else if(rhs == 0) return true;
		else return (lhs < rhs);
	}
};

double kmeans::kmeans_impl::xie_beni_validity(const km_data& kmd, double sep) const {
	//iterate through centers
	double cov = 0;
	double tmp;
	for(ulong i = 0; i < kmd.c_.row_num(); ++i) {
		//find points that belongs to this center
		for(ulong j = 0, end = kmd.aff_[i].size(); j < end; ++j) {
			tmp = kmd.norms_[kmd.aff_[i][j]];
			sep += tmp * tmp;
		}
	}
	//calc separation if needed
	if(sep == 0) {
		Matrix cdist;
		norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(kmd.c_, cdist);
		sep = ds.min_;
	}

	return sep / (kmd.data_.row_num() * sep * sep);
}

bool kmeans::kmeans_impl::join_phase(ulong maxiter)
{
	//merged centers indexes
	typedef set< ulong, greater< ulong > > merged_idx;
	merged_idx mc;

	km_data kmd;
	ulong c1, c2, f_ind;
	Matrix f, c, norm, expect;
	Matrix::indMatrix sel_ind;
	pat_sel& ps = get_ps();
	Matrix& p = kmd.data_;

	//calc distances matrix
	Matrix dist;
	norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(c_, dist);
	//find closest centers
	Matrix::indMatrix asc_di = norm_tools::closest_pairs< norm_tools::l2 >(dist);

	//discovered centers will be stored here
	Matrix new_c;
	bool do_merge;
	for(ulong d = 0; d < asc_di.size(); ++d) {
		//get index of minimum distance
		//ulong min_dist_ind = std::min_element(dist.begin(), dist.end(), less_except_null< double >()) - dist.begin();

		//main cycle starts here

		//c1 & c2 are centers to be merged
		c1 = asc_di[d] / c_.row_num(), c2 = asc_di[d] % c_.row_num();
		//check if any of those centers are already marked to merge
		if(mc.find(c1) != mc.end() || mc.find(c2) != mc.end())
			continue;

		//collect points belonging to c1 and c2
		f.Resize(aff_[c1].size() + aff_[c2].size(), 1);
		kmd.data_.Resize(aff_[c1].size() + aff_[c2].size(), data_.col_num());
		f_ind = 0;
		for(ulong i = 0; i < aff_[c1].size(); ++i) {
			f[f_ind] = f_[aff_[c1][i]];
			kmd.data_.SetRows(data_.GetRows(aff_[c1][i]), f_ind++);
		}
		for(ulong i = 0; i < aff_[c2].size(); ++i) {
			f[f_ind] = f_[aff_[c2][i]];
			kmd.data_.SetRows(data_.GetRows(aff_[c2][i]), f_ind++);
		}

		//move c1 to new location
		c = c_.GetRows(c1);
		for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
			//select points
			sel_ind <<= ps.selection(f);
			//move center
			for(ulong j = 0; j < sel_ind.size(); ++j)
				c += (*_pDerivFcn)(kmd.data_.GetRows(sel_ind[j]), c) * opt_.nu;
			//calc error
			norm <<= (*_pNormFcn)(c, p);
			e_ = norm.Sum()/norm.size();
			if(patience_check()) break;
		}

		do_merge = false;

		//test if new merged cluster is better than two previous
		//find closest points to each of centers c1, c2 and c
		//and test which one has better function value
		norm <<= (*_pNormFcn)(c_.GetRows(c1), p);
		ulong cl_c1 = norm.min_ind();
		norm <<= (*_pNormFcn)(c_.GetRows(c2), p);
		ulong cl_c2 = norm.min_ind();
		norm <<= (*_pNormFcn)(c, p);
		ulong cl_c = norm.min_ind();

		if(f[cl_c] <= f[cl_c1] && f[cl_c] <= f[cl_c2]) {
			do_merge = true;
		}
//		else {
//			//second test
//			//if found center closer to optimum point than c1 and c2 then merge
//			//find minimum point
//			Matrix min_point = p.GetRows(f_.min_ind());
//			//find distances to minimum point from c1, c2 and c
//			expect <<= c & c_.GetRows(c1) & c_.GetRows(c2);
//			norm <<= (*_pNormFcn)(min_point, expect);
//			if(norm.min_ind() == 0)
//				do_merge = true;
//		}

		if(do_merge) {
			//mark pair as merged
			mc.insert(c1); mc.insert(c2);
			//save new center
			new_c &= c;

			//cout << "Centers left: " << c_.row_num();
			//res = true;
			//return res;
		}
	}

	if(mc.size() == 0) return false;

	//remove merged centers
	for(merged_idx::const_iterator pc = mc.begin(), end = mc.end(); pc != end; ++pc) {
		c_.DelRows(*pc);
	}
	//add discovered centers
	c_ &= new_c;

	//merge centers
	//if(c1 < c2) {
	//	c_.DelRows(c2); c_.DelRows(c1);
	//}
	//else {
	//	c_.DelRows(c1); c_.DelRows(c2);
	//}
	//c_ &= c;
	return true;
}

void kmeans::kmeans_impl::find_clusters(const Matrix& data, ulong clust_num, ulong maxiter,
						   bool skip_online, const Matrix* pCent, bool use_prev_cent)
{
	//set initial centers
	seed(data, clust_num, pCent, use_prev_cent);
	//do kmeans
	batch_phase(maxiter);

	if(c_.row_num() > 1 && !skip_online) {
		_pOrderFcn = &kmeans_impl::simple_shuffle;
		online_phase(maxiter);
	}
}

void kmeans::kmeans_impl::find_clusters_f(const Matrix& data, const Matrix& f, ulong clust_num,
							 ulong maxiter, const Matrix* pCent, bool use_prev_cent)
{
	//set initial centers
	seed(data, clust_num, pCent, use_prev_cent);
	f_ = f;
	//do batch phase to minimize norm sum
	//batch_phase(maxiter);

	//calc initial centers-winners for each data point
	/*
	Matrix dv, norm;
	for(ulong i = 0; i < data_.row_num(); ++i) {
		dv <<= data_.GetRows(i);
		(this->*pNormFcn)(dv, c_, norm);
		//norms_.SetRows(norm, i);
		w_[i] = norm.ElementInd(norm.Min());
		norms_[i] = norm[w_[i]];
	}
	*/

	//now start main online phase
	//pDerivFcn = &kmeans_impl::l2f_deriv;
	_pOrderFcn = &kmeans_impl::selection_based_order;
	_pBUFcn = &kmeans_impl::selection_batch_update;

	ulong i = 0;
	do {
		//cout << "find_clusters_f: iteration " << i << " started" << endl;
		batch_phase(maxiter);
		cout << "find_clusters_f: iteration " << i << " finished" << endl;
		cout << "centers dump (" << c_.row_num() << "):" << endl;
		c_.Print(cout);
		//do join phase
		if(join_phase(maxiter)) {
			//spin until there are no merges left
			while(join_phase(maxiter)) {};
		}
		else	//no merges - exit
			break;
		++i;
	} while(join_phase(maxiter));
	//ensure correct center locations after last merging
	//calc_winners(*this);
	//(this->*_pBUFcn)();
	//batch_phase(maxiter);
}

template<class dist_buf>
double kmeans::kmeans_impl::_dist_ij(const dist_buf& db, ulong i, ulong j)
{
	return 0;
}

Matrix kmeans::kmeans_impl::drops_homo(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter,
	double quant_mult)
{
	switch(opt_.norm_t) {
		default:
		case eucl_l2:
			_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
			_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
			_pBUFcn = &kmeans_impl::l2_batch_update;
			break;
	}

	//calc distance matrix
	Matrix dist;
	norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(data, dist);
	//calc drop radius
	const double drop_r = (ds.mean_ - ds.mse_)*quant_mult;
	//const double drop_r = 2*mind;

	//generate random initial drops ind
	ul_vec d_ind(static_cast<ulong>(drops_mult * data.row_num()));
	//create pool of free indicies
	ul_vec free_ind(data.row_num());
	for(ulong i = 0; i < d_ind.size(); ++i)
		free_ind[i] = i;
	random_shuffle(free_ind.begin(), free_ind.end(), prg::randIntUB);
	//generate initial points
	ulong ind;
	bool add_ok;
	for(ulong i = 0; i < d_ind.size(); ++i) {
		ind = 0;
		do {
			add_ok = true;
			for(ulong j = 0; j < i; ++j)
				if(dist(d_ind[j], free_ind[ind]) < drop_r) {
					if(++ind >= free_ind.size()) {
						ind = 0; add_ok = true;
					}
					else add_ok = false;
					break;
				}
		} while(!add_ok);
		//add this point
		d_ind[i] = free_ind[ind];
		free_ind.erase(free_ind.begin() + ind);
	}

	//main algorithm starts here
	Matrix dist_row;
	Matrix::indMatrix cl_ind;
	double minf;
	ulong cnt, moves_cnt;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		moves_cnt = 0;
		//find points within drop radius if any, but not less than 5
		for(ulong i = 0; i < d_ind.size(); ++i) {
			//find closest points
			dist_row <<= dist.GetRows(d_ind[i]);
			cl_ind <<= dist_row.RawSort();

			//find which of closest points have lowest function value
			ind = d_ind[i]; minf = f[ind];
			for(ulong j = 1; j < cl_ind.size(); ++j) {
				if(dist_row[j] > drop_r && j >= 5) continue;
				if(f[cl_ind[j]] < minf) {
					ind = cl_ind[j];
					minf = f[ind];
				}
			}

			//move current drop to found point
			if(d_ind[i] != ind) ++moves_cnt;
			d_ind[i] = ind;
		}

		//no moves - exit
		if(moves_cnt == 0) break;

		//test if some drops are too close to other - merge
		cl_ind.Resize(1, d_ind.size() - 1);
		for(ulong i = 0; i < d_ind.size(); ++i) {
			//find drops to merge && drop with lowest function value
			cnt = 0; ind = d_ind[i]; minf = f[ind];
			for(ulong j = 0; j < d_ind.size(); ++j) {
				if(j == i) continue;
				if(dist(d_ind[i], d_ind[j]) <= drop_r) {
					cl_ind[cnt++] = j;
					if(f[d_ind[j]] < minf) {
						ind = d_ind[j];
						minf = f[ind];
					}
				}
			}

			//merge drops
			d_ind[i] = ind;
			//cl_ind.RawSort();
			//sort(cl_ind.begin(), cl_ind.begin() + merge_cnt);
			for(ulong j = cnt - 1; j < cnt; --j)
				d_ind.erase(d_ind.begin() + cl_ind[j]);
		}
	}

	//return drops positions
	c_.NewMatrix(d_ind.size(), data.col_num());
	for(ulong i = 0; i < d_ind.size(); ++i)
		c_.SetRows(data.GetRows(d_ind[i]), i);

	aff_.resize(data.row_num());
	//calc affilations
	calc_winners(*this);

	return c_;
}

Matrix kmeans::kmeans_impl::drops_hetero(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter,
	double quant_mult)
{
	return drops_hetero_map(data, f, drops_mult, maxiter);

	switch(opt_.norm_t) {
		default:
		case eucl_l2:
			_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
			_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
			_pBUFcn = &kmeans_impl::l2_batch_update;
			break;
	}

	//save data internally
	data_ = data;

	const double dim = data.col_num();
	const double rev_dim = 1.0/dim;

	//calc distance matrix
	Matrix dist;
	norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(data, dist);
	//calc drop radius
	const double quant = (ds.mean_ - ds.mse_)*quant_mult;
	//const double quant = 2*mind;
	const double quant_v = pow(quant, dim);

	//create pool of free indicies
	ul_vec free_ind(data.row_num());
	for(ulong i = 0; i < free_ind.size(); ++i)
		free_ind[i] = i;
	random_shuffle(free_ind.begin(), free_ind.end(), prg::randIntUB);

	//randomly generate initial drops positions
	ul_vec d_ind(static_cast<ulong>(drops_mult * data.row_num()));
	//generate initial points

	copy(free_ind.begin(), free_ind.begin() + d_ind.size(), d_ind.begin());

	//drops radiuses & drops count
	vector<double> drop_r(d_ind.size(), quant);
	ul_vec d_cnt(d_ind.size(), 1);
	//drop_r = quant;
	//fill(d_cnt.begin(), d_cnt.end(), 1);

	//disabled - COMPILER BUG???
	/*
	bool add_ok;
	for(ulong i = 0; i < d_ind.size(); ++i) {
		ind = 0;
		do {
			add_ok = true;
			for(ulong j = 0; j < i; ++j) {
				if(dist(d_ind[j], free_ind[ind]) < drop_r[i]) {
					if(++ind >= free_ind.size()) {
						ind = 0; add_ok = true;
					}
					else add_ok = false;
					break;
				}
			}
		} while(!add_ok);

		//add this point
		d_ind[i] = free_ind[ind];
		//at starting moment there is one drop in each point
		d_cnt[i] = 1;
		drop_r[i] = quant;

		free_ind.erase(free_ind.begin() + ind);
	}
	*/

	//main algorithm starts here
	Matrix dist_row;
	Matrix::indMatrix cl_ind;
	double minf;
	ulong ind, cnt, moves_cnt;
	ul_vec::iterator p_drop;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		moves_cnt = 0;
		for(ulong i = 0; i < d_ind.size(); ++i) {
			//find closest points
			dist_row <<= dist.GetRows(d_ind[i]);
			cl_ind <<= dist_row.RawSort();

			//find which of closest points have lowest function value
			ind = d_ind[i]; minf = f[ind];
			for(ulong j = 1; j < cl_ind.size(); ++j) {
				if(dist_row[j] > drop_r[i] && j >= 5) break;
				if(f[cl_ind[j]] < minf) {
					ind = cl_ind[j];
					minf = f[ind];
				}
			}
			//cnt = min(cl_ind.size(), max(cnt, 5));
			if(d_ind[i] == ind) continue;

			//move current drop to found point
			//increase moves count
			++moves_cnt;

			//search if there is a drop in a new point
			p_drop = find(d_ind.begin(), d_ind.end(), ind);
			//update source drop
			if(--d_cnt[i] > 0)
				drop_r[i] = pow(quant_v * d_cnt[i], rev_dim);
			if(p_drop != d_ind.end()) {
				//there is a drop in a new point - update destination drop
				ind = p_drop - d_ind.begin();
				//update quants count & radius of destination drop
				++d_cnt[ind];
				drop_r[ind] = pow(quant_v * d_cnt[ind], rev_dim);
			}
			else {
				//new point is clear
				if(d_cnt[i] == 0) {
					//move drop to new point
					d_ind[i] = ind;
					++d_cnt[i];
				}
				else {
					//if there are many drops in this point - move only one to new location
					d_ind.push_back(ind);
					d_cnt.push_back(1);
					drop_r.push_back(quant);
				}
			}
		}

		//erase empty drops
		for(ulong i = d_ind.size() - 1; i < d_ind.size(); --i) {
			if(d_cnt[i] == 0) {
				d_ind.erase(d_ind.begin() + i);
				d_cnt.erase(d_cnt.begin() + i);
				drop_r.erase(drop_r.begin() + i);
			}
		}

		/*
		//test if some drops are in one point - merge
		cl_ind.Resize(1, d_ind.size() - 1);
		//double new_v;
		for(ulong i = 0; i < d_ind.size(); ++i) {
			//find dublicating drops to merge
			cnt = 0; ind = d_ind[i]; //new_v = 0;
			for(ulong j = 0; j < d_ind.size(); ++j) {
				if(j == i) continue;
				if(ind == d_ind[j]) {
					cl_ind[cnt++] = j;
					d_cnt[i] += d_cnt[j];
					//new_v += quant_v * d_cnt[j];
				}
			}

			//update drop radius
			if(cnt > 0)
				drop_r[i] = pow(quant_v * d_cnt[i], rev_dim);

			//remove dublicates
			for(ulong j = cnt - 1; j < cnt; --j) {
				d_ind.erase(d_ind.begin() + cl_ind[j]);
				d_cnt.erase(d_cnt.begin() + cl_ind[j]);
				drop_r.DelColumns(cl_ind[j]);
			}
		}
		*/

		//no moves - exit
		if(moves_cnt == 0) break;

		//DEBUG - dump current centers positions
		Matrix d_pos(d_ind.size(), data.col_num());
		for(ulong i = 0; i < d_ind.size(); ++i)
			d_pos.SetRows(data.GetRows(d_ind[i]), i);
		DumpMatrix(d_pos, "centers.txt");
	}

	//save drops positions
	c_.NewMatrix(d_ind.size(), data.col_num());
	for(ulong i = 0; i < d_ind.size(); ++i)
		c_.SetRows(data.GetRows(d_ind[i]), i);

	//initialization of arrays with results
	aff_.resize(c_.row_num());
	//mem reserve
	for(ulong i = 0; i < c_.row_num(); ++i)
		aff_[i].reserve(data.row_num());

	w_.NewMatrix(1, data.row_num());
	norms_.NewMatrix(1, data.row_num());

	//calc affilations
	calc_winners(*this);

	return c_;
}

Matrix kmeans::kmeans_impl::drops_hetero_map(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter,
	double quant_mult)
{
	switch(opt_.norm_t) {
		default:
		case eucl_l2:
			_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
			_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
			_pBUFcn = &kmeans_impl::l2_batch_update;
			break;
	}

	//save data internally
	data_ = data;

	const double dim = data.col_num();
	const double rev_dim = 1.0/dim;

	//calc distance matrix
	Matrix dist;
	norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(data, dist);
	//calc drop radius
	const double quant = ds.mean_nn_; //(md - qd)*quant_mult;
	//const double quant = 2*mind;
	const double quant_v = pow(quant, dim);

	Matrix dv, norm, _data, dist_row;
	//drops containing map instantiation
	drops_map drops;
	//randomly select drops positions
	ulong init_cnt = static_cast<ulong>(drops_mult * data.row_num());
	dv.NewMatrix(1, dim);
	_data <<= data.minmax(); dist_row <<= _data.GetRows(1) - _data.GetRows(0);
	_data <<= _data.GetRows(0);
	ulong ind;
	for(ulong i = 0; i < init_cnt; ++i) {
		generate(dv.begin(), dv.end(), prg::rand01);
		dv *= dist_row; dv += _data;
		norm <<= (*_pNormFcn)(dv, data);
		ind = norm.min_ind();
		++drops[ind].qcnt_;
	}
#ifdef _DEBUG
	//DEBUG - dump current centers positions
	Matrix d_pos(drops.size(), data.col_num());
#endif
	//now calc drops radiuses
	for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
		p_drop->second.r_ = pow(quant_v * p_drop->second.qcnt_, rev_dim);
	}

	//main algorithm starts here
	Matrix::indMatrix cl_ind;
	double minf, R;
	ulong moves_cnt, mov_q; //mov_q - number of moving quants
	drops_iterator p_empty;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		moves_cnt = 0;
		for(drops_iterator p_drop = drops.begin(); p_drop != drops.end(); ) {
			//find closest points
			dist_row <<= dist.GetRows(p_drop->first);
			cl_ind <<= dist_row.RawSort();

			//find which of closest points have lowest function value
			ind = p_drop->first; minf = f[ind]; R = 0;
			for(ulong j = 1; j < cl_ind.size(); ++j) {
				if(dist_row[j] > p_drop->second.r_ && j >= 5) break;
				if(f[cl_ind[j]] < minf) {
					ind = cl_ind[j];
					minf = f[ind];
					R = dist_row[j];
				}
			}
			//if drop isn't moving - jump to next
			if(p_drop->first == ind) {
				++p_drop; continue;
			}

			//move current drop to found point
			//increase moves count
			++moves_cnt;

			//update source drop
			mov_q = p_drop->second.qcnt_;
			if(R < p_drop->second.r_)
				//calc how many drops to move
				mov_q = min(mov_q, mov_q - static_cast<ulong>(floor(pow(R/quant, dim))) + 1);
			p_drop->second.qcnt_ -= mov_q;

			if(p_drop->second.qcnt_ > 0) {
				p_drop->second.r_ = pow(quant_v * p_drop->second.qcnt_, rev_dim);
				++p_drop;
			}
			else {
				p_empty = p_drop; ++p_drop;
				drops.erase(p_empty);
			}

			//update destination drop
			_drop_params& dst_drop = drops[ind];
			dst_drop.qcnt_ += mov_q;
			dst_drop.r_ = pow(quant_v * dst_drop.qcnt_, rev_dim);
		}

		//verbose print
		cout << "kmeans.drops_hetero_map iteration " << cycle_ << ", moves_cnt = " << moves_cnt;
		cout << ", centers_cnt = " << drops.size() << endl;

		//no moves - exit
		if(moves_cnt == 0) break;

#ifdef _DEBUG
		ind = 0;
		for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
			d_pos.SetRows(data.GetRows(p_drop->first), ind);
			++ind;
		}
		DumpMatrix(d_pos, "centers.txt");
#endif
	}

	//save drops positions
	c_.NewMatrix(drops.size(), data.col_num());
	ind = 0;
	for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
		c_.SetRows(data.GetRows(p_drop->first), ind);
		++ind;
	}

	//initialization of arrays with results
	aff_.resize(c_.row_num());
	//mem reserve
	for(ulong i = 0; i < c_.row_num(); ++i)
		aff_[i].reserve(data.row_num());

	w_.NewMatrix(1, data.row_num());
	norms_.NewMatrix(1, data.row_num());

	//calc affilations
	calc_winners(*this);

	return c_;
}

void kmeans::kmeans_impl::gen_uniform_cent(const Matrix& bound, const Matrix& dim, Matrix& dv,
										   drops_map& drops, const double quant)
{
	const ulong dim_ind = dv.size() - dim.size();
	const bool last_dim = (dim.size() == 1);
	const Matrix sub_dim = dim.GetColumns(dim_ind + 1, -1);

	Matrix norm;
	ulong drop_ind;
	dv[dim_ind] = bound[dim_ind] + quant/2;
	for(ulong i = 0; i < dim[0]; ++i) {
		if(last_dim) {
			norm <<= (*_pNormFcn)(dv, data_);
			drop_ind = norm.min_ind();
			++drops[drop_ind].qcnt_;
		}
		else
			gen_uniform_cent(bound, sub_dim, dv, drops, quant);

		//make new position
		dv[dim_ind] += quant;
	}
}

Matrix kmeans::kmeans_impl::drops_hetero_simple(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter,
	double quant_mult)
{
	switch(opt_.norm_t) {
		default:
		case eucl_l2:
			_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
			_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
			_pBUFcn = &kmeans_impl::l2_batch_update;
			break;
	}

	//save data internally
	data_ <<= data;

	//some constants
	const ulong dim = data.col_num();
	const ulong points_num = data.row_num();
	const double rev_dim = 1.0/dim;

	//calc distance matrix
	Matrix dist;
	norm_tools::dist_stat ds = norm_tools::calc_dist_matrix< norm_tools::l2 >(data, dist);
	//calc drop radius
	const double quant = ds.mean_nn_; //(md - qd)*quant_mult;
	_drop_params::init_r_ = quant;
	//const double quant = 2*mind;
	//const double quant_v = pow(quant, dim);

	Matrix dv, norm, _data, dist_row;
	//drops containing map instantiation
	drops_map drops;

	ulong init_cnt, ind;
	//calculate based on given ratio
	init_cnt = static_cast<ulong>(drops_mult * points_num);

	//heuristic calculation assuming uniform coverage by drops
	//calc drops count
	_data <<= data.minmax();
	init_cnt = 1;
	Matrix dcnt_bydim(1, dim);
	for(ulong i = 0; i < dim; ++i) {
		dist_row <<= _data.GetColumns(i);
		dcnt_bydim[i] = ceil((dist_row[1] - dist_row[0])/quant);
		init_cnt *= dcnt_bydim[i];
	}

	//select drops positions
	dv.NewMatrix(1, dim);

	//random selection
	//dist_row <<= _data.GetRows(1) - _data.GetRows(0);
	//_data <<= _data.GetRows(0);
	//for(ulong i = 0; i < init_cnt; ++i) {
	//	generate(dv.begin(), dv.end(), prg::rand01);
	//	dv *= dist_row; dv += _data;
	//	(this->*_pNormFcn)(dv, data, norm);
	//	ind = norm.min_ind();
	//	++drops[ind].qcnt_;
	//}

	//uniform selection
	gen_uniform_cent(_data.GetRows(0), dcnt_bydim, dv, drops, quant);

	//now calc drops radiuses
	//for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
	//	p_drop->second.r_ = quant;
	//}

	//main algorithm starts here
	Matrix::indMatrix cl_ind;
	double minf, R;
	ulong moves_cnt, mov_q; //mov_q - number of moving quants
	drops_iterator p_empty;
	for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
		moves_cnt = 0;
		for(drops_iterator p_drop = drops.begin(); p_drop != drops.end(); ) {
			//find closest points
			dist_row <<= dist.GetRows(p_drop->first);
			cl_ind <<= dist_row.RawSort();

			//find which of closest points have lowest function value
			ind = p_drop->first; minf = f[ind]; R = 0;
			for(ulong j = 1; j < cl_ind.size(); ++j) {
				if(dist_row[j] > p_drop->second.r_) break;
				if(f[cl_ind[j]] < minf) {
					ind = cl_ind[j];
					minf = f[ind];
					R = dist_row[j];
					break;
				}
			}
			//if drop isn't moving - jump to next
			if(p_drop->first == ind) {
				++p_drop; continue;
			}

			//move current drop to found point
			//increase moves count
			++moves_cnt;

			//update destination drop
			_drop_params& dst_drop = drops[ind];
			dst_drop.qcnt_ += p_drop->second.qcnt_;
			//dst_drop.r_ = quant;

			//delete source drop
			p_empty = p_drop; ++p_drop;
			drops.erase(p_empty);

			//dst_drop.r_ = pow(quant_v * dst_drop.qcnt_, rev_dim);
		}

		//verbose print
		cout << "kmeans.drops iteration " << cycle_ << "; moves_cnt = " << moves_cnt;
		cout << ", drops_cnt = " << drops.size() << endl;

		//no moves - exit
		if(moves_cnt == 0) break;

#ifdef _DEBUG
		ind = 0;
		c_.NewMatrix(drops.size(), dim);
		for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
			c_.SetRows(data.GetRows(p_drop->first), ind);
			++ind;
		}
		DumpMatrix(c_, "centers.txt");
#endif
	}

	//save drops positions
	c_.NewMatrix(drops.size(), data.col_num());
	ind = 0;
	for(drops_iterator p_drop = drops.begin(), end = drops.end(); p_drop != end; ++p_drop) {
		c_.SetRows(data.GetRows(p_drop->first), ind);
		++ind;
	}

	//initialization of arrays with results
	aff_.resize(c_.row_num());
	//mem reserve
	for(ulong i = 0; i < c_.row_num(); ++i)
		aff_[i].reserve(points_num);

	w_.NewMatrix(1, points_num);
	norms_.NewMatrix(1, points_num);

	//calc affilations
	calc_winners(*this);

	return c_;
}

void kmeans::kmeans_impl::restart(const Matrix& data, ulong clust_num, ulong maxiter,
					 bool skip_online, const Matrix* pCent, bool use_prev_cent)
{
	Matrix cur_c;
	if(pCent) cur_c &= *pCent;
	cur_c &= c_;
	if(clust_num > 0)
		seed(data, clust_num, &cur_c, use_prev_cent);
	else
		seed(data, cur_c.row_num(), &cur_c, use_prev_cent);

	//do kmeans
	batch_phase(maxiter);
	if(!skip_online && c_.row_num() > 1) {
		_pOrderFcn = &kmeans_impl::simple_shuffle;
		online_phase(maxiter);
	}
}

//---------------------------- kmeans  implementation ----------------------------------------------
kmeans::kmeans() : pimpl_(new kmeans_impl(opt_))
{
	set_def_opt();
}

void kmeans::set_def_opt()
{
	opt_.set_def_opt();
	pimpl_->_pNormFcn = &norm_tools::vm_norm< norm_tools::l2 >;
	pimpl_->_pDerivFcn = &norm_tools::deriv< norm_tools::l2 >;
}

const Matrix& kmeans::get_centers() const {
	return pimpl_->c_;
}

const ulMatrix& kmeans::get_ind() const {
	return pimpl_->w_;
}

const Matrix& kmeans::get_norms() const {
	return pimpl_->norms_;
}

const kmeans::vvul& kmeans::get_aff() const {
	return pimpl_->aff_;
}

void kmeans::find_clusters(const Matrix& data, ulong clust_num, ulong maxiter,
						   bool skip_online, const Matrix* pCent, bool use_prev_cent)
{
	pimpl_->find_clusters(data, clust_num, maxiter, skip_online, pCent, use_prev_cent);
}

void kmeans::find_clusters_f(const Matrix& data, const Matrix& f, ulong clust_num,
							 ulong maxiter, const Matrix* pCent, bool use_prev_cent)
{
	pimpl_->find_clusters_f(data, f, clust_num, maxiter, pCent, use_prev_cent);
}

void kmeans::restart(const Matrix& data, ulong clust_num, ulong maxiter,
					 bool skip_online, const Matrix* pCent, bool use_prev_cent)
{
	pimpl_->restart(data, clust_num, maxiter, skip_online, pCent, use_prev_cent);
}

Matrix kmeans::drops_homo(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter, double quant_mult)
{
	return pimpl_->drops_homo(data, f, drops_mult, maxiter, quant_mult);
}

Matrix kmeans::drops_hetero(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter, double quant_mult)
{
	return pimpl_->drops_hetero(data, f, drops_mult, maxiter, quant_mult);
}

Matrix kmeans::drops_hetero_map(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter, double quant_mult)
{
	return pimpl_->drops_hetero_map(data, f, drops_mult, maxiter, quant_mult);
}

Matrix kmeans::drops_hetero_simple(const Matrix& data, const Matrix& f, double drops_mult, ulong maxiter, double quant_mult)
{
	return pimpl_->drops_hetero_simple(data, f, drops_mult, maxiter, quant_mult);
}

} //namespace KM
