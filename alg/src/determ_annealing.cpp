#include "determ_annealing.h"
#include "ga.h"
#include "m_algorithm.h"
#include "prg.h"

#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

using namespace GA;
using namespace prg;
using namespace std;

namespace DA {

typedef vector< vector< ulong > > vvul;

//========================== implementation of deterministic annealing =================================================
//names of variables follow original notation
struct da_data {
	//source points
	Matrix x_;
	//codevectors
	Matrix y_;
	//probabilities p(y)
	Matrix p_y;
	//probabilities p(y|x)
	Matrix p_yx;
	//temperature
	double T_;
};

struct hard_clusters_data {
	//winners matrix - point->center relation
	//ulMatrix w_;
	//points affiliation - center->points collection
	vvul aff_;
	//sum of distances from points to corresponding cluster centers
	double e_;

	hard_clusters_data() : e_(0) {}
};

struct cov_data {
	Matrix cov_;
	double variance_;
};

class determ_annealing::da_impl : public da_data {
	//---------------------------- pat_sel implementation ----------------------------------------------
	class pat_sel
	{
		typedef Matrix::indMatrix indMatrix;

		indMatrix _selection(const Matrix& f)
		{
			Matrix expect;

			expect <<= ga_.ScalingCall(f);
			return ga_.SelectionCall(expect, expect.size());

		}

	public:
		ga ga_;

		pat_sel(const da_impl& da) //: km_(km)
		{
			ga_.ReadOptions("da.ini");
			//km.opt_.add_embopt(ga_.opt_);
			//set_def_opt();
			ga_.prepare2run(false);
		}
		~pat_sel() {};

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

	pat_sel& get_ps() const {
		static pat_sel ps(*this);
		return ps;
	};

public:

	//probability threshold for making hard clusters - see calc_winners
	double prob_thresh_;
	double patience_;
	double alfa_;
	double Tmin_;
	ulong patience_cycles_;

	hard_clusters_data hcd_;
	//function values
	Matrix f_;
	ulong cycle_;

	double (*norm_fcn_)(const Matrix&, const Matrix&);
	double (*norm2_fcn_)(const Matrix&, const Matrix&);
	Matrix (*deriv_fcn_)(const Matrix&, const Matrix&);
	void (da_impl::*order_fcn_)(ulong, ul_vec&) const;

	da_impl()
		: prob_thresh_(1/3), alfa_(0.9), Tmin_(0.5), patience_(0.01), patience_cycles_(10)
	{}

	void calc_winners(const da_data& dad, hard_clusters_data& hcd) const
	{
		Matrix dv;

		//clear affiliation list
		hcd.aff_.clear();
		hcd.aff_.resize(dad.y_.row_num());
		//for(ulong i = 0; i < aff_.size(); ++i)
		//	aff_[i].clear();

		hcd.e_ = 0;
		ulong cnt = 0;
		//calc affiliation for each center
		for(ulong i = 0; i < dad.y_.row_num(); ++i) {
			dv <<= dad.p_yx.GetRows(i);
			ul_vec& cur_aff = hcd.aff_[i];
			for(ulong j = 0; j < dv.size(); ++j) {
				if(dv[j] > prob_thresh_) {
					cur_aff.push_back(j);
					hcd.e_ += (*norm_fcn_)(dad.x_.GetRows(j), dad.y_.GetRows(i));
					++cnt;
				}
			}
		}
		hcd.e_ /= cnt;
	}

	bool patience_check(ulong cycle, double e)
	{
		static ulong ni_cycles = 0;
		static double preve = 0;

		if(cycle == 0 || preve - e > patience_ * preve) {
			ni_cycles = 0;
			preve = e;
		}
		else if(++ni_cycles >= patience_cycles_)
			return true;

		return false;
	}

	//norms used - currently only l2
	static double l2_norm(const Matrix& x1, const Matrix& x2) {
		return (x1 - x2).norm2();

//		norm.Resize(1, points.row_num());
//		//Matrix diff;
//		for(ulong i = 0; i < points.row_num(); ++i) {
//			norm[i] = (dv - points.GetRows(i)).norm2();
//		}
	}

	static double l2_norm2(const Matrix& x1, const Matrix& x2) {
		Matrix diff = x1 - x2;
		return diff.Mul(diff).Sum();

//		norm.Resize(1, points.row_num());
//		//Matrix diff;
//		for(ulong i = 0; i < points.row_num(); ++i) {
//			norm[i] = (dv - points.GetRows(i)).norm2();
//		}
	}

	//norm function derivatives
	static Matrix l2_deriv(const Matrix& dv, const Matrix& center) {
		return (dv - center);
	}


	//patterns order processing
	void conseq_order(ulong cv_ind, ul_vec& porder) const
	{
		porder.clear();
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		for(ul_vec::const_iterator pos = aff_cv.begin(); pos != aff_cv.end(); ++pos)
			porder.push_back(aff_cv[*pos]);

		//random_shuffle(porder.begin(), porder.end(), prg::randIntUB);
	}

	//resulting porder will contain ABSOLUTE indexes applicable directly to da_data::x_
	void selection_based_order(ulong cv_ind, ul_vec& porder) const
	{
		//first of all determine hard clusters
		//calc_winners(*this, hcd_);
		//assume that hard affiliation is already calculated
		if(cv_ind >= hcd_.aff_.size()) return;

		//select f values belonging to given cluster
		Matrix f_cv;
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		for(ul_vec::const_iterator pos = aff_cv.begin(); pos != aff_cv.end(); ++pos)
			f_cv.push_back(f_[*pos]);

		//now do a selection
		ulMatrix sel_ind = get_ps().selection(f_cv);
		//translate to absolute indexes and store them in porder
		porder.clear();
		for(ulMatrix::cr_iterator pos = sel_ind.begin(), end = sel_ind.end(); pos != end; ++pos)
			porder.push_back(aff_cv[*pos]);

		//copy(sel_ind.begin(), sel_ind.end(), back_inserter(porder));

		//shuffle order obtained
		//random_shuffle(porder.begin(), porder.end(), prg::randIntUB);
	}

	double calc_var(const da_data& dad, ulong cv_ind) const {
		//assume that hard affiliation is already calculated
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		if(cv_ind >= aff_cv.size()) return 0;

		//collect data vectors belonging to given cluster
		const Matrix y = dad.y_.GetRows(cv_ind);
		//const double mult = 1.0 / aff_cv.size();
		Matrix t;
		for(ulong i = 0; i < aff_cv.size(); ++i)
			t &= dad.x_.GetRows(aff_cv[i]) - y;

		//calculate variance along principal axis
		//first calc SVD
		Matrix U, E, V;
		svd(!t, U, E, V);
		//we need only maximum singular value
		double var = E.Max();
		return var * var / aff_cv.size();
	}

	void update_epoch() {
		ul_vec x_order, uniq_xor;
		Matrix::r_iterator cur_py = p_y.begin(), cur_pyx;
		Matrix x, y;

		hcd_.e_ = 0;
		//main cycle starts here - update all available centers
		for(ulong i = 0; i < y_.row_num(); ++i, ++cur_py) {
			y <<= y_.GetRows(i);
			//get patterns presentation order
			(this->*order_fcn_)(0, x_order);
			//find unique elements
			sort(x_order.begin(), x_order.end());
			uniq_xor.clear();
			unique_copy(x_order.begin(), x_order.end(), back_inserter(uniq_xor));

			//update probabilities p(y[i] | x)
			cur_pyx = p_yx.begin() + i * x_.col_num();
			for(ul_vec::const_iterator pat = uniq_xor.begin(), pat_end = uniq_xor.end(); pat != pat_end; ++pat) {
				x <<= x_.GetRows(*pat);
				//first calc denominator for further calculation of p_yx
				double pyx_den = 0;
				for(ulong j = 0; j < y_.row_num(); ++j)
					//pyx_den += p(y[j]) * exp(-||x - y[j]||^2 / T)
					pyx_den += p_y[j] * exp(-(*norm2_fcn_)(x, y_.GetRows(j)) / T_);

				//update p(y[i] | x) = p(y[i]) * exp(-||x - y[i]||^2 / T)
				cur_pyx[*pat] = p_y[i] * exp(-(*norm2_fcn_)(x, y) / T_) / pyx_den;
			}

			//update probabilities p(y[i]) & calculate new center position
			//go through all patterns selected for that cluster
			p_y[i] = 0;
			y = 0;
			for(ul_vec::const_iterator pat = x_order.begin(), pat_end = x_order.end(); pat != pat_end; ++pat) {
				x <<= x_.GetRows(*pat);
				y += x * cur_pyx[*pat];
				p_y[i] += cur_pyx[*pat];
			}
			//complete calculations
			p_y[i] /= x_order.size();
			y /= p_y[i] * x_order.size();
			//update error
			hcd_.e_ += (*norm_fcn_)(y, y_.GetRows(i));
			//save new center position
			y_.SetRows(y, i);
		}	//end of centers loop

		//complete mse calculation
		hcd_.e_ /= y_.row_num();
	}

	void phase_transition_epoch() {
		//main cycle starts here - check all available centers
		double var;
		Matrix y;
		Matrix perturb(1, y_.col_num());
		ulong cnt = y_.row_num();
		for(ulong i = 0; i < cnt; ++i) {
			y <<= y_.GetRows(i);
			//calc variance for current center
			var = calc_var(*this, i);
			//if current T below critical - add new center
			if(T_ < 2 * var) {
				//generate perturbation
				generate(perturb.begin(), perturb.end(), prg::rand01);
				perturb -= 0.5; perturb *= y.norm2() * 0.05;

				//add new center
				y_ &= y + perturb;
				p_y[i] *= 0.5;
				p_y.push_back(p_y[i], false);
				p_yx.Resize(y_.row_num());
			}
		}
	}

	//clusterization using deterministic annealing
	void find_clusters(const Matrix& data, ulong clust_num, ulong maxiter,
					   bool skip_online, const Matrix* pCent, bool use_prev_cent)
	{
		order_fcn_ = &da_impl::selection_based_order;
		norm_fcn_ = &da_impl::l2_norm;
		norm2_fcn_ = &da_impl::l2_norm2;

		//initialization
		x_ = data;
		p_y.Resize(1, 1);
		p_yx.Resize(1, x_.row_num());
		//set initial probabilities to 1
		p_y = 1; p_yx = 1;
		//calc first hard cluster
		calc_winners(*this, hcd_);
		//get patterns presentation order
		ul_vec x_order;
		(this->*order_fcn_)(0, x_order);
		//set position of the first center
		double mult = 1.0 / x_order.size();
		for(ulong i = 0; i < x_order.size(); ++i)
			y_ += x_.GetRows(x_order[i]) * mult;
		//set initial T > 2 * max variation along principal axis of all data
		T_ = calc_var(*this, 0) * 2;
		T_ += T_ * 0.1;

		//main cycle starts here
		for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
			for(ulong i = 0; i < maxiter; ++i) {
				//update probabilities and centers positions
				update_epoch();
				//convergence test
				if(patience_check(i, hcd_.e_)) break;
			}

			//temperature test
			if(T_ <= Tmin_) T_ = 0;
			else if(T_ == 0) break;

			//cooling step
			T_ *= alfa_;

			//add new centers if nessessary
			if(y_.row_num() < clust_num)
				phase_transition_epoch();
		}	//end of main loop
	}

};


//========================================== determ_annealing implementation ===========================================
determ_annealing::determ_annealing()
{
	//ctor
}

determ_annealing::~determ_annealing()
{
	//dtor
}

}	//end of namespace DA
