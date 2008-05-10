#include "determ_annealing.h"
#include "ga.h"
#include "m_algorithm.h"
#include "prg.h"

#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#define EPS 0.000000001
#define T_EPS 0.0001
#define MERGE_EPS 0.0001

using namespace GA;
using namespace prg;
using namespace std;

namespace DA {

typedef determ_annealing::vvul vvul;

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
	//beta = 1/T
	double beta_;
	//variances of each cluster stored here
	Matrix var_;
};

//structure to track annealing process
struct da_hist {
	Matrix y_;
	Matrix p_y;
	double T_;

	da_hist(const da_data& dad) : y_(dad.y_), p_y(dad.p_y), T_(dad.T_) {}
};

struct hard_clusters_data {
	//winners matrix - point->center relation
	ulMatrix w_;
	Matrix norms_;
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

		Matrix scaling(const Matrix& f) {
			return ga_.ScalingCall(f);
		}

		Matrix selection_prob(const Matrix& f, ulong maxcycles = 100, ulong buf_size = 0) {
			//const ulong how_many = f.size(); //min(max(f.size(), (ulong)1000), (ulong)1000);
			if(buf_size == 0) buf_size = f.size();
			const double prob_quant = 1.0/buf_size;
			Matrix prob(1, f.size(), 0);
			Matrix new_prob(1, f.size(), 0);
			Matrix diff;
			ulMatrix sel_ind;
			double dist;
			bool stop_patience = false;
			for(ulong i = 0; i < maxcycles && !stop_patience; ++i) {
				//first of all invoke selection
				sel_ind <<= selection(f, buf_size);
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
	double alpha_, alpha_max_;
	double Tmin_;
	ulong patience_cycles_;

	hard_clusters_data hcd_;
	//function values
	Matrix f_;
	ulong cycle_;

	//annealing history
	vector< da_hist > hist_;

	double (*norm_fcn_)(const Matrix&, const Matrix&);
	double (*norm2_fcn_)(const Matrix&, const Matrix&);
	Matrix (*deriv_fcn_)(const Matrix&, const Matrix&);
	void (da_impl::*order_fcn_)(ulong, ul_vec&) const;
	const Matrix& (da_impl::*px_fcn_)(ulong) const;

	da_impl()
		: prob_thresh_(1.0/3), alpha_(0.9), alpha_max_(0.99), Tmin_(0.01), patience_(0.00001), patience_cycles_(10)
	{}

	void calc_winners(const da_data& dad, hard_clusters_data& hcd) const
	{
		hcd.aff_.clear();
		hcd.aff_.resize(dad.y_.row_num());

		//normalize threshold
		double cur_pt = prob_thresh_ / dad.y_.row_num();

		//move through all data points
		double cur_p, max_p;
		ulong winner;
		for(ulong i = 0; i < dad.x_.row_num(); ++i) {
			//iterate through centers
			for(ulong j = 0; j < dad.y_.row_num(); ++j) {
				cur_p = dad.p_yx(j, i);
				if(cur_p > cur_pt)
					hcd.aff_[j].push_back(i);
				if(j == 0 || cur_p > max_p) {
					max_p = cur_p;
					winner = j;
				}
			}

			//save winner
			hcd.w_[i] = winner;
			hcd.norms_[i] = (*norm_fcn_)(dad.x_.GetRows(i), dad.y_.GetRows(winner));
		}

//		hcd.e_ = 0;
//		ulong cnt = 0;
//		//calc affiliation for each center
//		for(ulong i = 0; i < dad.y_.row_num(); ++i) {
//			dv <<= dad.p_yx.GetRows(i);
//			ul_vec& cur_aff = hcd.aff_[i];
//			for(ulong j = 0; j < dv.size(); ++j) {
//				if(dv[j] > prob_thresh_) {
//					cur_aff.push_back(j);
//					hcd.e_ += (*norm_fcn_)(dad.x_.GetRows(j), dad.y_.GetRows(i));
//					++cnt;
//				}
//			}
//		}
//		hcd.e_ /= cnt;
	}

	//calc affiliation for particular cluster only
	void calc_aff(const da_data& dad, hard_clusters_data& hcd, ulong cv_ind) const
	{
		if(cv_ind >= hcd.aff_.size()) return;
		//clear affiliation list
		ul_vec& cur_aff = hcd.aff_[cv_ind];
		cur_aff.clear();

		//normalize threshold
		double cur_pt = prob_thresh_ / dad.y_.row_num();

		//move through all data points
		Matrix::cr_iterator cur_pyx = dad.p_yx.begin() + cv_ind * dad.x_.row_num();
		for(ulong i = 0; i < dad.x_.row_num(); ++i, ++cur_pyx) {
			//iterate through centers
			if(*cur_pyx > cur_pt)
				cur_aff.push_back(i);
		}
	}

	//calc winners based on nearest center
	void calc_winners_kmeans(const da_data& dad, hard_clusters_data& hcd) const {
		//clear affiliation list
		hcd.aff_.clear();
		hcd.aff_.resize(dad.y_.row_num());

		Matrix dv;
		hcd.e_ = 0;
		double tmp, min_dist;
		ulong winner;
		//calc centers-winners for each data point
		for(ulong i = 0; i < dad.x_.row_num(); ++i) {
			dv <<= dad.x_.GetRows(i);
			for(ulong j = 0; j < dad.y_.row_num(); ++j) {
				tmp = (*norm2_fcn_)(dv, dad.y_.GetRows(j));
				if(j == 0 || tmp < min_dist) {
					min_dist = tmp;
					winner = j;
				}
			}
			//save winner
			hcd.w_[i] = winner;
			//save affiliation
			hcd.aff_[winner].push_back(i);

			hcd.norms_[i] = min_dist;
			hcd.e_ += min_dist;
		}
		hcd.e_ /= dad.x_.row_num();
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
			porder.push_back(*pos);

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
			f_cv.push_back(f_[*pos], false);

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

	//p(x) generators
	//standard prob distribution where each p(x) = 1/N
	const Matrix& unifrom_px(ulong cv_ind) const {
		struct px {
			Matrix expect_;
			px(const Matrix& f) : expect_(1, f.size()) {
				expect_ = 1.0/(double)f.size();
			}
		};
		static px p(f_);
		return p.expect_;
	}

	//p(x) distribution based on static GA.ScalingCall calculated only when first time called
	const Matrix& scaling_px(ulong cv_ind) const {
		static Matrix expect(get_ps().scaling(f_));
		return expect;
	}

	//p(x) distribution based dynamically calculated expectation for given cluster from it's affiliation
	const Matrix& scaling_px_aff(ulong cv_ind) const {
		static Matrix expect(1, f_.size());

		//assume that hard affiliation is already calculated
		if(cv_ind >= hcd_.aff_.size()) return scaling_px(cv_ind);

		//select f values belonging to given cluster
		Matrix f_cv;
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		for(ul_vec::const_iterator pos = aff_cv.begin(); pos != aff_cv.end(); ++pos)
			f_cv.push_back(f_[*pos], false);

		//calc expectation
		Matrix exp_cv = get_ps().scaling(f_cv);

		//make full expectation matrix
		expect = 0;
		for(ulong i = 0; i < exp_cv.size(); ++i) {
			expect[aff_cv[i]] = exp_cv[i];
		}
		return expect;
	}

	//p(x) distribution based on estimated selection probabilities
	const Matrix& selection_px(ulong cv_ind) const {
		static Matrix expect(get_ps().selection_prob(f_, 100, 1000));
		return expect;
	}

	const Matrix& selection_px_aff(ulong cv_ind) const {
		static Matrix expect(1, f_.size());

		//assume that hard affiliation is already calculated
		if(cv_ind >= hcd_.aff_.size()) return scaling_px(cv_ind);

		//select f values belonging to given cluster
		Matrix f_cv;
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		for(ul_vec::const_iterator pos = aff_cv.begin(); pos != aff_cv.end(); ++pos)
			f_cv.push_back(f_[*pos], false);

		//calc expectation
		Matrix exp_cv = get_ps().selection_prob(f_cv);

		//make full expectation matrix
		expect = 0;
		for(ulong i = 0; i < exp_cv.size(); ++i) {
			expect[aff_cv[i]] = exp_cv[i];
		}
		return expect;
	}

	//order of patterns to p(x) distribution converter
	const Matrix& order2px(ulong cv_ind) const {
		static Matrix expect(1, f_.size());

		//calc affiliation for given center
		//calc_aff(*this, hcd_, cv_ind);

		//assume that hard affiliation is already calculated
		//build patterns order
		ul_vec order;
		(this->*order_fcn_)(cv_ind, order);

		//convert order to expectation
		expect = 0;
		const double p_quant = 1.0/order.size();
		for(ulong i = 0; i < order.size(); ++i) {
			expect[order[i]] += p_quant;
		}
		return expect;
	}

	double calc_var(const da_data& dad, ulong cv_ind) const {
		//assume that hard affiliation is already calculated
		if(cv_ind >= hcd_.aff_.size()) return 0;
		const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		if(aff_cv.size() == 0) return 0;

		//collect data vectors belonging to given cluster
		const Matrix y = dad.y_.GetRows(cv_ind);

		Matrix t;
		for(ulong i = 0; i < aff_cv.size(); ++i)
			t &= dad.x_.GetRows(aff_cv[i]) - y;
		//DEBUG - substitute mean
		//y <<= t.vMean();
		//Matrix::r_iterator pos = t.begin();
		//for(ulong i = 0; i < t.row_num(); ++i) {
		//	pos = transform(pos, pos + t.col_num(), y.begin(), pos, std::minus< double >());
		//}

		//calculate variance along principal axis
		//first calc SVD
		Matrix U, E, V;
		svd(!t, U, E, V);
		//we need only maximum singular value
		double var = E.Max();
		return var * var / aff_cv.size();
	}

	double calc_var_honest(const da_data& dad, ulong cv_ind, const Matrix& p_x) const {
		//assume that hard affiliation is already calculated
		//if(cv_ind >= hcd_.aff_.size()) return 0;
		//const ul_vec& aff_cv = hcd_.aff_[cv_ind];
		//if(aff_cv.size() == 0) return 0;

		//collect data vectors belonging to given cluster
		const Matrix y = dad.y_.GetRows(cv_ind);

		Matrix covar(x_.col_num(), x_.col_num()), x;
		double p_xiy;
		const double mult = 1.0 / p_y[cv_ind];
		Matrix::cr_iterator pp_x = p_x.begin(), pp_yx = p_yx.begin() + cv_ind * x_.row_num();
		covar = 0;
		for(ulong i = 0; i < x_.row_num(); ++i, ++pp_x, ++pp_yx) {
			p_xiy = *pp_yx * (*pp_x) * mult;
			x <<= dad.x_.GetRows(i) - y;
			covar += ((!x) * x) * p_xiy;
		}

		//calculate eigenvalues of covariance matrix
		//first calc SVD
		Matrix E;
		eig(covar, E);
		//we need only maximum singular value
		return E.Max();
		//return var * var / aff_cv.size();
	}

	void update_variances() {
		//force winners calculation
		calc_winners_kmeans(*this, hcd_);

		var_.Resize(y_.row_num());
		//DEBUG
		cout << y_.row_num() << " centers & their variances:" << endl;
		for(ulong i = 0; i < y_.row_num(); ++i) {
			y_.GetRows(i).Print(cout, false);
			//get p(x) distribution
			const Matrix& px = (this->*px_fcn_)(i);
			var_[i] = calc_var_honest(*this, i, px);
			cout << " : " << var_[i] << endl;
		}
		cout << endl;
	}

	void update_epoch() {
		ul_vec x_order; //, uniq_xor;
		Matrix::r_iterator cur_py = p_y.begin(), cur_pyx = p_yx.begin();
		Matrix x, y, new_y(1, y_.col_num());
		double new_py;

		//make hard clusters
		calc_winners(*this, hcd_);

		hcd_.e_ = 0;
		//main cycle starts here - update all available centers
		for(ulong i = 0; i < y_.row_num(); ++i, ++cur_py) {
			y <<= y_.GetRows(i);

			//get p(x) distribution
			const Matrix& px = (this->*px_fcn_)(i);

			//update probabilities p(y[i] | x) - independent from custom p(x)
			//update probabilities p(y[i]) & calculate new center position
			new_py = 0;
			new_y = 0;
			for(ulong j = 0; j < x_.row_num(); ++j, ++cur_pyx) {
				x <<= x_.GetRows(j);
				//first calc denominator for further calculation of p_yx
				double pyx_den = 0;
				for(ulong k = 0; k < y_.row_num(); ++k)
					//pyx_den += p(y[k]) * exp(-||x - y[k]||^2 / T)
					pyx_den += p_y[k] * exp(-(*norm2_fcn_)(x, y_.GetRows(k)) * beta_);

				//update p(y[i] | x) = p(y[i]) * exp(-||x - y[i]||^2 / T)
				*cur_pyx = (*cur_py) * exp(-(*norm2_fcn_)(x, y) * beta_) / pyx_den;

				//update p(y[i])
				new_py += (*cur_pyx) * px[j];
				//update y position
				new_y += x * (*cur_pyx) * px[j];
			}
			//complete y calculations
			new_y /= new_py;
			//update error
			hcd_.e_ += (*norm_fcn_)(y, new_y);
			//save values
			*cur_py = new_py;
			y_.SetRows(new_y, i);

		}	//end of centers loop

		//complete mse calculation
		hcd_.e_ /= y_.row_num();
	}

	ulong cooling_step() {
		//sort variances in descending order
		Matrix svar;
		svar = var_;
		Matrix::indMatrix ind = svar.RawSort(greater< double >());

		//compute next bifurcation prediction
		double pt_T;
		ulong pt_ind = svar.size();
		for(ulong i = 0; i < svar.size(); ++i)
			if((pt_T = 2 * svar[i]) < T_) {
				pt_ind = i;
				break;
			}

		//T_ *= alpha_;
		if(pt_ind < svar.size())
			//T_ = min(T_ * alpha_max_, max(T_ * alpha_, pt_T));
			T_ = max(T_ * alpha_, pt_T);
		else {
			T_ *= alpha_;
			pt_ind = svar.size();
		}

		beta_ = 1 / T_;
		cout << "cooling step done, new T = " << T_ << endl;

		//calc dCnum / dT


		return pt_ind;
	}

	void fork_center(ulong cv_ind) {
		if(cv_ind >= y_.row_num()) return;

		Matrix y = y_.GetRows(cv_ind);
		//generate perturbation
		Matrix perturb(1, y_.col_num());
		generate(perturb.begin(), perturb.end(), prg::rand01);
		perturb -= 0.5; perturb *= y.norm2() * 0.2;

		//add new center
		y_ &= y + perturb;
		p_y[cv_ind] *= 0.5;
		p_y.push_back(p_y[cv_ind], false);
		//assign the same affiliation probability to new cluster as the parent had
		p_yx &= p_yx.GetRows(cv_ind);
		//p_yx.Resize(y_.row_num());

		//resize affiliations
		hcd_.aff_.resize(y_.row_num());
	}

	bool phase_transition_epoch(ulong max_clust) {
		//main cycle starts here - check all available centers

		//do cooling step
		ulong pt_ind = cooling_step();
		//check low-temp condition
		if(T_ < Tmin_) return false;
		//fork centers if any
//		if(pt_ind < y_.row_num() && y_.row_num() < max_clust)
//			fork_center(pt_ind);

		//check every center for bifurcation
		ulong cnt = y_.row_num();
		for(ulong i = 0; i < cnt && y_.row_num() < max_clust; ++i)
			if(T_ < 2 * var_[i]) fork_center(i);
		return true;
	}

	void merge_step() {
		//test if any centers are coinsident and merge them
		//first of all calc centers distance matrix
		Matrix dist;
		norm_tools::calc_dist_matrix< norm_tools::l2 >(y_, dist);
		//DEBUG
		dist.Resize(1, dist.row_num() * dist.col_num());
		//find closest pairs
		Matrix::indMatrix pairs;
		pairs = norm_tools::closest_pairs< norm_tools::l2 >(dist);
		//cut tails that exeeds distance thresold
		ulong merge_cnt = (ulong)(find_if(dist.begin(), dist.end(), bind2nd(greater< double >(), MERGE_EPS))
									- dist.begin());
		if(merge_cnt == dist.size())
			//nothing to merge
			return;

		//merged centers indexes
		typedef set< ulong, greater< ulong > > merged_idx;
		merged_idx mc;

		//codebook with merged cenrters
		da_data new_cb;
		ulong cv1, cv2;
		//first pass - compute merged centers
		for(Matrix::indMatrix::cr_iterator i = pairs.begin(), end = pairs.end(); i != end; ++i) {
			//extract centers pair
			cv1 = (*i) / y_.row_num(); cv2 = (*i) % y_.row_num();
			//check if any of these codevectors already marked to merge
			if(mc.find(cv1) != mc.end() || mc.find(cv2) != mc.end())
				continue;
			//mark centers as merged
			mc.insert(cv1); mc.insert(cv2);
			//compute new merged center
			new_cb.y_ &= y_.GetRows(cv1);
			new_cb.p_y.push_back(p_y[cv1] + p_y[cv2], false);
			new_cb.p_yx &= p_yx.GetRows(cv1) + p_yx.GetRows(cv2);
		}

		//second pass - clear existing codebook
		for(merged_idx::const_iterator mi = mc.begin(), end = mc.end(); mi != end; ++mi) {
			y_.DelRows(*mi);
			p_yx.DelRows(*mi);
			p_y.DelColumns(*mi);
		}
		//append merged codebook
		y_ &= new_cb.y_;
		p_yx &= new_cb.p_yx;
		p_y |= new_cb.p_y;
		//resize affiliations
		hcd_.aff_.resize(y_.row_num());
	}

	//clusterization using deterministic annealing
	void find_clusters(const Matrix& data, const Matrix& f, ulong clust_num, ulong maxiter) {
		px_fcn_ = &da_impl::scaling_px;
		order_fcn_ = &da_impl::selection_based_order;
		norm_fcn_ = &da_impl::l2_norm;
		norm2_fcn_ = &da_impl::l2_norm2;

		//initialization
		f_ = f;
		x_ = data;
		y_.Resize(1, x_.col_num());
		p_y.Resize(1, 1);
		p_yx.Resize(1, x_.row_num());

		hcd_.w_.Resize(x_.row_num());
		hcd_.norms_.Resize(x_.row_num());

		//set initial probabilities to 1
		p_y = 1; p_yx = 1;
		//all points initially belongs to single center
		hcd_.aff_.resize(1);
		for(ulong i = 0; i < x_.row_num(); ++i)
			hcd_.aff_[0].push_back(i);

		//set position of the first center
		//get p(x) distribution
		const Matrix& px = (this->*px_fcn_)(0);
//		for(ulong i = 0; i < x_.row_num(); ++i)
//			y_ += x_.GetRows(i) * px[i];

		//set initial T > 2 * max variation along principal axis of all data
		T_ = calc_var_honest(*this, 0, px) * 2;
		T_ *= 1.2;
		beta_ = 1 / T_;

		//clear history
		hist_.clear();
		//hist_.push_back(*this);

		//main cycle starts here
		for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
			for(ulong i = 0; i < maxiter; ++i) {
				//update probabilities and centers positions
				update_epoch();
				//convergence test
				//if(patience_check(i, hcd_.e_)) break;
				if(hcd_.e_ < EPS) break;
			}
			//perform merge step
			merge_step();

			//update variances
			update_variances();

			//save history
			hist_.push_back(*this);
			//display dCnum / dT info
			cout << "dCnum/dT = ";
			if(hist_.size() > 1) {
				cout << (hist_[cycle_].y_.row_num() - hist_[cycle_ - 1].y_.row_num()) /
						(hist_[cycle_].T_ - hist_[cycle_ - 1].T_);
			}
			else cout << "0";
			cout << endl;

			//temperature test
			//if(T_ <= Tmin_) break;

			//add new centers if nessessary
			if(!phase_transition_epoch(clust_num)) break;
		}	//end of main loop
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

};


//========================================== determ_annealing implementation ===========================================
determ_annealing::determ_annealing()
	: pimpl_(new da_impl)
{
	//ctor
}

determ_annealing::~determ_annealing()
{
	//dtor
}

void determ_annealing::find_clusters(const Matrix& data, const Matrix& f, ulong clust_num, ulong maxiter) {
	pimpl_->find_clusters(data, f, clust_num, maxiter);
}

const Matrix& determ_annealing::get_centers() const {
	return pimpl_->y_;
}

const ulMatrix& determ_annealing::get_ind() const {
	return pimpl_->hcd_.w_;
}

const Matrix& determ_annealing::get_norms() const {
	return pimpl_->hcd_.norms_;
}

const vvul& determ_annealing::get_aff() const {
	return pimpl_->hcd_.aff_;
}

}	//end of namespace DA
