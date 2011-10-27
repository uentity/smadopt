#include "determ_annealing.h"
#include "ga.h"
#include "m_algorithm.h"
#include "prg.h"

#include <math.h>
#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#define EPS 0.0000001
#define T_EPS 0.0001
#define MERGE_EPS 0.005
#define MERGE_PATIENCE 3
#define MERGE_THRESH 0.1
#define PERTURB_MULT 0.1
#define EXPL_MAXGROW 1.7
#define EXPL_LENGTH 4
#define EXPL1_LENGTH 2
#define EXPL_THRESH_FACTOR 5
#define COOL_FACTOR 25

using namespace GA;
using namespace prg;
using namespace std;

namespace DA {

typedef determ_annealing::vvul vvul;

//========================== implementation of deterministic annealing =================================================
//norms used - currently only l2
double l2_norm(const Matrix& x1, const Matrix& x2) {
	return (x1 - x2).norm2();
}

double l2_norm2(const Matrix& x1, const Matrix& x2) {
	Matrix diff = x1 - x2;
	return diff.Mul(diff).Sum();
}

//norm function derivatives
Matrix l2_deriv(const Matrix& dv, const Matrix& center) {
	return (dv - center);
}

//forward declaration
struct da_hist;

//codevector info
struct cv_info {
	//location of codevector
	Matrix loc_;
	//parent identifier
	ulong parent_;
	//previous distance to parent
	double pdist_;
	//patience steps counts consequent steps of rapproachement with parent
	ulong pat_steps_;
	//probability p(y)
	double p_;
	//parobabilities p(y|x)
	Matrix px_;
	//variance of cluster
	double var_;

	//ctors
	cv_info() : parent_(0), pdist_(0), pat_steps_(0) {};
	cv_info(ulong parent, double pdist) : parent_(parent), pdist_(pdist), pat_steps_(0) {};
	cv_info(const Matrix& loc, ulong parent, double pdist)
		: loc_(loc), parent_(parent), pdist_(pdist), pat_steps_(0)
	{}

	//copy ctor - ensure that loc_ is copied
	cv_info(const cv_info& cvi)
		: parent_(cvi.parent_), pdist_(cvi.pdist_), pat_steps_(cvi.pat_steps_),
		p_(cvi.p_), var_(cvi.var_)
	{
		loc_ = cvi.loc_;
		px_ = cvi.px_;
	}

	//swaps 2 cv_info structures
	void swap(cv_info& cvi) {
		Matrix tmp;

		//swap loc_
		tmp = loc_;
		loc_ = cvi.loc_;
		cvi.loc_ = tmp;
		//swap px_
		tmp = px_;
		px_ = cvi.px_;
		cvi.px_ = tmp;

		//swap other
		std::swap(parent_, cvi.parent_);
		std::swap(pdist_, cvi.pdist_);
		std::swap(pat_steps_, cvi.pat_steps_);
		std::swap(p_, cvi.p_);
		std::swap(var_, cvi.var_);
	}

	//assignment through swap
	cv_info& operator=(const cv_info& cvi) {
		cv_info(cvi).swap(*this);
		return *this;
	}
};

typedef map< ulong, cv_info > cv_map;
typedef pair< cv_map::iterator, bool > cv_map_insres;

//distance calculation between 2 codevectors
double cv_dist(const cv_map::const_iterator& cv1, const cv_map::const_iterator& cv2) {
	return l2_norm2(cv1->second.loc_, cv2->second.loc_);
}

//descending sorting criteria for cv_info
struct cvip_sort_desc {
	bool operator()(cv_map::value_type* const& lhs, cv_map::value_type* const& rhs) const {
		return (lhs->second.var_ > rhs->second.var_);
	}
};

typedef set< cv_map::value_type*, cvip_sort_desc > cvp_set_desc;

//names of variables follow original notation by K. Rose
struct da_data {
	//source points
	Matrix x_;
	//codevectors with unique identifier
	cv_map y_;
	//temperature
	double T_;
	//beta = 1/T
	double beta_;

	da_data& operator =(const da_hist& hist);

	void clear() {
		x_.clear();
		y_.clear();
		T_ = 0;
		beta_ = 0;
	}

	//generate new unique identifier for cv_map
	ulong cv_id() const {
		if(y_.size() == 0) return 0;
		return (--(y_.end()))->first + 1;
	}

	ulong push_cv(const Matrix& new_cv, ulong parent) {
		//calc distance to parent
		ulong new_id = cv_id();
		double pdist = 0;
		cv_map::const_iterator par = y_.find(parent);
		if(par != y_.end())
			pdist = l2_norm2(new_cv, par->second.loc_);
		else
			parent = new_id;

		//add new codevector
		cv_map_insres p_ny = y_.insert(cv_map::value_type(new_id, cv_info(new_cv, parent, pdist)));

		//assign the same affiliation probability to new cluster as the parent had
		if(parent != new_id)
			p_ny.first->second.px_ = par->second.px_;

		return new_id;
	}

	void update_pdist() {
		double dist;
		cv_map::const_iterator par;
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			if(p_cv->first == p_cv->second.parent_) continue;
			//check if parent index is valid
			if((par = y_.find(p_cv->second.parent_)) == y_.end()) {
				//invalid parent index
				p_cv->second.parent_ = p_cv->first;
				continue;
			}
			//calc distance to parent
			dist = cv_dist(p_cv, par);
			//check if cv is moving to parent
			if(dist < p_cv->second.pdist_)
				++p_cv->second.pat_steps_;
			else if(dist > p_cv->second.pdist_)
				p_cv->second.pat_steps_ = 0;
			//update distance
			p_cv->second.pdist_ = dist;
		}
	}

	//dynamically change parent to the closest center
	void update_pdist_cl() {
		if(y_.size() <= 1) return;

		//distance matrix
		Matrix dist;
		//build id -> ind relation
		map< ulong, ulong > id2ind;
		map< ulong, ulong > ind2id;
		ulong i = 0;
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv, ++i) {
			id2ind[p_cv->first] = i;
			ind2id[i] = p_cv->first;
		}

		//compute distances between cv
		norm_tools::calc_dist_matrix< norm_tools::l2 >(get_y(), dist);

		//update parent to be the closest cv
		Matrix drow;
		ulMatrix ind;
		double mdist;
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			//find closest cv
			drow <<= dist.GetRows(id2ind[p_cv->first]);
			ind <<= drow.RawSort();
			p_cv->second.parent_ = ind2id[ind[1]];
			mdist = drow[ind[1]];

			//check if cv is moving to parent
			if(mdist < p_cv->second.pdist_)
				++p_cv->second.pat_steps_;
			else if(mdist > p_cv->second.pdist_)
				p_cv->second.pat_steps_ = 0;

			p_cv->second.pdist_ = mdist;
		}
	}

	Matrix get_y() const {
		Matrix y;
		for(cv_map::const_iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv)
			y &= p_cv->second.loc_;

		return y;
	}
};

//structure to track annealing process
struct da_hist {
	cv_map y_;
	//Matrix p_y;
	double T_;

	da_hist(const da_data& dad) : y_(dad.y_), T_(dad.T_) {}
};

da_data& da_data::operator =(const da_hist& hist) {
	y_ = hist.y_;
	T_ = hist.T_;
	beta_ = 1. / T_;
	return *this;
}

template < class charT, class traits >
inline
std::basic_ostream< charT, traits >&
operator <<(std::basic_ostream< charT, traits >& strm, const da_hist& h)
{
	strm << h.y_.size() << "	" << h.T_ << "	" << 1. / h.T_;
	return strm;
}

struct da_log {
	typedef vector< da_hist > hist_v;
	hist_v hist_;
	Matrix dt_;

	hist_v::const_reference head(ulong steps_back = 0) const {
		return hist_[hist_.size() - 1 - steps_back];
	}

	double head_dT(ulong steps_back = 0) const {
		return dt_[dt_.size() - 1 - steps_back];
	}

	void push_back(const da_hist& hist) {
		hist_.push_back(hist);
		if(hist_.size() > 1) {
			const da_hist& prev = head(1);
			dt_.push_back( (double(hist.y_.size()) - double(prev.y_.size())) / (prev.T_ - hist.T_) );
		}
		else dt_.push_back(0);
	}

	void clear() {
		hist_.clear();
		dt_.clear();
	}

	Matrix get_vT() const {
		Matrix vt(hist_.size(), 1);
		Matrix::r_iterator p_vt = vt.begin();
		for(hist_v::const_iterator h = hist_.begin(), end = hist_.end(); h != end; ++h, ++p_vt)
			*p_vt = h->T_;
		return vt;
	}

	void dump() const {
		ofstream f("da_log.txt", ios::out | ios::trunc);
		for(ulong i = 0; i < hist_.size(); ++i)
			f << hist_[i] << "	" << dt_[i] << endl;
		//DumpMatrix(get_vT() | dt_, "da_log.txt");
	}
};

typedef map< ulong, ul_vec > aff_map;

struct hard_clusters_data {
	//winners matrix - point->center relation
	ulMatrix w_;
	Matrix norms_;
	//points affiliation - center->points collection
	aff_map aff_;
	//sum of distances from points to corresponding cluster centers
	double e_;

	hard_clusters_data() : e_(0) {}

	vvul get_aff() const {
		vvul aff;
		for(aff_map::const_iterator p_cv = aff_.begin(), end = aff_.end(); p_cv != end; ++p_cv)
			aff.push_back(p_cv->second);
		return aff;
	}
};

struct cov_data {
	Matrix cov_;
	double variance_;
};


//============================================= DA implementation ======================================================
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
				new_prob *= 0.5;
				//now we have updated probability distribution - check patience
				//diff = distance between prob & new_prob
				diff <<= new_prob - prob;
				diff *= diff;
				//dist = sqrt(diff.Sum());
				//if(dist < sqrt(prob.Mul(prob).Sum()) * 0.05)
				dist = diff.Sum();
				if(dist < prob.Mul(prob).Sum() * 0.0025)
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

	//near-zero variance condition checker
	struct nz_var {
		nz_var(bool enabled) : enabled_(enabled) {}

		bool operator()(const cv_map::const_iterator& p_cv) const {
			return (enabled_ && p_cv->second.var_ < EPS);
		}

		bool enabled_;
	};

	//near-zero probability condition checker
	struct nz_prob {
		nz_prob(bool enabled) : enabled_(enabled) {}

		bool operator()(const cv_map::const_iterator& p_cv) const {
			return (enabled_ && p_cv->second.p_ < EPS);
		}

		bool enabled_;
	};

public:
	//probability threshold for making hard clusters - see calc_winners
	double prob_thresh_;
	double patience_;
	double alpha_, alpha_max_;
	double Tmin_, Texpl_;
	double dither_amount_;
	double merge_thresh_;
	ulong patience_cycles_, expl_length_, expl_steps_;
	ulong expl_det_start_;
	bool is_exploded;
	bool recalc_px_, kill_zero_prob_cent_, kill_zero_var_cent_;

	hard_clusters_data hcd_;
	//function values
	Matrix f_;
	ulong cycle_;

	//statistics for source points
	norm_tools::dist_stat srcp_stat_;

	//annealing history
	//vector< da_hist > hist_;
	da_log log_;

	double (*norm_fcn_)(const Matrix&, const Matrix&);
	double (*norm2_fcn_)(const Matrix&, const Matrix&);
	Matrix (*deriv_fcn_)(const Matrix&, const Matrix&);
	void (da_impl::*order_fcn_)(ulong, ul_vec&) const;

	typedef const Matrix& (da_impl::*px_fcn_t)(ulong) const;
	px_fcn_t px_fcn_;

	da_impl()
		: prob_thresh_(1.0/3), patience_(0.001), alpha_(0.9), alpha_max_(0.99), Tmin_(0.01), dither_amount_(0.005),
		patience_cycles_(10), expl_length_(EXPL_LENGTH), kill_zero_prob_cent_(true), kill_zero_var_cent_(true)
	{}

	void calc_winners(const da_data& dad, hard_clusters_data& hcd) const
	{
		hcd.aff_.clear();
		//hcd.norms_.Resize(1, dad.x_.row_num());
		//hcd.w_.Resize(1, dad.x_.row_num());

		//normalize threshold
		double cur_pt = prob_thresh_ / dad.y_.size();

		//move through all data points
		double cur_p, max_p;
		ulong winner;
		for(ulong i = 0; i < dad.x_.row_num(); ++i) {
			//iterate through centers
			for(cv_map::const_iterator p_cv = dad.y_.begin(), end = dad.y_.end(); p_cv != end; ++p_cv) {
				cur_p = p_cv->second.px_[i];
				ul_vec& cur_aff = hcd.aff_[p_cv->first];
				if(cur_p > cur_pt)
					cur_aff.push_back(i);
				if(p_cv == dad.y_.begin() || cur_p > max_p) {
					max_p = cur_p;
					winner = p_cv->first;
				}
			}

			//save winner
			hcd.w_[i] = winner;
			hcd.norms_[i] = (*norm_fcn_)(dad.x_.GetRows(i), dad.y_.find(winner)->second.loc_);
		}
	}

	//calc affiliation for particular cluster only
	void calc_aff(const da_data& dad, hard_clusters_data& hcd, ulong cv_id) const
	{
		if(dad.y_.find(cv_id) == dad.y_.end()) return;
		//clear affiliation list
		ul_vec& cur_aff = hcd.aff_[cv_id];
		cur_aff.clear();

		//normalize threshold
		double cur_pt = prob_thresh_ / dad.y_.size();

		//move through all data points
		Matrix::cr_iterator cur_pyx = dad.y_.find(cv_id)->second.px_.begin();
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
		//hcd.norms_.Resize(1, dad.x_.row_num());
		//hcd.w_.Resize(1, dad.x_.row_num());

		Matrix dv;
		hcd.e_ = 0;
		double tmp, min_dist;
		ulong winner;
		//calc centers-winners for each data point
		for(ulong i = 0; i < dad.x_.row_num(); ++i) {
			dv <<= dad.x_.GetRows(i);
			for(cv_map::const_iterator p_cv = dad.y_.begin(), end = dad.y_.end(); p_cv != end; ++p_cv) {
				tmp = (*norm2_fcn_)(dv, p_cv->second.loc_);
				if(p_cv == dad.y_.begin() || tmp < min_dist) {
					min_dist = tmp;
					winner = p_cv->first;
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

	//patterns order processing
	void conseq_order(ulong cv_id, ul_vec& porder) const {
		//assume that hard affiliation is already calculated
		//check that requested cv exists
		aff_map::const_iterator res = hcd_.aff_.find(cv_id);
		if(res == hcd_.aff_.end()) return;

		porder.clear();
		const ul_vec& aff_cv = res->second;
		for(ul_vec::const_iterator pos = aff_cv.begin(); pos != aff_cv.end(); ++pos)
			porder.push_back(*pos);

		//random_shuffle(porder.begin(), porder.end(), prg::randIntUB);
	}

	//resulting porder will contain ABSOLUTE indexes applicable directly to da_data::x_
	void selection_based_order(ulong cv_id, ul_vec& porder) const {
		//assume that hard affiliation is already calculated
		//check that requested cv exists
		aff_map::const_iterator res = hcd_.aff_.find(cv_id);
		if(res == hcd_.aff_.end()) return;

		//select f values belonging to given cluster
		Matrix f_cv;
		const ul_vec& aff_cv = res->second;
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
	const Matrix& unifrom_px(ulong cv_id) const {
		struct px {
			Matrix expect_;

			void reset(const Matrix& f) {
				expect_.Resize(1, f.size());
				expect_ = 1.0/(double)f.size();
			}
		};
		static px p;

		if(recalc_px_) p.reset(f_);
		return p.expect_;
	}

	//p(x) distribution based on static GA.ScalingCall calculated only when first time called
	const Matrix& scaling_px(ulong cv_id) const {
		static Matrix expect;
		if(recalc_px_)
			expect <<= get_ps().scaling(f_);
		return expect;
	}

	//p(x) distribution based dynamically calculated expectation for given cluster from it's affiliation
	const Matrix& scaling_px_aff(ulong cv_id) const {
		static Matrix expect(1, f_.size());
		if(expect.size() != f_.size())
			expect.Resize(1, f_.size());

		//assume that hard affiliation is already calculated
		//check that requested cv exists
		aff_map::const_iterator res = hcd_.aff_.find(cv_id);
		if(res == hcd_.aff_.end()) throw ga_except("scaling_px_aff: invalid codevector id passed");
		const ul_vec& aff_cv = res->second;

		//select f values belonging to given cluster
		Matrix f_cv;
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
	const Matrix& selection_px(ulong cv_id) const {
		static Matrix expect;
		if(recalc_px_)
			expect <<= get_ps().selection_prob(f_, 100, 1000);
		return expect;
	}

	const Matrix& selection_px_aff(ulong cv_id) const {
		static Matrix expect;
		if(expect.size() != f_.size())
			expect.Resize(1, f_.size());

		//assume that hard affiliation is already calculated
		//check that requested cv exists
		aff_map::const_iterator res = hcd_.aff_.find(cv_id);
		if(res == hcd_.aff_.end()) throw ga_except("selection_px_aff: invalid codevector id passed");

		//select f values belonging to given cluster
		Matrix f_cv;
		const ul_vec& aff_cv = res->second;
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
	const Matrix& order2px(ulong cv_id) const {
		static Matrix expect;
		if(expect.size() != f_.size())
			expect.Resize(1, f_.size());

		//calc affiliation for given center
		//calc_aff(*this, hcd_, cv_ind);

		//assume that hard affiliation is already calculated
		//build patterns order
		ul_vec order;
		(this->*order_fcn_)(cv_id, order);

		//convert order to expectation
		expect = 0;
		const double p_quant = 1.0 / order.size();
		for(ulong i = 0; i < order.size(); ++i) {
			expect[order[i]] += p_quant;
		}
		return expect;
	}

	template< px_fcn_t px_fcn >
	const Matrix& dithered_px(ulong cv_id) const {
		//get original distribution
		Matrix& px = const_cast< Matrix& >((this->*px_fcn)(cv_id));

		//generate random noise
		Matrix noise(px.row_num(), px.col_num());
		generate(noise.begin(), noise.end(), prg::rand01);
		//noise -= 0.5;
		//calc correction to make sum of noise = 0
		double n_item = noise.Sum() / noise.size();
		//make correction
		noise -= n_item;
		//if(n_item != 0) transform(noise, bind2nd(minus< double >(), n_item));

		//scale noise
		noise *= (px.norm2() * dither_amount_) / noise.norm2();
		//add noise
		px += noise;

		//correct px to ensure that there are no subzero elements
		//disabled - no positive effect, only slowdown
//		for(Matrix::r_iterator pos = px.begin(), end = px.end(); pos != end; ++pos) {
//			if(*pos >= 0) continue;
//			//randomly find i, such that px[i] > abs(*pos)
//			ulong i = prg::randIntUB(px.size());
//			while(px[i] <= abs(*pos)) i = prg::randIntUB(px.size());
//			//correct *pos and px[i]
//			px[i] += *pos;
//			*pos = 0;
//		}
		return px;
	}

	double calc_var(const da_data& dad, ulong cv_id) const {
		//assume that hard affiliation is already calculated
		//check that requested cv exists
		aff_map::const_iterator res = hcd_.aff_.find(cv_id);
		if(res == hcd_.aff_.end()) throw ga_except("calc_var: invalid codevector id passed");
		const ul_vec& aff_cv = res->second;

		//collect data vectors belonging to given cluster
		const Matrix y = dad.y_.find(cv_id)->second.loc_;

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

	double calc_var_honest(const da_data& dad, ulong cv_id, const Matrix& p_x) const {
		//assume that hard affiliation is already calculated
		//check that requested cv exists
		cv_map::const_iterator res = dad.y_.find(cv_id);
		if(res == dad.y_.end()) throw ga_except("calc_var_honest: invalid codevector id passed");
		//const ul_vec& aff_cv = res->second;

		//collect data vectors belonging to given cluster
		const cv_info& cvi = res->second;
		const Matrix y = cvi.loc_;

		Matrix covar(x_.col_num(), x_.col_num()), x;
		double p_xiy;
		const double mult = 1.0 / cvi.p_;
		Matrix::cr_iterator pp_x = p_x.begin(), pp_yx = cvi.px_.begin();
		covar = 0;
		for(ulong i = 0; i < x_.row_num(); ++i, ++pp_x, ++pp_yx) {
			p_xiy = *pp_yx * (*pp_x) * mult;
			x <<= dad.x_.GetRows(i) - y;
			covar += ((!x) * x) * p_xiy;
		}

		//calculate eigenvalues of covariance matrix
		Matrix E;
		eig(covar, E);
		//we need only maximum singular value
		return E.Max();
		//return var * var / aff_cv.size();
	}

	void update_variances() {
		//force winners calculation
		calc_winners_kmeans(*this, hcd_);

		//dump info
		cout << y_.size() << " centers & their variances:" << endl;
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			p_cv->second.loc_.Print(cout, false);
			//get p(x) distribution
			const Matrix& px = (this->*px_fcn_)(p_cv->first);
			p_cv->second.var_ = calc_var_honest(*this, p_cv->first, px);
			cout << " : " << p_cv->second.var_ << endl;
		}
		//cout << endl;
	}

	template< class cond_checker >
	void kill_weak_centers(const cond_checker& ck) {
//		if(!kill_zero_var_cent_) return;
//
//		//kill centers with near-zero probability p_y
//		if(kill_zero_prob_cent_) {
//			cv_map::iterator p_cv = y_.begin();
//			while(p_cv != y_.end()) {
//				if(y_.size() == 1) break;
//				if(p_cv->second.p_ < EPS)
//					y_.erase(p_cv++);
//				else ++p_cv;
//			}
//		}

		//kill centers with near zero variance or near zero probability
		double mind, curd;
		ulong cl_cv;

		typedef set< ulong > kill_set;
		kill_set to_kill;
		//to_kill.reserve(y_.size());

		//make first pass - find centers to kill
		for(cv_map::const_iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			//check kill conditions
//			if( (kill_zero_prob_cent_ && p_cv->second.p_ < EPS) ||
//				(var_known && kill_zero_var_cent_ && p_cv->second.var_ < EPS) )
			if(ck(p_cv))
				to_kill.insert(p_cv->first);
		}

		//check that at least one center will be alive
		if(to_kill.size() == y_.size())
			to_kill.erase(to_kill.begin());

		//second pass - kill cv
		for(kill_set::const_iterator pk = to_kill.begin(), end = to_kill.end(); pk != end; ++pk) {
			//find closest center
			mind = 0;
			for(cv_map::const_iterator p_cv = y_.begin(), end1 = y_.end(); p_cv != end1; ++p_cv) {
				if(p_cv->first == *pk || to_kill.find(p_cv->first) != to_kill.end()) continue;
				curd = cv_dist(y_.find(*pk), p_cv);
				if(mind == 0 || curd < mind) {
					mind = curd;
					cl_cv = p_cv->first;
				}
			}

			//add probability to closest center
			y_[cl_cv].p_ += y_[*pk].p_;

			//erase cv
			y_.erase(*pk);
		}
	}

	void update_epoch() {
		ul_vec x_order; //, uniq_xor;
		Matrix x, y, new_y(1, x_.col_num());
		double new_py, pyx_den, pyx_num;

		//newly calculated centers positions & p_y will be stored here
		cv_map res(y_);

		//make hard clusters
		calc_winners(*this, hcd_);
		hcd_.e_ = 0;

		//main cycle starts here - update all available centers
		double cur_py;
		Matrix::r_iterator cur_pyx;
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			//setup
			cv_info& src_cvi = p_cv->second;
			cv_info& dst_cvi = res[p_cv->first];
			//those source values shouldn't change during whole update epoch
			y <<= src_cvi.loc_;
			cur_py = src_cvi.p_;
			//values p(y_i|x) can be calculated directly to destination
			cur_pyx = dst_cvi.px_.begin();

			//get p(x) distribution
			const Matrix& px = (this->*px_fcn_)(p_cv->first);

			//update probabilities p(y[i] | x) - independent from custom p(x)
			//update probabilities p(y[i]) & calculate new center position
			new_py = 0;
			new_y = 0;
			for(ulong j = 0; j < x_.row_num(); ++j, ++cur_pyx) {
				x <<= x_.GetRows(j);
				//first calc numerator of p_yx
				pyx_num = cur_py * exp(-(*norm2_fcn_)(x, y) * beta_);
				//next calc denominator for further calculation of p_yx
				pyx_den = 0;
				for(cv_map::iterator p_cv1 = y_.begin(); p_cv1 != end; ++p_cv1) {
					if(p_cv->first == p_cv1->first) continue;
					//pyx_den += p(y[k]) * exp(-||x - y[k]||^2 / T)
					pyx_den += p_cv1->second.p_ * exp(-(*norm2_fcn_)(x, p_cv1->second.loc_) * beta_);
				}
				pyx_den += pyx_num;

				//update p(y[i] | x) = p(y[i]) * exp(-||x - y[i]||^2 / T)
				if(pyx_den > EPS)
					*cur_pyx = pyx_num / pyx_den;
				else
					*cur_pyx = 0;

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
			dst_cvi.loc_ = new_y;
			dst_cvi.p_ = new_py;
		}	//end of centers loop

		//complete mse calculation
		hcd_.e_ /= y_.size();
		//assign new values
		y_ = res;
	}

	void null_step() {
		//this step is equal to offline kmeans iteration with explicit p(x) distribution

		//first of all calc winners
		calc_winners_kmeans(*this, hcd_);

		//now calc centers positions depending on prob. distribution
		double norm_mult;
		Matrix new_y(1, x_.col_num());
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			//get p(x) distribution
			const Matrix& px = (this->*px_fcn_)(p_cv->first);

			norm_mult = 0;
			new_y = 0;
			const ul_vec& aff_i = hcd_.aff_[p_cv->first];
			for(ulong j = 0; j < aff_i.size(); ++j) {
				//calc center position
				new_y += x_.GetRows(aff_i[j]) * px[aff_i[j]];
				//calc prob normalization multiplyer
				norm_mult += px[aff_i[j]];
			}
			//finish center position calculation
			new_y /= norm_mult;
			//save calculated center
			p_cv->second.loc_ = new_y;
		}
	}

	ulong cooling_step(ulong maxclust) {
		//compute next bifurcation prediction
		double pt_T;
		ulong pt_ind = ulong(-1);
		if(!is_exploded && y_.size() < maxclust) {
			//sort variances in descending order
			cvp_set_desc svar;
			for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
				svar.insert(&(*p_cv));
			}

			//calc pt_T prediction
			for(cvp_set_desc::iterator p_scv = svar.begin(), end = svar.end(); p_scv != end; ++p_scv)
				if((pt_T = (*p_scv)->second.var_ * 2)  < T_) {
					pt_ind = (*p_scv)->first;
					break;
				}

			//find cluster with max variance
//			pt_T = 0;
//			for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
//				if(p_cv->second.var_ > pt_T) {
//					pt_T = p_cv->second.var_;
//					pt_ind = p_cv->first;
//				}
//			}
//			pt_T *= 2;

			//T_ *= alpha_;
			if(pt_ind != ulong(-1) && pt_T > T_ * alpha_) {
				cout << "PT point, ";
				//T_ = min(pt_T, T_) - T_EPS;
				T_ = min(pt_T - T_EPS, T_ * alpha_max_);

				//T_ = min(T_ * alpha_max_, max(T_ * alpha_, pt_T));
				//T_ = max(T_ * alpha_, pt_T);
			}
			else {
				T_ *= alpha_;
				pt_ind = ulong(-1);
			}
		}
		else {
			cout << "freeze, ";
			T_ *= alpha_ * alpha_;
		}

		beta_ = 1 / T_;
		cout << "cooling step done, T = " << T_ << ", T - Tmin = " << T_ - Tmin_ << endl;

		//calc dCnum / dT

		return pt_ind;
	}

	void fork_center(ulong cv_id) {
		cv_map::iterator src = y_.find(cv_id);
		if(src == y_.end()) return;

		Matrix y = src->second.loc_;
		//generate perturbation
		Matrix perturb(1, x_.col_num());
		generate(perturb.begin(), perturb.end(), prg::rand01);
		perturb -= 0.5; perturb *= y.norm2() * PERTURB_MULT;
		//calc length multiplier, so that
		//distance between perturbed & original cv's will be slightly less than merge_thresh_
		//double mult = merge_thresh_ * 0.98 / perturb.norm2();
		//perturb *= mult;

		//add new center
		src->second.p_ *= 0.5;
		ulong child_id = push_cv(y + perturb, src->first);
		y_[child_id].p_ = src->second.p_;

		//add affiliation record
		//hcd_.aff_.resize(y_.row_num());
	}

	bool phase_transition_epoch(ulong max_clust) {
		//main cycle starts here - check all available centers

		//do cooling step
		ulong pt_ind = cooling_step(max_clust);
		//check low-temp condition
		if(T_ < Tmin_) return false;
		//fork centers if any
//		if(pt_ind != ulong(-1) && y_.find(pt_ind) != y_.end() && y_.size() < max_clust)
//			fork_center(pt_ind);

		//check every center for bifurcation
		ulong cnt = y_.size();
		cv_map::iterator p_cv = y_.begin();
		for(ulong i = 0; i < cnt && y_.size() < max_clust; ++i, ++p_cv)
			if(T_ < 2 * p_cv->second.var_) fork_center(p_cv->first);
		return true;
	}

	bool merge_step() {
		//perform classic merge step first
		bool ret = false;
//		bool ret = merge_step_classic();
//		return ret;

		//update distances to parents
		update_pdist();

		//merge
		typedef vector< cv_map::iterator > kill_map;
		kill_map to_kill;
		//look for cv to merge
		//first pass - find centers to merge
		for(cv_map::iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv) {
			if(p_cv->second.pat_steps_ > MERGE_PATIENCE)
				//update p(y) of parent cv
				to_kill.push_back(p_cv);
		}

		//second pass - update parent's p(y)
		bool py_flow;
		do {
			py_flow = false;
			for(kill_map::iterator pk = to_kill.begin(), end = to_kill.end(); pk != end; ++pk) {
				if((*pk)->second.p_ != 0) {
					py_flow = true;
					//pass p(y) to parent
					y_.find((*pk)->second.parent_)->second.p_ += (*pk)->second.p_;
					(*pk)->second.p_ = 0;
				}
			}
		} while(py_flow);

		//third pass - erase merged cv
		for(kill_map::iterator pk = to_kill.begin(), end = to_kill.end(); pk != end; ++pk)
			y_.erase(*pk);

		//forth pass - classic merge based on distances
		return ret || (to_kill.size() > 0);
	}


	bool merge_step_classic() {
		//create y matrix & ind2id map
		map< ulong, ulong > ind2id;
		Matrix y;
		y.reserve(y_.size() * x_.col_num());
		ulong i = 0;
		for(cv_map::const_iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv, ++i) {
			y &= p_cv->second.loc_;
			ind2id[i] = p_cv->first;
		}

		//test if any centers are coinsident and merge them
		//first of all calc centers distance matrix
		Matrix dist;
		//compute distance between p_xy
		//norm_tools::calc_dist_matrix< norm_tools::l2 >(p_yx, dist);
		norm_tools::dist_stat stat = norm_tools::calc_dist_matrix< norm_tools::l2 >(y, dist);

		//dynamically calc min distance between centers
		merge_thresh_ = MERGE_EPS;
		if(y_.size() > 2)
			merge_thresh_ = (stat.mean_nn_ + srcp_stat_.mean_nn_) * MERGE_THRESH * 0.5;

		//DEBUG
		//dist.Resize(1, dist.row_num() * dist.col_num());
		//find closest pairs
		Matrix::indMatrix pairs;
		pairs = norm_tools::closest_pairs< norm_tools::l2 >(dist);

		//find merge cutoff
		ulong merge_cnt = (ulong)(find_if(dist.begin(), dist.end(), bind2nd(greater< double >(), merge_thresh_))
									- dist.begin());
		if(merge_cnt == dist.size())
			//nothing to merge
			return false;

		//merged centers indexes
		typedef set< ulong, greater< ulong > > merged_idx;
		merged_idx mc;
		ul_vec to_kill;

		//codebook with merged centers
		//da_data new_cb;
		ulong cv1, cv2;
		//first pass - compute merged centers
		for(ulong i = 0; i < merge_cnt; ++i) {
			//extract centers pair
			cv1 = pairs[i] / y_.size(); cv2 = pairs[i] % y_.size();
			//check if any of these codevectors already marked to merge
			if(mc.find(cv1) != mc.end() || mc.find(cv2) != mc.end())
				continue;
			//mark centers as merged
			mc.insert(cv1); mc.insert(cv2);
			//transform indexes to IDs
			cv1 = ind2id[cv1]; cv2 = ind2id[cv2];
			//mark m2 to be erased
			to_kill.push_back(cv2);

			//update prob of cv1
			y_[cv1].p_ += y_[cv2].p_;
		}

		//second pass - clear existing codebook
		for(ulong i = 0; i < to_kill.size(); ++i)
			y_.erase(to_kill[i]);

		return mc.size() > 0;
	}

	ulong detect_explosion(ulong cycle) {
		if(is_exploded) return 0;
		//static ulong expl_steps = 0;

//		struct expl_trigger {
//			static ulong cancel_expl(da_impl& di, ulong rev_steps) {
//				//check for "false" explosion that can occur from the beginning
//				if(di.log_.head(rev_steps).y_.size() == 1) return 0;
//
//				//explosion detected
//				cout << "EXPLOSION detected! Rolling " << rev_steps << " steps back" << endl;
//				(da_data&)(di) = di.log_.head(rev_steps);
//				//update variances
//				//di.update_variances();
//				//save new log entry
//				//di.log_.push_back(di);
//				return di.y_.size();
//			}
//		};

		ulong rev_steps = 0;

		//check whether we should try to detect explosion
		if(T_ > Texpl_) return 0;
		else if(expl_det_start_ == 0) expl_det_start_ = cycle;

		//explosion detection
		//if we had 3 sequental steps of positive dCnum / dbeta
		//and next step wasn't negative (centers were keeped)
		//then roll 4 steps back and freeze centers number
		double dt = log_.head_dT();
		if(cycle == 0 || dt < -EPS) expl_steps_ = 0;
		else if(++expl_steps_ >= expl_length_ 	//too long cluster number increasing
				//|| y_.row_num() >= log_.head(expl_steps).y_.row_num() * 2		//too fast increasing
				)
		{
			if(log_.head(expl_steps_).y_.size() > 1) {
				//freeze centers number
				is_exploded = true;
				rev_steps = expl_steps_ - 1;
			}
			else expl_steps_ = 0;
//			cout << "expl_steps = " << expl_steps_ << endl;
//			ret = expl_trigger::cancel_expl(*this, expl_steps_);
//			//zero expl_steps counter
//			expl_steps_ = 0;
		}

		if(dt <= EPS)
			//zero expl_steps counter
			expl_steps_ = 0;
		//show expl_steps
		cout << "expl_steps = " << expl_steps_ << endl;

		//another explosion check
		//if number of centers has grown more than EXPL_MAXGROW times from expl_length_ steps back to current iteration
		//then explosion is detected and we rewind 1 step back
		if(cycle > expl_det_start_ + EXPL1_LENGTH) {
			ulong s = log_.head(EXPL1_LENGTH).y_.size();
			if(s > 1 && double(y_.size()) > double(s * EXPL_MAXGROW)) {
				is_exploded = true;
				rev_steps = 1;
				//ret = expl_trigger::cancel_expl(*this, 1);
			}
		}

		ulong ret = 0;
		//if explosion detected
		if(is_exploded && rev_steps > 0) {
			//check for "false" explosion that can occur from the beginning
			//if(log_.head(rev_steps).y_.size() > 1) {
			//explosion detected
			cout << "EXPLOSION detected! Rolling " << rev_steps << " steps back" << endl;
			//rewind codebook
			y_ = log_.head(rev_steps).y_;
			//}
			//else is_exploded = false;
			expl_steps_ = 0;
			ret = y_.size();
		}

		return ret;
	}

	//! \return - if nonzero then explosion is detected and returned number of clusters should be fixed
	ulong log_step(ulong cycle) {
		//save history
		log_.push_back(*this);
		//display dCnum / dT info
		cout << "-dCnum/dT = " << log_.head_dT() << endl;

		return detect_explosion(cycle);
	}

	//clusterization using deterministic annealing
	void find_clusters(const Matrix& data, const Matrix& f, ulong clust_num, ulong maxiter) {
		px_fcn_ = &da_impl::dithered_px< &da_impl::scaling_px >;
		//px_fcn_ = &da_impl::scaling_px;
		//px_fcn_ = &da_impl::order2px;
		//expl_length_ = 4;
		order_fcn_ = &da_impl::selection_based_order;
		norm_fcn_ = &l2_norm;
		norm2_fcn_ = &l2_norm2;

		//calc statistics for source points distribution
		Matrix dist;
		srcp_stat_ = norm_tools::calc_dist_matrix< norm_tools::l2 >(data, dist);

		//initialization
		clear();
		f_ = f;
		x_ = data;
		cv_info cv1(0, 0);
		cv1.px_.Resize(1, x_.row_num());
		cv1.loc_.Resize(1, x_.col_num());
		//set initial probabilities to 1
		cv1.p_ = 1; cv1.px_ = 1;
		//save first cv
		y_[0] = cv1;

		hcd_.w_.Resize(1, x_.row_num());
		hcd_.norms_.Resize(1, x_.row_num());

		//set position of the first center
		//signal to recalculate static px (if used)
		recalc_px_ = true;
		//get p(x) distribution
		const Matrix& px = (this->*px_fcn_)(0);
		//drop signal
		recalc_px_ = false;
		//all points initially belongs to single center that is placed in the M(x)
		null_step();
		//debug
		//cv1 = y_[0];

		//set initial T > 2 * max variation along principal axis of all data
		T_ = calc_var_honest(*this, 0, px) * 2;
		T_ *= 1.2;
		beta_ = 1 / T_;

		//calc Tfreeze
		//double Tf = T_ / COOL_FACTOR;
		Tmin_ = T_ / COOL_FACTOR;
		//calc Texpl, from which explosion detection starts
		Texpl_ = T_ / EXPL_THRESH_FACTOR;

		//display startup info
		cout << "-------------------------------------------------------------------" << endl;
		cout << "DA startup conditions listed below" << endl;
		cout << "Initial T: " << T_ << endl;
		cout << "Minimum T (main stop condition): " << Tmin_ << endl;
		cout << "Explosion detection starts from T: " << Texpl_ << endl;
		cout << "-------------------------------------------------------------------" << endl << endl;

		//clear history
		log_.clear();
		//hist_.push_back(*this);

		//zero explosion counters
		expl_steps_ = 0;
		expl_det_start_ = 0;
		is_exploded = false;

		//main cycle starts here
		bool spin_update = false;
		for(cycle_ = 0; cycle_ < maxiter; ++cycle_) {
			for(ulong i = 0; i < maxiter; ++i) {
				//update probabilities and centers positions
				do {
					update_epoch();
					//remove nonsignificant centers
					kill_weak_centers(nz_prob(kill_zero_prob_cent_));
					//do_merge
					spin_update = merge_step_classic();
					while(spin_update && merge_step()) {};
				} while(spin_update);

				//convergence test
				if(patience_check(i, hcd_.e_)) break;
				if(hcd_.e_ < EPS) break;
			}
			//perform merge step
			//while(merge_step()) {};

			//update variances
			update_variances();
			kill_weak_centers(nz_var(kill_zero_var_cent_));

			if(log_step(cycle_)) clust_num = y_.size();
			//separate iterations
			cout << endl;

			//check freeze condition
//			if(T_ < Tf) {
//				//simulate explosion
//				is_exploded = true;
//				clust_num = y_.size();
//			}
			//add new centers if nessessary
			if(!phase_transition_epoch(clust_num)) break;
		}	//end of main loop

		//perform last step
		null_step();

		//VERBOSE - save log of dT / dCnum
		log_.dump();

		//recode aff & winners
		id2ind_recode();
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

	//convert id-based winner and affiliation to index-based
	void id2ind_recode() {
		//create id->index map
		map< ulong, ulong > id2ind;
		ulong i = 0;
		for(cv_map::const_iterator p_cv = y_.begin(), end = y_.end(); p_cv != end; ++p_cv, ++i)
			id2ind[p_cv->first] = i;

		//convert affiliation
		aff_map ind_aff;
		i = 0;
		for(aff_map::iterator p_aff = hcd_.aff_.begin(), end = hcd_.aff_.end(); p_aff != end; ++p_aff, ++i)
			ind_aff[id2ind[p_aff->first]] = p_aff->second;
		//replace aff
		hcd_.aff_ = ind_aff;

		//convert winners
		for(i = 0; i < hcd_.w_.size(); ++i)
			hcd_.w_[i] = id2ind[hcd_.w_[i]];
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

Matrix determ_annealing::get_centers() const {
	Matrix y;
	for(cv_map::const_iterator p_cv = pimpl_->y_.begin(), end = pimpl_->y_.end(); p_cv != end; ++p_cv)
		y &= p_cv->second.loc_;
	return y;
}

const ulMatrix& determ_annealing::get_ind() const {
	return pimpl_->hcd_.w_;
}

const Matrix& determ_annealing::get_norms() const {
	return pimpl_->hcd_.norms_;
}

vvul determ_annealing::get_aff() const {
	vvul aff;
	for(aff_map::const_iterator p_cv = pimpl_->hcd_.aff_.begin(), end = pimpl_->hcd_.aff_.end(); p_cv != end; ++p_cv)
		aff.push_back(p_cv->second);

	return aff;
}

void determ_annealing::calc_variances() const {
	pimpl_->update_variances();
}

Matrix determ_annealing::get_variances() const {
	Matrix var;
	for(cv_map::const_iterator p_cv = pimpl_->y_.begin(), end = pimpl_->y_.end(); p_cv != end; ++p_cv)
		var.push_back(p_cv->second.var_, false);

	return var;
}

}	//end of namespace DA
