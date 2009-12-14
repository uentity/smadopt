#include "objnet.h"
#include "polarssl/sha1.h"
#include "polarssl/bignum.h"
#include <map>
#include <string.h>
#include <tbb/tbb.h>

#define HASH_SIZE 20
const unsigned int hash_buf_size = HASH_SIZE / sizeof(t_int);

using namespace std;
using namespace NN;

//-------------------------------falman layer implementation-------------------------------------
struct falman_layer::cache_prop {
	// all const_casts needed to wrap around
	// stupid C iface of polarssl library
	struct mpi_hash : public mpi {
		mpi_hash() {
			s = 0;
			n = hash_buf_size;
			p = buf_;
			memset(buf(), 0, HASH_SIZE);
		}

		mpi_hash(const char* p_hash) {
			s = 0;
			n = hash_buf_size;
			p = buf_;
			memcpy(buf(), p_hash, HASH_SIZE);
		}

		mpi_hash(const mpi_hash& lhs) {
			s = 0;
			n = hash_buf_size;
			p = buf_;
			memcpy(buf(), lhs.buf(), HASH_SIZE);
		}

		//mpi_hash& operator=(const char* p_hash) {
		//	memcpy(buf(), p_hash, HASH_SIZE);
		//}

		bool operator <(const mpi_hash& lhs) const {
			return mpi_cmp_abs(
					const_cast< mpi* >((const mpi*)&lhs),
					const_cast< mpi* >((const mpi*)this)
					) > 0 ? true : false;
		}

		unsigned char* buf() {
			return (unsigned char*)buf_;
		}

		const unsigned char* buf() const {
			return (const unsigned char*)buf_;
		}

		t_int buf_[hash_buf_size];
	};

	struct m_hash {
		m_hash(const Matrix& m) {
			assign(m);
		}

		template< template < class > class buf_traits_t >
		m_hash(const TMatrix< double, buf_traits_t >& m) {
			Matrix tmp;
			tmp = m;
			assign(tmp);
		}

		void assign(const Matrix& m) {
			sha1(
				const_cast< unsigned char* >((const unsigned char*)m.GetBuffer()), 
				m.raw_size(),
				hash_.buf()
				);
		}

		bool operator <(const m_hash& lhs) const {
			return hash_ < lhs.hash_;
		}

		void dump(ostream& outs) const {
			outs << hex;
			for(uint i = 0; i < HASH_SIZE; ++i)
				outs << int(hash_.buf()[i]);
			outs << dec;
		}

		ostream& operator<<(ostream& outs) const {
			dump(outs);
			return outs;
		}

	private:
		mpi_hash hash_;
	};

	typedef map< m_hash, Matrix > cache_t;
	typedef pair< m_hash, bool > find_ret_t;

	cache_t cache_;
	int mode_;
	//falman_layer& l_;
	//ulong ind;
	//int mode, save_mode;

	cache_prop() : mode_(no_cache)
	{}

	//copy constructor
	cache_prop(const cache_prop& cp)
		: cache_(cp.cache_), mode_(cp.mode_)
	{}
	//swaps 2 cache_props
	void swap(cache_prop& cp) {
		std::swap(cache_, cp.cache_);
		std::swap(mode_, cp.mode_);
	}

	cache_prop& operator =(const cache_prop& cp) {
		cache_prop(cp).swap(*this);
		return *this;
	}

	// use input pattern for hashing
	// this may not work properly for NN with feedbacks
	find_ret_t cached_propagate(falman_layer& l) {
		m_hash h(l.net_.get_input().out());
		cache_t::const_iterator res = cache_.find(h);
		if(res != cache_.end()) {
			l.axons_ = res->second;
			return find_ret_t(h, true);
		}
		else
			return find_ret_t(h, false);
	}

	Matrix& ss(const m_hash& h) {
		return cache_[h];
	}

	void save_axons(falman_layer& l) {
		cache_[m_hash(l.net_.get_input().out())] = l.axons_;
	}
};

falman_layer::falman_layer(objnet& net, ulong candidates_count)
	:layer(net, candidates_count, 0), cp_(new cache_prop)
{
	if(candidates_count < MIN_CAND_COUNT)
		init(MIN_CAND_COUNT);
	_construct_aft();
}

falman_layer::falman_layer(objnet& net, const iMatrix& act_fun)
	:layer(net, act_fun), cp_(new cache_prop)
{}

falman_layer::falman_layer(const layer& l)
	: layer(l), cp_(new cache_prop)
{}

//copy constructor
falman_layer::falman_layer(const falman_layer& l)
	:layer(l), cp_(new cache_prop(*l.cp_)), winner_ind_(l.winner_ind_)
{}

void falman_layer::_construct_aft()
{
	vector<int> aft;
	aft.push_back(logsig);
	aft.push_back(tansig);
	aft.push_back(radbas);
	//aft.push_back(revradbas);
	//aft.push_back(multiquad);
	//aft.push_back(revmultiquad);
	for(iMatrix::r_iterator p_aft(aft_.begin()); p_aft != aft_.end(); ++p_aft) {
		*p_aft = aft[prg::randIntUB(aft.size())];
		//*p_aft = radbas;
	}
}

void falman_layer::delete_losers(ulong survivals)
{
	//winner_ind_ is arranged in order of increasing neuron majority
	ulong cand_count = size();
	//if everybody survives - return
	if(cand_count - survivals == 0) return;
	//select dying neurons
	ulMatrix to_kill = winner_ind_.GetColumns(0, cand_count - survivals);
	//sort it in descending order
	to_kill <<= to_kill.Sort(std::greater< ulong >(), true);
	//kill neurons
	for(ulong i = 0; i < to_kill.size(); ++i) {
		neurons_.DelRows(to_kill[i]);
		aft_.DelRows(to_kill[i]);
		B_.DelRows(to_kill[i]);
		BD_.DelRows(to_kill[i]);
		BG_.DelRows(to_kill[i]);
		OBG_.DelRows(to_kill[i]);
	}
	_construct_axons();
	//winner_ind_ = 0;
}

void falman_layer::cache_mode_on()
{
	cp_->mode_ = use_cache;
}

void falman_layer::cache_mode_off()
{
	cp_->cache_.clear();
	cp_->mode_ = no_cache;
}

void falman_layer::propagate()
{
	if(cp_->mode_ == use_cache) {
		cache_prop::find_ret_t res = cp_->cached_propagate(*this);
		// DEBUG
		//cout << "For input " << endl;
		//net_.get_input().out().Print(cout);
		//cout << " hash = "; res.first.dump(cout);
		if(!res.second) {
			layer::propagate();
			cp_->ss(res.first) = axons_;
			//cout << " not found!" << endl;
		}
		//else {
		//	cout << " found!" << endl;
		//}
	}
	else {
		//standart propagation
		layer::propagate();
	}
}
//-------------------------------Cascade correlation network implementation-------------------
class ccn::ccn_impl {
public:
	struct mt_calc_grad {
		falman_layer& fl;
		bool palsy_;
		const Matrix& V;
		const Matrix& S;

		mt_calc_grad(falman_layer& l, const Matrix& Vv, const Matrix& Ss)
			: fl(l), palsy_(true), V(Vv), S(Ss)
		{
			//V = Vv;
			//S = Ss;
		}
		// split ctor
		mt_calc_grad(const mt_calc_grad& lhs, tbb::split)
			: fl(lhs.fl), palsy_(true), V(lhs.V), S(lhs.S)
		{
			//V = lhs.V;
			//S = lhs.S;
		}

		void join(const mt_calc_grad& lhs) {
			palsy_ &= lhs.palsy_;
		}

		void operator()(const tbb::blocked_range< ulong >& r) {
			//now calc gradient for each fl's weight
			const double epsilon = fl.net_.opt_.epsilon;
			bool palsy = true;
			double g;

			//ulong j = 0;
			r_iterator p_b = fl.B_.begin() + r.begin();
			r_iterator p_bg = fl.BG_.begin() + r.begin();
			iMatrix::cr_iterator p_aft = fl.aft_.begin() + r.begin();
			MatrixPtr::r_iterator p_state = fl.states_.begin() + r.begin();
			n_iterator p_n = fl.neurons_.begin() + r.begin();

			for(ulong i = r.begin(); i != r.end(); ++i) {
				g = V.Mul(S.GetColumns(i).sign()).Sum() * p_n->axon_;

				//process biases
				if(fl.net_.opt_.use_biases_) {
					if(*p_aft == radbas || *p_aft == revradbas)
						*p_bg += g * 2 * (*p_b) * (*p_state);
					else if(*p_aft == multiquad || *p_aft == revmultiquad)
						*p_bg += g * 2 * (*p_b);
					else *p_bg += g;

					if(palsy && *p_bg > epsilon) palsy = false;
					++p_bg;
				}

				//process weights
				if(*p_aft == radbas || *p_aft == revradbas)
					g *= 2 * (*p_b) * (*p_b);
				else if(*p_aft == multiquad || *p_aft == revmultiquad)
					g *= 2;

				np_iterator p_in = p_n->inputs_.begin();
				r_iterator p_w = p_n->weights_.begin();
				for(r_iterator p_g = p_n->grad_.begin(); p_g != p_n->grad_.end(); ++p_g) {
					if(*p_aft == radbas || *p_aft == multiquad || *p_aft == revradbas || *p_aft == revmultiquad)
						*p_g += (*p_w - p_in->axon_)*g;
					else
						*p_g += g * p_in->axon_;
					//palsy check
					if(palsy && abs(*p_g) > epsilon) palsy = false;

					++p_in; ++p_w;
				}

				++p_n;
				++p_aft; ++p_b; ++p_state;
			}
			// save palsy state
			palsy_ = palsy;
		}
	};
};

//const Matrix* NN::ccn::ccn_impl::mt_calc_grad::V = NULL;
//const Matrix* NN::ccn::ccn_impl::mt_calc_grad::S = NULL;

ccn::ccn() :
	objnet(new ccn_opt), opt_((ccn_opt&)*opt_holder_), rbfl_(false)
{
	cur_fl_ = NULL;
	//opt_.set_def_opt(false);
}

falman_layer& ccn::add_falman_layer(ulong candidates_count)
{
	//check if no input size specified
	if(inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	flayers_.push_back(smart_ptr<falman_layer>(new falman_layer(*this, candidates_count)), false);
	//get new layer's iterator
	fl_iterator p_newl = flayers_.begin() + (flayers_.size() - 1);
	//fully connect to input
	p_newl->set_links(create_ptr_mat(input_.neurons()));
	if(!opt_.grow_vlayer_) {
		//connect to all previous falman layers
		for(fl_iterator p_l = flayers_.begin(); p_l != p_newl; ++p_l)
			p_newl->add_links(create_ptr_mat(p_l->neurons_));
	}

	return *p_newl;
}

void ccn::add_rb_layer(const Matrix& inputs, const Matrix& targets, const Matrix& centers, double stock_mult)
{
	//first create rb_layer
	smart_ptr< rb_layer > rbl(new rb_layer(*this));
	//now construct neurons
	rbl->construct_drops(inputs, targets, centers, stock_mult);
	//now create first falman layer based on rb_layer
	flayers_.insert(smart_ptr<falman_layer>(new falman_layer(*rbl)), 0, false);
	//get new layer's iterator
	fl_iterator p_newl = flayers_.begin();
	//fully connect to input
	p_newl->set_links(create_ptr_mat(input_.neurons()));
	neurMatrixPtr rbfl_neurons = create_ptr_mat(p_newl->neurons_);
	if(!opt_.grow_vlayer_) {
		//connect to all subsequent falman layers
		for(fl_iterator p_l = p_newl + 1; p_l != flayers_.end(); ++p_l)
			p_l->add_links(rbfl_neurons);
	}

	//set flag
	rbfl_ = true;
}

void ccn::set_output_layer(ulong neurons_count, int af_type, int layer_type) {
	layers_.clear(); flayers_.clear();
	sp_layer outl = add_layer(neurons_count, af_type, layer_type);
	//fully connect to input
	if(!opt_.insert_between_)
		outl->set_links(create_ptr_mat(input_.neurons()));
}

void ccn::propagate()
{
	//first calc falman layers
	for(fl_iterator p_fl = flayers_.begin(); p_fl != flayers_.end(); ++p_fl)
		p_fl->propagate();
	//calc network output
	layers_[0].propagate();
}

void ccn::falman_epoch(const Matrix& inputs, const Matrix& targets)
{
	//double dPerf;
	//calc correlation matrices
	falman_layer& fl(*cur_fl_);
	Matrix E(layers_.begin()->size(), inputs.col_num()), V(fl.size(), inputs.col_num());
	MatrixPtr y = layers_.begin()->out();
	for(ulong i=0; i < inputs.col_num(); ++i) {
		set_input(inputs.GetColumns(i));
		propagate();
		E.SetColumns(targets.GetColumns(i) - y, i);
		V.SetColumns(fl.out(), i);
	}
#ifdef VERBOSE
	DumpMatrix(E, "E.txt");
	DumpMatrix(V, "V.txt");
#endif
	//calculate covariance
	E <<= E.SubMean(E.vMean(true), true);
	V <<= V.SubMean(V.vMean(true), true);
#ifdef VERBOSE
	DumpMatrix(E, "E1.txt");
	DumpMatrix(V, "V1.txt");
#endif
	//calc S matrix
	Matrix S = E * (!V);
	//calc final correlation for each candidate neuron
	for(ulong i = 0; i < fl.size(); ++i)
		fl.Goal_[i] = S.GetColumns(i).Abs().Sum();
	state_.perf = fl.Goal_.Max();
	//fl.winner_ind_ = fl.Goal_.ElementInd(state_.perf);

	//now calc gradient for each fl's weight
	double g;
	Matrix E_col;
	bool palsy = true;
	r_iterator p_b, p_w;
	iMatrix::r_iterator p_aft;
	mp_iterator p_state;
	np_iterator p_in;
	for(ulong p = 0; p < inputs.col_num(); ++p) {
		set_input(inputs.GetColumns(p));
		propagate();
		fl.deriv_af();

		E_col = E.GetColumns(p);
		ccn_impl::mt_calc_grad cg(fl, E_col, S);
		//cg.V = &E_col;
		tbb::parallel_reduce(tbb::blocked_range< ulong >(0, fl.neurons_.size()), cg);
		palsy &= cg.palsy_;

		//ulong j = 0;
		//V <<= E.GetColumns(p);
		//p_aft = fl.aft_.begin(); p_b = fl.B_.begin(); p_state = fl.states_.begin();
		//for(n_iterator p_n = fl.neurons_.begin(); p_n != fl.neurons_.end(); ++p_n) {
		//	g = V.Mul(S.GetColumns(j).sign()).Sum() * p_n->axon_;

		//	//process biases
		//	if(opt_.use_biases_) {
		//		if(*p_aft == radbas || *p_aft == revradbas)
		//			fl.BG_[j] += g * 2 * (*p_b) * (*p_state);
		//		else if(*p_aft == multiquad || *p_aft == revmultiquad)
		//			fl.BG_[j] += g * 2 * (*p_b);
		//		else fl.BG_[j] += g;

		//		if(palsy && fl.BG_[j] > opt_.epsilon) palsy = false;
		//	}

		//	//process weights
		//	if(*p_aft == radbas || *p_aft == revradbas)
		//		g *= 2 * (*p_b) * (*p_b);
		//	else if(*p_aft == multiquad || *p_aft == revmultiquad)
		//		g *= 2;

		//	p_in = p_n->inputs_.begin();
		//	p_w = p_n->weights_.begin();
		//	for(r_iterator p_g = p_n->grad_.begin(); p_g != p_n->grad_.end(); ++p_g) {
		//		if(*p_aft == radbas || *p_aft == multiquad || *p_aft == revradbas || *p_aft == revmultiquad)
		//			*p_g += (*p_w - p_in->axon_)*g;
		//		else
		//			*p_g += g * p_in->axon_;
		//		//palsy check
		//		if(palsy && abs(*p_g) > opt_.epsilon) palsy = false;

		//		++p_in; ++p_w;
		//	}

		//	++j;
		//	++p_aft; ++p_b; ++p_state;
		//}
	}

	if(state_.cycle == 0 && opt_.learnFun == R_BP) {
		fl.OBG_ = fl.BG_;
		for(n_iterator p_n = fl.neurons_.begin(); p_n != fl.neurons_.end(); ++p_n)
			p_n->prevg_ = p_n->grad_;
	}

	if(palsy)
		state_.status = stop_palsy;
}

void ccn::prepare2learn()
{
	switch(mainState_.status) {
		case ccn_fully_bp:
			for(flMatrix::r_iterator p_l = flayers_.begin(); p_l != flayers_.end(); ++p_l) {
				p_l->_prepare2learn< anti_grad >();
				p_l->backprop_lg_ = true;
				p_l->Goal_ = 0;
			}
			if(flayers_num() > 0) flayers_[0].backprop_lg_ = false;
		default:
		case learning:
		case ccn_bp:
			objnet::prepare2learn();
			if(cur_fl_) {
				layers_[0].backprop_lg_ = true;
				cur_fl_->_prepare2learn< anti_grad >();
				cur_fl_->backprop_lg_ = false;
				cur_fl_->Goal_ = 0;
			}
			break;
		case ccn_maxcor:
			if(cur_fl_) {
				cur_fl_->_prepare2learn< follow_grad >();
				cur_fl_->backprop_lg_ = false;
				cur_fl_->Goal_ = 0;
			}
			break;
	}

	if(opt_.use_lsq && mainState_.status == learning) {
		//disable updates in last layer
		layers_[layers_num() - 1]._pUpdateFun = &layer::empty_update;
	}
}

//override learn epoch
void ccn::learn_epoch(const Matrix& inputs, const Matrix& targets) {
	switch(mainState_.status) {
		default:
		case ccn_bp:
		case ccn_fully_bp:
			objnet::learn_epoch(inputs, targets);
			break;
		case learning:
			if(opt_.use_lsq)
				objnet::lsq_epoch(inputs, targets);
			else
				objnet::bp_epoch(inputs, targets);
			break;
		case ccn_maxcor:
			falman_epoch(inputs, targets);
			break;
	}
}

//custom learn stop function
void ccn::is_goal_reached() {
	switch(mainState_.status) {
		default:
		case ccn_fully_bp:
			objnet::is_goal_reached();
			break;
		case learning:
			if(opt_.use_lsq) {
				if(state_.perf < opt_.goal)
					state_.status = learned;
				else
					state_.status = stop_patience;
			}
			else
				objnet::is_goal_minimized();
			break;
		case ccn_bp:
			//if(state_.perf < opt_.goal)
			//	state_.status = learned;
			//break;
		case ccn_maxcor:
			if(state_.cycle >= opt_.maxFLLcycles_)
				state_.status = stop_maxcycle;
			break;
	}
}

int ccn::check_patience(nnState& state, double patience, ulong patience_cycles, int patience_status)
{
	if(mainState_.status != ccn_maxcor)
		return objnet::_check_patience< anti_grad >(state, patience, patience_cycles, patience_status);
	else
		return objnet::_check_patience< follow_grad >(state, patience, patience_cycles, patience_status);
}

void ccn::bp_after_grad()
{
	//calculate gradient for falman layers
	if(mainState_.status == ccn_fully_bp && flayers_num() > 0) {
		for(flMatrix::r_iterator p_l = flayers_.end() - 1; p_l >= flayers_.begin(); --p_l) {
			p_l->calc_grad();
			p_l->Goal_ = 0;
		}
	}
	else if(cur_fl_) {
		cur_fl_->calc_grad();
		//zero goal for next epoch
		cur_fl_->Goal_ = 0;
	}
}

void ccn::update_epoch()
{
	switch(mainState_.status) {
		case ccn_fully_bp:
			for(flMatrix::r_iterator p_l = flayers_.begin(); p_l != flayers_.end(); ++p_l)
				p_l->update_epoch();
			objnet::update_epoch();
			break;

		default:
		case ccn_bp:
		case learning:
			objnet::update_epoch();

		case ccn_maxcor:
			if(cur_fl_) cur_fl_->update_epoch();
			break;
	}
}

void ccn::_calc_winner_maxw()
{
	//winner is neuron with most strong connection to output layer
	neurMatrixSP& o_neurons = (neurMatrixSP&)layers_.begin()->neurons();
	//double max_ws = 0, cur_ws;
	//ulong wi, winner_ind = 0;
	//Matrix cur_w(o_neurons.size(), 1), winner_w;
	ulong cur_n = 0, syn_ind;
	Matrix rank(1, cur_fl_->neurons().size());
	for(n_iterator p_n = cur_fl_->neurons_.begin(), end = cur_fl_->neurons_.end(); p_n != end; ++p_n) {
		//wi = 0;
		rank[cur_n] = 0;
		for(n_iterator p_on = o_neurons.begin(), o_end = o_neurons.end(); p_on != o_end; ++p_on) {
			if((syn_ind = p_on->inputs_.BufElementInd(&(*p_n))) < p_on->inputs_.size())
				rank[cur_n] += abs(p_on->weights_[syn_ind]);
				//cur_w[wi++] = p_on->weights_[syn_ind];
				//cur_ws += abs(p_on->weights_[syn_ind]);
		}
		//if((cur_ws = cur_w.Abs().Sum()) > max_ws) {
		//	max_ws = cur_ws;
		//	winner_w = cur_w;
		//	winner_ind = cur_n;
		//}
		++cur_n;
	}

	//delete all links from last layer
	layers_.begin()->rem_links(create_ptr_mat(cur_fl_->neurons_));

	//cur_fl_->winner_ind_ = winner_ind;
	cur_fl_->winner_ind_ <<= rank.RawSort();
	cur_fl_->delete_losers(opt_.fl_candidates_survive_);

	//connect winners to last layer
	layers_.begin()->add_links(create_ptr_mat(cur_fl_->neurons_));

	//add links to winner with saved weights
	//neurMatrixPtr syn = create_ptr_mat(cur_fl_->neurons_);
	//for(ulong i = 0; i < o_neurons.size(); ++i)
	//	o_neurons[i].add_synapses(syn, &winner_w.GetColumns(i));
}

void ccn::_calc_winner_corr(const Matrix& inputs, const Matrix& targets)
{
	//delete connections to new falman layer from last layer
	layers_.begin()->rem_links(create_ptr_mat(cur_fl_->neurons_));

	//calc correlation matrices
	falman_layer& fl(*cur_fl_);
	Matrix E(layers_.begin()->size(), inputs.col_num()), V(fl.size(), inputs.col_num());
	MatrixPtr y = layers_.begin()->out();
	for(ulong i=0; i < inputs.col_num(); ++i) {
		set_input(inputs.GetColumns(i));
		propagate();
		E.SetColumns(targets.GetColumns(i) - y, i);
		V.SetColumns(fl.out(), i);
	}
	//calculate covariance
	E <<= E.SubMean(E.vMean(true), true);
	V <<= V.SubMean(V.vMean(true), true);
	//calc S matrix
	Matrix S = E * (!V);
	//calc final correlation for each candidate neuron
	for(ulong i = 0; i < fl.size(); ++i)
		fl.Goal_[i] = S.GetColumns(i).Abs().Sum();
	//sort winners
	fl.winner_ind_ <<= fl.Goal_.RawSort();
	//fl.winner_ind_ = fl.Goal_.ElementInd(fl.Goal_.Max());

	//delete losers
	cur_fl_->delete_losers(opt_.fl_candidates_survive_);

	//connect winners to last layer
	layers_.begin()->add_links(create_ptr_mat(cur_fl_->neurons_));

	//extract weight matrix
	//winner_w.NewMatrix(o_neurons.size(), 1);
	//neuron* p_winner = &fl.neurons_[winner_ind];
	//ulong wi =  0, syn_ind;
	//for(n_iterator p_on = o_neurons.begin(); p_on != o_neurons.end(); ++p_on) {
	//	if((syn_ind = p_on->inputs_.BufElementInd(p_winner)) < p_on->inputs_.size())
	//		winner_w[wi++] = p_on->weights_[syn_ind];
	//}
}

bool ccn::delete_losers(const Matrix& inputs, const Matrix& targets)
{
	if(cur_fl_ == NULL) return false;
	if(mainState_.status != ccn_maxcor) {
		//cur_fl_->winner_ind_ = 0;
		//_calc_winner_maxw();
		_calc_winner_corr(inputs, targets);
		//return false;
	}
	else {
		cur_fl_->winner_ind_ <<= cur_fl_->Goal_.RawSort();
		cur_fl_->delete_losers(opt_.fl_candidates_survive_);
		layers_.begin()->add_links(create_ptr_mat(cur_fl_->neurons_));
	}
	return true;
}

int ccn::learn(const Matrix& inputs, const Matrix& targets, bool initialize, pLearnInformer pProc, 
		smart_ptr< const Matrix > test_inp, smart_ptr< const Matrix > test_tar)
{
	int save_lsq_stat = opt_.use_lsq;

	// prepare learn & validation sets
	Matrix real_inputs;
	Matrix real_targets;
	prep_learn_valid_sets(inputs, targets, test_inp, test_tar, real_inputs, real_targets);

	// save & reset test_validation bit
	unsigned int save_test_valid = opt_.goal_checkFun & test_validation;
	// TODO: remove this ugly thing after proper goal_checkFun reading
	if(save_test_valid)
		opt_.goal_checkFun = patience;
	else
		opt_.goal_checkFun ^= save_test_valid;

	try {
		//check if learning can start
		if(layers_num() == 0 || layers_[0].size() == 0)
			throw nn_except("An output layer must be created before learning can start");
		//if(mainState_.status == learning || mainState_.status == opt_.learnType)
		//	throw nn_except(NN_Busy, "This network is already in learning state");

		if(initialize) reset();

		//if(opt_.learnFun != ccn_fully_bp) {
		//	//turn on caching in prev falman layers
		//	for(fl_iterator p_fl = flayers_.begin(); p_fl != flayers_.end(); ++p_fl)
		//		p_fl->cache_mode_on();
		//}

		bool learn_outl = true;
		if(opt_.insert_between_) {
			layers_.begin()->init_weights(real_inputs);
			if(flayers_num() == 0) learn_outl = false;
		}

		//main learn cycle
		mainState_.lastPerf = 0;
		mainState_.cycle = 0;
		if(save_test_valid)
			check_early_stop(mainState_, *test_inp, *test_tar);

		for(++mainState_.cycle; mainState_.cycle <= opt_.maxFL_; ++mainState_.cycle) {
			//learn last layer
			if(learn_outl) {
				//check if first falman layer is pre-creted
			//	if(rbfl_) {
			//		mainState_.status = NN::ccn_fully_bp;
			//		rbfl_ = false;
			//	}
			//	else
					mainState_.status = learning;
				cur_fl_ = NULL;
				if(mainState_.cycle == 0) {
					layers_.begin()->init_weights(real_inputs);
				}

				//layers_.begin()->init_weights(real_inputs);
				//init falman layers weights
				//for(fl_iterator p_fl = flayers_.begin(); p_fl != flayers_.end(); ++p_fl)
				//	p_fl->init_weights(real_inputs);

				common_learn(real_inputs, real_targets, false, pProc);
				mainState_.perf = state_.perf;

				// drop rbfl_ flag
				rbfl_ = false;
				if(state_.status == learned || state_.status == stop_breaked) {
					mainState_.status = state_.status;
					break;
				}

				//lsq can become unstable. If performance isn't improving during last opt_.lsq_patience_cycles epochs
				//then turn it off

				if(opt_.use_lsq) {
					_check_patience< anti_grad >(mainState_, opt_.patience, opt_.lsq_patience_cycles_);
					if(mainState_.status == stop_patience)
						opt_.use_lsq = false;
				}

				if(save_test_valid) {
					mainState_.patience_counter = check_early_stop(mainState_, *test_inp, *test_tar);
					// print xvalidation status
					cout << xvalid_info() << endl;
					if(mainState_.status == stop_test_validation)
						break;
				}
			}

			//error still high - add new falman_layer
			mainState_.status = opt_.learnType;
			//add new layer
			cur_fl_ = &add_falman_layer(opt_.fl_candidates_);
			if(mainState_.status != ccn_maxcor && opt_.insert_between_) {
				//connect output layer to this new layer
				layers_.begin()->add_links(create_ptr_mat(cur_fl_->neurons_));
			}
			//cur_fl_->Goal_ = 0;
			cur_fl_->init_weights(real_inputs);

			//if(!opt_.batch) {
			//	//turn on caching in prev falman layers
			//	for(fl_iterator p_fl = flayers_.begin(); &(*p_fl) != cur_fl_; ++p_fl)
			//		p_fl->cache_mode_on(real_inputs.col_num());
			//}
			common_learn(real_inputs, real_targets, false, pProc);

			//select the most performing element, kill others
			if(!(learn_outl = delete_losers(real_inputs, real_targets))) {
				mainState_.perf = state_.perf;
				if(state_.status == learned || state_.status == stop_breaked) {
					mainState_.status = state_.status;
					break;
				}
			}

			// we should disable caching only in case of
			// fully_bp learning, because it's weights won't be fixed
			if(opt_.learnType == ccn_fully_bp)
				cur_fl_->cache_mode_off();
			else
				cur_fl_->cache_mode_off();

			//connect output layer to this new layer
			//layers_.begin()->add_links(create_ptr_mat(cur_fl_->neurons_));

			//if(opt_.batch) {
			//	cur_fl_->cache_mode_on(real_inputs.col_num());
			//}
			//else {
			//	//pause caching in prev falman layers
			//	for(fl_iterator p_fl = flayers_.begin(); &(*p_fl) != cur_fl_; ++p_fl)
			//		p_fl->cache_mode_pause();
			//}

#if 0
			Matrix ol, fl, er;
			ol.reserve(real_targets.col_num());
			fl.reserve(real_targets.col_num());
			er.reserve(real_targets.col_num());
			MatrixPtr res = layers_.begin()->out();
			for(ulong i=0; i<real_inputs.col_num(); ++i) {
				set_input(real_inputs.GetColumns(i));
				propagate();
				ol.push_back(res[0]);
				fl.push_back(cur_fl_->out()[0]);
				er.push_back(real_targets[i] - res[0]);
			}
			DumpMatrix(ol | fl | er, "corr.txt");
			//DumpMatrix(fl, "fl.txt");
			//DumpMatrix(er, "er.txt");
#endif
		}

		//if(mainState_.status != learned && state_.status != stop_breaked)
		//	mainState_.status = stop_maxcycle;

		//call informer
		if(pProc) pProc(mainState_.cycle, mainState_.perf, (void*)this);

		if(mainState_.status == stop_test_validation) {
			cout << "Remove Fahlman layers added during patience_counter cycles" << endl;
			ulong reduced_sz = flayers_.size() - mainState_.patience_counter;
			layer& outl = layers_[layers_.size() - 1];
			for(ulong i = reduced_sz; i < flayers_.size(); ++i)
				outl.rem_links(create_ptr_mat(flayers_[i].neurons()));
			flayers_.DelRows(reduced_sz, mainState_.patience_counter);

			cout << "Relearn last layer" << endl;
			mainState_.status = learning;
			common_learn(real_inputs, real_targets, false, pProc);
			mainState_.perf = state_.perf;
		}

		//turn off caching everywhere
		for(fl_iterator p_fl = flayers_.begin(); p_fl != flayers_.end(); ++p_fl)
			p_fl->cache_mode_off();
	}
	//errors handling
	catch(alg_except& ex) {
		mainState_.status = error;
		_print_err(ex.what());
		throw;
	}
	catch(exception ex) {
		mainState_.status = error;
		_print_err(ex.what());
		throw nn_except(ex.what());
	}
	catch(...) {
		mainState_.status = error;
		_print_err("Unknown run-time exception thrown");
		throw nn_except(-1, state_.lastError.c_str());
	}

	opt_.use_lsq = save_lsq_stat;
	return state_.status;
}

void ccn::_init_fl_radbas(const Matrix& inputs)
{
	Matrix m = inputs.vMean(true), std = inputs.vStd(true);
	Matrix div(inputs.row_num(), 1);
	for(n_iterator p_n = cur_fl_->neurons_.begin(); p_n != cur_fl_->neurons_.end(); ++p_n) {
		generate(div.begin(), div.end(), prg::rand01);
		div -= 0.5; div *= 2;
		div *= std; div += m;
		p_n->weights_ = div;
	}
}

void ccn::reset()
{
	//delete links to falman layers
	layer& outl(*layers_.begin());
	if(!opt_.insert_between_) {
		outl.set_links(create_ptr_mat(input_.neurons()));
	}
	if(rbfl_ && flayers_.size() > 0) outl.set_links(create_ptr_mat(flayers_.begin()->neurons()));

	//delete falman layers
	if(!rbfl_)
		//remove falman layers
		flayers_.clear();
	else
		//save only first falman layer
		flayers_.Resize(1, 1, smart_ptr< falman_layer >());
}

std::string ccn::status_info(int level) const {
	ostringstream os;
	string stop_state = decode_nn_state(state_.status);
	if(mainState_.status == learning) {
		// outpul layer learning status
		if(state_.cycle == 1) {
			if(rbfl_)
				os << "Initial Radial-Basis Network learning started";
			else
				os << "Output layer learning started";
			os << endl;
		}
		os << objnet::status_info();
		if(stop_state != "")
			os << "Output layer: " << stop_state << endl << endl;
	}
	else {
		// Fahlman layer learning status
		if(state_.status == learning) {
			if(state_.cycle == 1)
				os << "Error still high - adding new falman layer #" << flayers_num() << endl;
			os << "cycle " << state_.cycle << ", goal " << state_.perf << endl;
		}
		// Result of fahlman layer learning
		if(stop_state != "") {
			os << "Fahlman layer: " << stop_state << endl;
			if(mainState_.status == ccn_maxcor) {
				os << "The winner is: " << decode_neuron_type(cur_fl_->aft()[cur_fl_->get_winner_ind()]);
				os << " neuron " << endl << endl;
			}
			else os << endl;
		}
		// Overall NN learning status
		if(mainState_.status != opt_.learnType) {
			os << "We end up with " << flayers_num() << " falman layers" << endl;
			//stop_state = decode_nn_state(mainState_.status);
			//os << "CCN leraning stopped: " << stop_state << endl << endl;
		}
	}

	// print resulting status of main CCN
	if((stop_state = decode_nn_state(mainState_.status)) != "")
		os << "CCN: " << decode_nn_state(mainState_.status) << endl << endl;

	return os.str();
}

text_table ccn::detailed_info(int level) const {
	struct fmt_layer_info {
		static text_table& go(text_table& tt, const layer& l, int level) {
			// decode layer's info about neuron types
			TMatrix< string > neur_info = decode_neuron_type(l.aft(), level > 2 ? false : true).content();
			for(ulong j = 0; j < neur_info.row_num(); ++j) {
				tt << "&" << decode_layer_type(l.layer_type()) << "&" << neur_info(j, 0);
				tt << "&" << neur_info(j, 1) << tt_endr();
			}
			tt << "\\hline" << tt_endr();
			return tt;
		}
	};

	if(!level) return objnet::detailed_info(level);

	text_table tt;
	tt.fmt().sep_cols = true;
	tt.fmt().align = 2;

	// deep level of information about network
	// display table with detailed information about layers
	tt << tt_begh() << "| 0 | 0 | 0 | 0 |" << tt_endrh();
	tt << "Layer # & Layer type & Neurons num & Neuron types" << tt_endrh();
	// display fahlman layers
	ulong cnt = 0;
	for(ulong i = 0; i < flayers_.size(); ++i, ++cnt) {
		tt << cnt;
		fmt_layer_info::go(tt, flayers_[i], level);
	}
	// display other layers
	for(ulong i = 0; i < layers_.size(); ++i, ++cnt) {
		tt << cnt;
		fmt_layer_info::go(tt, layers_[i], level);
	}
	return tt;
}

