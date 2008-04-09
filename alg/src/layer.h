#include "prg.h"
#include "objnet.h"
#include <cmath>

using namespace std;
//using namespace NN;

namespace NN {
//--------------------------------layer  class implementation---------------------------------

layer::layer(objnet& net, ulong neurons_count, int af_type) :
	net_(net) //, neurons_(neurons_count, 1), aft_(neurons_count, 1),
	//B_(neurons_count, 1), BD_(neurons_count, 1)
{
	init(neurons_count, af_type);
	//aft_ = af_type;
	//_construct_axons();
}

layer::layer(objnet& net, const iMatrix& act_fun) :
	net_(net) //, neurons_(act_fun.size(), 1),
	//B_(act_fun.size(), 1), BD_(act_fun.size(), 1)
{
	init(act_fun.size(), logsig, 0, &act_fun);
	//aft_ = act_fun;
	//_construct_axons();
}

//copy constructor - reference semantics
layer::layer(const layer& l) :
	neurons_(l.neurons_), aft_(l.aft_),
	B_(l.B_), BD_(l.BD_), BG_(l.BG_), OBG_(l.OBG_),
	net_(l.net_), backprop_lg_(l.backprop_lg_)
{
	_construct_axons();
}

layer& layer::operator =(const layer& l)
{
	//memcpy((void*)&opt_, &l.opt_, sizeof(nnOptions));
	if(&net_ == &l.net_) {
		aft_ = l.aft_;
		B_ = l.B_; BD_ = l.BD_; BG_ = l.BG_; OBG_ = l.OBG_;
		backprop_lg_ = l.backprop_lg_;
		//neurons should be really copied
		neurons_.clear();
		smart_ptr<neuron> sp_dst;
		for(cn_iterator sp_src = l.neurons_.begin(); sp_src != l.neurons_.end(); ++sp_src) {
			sp_dst = new neuron;
			*sp_dst = *sp_src;
			neurons_.push_back(sp_dst);
		}
		_construct_axons();
	}
	return *this;
}

void layer::_construct_axons()
{
	if(axons_.size() != neurons_.size()) {
		axons_.NewMatrix(neurons_.size(), 1);
		states_.NewMatrix(neurons_.size(), 1);
		Goal_.NewMatrix(neurons_.size(), 1);
	}
	MatrixPtr::buf_iterator p_axon(axons_.buf_begin()), p_goal(Goal_.buf_begin()), p_state(states_.begin());
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		*p_axon = &p_n->axon_;
		*p_goal = &p_n->error_;
		*p_state = &p_n->state_;
		++p_axon; ++p_goal; ++p_state;
	}
}

void layer::init(ulong neurons_count, int af_type, ulong inp_count, const iMatrix* p_af_mat)
{
	if(neurons_.size() != neurons_count) {
		aft_.NewMatrix(neurons_count, 1);
		B_.NewMatrix(neurons_count, 1);
		BD_.NewMatrix(neurons_count, 1);
		if(neurons_.size() > neurons_count)
			neurons_.DelRows(neurons_count, neurons_.size() - neurons_count);
		else {
			//add neurons
			while(neurons_.size() < neurons_count)
				neurons_.push_back(smart_ptr<neuron>(new neuron(inp_count, af_type)));
		}
		_construct_axons();
	}
	if(p_af_mat && p_af_mat->size() == neurons_count)
		aft_ = *p_af_mat;
	else aft_ = af_type;
}

//fully connect to inputs
void layer::set_links(const neurMatrixPtr& inputs)
{
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n)
		p_n->set_synapses(inputs);
}

//connect using connection matrix
//rows - neurons, columns - input elements
void layer::set_links(const neurMatrixPtr& inputs, const bitMatrix& con_mat)
{
	neurMatrixPtr syn;
	bitMatrix con_row;
	ulong i = 0, con_ind;
	syn.reserve(con_mat.col_num());
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		syn.Resize(inputs.size(), 1);
		con_ind = 0;
		for(ulong j=0; j < inputs.size(); ++j) {
			if(con_mat(i, j))
				syn.at_buf(con_ind++) = inputs.at_buf(j);
		}
		syn.Resize(con_ind - 1);
		p_n->set_synapses(syn);
		++i;
	}
}

void layer::add_links(const neurMatrixPtr& inputs, const Matrix* p_weights, ulong neur_ind)
{
	if(p_weights) {
		if(neur_ind < neurons_.size())
			neurons_[neur_ind].add_synapses(inputs, p_weights);
		else {
			for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n)
				p_n->add_synapses(inputs, p_weights);
		}
	}
	else {
		//randomly init new weights
		Matrix nw(inputs.size(), 1);
		if(neur_ind < neurons_.size()) {
			generate(nw.begin(), nw.end(), prg::rand01);
			nw -= 0.5; nw *= net_.opt_.wiRange;
			neurons_[neur_ind].add_synapses(inputs, &nw);
		}
		else {
			for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
				generate(nw.begin(), nw.end(), prg::rand01);
				nw -= 0.5; nw *= net_.opt_.wiRange;
				p_n->add_synapses(inputs, &nw);
			}
		}
	}
}

void layer::rem_links(const neurMatrixPtr& inputs, ulong neur_ind)
{
	if(neur_ind < neurons_.size())
		neurons_[neur_ind].rem_synapses(inputs);
	else {
		for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n)
			p_n->rem_synapses(inputs);
	}
}

neuron& layer::add_neuron(int af_type, ulong inp_num)
{
	smart_ptr<neuron> p_n(new neuron(inp_num, af_type));
	neurons_.push_back(p_n);

	aft_.push_back(af_type);
	B_.push_back(0);
	BD_.push_back(0);
	//construct pointers
	axons_.push_back(&p_n->axon_);
	states_.push_back(&p_n->state_);
	Goal_.push_back(&p_n->error_);

	if(BG_.size() == size() - 1) {
		BG_.push_back(0); OBG_.push_back(0);
	}
	return *p_n;
}

neuron& layer::add_neuron(int af_type, const neurMatrixPtr& inputs)
{
	neuron& n = add_neuron(af_type, inputs.size());
	n.set_synapses(inputs);
	return n;
}

void layer::rem_neuron(ulong ind)
{
	if(ind >= size()) return;
	neurons_.DelRows(ind);
	aft_.DelRows(ind);
	B_.DelRows(ind);
	BD_.DelRows(ind);
	//delete pointers
	axons_.DelRows(ind);
	states_.DelRows(ind);
	Goal_.DelRows(ind);

	if(BG_.size() == size() + 1) {
		BG_.DelRows(ind); OBG_.DelRows(ind);
	}
}

/*
void layer::activate()
{
	iMatrix::r_iterator p_aft = aft_.begin();
	for(mp_iterator p_axon = axons_.begin(); p_axon != axons_.end(); ++p_axon) {
		switch(*p_aft)
		{
		case tansig:
			*p_axon = net_.opt_.tansig_a*tanh(net_.opt_.tansig_b*(*p_axon));
			if(net_.opt_.saturate) {
				*p_axon = min(*p_axon, net_.opt_.tansig_a - net_.opt_.tansig_e);
				*p_axon = max(*p_axon, net_.opt_.tansig_e - net_.opt_.tansig_a);
			}
			break;
		case logsig:
			*p_axon = 1/(1 + exp(-net_.opt_.logsig_a*(*p_axon)));
			if(net_.opt_.saturate) {
				*p_axon = min(*p_axon, 1 - net_.opt_.tansig_e);
				*p_axon = max(*p_axon, net_.opt_.tansig_e - 1);
			}
			break;
		case poslin:
			*p_axon = max(*p_axon, 0);
			break;
		case radbas:
			*p_axon = exp(-(*p_axon)*(*p_axon));
			break;
		}
		++p_aft;
	}
}
*/

Matrix layer::active_af_region()
{
	Matrix ret; ret.reserve(2*neurons_.size());
	Matrix res(1, 2);
	double a = 0, b = 0;
	iMatrix::r_iterator p_aft = aft_.begin();
	for(n_iterator p_n(neurons_.begin()); p_n != neurons_.end(); ++p_n) {
		switch(*p_aft) {
			case logsig:
				a = -net_.opt_.logsig_a*log(1./net_.opt_.logsig_e - 1);
				b = -net_.opt_.logsig_a*log(1./(1 - net_.opt_.logsig_e) - 1);
				break;
			case tansig:
				a = 1./(2*net_.opt_.tansig_b)*log(net_.opt_.tansig_e/(2*net_.opt_.tansig_a - net_.opt_.tansig_e));
				b = 1./(2*net_.opt_.tansig_b)*log((2*net_.opt_.tansig_a - net_.opt_.tansig_e)/net_.opt_.tansig_e);
				break;
		}

		if(b > a) {
			res[0] = a; res[1] = b;
		}
		else {
			res[0] = b; res[1] = a;
		}
		ret &= res;
	}
	return ret;
}

void layer::propagate()
{
	r_iterator p_b = B_.begin();
	iMatrix::r_iterator p_aft = aft_.begin();
	mp_iterator p_axon = axons_.begin(), p_state = states_.begin();
	for(n_iterator p_n(neurons_.begin()); p_n != neurons_.end(); ++p_n) {
		switch(*p_aft)
		{
		case tansig:
			p_n->weighted_sum_sf();
			*p_axon = net_.opt_.tansig_a*tanh(net_.opt_.tansig_b*(*p_state + *p_b));
			if(net_.opt_.saturate) {
				*p_axon = min(*p_axon, net_.opt_.tansig_a - net_.opt_.tansig_e);
				*p_axon = max(*p_axon, net_.opt_.tansig_e - net_.opt_.tansig_a);
			}
			break;
		case logsig:
			p_n->weighted_sum_sf();
			*p_axon = 1/(1 + exp(-net_.opt_.logsig_a*(*p_state + *p_b)));
			if(net_.opt_.saturate) {
				*p_axon = min(*p_axon, 1 - net_.opt_.tansig_e);
				*p_axon = max(*p_axon, net_.opt_.tansig_e - 1);
			}
			break;
		default:
		case purelin:
			p_n->weighted_sum_sf();
			*p_axon = *p_state + *p_b;
			break;
		case poslin:
			p_n->weighted_sum_sf();
			*p_axon = std::max<double>(*p_state + *p_b, 0);
			break;
		case radbas:
			p_n->eucl_dist_sf();
			//bias contains sigma^-1
			*p_axon = exp(-(*p_state)*(*p_b)*(*p_b));
			break;
		case revradbas:
			p_n->eucl_dist_sf();
			//bias contains sigma^-1
			*p_axon = 1 - exp(-(*p_state)*(*p_b)*(*p_b));
			break;
		case expws:
			p_n->weighted_sum_sf();
			*p_axon = exp(*p_state + *p_b);
			break;
		case multiquad:
			p_n->eucl_dist_sf();
			*p_axon = sqrt(*p_state + (*p_b)*(*p_b));
			break;
		case revmultiquad:
			p_n->eucl_dist_sf();
			*p_axon = 1./sqrt(*p_state + (*p_b)*(*p_b));
			break;
		}

		++p_aft; ++p_b;
		++p_state; ++p_axon;
	}
	//activate();
}

void layer::deriv_af()
{
	iMatrix::r_iterator p_aft = aft_.begin();
	mp_iterator p_state = states_.begin();
	r_iterator p_b = B_.begin();
	for(mp_iterator p_axon = axons_.begin(); p_axon != axons_.end(); ++p_axon) {
		switch(*p_aft)
		{
		case tansig:
			*p_axon = net_.opt_.tansig_b*(net_.opt_.tansig_a - *p_axon)*(net_.opt_.tansig_a + *p_axon)/net_.opt_.tansig_a;
			break;
		case logsig:
			*p_axon *= net_.opt_.logsig_a*(1 - *p_axon);
			break;
		case poslin:
			if(*p_axon < 0) *p_axon = 0;
			else *p_axon = 1;
			break;
		case purelin:
			*p_axon = 1;
			break;
		case radbas:
			//exp(-x)' = - exp(-x)
			*p_axon = - *p_axon;
			break;
		case revradbas:
			//(1 - exp(-x))' = exp(-x)
			*p_axon = 1 - *p_axon;
			break;
		case expws:
			*p_axon *= (*p_state + *p_b);
			break;
		case multiquad:
			if(*p_axon != 0) *p_axon = 1. / (*p_axon * 2);
			break;
		case revmultiquad:
			if(*p_axon != 0) {
				double r = *p_state + (*p_b)*(*p_b);
				*p_axon = - 1. / (sqrt(r*r*r) * 2);
			}
			break;
		}
		++p_aft; ++p_state;
		++p_b;
	}
}

void layer::init_weights_random()
{
	//biases
	if(net_.opt_.use_biases_) {
		generate(B_.begin(), B_.end(), prg::rand01);
		B_ -= 0.5; B_ *= net_.opt_.wiRange;
	}
	//else B_ = 0;
	//neurons
	for(n_iterator p_n(neurons_.begin()); p_n != neurons_.end(); ++p_n) {
		generate(p_n->weights_.begin(), p_n->weights_.end(), prg::rand01);
		p_n->weights_ -= 0.5; p_n->weights_ *= net_.opt_.wiRange;
	}
}

void layer::init_weights_nw()
{
}

template< >
void layer::init_weights_radbas<true>(const Matrix& inputs)
{
	//distribute centers randomly in volume
	Matrix mm = inputs.minmax(true);
	Matrix vol = mm.GetColumns(1) - mm.GetColumns(0);
	mm <<= mm.GetColumns(0);
	//deviation
	Matrix dev(inputs.row_num(), 1);
	r_iterator p_b = B_.begin();
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		generate(dev.begin(), dev.end(), prg::rand01);
		dev *= vol; dev += mm;
		p_n->weights_ = dev;
		//set sigma^-1 = 1
		*p_b = 1; ++p_b;
	}
}

template< >
void layer::init_weights_radbas<false>(const Matrix& inputs)
{
	//assume patterns have gaussian distribution around mean
	Matrix mean = inputs.vMean(true);
	Matrix std = inputs.vStd(true);
	//deviation
	Matrix dev(inputs.row_num(), 1);
	//iMatrix::r_iterator p_aft = aft_.begin();
	r_iterator p_b = B_.begin();
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		for(r_iterator pos = dev.begin(), end = dev.end(); pos != end; ++pos)
			*pos = prg::randn();
		dev *= std; dev += mean;
		p_n->weights_ = dev;
		//set sigma^-1 = 1
		*p_b = 1; ++p_b;
	}
}

void layer::init_weights(const Matrix& inputs)
{
	init_weights_random();
	//init_weights_radbas(inputs);
}

template<class goal_action>
void layer::_prepare2learn()
{
	ulong flags = 0;
	//always construct gradient
	flags |= 1;
	//prev gradient
	if(net_.opt_.learnFun == R_BP || net_.opt_.learnFun == QP || net_.opt_.learnFun == R_BP_PLUS)
		flags |= 2;

	//biases
	//gradient
	if((flags & 1) > 0) {
		BG_.Resize(B_.size(), 1); BG_ = 0;
		if((flags & 2) > 0) {
			OBG_.Resize(B_.size(), 1); OBG_ = 0;
		}
	}
	//other initialization
	if(net_.opt_.learnFun == R_BP)
		BD_ = net_.opt_.rp_delta0;
	else
		BD_ = 0;

	//neurons
	for(n_iterator p_n(neurons_.begin()); p_n != neurons_.end(); ++p_n) {
		//gradient
		if((flags & 1) > 0) {
			p_n->grad_.Resize(p_n->weights_.size(), 1);
			p_n->grad_ = 0;
			if((flags & 2) > 0) {
				p_n->prevg_.Resize(p_n->weights_.size(), 1);
				p_n->prevg_ = 0;
			}
		}
		//additional space for QP
		//if(flags & 4 > 0) {
		//	p_n->_a.Resize(p_n->weights_.size(), 1);
		//	p_n->_a = 0;
		//}

		if(net_.opt_.learnFun == R_BP)
			p_n->deltas_ = net_.opt_.rp_delta0;
		else p_n->deltas_ = 0;
	}

	//set update function
	switch(net_.opt_.learnFun) {
		default:
		case BP: _pUpdateFun = &layer::_bp_update<goal_action>;
			break;
		case R_BP: _pUpdateFun = &layer::_rbp_update<goal_action>;
			if(net_.opt_.useSimpleRP) _pRP_alg = &layer::_rp_simple<goal_action>;
			else _pRP_alg = &layer::_rp_original<goal_action>;
			break;
		case R_BP_PLUS:
			_pUpdateFun = &layer::_rp_plus_update<goal_action>;
			break;
		case QP: _pUpdateFun = &layer::_qp_update<goal_action>;
			if(net_.opt_.useSimpleQP) _pQP_alg = &layer::_qp_simple<goal_action>;
			else _pQP_alg = &layer::_qp_original<goal_action>;
			break;
	}

	if(!net_.opt_.use_biases_) {
		//properly disable biases influence for different aft
		iMatrix::r_iterator p_aft = aft_.begin();
		for(r_iterator p_b = B_.begin(); p_b != B_.end(); ++p_b) {
			if(*p_aft != radbas && *p_aft != revradbas) *p_b = 0;
			else *p_b = 1;
			++p_aft;
		}
	}

	//by default backprop local gradient
	backprop_lg_ = true;
}

bool layer::calc_grad()
{
	//pure weight gradient = -(local gradiend)*(dv/dw)
	//thats why using "-=" in gradient calculation
	bool palsy = true;
	//calc error
	deriv_af();
	Goal_ *= axons_;

	//process biases
	/*
	if(net_.opt_.use_biases_) {
		BG_ -= Goal_;
		for(r_iterator p_bg(BG_.begin()); p_bg != BG_.end(); ++p_bg) {
			if(abs(*p_bg) > net_.opt_.epsilon) {
				palsy = false;
				break;
			}
		}
	}
	*/

	//calc weights gradient
	mp_iterator p_er = Goal_.begin(), p_state = states_.begin();
	r_iterator p_b = B_.begin(), p_bg = BG_.begin();
	iMatrix::r_iterator p_aft = aft_.begin();
	r_iterator p_g, p_w;
	double g;
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		if(net_.opt_.use_biases_) {
			if(*p_aft == radbas || *p_aft == revradbas)
				*p_bg -= *p_er * 2 * (*p_b) * (*p_state);
			else if(*p_aft == multiquad || *p_aft == revmultiquad)
				*p_bg -= *p_er * 2 * (*p_b);
			else
				*p_bg -= *p_er;
			//palsy check
			if(palsy && abs(*p_bg) > net_.opt_.epsilon)
				palsy = false;
		}

		if(*p_aft == radbas || *p_aft == revradbas)
			g = 2*(*p_b)*(*p_b) * (*p_er);
		else if(*p_aft == multiquad || *p_aft == revmultiquad)
			g = 2 * (*p_er);
		p_g = p_n->grad_.begin(); p_w = p_n->weights_.begin();
		for(np_iterator p_in = p_n->inputs_.begin(); p_in != p_n->inputs_.end(); ++p_in) {
			//calc grad element
			if(*p_aft == radbas || *p_aft == revradbas || *p_aft == multiquad || *p_aft == revmultiquad)
				*p_g -= (*p_w - p_in->axon_)*g;
			else
				*p_g -= p_in->axon_*(*p_er);

			//palsy check
			if(palsy && abs(*p_g) > net_.opt_.epsilon)
				palsy = false;

			//calc local gradient in prev layer
			//local gradient is opposite to weight gradient - so using "+="
			if(backprop_lg_)
				p_in->error_ += *p_w * (*p_er);

			++p_g; ++p_w;
		}

		++p_er;
		++p_b; ++p_bg; ++p_aft;
		++p_state;
	}

	return palsy;
}

template<class goal_action>
bool layer::_bp_gradless_update(bool backprop_er)
{
	bool palsy = true;
	//calc error
	deriv_af();
	Goal_ *= axons_;

	//process biases
	if(net_.opt_.use_biases_) {
		//add moment
		BD_ *= net_.opt_.mu;
		//backprop update rule
		goal_action::update(BD_, Goal_ * net_.opt_.nu);
		//palsy check
		for(r_iterator p_bd(BD_.begin()); p_bd != BD_.end(); ++p_bd) {
			if(*p_bd > net_.opt_.epsilon) {
				palsy = false;
				break;
			}
		}
		//update biases
		if(!palsy) B_ += BD_;
	}

	//calc weights gradient
	mp_iterator p_er = Goal_.begin();
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		r_iterator p_d = p_n->deltas_.begin(), p_w = p_n->weights_.begin();
		for(np_iterator p_in = p_n->inputs_.begin(); p_in != p_n->inputs_.end(); ++p_in) {
			//add moment
			*p_d *= net_.opt_.mu;
			//backprop update rule
			goal_action::update(*p_d, p_in->axon_*(*p_er) * net_.opt_.nu);
			//palsy check
			if(palsy && *p_d > net_.opt_.epsilon)
				palsy = false;
			//update weight
			if(!palsy) *p_w += *p_d;

			//calc error element in prev layer
			if(backprop_er)
				p_in->error_ += *p_w * (*p_er);

			++p_d; ++p_w;
		}
		++p_er;
	}

	return palsy;
}

template<class goal_action>
void layer::_qp_original(Matrix& W, Matrix& D, Matrix& G, Matrix& OG)
{
	double beta;
	r_iterator p_d(D.begin()), p_g(G.begin()), p_og(OG.begin());
	for(r_iterator p_w = W.begin(); p_w != W.end(); ++p_w) {
		if(*p_d != 0) {
			beta = *p_g / (*p_og - *p_g);
			//add moment
			if(beta > net_.opt_.qp_alfamax)
				beta = net_.opt_.qp_alfamax;
			beta *= *p_d;
		}
		else beta = 0;

		if(beta == 0 || *p_d * beta < 0)
			goal_action::update(beta, *p_g * net_.opt_.nu);
		//*p_d = beta*(1 - net_.opt_.qp_lambda);
		*p_og = *p_g;
		//update weight
		*p_w += *p_d;
		//update iterators
		++p_d; ++p_g; ++p_og;
	}
}

template<class goal_action>
void layer::_qp_simple(Matrix& W, Matrix& D, Matrix& G, Matrix& OG)
{
	//double beta;
	r_iterator p_d(D.begin()), p_g(G.begin()), p_og(OG.begin());
	for(r_iterator p_w = W.begin(); p_w != W.end(); ++p_w) {
		if(*p_d != 0)
			*p_d *= min(*p_g / (*p_og - *p_g), net_.opt_.qp_alfamax);
		else
			goal_action::update(*p_d, *p_g * net_.opt_.nu);
		//*p_d *= (1 - net_.opt_.qp_lambda);
		*p_og = *p_g;
		//update weight
		*p_w += *p_d;
		//update iterators
		++p_d; ++p_g; ++p_og;
	}
}

template<class goal_action>
void layer::_qp_modified(Matrix& W, Matrix& D, Matrix& G, Matrix& OG)
{
	double beta;
	r_iterator p_d(D.begin()), p_g(G.begin()), p_og(OG.begin());
	for(r_iterator p_w = W.begin(); p_w != W.end(); ++p_w) {
		if(*p_d != 0) {
			beta = abs(*p_og - *p_g)/abs(*p_d);
		}
		else beta = 0;
		*p_d = 0;
		if(beta != 0)
			goal_action::update(*p_d, *p_g * (1/beta) * net_.opt_.nu);
		else
			goal_action::update(*p_d, *p_g * net_.opt_.nu);
		*p_og = *p_g;
		//update weight
		*p_w += *p_d;
		//update iterators
		++p_d; ++p_g; ++p_og;
	}
}

template<class goal_action>
void layer::_qp_update(bool zero_grad)
{
	//biases
	if(net_.opt_.use_biases_) {
		if(net_.opt_.useSimpleQP)
			_qp_modified<goal_action>(B_, BD_, BG_, OBG_);
		else
			_qp_original<goal_action>(B_, BD_, BG_, OBG_);
		if(zero_grad) BG_ = 0;
	}

	//neurons
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		if(net_.opt_.useSimpleQP)
			_qp_modified<goal_action>(p_n->weights_, p_n->deltas_, p_n->grad_, p_n->prevg_);
		else
			_qp_original<goal_action>(p_n->weights_, p_n->deltas_, p_n->grad_, p_n->prevg_);
		if(zero_grad) p_n->grad_ = 0;
	}
}

template<class goal_action>
void layer::_bp_update(bool zero_grad)
{
	//process biases
	if(net_.opt_.use_biases_) {
		//add moment
		BD_ *= net_.opt_.mu;
		//backprop update rule
		goal_action::update(BD_, BG_ * net_.opt_.nu);
		//update biases
		B_ += BD_;
		if(zero_grad) BG_ = 0;
	}

	//process weights
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		//add moment
		p_n->deltas_ *= net_.opt_.mu;
		//backprop update rule
		goal_action::update(p_n->deltas_, p_n->grad_ * net_.opt_.nu);
		//update weights
		p_n->weights_ += p_n->deltas_;
		if(zero_grad) p_n->grad_ = 0;
	}
}

template<class goal_action>
void layer::_rp_original(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights)
{
/*
	Matrix gc = grad.Mul(old_grad);
	//extract only changes in grad sign
	Matrix gc_lz(gc.row_num(), gc.col_num(), gc.GetBuffer());
	replace_if(gc_lz.begin(), gc_lz.end(), bind2nd(greater<double>(), 0), 0);
	replace_if(gc_lz.begin(), gc_lz.end(), bind2nd(less<double>(), 0), -1);
	//revert these steps back
	weights += deltas.Mul(gc_lz.Mul(old_grad.sign()));
	//zero grad where changes in sign happen
	grad += grad.Mul(gc_lz);

	//correct deltas
	replace_if(gc.begin(), gc.end(), bind2nd(greater<double>(), 0), net_.opt_.rp_delt_inc);
	replace_if(gc.begin(), gc.end(), bind2nd(less<double>(), 0), net_.opt_.rp_delt_dec);
	replace_if(gc.begin(), gc.end(), bind2nd(equal_to<double>(), 0), 1);
	//deltas += deltas.Mul(gc);
	deltas *= gc;
	//update weights
	weights += deltas.Mul(grad.sign());
*/

	//iterator form

	r_iterator p_w = weights.begin(), p_g = grad.begin(), pold_g = old_grad.begin();
	double dT;
	for(r_iterator p_d = deltas.begin(); p_d != deltas.end(); ++p_d) {
		dT = (*p_g)*(*pold_g);
		if(dT >= 0) {
			if(dT > 0)
				*p_d = min(*p_d * net_.opt_.rp_delt_inc, net_.opt_.rp_deltamax);
			if(*p_g > 0) dT = *p_d;
			else if(*p_g < 0) dT = - *p_d;
			else dT = 0;
		}
		else {
			if(*pold_g > 0) dT = - *p_d;
			else dT = *p_d;
			*p_d *= net_.opt_.rp_delt_dec;
			*p_g = 0;
		}

		//*p_w += dT;
		*pold_g = *p_g;
		goal_action::update(*p_w, dT);
		++p_g; ++pold_g;
		++p_w;
	}
}

template<class goal_action>
void layer::_rp_simple(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights)
{
/*
	Matrix gc = grad.Mul(old_grad);
	replace_if(gc.begin(), gc.end(), bind2nd(greater<double>(), 0), net_.opt_.rp_delt_inc);
	replace_if(gc.begin(), gc.end(), bind2nd(less<double>(), 0), net_.opt_.rp_delt_dec);
	replace_if(gc.begin(), gc.end(), bind2nd(equal_to<double>(), 0), 1);
	//deltas += deltas.Mul(gc);
	deltas *= gc;
	replace_if(deltas.begin(), deltas.end(), bind2nd(greater<double>(), net_.opt_.rp_deltamax), net_.opt_.rp_deltamax);
	//update weights
	weights += deltas.Mul(grad.sign());
*/

	//iterator form

	r_iterator p_w = weights.begin(), p_g = grad.begin(), pold_g = old_grad.begin();
	double dT;
	for(r_iterator p_d = deltas.begin(); p_d != deltas.end(); ++p_d) {
		dT = (*p_g)*(*pold_g);
		if(dT > 0)
			*p_d = min(*p_d * net_.opt_.rp_delt_inc, net_.opt_.rp_deltamax);
		else if(dT < 0)
			*p_d *= net_.opt_.rp_delt_dec;
		if(*p_g > 0) dT = *p_d;
		else if(*p_g < 0) dT = - *p_d;
		else dT = 0;

		//*p_w += dT;
		*pold_g = *p_g;
		goal_action::update(*p_w, dT);
		++p_g; ++pold_g;
		++p_w;
	}
}

template<class goal_action>
void layer::_rbp_update(bool zero_grad)
{
	//process biases
	if(net_.opt_.use_biases_) {
		if(net_.opt_.useSimpleRP)
			_rp_simple<goal_action>(BG_, OBG_, BD_, B_);
		else
			_rp_original<goal_action>(BG_, OBG_, BD_, B_);
		//OBG_ = BG_;
		if(zero_grad) BG_ = 0;
	}
	//process weights
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		if(net_.opt_.useSimpleRP)
			_rp_simple<goal_action>(p_n->grad_, p_n->prevg_, p_n->deltas_, p_n->weights_);
		else
			_rp_original<goal_action>(p_n->grad_, p_n->prevg_, p_n->deltas_, p_n->weights_);
		//p_n->prevg_ = p_n->grad_;
		if(zero_grad) p_n->grad_ = 0;
	}
}

//struct rp_plus_rule {
//	template< class _goal_action >
//	static bool calc_delta(const double& slope, double& delta);

//	template< >
//	static bool calc_delta< anti_grad >(const double& slope, double& delta) {
//		if(slope < 0) return true;
//		else return false;
//	}

//	template< >
//	static bool check_slope< follow_grad >(const double& a) {
//		if(a < 0) return true;
//		else return false;
//	}
//};

template< class goal_action >
void layer::_rp_plus(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights)
{
	r_iterator p_w = weights.begin(), p_g = grad.begin(), pold_g = old_grad.begin();
	double dT;
	bool fin_rp_step;
	for(r_iterator p_d = deltas.begin(); p_d != deltas.end(); ++p_d) {
		fin_rp_step = true;
		//if(*p_g != 0) {
		if(*p_g - *pold_g != 0)
			//calc slope
			if(*p_d == 0) {
				//initial case - best we can do is a little step following gradient
				*p_d = net_.opt_.rp_delta0;
			}
			else {
				dT = (*p_d) / (*p_g - *pold_g);
				if(dT < 0) {
					//main update rule
					//we need negative slope - in other case using rp boost rule
					goal_action::assign(dT, dT);
					goal_action::assign(*p_d, -(*p_g) * dT);
					fin_rp_step = false;
				}
				else
					*p_d = min(*p_d * net_.opt_.rp_delt_inc, net_.opt_.rp_deltamax);
			}
		else	//gradient hasn't changed - just boost current step
			*p_d = min(*p_d * net_.opt_.rp_delt_inc, net_.opt_.rp_deltamax);
		//dT = *p_d;
		//}
		//else dT = 0;

		if(fin_rp_step) {
			if(*p_g == 0) *p_d = 0;
			else if(*p_g < 0) *p_d = -(*p_d);
		}


		*pold_g = *p_g;
		goal_action::update(*p_w, *p_d);
		++p_g; ++pold_g;
		++p_w;
	}
}

template<class goal_action>
void layer::_rp_plus_update(bool zero_grad)
{
	//process biases
	if(net_.opt_.use_biases_) {
		_rp_plus< goal_action >(BG_, OBG_, BD_, B_);
		if(zero_grad) BG_ = 0;
	}
	//process weights
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
		_rp_plus< goal_action >(p_n->grad_, p_n->prevg_, p_n->deltas_, p_n->weights_);
		if(zero_grad) p_n->grad_ = 0;
	}
}

void layer::empty_update(bool zero_grad)
{
	//just zero gradient if needed
	for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n)
		if(zero_grad) p_n->grad_ = 0;
}

}	//end of namespace NN
