#include "objnet.h"

using namespace std;
using namespace NN;

//-------------------------------- neuron strategies -----------------------------------------

//--------------------------------neuron class implementation---------------------------------

//const double neuron::s_bias1 = 1;

neuron::neuron(ulong inp_count, int aft, bool backprop_lg) :
	inputs_(inp_count, 1), weights_(inp_count, 1),
	deltas_(inp_count, 1)
{
	init(inp_count, aft, backprop_lg);
}

neuron::neuron(const neurMatrixPtr& synapses, int aft, bool backprop_lg)
{
	set_synapses(synapses);
	init(inputs_.size(), aft, backprop_lg);
}

//reference semantics - disabled
/*
neuron::neuron(const neuron& n) :
	weights_(n.weights_), inputs_(n.inputs_),
	deltas_(n.deltas_), grad_(n.grad_),
	prevg_(n.prevg_),	error_(n.error_),
	_pStateFcn(n._pStateFcn)
{
}
*/
neuron::neuron(const neuron& n)
{
	*this = n;
}

//copy semantics
neuron& neuron::operator =(const neuron& n)
{
	weights_ = n.weights_; inputs_ ^= n.inputs_;
	deltas_ = n.deltas_;
	grad_ = n.grad_; prevg_ = n.prevg_;
	error_ = n.error_; state_ = n.state_; axon_ = n.axon_;
	bias_ = n.bias_;

	state_fcn = n.state_fcn;
	grad_fcn = n.grad_fcn;
	act_fcn = n.act_fcn;
	deriv_fcn = n.deriv_fcn;
	return *this;
}

void neuron::init(ulong inp_count, int aft, bool backprop_lg)
{
	if(inputs_.row_num() != inp_count) {
		weights_.NewMatrix(inp_count, 1); inputs_.NewMatrix(inp_count, 1);
		deltas_.NewMatrix(inp_count, 1);
	}

	switch(aft) {
		default:
		case tansig:
		case logsig:
		case purelin:
		case poslin:
		case expws:
			state_fcn = &neuron::weighted_sum_sf;
			if(backprop_lg)
				grad_fcn = &neuron::grad_ws_blg;
			else
				grad_fcn = &neuron::grad_ws;
			break;

		case radbas:
		case revradbas:
		case multiquad:
		case revmultiquad:
			state_fcn = &neuron::eucl_dist_sf;
			if(backprop_lg)
				grad_fcn = &neuron::grad_ed_blg;
			else
				grad_fcn = &neuron::grad_ed;
			break;
	}

	switch(aft) {
		default:
		case tansig:
			act_fcn = &neuron::act_tansig;
			deriv_fcn = &neuron::d_tansig;
			break;
		case logsig:
			act_fcn = &neuron::act_logsig;
			deriv_fcn = &neuron::d_logsig;
			break;
		case purelin:
			act_fcn = &neuron::act_purelin;
			deriv_fcn = &neuron::d_purelin;
			break;
		case poslin:
			act_fcn = &neuron::act_poslin;
			deriv_fcn = &neuron::d_poslin;
			break;
		case radbas:
			act_fcn = &neuron::act_gauss;
			deriv_fcn = &neuron::d_gauss;
			break;
		case revradbas:
			act_fcn = &neuron::act_revgauss;
			deriv_fcn = &neuron::d_revgauss;
			break;
		case expws:
			act_fcn = &neuron::act_expws;
			deriv_fcn = &neuron::d_expws;
			break;
		case multiquad:
			act_fcn = &neuron::act_multiquad;
			deriv_fcn = &neuron::d_multiquad;
			break;
		case revmultiquad:
			act_fcn = &neuron::act_revmultiquad;
			deriv_fcn = &neuron::d_revmultiquad;
			break;
	}
}

void neuron::calc_state()
{
	(this->*state_fcn)();
}

void neuron::calc_grad(double mult)
{
	(this->*grad_fcn)(mult);
}

void neuron::activate(const new_nnOptions& opt)
{
	(this->*act_fcn)(opt);
}

void neuron::calc_deriv(const new_nnOptions& opt)
{
	(this->*deriv_fcn)(opt);
}

//-------------------------------------- State Functions -----------------------------------
void neuron::weighted_sum_sf()
{
	//axon_ = weights_.Mul(inputs_).Sum(); //+ _bias;
	state_ = 0;
	r_iterator p_w = weights_.begin();
	for(np_iterator p_n(inputs_.begin()); p_n != inputs_.end(); ++p_n) {
		state_ += *p_w * p_n->axon_;
		++p_w;
	}
}

void neuron::eucl_dist_sf()
{
	//Matrix res = inputs_ - weights_;
	//axon_ = sqrt(res.Mul(res).Sum());
	state_ = 0;
	r_iterator p_w = weights_.begin();
	for(np_iterator p_n(inputs_.begin()); p_n != inputs_.end(); ++p_n) {
		state_ += (p_n->axon_ - *p_w)*(p_n->axon_ - *p_w);
		++p_w;
	}
	//state_ = sqrt(state_);
}

//--------------------------------- Gradient Functions --------------------------------------------
void neuron::grad_ws_blg(double mult)
{
	r_iterator p_g = grad_.begin();
	neurMatrixPtr::r_iterator p_inp = inputs_.begin();
	for(r_iterator p_w = weights_.begin(); p_w != weights_.end(); ++p_w) {
		//calc grad element
		*p_g -= p_inp->axon_ * error_ * mult;
		p_inp->error_ += *p_w * error_;

		++p_g; ++p_inp;
	}
}

void neuron::grad_ws(double mult)
{
	r_iterator p_g = grad_.begin();
	neurMatrixPtr::r_iterator p_inp = inputs_.begin();
	for(r_iterator p_w = weights_.begin(); p_w != weights_.end(); ++p_w) {
		//calc grad element
		*p_g -= p_inp->axon_ * error_ * mult;
		++p_g; ++p_inp;
	}
}

void neuron::grad_ed_blg(double mult)
{
	r_iterator p_g = grad_.begin();
	neurMatrixPtr::r_iterator p_inp = inputs_.begin();
	for(r_iterator p_w = weights_.begin(); p_w != weights_.end(); ++p_w) {
		//calc grad element
		if(state_ != 0) 
			*p_g -= 2*(*p_w - p_inp->axon_) * error_ * mult;
		p_inp->error_ += *p_w * error_;

		++p_g; ++p_inp;
	}
}

void neuron::grad_ed(double mult)
{
	calc_state();
	r_iterator p_g = grad_.begin();
	neurMatrixPtr::r_iterator p_inp = inputs_.begin();
	for(r_iterator p_w = weights_.begin(); p_w != weights_.end(); ++p_w) {
		//calc grad element
		if(state_ != 0) 
			*p_g -= 2*(*p_w - p_inp->axon_) * error_ * mult;
		++p_g; ++p_inp;
	}
}

//----------------------------- Activation functions ---------------------------------------
void neuron::act_tansig(const new_nnOptions& opt)
{
	calc_state();
	axon_ = opt.tansig_a * tanh(opt.tansig_b*(state_ + bias_));
	if(opt.saturate) {	
		axon_ = min(axon_, opt.tansig_a - opt.tansig_e);
		axon_ = max(axon_, opt.tansig_e - opt.tansig_a);
	}
}

void neuron::act_logsig(const new_nnOptions& opt)
{
	calc_state();
	axon_ = 1/(1 + exp(-opt.logsig_a * (state_ + bias_)));
	if(opt.saturate) {
		axon_ = min(axon_, 1 - opt.tansig_e);
		axon_ = max(axon_, opt.tansig_e - 1);
	}
}

void neuron::act_purelin(const new_nnOptions& opt)
{
	calc_state();
	axon_ = state_ + bias_;
}

void neuron::act_poslin(const new_nnOptions& opt)
{
	calc_state();
	axon_ = std::max<double>(state_ + bias_, 0);
}

void neuron::act_gauss(const new_nnOptions& opt)
{
	calc_state();
	//bias contains sigma^-1
	axon_ = exp(-(state_)*(bias_)*(bias_));
}

void neuron::act_revgauss(const new_nnOptions& opt)
{
	calc_state();
	//bias contains sigma^-1
	axon_ = 1 - exp(-(state_)*(bias_)*(bias_));
}

void neuron::act_expws(const new_nnOptions& opt)
{
	calc_state();
	//bias contains sigma^-1
	axon_ = exp(state_ + bias_);
}

void neuron::act_multiquad(const new_nnOptions& opt)
{
	calc_state();
	//bias contains sigma^-1
	axon_ = sqrt(state_ + (bias_)*(bias_));
}

void neuron::act_revmultiquad(const new_nnOptions& opt)
{
	calc_state();
	//bias contains sigma^-1
	axon_ = 1/sqrt(state_ + (bias_)*(bias_));
}

//----------------------------------- Derivative Functions -----------------------------------------
void neuron::d_tansig(const new_nnOptions& opt)
{
	axon_ = opt.tansig_b*(opt.tansig_a - axon_)*(opt.tansig_a + axon_)/opt.tansig_a;
}

void neuron::d_logsig(const new_nnOptions& opt)
{
	axon_ *= opt.logsig_a*(1 - axon_);
}

void neuron::d_poslin(const new_nnOptions& opt)
{
	if(axon_ < 0) axon_ = 0;
	else axon_ = 1;
}

void neuron::d_purelin(const new_nnOptions& opt)
{
	axon_ = 1;
}

void neuron::d_gauss(const new_nnOptions& opt)
{
	axon_ = - axon_;
}

void neuron::d_revgauss(const new_nnOptions& opt)
{
	axon_ = 1 - axon_;
}

void neuron::d_expws(const new_nnOptions& opt)
{
	axon_ *= (state_ + bias_);
}

void neuron::d_multiquad(const new_nnOptions& opt)
{
	if(axon_ != 0) axon_ = 1. / (axon_ * 2);
}

void neuron::d_revmultiquad(const new_nnOptions& opt)
{
	if(axon_ != 0) {
		double r = state_ + (bias_)*(bias_);
		axon_ = - 1. / (sqrt(r*r*r) * 2);
	}
}

//----------------------------------- Other Functions ----------------------------------------------
void neuron::set_synapses(const neurMatrixPtr& synapses)
{
	//make a copy!!! not a reference
	inputs_ ^= synapses;
	weights_.Resize(inputs_.size(), 1);
	deltas_.Resize(inputs_.size(), 1);
	//weights_.NewMatrix(inputs_.size(), 1);
	//deltas_.NewMatrix(inputs_.size(), 1);
}

void neuron::add_synapses(const neurMatrixPtr& synapses, const Matrix* p_weights)
{
	inputs_ &= synapses;
	//preserve old weights!
	if(p_weights && p_weights->size() == synapses.size())
		weights_ &= *p_weights;
	else 
		weights_.Resize(inputs_.size(), 1);
	deltas_.Resize(inputs_.size(), 1);
	if(grad_.size() > 0) grad_.Resize(inputs_.size(), 1);
	if(prevg_.size() > 0) prevg_.Resize(inputs_.size(), 1);
}

void neuron::add_synapse(neuron *const p_n)
{
	inputs_.push_back(p_n);
	weights_.Resize(inputs_.size() + 1, 1);
	deltas_.Resize(inputs_.size() + 1, 1);
	if(grad_.size() > 0) grad_.Resize(inputs_.size() + 1, 1);
	if(grad_.size() > 0) prevg_.Resize(inputs_.size() + 1, 1);
}

void neuron::rem_synapse(neuron *const p_n)
{
	ulong ind;
	if((ind = inputs_.BufElementInd(p_n)) < inputs_.size()) {
		inputs_.DelRows(ind); 
		weights_.DelRows(ind);
		deltas_.DelRows(ind);
		grad_.DelRows(ind);
		prevg_.DelRows(ind);
	}
}

void neuron::rem_synapses(const neurMatrixPtr& synapses)
{
	for(neurMatrixPtr::cbuf_iterator pos = synapses.buf_begin(); pos != synapses.buf_end(); ++pos)
		rem_synapse(*pos);
}
