#include "objnet.h"

using namespace std;
using namespace NN;

//--------------------------------PCA network implementation---------------------------------------------------
pcan::pcan(ulong input_size, ulong prin_comp_num)
{
	set_input_size(input_size);
	set_output_layer(prin_comp_num);
}

void pcan::set_output_layer(ulong prin_comp_num)
{
	if(inp_size() == 0) throw nn_except(NoInputSize);
	if(prin_comp_num == 0) prin_comp_num = inp_size();
	layer* p_l;
	if(layers_num() == 0)
		p_l = &objnet::add_layer<bp_layer>(prin_comp_num, purelin);
	else {
		p_l = &layers_[0];
		p_l->init(prin_comp_num, purelin);
	}
	p_l->set_links(create_ptr_mat(input_.neurons()));
}

void pcan::prepare2learn()
{
	//do nothing
}

void pcan::gha_step()
{
	//only update rule
	layer& l = layers_[0];

	double tmp;
	r_iterator p_w, p_d;
	np_iterator p_in;
	for(n_iterator p_n = l.neurons_.begin(); p_n != l.neurons_.end(); ++p_n) {
		p_d = p_n->deltas_.begin();
		p_in = p_n->inputs_.begin();
		for(p_w = p_n->weights_.begin(); p_w != p_n->weights_.end(); ++p_w) {
			tmp = *p_w * p_in->axon_;
			*p_d = state_.nu * p_n->axon_ * (p_in->axon_ - tmp);
			//update input
			p_in->axon_ -= tmp;
			//update weight
			*p_w += *p_d;

			++p_d; ++p_in;
		}
	}
}

void pcan::learn_epoch(const Matrix& inputs, const Matrix& targets)
{
	//update weights
	for(ulong i=0; i < inputs.col_num(); ++i) {
		set_input(inputs.GetColumns(i));
		propagate();
		gha_step();
	}

	//calc performance
	Matrix wt;
	double n = 0, tmp;
	layer& l = layers_[0];
	for(n_iterator p_n = l.neurons_.begin(); p_n != l.neurons_.end(); ++p_n) {
		wt <<= !p_n->weights_;
		for(n_iterator p_n1 = l.neurons_.begin(); p_n1 != l.neurons_.end(); ++p_n1) {
			tmp = (wt * p_n1->weights_)[0];
			n += tmp*tmp;
		}
	}
	state_.perf = sqrt(n) - l.neurons_.size();
	
	//update speed using simple alg
	if(opt_.adaptive) {
		if(state_.perf < state_.lastPerf)
			state_.nu *= 1.2;
		else if(state_.perf > state_.lastPerf) state_.nu *= 0.5;
		if(state_.nu < 0.0001) state_.nu = 0.0001;
	}
}

int pcan::learn(const Matrix& inputs, const Matrix& targets, bool initialize, pLearnInformer pProc,
		smart_ptr< const Matrix > test_inp, smart_ptr< const Matrix > test_tar)
{
	opt_.goal_checkFun = no_check;
	return objnet::common_learn(inputs, targets, initialize, pProc);
}

