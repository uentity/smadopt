#include "alg_api.h"
#include "matrix.h"
#include "objnet.h"

#include <fstream>
#include <iostream>

using namespace std;
using namespace NN;

namespace {
	// hide learn informer from being exported
	bool learn_informer(ulong uCycle, double perf, void* pNet) {
		cout << ((objnet*)pNet)->status_info();

		ifstream f("stop.txt");
		if(f.rdbuf()->sgetc() == '1') return false;
		else return true;
	}

} // eof hidden namespace

smart_ptr< mlp > p_net;

void BuildMP(ulong layers_num, ulong* neurons_num,
	unsigned int* neuron_af) {

	p_net = new mlp;
	p_net->opt_.ReadOptions();
	// set input layer size
	if(layers_num > 0)
		p_net->set_input_size(neurons_num[0]);
	// set hidden layers
	for(ulong i = 1; i < layers_num; ++i)
		p_net->add_layer(neurons_num[i], neuron_af[i - 1]);
}

double LearnMP(ulong sampl_num, const double* samples, const double* want_resp) {
	// sanity check
	if(!p_net) return -1;

	// make learn set matrices
	const ulong inp_sz = p_net->inp_size();
	const ulong outp_sz = p_net->output().size();
	Matrix train_set(inp_sz, sampl_num, samples), targets(outp_sz, sampl_num, want_resp);

	// learn network
	p_net->learn(train_set, targets, true, learn_informer);
	// return learning error
	return p_net->state().perf;
}

void SimMP(ulong sampl_num, const double* samples, double* res) {
	const ulong inp_sz = p_net->inp_size();
	Matrix input(inp_sz, sampl_num, samples);
	Matrix sim_r = p_net->sim(input);
	std::copy(sim_r.begin(), sim_r.end(), res);
	//memcpy(res, sim_r.GetBuffer(), sim_r.raw_size());
}

