#include "matrix.h"
#include "objnet.h"

namespace NN {
//void layer_jump::deriv_af() {
//	// jump function has no meaningful derivative
//	// empty here
//}

void layer_jump::prepare2learn() {
	layer::prepare2learn();
	// we'll always use Resilent Propagation
	//_pUpdateFun = &layer::_rbp_update< anti_grad >;
}

bool layer_jump::calc_grad() {
	// propagate gradient for prev layers
	layer::calc_grad();
	// update weights using perceptron rule
	r_iterator p_b = B_.begin();
	mp_iterator p_er = Goal_.begin();
	for(n_iterator p_n = neurons_.begin(), end = neurons_.end(); p_n != end; ++p_n) {
		// process bias
		*p_b += *p_er * net_.opt_.nu;
		// ... and weights
		r_iterator p_w = p_n->weights_.begin();
		for(np_iterator p_in = p_n->inputs_.begin(), end_in = p_n->inputs_.end(); p_in != end_in; ++p_in) {
			*p_w += *p_er * net_.opt_.nu * p_in->axon_;
			++p_w;
		}
	}
	return false;
}

void layer_jump::update_epoch() {
	// weights are updated online
}

void layer_jump::init_weights(const Matrix& inputs) {
	// all weights are zero at the beginning
	for(n_iterator p_n = neurons_.begin(), end = neurons_.end(); p_n != end; ++p_n) {
		p_n->weights_ = 0;
	}
}

}	// namespace NN

